# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Download guards: offline, hash policy, timeout, and HTTPS for COCO."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from autovision.data.downloads import (
    DATASET_REGISTRY,
    DatasetFile,
    DatasetSpec,
    _download_file,
    _extract,
    download_dataset,
)
from autovision.errors import AutoVisionError
from autovision.registry.downloadable import (
    ASSET_MANIFEST,
    DownloadableModelError,
    _download,
    register_downloadable,
    resolve_downloadable,
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class _EmptyResponse:
    """Minimal urlopen response: one EOF read, no Content-Length."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getheader(self, name):
        return None

    def read(self, size=-1):
        return b""


class _PayloadResponse:
    """Minimal urlopen response serving ``payload`` in chunks."""

    def __init__(self, payload: bytes, *, send_content_length: bool = True):
        self._payload = payload
        self._offset = 0
        self._send_content_length = send_content_length

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getheader(self, name):
        return str(len(self._payload)) if self._send_content_length else None

    def read(self, size=-1):
        if self._offset >= len(self._payload):
            return b""
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _TruncatedResponse(_PayloadResponse):
    """Serves one payload chunk, then raises mid-transfer."""

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self._served = False

    def read(self, size=-1):
        if self._served:
            raise ConnectionResetError("connection reset by peer")
        self._served = True
        return super().read(size)


def _register_dataset(
    monkeypatch, name: str, files: tuple[DatasetFile, ...]
) -> DatasetSpec:
    spec = DatasetSpec(
        description="test dataset",
        homepage="https://example.invalid/dataset",
        files=files,
    )
    monkeypatch.setitem(DATASET_REGISTRY, name, spec)
    return spec


def test_dataset_download_is_refused_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
    with pytest.raises(AutoVisionError, match="AUTOVISION_OFFLINE"):
        download_dataset("coco2017-val", tmp_path)


def test_dataset_download_refuses_unpinned_hashes_when_required(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AUTOVISION_REQUIRE_DATASET_HASHES", "1")
    with pytest.raises(AutoVisionError, match="no pinned sha256"):
        download_dataset("coco2017-val", tmp_path)


def test_coco_urls_use_https():
    for name in ("coco2017-val", "coco2017-train"):
        assert all(
            entry.url.startswith("https://")
            for entry in DATASET_REGISTRY[name].files
        )


def test_dataset_download_passes_a_timeout(monkeypatch, tmp_path):
    seen = {}

    def _fake_urlopen(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _EmptyResponse()

    monkeypatch.setenv("AUTOVISION_DOWNLOAD_CHUNK_TIMEOUT_SEC", "5")
    monkeypatch.setattr(
        "autovision.data.downloads.urllib.request.urlopen", _fake_urlopen
    )
    spec = DatasetFile(url="https://example.invalid/empty.bin", sha256=_EMPTY_SHA256)
    target = tmp_path / "empty.bin"
    _download_file(spec, target)
    assert target.read_bytes() == b""
    assert seen == {"url": spec.url, "timeout": 5.0}


def _patch_urlopen(monkeypatch, response_factory):
    monkeypatch.setattr(
        "autovision.data.downloads.urllib.request.urlopen",
        lambda url, timeout=None: response_factory(),
    )


def test_failed_download_leaves_no_final_file_and_cleans_temp(monkeypatch, tmp_path):
    payload = b"partial payload"
    spec = DatasetFile(
        url="https://example.invalid/data.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    _patch_urlopen(monkeypatch, lambda: _TruncatedResponse(payload))
    target = tmp_path / "data.bin"
    with pytest.raises(AutoVisionError, match="Failed to download"):
        _download_file(spec, target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_sha_mismatch_leaves_no_final_file(monkeypatch, tmp_path):
    payload = b"tampered bytes"
    spec = DatasetFile(
        url="https://example.invalid/data.bin",
        sha256=hashlib.sha256(b"expected bytes").hexdigest(),
    )
    _patch_urlopen(monkeypatch, lambda: _PayloadResponse(payload))
    target = tmp_path / "data.bin"
    with pytest.raises(AutoVisionError, match="sha256 mismatch"):
        _download_file(spec, target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_writes_temp_then_atomically_replaces_final(monkeypatch, tmp_path):
    payload = b"fresh content"
    spec = DatasetFile(
        url="https://example.invalid/data.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    target = tmp_path / "data.bin"
    target.write_bytes(b"stale content")
    real_replace = os.replace
    seen = {}

    def _checked_replace(src, dst):
        src, dst = Path(src), Path(dst)
        seen["src"], seen["dst"] = src, dst
        assert src.parent == dst.parent
        assert src.name.startswith(f".{dst.name}.")
        assert src.name.endswith(".part")
        # The final path is untouched until the atomic rename.
        assert dst.read_bytes() == b"stale content"
        real_replace(src, dst)

    _patch_urlopen(monkeypatch, lambda: _PayloadResponse(payload))
    monkeypatch.setattr("autovision.data.downloads.os.replace", _checked_replace)
    _download_file(spec, target)
    assert target.read_bytes() == payload
    assert seen["dst"] == target
    assert not seen["src"].exists()
    assert list(tmp_path.glob("*.part")) == []


def test_cached_archive_with_bad_sha256_is_rejected_before_extraction(
    monkeypatch, tmp_path
):
    name = "test-tampered-cache"
    archive_name = "payload.zip"
    root = tmp_path / name
    root.mkdir()
    with zipfile.ZipFile(root / archive_name, "w") as bundle:
        bundle.writestr("marker.txt", "must not be extracted")
    _register_dataset(
        monkeypatch,
        name,
        (
            DatasetFile(
                url=f"https://example.invalid/{archive_name}",
                sha256=hashlib.sha256(b"different bytes").hexdigest(),
            ),
        ),
    )
    with pytest.raises(
        AutoVisionError, match="cached archive failed sha256 verification"
    ):
        download_dataset(name, tmp_path)
    # Fail closed: the archive stays in place and nothing is extracted.
    assert (root / archive_name).exists()
    assert not (root / "marker.txt").exists()


def test_cached_archive_with_bad_size_is_rejected_before_extraction(
    monkeypatch, tmp_path
):
    name = "test-truncated-cache"
    archive_name = "payload.zip"
    root = tmp_path / name
    root.mkdir()
    with zipfile.ZipFile(root / archive_name, "w") as bundle:
        bundle.writestr("marker.txt", "must not be extracted")
    _register_dataset(
        monkeypatch,
        name,
        (
            DatasetFile(
                url=f"https://example.invalid/{archive_name}",
                size_bytes=1,
            ),
        ),
    )
    with pytest.raises(
        AutoVisionError, match="cached archive failed size verification"
    ):
        download_dataset(name, tmp_path)
    assert not (root / "marker.txt").exists()


def test_unpinned_cached_archive_extracts_with_warning(
    monkeypatch, tmp_path, caplog
):
    name = "test-unpinned-cache"
    archive_name = "payload.zip"
    root = tmp_path / name
    root.mkdir()
    with zipfile.ZipFile(root / archive_name, "w") as bundle:
        bundle.writestr("marker.txt", "extracted")
    _register_dataset(
        monkeypatch,
        name,
        (DatasetFile(url=f"https://example.invalid/{archive_name}"),),
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        download_dataset(name, tmp_path)
    assert (root / "marker.txt").read_text() == "extracted"
    assert "integrity is not verified" in caplog.text


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "nested/../../escape.txt",
        "/absolute/escape.txt",
        "..\\escape.txt",
    ],
)
def test_zip_path_traversal_is_rejected(tmp_path, member):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "boom")
        bundle.writestr("safe.txt", "ok")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(AutoVisionError, match="traversal|absolute-path"):
        _extract(archive, out_dir)
    assert not (tmp_path / "escape.txt").exists()
    # Validation happens before extraction, so even safe members stay out.
    assert not (out_dir / "safe.txt").exists()


def _write_tar(archive: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(archive, "w") as bundle:
        for member_name, payload in members.items():
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))


def _disable_tar_filter(monkeypatch):
    """Simulate a Python 3.10 build whose extractall lacks ``filter``."""
    original = tarfile.TarFile.extractall

    def _without_filter(self, *args, **kwargs):
        if "filter" in kwargs:
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", _without_filter)


def test_tar_path_traversal_is_rejected_without_filter_support(
    monkeypatch, tmp_path
):
    archive = tmp_path / "evil.tar"
    _write_tar(archive, {"../escape.txt": b"boom", "safe.txt": b"ok"})
    _disable_tar_filter(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(AutoVisionError, match="traversal"):
        _extract(archive, out_dir)
    assert not (tmp_path / "escape.txt").exists()
    assert not (out_dir / "safe.txt").exists()


def test_tar_safe_members_extract_without_filter_support(monkeypatch, tmp_path):
    archive = tmp_path / "safe.tar"
    _write_tar(archive, {"nested/safe.txt": b"ok"})
    _disable_tar_filter(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _extract(archive, out_dir)
    assert (out_dir / "nested" / "safe.txt").read_bytes() == b"ok"


def test_downloadable_is_refused_offline(monkeypatch, tmp_path):
    name = "test/offline-only"
    register_downloadable(name, "https://example.invalid/x.pt", "0" * 64)
    monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
    try:
        with pytest.raises(DownloadableModelError, match="AUTOVISION_OFFLINE"):
            resolve_downloadable(name, cache_dir=tmp_path)
    finally:
        ASSET_MANIFEST.pop(name, None)


def test_downloadable_download_passes_a_timeout(monkeypatch, tmp_path):
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"payload"

    def _fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("AUTOVISION_DOWNLOAD_CHUNK_TIMEOUT_SEC", "7")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    target = tmp_path / "model.pt"
    _download("https://example.invalid/model.pt", target)
    assert target.read_bytes() == b"payload"
    assert seen["timeout"] == 7.0
