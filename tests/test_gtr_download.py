# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR upstream downloader: selection, revision resolution, fetch + verify.

No network: the downloader is injected and writes fixture bytes into a
temporary root. The real assets are only referenced for name/path and
``hf_filename`` checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from autovision.registry.gtr_assets import GTR_ASSET_ROOT, GTR_ASSETS, GTRAsset
from autovision.registry.gtr_download import (
    GTRDownloadError,
    download_gtr_asset,
    resolve_revision,
    select_assets,
    target_path,
    verify_asset,
)

_PAYLOAD = b"fake-gtr-checkpoint"
_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


def _asset(name: str = "fake-gtr") -> GTRAsset:
    return GTRAsset(
        name=name,
        path=GTR_ASSET_ROOT / "det" / f"{name}.pth",
        sha256=_SHA,
        license="MIT",
    )


def _writer(payload: bytes, calls: list[dict] | None = None):
    def download(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    return download


class TestSelection:
    def test_all_returns_the_registered_family(self):
        assets = select_assets(all_assets=True)
        assert len(assets) == len(GTR_ASSETS) == 26
        assert {asset.name for asset in assets} == set(GTR_ASSETS)

    def test_task_group(self):
        assets = select_assets(tasks=["det"])
        assert [asset.name for asset in assets] == [
            "gtr_l_coco",
            "gtr_m_coco",
            "gtr_s_coco",
            "gtr_x_coco",
        ]

    def test_names_are_deduplicated_and_sorted(self):
        assets = select_assets(
            names=["gtrpose_s_coco", "gtr_s_coco", "gtrpose_s_coco"]
        )
        assert [asset.name for asset in assets] == [
            "gtr_s_coco",
            "gtrpose_s_coco",
        ]

    def test_all_plus_filter_is_rejected(self):
        with pytest.raises(GTRDownloadError, match="not both"):
            select_assets(all_assets=True, names=["gtr_s_coco"])

    def test_empty_selection_is_rejected(self):
        with pytest.raises(GTRDownloadError, match="choose"):
            select_assets()

    def test_unknown_name_and_task_are_rejected(self):
        with pytest.raises(GTRDownloadError, match="unknown GTR asset"):
            select_assets(names=["nope"])
        with pytest.raises(GTRDownloadError, match="unknown GTR task"):
            select_assets(tasks=["nope"])

    def test_hf_filename_mirrors_the_store_layout(self):
        assert GTR_ASSETS["gtr_s_coco"].hf_filename == "det/gtr_s_coco.pth"
        assert GTR_ASSETS["gtrpose_s_coco"].hf_filename == "pose/gtrpose_s_coco.pth"


class TestTargetAndVerify:
    def test_root_override(self, tmp_path):
        asset = _asset()
        assert target_path(asset, tmp_path) == tmp_path / "det" / "fake-gtr.pth"

    def test_verify_asset_accepts_a_valid_file(self, tmp_path):
        asset = _asset()
        target = target_path(asset, tmp_path)
        target.parent.mkdir(parents=True)
        target.write_bytes(_PAYLOAD)
        assert verify_asset(asset, tmp_path) == target

    def test_verify_asset_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            verify_asset(_asset(), tmp_path)


class TestDownload:
    def test_existing_valid_file_is_reused(self, tmp_path):
        asset = _asset()
        target = target_path(asset, tmp_path)
        target.parent.mkdir(parents=True)
        target.write_bytes(_PAYLOAD)

        def fail(**_kwargs):
            raise AssertionError("downloader must not be called")

        path, downloaded = download_gtr_asset(
            asset, revision="r1", root=tmp_path, downloader=fail
        )
        assert path == target
        assert downloaded is False

    def test_download_writes_and_verifies(self, tmp_path):
        asset = _asset()
        calls: list[dict] = []
        path, downloaded = download_gtr_asset(
            asset,
            revision="r1",
            root=tmp_path,
            downloader=_writer(_PAYLOAD, calls),
        )
        assert downloaded is True
        assert path.read_bytes() == _PAYLOAD
        assert calls == [
            {
                "repo_id": "Phoenix8125/GTR",
                "filename": "det/fake-gtr.pth",
                "revision": "r1",
                "local_dir": str(tmp_path),
            }
        ]

    def test_sha_mismatch_removes_the_bad_file(self, tmp_path):
        asset = _asset()
        target = target_path(asset, tmp_path)
        with pytest.raises(Exception):
            download_gtr_asset(
                asset,
                revision="r1",
                root=tmp_path,
                downloader=_writer(b"wrong-bytes"),
            )
        assert not target.exists()

    def test_corrupt_existing_file_is_replaced(self, tmp_path):
        asset = _asset()
        target = target_path(asset, tmp_path)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"corrupt")
        _path, downloaded = download_gtr_asset(
            asset,
            revision="r1",
            root=tmp_path,
            downloader=_writer(_PAYLOAD),
        )
        assert downloaded is True
        assert target.read_bytes() == _PAYLOAD

    def test_offline_missing_file_fails(self, tmp_path):
        with pytest.raises(GTRDownloadError, match="offline"):
            download_gtr_asset(
                _asset(),
                revision="r1",
                root=tmp_path,
                offline=True,
            )

    def test_force_redownloads_even_when_valid(self, tmp_path):
        asset = _asset()
        target = target_path(asset, tmp_path)
        target.parent.mkdir(parents=True)
        target.write_bytes(_PAYLOAD)
        calls: list[dict] = []
        _path, downloaded = download_gtr_asset(
            asset,
            revision="r1",
            root=tmp_path,
            force=True,
            downloader=_writer(_PAYLOAD, calls),
        )
        assert downloaded is True
        assert len(calls) == 1


class TestRevision:
    def test_explicit_revision_wins(self):
        assert resolve_revision("deadbeef") == "deadbeef"

    def test_resolves_the_upstream_commit(self):
        class Info:
            sha = "abc123"

        class Api:
            def model_info(self, repo_id):
                assert repo_id == "Phoenix8125/GTR"
                return Info()

        assert resolve_revision(api=Api()) == "abc123"

    def test_pinned_registry_revision_is_used(self, monkeypatch):
        monkeypatch.setattr(
            "autovision.registry.gtr_download.GTR_HF_REVISION", "pinned123"
        )
        assert resolve_revision() == "pinned123"

    def test_resolution_failure_is_actionable(self):
        class Api:
            def model_info(self, repo_id):
                raise RuntimeError("gated repo")

        with pytest.raises(GTRDownloadError, match="failed to resolve"):
            resolve_revision(api=Api())
