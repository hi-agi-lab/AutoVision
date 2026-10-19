# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for verified pretrained-weight loading (supply-chain integrity)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from autovision.encoders.weight_hashes import WEIGHT_MANIFEST
from autovision.encoders.weight_verification import (
    WeightArtifact,
    WeightVerificationError,
    _render_manifest,
    compute_file_sha256,
    get_weight_artifact,
    load_pretrained_verified,
)
from autovision.errors import CheckpointError

pytest.importorskip("huggingface_hub")

GOOD_CONTENT = b"weights-bytes"
BAD_CONTENT = b"corrupted-bytes"
GOOD_HASH = hashlib.sha256(GOOD_CONTENT).hexdigest()
BAD_HASH = hashlib.sha256(BAD_CONTENT).hexdigest()


class FakeModelCls:
    """Stands in for a transformers model class."""

    calls: list[Path] = []

    @classmethod
    def from_pretrained(cls, path: Any, **kwargs: Any) -> "FakeModelCls":
        cls.calls.append(Path(path))
        return cls()


def make_artifact(**overrides: Any) -> WeightArtifact:
    defaults = dict(
        repo_id="org/model",
        license="apache-2.0",
        source="huggingface",
        revision=None,
        sha256={"model.safetensors": GOOD_HASH},
    )
    defaults.update(overrides)
    return WeightArtifact(**defaults)


@pytest.fixture()
def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Queue-driven fake snapshot_download: each call writes one content set.

    ``contents`` items are dicts of filename -> bytes, or an Exception to
    raise instead.
    """

    calls: list[list] = []

    def download(repo_id: str, revision: str | None) -> Path:
        snapshot = tmp_path / "cache" / f"models--{repo_id.replace('/', '--')}" / "snap"
        payload = calls.pop(0)
        if isinstance(payload, Exception):
            raise payload
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename, content in payload.items():
            (snapshot / filename).write_bytes(content)
        return snapshot

    monkeypatch.setattr(
        "autovision.encoders.weight_verification._download_snapshot",
        lambda repo_id, revision, *a, **k: download(repo_id, revision),
    )
    monkeypatch.setattr(
        "autovision.encoders.weight_verification._lfs_sha256",
        lambda repo_id, revision: {"model.safetensors": GOOD_HASH},
    )
    monkeypatch.setattr(
        "autovision.encoders.weight_verification.get_weight_artifact",
        lambda repo_id: make_artifact(),
    )
    calls.append({"config.json": b"{}", "model.safetensors": GOOD_CONTENT})
    return calls


class TestManifest:
    def test_covers_dinov2_registry(self):
        from autovision.encoders.dinov2 import dinov2_source

        repos = {
            info["hf"] if use_reg else info["hf_noreg"]
            for info in dinov2_source._MODEL_INFO.values()
            for use_reg in (True, False)
        }
        for repo in repos:
            artifact = get_weight_artifact(repo)
            assert artifact is not None, f"missing manifest entry: {repo}"
            assert artifact.license
            assert artifact.source

    def test_covers_dinov3_registry(self):
        from autovision.encoders.dinov3 import dinov3_source

        for info in dinov3_source._MODEL_INFO.values():
            if info["hf"] is None:
                # No public HF checkpoint (distilled vitt16/vitt16plus):
                # config-built only, weights arrive via backbone_weights /
                # detector loads — nothing to pin in the manifest.
                continue
            artifact = get_weight_artifact(info["hf"])
            assert artifact is not None, f"missing manifest entry: {info['hf']}"
            assert artifact.license
            assert artifact.source

    def test_renderer_round_trip(self):
        rendered = _render_manifest(dict(WEIGHT_MANIFEST))
        assert f"'{next(iter(WEIGHT_MANIFEST))}'" in rendered
        # Re-rendering the rendered file's data must be byte-identical.
        namespace: dict[str, Any] = {}
        exec(compile(rendered, "<manifest>", "exec"), namespace)  # noqa: S102
        assert _render_manifest(namespace["WEIGHT_MANIFEST"]) == rendered


class TestSha256:
    def test_known_vector(self, tmp_path: Path):
        path = tmp_path / "blob"
        path.write_bytes(b"abc")
        assert compute_file_sha256(path) == hashlib.sha256(b"abc").hexdigest()

    def test_large_file_chunked(self, tmp_path: Path):
        path = tmp_path / "blob"
        payload = GOOD_CONTENT * 500_000  # ~6 MB: multiple chunks
        path.write_bytes(payload)
        assert compute_file_sha256(path) == hashlib.sha256(payload).hexdigest()


class TestLoadPretrainedVerified:
    def test_happy_path_pinned(self, fake_download):
        calls = fake_download
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert FakeModelCls.calls[-1].name == "snap"
        assert (FakeModelCls.calls[-1] / "model.safetensors").read_bytes() == GOOD_CONTENT
        assert len(calls) == 0  # exactly the one queued download consumed

    def test_mismatch_retries_once_then_succeeds(self, fake_download):
        calls = fake_download
        calls.insert(0, {"config.json": b"{}", "model.safetensors": BAD_CONTENT})
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert len(calls) == 0  # both queued downloads consumed
        final_snapshot = FakeModelCls.calls[-1]
        assert (final_snapshot / "model.safetensors").read_bytes() == GOOD_CONTENT

    def test_double_mismatch_raises_with_guidance(
        self, monkeypatch, fake_download, tmp_path: Path
    ):
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            lambda *a, **k: _write_snapshot(tmp_path, BAD_CONTENT),
        )
        with pytest.raises(WeightVerificationError, match="sha256 mismatch"):
            load_pretrained_verified(FakeModelCls, "org/model")

    def test_unpinned_uses_lfs_metadata_tier(self, monkeypatch, fake_download):
        calls = fake_download
        monkeypatch.setattr(
            "autovision.encoders.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(sha256={}),
        )
        lfs_calls: list[str] = []
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._lfs_sha256",
            lambda repo_id, revision: lfs_calls.append(repo_id)
            or {"model.safetensors": GOOD_HASH},
        )
        load_pretrained_verified(FakeModelCls, "org/model")
        assert lfs_calls == ["org/model"]
        assert len(calls) == 0  # single queued download consumed

    def test_pinned_skips_lfs_metadata(self, monkeypatch, fake_download):
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._lfs_sha256",
            lambda repo_id, revision: (_ for _ in ()).throw(
                AssertionError("network metadata must not be fetched when pinned")
            ),
        )
        load_pretrained_verified(FakeModelCls, "org/model")

    def test_unknown_repo_loads_unverified_with_warning(
        self, monkeypatch, caplog: pytest.LogCaptureFixture
    ):
        seen: list[Any] = []

        class PlainCls:
            @classmethod
            def from_pretrained(cls, repo: str, **kwargs: Any) -> "PlainCls":
                seen.append(repo)
                return cls()

        with caplog.at_level("WARNING", logger="autovision"):
            load_pretrained_verified(PlainCls, "org/unlisted")
        assert seen == ["org/unlisted"]
        assert "WITHOUT" in caplog.text


class TestOfflineAndMirror:
    def test_offline_uses_cache_without_network(self, monkeypatch, tmp_path: Path):
        snapshot = _write_snapshot(tmp_path, GOOD_CONTENT, write=True)
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        network_attempts: list[tuple] = []

        def fake_download(
            repo_id: str, revision: str | None, local_files_only: bool = False
        ) -> Path:
            # Offline lookups arrive with local_files_only=True; anything
            # else would be a network attempt.
            assert local_files_only, "offline mode must not hit the network"
            network_attempts.append((repo_id, revision, local_files_only))
            return snapshot

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            fake_download,
        )
        monkeypatch.setattr(
            "autovision.encoders.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(
                sha256={"model.safetensors": hashlib.sha256(GOOD_CONTENT).hexdigest()}
            ),
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert network_attempts == [("org/model", None, True)]

    def test_offline_missing_cache_actionable_error(
        self, monkeypatch, fake_download, tmp_path: Path
    ):
        from huggingface_hub.errors import LocalEntryNotFoundError

        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            lambda *a, **k: (_ for _ in ()).throw(
                LocalEntryNotFoundError("not cached")
            ),
        )
        with pytest.raises(CheckpointError, match="AUTOVISION_OFFLINE"):
            load_pretrained_verified(FakeModelCls, "org/model")

    def test_offline_unpinned_skips_verification(self, monkeypatch, tmp_path: Path):
        snapshot = _write_snapshot(tmp_path, BAD_CONTENT, write=True)
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        monkeypatch.setattr(
            "autovision.encoders.weight_verification.get_weight_artifact",
            lambda repo_id: make_artifact(sha256={}),
        )
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            lambda *a, **k: snapshot,
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)  # no hash check, loads anyway

    def test_mirror_url_bridges_to_hf_endpoint(
        self, monkeypatch, fake_download
    ):
        monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://mirror.example/hf")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        load_pretrained_verified(FakeModelCls, "org/model")
        import os

        assert os.environ["HF_ENDPOINT"] == "https://mirror.example/hf"

    def test_existing_hf_endpoint_wins(self, monkeypatch, fake_download):
        import os

        monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://mirror.example/hf")
        monkeypatch.setenv("HF_ENDPOINT", "https://explicit.example")
        load_pretrained_verified(FakeModelCls, "org/model")
        assert os.environ["HF_ENDPOINT"] == "https://explicit.example"


class TestModelScopeSource:
    """AUTOVISION_WEIGHT_SOURCE=modelscope routes pinned mirrors to
    modelscope.cn; verification semantics are identical."""

    def _fake_ms_download(self, monkeypatch, tmp_path, content=GOOD_CONTENT):
        calls: list[str] = []

        def download(ms_id: str, local_files_only: bool = False) -> Path:
            calls.append(ms_id)
            snapshot = tmp_path / "ms" / ms_id.replace("/", "--")
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "model.safetensors").write_bytes(content)
            return snapshot

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_modelscope_snapshot",
            download,
        )
        return calls

    def _ms_artifact(self, monkeypatch, **overrides):
        artifact = make_artifact(modelscope_id="merve/model", **overrides)
        monkeypatch.setattr(
            "autovision.encoders.weight_verification.get_weight_artifact",
            lambda repo_id: artifact,
        )
        return artifact

    def test_modelscope_source_loads_mirror(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("AUTOVISION_WEIGHT_SOURCE", "modelscope")
        self._ms_artifact(monkeypatch)
        calls = self._fake_ms_download(monkeypatch, tmp_path)
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert calls == ["merve/model"]
        assert FakeModelCls.calls[-1] == tmp_path / "ms" / "merve--model"

    def test_modelscope_hash_mismatch_retries_once(
        self, monkeypatch, tmp_path
    ):
        contents = iter([BAD_CONTENT, GOOD_CONTENT])
        monkeypatch.setenv("AUTOVISION_WEIGHT_SOURCE", "modelscope")
        self._ms_artifact(monkeypatch)

        def download(ms_id: str, local_files_only: bool = False) -> Path:
            snapshot = tmp_path / "ms" / ms_id.replace("/", "--")
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "model.safetensors").write_bytes(next(contents))
            return snapshot

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_modelscope_snapshot",
            download,
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)  # second download verified

    def test_modelscope_falls_back_without_mirror_id(
        self, monkeypatch, fake_download
    ):
        # Manifest entry without modelscope_id: the env switch must not
        # break the hub path.
        monkeypatch.setenv("AUTOVISION_WEIGHT_SOURCE", "modelscope")
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)

    def test_modelscope_offline_missing_cache_raises(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("AUTOVISION_WEIGHT_SOURCE", "modelscope")
        monkeypatch.setenv("AUTOVISION_OFFLINE", "1")
        self._ms_artifact(monkeypatch)

        def no_cache(ms_id: str, local_files_only: bool = False) -> Path:
            raise CheckpointError("no cached snapshot")

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_modelscope_snapshot",
            no_cache,
        )
        with pytest.raises(CheckpointError):
            load_pretrained_verified(FakeModelCls, "org/model")

    def test_auto_default_falls_back_to_modelscope_on_hub_failure(
        self, monkeypatch, tmp_path
    ):
        # Default source ("auto"): hub first, ModelScope mirror on failure.
        self._ms_artifact(monkeypatch)

        def hub_fail(*args: Any, **kwargs: Any) -> Path:
            raise RuntimeError("connection refused")

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            hub_fail,
        )
        calls = self._fake_ms_download(monkeypatch, tmp_path)
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert calls == ["merve/model"]

    def test_auto_default_prefers_hub_when_available(
        self, monkeypatch, tmp_path
    ):
        # Hub works -> the ModelScope mirror must not be touched.
        self._ms_artifact(monkeypatch)
        ms_calls: list[str] = []

        def ms_download(ms_id: str, local_files_only: bool = False) -> Path:
            ms_calls.append(ms_id)
            raise AssertionError("fallback must not run")

        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_modelscope_snapshot",
            ms_download,
        )
        snapshot = tmp_path / "hub"
        snapshot.mkdir()
        (snapshot / "config.json").write_bytes(b"{}")
        (snapshot / "model.safetensors").write_bytes(GOOD_CONTENT)
        monkeypatch.setattr(
            "autovision.encoders.weight_verification._download_snapshot",
            lambda *a, **k: snapshot,
        )
        model = load_pretrained_verified(FakeModelCls, "org/model")
        assert isinstance(model, FakeModelCls)
        assert FakeModelCls.calls[-1] == snapshot

    def test_render_manifest_emits_modelscope_id(self):
        rendered = _render_manifest(
            {"org/model": make_artifact(modelscope_id="merve/model")}
        )
        assert "modelscope_id='merve/model'" in rendered
        # Entries without a mirror keep the old shape (no None noise).
        plain = _render_manifest({"org/model": make_artifact()})
        assert "modelscope_id" not in plain


def _write_snapshot(root: Path, content: bytes, write: bool = True) -> Path:
    snapshot = root / "cache" / "models--org--model" / "snap"
    if write:
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b"{}")
        (snapshot / "model.safetensors").write_bytes(content)
    return snapshot
