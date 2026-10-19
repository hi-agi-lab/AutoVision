# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the model asset registry and downloadable model cache."""

import json
import threading

import pytest

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.registry import (
    ASSET_MANIFEST,
    DownloadableModelError,
    ModelRegistry,
    RegistryError,
    register_downloadable,
    resolve_downloadable,
)


@pytest.fixture()
def model_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"model-bytes-" + b"x" * 32)
    return path


# ── registration ────────────────────────────────────────────────────────


class TestRegister:
    def test_creates_manifest_and_layout(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(
            model_file, name="detector", version="1.0.0", metadata={"map": 0.42}
        )
        assert entry.stage == "dev"
        assert entry.sha256 == compute_file_sha256(model_file)
        assert entry.path.is_file()
        manifest = json.loads(
            (entry.path.parent / "manifest.json").read_text()
        )
        assert manifest["name"] == "detector"
        assert manifest["version"] == "1.0.0"
        assert manifest["metadata"] == {"map": 0.42}

    def test_versions_are_immutable(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="immutable"):
            registry.register(model_file, name="detector", version="1.0.0")

    def test_register_sweeps_stale_current_tmp(self, tmp_path, model_file):
        """A crashed writer's current.json temp residue is swept on rewrite."""
        import os

        registry = ModelRegistry(tmp_path / "registry")
        current_dir = tmp_path / "registry" / "dev" / "detector"
        current_dir.mkdir(parents=True)
        stale = current_dir / "current.tmp"
        stale.write_text("{ broken")
        stat = stale.stat()
        os.utime(stale, (stat.st_atime - 7200, stat.st_mtime - 7200))
        registry.register(model_file, name="detector", version="1.0.0")
        assert not stale.exists()
        assert (current_dir / "current.json").is_file()

    def test_missing_file_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="does not exist"):
            registry.register(tmp_path / "nope.pt", name="m", version="1.0.0")

    def test_invalid_key_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.register(model_file, name="a/b", version="1.0.0")

    def test_unknown_stage_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Unknown stage"):
            registry.register(
                model_file, name="m", version="1.0.0", stage="qa"
            )


# ── promotion gate ──────────────────────────────────────────────────────


class TestPromote:
    def test_single_step_promotion(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        entry = registry.promote(name="detector", version="1.0.0", to_stage="staging", gates=[])
        assert entry.stage == "staging"
        assert entry.sha256 == compute_file_sha256(model_file)
        entry = registry.promote(name="detector", version="1.0.0", to_stage="prod", gates=[])
        assert entry.stage == "prod"

    def test_skip_stage_rejected(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="not found in 'staging'"):
            registry.promote(name="detector", version="1.0.0", to_stage="prod", gates=[])

    def test_promotion_requires_existing_source(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="not found in 'dev'"):
            registry.promote(name="ghost", version="1.0.0", to_stage="staging", gates=[])

    def test_cannot_promote_to_dev(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="Cannot promote to 'dev'"):
            registry.promote(name="detector", version="1.0.0", to_stage="dev")

    def test_promotion_is_idempotent_guarded(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        registry.promote(name="detector", version="1.0.0", to_stage="staging", gates=[])
        with pytest.raises(RegistryError, match="immutable"):
            registry.promote(name="detector", version="1.0.0", to_stage="staging", gates=[])


# ── lookup ──────────────────────────────────────────────────────────────


class TestLookup:
    def test_get_latest_version(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        for version in ("1.0.0", "1.2.0", "1.10.0"):
            registry.register(model_file, name="detector", version=version)
        entry = registry.get(name="detector", stage="dev")
        assert entry.version == "1.10.0"

    def test_get_explicit_version(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        registry.register(model_file, name="detector", version="2.0.0")
        entry = registry.get(name="detector", stage="dev", version="1.0.0")
        assert entry.version == "1.0.0"

    def test_get_missing_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="No versions"):
            registry.get(name="ghost", stage="prod")

    def test_list_entries_filters_by_stage(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="a", version="1.0.0")
        registry.register(model_file, name="b", version="1.0.0")
        registry.promote(name="a", version="1.0.0", to_stage="staging", gates=[])
        assert {e.name for e in registry.list_entries(stage="dev")} == {"a", "b"}
        assert {e.name for e in registry.list_entries(stage="staging")} == {"a"}
        assert len(registry.list_entries()) == 3

    def test_corrupt_entry_raises(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(model_file, name="detector", version="1.0.0")
        entry.path.unlink()
        with pytest.raises(RegistryError, match="missing"):
            registry.get(name="detector", stage="dev", version="1.0.0")

    def test_get_verifies_artifact_sha(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(model_file, name="detector", version="1.0.0")
        entry.path.write_bytes(b"tampered")
        with pytest.raises(RegistryError, match="sha256 mismatch"):
            registry.get(name="detector", stage="dev", version="1.0.0")

    def test_get_verifies_sidecar_sha(self, tmp_path, model_file):
        source = tmp_path / "model.pt"
        source.write_bytes(b"sidecar-model")
        sidecar = source.with_name(source.name + ".parity.json")
        sidecar.write_text('{"passed": true}')
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(source, name="detector", version="1.0.0")
        assert entry.sidecars
        entry.path.with_name(entry.path.name + ".parity.json").write_text(
            '{"passed": false}'
        )
        with pytest.raises(RegistryError, match="sidecar sha256"):
            registry.get(name="detector", stage="dev", version="1.0.0")

    def test_audit_log_detects_tampering(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(model_file, name="detector", version="1.0.0")
        audit = registry.root / "audit.jsonl"
        audit.write_text(audit.read_text().replace('"register"', '"promote"'))
        with pytest.raises(RegistryError, match="hash mismatch"):
            registry.audit_log()


# ── downloadable models ─────────────────────────────────────────────────


class TestDownloadable:
    def test_unknown_name_raises(self, tmp_path):
        with pytest.raises(DownloadableModelError, match="Unknown downloadable"):
            resolve_downloadable("nope", cache_dir=tmp_path)

    def test_register_and_resolve_local_url(self, tmp_path):
        import hashlib

        source = tmp_path / "hosted.pt"
        payload = b"hosted-checkpoint"
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # A file:// URL keeps the download fully offline for the test.
        register_downloadable(
            "test-model", source.as_uri(), digest
        )
        try:
            path = resolve_downloadable("test-model", cache_dir=tmp_path / "cache")
            assert path.read_bytes() == payload
            # Second resolve hits the verified cache.
            again = resolve_downloadable("test-model", cache_dir=tmp_path / "cache")
            assert again == path
        finally:
            ASSET_MANIFEST.pop("test-model", None)

    def test_nested_name_creates_cache_parent(self, tmp_path):
        import hashlib

        source = tmp_path / "hosted.pt"
        payload = b"nested-checkpoint"
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        register_downloadable("nested/model", source.as_uri(), digest)
        try:
            cache = tmp_path / "cache"
            path = resolve_downloadable("nested/model", cache_dir=cache)
            assert path == cache / "nested" / "model.pt"
            assert path.read_bytes() == payload
        finally:
            ASSET_MANIFEST.pop("nested/model", None)

    def test_corrupt_cache_is_redownloaded(self, tmp_path):
        import hashlib

        source = tmp_path / "hosted.pt"
        payload = b"hosted-checkpoint"
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "test-corrupt.pt").write_bytes(b"corrupted")
        register_downloadable(
            "test-corrupt", source.as_uri(), digest
        )
        try:
            path = resolve_downloadable("test-corrupt", cache_dir=cache)
            assert path.read_bytes() == payload
        finally:
            ASSET_MANIFEST.pop("test-corrupt", None)


# ── mirror fallback (design doc 28 §6.3, regional mirrors) ──────────────


class TestMirrorFallback:
    def test_falls_back_to_mirror_on_primary_failure(self, monkeypatch, tmp_path):
        import hashlib

        from autovision.registry import downloadable as dl

        payload = b"weights"
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setitem(
            dl.ASSET_MANIFEST,
            "mirror-test",
            dl.DownloadableModel(
                url="https://primary.invalid/w.pt",
                sha256=sha,
                mirrors=("https://mirror.invalid/w.pt",),
            ),
        )
        attempted: list[str] = []

        def fake_download(url: str, target) -> None:
            attempted.append(url)
            if url.startswith("https://primary"):
                raise dl.DownloadableModelError(f"primary down: {url}")
            target.write_bytes(payload)

        monkeypatch.setattr(dl, "_download", fake_download)
        out = dl.resolve_downloadable("mirror-test", cache_dir=tmp_path)
        assert out.read_bytes() == payload
        assert attempted == ["https://primary.invalid/w.pt", "https://mirror.invalid/w.pt"]

    def test_mirror_sha_mismatch_tries_next(self, monkeypatch, tmp_path):
        import hashlib

        from autovision.registry import downloadable as dl

        payload = b"good"
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setitem(
            dl.ASSET_MANIFEST,
            "mirror-sha-test",
            dl.DownloadableModel(
                url="https://a.invalid/w.pt",
                sha256=sha,
                mirrors=("https://b.invalid/w.pt", "https://c.invalid/w.pt"),
            ),
        )

        def fake_download(url: str, target) -> None:
            target.write_bytes(b"corrupt" if "b." in url else payload)

        monkeypatch.setattr(dl, "_download", fake_download)
        out = dl.resolve_downloadable("mirror-sha-test", cache_dir=tmp_path)
        assert out.read_bytes() == payload


# ── hardening: key validation + cross-process mutation lock ─────────────


class TestRegistryHardening:
    @pytest.mark.parametrize("name", ["..", ".", "a/b", "a\\b", "", "  "])
    def test_invalid_names_are_rejected(self, tmp_path, model_file, name):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.register(model_file, name=name, version="1.0.0")

    @pytest.mark.parametrize("version", ["..", ".", "1/2", "1\\2", "", "  "])
    def test_invalid_versions_are_rejected(self, tmp_path, model_file, version):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.register(model_file, name="detector", version=version)

    def test_lookup_paths_validate_too(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.get(name="..", version="escape")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.exists(name="..", version="escape", stage="dev")
        with pytest.raises(RegistryError, match="Invalid registry key"):
            registry.current(name="..", stage="prod")

    def test_concurrent_registration_has_exactly_one_winner(self, tmp_path, model_file):
        registry = ModelRegistry(tmp_path / "registry")
        barrier = threading.Barrier(2)
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            try:
                registry.register(model_file, name="detector", version="1.0.0")
                outcome = "ok"
            except RegistryError:
                outcome = "error"
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert sorted(outcomes) == ["error", "ok"]
        # The winning entry is complete and readable.
        entry = registry.get(name="detector", version="1.0.0", stage="dev")
        assert entry.path.is_file()
        assert entry.sha256 == compute_file_sha256(model_file)


def test_promote_defaults_to_fail_closed_gates(tmp_path, model_file):
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model_file, name="ungated", version="1.0.0")
    with pytest.raises(RegistryError, match="Deployment gates failed|missing model card"):
        registry.promote(name="ungated", version="1.0.0", to_stage="staging")
    # Nothing is promoted when the default gates fail.
    assert not registry.exists(name="ungated", version="1.0.0", stage="staging")
