# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the MLflow registry backend (same contract as ModelRegistry).

The contract tests run against an in-memory fake client implementing the
MLflow client subset the backend uses, so they need no ``mlflow`` install.
The two integration tests run against a real local file store and are
gated behind ``AUTOVISION_MLFLOW_TESTS=1`` (mlflow is an optional extra).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autovision.registry.mlflow_registry import MLflowModelRegistry
from autovision.registry.model_registry import RegistryError

# ── fake MLflow client ──────────────────────────────────────────────────


class _FakeMlflowException(Exception):
    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass
class _FakeRunInfo:
    run_id: str
    artifact_uri: str


@dataclass
class _FakeRun:
    info: _FakeRunInfo


@dataclass
class _FakeExperiment:
    experiment_id: str
    name: str


@dataclass
class _FakeModelVersion:
    name: str
    version: str
    source: str
    run_id: str | None
    tags: dict[str, str] = field(default_factory=dict)
    creation_timestamp: int = 1_700_000_000_000


class FakeMlflowClient:
    """In-memory MlflowClient subset used by MLflowModelRegistry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.experiments: dict[str, _FakeExperiment] = {}
        self._next_experiment = 0
        self.runs: dict[str, dict[str, bytes]] = {}
        self._next_run = 0
        self.model_versions: list[_FakeModelVersion] = []
        self._next_version = 0
        self.deleted: list[tuple[str, str]] = []

    # experiments / runs
    def get_experiment_by_name(self, name: str) -> _FakeExperiment | None:
        return self.experiments.get(name)

    def create_experiment(self, name: str) -> str:
        if name in self.experiments:
            raise _FakeMlflowException(
                f"Experiment '{name}' already exists.",
                error_code="RESOURCE_ALREADY_EXISTS",
            )
        self._next_experiment += 1
        experiment = _FakeExperiment(str(self._next_experiment), name)
        self.experiments[name] = experiment
        return experiment.experiment_id

    def create_run(self, experiment_id: str) -> _FakeRun:
        self._next_run += 1
        run_id = f"run-{self._next_run}"
        artifact_uri = (self.root / run_id / "artifacts").as_uri()
        self.runs[run_id] = {}
        return _FakeRun(_FakeRunInfo(run_id, artifact_uri))

    def get_run(self, run_id: str) -> _FakeRun:
        return _FakeRun(_FakeRunInfo(run_id, (self.root / run_id / "artifacts").as_uri()))

    def log_artifact(
        self, run_id: str, local_path: str, artifact_path: str | None = None
    ) -> None:
        assert artifact_path is None  # the backend logs at the run root
        self.runs[run_id][Path(local_path).name] = Path(local_path).read_bytes()

    # registered models / versions
    def create_registered_model(self, name: str) -> None:
        if any(mv.name == name for mv in self.model_versions):
            raise _FakeMlflowException(
                f"Registered model '{name}' already exists.",
                error_code="RESOURCE_ALREADY_EXISTS",
            )
        # A model with zero versions cannot be distinguished from a missing
        # one here; _ensure_registered_model tolerates that (idempotent).

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _FakeModelVersion:
        self._next_version += 1
        mv = _FakeModelVersion(
            name=name,
            version=str(self._next_version),
            source=source,
            run_id=run_id,
            tags=dict(tags or {}),
        )
        self.model_versions.append(mv)
        return mv

    def search_model_versions(self, filter_string: str = "") -> list[_FakeModelVersion]:
        if not filter_string.strip():
            return list(self.model_versions)
        assert filter_string.startswith("name = '") and filter_string.endswith("'")
        name = filter_string[len("name = '"):-1]
        return [mv for mv in self.model_versions if mv.name == name]

    def delete_model_version(self, name: str, version: str) -> None:
        self.deleted.append((name, version))
        self.model_versions = [
            mv
            for mv in self.model_versions
            if not (mv.name == name and mv.version == version)
        ]

    def download_artifacts(
        self, run_id: str, path: str, dst_path: str | None = None
    ) -> str:
        data = self.runs[run_id][Path(path).name]
        target = self.root / "downloads" / run_id / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)


# ── fixtures / helpers ──────────────────────────────────────────────────


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "assets" / "model.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weights-bytes-v1")
    return path


@pytest.fixture()
def registry(tmp_path: Path) -> MLflowModelRegistry:
    return MLflowModelRegistry(
        client=FakeMlflowClient(tmp_path),
        experiment_name="autovision-tests",
    )


def _corrupt_run_artifact(client: FakeMlflowClient, run_id: str) -> None:
    client.runs[run_id]["model.pt"] = b"tampered"


class TestRegister:
    def test_creates_entry_with_tags_and_artifact(self, registry, model_file):
        entry = registry.register(
            model_file,
            name="detector",
            version="1.0.0",
            stage="dev",
            metadata={"task": "detection"},
        )
        assert entry.name == "detector"
        assert entry.version == "1.0.0"
        assert entry.stage == "dev"
        assert entry.path.is_file()
        assert entry.sha256 and len(entry.sha256) == 64
        assert entry.metadata == {"task": "detection"}
        assert entry.registered_at.endswith("Z")
        client: FakeMlflowClient = registry.client
        assert len(client.model_versions) == 1
        mv = client.model_versions[0]
        assert mv.tags["autovision.version"] == "1.0.0"
        assert mv.tags["autovision.stage"] == "dev"
        assert mv.tags["autovision.sha256"] == entry.sha256
        assert json.loads(mv.tags["autovision.metadata"]) == {"task": "detection"}
        assert mv.source.startswith("file://")

    def test_duplicate_version_in_stage_rejected(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0")
        with pytest.raises(RegistryError, match="immutable"):
            registry.register(model_file, name="detector", version="1.0.0")

    def test_same_version_across_stages_allowed(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        entry = registry.register(
            model_file, name="detector", version="1.0.0", stage="staging"
        )
        assert entry.stage == "staging"
        assert registry.exists(name="detector", version="1.0.0", stage="dev")
        assert registry.exists(name="detector", version="1.0.0", stage="staging")

    def test_missing_model_file_rejected(self, registry, tmp_path):
        with pytest.raises(RegistryError, match="does not exist"):
            registry.register(tmp_path / "nope.pt", name="detector", version="1.0.0")

    def test_invalid_stage_rejected(self, registry, model_file):
        with pytest.raises(RegistryError, match="Unknown stage"):
            registry.register(model_file, name="detector", version="1.0.0", stage="qa")

    def test_corrupt_artifact_aborts_and_cleans_up(self, registry, model_file):
        client: FakeMlflowClient = registry.client
        original_log = client.log_artifact

        def _log_then_corrupt(run_id: str, local_path: str, artifact_path=None):
            original_log(run_id, local_path, artifact_path)
            _corrupt_run_artifact(client, run_id)

        client.log_artifact = _log_then_corrupt
        with pytest.raises(RegistryError, match="sha256"):
            registry.register(model_file, name="detector", version="1.0.0")
        # The half-registered version is deleted (immutability not poisoned).
        assert client.deleted == [("detector", "1")]
        assert registry.list_entries() == []


class TestPromote:
    def test_dev_to_staging_creates_new_version(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        promoted = registry.promote(
            name="detector", version="1.0.0", to_stage="staging", gates=[]
        )
        assert promoted.stage == "staging"
        assert promoted.version == "1.0.0"
        assert promoted.path.read_bytes() == b"weights-bytes-v1"
        client: FakeMlflowClient = registry.client
        assert len(client.model_versions) == 2
        stages = {mv.tags["autovision.stage"] for mv in client.model_versions}
        assert stages == {"dev", "staging"}

    def test_skip_stage_rejected(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        with pytest.raises(RegistryError, match="one stage at a time"):
            registry.promote(name="detector", version="1.0.0", to_stage="prod")

    def test_missing_source_rejected(self, registry):
        with pytest.raises(RegistryError, match="not found in 'dev'"):
            registry.promote(name="ghost", version="1.0.0", to_stage="staging")

    def test_promote_to_dev_rejected(self, registry, model_file):
        with pytest.raises(RegistryError, match="Cannot promote to 'dev'"):
            registry.promote(name="detector", version="1.0.0", to_stage="dev")

    def test_default_gates_fail_closed_without_model_card(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        with pytest.raises(RegistryError, match="Deployment gates failed"):
            registry.promote(name="detector", version="1.0.0", to_stage="staging")
        assert not registry.exists(
            name="detector", version="1.0.0", stage="staging"
        )

    def test_duplicate_target_rejected(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        registry.register(model_file, name="detector", version="1.0.0", stage="staging")
        with pytest.raises(RegistryError, match="immutable"):
            registry.promote(name="detector", version="1.0.0", to_stage="staging")

    def test_corrupt_source_aborts(self, registry, model_file):
        entry = registry.register(model_file, name="detector", version="1.0.0")
        client: FakeMlflowClient = registry.client
        run_id = client.model_versions[0].run_id
        assert run_id is not None
        _corrupt_run_artifact(client, run_id)
        with pytest.raises(RegistryError, match="promotion aborted"):
            registry.promote(name="detector", version="1.0.0", to_stage="staging")
        # Nothing was created in the target stage.
        assert not registry.exists(name="detector", version="1.0.0", stage="staging")
        assert entry.sha256  # original tag untouched


class TestLookup:
    def test_get_latest_resolves_semantic_max(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        registry.register(model_file, name="detector", version="0.9.0", stage="dev")
        registry.register(model_file, name="detector", version="1.10.0", stage="dev")
        entry = registry.get(name="detector", stage="dev")
        assert entry.version == "1.10.0"  # numeric, not lexicographic

    def test_get_explicit_version(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        entry = registry.get(name="detector", stage="dev", version="1.0.0")
        assert entry.path.read_bytes() == b"weights-bytes-v1"
        assert entry.sha256

    def test_get_missing_version_rejected(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        with pytest.raises(RegistryError, match="not found in 'dev'"):
            registry.get(name="detector", stage="dev", version="9.9.9")

    def test_get_empty_stage_rejected(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        with pytest.raises(RegistryError, match="No versions"):
            registry.get(name="detector", stage="prod")

    def test_exists(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        assert registry.exists(name="detector", version="1.0.0", stage="dev")
        assert not registry.exists(name="detector", version="2.0.0", stage="dev")

    def test_list_entries_filters_by_stage(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        registry.register(model_file, name="detector", version="1.0.0", stage="staging")
        registry.register(model_file, name="classifier", version="2.0.0", stage="dev")
        everything = registry.list_entries()
        assert [(e.name, e.stage, e.version) for e in everything] == [
            ("classifier", "dev", "2.0.0"),
            ("detector", "dev", "1.0.0"),
            ("detector", "staging", "1.0.0"),
        ]
        staged = registry.list_entries(stage="staging")
        assert [(e.name, e.version) for e in staged] == [("detector", "1.0.0")]


class TestClientSeam:
    def test_missing_mlflow_package_error_message(self, monkeypatch, tmp_path):
        """Lazy import failure surfaces the install guidance."""
        import builtins

        real_import = builtins.__import__

        def _no_mlflow(name, *args, **kwargs):
            if name == "mlflow" or name.startswith("mlflow."):
                raise ModuleNotFoundError("No module named 'mlflow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_mlflow)
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"weights")
        registry = MLflowModelRegistry(tracking_uri=str(tmp_path / "mlruns"))
        with pytest.raises(RegistryError, match="pip install"):
            registry.register(model_file, name="detector", version="1.0.0")

    def test_tracking_uri_stored(self, tmp_path):
        registry = MLflowModelRegistry(tracking_uri=str(tmp_path / "mlruns"))
        assert registry.tracking_uri == str(tmp_path / "mlruns")


# ── real-mlflow integration (gated) ─────────────────────────────────────


def _require_real_mlflow() -> None:
    pytest.importorskip("mlflow", reason="mlflow extra not installed (pip install -e '.[mlflow]')")
    if os.environ.get("AUTOVISION_MLFLOW_TESTS") != "1":
        pytest.skip("set AUTOVISION_MLFLOW_TESTS=1 to run real-mlflow tests")


class TestRealMlflowFileStore:
    def test_register_get_promote_roundtrip(self, tmp_path, model_file):
        _require_real_mlflow()
        registry = MLflowModelRegistry(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="autovision-it",
        )
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        entry = registry.get(name="detector", stage="dev", version="1.0.0")
        assert entry.path.read_bytes() == b"weights-bytes-v1"
        promoted = registry.promote(
            name="detector", version="1.0.0", to_stage="staging", gates=[]
        )
        assert promoted.stage == "staging"
        assert promoted.path.read_bytes() == b"weights-bytes-v1"

    def test_promotion_gate_enforced(self, tmp_path, model_file):
        _require_real_mlflow()
        registry = MLflowModelRegistry(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="autovision-it",
        )
        registry.register(model_file, name="detector", version="1.0.0", stage="dev")
        with pytest.raises(RegistryError, match="one stage at a time"):
            registry.promote(name="detector", version="1.0.0", to_stage="prod")


class TestRegistryHardening:
    def test_sidecars_are_logged_and_downloaded(self, registry, model_file):
        card = Path(str(model_file) + ".modelcard.json")
        card.write_text('{"license": "Apache-2.0"}')
        entry = registry.register(model_file, name="detector", version="1.0.0")
        local_card = Path(str(entry.path) + ".modelcard.json")
        assert local_card.is_file()
        assert local_card.read_text() == '{"license": "Apache-2.0"}'

    def test_get_rejects_tampered_artifact(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0")
        client: FakeMlflowClient = registry.client
        run_id = client.model_versions[0].run_id
        assert run_id is not None
        _corrupt_run_artifact(client, run_id)
        with pytest.raises(RegistryError, match="failed sha256"):
            registry.get(name="detector", stage="dev", version="1.0.0")

    def test_list_entries_skips_foreign_model_versions(self, registry, model_file):
        registry.register(model_file, name="detector", version="1.0.0")
        client: FakeMlflowClient = registry.client
        client.create_model_version("foreign", "file:///tmp/x.pt", None, {})
        entries = registry.list_entries()
        assert [entry.name for entry in entries] == ["detector"]


def test_gates_run_against_downloaded_sidecars(registry, model_file):
    card = Path(str(model_file) + ".modelcard.json")
    card.write_text('{"license": "Apache-2.0"}')
    registry.register(model_file, name="detector", version="1.0.0", stage="dev")

    class _SeesCard:
        name = "sees_card"

        def check(self, entry):
            sidecar = Path(str(entry.path) + ".modelcard.json")
            return [] if sidecar.is_file() else ["model card sidecar missing"]

    promoted = registry.promote(
        name="detector", version="1.0.0", to_stage="staging", gates=[_SeesCard()]
    )
    assert promoted.stage == "staging"
