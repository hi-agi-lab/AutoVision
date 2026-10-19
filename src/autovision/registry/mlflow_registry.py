# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""MLflow backend for the model asset registry.

Implements the exact :class:`autovision.registry.model_registry.ModelRegistry`
contract (``register`` / ``promote`` / ``get`` / ``exists`` / ``list_entries``)
on top of the MLflow model registry, so backends are interchangeable.

Mapping (design docs 08-assets-and-deployment §12.6):

- One MLflow registered model per registry ``name``; one MLflow model
  version per ``(name, version, stage)`` triple.
- Stages ride in the ``autovision.stage`` tag (``dev``/``staging``/``prod``)
  instead of MLflow's legacy stage enum, which has no ``dev`` value and is
  deprecated since MLflow 2.9.
- The logical ``version`` string rides in ``autovision.version``; MLflow's
  own version numbers are server-assigned integers and are ignored.
- ``autovision.sha256`` and ``autovision.metadata`` (JSON) carry the entry
  metadata; ``registered_at`` is derived from the model version creation
  timestamp.
- Artifacts are logged to a dedicated experiment (one run per registration)
  so the model file is owned by MLflow, not by the caller's filesystem.
  Registration and promotion both verify the artifact sha256 by downloading
  it back (byte-identical guarantee, mirroring the filesystem backend's
  copy + verify).

MLflow is an optional dependency (``autovision[mlflow]``); it is imported
lazily on first client use. Tests inject a client object implementing the
used subset instead of standing up a tracking server.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.registry.model_registry import (
    MODEL_STAGES,
    RegistryEntry,
    RegistryError,
    RegistryStage,
    _version_sort_key,
)

__all__ = ["MLflowModelRegistry"]

_TAG_PREFIX = "autovision."
_TAG_VERSION = _TAG_PREFIX + "version"
_TAG_STAGE = _TAG_PREFIX + "stage"
_TAG_SHA256 = _TAG_PREFIX + "sha256"
_TAG_METADATA = _TAG_PREFIX + "metadata"

_RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
_SIDECAR_SUFFIXES = (".modelcard.json", ".postprocess.json")


def _strip_file_scheme(uri: str) -> Path:
    """Best-effort local path for an artifact URI (``file://`` unwrapped)."""
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    return Path(uri)


class MLflowModelRegistry:
    """Registry backend backed by an MLflow tracking server / file store.

    Args:
        tracking_uri: MLflow tracking URI; ``None`` uses the MLflow default
            (local ``./mlruns`` file store).
        experiment_name: Experiment receiving the registered artifacts.
        client: Pre-built client (test seam). When given, it must implement
            the MLflow client subset used below and MLflow itself is never
            imported.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        experiment_name: str = "autovision-registry",
        client: Any | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._client_override = client

    # ── client seam ─────────────────────────────────────────────────────
    @property
    def client(self) -> Any:
        if self._client_override is None:
            self._client_override = self._new_client(self.tracking_uri)
        return self._client_override

    @staticmethod
    def _new_client(tracking_uri: str | None) -> Any:
        try:
            from mlflow import MlflowClient
        except ModuleNotFoundError as error:
            raise RegistryError(
                "The MLflow registry backend requires the 'mlflow' package; "
                "install it with: pip install mlflow (or pip install "
                "autovision[mlflow])."
            ) from error
        return MlflowClient(tracking_uri=tracking_uri)

    # ── registration ────────────────────────────────────────────────────
    def register(
        self,
        model_file: Path | str,
        *,
        name: str,
        version: str,
        stage: RegistryStage = "dev",
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEntry:
        """Register a model file as an immutable (name, version, stage)."""
        _validate_stage(stage)
        source = Path(model_file)
        if not source.is_file():
            raise RegistryError(f"Model file '{source}' does not exist.")
        client = self.client
        versions = self._search(client, name)
        if any(
            v.tags.get(_TAG_VERSION) == version and v.tags.get(_TAG_STAGE) == stage
            for v in versions
        ):
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{stage}'; versions are immutable."
            )
        run_id = self._log_artifact(client, source)
        sha256 = compute_file_sha256(source)
        run = client.get_run(run_id)
        source_uri = f"{run.info.artifact_uri.rstrip('/')}/{source.name}"
        created = self._create_model_version(
            client,
            name=name,
            source=source_uri,
            run_id=run_id,
            tags=_version_tags(version, stage, sha256, metadata),
        )
        # Byte-identical guarantee: download the artifact back and verify.
        local = self._download(client, created)
        if compute_file_sha256(local) != sha256:
            client.delete_model_version(name, created.version)
            raise RegistryError(
                f"Registered artifact '{source_uri}' failed sha256 "
                f"verification; registration aborted."
            )
        return _entry(name, created, local)

    # ── promotion gate ──────────────────────────────────────────────────
    def promote(
        self,
        *,
        name: str,
        version: str,
        to_stage: RegistryStage = "staging",
        gates: list[Any] | None = None,
    ) -> RegistryEntry:
        """Promote an entry one stage forward (dev -> staging -> prod).

        Artifact sidecars (model card / postprocess spec) are downloaded
        alongside the model, so the same deployment gates as the filesystem
        backend run against the promoted copy. ``gates=None`` applies the
        default gates (fail-closed); ``gates=[]`` is an explicit audited
        bypass.
        """
        _validate_stage(to_stage)
        if gates is None:
            from autovision.registry.deployment_gates import default_gates

            gates = default_gates()
        if to_stage == "dev":
            raise RegistryError("Cannot promote to 'dev'.")
        client = self.client
        versions = self._search(client, name)
        source_stage = MODEL_STAGES[MODEL_STAGES.index(to_stage) - 1]
        source_mv = _find(versions, version, source_stage)
        if source_mv is None:
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in "
                f"'{source_stage}'; promotion must move one stage at a time."
            )
        if _find(versions, version, to_stage) is not None:
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{to_stage}'; versions are immutable."
            )
        # Promoted artifacts are byte-identical: re-verify the source before
        # creating the promoted model version.
        local = self._download(client, source_mv)
        sha256 = source_mv.tags.get(_TAG_SHA256, "")
        if compute_file_sha256(local) != sha256:
            raise RegistryError(
                f"Source artifact for '{name}' v{version} failed sha256 "
                f"verification; promotion aborted."
            )
        source_entry = _entry(name, source_mv, local)
        failures: list[str] = []
        for gate in gates:
            gate_name = getattr(gate, "name", type(gate).__name__)
            try:
                problems = gate.check(source_entry)
            except Exception as error:  # noqa: BLE001 - gate boundary
                problems = [f"{type(error).__name__}: {error}"]
            failures.extend(f"{gate_name}: {problem}" for problem in problems)
        if failures:
            raise RegistryError("Deployment gates failed: " + "; ".join(failures))
        metadata = _load_metadata(source_mv.tags)
        created = self._create_model_version(
            client,
            name=name,
            source=source_mv.source,
            run_id=source_mv.run_id,
            tags=_version_tags(version, to_stage, sha256, metadata),
        )
        promoted_local = self._download(client, created)
        return _entry(name, created, promoted_local)

    # ── lookup ──────────────────────────────────────────────────────────
    def get(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        version: str | None = None,
    ) -> RegistryEntry:
        """Load an entry; ``version=None`` resolves the latest in the stage."""
        _validate_stage(stage)
        client = self.client
        versions = self._search(client, name)
        in_stage = [
            v
            for v in versions
            if _is_autovision_version(v) and v.tags.get(_TAG_STAGE) == stage
        ]
        if not in_stage:
            raise RegistryError(f"No versions of '{name}' in '{stage}'.")
        if version is None:
            mv = max(in_stage, key=lambda v: _version_sort_key(v.tags[_TAG_VERSION]))
        else:
            mv = _find(in_stage, version, stage)
            if mv is None:
                raise RegistryError(
                    f"Registry entry '{name}' v{version} not found in '{stage}'."
                )
        local = self._download(client, mv)
        declared = mv.tags.get(_TAG_SHA256, "")
        if declared and compute_file_sha256(local) != declared:
            raise RegistryError(
                f"MLflow artifact for '{name}' v{mv.version} failed sha256 "
                "verification; refusing the read."
            )
        return _entry(name, mv, local)

    def exists(
        self, *, name: str, version: str, stage: RegistryStage
    ) -> bool:
        _validate_stage(stage)
        return (
            _find(self._search(self.client, name), version, stage) is not None
        )

    def list_entries(self, stage: RegistryStage | None = None) -> list[RegistryEntry]:
        """List AutoVision entries, optionally filtered by stage.

        MLflow registries may hold unrelated models: versions without the
        framework tags are skipped rather than raising ``KeyError``.
        """
        versions = self._search(self.client)
        entries = [
            _entry(v.name, v, _strip_file_scheme(v.source))
            for v in versions
            if _is_autovision_version(v)
            and (stage is None or v.tags.get(_TAG_STAGE) == stage)
        ]
        entries.sort(
            key=lambda e: (e.name, MODEL_STAGES.index(e.stage), e.version)
        )
        return entries

    # ── internals ───────────────────────────────────────────────────────
    def _search(self, client: Any, name: str | None = None) -> list[Any]:
        """Search model versions; ``name=None`` searches across all names."""
        filter_string = f"name = '{name}'" if name is not None else ""
        try:
            return list(client.search_model_versions(filter_string))
        except Exception as error:  # noqa: BLE001 - wrapped as RegistryError
            raise RegistryError(
                f"MLflow backend error while searching model versions: {error}"
            ) from error

    def _log_artifact(self, client: Any, source: Path) -> str:
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self._create_experiment(client, self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        run = client.create_run(experiment_id)
        try:
            client.log_artifact(run.info.run_id, str(source))
            for suffix in _SIDECAR_SUFFIXES:
                sidecar = Path(str(source) + suffix)
                if sidecar.is_file():
                    client.log_artifact(run.info.run_id, str(sidecar))
        except Exception as error:  # noqa: BLE001
            raise RegistryError(
                f"MLflow backend error while logging artifact '{source}': {error}"
            ) from error
        return run.info.run_id

    @staticmethod
    def _create_experiment(client: Any, name: str) -> str:
        try:
            return client.create_experiment(name)
        except Exception as error:  # noqa: BLE001
            if getattr(error, "error_code", None) == _RESOURCE_ALREADY_EXISTS:
                experiment = client.get_experiment_by_name(name)
                if experiment is not None:
                    return experiment.experiment_id
            raise RegistryError(
                f"MLflow backend error while creating experiment '{name}': {error}"
            ) from error

    def _create_model_version(
        self,
        client: Any,
        *,
        name: str,
        source: str,
        run_id: str | None,
        tags: dict[str, str],
    ) -> Any:
        self._ensure_registered_model(client, name)
        try:
            return client.create_model_version(
                name, source, run_id=run_id, tags=tags
            )
        except Exception as error:  # noqa: BLE001
            raise RegistryError(
                f"MLflow backend error while creating model version for "
                f"'{name}': {error}"
            ) from error

    @staticmethod
    def _ensure_registered_model(client: Any, name: str) -> None:
        try:
            client.create_registered_model(name)
        except Exception as error:  # noqa: BLE001
            if getattr(error, "error_code", None) == _RESOURCE_ALREADY_EXISTS:
                return
            raise RegistryError(
                f"MLflow backend error while creating registered model "
                f"'{name}': {error}"
            ) from error

    @staticmethod
    def _download(client: Any, mv: Any) -> Path:
        """Materialize the model artifact and its sidecars locally."""
        if not mv.run_id:
            return _strip_file_scheme(mv.source)
        try:
            local = Path(client.download_artifacts(mv.run_id, _artifact_name(mv)))
        except Exception as error:  # noqa: BLE001
            raise RegistryError(
                f"MLflow backend error while downloading artifact for "
                f"'{mv.name}' v{mv.version}: {error}"
            ) from error
        for suffix in _SIDECAR_SUFFIXES:
            try:
                client.download_artifacts(mv.run_id, _artifact_name(mv) + suffix)
            except Exception:  # noqa: BLE001 - sidecars are optional
                continue
        return local


def _artifact_name(mv: Any) -> str:
    """Relative artifact path: the model file sits at the run artifact root."""
    return mv.source.rstrip("/").rsplit("/", 1)[-1]


def _validate_stage(stage: str) -> None:
    if stage not in MODEL_STAGES:
        raise RegistryError(
            f"Unknown stage '{stage}'; expected one of {MODEL_STAGES}."
        )


def _find(versions: list[Any], version: str, stage: str) -> Any | None:
    for mv in versions:
        if (
            _is_autovision_version(mv)
            and mv.tags.get(_TAG_VERSION) == version
            and mv.tags.get(_TAG_STAGE) == stage
        ):
            return mv
    return None


def _is_autovision_version(mv: Any) -> bool:
    """True for versions written by this backend (tags are complete)."""
    return (
        bool(mv.tags.get(_TAG_VERSION))
        and mv.tags.get(_TAG_STAGE) in MODEL_STAGES
        and bool(mv.tags.get(_TAG_SHA256))
    )


def _version_tags(
    version: str, stage: str, sha256: str, metadata: dict[str, Any] | None
) -> dict[str, str]:
    tags = {
        _TAG_VERSION: version,
        _TAG_STAGE: stage,
        _TAG_SHA256: sha256,
    }
    if metadata:
        tags[_TAG_METADATA] = json.dumps(metadata, sort_keys=True)
    return tags


def _load_metadata(tags: dict[str, str]) -> dict[str, Any]:
    raw = tags.get(_TAG_METADATA)
    return json.loads(raw) if raw else {}


def _entry(name: str, mv: Any, local_path: Path) -> RegistryEntry:
    created_ms = getattr(mv, "creation_timestamp", 0) or 0
    return RegistryEntry(
        name=name,
        version=mv.tags[_TAG_VERSION],
        stage=mv.tags[_TAG_STAGE],
        path=local_path,
        sha256=mv.tags[_TAG_SHA256],
        metadata=_load_metadata(mv.tags),
        registered_at=datetime.fromtimestamp(
            created_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
