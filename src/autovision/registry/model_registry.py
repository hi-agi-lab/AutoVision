# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model asset registry: filesystem backend with dev/staging/prod promotion.

The registry stores immutable model versions in a plain directory tree::

    {root}/dev/{name}/{version}/<model file>
    {root}/dev/{name}/{version}/manifest.json
    {root}/staging/...
    {root}/prod/...

Design (asset & deployment layer, §12.6):
- Backend abstraction: the filesystem backend is the reference; a future
  MLflow backend implements the same :class:`ModelRegistry` contract.
- Immutable versions: registering over an existing version raises.
- Promotion gate: assets move one stage at a time (dev -> staging -> prod);
  skipping a stage or moving backwards is rejected. Promoted artifacts are
  byte-identical (sha256 re-verified after copy).

The download-and-verify asset counterpart is
:mod:`autovision.registry.downloadable`: name -> (url, sha256) entries
with verified download + cache.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock, Timeout

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.errors import AutoVisionError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = [
    "MODEL_STAGES",
    "RegistryEntry",
    "RegistryError",
    "ModelRegistry",
]

MODEL_STAGES = ("dev", "staging", "prod")
RegistryStage = Literal["dev", "staging", "prod"]

_MANIFEST_NAME = "manifest.json"


class RegistryError(AutoVisionError):
    """Raised for registry layout, immutability or promotion-gate violations."""


@dataclass
class RegistryEntry:
    """One immutable model asset in the registry."""

    name: str
    version: str
    stage: str
    path: Path
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""
    sidecars: dict[str, str] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path.name)
        return payload

    @classmethod
    def from_manifest(
        cls, payload: dict[str, Any], entry_dir: Path
    ) -> "RegistryEntry":
        data = dict(payload)
        data["path"] = entry_dir / str(data.pop("path"))
        return cls(**data)


def _version_sort_key(version: str) -> tuple[int, ...] | tuple[str, ...]:
    """Semantic-ish sort: numeric versions sort naturally, others fallback."""
    parts = version.split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return (version,)


_LOCK_TIMEOUT_SEC = 30.0


def _locked(method):
    """Serialize registry operations across processes/threads.

    The filesystem registry is shared state: writes must not race, and a
    reader must not observe a half-copied entry (the manifest is written
    last). ``FileLock`` is reentrant inside one thread, so nested
    ``_write_current``/``_append_audit`` calls do not deadlock.
    """

    @functools.wraps(method)
    def wrapper(self: "ModelRegistry", *args: Any, **kwargs: Any) -> Any:
        try:
            # FileLock serializes processes; the RLock serializes threads
            # sharing this instance (FileLock alone is reentrant per thread
            # and would let two threads through on the same fd). Both are
            # reentrant, so nested _write_current/_append_audit calls work.
            with self._thread_lock, self._lock:
                return method(self, *args, **kwargs)
        except Timeout as error:
            raise RegistryError(
                f"Timed out after {_LOCK_TIMEOUT_SEC:.0f}s waiting for the "
                f"registry lock '{self._lock_path}'."
            ) from error

    return wrapper


def _validate_key_part(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} "
            "(must be a non-empty string)."
        )
    if value != value.strip():
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} "
            "(leading/trailing whitespace)."
        )
    if value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} (path separators, NUL, "
            "'/' and '..' are not allowed)."
        )


def _validate_key(name: str, version: str) -> None:
    _validate_key_part("name", name)
    _validate_key_part("version", version)


class ModelRegistry:
    """Filesystem-backed model asset registry with a promotion gate."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".registry.lock"
        self._lock = FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_SEC)
        self._thread_lock = threading.RLock()

    # ── registration ────────────────────────────────────────────────────
    @_locked
    def register(
        self,
        model_file: Path | str,
        *,
        name: str,
        version: str,
        stage: RegistryStage = "dev",
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> RegistryEntry:
        """Register a model file as an immutable (name, version, stage).

        Model-card and postprocess sidecars next to the file are copied
        alongside the artifact so deployment gates can read them.
        """
        self._validate_stage(stage)
        source = Path(model_file)
        if not source.is_file():
            raise RegistryError(f"Model file '{source}' does not exist.")
        entry_dir = self._entry_dir(name, stage, version)
        if entry_dir.exists():
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{stage}'; versions are immutable."
            )
        target = entry_dir / source.name
        entry_dir.mkdir(parents=True)
        try:
            shutil.copy2(source, target)
            sidecars = self._copy_sidecars(source, target)
            entry = RegistryEntry(
                name=name,
                version=version,
                stage=stage,
                path=target,
                sha256=compute_file_sha256(target),
                metadata=dict(metadata or {}),
                registered_at=_utc_now(),
                sidecars=sidecars,
            )
            self._write_manifest(entry_dir, entry)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise
        if self._read_current(name, stage) is None:
            self._write_current(name, stage, version)
        self._append_audit(
            action="register", name=name, version=version, stage=stage,
            actor=actor,
        )
        return entry

    # ── promotion gate ──────────────────────────────────────────────────
    @_locked
    def promote(
        self,
        *,
        name: str,
        version: str,
        to_stage: RegistryStage = "staging",
        gates: Sequence[Any] | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> RegistryEntry:
        """Promote an entry one stage forward (dev -> staging -> prod).

        ``gates`` are evaluated against the promoted copy; any failure
        removes the target and raises. The stage's ``current`` pointer is
        updated only after the gates pass. ``gates=None`` applies the
        default deployment gates (fail-closed); pass ``gates=[]`` only for
        an explicit, audited bypass.
        """
        self._validate_stage(to_stage)
        if to_stage == "dev":
            raise RegistryError("Cannot promote to 'dev'.")
        if gates is None:
            from autovision.registry.deployment_gates import default_gates

            gates = default_gates()
        # Promotion gate: exactly one stage forward (dev -> staging -> prod).
        source_stage = MODEL_STAGES[MODEL_STAGES.index(to_stage) - 1]
        if not self.exists(name=name, version=version, stage=source_stage):
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in "
                f"'{source_stage}'; promotion must move one stage at a time."
            )
        source_entry = self.get(name=name, version=version, stage=source_stage)
        target_dir = self._entry_dir(name, to_stage, version)
        if target_dir.exists():
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{to_stage}'; versions are immutable."
            )
        target_dir.mkdir(parents=True)
        target = target_dir / Path(source_entry.path).name
        try:
            shutil.copy2(source_entry.path, target)
            sidecars = self._copy_sidecars(source_entry.path, target)
            if compute_file_sha256(target) != source_entry.sha256:
                raise RegistryError(
                    f"Promoted artifact '{target}' failed sha256 verification; "
                    f"promotion aborted."
                )
            for sidecar_name, expected_sha in source_entry.sidecars.items():
                copied_sidecar = target_dir / sidecar_name
                if (
                    not copied_sidecar.is_file()
                    or compute_file_sha256(copied_sidecar) != expected_sha
                ):
                    raise RegistryError(
                        f"Promoted sidecar '{sidecar_name}' failed sha256 "
                        "verification; promotion aborted."
                    )
            entry = RegistryEntry(
                name=name,
                version=version,
                stage=to_stage,
                path=target,
                sha256=source_entry.sha256,
                metadata=dict(source_entry.metadata),
                registered_at=_utc_now(),
                sidecars=sidecars,
            )
            if gates:
                failures: list[str] = []
                for gate in gates:
                    gate_name = getattr(gate, "name", type(gate).__name__)
                    try:
                        problems = gate.check(entry)
                    except Exception as error:  # noqa: BLE001 - gate boundary
                        problems = [f"{type(error).__name__}: {error}"]
                    failures.extend(f"{gate_name}: {problem}" for problem in problems)
                if failures:
                    raise RegistryError(
                        "Deployment gates failed: " + "; ".join(failures)
                    )
            self._write_manifest(target_dir, entry)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        self._write_current(name, to_stage, version)
        self._append_audit(
            action="promote", name=name, version=version, stage=to_stage,
            actor=actor, reason=reason,
            details={"from_stage": source_stage},
        )
        return entry

    # ── lookup ──────────────────────────────────────────────────────────
    @_locked
    def get(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        version: str | None = None,
    ) -> RegistryEntry:
        """Load an entry; ``version=None`` resolves the latest in the stage."""
        self._validate_stage(stage)
        _validate_key_part("name", name)
        if version is not None:
            _validate_key_part("version", version)
        stage_dir = self.root / stage / name
        if version is None:
            versions = self._list_versions(stage_dir)
            if not versions:
                raise RegistryError(f"No versions of '{name}' in '{stage}'.")
            version = max(versions, key=_version_sort_key)
        entry_dir = self._entry_dir(name, stage, version)
        if not entry_dir.is_dir():
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in '{stage}'."
            )
        return self._read_entry(entry_dir)

    @_locked
    def exists(
        self, *, name: str, version: str, stage: RegistryStage
    ) -> bool:
        self._validate_stage(stage)
        return self._entry_dir(name, stage, version).is_dir()

    @_locked
    def list_entries(self, stage: RegistryStage | None = None) -> list[RegistryEntry]:
        """List entries, optionally filtered by stage."""
        stages = [stage] if stage is not None else list(MODEL_STAGES)
        entries: list[RegistryEntry] = []
        for current_stage in stages:
            self._validate_stage(current_stage)
            stage_root = self.root / current_stage
            if not stage_root.is_dir():
                continue
            for name_dir in sorted(stage_root.iterdir()):
                for version_dir in sorted(name_dir.iterdir()):
                    if (version_dir / _MANIFEST_NAME).is_file():
                        entries.append(self._read_entry(version_dir))
        return entries

    # ── deployment pointer / rollback / audit ───────────────────────────
    @_locked
    def current(
        self, *, name: str, stage: RegistryStage = "prod"
    ) -> RegistryEntry:
        """Resolve the stage's ``current`` pointer (latest as a fallback)."""
        self._validate_stage(stage)
        version = self._read_current(name, stage)
        if version is not None and self.exists(name=name, version=version, stage=stage):
            return self.get(name=name, stage=stage, version=version)
        return self.get(name=name, stage=stage)

    @_locked
    def rollback(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        to_version: str,
        reason: str | None = None,
        actor: str | None = None,
    ) -> RegistryEntry:
        """Point ``current`` at an existing older version (no file copy)."""
        self._validate_stage(stage)
        if not self.exists(name=name, version=to_version, stage=stage):
            raise RegistryError(
                f"Rollback target '{name}' v{to_version} not found in '{stage}'."
            )
        entry = self.get(name=name, stage=stage, version=to_version)
        self._write_current(name, stage, to_version)
        self._append_audit(
            action="rollback", name=name, version=to_version, stage=stage,
            actor=actor, reason=reason,
        )
        return entry

    @_locked
    def audit_log(self) -> list[dict[str, Any]]:
        """Read and verify the append-only audit log (empty when absent)."""
        path = self.root / "audit.jsonl"
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        previous_hash = ""
        hashed = False
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise RegistryError(
                    f"Malformed audit log line {line_number}: {error}."
                ) from error
            if "entry_hash" not in entry:
                if hashed:
                    raise RegistryError(
                        f"Audit log line {line_number} is unhashed after "
                        "hashed entries."
                    )
                entries.append(entry)
                continue
            hashed = True
            payload = {key: value for key, value in entry.items() if key != "entry_hash"}
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if entry["entry_hash"] != expected:
                raise RegistryError(
                    f"Audit log line {line_number} hash mismatch (tampered)."
                )
            if entry.get("prev_hash", "") != previous_hash:
                raise RegistryError(
                    f"Audit log line {line_number} breaks the hash chain."
                )
            previous_hash = expected
            entries.append(entry)
        return entries

    # ── internals ───────────────────────────────────────────────────────
    def _entry_dir(self, name: str, stage: str, version: str) -> Path:
        _validate_key(name, version)
        return self.root / stage / name / version

    def _read_entry(self, entry_dir: Path) -> RegistryEntry:
        manifest = entry_dir / _MANIFEST_NAME
        if not manifest.is_file():
            raise RegistryError(f"Corrupt registry entry: '{entry_dir}'.")
        try:
            entry = RegistryEntry.from_manifest(
                json.loads(manifest.read_text()), entry_dir
            )
        except (OSError, ValueError, TypeError) as error:
            raise RegistryError(
                f"Corrupt registry manifest '{manifest}': {error}."
            ) from error
        root = entry_dir.resolve()
        resolved = entry.path.resolve()
        if not resolved.is_relative_to(root):
            raise RegistryError(
                f"Registry manifest path escapes its entry directory: "
                f"'{entry.path}'."
            )
        if not resolved.is_file():
            raise RegistryError(
                f"Registered model file missing: '{entry.path}'."
            )
        actual = compute_file_sha256(resolved)
        if actual != entry.sha256:
            raise RegistryError(
                f"Registry artifact sha256 mismatch for '{resolved}': "
                f"{actual} != {entry.sha256}."
            )
        for sidecar_name, expected_sha in entry.sidecars.items():
            sidecar = (root / sidecar_name).resolve()
            if not sidecar.is_relative_to(root) or not sidecar.is_file():
                raise RegistryError(
                    f"Registered sidecar missing or outside the entry: "
                    f"'{sidecar_name}'."
                )
            if compute_file_sha256(sidecar) != expected_sha:
                raise RegistryError(
                    f"Registry sidecar sha256 mismatch for '{sidecar_name}'."
                )
        entry.path = resolved
        return entry

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in MODEL_STAGES:
            raise RegistryError(
                f"Unknown stage '{stage}'; expected one of {MODEL_STAGES}."
            )

    @staticmethod
    def _list_versions(stage_dir: Path) -> list[str]:
        if not stage_dir.is_dir():
            return []
        return [
            path.name
            for path in stage_dir.iterdir()
            if path.is_dir() and (path / _MANIFEST_NAME).is_file()
        ]

    @staticmethod
    def _copy_sidecars(source: Path, target: Path) -> dict[str, str]:
        """Copy sidecars and return ``{filename: sha256}`` for the manifest."""
        hashes: dict[str, str] = {}
        for suffix in (".modelcard.json", ".postprocess.json", ".parity.json"):
            sidecar = Path(str(source) + suffix)
            if sidecar.is_file():
                copied = Path(str(target) + suffix)
                shutil.copy2(sidecar, copied)
                hashes[copied.name] = compute_file_sha256(copied)
        return hashes

    @staticmethod
    def _write_manifest(entry_dir: Path, entry: RegistryEntry) -> None:
        """Write the manifest atomically; a crash never leaves a partial one."""
        target = entry_dir / _MANIFEST_NAME
        sweep_stale_temp_files(entry_dir, f".{glob.escape(_MANIFEST_NAME)}.*.tmp")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{_MANIFEST_NAME}.", suffix=".tmp", dir=entry_dir
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            temp_path.write_text(
                json.dumps(entry.to_manifest(), indent=2, sort_keys=True)
            )
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)

    def _current_path(self, name: str, stage: str) -> Path:
        _validate_key_part("name", name)
        return self.root / stage / name / "current.json"

    def _read_current(self, name: str, stage: str) -> str | None:
        path = self._current_path(name, stage)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        version = payload.get("version")
        return str(version) if version else None

    @_locked
    def _write_current(self, name: str, stage: str, version: str) -> None:
        path = self._current_path(name, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The temp file below is ``path.with_suffix(".tmp")`` — i.e.
        # "{stem}.tmp", NOT "{name}.tmp".
        sweep_stale_temp_files(path.parent, f"{glob.escape(path.stem)}.tmp")
        payload = {"version": version, "updated_at": _utc_now()}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)

    @_locked
    def _append_audit(
        self,
        *,
        action: str,
        name: str,
        version: str,
        stage: str,
        actor: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        path = self.root / "audit.jsonl"
        previous_hash = ""
        if path.is_file():
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            if lines:
                try:
                    previous_hash = str(json.loads(lines[-1]).get("entry_hash", ""))
                except json.JSONDecodeError as error:
                    raise RegistryError(
                        "Refusing to append to a malformed audit log: "
                        f"{error}."
                    ) from error
        entry: dict[str, Any] = {
            "timestamp": _utc_now(),
            "action": action,
            "name": name,
            "version": version,
            "stage": stage,
            "actor": actor or os.environ.get("AUTOVISION_ACTOR", "local"),
            "reason": reason,
            "details": dict(details or {}),
            "prev_hash": previous_hash,
        }
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        entry["entry_hash"] = hashlib.sha256(payload.encode()).hexdigest()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
