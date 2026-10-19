# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Machine-readable run manifest written at the start of every training run.

The checkpoint answers "which weights"; the manifest answers "which command,
code revision, environment and stack produced them". It deliberately records
only a small environment allowlist (no full ``os.environ`` snapshot), so
credentials or unrelated host secrets never leak into a run directory.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from autovision._version import __version__
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = ["RunManifest", "build_run_manifest", "write_run_manifest"]

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "run_manifest.json"

_ENV_ALLOWLIST = (
    "AUTOVISION_CACHE_DIR",
    "AUTOVISION_DATA_CACHE_DIR",
    "AUTOVISION_DETERMINISTIC",
    "AUTOVISION_LOG_LEVEL",
    "AUTOVISION_OFFLINE",
    "AUTOVISION_REQUIRE_DATASET_HASHES",
    "AUTOVISION_WEIGHT_SOURCE",
    # Deliberately omitted: AUTOVISION_MIRROR_URL (may embed credentials),
    # AUTOVISION_SERVING_API_KEY, AUTOVISION_TELEMETRY_ENDPOINT.
    "CUDA_VISIBLE_DEVICES",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "SLURM_JOB_ID",
    "SLURM_NNODES",
    "SLURM_PROCID",
)

_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pydantic",
    "pyarrow",
    "pycocotools",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "openvino",
)


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    created_at: str
    resume: bool
    framework_version: str
    python: str
    platform: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    packages: dict[str, str]
    git_commit: str | None
    cuda_available: bool
    cuda_version: str | None
    gpus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, out_dir: str | Path, *, resume: bool | None = None) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        sweep_stale_temp_files(directory, "run_manifest*.tmp")
        is_resume = self.resume if resume is None else resume
        if is_resume:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = directory / f"run_manifest.resume-{stamp}.json"
        else:
            path = directory / _MANIFEST_NAME
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    # An installed wheel lives under site-packages; only treat the package as
    # a checkout when it actually sits next to a .git entry.
    if not (repo_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _cuda_info() -> tuple[bool, str | None, list[str]]:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency
        return False, None, []
    try:
        available = bool(torch.cuda.is_available())
        version = getattr(torch.version, "cuda", None)
        gpus = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if available else []
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return False, None, []
    return available, version, gpus


def _environment() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in _ENV_ALLOWLIST
        if name in os.environ
    }


def build_run_manifest(*, resume: bool = False) -> RunManifest:
    cuda_available, cuda_version, gpus = _cuda_info()
    return RunManifest(
        schema_version=_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        resume=resume,
        framework_version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        command=list(sys.argv),
        cwd=str(Path.cwd()),
        env=_environment(),
        packages=_package_versions(),
        git_commit=_git_commit(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpus=gpus,
    )


def write_run_manifest(
    out_dir: str | Path, *, resume: bool = False
) -> Path:
    """Write ``run_manifest.json`` (or ``run_manifest.resume-*.json``)."""
    return build_run_manifest(resume=resume).save(out_dir)
