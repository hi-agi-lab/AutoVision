# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Collection of system and environment information for run metadata."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from autovision._version import __version__
from autovision.runtime.accelerator import npu_available

__all__ = ["GitInformation", "SystemInformation", "get_system_information"]


@dataclass(frozen=True)
class GitInformation:
    branch: str
    commit: str
    uncommitted_changes: bool


@dataclass(frozen=True)
class SystemInformation:
    platform: str
    python_version: str
    autovision_version: str
    torch_version: str
    torchvision_version: str
    cuda_version: str | None
    cpu_count: int
    gpus: list[str]
    git_info_autovision: GitInformation | None
    git_info_current_directory: GitInformation | None


def _get_git_information(directory: Path) -> GitInformation | None:
    """Best-effort git introspection; any failure degrades to ``None``."""
    if not directory.is_dir():
        return None
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return GitInformation(branch=branch, commit=commit, uncommitted_changes=bool(dirty))
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_system_information() -> SystemInformation:
    cuda_version = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(f"{props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    elif npu_available():
        for index in range(torch.npu.device_count()):
            gpus.append(f"{torch.npu.get_device_name(index)} (NPU)")
    return SystemInformation(
        platform=platform.platform(),
        python_version=platform.python_version(),
        autovision_version=__version__,
        torch_version=torch.__version__,
        torchvision_version=_package_version("torchvision") or "unknown",
        cuda_version=cuda_version,
        cpu_count=os.cpu_count() or 0,
        gpus=gpus,
        git_info_autovision=_get_git_information(
            Path(__file__).parent.parent.parent.parent
        ),
        git_info_current_directory=_get_git_information(Path.cwd()),
    )
