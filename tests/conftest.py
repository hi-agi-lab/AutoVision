# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared pytest fixtures: env isolation and synthetic data factories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every VTRAIN cache dir to a per-test temp directory."""
    cache_dir = tmp_path / "autovision-cache"
    monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(cache_dir))
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_framework_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all AUTOVISION_* and rank variables so tests start from defaults."""
    for name in list(os.environ):
        if name.startswith("AUTOVISION_") and not name.endswith("_TESTS"):
            monkeypatch.delenv(name, raising=False)
    for name in (
        "RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "NODE_RANK",
        "GROUP_RANK",
        "JSM_NAMESPACE_RANK",
        "JSM_NAMESPACE_LOCAL_RANK",
    ):
        monkeypatch.delenv(name, raising=False)


def create_image(
    path: Path, size: tuple[int, int] = (32, 32), seed: int = 0
) -> Path:
    """Create a deterministic random RGB image at ``path`` (parents created)."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    try:
        import torchvision.io as tio

        path = path.with_suffix(".jpg")
        tio.write_jpeg(torch_from_numpy(array), str(path), quality=90)
    except Exception:  # pragma: no cover - fallback without torchvision write
        from PIL import Image

        path = path.with_suffix(".png")
        Image.fromarray(array).save(path)
    return path


def torch_from_numpy(array: np.ndarray):
    import torch

    return torch.from_numpy(array).permute(2, 0, 1)


def create_images(
    directory: Path, num_images: int = 8, size: tuple[int, int] = (32, 32)
) -> list[Path]:
    """Create ``num_images`` deterministic images spread over subdirectories."""
    paths = []
    for index in range(num_images):
        subdir = directory / f"class_{index % 2}"
        paths.append(create_image(subdir / f"image_{index:03d}.jpg", size=size, seed=index))
    return paths
