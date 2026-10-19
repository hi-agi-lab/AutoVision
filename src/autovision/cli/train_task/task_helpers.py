# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared task-training helpers: result type, save-checkpoint args, dataset sizing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autovision.config.base import PydanticConfig

__all__ = ["TaskTrainResult"]


class _SaveCheckpointArgs(PydanticConfig):
    save_every_num_steps: int = 1000
    save_last: bool = True
    save_best: bool = True


@dataclass
class TaskTrainResult:
    out_dir: Path
    steps_run: int
    checkpoint_path: Path | None
    best_checkpoint_path: Path | None
    exported_model_path: Path | None
    best_exported_model_path: Path | None
    best_metrics: dict[str, float]
    classes: dict[int, str]
    stopped_early: bool = False
    stop_step: int | None = None


def _dataset_size(data_args, split: str) -> int:
    return sum(1 for _ in data_args.list_image_info(split))
