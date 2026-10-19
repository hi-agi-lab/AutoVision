# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Debug configuration for training-instability diagnostics."""

from __future__ import annotations

import pathlib

from pydantic import ConfigDict

from autovision.config.base import PydanticConfig

__all__ = ["DebugArgs", "DebugUnderflowOverflowArgs", "NaNCaptureArgs"]


class NaNCaptureArgs(PydanticConfig):
    """Capture a self-contained replay snapshot at the first non-finite loss
    or gradient: model weights, optimizer state, microbatches, and RNG states
    are saved before the run aborts."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_microbatches: int = 21

    def output_dir(self, out_dir: pathlib.Path) -> pathlib.Path:
        return out_dir / "debug" / "nan_capture"


class DebugUnderflowOverflowArgs(PydanticConfig):
    """Per-module activation min/max tracing (HF DebugUnderflowOverflow
    style): on the first inf/nan activation the last frames are dumped and
    the run optionally aborts."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_frames_to_save: int = 21
    trace_batch_nums: list[int] = []
    abort_after_batch_num: int | None = None

    def output_dir(self, out_dir: pathlib.Path) -> pathlib.Path:
        return out_dir / "debug" / "underflow_overflow"


class DebugArgs(PydanticConfig):
    model_config = ConfigDict(extra="forbid")

    nan_capture: NaNCaptureArgs | None = None
    underflow_overflow: DebugUnderflowOverflowArgs | None = None
