# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for EOMT instance segmentation training (DINOv2 variant)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.semantic_segmentation.dinov2_eomt.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)

__all__ = ["Dinov2EomtInstanceSegmentationTrainArgs"]


class Dinov2EomtInstanceSegmentationTrainArgs(Dinov2EomtSemanticSegmentationTrainArgs):
    """Instance segmentation shares the EOMT recipe; instance queries are
    class-agnostic and matched with the HF Hungarian mask matcher."""

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 360_000

    num_top_queries: int = 100
    score_threshold: float = 0.05
    mask_threshold: float = 0.5
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)

    model_config = {"extra": "forbid"}
