# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for EOMT instance segmentation training (DINOv3 variant)."""

from __future__ import annotations

from typing import ClassVar

from autovision.tasks.semantic_segmentation.dinov3_eomt.config import (
    Dinov3EomtSemanticSegmentationTrainArgs,
)

__all__ = ["Dinov3EomtInstanceSegmentationTrainArgs"]


class Dinov3EomtInstanceSegmentationTrainArgs(Dinov3EomtSemanticSegmentationTrainArgs):
    """Instance segmentation shares the DINOv3 EOMT recipe; instance queries
    are class-agnostic and matched with the HF Hungarian mask matcher."""

    default_steps: ClassVar[int] = 360_000

    num_top_queries: int = 100
    score_threshold: float = 0.05
    mask_threshold: float = 0.5
