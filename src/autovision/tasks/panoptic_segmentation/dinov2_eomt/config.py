# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for EOMT panoptic segmentation training (DINOv2 variant)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from autovision.tasks.semantic_segmentation.dinov2_eomt.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)

__all__ = ["Dinov2EomtPanopticSegmentationTrainArgs"]


class Dinov2EomtPanopticSegmentationTrainArgs(Dinov2EomtSemanticSegmentationTrainArgs):
    """Panoptic segmentation shares the EOMT recipe; ``num_thing_classes``
    splits the contiguous internal classes into things (first) and stuffs
    (rest), which drives stuff-segment merging and the PQ metric.

    ``num_blocks`` defaults to 3 following the baseline per-scale
    registry (s/b -> 3, l -> 4, g -> 5); override for larger backbones.
    The scale jitter lower bound follows the baseline panoptic transforms
    (0.1 instead of the semantic 0.5).
    """

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 90_000

    num_thing_classes: int
    num_blocks: int = 3

    # Evaluation thresholds (baseline naming: threshold ~ mask score
    # confidence, mask_threshold ~ binarization, mask_overlap_threshold ~
    # minimum retained area ratio per segment).
    threshold: float = 0.8
    mask_threshold: float = 0.5
    mask_overlap_threshold: float = 0.8

    scale_jitter: tuple[float, float] = Field(default=(0.1, 2.0), strict=False)

    model_config = {"extra": "forbid"}
