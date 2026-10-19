# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT panoptic segmentation (DINOv3 variant)."""

from autovision.tasks.panoptic_segmentation.dinov3_eomt.config import (
    Dinov3EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.panoptic_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtPanopticSegmentation,
)
from autovision.tasks.panoptic_segmentation.dinov3_eomt.train_model import (
    Dinov3EomtPanopticSegmentationTrain,
)

__all__ = [
    "Dinov3EomtPanopticSegmentation",
    "Dinov3EomtPanopticSegmentationTrain",
    "Dinov3EomtPanopticSegmentationTrainArgs",
]
