# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT panoptic segmentation (DINOv2 variant): encoder-only mask
transformer decoding queries into per-pixel (class, segment) maps."""

from autovision.tasks.panoptic_segmentation.dinov2_eomt.config import (
    Dinov2EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.panoptic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtPanopticSegmentation,
)
from autovision.tasks.panoptic_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtPanopticSegmentationTrain,
)

__all__ = [
    "Dinov2EomtPanopticSegmentation",
    "Dinov2EomtPanopticSegmentationTrain",
    "Dinov2EomtPanopticSegmentationTrainArgs",
]
