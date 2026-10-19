# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation (DINOv3 variant)."""

from autovision.tasks.instance_segmentation.dinov3_eomt.config import (
    Dinov3EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.instance_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtInstanceSegmentation,
)
from autovision.tasks.instance_segmentation.dinov3_eomt.train_model import (
    Dinov3EomtInstanceSegmentationTrain,
)

__all__ = [
    "Dinov3EomtInstanceSegmentation",
    "Dinov3EomtInstanceSegmentationTrain",
    "Dinov3EomtInstanceSegmentationTrainArgs",
]
