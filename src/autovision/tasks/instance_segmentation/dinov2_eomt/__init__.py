# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation (DINOv2 variant): encoder-only mask
transformer with class-agnostic queries and Hungarian mask matching."""

from autovision.tasks.instance_segmentation.dinov2_eomt.config import (
    Dinov2EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.instance_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtInstanceSegmentation,
)
from autovision.tasks.instance_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtInstanceSegmentationTrain,
)

__all__ = [
    "Dinov2EomtInstanceSegmentation",
    "Dinov2EomtInstanceSegmentationTrain",
    "Dinov2EomtInstanceSegmentationTrainArgs",
]
