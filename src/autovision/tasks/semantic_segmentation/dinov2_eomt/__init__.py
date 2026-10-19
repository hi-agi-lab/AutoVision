# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation (DINOv2 variant): encoder-only mask
transformer trained end to end via the HF implementation."""

from autovision.tasks.semantic_segmentation.dinov2_eomt.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtSemanticSegmentationTrain,
)

__all__ = [
    "Dinov2EomtSemanticSegmentation",
    "Dinov2EomtSemanticSegmentationTrain",
    "Dinov2EomtSemanticSegmentationTrainArgs",
]
