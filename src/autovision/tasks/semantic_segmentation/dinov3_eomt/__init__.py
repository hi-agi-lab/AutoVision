# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation (DINOv3 variant): encoder-only mask
transformer backed by the HF ``EomtDinov3`` implementation."""

from autovision.tasks.semantic_segmentation.dinov3_eomt.config import (
    Dinov3EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtSemanticSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.train_model import (
    Dinov3EomtSemanticSegmentationTrain,
)

__all__ = [
    "Dinov3EomtSemanticSegmentation",
    "Dinov3EomtSemanticSegmentationTrain",
    "Dinov3EomtSemanticSegmentationTrainArgs",
]
