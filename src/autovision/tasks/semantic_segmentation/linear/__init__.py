# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Linear semantic segmentation task model package."""

from autovision.tasks.semantic_segmentation.linear.config import (
    LinearSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.linear.task_model import (
    LinearSemanticSegmentation,
)
from autovision.tasks.semantic_segmentation.linear.train_model import (
    LinearSemanticSegmentationTrain,
)

__all__ = [
    "LinearSemanticSegmentation",
    "LinearSemanticSegmentationTrain",
    "LinearSemanticSegmentationTrainArgs",
]
