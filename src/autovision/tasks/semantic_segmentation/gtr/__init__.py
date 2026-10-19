# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR semantic segmentation task model package (roadmap doc 31 §11 S1)."""

from autovision.tasks.semantic_segmentation.gtr.config import GTRSemSegTrainArgs
from autovision.tasks.semantic_segmentation.gtr.task_model import (
    GTRSemanticSegmentation,
)
from autovision.tasks.semantic_segmentation.gtr.train_model import GTRSemSegTrain

__all__ = [
    "GTRSemSegTrain",
    "GTRSemSegTrainArgs",
    "GTRSemanticSegmentation",
]
