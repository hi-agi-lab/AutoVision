# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR instance-segmentation task model package (roadmap doc 31 §11 S2)."""

from autovision.tasks.instance_segmentation.gtr.config import GTRSegTrainArgs
from autovision.tasks.instance_segmentation.gtr.task_model import (
    GTRSegObjectDetection,
)
from autovision.tasks.instance_segmentation.gtr.train_model import GTRSegTrain

__all__ = [
    "GTRSegObjectDetection",
    "GTRSegTrain",
    "GTRSegTrainArgs",
]
