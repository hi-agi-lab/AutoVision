# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR oriented object detection family (YOLO-OBB training)."""

from autovision.tasks.oriented_detection.rtdetr.config import (
    RTDETROrientedObjectDetectionTrainArgs,
)
from autovision.tasks.oriented_detection.rtdetr.task_model import (
    RTDETROrientedObjectDetection,
)
from autovision.tasks.oriented_detection.rtdetr.train_model import (
    RTDETROrientedObjectDetectionTrain,
)

__all__ = [
    "RTDETROrientedObjectDetection",
    "RTDETROrientedObjectDetectionTrain",
    "RTDETROrientedObjectDetectionTrainArgs",
]
