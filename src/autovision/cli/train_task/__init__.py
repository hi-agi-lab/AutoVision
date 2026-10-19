# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task training commands: detection, classification, segmentation finetuning."""

from autovision.cli.train_task.depth_estimation import (
    train_depth_estimation,
)
from autovision.cli.train_task.image_classification import (
    train_image_classification,
)
from autovision.cli.train_task.instance_segmentation import (
    train_instance_segmentation,
)
from autovision.cli.train_task.object_detection import train_object_detection
from autovision.cli.train_task.oriented_object_detection import (
    train_oriented_object_detection,
)
from autovision.cli.train_task.panoptic_segmentation import (
    train_panoptic_segmentation,
)
from autovision.cli.train_task.pose_estimation import train_pose_estimation
from autovision.cli.train_task.semantic_segmentation import (
    train_semantic_segmentation,
)
from autovision.cli.train_task.task_helpers import TaskTrainResult

__all__ = [
    "TaskTrainResult",
    "train_image_classification",
    "train_instance_segmentation",
    "train_object_detection",
    "train_oriented_object_detection",
    "train_panoptic_segmentation",
    "train_semantic_segmentation",
    "train_depth_estimation",
    "train_pose_estimation",
]
