# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Command layer: public entry points and assembly helpers."""

from autovision.cli.train import PretrainConfig, PretrainResult, pretrain
from autovision.cli.train_task import (
    TaskTrainResult,
    train_depth_estimation,
    train_image_classification,
    train_instance_segmentation,
    train_object_detection,
    train_oriented_object_detection,
    train_panoptic_segmentation,
    train_pose_estimation,
    train_semantic_segmentation,
)

__all__ = [
    "PretrainConfig",
    "PretrainResult",
    "TaskTrainResult",
    "pretrain",
    "train_image_classification",
    "train_instance_segmentation",
    "train_object_detection",
    "train_oriented_object_detection",
    "train_panoptic_segmentation",
    "train_semantic_segmentation",
    "train_depth_estimation",
    "train_pose_estimation",
]
