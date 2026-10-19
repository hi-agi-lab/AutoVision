# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection transform family (torchvision v2 based, step scheduled)."""

from autovision.augment.detection.args import (
    OrientedObjectDetectionTransformArgs,
    RTDETRDetectionTransformArgs,
    ScaleJitterArgs,
)
from autovision.augment.detection.collate import (
    RTDETRDetectionCollateFunction,
    TwoViewDetectionCollateFunction,
)
from autovision.augment.detection.oriented_transform import (
    OrientedObjectDetectionCollateFunction,
    OrientedObjectDetectionTransform,
    build_oriented_val_transform_args,
)
from autovision.augment.detection.schedule import (
    StepSchedule,
    resolve_detection_step_schedule,
)
from autovision.augment.detection.transform import (
    RTDETRDetectionTransform,
    StepScheduledCompose,
    build_val_transform_args,
)

__all__ = [
    "OrientedObjectDetectionCollateFunction",
    "OrientedObjectDetectionTransform",
    "OrientedObjectDetectionTransformArgs",
    "RTDETRDetectionCollateFunction",
    "RTDETRDetectionTransform",
    "RTDETRDetectionTransformArgs",
    "ScaleJitterArgs",
    "StepSchedule",
    "StepScheduledCompose",
    "TwoViewDetectionCollateFunction",
    "build_oriented_val_transform_args",
    "build_val_transform_args",
    "resolve_detection_step_schedule",
]
