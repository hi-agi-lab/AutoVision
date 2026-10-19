# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR oriented object detection: model configuration."""

from __future__ import annotations

from typing import Any

from autovision.config.base import PydanticConfig
from autovision.tasks.detection.rtdetr.config import (
    RTDETRObjectDetectionTrainArgs,
)

__all__ = ["RTDETROrientedObjectDetectionTrainArgs"]


class RTDETROrientedObjectDetectionTrainArgs(
    RTDETRObjectDetectionTrainArgs, PydanticConfig
):
    """Training/architecture args for the oriented RT-DETR (OBB) family.

    Inherits the full RT-DETRv2 recipe; only the decode-time thresholds are
    OBB-specific (rotated NMS operates on exact rotated IoU).
    """

    model_config = {"extra": "forbid"}

    #: Validation score gate applied before the per-class rotated NMS.
    score_threshold: float = 0.05
    #: Rotated-NMS IoU threshold (per-class, exact rotated IoU).
    nms_iou_threshold: float = 0.65

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
