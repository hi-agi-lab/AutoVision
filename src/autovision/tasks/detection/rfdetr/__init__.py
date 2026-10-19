# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR object detection task model (HF transformers backend)."""

from autovision.tasks.detection.rfdetr.config import (
    RFDETRObjectDetectionTrainArgs,
    RFDETRSSLArgs,
    resolve_rfdetr_variant,
)
from autovision.tasks.detection.rfdetr.task_model import (
    RFDETRObjectDetection,
)
from autovision.tasks.detection.rfdetr.train_model import (
    RFDETRObjectDetectionTrain,
)

__all__ = [
    "RFDETRObjectDetection",
    "RFDETRObjectDetectionTrain",
    "RFDETRObjectDetectionTrainArgs",
    "RFDETRSSLArgs",
    "resolve_rfdetr_variant",
]
