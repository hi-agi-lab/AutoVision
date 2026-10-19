# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Ultralytics (YOLO/RT-DETR) package. AGPL-3.0 — extras-only dependency."""

from autovision.encoders.ultralytics.ultralytics import UltralyticsModelWrapper
from autovision.encoders.ultralytics.ultralytics_source import ULTRALYTICS_SOURCE

__all__ = ["ULTRALYTICS_SOURCE", "UltralyticsModelWrapper"]
