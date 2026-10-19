# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR package (extras-only dependency)."""

from autovision.encoders.rfdetr.rfdetr import RFDETRModelWrapper
from autovision.encoders.rfdetr.rfdetr_source import RFDETR_SOURCE

__all__ = ["RFDETR_SOURCE", "RFDETRModelWrapper"]
