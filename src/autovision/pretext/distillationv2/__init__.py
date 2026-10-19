# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Distillation V2: intermediate-layer MSE (no queue, no normalization)."""

from autovision.pretext.distillationv2.distillationv2 import (
    DistillationV2,
    DistillationV2AdamWArgs,
    DistillationV2Args,
    DistillationV2LARSArgs,
    DistillationV2Transform,
    DistillationV2TransformArgs,
)

__all__ = [
    "DistillationV2",
    "DistillationV2AdamWArgs",
    "DistillationV2Args",
    "DistillationV2LARSArgs",
    "DistillationV2Transform",
    "DistillationV2TransformArgs",
]
