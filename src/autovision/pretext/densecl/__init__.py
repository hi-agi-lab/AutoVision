# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DenseCL: single-image contrastive learning with dense correspondence."""

from autovision.pretext.densecl.densecl import (
    DenseCL,
    DenseCLArgs,
    DenseCLSGDArgs,
)

__all__ = ["DenseCL", "DenseCLArgs", "DenseCLSGDArgs"]
