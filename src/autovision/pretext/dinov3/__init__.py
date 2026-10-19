# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv3: DINOv2 post-training + PaKA dense-relational auxiliary loss."""

from autovision.pretext.dinov3.dinov3 import (
    DINOv3,
    DINOv3Args,
    DINOv3Transform,
    DINOv3TransformArgs,
)

__all__ = ["DINOv3", "DINOv3Args", "DINOv3Transform", "DINOv3TransformArgs"]
