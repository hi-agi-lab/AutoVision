# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv3 package backed by HuggingFace transformers (gated)."""

from autovision.encoders.dinov3.dinov3_source import DINOV3_SOURCE
from autovision.encoders.dinov3.dinov3_wrapper import (
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)

__all__ = ["DINOV3_SOURCE", "TransformersDinov3ConvNextWrapper", "TransformersDinov3ViTWrapper"]
