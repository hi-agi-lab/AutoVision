# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv2 package backed by HuggingFace transformers."""

from autovision.encoders.dinov2.dinov2_source import DINOV2_SOURCE
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper

__all__ = ["DINOV2_SOURCE", "TransformersDinov2Wrapper"]
