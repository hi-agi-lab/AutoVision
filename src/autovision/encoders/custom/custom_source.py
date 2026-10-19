# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Custom package: last resort for user-provided ``EncoderProtocol`` instances.

This package lists no models: it accepts any object implementing the
``EncoderProtocol`` protocol (duck-typed) and exports its state dict. It must
always be registered LAST so named packages are preferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from autovision.encoders.model_wrapper import (
    missing_encoder_members,
)
from autovision.encoders.package import ModelSource

__all__ = ["CustomSource"]


class CustomSource(ModelSource):
    """Last-resort package for user-provided ``EncoderProtocol`` instances."""

    name = "custom"

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        return (
            not isinstance(model, str)
            and not missing_encoder_members(model)
        )

    @classmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        state_dict = (
            model.state_dict()
            if isinstance(model, nn.Module)
            else model.raw_module().state_dict()
        )
        torch.save(state_dict, out)
        if log_example:
            class_name = type(model).__name__
            print(
                f"Exported custom model '{class_name}' state dict to '{out}'. "
                f"Rebuild the '{class_name}' instance and load the state dict "
                f"to reuse it."
            )
