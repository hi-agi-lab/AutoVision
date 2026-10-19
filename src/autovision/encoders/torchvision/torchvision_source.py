# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Torchvision package: resnet/convnext families.

Models are built **untrained by default** (no weight download); pretrained
weights require an explicit ``model_args={"weights": "DEFAULT"}`` and network
access. This mirrors the baseline torchvision package behavior.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from autovision.encoders.package import CatalogModelSource
from autovision.encoders.torchvision.wrappers import (
    _FEATURE_EXTRACTORS,
    TorchvisionModelWrapper,
)

__all__ = ["TORCHVISION_SOURCE", "TorchvisionSource"]

_LICENSE_LABEL = "bsd"


def _adapt_input_channels(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    from autovision.encoders._model_helpers import adapt_conv_input_channels

    return adapt_conv_input_channels(weight, in_chans)


class TorchvisionSource(CatalogModelSource):
    name = "torchvision"

    @classmethod
    def catalog_names(cls) -> list[str]:
        try:
            import torchvision.models as torchvision_models
        except ImportError:  # pragma: no cover
            return []
        pattern = "|".join(
            wrapper._torchvision_model_name_pattern for wrapper in _FEATURE_EXTRACTORS
        )
        return [
            f"{cls.name}/{name}"
            for name in torchvision_models.list_models()
            if re.fullmatch(f"({pattern})", name)
        ]

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        if isinstance(model, TorchvisionModelWrapper):
            return True
        if isinstance(model, nn.Module):
            return any(wrapper._matches(model) for wrapper in _FEATURE_EXTRACTORS)
        return False

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        import torchvision.models as torchvision_models

        args = dict(model_args or {})
        if not load_weights and "weights" not in args:
            args["weights"] = None
        model = torchvision_models.get_model(model_name, **args)
        if num_input_channels != 3:
            cls._patch_input_channels(model, num_input_channels)
        return model

    @classmethod
    def _patch_input_channels(cls, model: nn.Module, num_input_channels: int) -> None:
        first_conv: nn.Module | None = None
        if hasattr(model, "conv1"):
            first_conv = model.conv1
        elif hasattr(model, "features") and len(model.features) > 0:
            candidate = model.features[0]
            if isinstance(candidate, nn.Conv2d):
                first_conv = candidate
            elif isinstance(candidate, nn.Sequential) and isinstance(
                candidate[0], nn.Conv2d
            ):
                first_conv = candidate[0]
        if first_conv is None:
            raise ValueError(
                f"Cannot adapt '{type(model).__name__}' to {num_input_channels} "
                f"input channels: no first conv layer found."
            )
        if not isinstance(first_conv, nn.Conv2d):  # pragma: no cover - narrowed above
            raise ValueError("First layer is not a Conv2d.")
        new_conv = nn.Conv2d(
            num_input_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(
                _adapt_input_channels(first_conv.weight.data, num_input_channels)
            )
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias.data)
        if hasattr(model, "conv1"):
            model.conv1 = new_conv
        elif hasattr(model, "features"):
            candidate = model.features[0]
            if isinstance(candidate, nn.Conv2d):
                model.features[0] = new_conv
            else:
                candidate[0] = new_conv

    @classmethod
    def as_encoder(cls, model: nn.Module) -> TorchvisionModelWrapper:
        for wrapper_cls in _FEATURE_EXTRACTORS:
            if wrapper_cls._matches(model):
                return wrapper_cls(model)
        raise ValueError(
            f"Unsupported torchvision model: '{type(model).__name__}'."
        )

    @classmethod
    def export_weights(
        cls, model: Any, out: Path, log_example: bool = True
    ) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapped = model if isinstance(model, TorchvisionModelWrapper) else cls.as_encoder(model)
        torch.save(wrapped.raw_module().state_dict(), out)
        if log_example:
            print(
                f"Exported torchvision model to '{out}'. Load with "
                f"autovision.list_models() -> 'torchvision/...' and "
                f"model.load_state_dict(torch.load('{out}'))."
            )


TORCHVISION_SOURCE = TorchvisionSource()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(TORCHVISION_SOURCE)
