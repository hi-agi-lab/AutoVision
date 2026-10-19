# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Torchvision model wrappers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autovision.encoders.model_wrapper import (
    ArchTraits,
    FeatureMapsOut,
    PoolOut,
)


class TorchvisionModelWrapper(nn.Module):
    """Base wrapper for torchvision models.

    Stores the raw model as ``self._model`` (wrapped in a list-like helper is
    unnecessary here because torchvision models are real ``nn.Module``\ s).
    """

    _torchvision_model_name_pattern: str

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def raw_module(self) -> nn.Module:
        return self._model

    def pool_features(self, x: FeatureMapsOut) -> PoolOut:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}

    @classmethod
    def _matches(cls, model: nn.Module) -> bool:
        raise NotImplementedError

    def arch_traits(self) -> ArchTraits:  # pragma: no cover - overridden
        return {"model_type": "convolutional", "norm_type": "batchnorm"}


class ResNetModelWrapper(TorchvisionModelWrapper):
    _torchvision_model_name_pattern = r"resnet\d+|resnext\d+|wide_resnet\d+"

    @classmethod
    def _matches(cls, model: nn.Module) -> bool:
        return hasattr(model, "layer4") and hasattr(model, "fc")

    def feature_maps(self, x: Tensor) -> FeatureMapsOut:
        model = self._model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return {"features": x}

    def encoder_dim(self) -> int:
        return int(self._model.fc.in_features)

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}


class ConvNeXtModelWrapper(TorchvisionModelWrapper):
    _torchvision_model_name_pattern = r"convnext_(tiny|small|base|large)"

    @classmethod
    def _matches(cls, model: nn.Module) -> bool:
        return hasattr(model, "features") and hasattr(model, "classifier") and not hasattr(
            model, "layer4"
        )

    def feature_maps(self, x: Tensor) -> FeatureMapsOut:
        return {"features": self._model.features(x)}

    def encoder_dim(self) -> int:
        return int(self._model.classifier[-1].in_features)

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "convolutional", "norm_type": "layernorm"}


_FEATURE_EXTRACTORS: list[type[TorchvisionModelWrapper]] = [
    ResNetModelWrapper,
    ConvNeXtModelWrapper,
]
