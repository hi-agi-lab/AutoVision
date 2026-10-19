# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared test model helpers."""

from types import SimpleNamespace

import torch
from torch import nn


class TinyWrapper(nn.Module):
    """Minimal EncoderProtocol implementation used across tests."""

    def __init__(self, dim: int = 8, model_type: str = "convolutional"):
        super().__init__()
        self.dim = dim
        self._model_type = model_type
        self.conv = nn.Conv2d(3, dim, 3, padding=1)

    def feature_maps(self, x):
        return {
            "features": self.conv(x),
            "cls_token": self.conv(x).mean(dim=(2, 3)),
        }

    def pool_features(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x["features"], 1)
        return {"pooled_features": pooled}

    def encoder_dim(self) -> int:
        return self.dim

    def raw_module(self):
        return self.conv

    def arch_traits(self):
        return {"model_type": self._model_type, "norm_type": "batchnorm"}


class TinyMultiscaleWrapper(nn.Module):
    """Minimal ViT-like wrapper with intermediate-layer taps for tests.

    Each "block" is a residual 1×1-conv MLP over the patch grid;
    ``stage_feature_maps`` taps the requested block outputs
    (0-based, like the real ViT wrappers).
    """

    def __init__(self, dim: int = 8, grid: int = 4, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.proj = nn.Conv2d(3, dim, kernel_size=32 // grid, stride=32 // grid)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU()) for _ in range(num_layers)]
        )
        self.config = SimpleNamespace(num_hidden_layers=num_layers)

    def _block_outputs(self, x):
        hidden = self.proj(x)
        outputs = []
        for block in self.blocks:
            hidden = hidden + block(hidden)
            outputs.append(hidden)
        return outputs

    def stage_feature_maps(self, x, layer_indices):
        outputs = self._block_outputs(x)
        return [{"features": outputs[i]} for i in layer_indices]

    def feature_maps(self, x):
        features = self._block_outputs(x)[-1]
        return {"features": features, "cls_token": features.mean(dim=(2, 3))}

    def pool_features(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x["features"], 1)
        return {"pooled_features": pooled}

    def encoder_dim(self) -> int:
        return self.dim

    def patch_size(self) -> int:
        return 32 // self.grid

    def raw_module(self):
        return self.proj

    def arch_traits(self):
        return {"model_type": "transformer", "norm_type": "layernorm"}
