# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT optimizer: AdamW with layer-wise learning-rate decay.

Follows the ViT finetuning convention used by the EOMT recipes: block
``layers.N`` decays by ``layer_decay^(max_layer - N)`` while everything
outside the block stack (embeddings, queries, scale blocks, mask head,
class predictor) trains at the full learning rate. Norm layers and biases
are excluded from weight decay.
"""

from __future__ import annotations

import re

import torch
from torch import nn

__all__ = ["build_eomt_optimizer"]

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _num_layer(name: str, num_layers: int) -> int:
    """Block index of a parameter, or ``num_layers`` (top lr) if outside."""
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else num_layers


def _is_no_weight_decay(name: str) -> bool:
    return (
        name.endswith(".bias")
        or "norm" in name.lower()
        or ".lambda" in name  # layer-scale parameters are scalars
    )


def build_eomt_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.01,
    layer_decay: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    """AdamW parameter groups: per-block lr decay x (decay / no-decay)."""
    layer_indices = [
        int(match.group(1))
        for name, _ in model.named_parameters()
        if (match := _LAYER_RE.search(name))
    ]
    num_layers = (max(layer_indices) + 1) if layer_indices else 0
    groups: dict[tuple[int, bool], list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer = _num_layer(name, num_layers)
        no_decay = _is_no_weight_decay(name)
        groups.setdefault((layer, no_decay), []).append(parameter)
    optimizer_groups = []
    for (layer, no_decay), parameters in sorted(groups.items()):
        factor = layer_decay ** max(0, num_layers - layer)
        optimizer_groups.append(
            {
                "name": f"layer{layer}{'_nodecay' if no_decay else ''}",
                "params": parameters,
                "lr": lr * factor,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return torch.optim.AdamW(optimizer_groups, lr=lr, betas=betas)
