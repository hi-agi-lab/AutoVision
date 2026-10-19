# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Opt-in activation checkpointing for large task models.

Two mechanisms, in order of preference:

1. HuggingFace backbones expose ``gradient_checkpointing_enable()``; call it
   on every module that provides it (the HF implementation understands its
   own attention/MLP layout).
2. Custom ViT/GTR backbones expose their transformer layers as a
   ``blocks`` / ``layers`` ``ModuleList``; wrap each block's forward with
   ``torch.utils.checkpoint`` (non-reentrant).

Checkpointing trades compute for activation memory and changes nothing
numerically; it is opt-in because it roughly doubles the backbone forward
cost on small models.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["enable_gradient_checkpointing"]

_CHECKPOINTED_ATTR = "_autovision_gradient_checkpointed"
_BLOCK_CONTAINERS = ("blocks", "layers")


def _wrap_block(block: nn.Module) -> int:
    if getattr(block, _CHECKPOINTED_ATTR, False):
        return 0
    original = block.forward

    def checkpointed_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            original, *args, use_reentrant=False, **kwargs
        )

    block.forward = checkpointed_forward  # type: ignore[method-assign]
    setattr(block, _CHECKPOINTED_ATTR, True)
    return 1


def enable_gradient_checkpointing(model: nn.Module) -> int:
    """Enable activation checkpointing; return the number of patched modules.

    Only blocks in a ``ModuleList`` named ``blocks`` or ``layers`` are
    wrapped (the two conventions used by the in-tree ViT/GTR backbones).
    HF backbones use their native implementation. Calling the function twice
    is a no-op for already-patched blocks.
    """
    native = [
        module
        for module in model.modules()
        if callable(getattr(module, "gradient_checkpointing_enable", None))
    ]
    if native:
        for module in native:
            module.gradient_checkpointing_enable()
        return len(native)

    patched = 0
    seen: set[int] = set()
    for module in model.modules():
        for container_name in _BLOCK_CONTAINERS:
            container = getattr(module, container_name, None)
            if not isinstance(container, (nn.ModuleList, list, tuple)):
                continue
            for block in container:
                if not isinstance(block, nn.Module) or id(block) in seen:
                    continue
                seen.add(id(block))
                patched += _wrap_block(block)
    return patched
