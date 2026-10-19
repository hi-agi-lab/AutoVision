# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""8-bit AdamW via bitsandbytes (extras: ``autovision[bitsandbytes]``)."""

from __future__ import annotations

import importlib.util

import torch
from pydantic import Field

from autovision.optim.optimizer_args import AdamWArgs

__all__ = ["AdamW8BitArgs"]


def _require_bitsandbytes():
    """Resolve the bitsandbytes 8-bit AdamW class or fail with guidance."""
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "8-bit AdamW requires the optional 'bitsandbytes' package: "
            "pip install 'autovision[bitsandbytes]'"
        )
    from bitsandbytes.optim import AdamW8bit

    return AdamW8bit


class AdamW8BitArgs(AdamWArgs):
    """AdamW with bitsandbytes 8-bit storage; identical defaults to
    :class:`AdamWArgs`. The optional dependency is only touched when the
    optimizer is actually built."""

    betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)

    def kind(self):
        return "adamw8bit"

    def get_optimizer(self, params, lr_scale: float) -> torch.optim.Optimizer:
        adamw8bit_cls = _require_bitsandbytes()
        settings = self.model_dump(exclude={"lr"})
        return adamw8bit_cls(params=params, lr=self.lr * lr_scale, **settings)
