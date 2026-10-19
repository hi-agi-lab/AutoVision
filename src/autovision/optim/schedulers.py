# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Step-level LR schedule: linear warmup followed by cosine decay.

Semantics (step interval; the "epochs" argument names follow the baseline):

- steps ``[0, warmup_epochs)``: lr ramps linearly from 0 to the base lr.
- steps ``[warmup_epochs, max_epochs]``: cosine decay from the base lr to
  ``end_value * base_lr``, reaching it exactly at ``max_epochs``.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["CosineWarmupScheduler"]


class CosineWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        end_value: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if not 0.0 <= end_value <= 1.0:
            raise ValueError(f"end_value must be in [0, 1], got {end_value}")
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.end_value = float(end_value)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch
        if step < self.warmup_epochs:
            warmup = max(1, self.warmup_epochs)
            factor = step / warmup
        else:
            decay_steps = max(1, self.max_epochs - self.warmup_epochs)
            progress = min((step - self.warmup_epochs) / decay_steps, 1.0)
            factor = self.end_value + (1.0 - self.end_value) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * factor for base_lr in self.base_lrs]
