# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""MultiStep LR schedule used by the official Dome/D-FINE recipes.

Linear warmup → constant plateau with ``gamma`` multiplicative drops at
fixed milestones. The official Dome-S-VisDrone recipe trains this way
(``MultiStepLR, milestones [80, 120], gamma 0.8`` over 160 epochs): the
LR never drops below 64 % of peak, which empirically suits dense
tiny-object benchmarks — deep cosine annealing plus the flat-cosine
restart tail measurably damaged end-of-run mAP in our pilots.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["MultiStepWarmupScheduler"]


class MultiStepWarmupScheduler(LRScheduler):
    """Linear warmup → constant LR with ``gamma`` drops at milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        milestones: tuple[int, ...] = (),
        gamma: float = 0.8,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.milestones = tuple(sorted(int(m) for m in milestones))
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / max(1, self.warmup_steps)
        else:
            factor = self.gamma ** sum(1 for m in self.milestones if step >= m)
        return [base_lr * factor for base_lr in self.base_lrs]
