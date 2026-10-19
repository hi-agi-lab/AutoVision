# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Flat-cosine LR schedule used by the detection task family.

Linear warmup -> constant plateau -> cosine decay -> constant tail. The
plateau ends ``no_aug_steps`` before the end (aligned with the augmentation
disable window), mirroring the baseline detection recipes.

Source lineage: the DEIM flat-cosine schedule (``Intellindust-AI-Lab/DEIM``,
Apache-2.0, Copyright (c) 2024 The DEIM Authors; also vendored in the local
DEIMv2 mirror ``engine/optim/lr_scheduler.py``), re-implemented by AutoVision
on a step-based interface; provenance ledger entry (P1-5).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["FlatCosineLRScheduler"]


#: Baseline tail LR as a fraction of peak (the cosine floors here and the
#: tail HOLDS it — no restart; an earlier revision jumped back to 1.0 after
#: dipping to 0.01, a sawtooth that damaged end-of-run mAP).
DEFAULT_MIN_FACTOR = 0.5


class FlatCosineLRScheduler(LRScheduler):
    """Quadratic warmup -> flat plateau -> cosine decay -> tail at min LR.

    Baseline semantics: the cosine floors at ``min_factor × peak`` and the
    tail holds that value; warmup is quadratic (``progress²``).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 1000,
        flat_steps: int = 0,
        no_aug_steps: int = 0,
        min_factor: float = DEFAULT_MIN_FACTOR,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.flat_steps = max(0, int(flat_steps))
        self.no_aug_steps = max(0, int(no_aug_steps))
        self.min_factor = min_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch
        cosine_start = self.warmup_steps + self.flat_steps
        cosine_end = self.total_steps - self.no_aug_steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            factor = (step / float(self.warmup_steps)) ** 2
        elif step < cosine_start:
            factor = 1.0
        elif step >= cosine_end:
            factor = self.min_factor
        else:
            decay = 0.5 * (
                1.0
                + math.cos(
                    math.pi * (step - cosine_start) / (cosine_end - cosine_start)
                )
            )
            factor = self.min_factor + (1.0 - self.min_factor) * decay
        return [base_lr * factor for base_lr in self.base_lrs]


def resolve_flat_cosine_windows(
    total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
) -> tuple[int, int, int]:
    """Default (warmup, flat, no_aug) step windows for the flat-cosine recipe.

    Follows the baseline detection constants: warmup = 4 epochs-equivalent,
    flat until half the run, and the last 12/72 of the run (min 0) reserved
    for the no-augmentation tail.
    """
    from autovision.augment.detection.schedule import (
        resolve_detection_step_schedule,
    )

    steps_per_epoch = max(1, train_num_batches // max(1, gradient_accumulation_steps))
    schedule = resolve_detection_step_schedule(total_steps, steps_per_epoch)
    warmup = schedule.start
    flat = max(0, schedule.flat - schedule.start)
    no_aug = max(0, total_steps - schedule.stop)
    return warmup, flat, no_aug
