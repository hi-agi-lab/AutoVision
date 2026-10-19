# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Step schedule resolution for detection training (baseline constants)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["StepSchedule", "resolve_detection_step_schedule"]

REFERENCE_WARMUP_EPOCHS = 4
SHORT_RUN_TOTAL_EPOCHS = 12
REFERENCE_TOTAL_EPOCHS = 72
REFERENCE_NO_AUG_EPOCHS = 12


@dataclass(frozen=True)
class StepSchedule:
    """Augmentation windows in optimizer steps: ``[start, flat)`` for
    mosaic/mixup-style ops and ``[start, stop)`` for the rest."""

    start: int
    flat: int
    stop: int

    def window(self, kind: str) -> tuple[int | None, int | None]:
        if kind in ("mosaic", "mixup"):
            return self.start, self.flat
        return self.start, self.stop


def resolve_detection_step_schedule(
    total_steps: int, steps_per_epoch: int
) -> StepSchedule:
    """Epoch-domain recipe (baseline constants) mapped to steps.

    ``no_aug = min(12, round(total_epochs * 12 / 72))``; short runs (<= 12
    epochs) start the decay at ``total/3``, long runs at epoch 4; the flat
    window extends half of the remaining epochs; empty windows disable the
    augmentation instead of clamping.
    """
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    total_epochs = max(1, round(total_steps / steps_per_epoch))
    steps_per_epoch = total_steps / total_epochs

    no_aug_epochs = min(
        REFERENCE_NO_AUG_EPOCHS,
        round(total_epochs * REFERENCE_NO_AUG_EPOCHS / REFERENCE_TOTAL_EPOCHS),
    )
    stop_epoch = max(0, total_epochs - no_aug_epochs)
    if total_epochs <= SHORT_RUN_TOTAL_EPOCHS:
        start_epoch = total_epochs // 3
        flat_epoch = min(stop_epoch, start_epoch + total_epochs // 2)
    else:
        start_epoch = REFERENCE_WARMUP_EPOCHS
        flat_epoch = start_epoch + total_epochs // 2

    return StepSchedule(
        start=round(start_epoch * steps_per_epoch),
        flat=round(flat_epoch * steps_per_epoch),
        stop=round(stop_epoch * steps_per_epoch),
    )
