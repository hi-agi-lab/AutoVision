# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Optimizer discriminator values accepted by ``optimizer_args={"type": …}``."""

from __future__ import annotations

from typing import Literal

__all__ = ["OPTIMIZER_KINDS", "OptimizerKind"]

#: The ``type`` value selects the optimizer settings class in the
#: ``optimizer_args`` config block.
OptimizerKind = Literal["adamw", "adamw8bit", "sgd", "lars"]

#: Every valid value, for validators and help text.
OPTIMIZER_KINDS: tuple[OptimizerKind, ...] = ("adamw", "adamw8bit", "sgd", "lars")
