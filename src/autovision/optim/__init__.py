# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Optimizer selection, parameter grouping, and LR scheduling."""

from autovision.optim.optimizer_args import (
    AdamWArgs,
    LARSArgs,
    OptimizerArgs,
    SGDArgs,
    get_optimizer_args_cls,
)
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_optimizer,
    get_weight_decay_parameters,
)
from autovision.optim.optimizer_type import OPTIMIZER_KINDS, OptimizerKind
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.optim.trainable_modules import OptimizationTargets

__all__ = [
    "AdamWArgs",
    "CosineWarmupScheduler",
    "LARSArgs",
    "OptimizerArgs",
    "OptimizerKind",
    "OPTIMIZER_KINDS",
    "SGDArgs",
    "OptimizationTargets",
    "build_adamw",
    "get_optimizer",
    "get_optimizer_args_cls",
    "get_weight_decay_parameters",
]
