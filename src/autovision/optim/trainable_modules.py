# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Which model parts the optimizer drives, and weight-decay exemptions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from torch import nn

__all__ = ["OptimizationTargets"]


@dataclass
class OptimizationTargets:
    """The parameters the optimizer drives.

    ``decay_free`` modules never receive weight decay, regardless of the
    name/shape rules applied to everything else.
    """

    targets: Iterable[nn.Module]
    decay_free: Iterable[nn.Module] = field(default_factory=tuple)
