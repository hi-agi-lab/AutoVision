# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset-size-based scaling: linear interpolation and bucket lookup.

These helpers are the single source of truth for "auto" hyperparameters that
depend on the dataset size (queue sizes, epochs derived from steps, ...).
Every bucket table lives next to the hyperparameter that consumes it; the
helpers here are the shared mechanics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    "IMAGENET_SIZE",
    "QUEUE_SIZE_BUCKETS",
    "ScalingInfo",
    "epochs_from_steps",
    "get_bucket_value",
    "interpolate",
]

T = TypeVar("T", int, float)

IMAGENET_SIZE = 1_000_000
"""Default dataset size for datasets whose size cannot be determined."""

QUEUE_SIZE_BUCKETS: tuple[tuple[float, int], ...] = (
    # (dataset_size <, queue_size) — memory bank ≈ 50% of the minimal dataset
    # size and 25% of the maximal dataset size of the given bucket.
    (50, 16),
    (100, 32),
    (250, 64),
    (500, 128),
    (1_000, 256),
    (2_000, 512),
    (4_000, 1024),
    (10_000, 2048),
    (20_000, 4096),
    (float("inf"), 8192),
)


@dataclass(frozen=True)
class ScalingInfo:
    """Inputs shared by all dataset-size-dependent "auto" resolutions."""

    dataset_size: int
    epochs: int


def interpolate(
    input: float,
    input_start: float,
    input_end: float,
    value_start: float,
    value_end: float,
    round_ndigits: int | None = None,
) -> float:
    """Linearly interpolate ``input`` from [input_start, input_end] to
    [value_start, value_end], clamped at both ends."""
    if input_end <= input_start:
        raise ValueError(
            f"input_end ({input_end}) must be greater than input_start ({input_start})"
        )
    frac = min(max((input - input_start) / (input_end - input_start), 0.0), 1.0)
    value = value_start + frac * (value_end - value_start)
    if round_ndigits is not None:
        value = round(value, round_ndigits)
    return value


def get_bucket_value(
    input: float, buckets: Iterable[tuple[float, T]]
) -> T:
    """Map ``input`` to the value of the first bucket whose threshold exceeds it."""
    for threshold, value in buckets:
        if input < threshold:
            return value
    raise ValueError(f"Input {input} is larger than all bucket thresholds.")


def epochs_from_steps(
    default_steps: int, batch_size: int, dataset_size: int
) -> int:
    """Convert a step-based recipe into an epoch count for the given dataset."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    return max(1, math.ceil(default_steps * batch_size / dataset_size))
