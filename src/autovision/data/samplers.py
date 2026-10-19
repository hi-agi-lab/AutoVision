# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Long-tail sampling strategies (roadmap 90d, phase 3).

Two standard, dependency-free options over the detection loader rows:

- ``class_balanced``: per-image weight from the inverse class frequency
  (``1/count**power``), normalized to ``[1, max_repeat]``;
- ``repeat_factor``: LVIS-style, ``sqrt(threshold/frequency)`` per class,
  capped at ``max_repeat`` (image weight = max over its classes).

Both return a deterministic ``WeightedRandomSampler``. Like hard-example
sampling, v1 is single-device: the training entry rejects DDP rather than
letting DistributedSampler silently override the weights.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from autovision.errors import AutoVisionError

__all__ = [
    "build_long_tail_sampler",
    "class_counts",
    "image_weights",
]

_STRATEGIES = ("class_balanced", "repeat_factor")


def _labels_of(row: Any) -> list[int]:
    if isinstance(row, Mapping):
        raw = row.get("class_labels", "[]")
        if isinstance(raw, str):
            return [int(value) for value in json.loads(raw)]
        return [int(value) for value in raw]
    return [int(value) for value in row]


def _rows_labels(rows: Sequence[Any]) -> tuple[list[list[int]], int]:
    labels = [_labels_of(row) for row in rows]
    return labels, len(labels)


def class_counts(rows: Sequence[Any]) -> dict[int, int]:
    """Number of images containing each class id."""
    counts: dict[int, int] = {}
    for image_labels in _rows_labels(rows)[0]:
        for class_id in set(image_labels):
            counts[class_id] = counts.get(class_id, 0) + 1
    return dict(sorted(counts.items()))


def image_weights(
    rows: Sequence[Any],
    *,
    strategy: str = "class_balanced",
    power: float = 0.5,
    threshold: float = 0.01,
    max_repeat: float = 5.0,
) -> torch.Tensor:
    """Per-image sampling weights (length == len(rows))."""
    if strategy not in _STRATEGIES:
        raise AutoVisionError(
            f"Unknown long-tail strategy '{strategy}'; expected one of "
            f"{list(_STRATEGIES)}."
        )
    if power < 0:
        raise AutoVisionError(f"power must be >= 0, got {power}.")
    if max_repeat < 1:
        raise AutoVisionError(f"max_repeat must be >= 1, got {max_repeat}.")
    if not 0 < threshold <= 1:
        if strategy == "repeat_factor":
            raise AutoVisionError(f"threshold must be in (0, 1], got {threshold}.")
    labels, total = _rows_labels(rows)
    if total == 0:
        raise AutoVisionError("Long-tail sampling needs at least one image.")
    counts = class_counts(rows)

    if strategy == "repeat_factor":
        factors = {
            class_id: min(
                max_repeat,
                max(1.0, math.sqrt(threshold / (count / total))),
            )
            for class_id, count in counts.items()
        }
        weights = [
            max((factors[class_id] for class_id in image_labels), default=1.0)
            for image_labels in labels
        ]
        return torch.tensor(weights, dtype=torch.double)

    raw = [
        (
            sum(1.0 / (counts[class_id] ** power) for class_id in image_labels)
            / len(image_labels)
            if image_labels
            else 0.0
        )
        for image_labels in labels
    ]
    labeled_raw = [value for value, image_labels in zip(raw, labels) if image_labels]
    if not labeled_raw:
        return torch.ones(total, dtype=torch.double)
    low, high = min(labeled_raw), max(labeled_raw)
    weights = []
    for value, image_labels in zip(raw, labels):
        if not image_labels or high <= low:
            # Unlabeled/background images keep the baseline weight; they
            # still participate in training.
            weights.append(1.0)
        else:
            weights.append(
                1.0 + (value - low) / (high - low) * (max_repeat - 1.0)
            )
    return torch.tensor(weights, dtype=torch.double)


def build_long_tail_sampler(
    rows: Sequence[Any],
    *,
    strategy: str = "class_balanced",
    power: float = 0.5,
    threshold: float = 0.01,
    max_repeat: float = 5.0,
    seed: int = 0,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Deterministic replacement sampler over the training loader rows."""
    weights = image_weights(
        rows,
        strategy=strategy,
        power=power,
        threshold=threshold,
        max_repeat=max_repeat,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights) if num_samples is None else int(num_samples),
        replacement=True,
        generator=generator,
    )
