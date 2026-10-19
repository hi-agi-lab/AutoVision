# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset class-id compaction: arbitrary ids -> dense ids from 0."""

from __future__ import annotations

__all__ = ["contiguous_class_id_map"]


def contiguous_class_id_map(class_ids: list[int]) -> dict[int, int]:
    """Map each given (non-ignored) class id to its rank in sorted order."""
    return {class_id: rank for rank, class_id in enumerate(sorted(class_ids))}
