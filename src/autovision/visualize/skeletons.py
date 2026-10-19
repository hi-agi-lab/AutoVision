# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Keypoint skeleton definitions shared by pose annotators.

``COCO_SKELETON`` keeps the historical 1-based convention (keypoint 1 is the
first point) used by the existing COCO-17 renderer. ``SKELETONS`` exposes
zero-based edge lists for custom multi-skeleton models; annotators accept
either registry names or an explicit edge sequence.
"""

from __future__ import annotations

__all__ = ["COCO_SKELETON", "GHUM_SKELETON", "SKELETONS"]

#: COCO-17 skeleton, 1-based pairs (the official keypoint order).
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
)

#: GHUM-33 skeleton, 1-based pairs (full-body pose models).
GHUM_SKELETON: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (4, 8),
    (5, 6),
    (6, 7),
    (7, 9),
    (10, 11),
    (12, 13),
    (12, 14),
    (12, 24),
    (13, 15),
    (13, 25),
    (14, 16),
    (15, 17),
    (16, 18),
    (15, 19),
    (16, 22),
    (17, 19),
    (17, 21),
    (17, 23),
    (18, 20),
    (19, 21),
    (24, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (28, 32),
    (29, 31),
    (29, 33),
    (30, 32),
    (31, 33),
)


def _zero_based(
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple((start - 1, end - 1) for start, end in edges)


#: Zero-based edge registry consumed by :class:`KeypointAnnotator`.
SKELETONS: dict[str, tuple[tuple[int, int], ...]] = {
    "coco": _zero_based(COCO_SKELETON),
    "ghum": _zero_based(GHUM_SKELETON),
}
