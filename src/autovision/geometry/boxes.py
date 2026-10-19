# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""NumPy bounding-box utilities for inference postprocessing and rendering.

Training keeps its differentiable torch implementations in
:mod:`autovision.geometry.box_ops`; this module covers the inference side,
where detections are already NumPy arrays and the operations are not part of
the gradient graph.

All boxes use ``(x_min, y_min, x_max, y_max)`` pixels unless a function name
says otherwise. NMS/NMM follow the per-class convention: a ``(N, 6)``
prediction array carries the class id in the last column and boxes of
different classes never suppress or merge each other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autovision.errors import AutoVisionError

__all__ = [
    "box_area",
    "box_iou",
    "box_non_max_merge",
    "box_non_max_suppression",
    "clip_boxes",
    "denormalize_boxes",
    "move_boxes",
    "pad_boxes",
    "scale_boxes",
    "xcycwh_to_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_xcycarh",
    "xyxy_to_xywh",
]


def _as_boxes(value: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    """Return ``value`` as a float ``(N, 4)`` array (empty stays ``(0, 4)``)."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise AutoVisionError(f"{name} must have shape (N, 4), got {array.shape}.")
    return array


def _as_predictions(
    value: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return ``value`` as a float ``(N, 5)``/``(N, 6)`` prediction array."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 5)
    if array.ndim != 2 or array.shape[1] not in (5, 6):
        raise AutoVisionError(
            "predictions must have shape (N, 5) or (N, 6), "
            f"got {array.shape}."
        )
    return array


def _validate_resolution(resolution_wh: tuple[int, int]) -> tuple[float, float]:
    width, height = resolution_wh
    if float(width) <= 0 or float(height) <= 0:
        raise AutoVisionError(
            f"resolution_wh must be positive, got {resolution_wh!r}."
        )
    return float(width), float(height)


def _validate_iou_threshold(iou_threshold: float) -> float:
    value = float(iou_threshold)
    if not 0.0 <= value <= 1.0:
        raise AutoVisionError(
            f"iou_threshold must be in [0, 1], got {iou_threshold!r}."
        )
    return value


def box_area(boxes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the non-negative area of every box."""
    array = _as_boxes(boxes, name="boxes")
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.clip(array[:, 2] - array[:, 0], 0.0, None) * np.clip(
        array[:, 3] - array[:, 1], 0.0, None
    )


def box_iou(
    boxes1: Sequence[Sequence[float]] | np.ndarray,
    boxes2: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the pairwise IoU matrix of shape ``(len(boxes1), len(boxes2))``."""
    first = _as_boxes(boxes1, name="boxes1")
    second = _as_boxes(boxes2, name="boxes2")
    if first.size == 0 or second.size == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)

    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    union = (
        box_area(first)[:, None]
        + box_area(second)[None, :]
        - intersection_area
    )
    return np.divide(
        intersection_area,
        union,
        out=np.zeros_like(intersection_area),
        where=union > 0,
    )


def box_non_max_suppression(
    predictions: Sequence[Sequence[float]] | np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean keep-mask for greedy per-class NMS.

    ``predictions`` is ``(N, 5)`` ``[x1, y1, x2, y2, score]`` or ``(N, 6)``
    with a trailing class id. The returned mask is in the input order.
    """
    threshold = _validate_iou_threshold(iou_threshold)
    array = _as_predictions(predictions)
    if array.size == 0:
        return np.zeros(0, dtype=bool)

    boxes = array[:, :4]
    scores = array[:, 4]
    classes = (
        array[:, 5].astype(np.int64)
        if array.shape[1] == 6
        else np.zeros(array.shape[0], dtype=np.int64)
    )
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(array.shape[0], dtype=bool)
    keep = np.zeros(array.shape[0], dtype=bool)

    for position, index in enumerate(order):
        if suppressed[index]:
            continue
        keep[index] = True
        remaining = order[position + 1 :]
        remaining = remaining[~suppressed[remaining] & (classes[remaining] == classes[index])]
        if remaining.size == 0:
            continue
        overlaps = box_iou(boxes[index : index + 1], boxes[remaining])[0]
        suppressed[remaining[overlaps >= threshold]] = True
    return keep


def box_non_max_merge(
    predictions: Sequence[Sequence[float]] | np.ndarray,
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    """Group overlapping same-class boxes for non-maximum merging.

    Returns index groups whose highest-scoring member comes first. Merging
    the boxes in a group (weighted average, union box, ...) is left to the
    caller; this helper only answers "which detections describe one object".
    """
    threshold = _validate_iou_threshold(iou_threshold)
    array = _as_predictions(predictions)
    if array.size == 0:
        return []

    boxes = array[:, :4]
    scores = array[:, 4]
    classes = (
        array[:, 5].astype(np.int64)
        if array.shape[1] == 6
        else np.zeros(array.shape[0], dtype=np.int64)
    )
    groups: list[list[int]] = []
    for class_id in np.unique(classes):
        indices = np.nonzero(classes == class_id)[0]
        remaining = list(indices[np.argsort(-scores[indices], kind="stable")])
        while remaining:
            head = int(remaining[0])
            tail = remaining[1:]
            if not tail:
                groups.append([head])
                break
            overlaps = box_iou(boxes[head : head + 1], boxes[tail])[0]
            merged = [head, *[int(tail[i]) for i in np.nonzero(overlaps >= threshold)[0]]]
            groups.append(merged)
            merged_set = set(merged[1:])
            remaining = [index for index in tail if index not in merged_set]
    return groups


def clip_boxes(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
    """Clip every box to the ``(0, 0, width, height)`` frame."""
    array = _as_boxes(xyxy, name="xyxy")
    width, height = _validate_resolution(resolution_wh)
    clipped = array.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0.0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0.0, height)
    return clipped


def pad_boxes(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
    px: int,
    py: int | None = None,
) -> np.ndarray:
    """Expand every box by ``px`` horizontally and ``py`` vertically."""
    array = _as_boxes(xyxy, name="xyxy")
    horizontal = int(px)
    vertical = horizontal if py is None else int(py)
    padded = array.copy()
    padded[:, [0, 1]] -= (horizontal, vertical)
    padded[:, [2, 3]] += (horizontal, vertical)
    return padded


def denormalize_boxes(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
    resolution_wh: tuple[int, int],
    normalization_factor: float = 1.0,
) -> np.ndarray:
    """Map normalized boxes in ``[0, normalization_factor]`` to pixels."""
    array = _as_boxes(xyxy, name="xyxy")
    width, height = _validate_resolution(resolution_wh)
    factor = float(normalization_factor)
    if factor <= 0:
        raise AutoVisionError(
            f"normalization_factor must be positive, got {normalization_factor!r}."
        )
    pixels = array.copy()
    pixels[:, [0, 2]] = pixels[:, [0, 2]] * width / factor
    pixels[:, [1, 3]] = pixels[:, [1, 3]] * height / factor
    return pixels


def scale_boxes(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
    factor: float,
) -> np.ndarray:
    """Scale every box around its center by ``factor``."""
    array = _as_boxes(xyxy, name="xyxy")
    value = float(factor)
    if value <= 0:
        raise AutoVisionError(f"factor must be positive, got {factor!r}.")
    centers = (array[:, :2] + array[:, 2:]) / 2.0
    sizes = (array[:, 2:] - array[:, :2]) * value
    return np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)


def move_boxes(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
    offset_xy: tuple[float, float],
) -> np.ndarray:
    """Translate every box by ``(dx, dy)``."""
    array = _as_boxes(xyxy, name="xyxy")
    dx, dy = (float(offset_xy[0]), float(offset_xy[1]))
    return array + np.asarray((dx, dy, dx, dy), dtype=np.float64)


def xywh_to_xyxy(
    xywh: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Convert ``(x, y, width, height)`` boxes to ``(x1, y1, x2, y2)``."""
    array = _as_boxes(xywh, name="xywh")
    converted = array.copy()
    converted[:, 2] = array[:, 0] + array[:, 2]
    converted[:, 3] = array[:, 1] + array[:, 3]
    return converted


def xyxy_to_xywh(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(x, y, width, height)``."""
    array = _as_boxes(xyxy, name="xyxy")
    converted = array.copy()
    converted[:, 2] = array[:, 2] - array[:, 0]
    converted[:, 3] = array[:, 3] - array[:, 1]
    return converted


def xcycwh_to_xyxy(
    xcycwh: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Convert center boxes ``(cx, cy, width, height)`` to xyxy."""
    array = _as_boxes(xcycwh, name="xcycwh")
    converted = array.copy()
    converted[:, 0] = array[:, 0] - array[:, 2] / 2.0
    converted[:, 1] = array[:, 1] - array[:, 3] / 2.0
    converted[:, 2] = array[:, 0] + array[:, 2] / 2.0
    converted[:, 3] = array[:, 1] + array[:, 3] / 2.0
    return converted


def xyxy_to_xcycarh(
    xyxy: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Convert xyxy boxes to ``(cx, cy, aspect_ratio, height)``."""
    array = _as_boxes(xyxy, name="xyxy")
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    width = array[:, 2] - array[:, 0]
    height = array[:, 3] - array[:, 1]
    aspect_ratio = np.divide(
        width,
        height,
        out=np.zeros_like(width, dtype=np.float64),
        where=height != 0,
    )
    return np.column_stack(
        (array[:, 0] + width / 2.0, array[:, 1] + height / 2.0, aspect_ratio, height)
    )
