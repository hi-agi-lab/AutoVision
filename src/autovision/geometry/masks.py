# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Mask and polygon utilities for inference-side postprocessing.

Binary masks use half-open boxes (``x_max``/``y_max`` are exclusive) unless a
function says otherwise, matching NumPy/Pillow slicing. Polygon simplification
is a NumPy Ramer-Douglas-Peucker implementation; mask-to-polygon conversion
needs OpenCV and therefore imports it lazily from the ``video`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autovision.errors import AutoVisionError

__all__ = [
    "approximate_polygon",
    "filter_polygons_by_area",
    "mask_to_polygons",
    "mask_to_xyxy",
    "polygon_area",
    "polygon_to_mask",
    "polygon_to_xyxy",
    "xyxy_to_mask",
]


def _as_masks(masks: Sequence | np.ndarray) -> np.ndarray:
    """Return a boolean ``(N, H, W)`` stack; a single 2-D mask becomes N=1."""
    array = np.asarray(masks)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise AutoVisionError(
            f"masks must have shape (N, H, W) or (H, W), got {array.shape}."
        )
    return array.astype(bool, copy=False)


def _as_polygon(polygon: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return ``polygon`` as a float ``(M, 2)`` array."""
    array = np.asarray(polygon, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise AutoVisionError(
            f"polygon must have shape (M, 2), got {array.shape}."
        )
    return array


def mask_to_xyxy(masks: Sequence | np.ndarray) -> np.ndarray:
    """Return half-open ``(x1, y1, x2, y2)`` boxes for every non-empty mask.

    Empty masks produce an all-zero box, mirroring the convention used by
    detector postprocessing when a row has no pixels.
    """
    stack = _as_masks(masks)
    count, height, width = (int(value) for value in stack.shape)
    boxes: np.ndarray = np.zeros((count, 4), dtype=np.int64)
    if stack.size == 0:
        return boxes
    rows = np.any(stack, axis=2)
    columns = np.any(stack, axis=1)
    non_empty = np.any(rows, axis=1)
    if not bool(np.any(non_empty)):
        return boxes
    x_min = np.argmax(columns, axis=1)
    y_min = np.argmax(rows, axis=1)
    x_max = width - np.argmax(columns[:, ::-1], axis=1)
    y_max = height - np.argmax(rows[:, ::-1], axis=1)
    boxes[:, 0] = x_min
    boxes[:, 1] = y_min
    boxes[:, 2] = x_max
    boxes[:, 3] = y_max
    boxes[~non_empty] = 0
    return boxes


def xyxy_to_mask(
    boxes: Sequence[Sequence[float]] | np.ndarray,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
    """Rasterize half-open xyxy boxes into an ``(N, H, W)`` boolean stack."""
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise AutoVisionError(f"boxes must have shape (N, 4), got {array.shape}.")
    width, height = (int(resolution_wh[0]), int(resolution_wh[1]))
    if width <= 0 or height <= 0:
        raise AutoVisionError(
            f"resolution_wh must be positive, got {resolution_wh!r}."
        )
    masks = np.zeros((array.shape[0], height, width), dtype=bool)
    for index, (x1, y1, x2, y2) in enumerate(array):
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        if right > left and bottom > top:
            masks[index, top:bottom, left:right] = True
    return masks


def polygon_to_mask(
    polygon: Sequence[Sequence[float]] | np.ndarray,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
    """Rasterize a polygon into a boolean ``(H, W)`` mask."""
    points = _as_polygon(polygon)
    width, height = (int(resolution_wh[0]), int(resolution_wh[1]))
    if width <= 0 or height <= 0:
        raise AutoVisionError(
            f"resolution_wh must be positive, got {resolution_wh!r}."
        )
    if points.shape[0] < 3:
        return np.zeros((height, width), dtype=bool)
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [tuple(float(value) for value in point) for point in points], fill=1
    )
    return np.asarray(canvas) > 0


def mask_to_polygons(
    mask: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> list[np.ndarray]:
    """Extract external polygon contours from a 2-D boolean mask.

    ``tolerance`` is passed to OpenCV's ``approxPolyDP`` when positive.
    OpenCV is the optional ``video`` extra; without it this raises
    :class:`AutoVisionError` with the install command.
    """
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise AutoVisionError(
            "mask_to_polygons needs OpenCV: pip install -e '.[video]'"
        ) from error

    array = np.asarray(mask)
    if array.ndim != 2:
        raise AutoVisionError(f"mask must be 2-D, got {array.shape}.")
    contours, _ = cv2.findContours(
        array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[np.ndarray] = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if tolerance > 0 and points.shape[0] >= 3:
            points = cv2.approxPolyDP(
                contour, float(tolerance), True
            ).reshape(-1, 2)
        if points.shape[0] >= 3:
            polygons.append(points.astype(np.int32))
    return polygons


def polygon_area(polygon: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Return the absolute shoelace area of one polygon."""
    points = _as_polygon(polygon)
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_to_xyxy(
    polygon: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the axis-aligned envelope of one polygon as xyxy."""
    points = _as_polygon(polygon)
    if points.size == 0:
        return np.zeros(4, dtype=np.float64)
    return np.concatenate((points.min(axis=0), points.max(axis=0)))


def filter_polygons_by_area(
    polygons: Sequence[Sequence[Sequence[float]] | np.ndarray],
    *,
    min_area: float | None = None,
    max_area: float | None = None,
) -> list[np.ndarray]:
    """Keep polygons whose absolute area is within the requested bounds."""
    lower = 0.0 if min_area is None else float(min_area)
    upper = np.inf if max_area is None else float(max_area)
    if lower < 0 or upper < lower:
        raise AutoVisionError(
            f"Invalid area bounds: min_area={min_area!r}, max_area={max_area!r}."
        )
    kept: list[np.ndarray] = []
    for polygon in polygons:
        points = _as_polygon(polygon)
        if lower <= polygon_area(points) <= upper:
            kept.append(points)
    return kept


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify an open polyline with the Ramer-Douglas-Peucker algorithm."""
    if points.shape[0] < 3:
        return points
    start, end = points[0], points[-1]
    delta = end - start
    length = float(np.hypot(delta[0], delta[1]))
    if length == 0:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        distances = np.abs(
            delta[0] * (start[1] - points[:, 1])
            - (start[0] - points[:, 0]) * delta[1]
        ) / length
    index = int(np.argmax(distances))
    if distances[index] <= epsilon:
        return np.stack((start, end), axis=0)
    left = _rdp(points[: index + 1], epsilon)
    right = _rdp(points[index:], epsilon)
    return np.concatenate((left[:-1], right), axis=0)


def approximate_polygon(
    polygon: Sequence[Sequence[float]] | np.ndarray,
    percentage: float,
    epsilon_step: float = 0.05,
) -> np.ndarray:
    """Simplify a polygon by removing roughly ``percentage`` of its points.

    ``epsilon_step`` is a fraction of the polygon's diagonal per RDP step
    (0.05 = 5%), so the loop is bounded for both tiny and 4K contours. At
    least three points are preserved; when no valid simplification reaches
    the target, the last valid polygon is returned.
    """
    points = _as_polygon(polygon)
    ratio = float(percentage)
    if not 0.0 <= ratio < 1.0:
        raise AutoVisionError(
            f"percentage must be in [0, 1), got {percentage!r}."
        )
    if epsilon_step <= 0:
        raise AutoVisionError(
            f"epsilon_step must be positive, got {epsilon_step!r}."
        )
    if points.shape[0] < 4 or ratio == 0.0:
        return points
    target = max(3, int(np.floor(points.shape[0] * (1.0 - ratio))))
    if points.shape[0] <= target:
        return points
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent == 0:
        return points
    best = points
    for step in range(1, 201):
        simplified = _rdp(points, step * epsilon_step * extent)
        if simplified.shape[0] < 3:
            break
        best = simplified
        if simplified.shape[0] <= target:
            break
    return best
