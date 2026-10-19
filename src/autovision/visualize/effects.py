# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Privacy, tracking and attention overlays for :class:`InferenceResult`.

These annotators complement the task-aware renderers in
:mod:`autovision.visualize.annotators`: they change *how* a detection is
presented rather than adding another result type. All drawing is Pillow +
NumPy, so the module stays in the optional-dependency-light core path.

The effects mirror the useful parts of Roboflow ``supervision`` (MIT); the
implementation is AutoVision-authored. See ``THIRD_PARTY_LICENSES.md`` and
``docs/provenance/LEDGER.md``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from autovision.errors import AutoVisionError
from autovision.results import InferenceResult, mask_to_bool
from autovision.visualize.annotators import (
    BaseAnnotator,
    ColorPalette,
    _as_numpy,
    _clamped_box,
    _class_index,
)

__all__ = [
    "BackgroundOverlayAnnotator",
    "BlurAnnotator",
    "BoxCornerAnnotator",
    "EdgeAnnotator",
    "HaloAnnotator",
    "HeatMapAnnotator",
    "PixelateAnnotator",
    "RoundBoxAnnotator",
    "TraceAnnotator",
]

RGB = tuple[int, int, int]

_ANCHORS = {
    "center": lambda x1, y1, x2, y2: ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    "top_center": lambda x1, y1, x2, y2: ((x1 + x2) / 2.0, y1),
    "bottom_center": lambda x1, y1, x2, y2: ((x1 + x2) / 2.0, y2),
    "center_left": lambda x1, y1, x2, y2: (x1, (y1 + y2) / 2.0),
    "center_right": lambda x1, y1, x2, y2: (x2, (y1 + y2) / 2.0),
    "top_left": lambda x1, y1, x2, y2: (x1, y1),
    "top_right": lambda x1, y1, x2, y2: (x2, y1),
    "bottom_left": lambda x1, y1, x2, y2: (x1, y2),
    "bottom_right": lambda x1, y1, x2, y2: (x2, y2),
}


def _check_anchor(position: str) -> str:
    if position not in _ANCHORS:
        raise AutoVisionError(
            f"Unknown anchor '{position}'; available: {sorted(_ANCHORS)}."
        )
    return position


def _labels(result: InferenceResult) -> np.ndarray | None:
    if result.labels is None:
        return None
    return _as_numpy(result.labels).reshape(-1)


def _palette_color(
    palette: ColorPalette,
    result: InferenceResult,
    index: int,
    labels: np.ndarray | None,
) -> RGB:
    return palette.by_index(_class_index(result, index, labels=labels))


def _masks(result: InferenceResult) -> np.ndarray | None:
    if result.masks is None or not result.has_instance_masks:
        return None
    return mask_to_bool(result.masks)


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return a half-open ``(x1, y1, x2, y2)`` crop for a non-empty mask."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _box_regions(
    result: InferenceResult, image_size: tuple[int, int]
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Return ``(index, half-open pixel box)`` for every clamped detection box."""
    if result.bboxes is None:
        return []
    boxes = _as_numpy(result.bboxes).reshape(-1, 4)
    regions: list[tuple[int, tuple[int, int, int, int]]] = []
    for index in range(boxes.shape[0]):
        clamped = _clamped_box(boxes[index].tolist(), *image_size)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        regions.append((index, (x1, y1, x2 + 1, y2 + 1)))
    return regions


def _mask_regions(
    result: InferenceResult,
    image_size: tuple[int, int] | None = None,
) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    """Return ``(index, half-open box, crop mask)`` for every non-empty mask."""
    masks = _masks(result)
    if masks is None:
        return []
    if image_size is not None and masks.shape[1:] != (
        image_size[1],
        image_size[0],
    ):
        return []
    regions: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for index in range(masks.shape[0]):
        box = _mask_box(masks[index])
        if box is None:
            continue
        x1, y1, x2, y2 = box
        regions.append((index, box, masks[index, y1:y2, x1:x2]))
    return regions


def _anchor_xy(box: Sequence[float], position: str) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return _ANCHORS[position](x1, y1, x2, y2)


class BoxCornerAnnotator(BaseAnnotator):
    """Draw L-shaped corner brackets instead of full boxes."""

    def __init__(
        self,
        color: RGB | ColorPalette | None = None,
        thickness: int = 2,
        corner_ratio: float = 0.2,
    ) -> None:
        self.color = color
        self.thickness = max(1, int(thickness))
        self.corner_ratio = min(max(float(corner_ratio), 0.01), 0.5)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.bboxes is None:
            return image
        palette = (
            self.color
            if isinstance(self.color, ColorPalette)
            else ColorPalette((self.color,))
            if self.color
            else ColorPalette()
        )
        labels = _labels(result)
        draw = ImageDraw.Draw(image)
        for index, (x1, y1, x2, y2) in _box_regions(result, image.size):
            x2 -= 1
            y2 -= 1
            width, height = x2 - x1, y2 - y1
            arm_x = max(2, int(width * self.corner_ratio))
            arm_y = max(2, int(height * self.corner_ratio))
            color = _palette_color(palette, result, index, labels)
            segments = (
                ((x1, y1), (x1 + arm_x, y1)),
                ((x1, y1), (x1, y1 + arm_y)),
                ((x2, y1), (x2 - arm_x, y1)),
                ((x2, y1), (x2, y1 + arm_y)),
                ((x1, y2), (x1 + arm_x, y2)),
                ((x1, y2), (x1, y2 - arm_y)),
                ((x2, y2), (x2 - arm_x, y2)),
                ((x2, y2), (x2, y2 - arm_y)),
            )
            for start, end in segments:
                draw.line([start, end], fill=color, width=self.thickness)
        return image


class RoundBoxAnnotator(BaseAnnotator):
    """Draw boxes with rounded corners."""

    def __init__(
        self,
        color: RGB | ColorPalette | None = None,
        thickness: int = 2,
        corner_radius: int = 8,
    ) -> None:
        self.color = color
        self.thickness = max(1, int(thickness))
        self.corner_radius = max(0, int(corner_radius))

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.bboxes is None:
            return image
        palette = (
            self.color
            if isinstance(self.color, ColorPalette)
            else ColorPalette((self.color,))
            if self.color
            else ColorPalette()
        )
        labels = _labels(result)
        draw = ImageDraw.Draw(image)
        for index, (x1, y1, x2, y2) in _box_regions(result, image.size):
            x2 -= 1
            y2 -= 1
            radius = min(self.corner_radius, max(0, min(x2 - x1, y2 - y1) // 2))
            color = _palette_color(palette, result, index, labels)
            if radius > 0:
                draw.rounded_rectangle(
                    (x1, y1, x2, y2),
                    radius=radius,
                    outline=color,
                    width=self.thickness,
                )
            else:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=self.thickness)
        return image


class BlurAnnotator(BaseAnnotator):
    """Blur detections for privacy.

    With ``use_masks=None`` (default) masks are used when present (only
    masked pixels are blurred), otherwise each detection box is blurred.
    ``use_masks=True`` never falls back to boxes; ``use_masks=False`` always
    blurs boxes. ``radius=None`` picks a radius from the region's shorter
    side.
    """

    def __init__(
        self, radius: int | None = None, *, use_masks: bool | None = None
    ) -> None:
        if radius is not None and radius < 1:
            raise AutoVisionError(f"radius must be >= 1, got {radius!r}.")
        self.radius = radius
        self.use_masks = use_masks

    def _radius_for(self, width: int, height: int) -> int:
        if self.radius is not None:
            return self.radius
        return max(1, min(width, height) // 6)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        mask_regions = (
            _mask_regions(result, image.size) if self.use_masks is not False else []
        )
        if mask_regions:
            for _, (x1, y1, x2, y2), mask in mask_regions:
                radius = self._radius_for(x2 - x1, y2 - y1)
                blurred = image.crop((x1, y1, x2, y2)).filter(
                    ImageFilter.GaussianBlur(radius)
                )
                mask_image = Image.fromarray(
                    (mask.astype(np.uint8) * 255), mode="L"
                )
                image.paste(blurred, (x1, y1), mask_image)
            return image
        if self.use_masks is True:
            return image
        for _, (x1, y1, x2, y2) in _box_regions(result, image.size):
            radius = self._radius_for(x2 - x1, y2 - y1)
            blurred = image.crop((x1, y1, x2, y2)).filter(
                ImageFilter.GaussianBlur(radius)
            )
            image.paste(blurred, (x1, y1))
        return image


class PixelateAnnotator(BaseAnnotator):
    """Pixelate detections (mosaic) for privacy.

    ``pixel_size=None`` picks a block size from the region's shorter side.
    Mask handling matches :class:`BlurAnnotator` (default: masks when
    available, boxes otherwise).
    """

    def __init__(
        self, pixel_size: int | None = None, *, use_masks: bool | None = None
    ) -> None:
        if pixel_size is not None and pixel_size < 1:
            raise AutoVisionError(
                f"pixel_size must be >= 1, got {pixel_size!r}."
            )
        self.pixel_size = pixel_size
        self.use_masks = use_masks

    def _block_for(self, width: int, height: int) -> int:
        if self.pixel_size is not None:
            return self.pixel_size
        return max(2, min(width, height) // 12)

    def _pixelate(self, crop: Image.Image, block: int) -> Image.Image:
        width, height = crop.size
        small = (max(1, width // block), max(1, height // block))
        return crop.resize(small, Image.NEAREST).resize(
            (width, height), Image.NEAREST
        )

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        mask_regions = (
            _mask_regions(result, image.size) if self.use_masks is not False else []
        )
        if mask_regions:
            for _, (x1, y1, x2, y2), mask in mask_regions:
                block = self._block_for(x2 - x1, y2 - y1)
                pixelated = self._pixelate(image.crop((x1, y1, x2, y2)), block)
                mask_image = Image.fromarray(
                    (mask.astype(np.uint8) * 255), mode="L"
                )
                image.paste(pixelated, (x1, y1), mask_image)
            return image
        if self.use_masks is True:
            return image
        for _, (x1, y1, x2, y2) in _box_regions(result, image.size):
            block = self._block_for(x2 - x1, y2 - y1)
            image.paste(
                self._pixelate(image.crop((x1, y1, x2, y2)), block), (x1, y1)
            )
        return image


class TraceAnnotator(BaseAnnotator):
    """Draw per-identity motion trails.

    Trails need a stable identity per row. Pass ``tracker_ids=`` explicitly
    or put ``tracker_id`` / ``tracker_ids`` in ``result.extra`` (the stream
    pipeline records it there). Rows without an identity are skipped.
    """

    def __init__(
        self,
        color: RGB | ColorPalette | None = None,
        position: str = "center",
        trace_length: int = 30,
        thickness: int = 2,
        smooth: bool = False,
        by_track: bool = False,
    ) -> None:
        self.color = color
        self.position = _check_anchor(position)
        self.trace_length = max(2, int(trace_length))
        self.thickness = max(1, int(thickness))
        self.smooth = bool(smooth)
        self.by_track = bool(by_track)
        self._history: dict[int, deque[tuple[float, float]]] = {}

    def reset(self) -> None:
        """Drop all accumulated trails (reuse across independent streams)."""
        self._history.clear()

    def _resolve_tracker_ids(
        self, result: InferenceResult, tracker_ids: Any
    ) -> np.ndarray | None:
        if tracker_ids is None:
            for source in (result.extra, result.metadata):
                for key in ("tracker_id", "tracker_ids"):
                    if key in source:
                        tracker_ids = source[key]
                        break
                if tracker_ids is not None:
                    break
        if tracker_ids is None:
            return None
        array = _as_numpy(tracker_ids).reshape(-1)
        if array.shape[0] != len(result):
            raise AutoVisionError(
                "tracker_ids must have one entry per detection, got "
                f"{array.shape[0]} for {len(result)} rows."
            )
        return array

    def _smooth(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not self.smooth or len(points) < 4:
            return points
        unique = [points[0]]
        for point in points[1:]:
            if point != unique[-1]:
                unique.append(point)
        if len(unique) < 4:
            return unique
        try:
            from scipy.interpolate import splev, splprep

            coordinates = np.asarray(unique, dtype=np.float64)
            spline, _ = splprep(
                [coordinates[:, 0], coordinates[:, 1]], s=0.0
            )
            x, y = splev(np.linspace(0.0, 1.0, 100), spline)
            return list(zip(np.asarray(x).tolist(), np.asarray(y).tolist()))
        except (ValueError, TypeError):  # pragma: no cover - degenerate tracks
            return unique

    def annotate(
        self,
        image: Image.Image,
        result: InferenceResult,
        *,
        tracker_ids: Any = None,
    ) -> Image.Image:
        if result.bboxes is None:
            raise AutoVisionError(
                "TraceAnnotator requires result.bboxes (and tracker ids)."
            )
        if len(result) == 0:
            return image
        ids = self._resolve_tracker_ids(result, tracker_ids)
        if ids is None:
            raise AutoVisionError(
                "TraceAnnotator needs tracker ids: pass tracker_ids= or set "
                "result.extra['tracker_id'] / result.extra['tracker_ids']."
            )
        palette = (
            self.color
            if isinstance(self.color, ColorPalette)
            else ColorPalette((self.color,))
            if self.color
            else ColorPalette()
        )
        labels = _labels(result)
        boxes = _as_numpy(result.bboxes).reshape(-1, 4)
        draw = ImageDraw.Draw(image)
        for index in range(min(len(result), ids.shape[0])):
            try:
                tracker_id = int(float(ids[index]))
            except (TypeError, ValueError):
                continue
            if tracker_id < 0 or not np.isfinite(float(ids[index])):
                continue
            anchor = _anchor_xy(boxes[index].tolist(), self.position)
            history = self._history.setdefault(
                tracker_id, deque(maxlen=self.trace_length)
            )
            if not history or history[-1] != anchor:
                history.append(anchor)
            if len(history) < 2:
                continue
            color = _palette_color(
                palette,
                result,
                tracker_id if self.by_track else index,
                labels,
            )
            draw.line(
                self._smooth(list(history)),
                fill=color,
                width=self.thickness,
                joint="curve",
            )
        return image


_RAMP_POSITIONS = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
_RAMP_COLORS = np.asarray(
    (
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
    ),
    dtype=np.float32,
)


def _colorize_heat(normalized: np.ndarray) -> np.ndarray:
    """Map ``[0, 1]`` heat to an RGB blue-to-red ramp."""
    channels = [
        np.interp(normalized, _RAMP_POSITIONS, _RAMP_COLORS[:, index])
        for index in range(3)
    ]
    return np.stack(channels, axis=-1)


class HeatMapAnnotator(BaseAnnotator):
    """Accumulate detection anchors into a temporal heatmap overlay."""

    def __init__(
        self,
        position: str = "bottom_center",
        radius: int = 40,
        opacity: float = 0.35,
        blur_radius: int = 15,
    ) -> None:
        self.position = _check_anchor(position)
        self.radius = max(1, int(radius))
        self.opacity = min(max(float(opacity), 0.0), 1.0)
        self.blur_radius = max(0, int(blur_radius))
        self.heat: np.ndarray | None = None

    def reset(self) -> None:
        """Discard accumulated heat (reuse across independent streams)."""
        self.heat = None

    def _stamp(self, center: tuple[float, float]) -> None:
        heat = self.heat
        if heat is None:
            return
        size = 2 * self.radius + 1
        y, x = np.ogrid[
            -self.radius : self.radius + 1, -self.radius : self.radius + 1
        ]
        disk = (x * x + y * y <= self.radius * self.radius).astype(np.float32)
        cx, cy = int(round(center[0])), int(round(center[1]))
        x1, y1 = cx - self.radius, cy - self.radius
        x2, y2 = x1 + size, y1 + size
        height, width = heat.shape
        sx1, sy1 = max(0, -x1), max(0, -y1)
        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(width, x2), min(height, y2)
        if dx2 <= dx1 or dy2 <= dy1:
            return
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)
        heat[dy1:dy2, dx1:dx2] += disk[sy1:sy2, sx1:sx2]

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.bboxes is None or len(result) == 0:
            return image
        if self.heat is None or self.heat.shape != (image.height, image.width):
            self.heat = np.zeros((image.height, image.width), dtype=np.float32)
        boxes = _as_numpy(result.bboxes).reshape(-1, 4)
        for box in boxes:
            self._stamp(_anchor_xy(box.tolist(), self.position))
        maximum = float(self.heat.max())
        if maximum <= 0:
            return image
        normalized = np.clip(self.heat / maximum, 0.0, 1.0)
        if self.blur_radius > 0:
            heat_image = Image.fromarray(
                (normalized * 255.0).astype(np.uint8), mode="L"
            ).filter(ImageFilter.GaussianBlur(self.blur_radius))
            normalized = np.asarray(heat_image).astype(np.float32) / 255.0
            normalized = np.clip(
                normalized / max(float(normalized.max()), 1e-6), 0.0, 1.0
            )
        colors = _colorize_heat(normalized)
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        alpha = (self.opacity * normalized)[..., None]
        base = base * (1.0 - alpha) + colors * alpha
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


class HaloAnnotator(BaseAnnotator):
    """Draw a colored halo around instance masks."""

    def __init__(
        self,
        palette: ColorPalette | None = None,
        halo_width: int = 12,
        opacity: float = 0.5,
    ) -> None:
        self.palette = palette or ColorPalette()
        self.halo_width = max(1, int(halo_width))
        self.opacity = min(max(float(opacity), 0.0), 1.0)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        masks = _masks(result)
        if masks is None or masks.shape[1:] != (image.height, image.width):
            return image
        labels = _labels(result)
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        height, width = base.shape[:2]
        size = 2 * self.halo_width + 1
        for index in range(min(masks.shape[0], len(result))):
            box = _mask_box(masks[index])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            left = max(0, x1 - self.halo_width)
            top = max(0, y1 - self.halo_width)
            right = min(width, x2 + self.halo_width)
            bottom = min(height, y2 + self.halo_width)
            crop = masks[index, top:bottom, left:right]
            if crop.size == 0:
                continue
            dilated = np.asarray(
                Image.fromarray((crop.astype(np.uint8) * 255), mode="L").filter(
                    ImageFilter.MaxFilter(size)
                )
            ) > 0
            ring = dilated & ~crop
            if not ring.any():
                continue
            color = np.asarray(
                _palette_color(self.palette, result, index, labels),
                dtype=np.float32,
            )
            region = base[top:bottom, left:right]
            region[ring] = region[ring] * (1.0 - self.opacity) + color * self.opacity
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


class EdgeAnnotator(BaseAnnotator):
    """Outline instance-mask boundaries with a solid color."""

    def __init__(
        self,
        palette: ColorPalette | None = None,
        thickness: int = 1,
        opacity: float = 1.0,
    ) -> None:
        self.palette = palette or ColorPalette()
        self.thickness = max(1, int(thickness))
        self.opacity = min(max(float(opacity), 0.0), 1.0)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        masks = _masks(result)
        if masks is None or masks.shape[1:] != (image.height, image.width):
            return image
        labels = _labels(result)
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        dilate_size = 2 * self.thickness + 1
        for index in range(min(masks.shape[0], len(result))):
            box = _mask_box(masks[index])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = masks[index, y1:y2, x1:x2]
            if crop.size == 0:
                continue
            padded = np.pad(crop, 1, mode="constant", constant_values=False)
            eroded = padded.copy()
            eroded[1:, :] &= padded[:-1, :]
            eroded[:-1, :] &= padded[1:, :]
            eroded[:, 1:] &= padded[:, :-1]
            eroded[:, :-1] &= padded[:, 1:]
            boundary = (padded & ~eroded)[1:-1, 1:-1]
            if self.thickness > 1:
                boundary = (
                    np.asarray(
                        Image.fromarray(
                            (boundary.astype(np.uint8) * 255), mode="L"
                        ).filter(ImageFilter.MaxFilter(dilate_size))
                    )
                    > 0
                )
            if not boundary.any():
                continue
            color = np.asarray(
                _palette_color(self.palette, result, index, labels),
                dtype=np.float32,
            )
            region = base[y1:y2, x1:x2]
            region[boundary] = (
                region[boundary] * (1.0 - self.opacity) + color * self.opacity
            )
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


class BackgroundOverlayAnnotator(BaseAnnotator):
    """Dim everything outside the union of instance masks."""

    def __init__(
        self,
        color: RGB = (0, 0, 0),
        opacity: float = 0.6,
    ) -> None:
        self.color = (int(color[0]), int(color[1]), int(color[2]))
        self.opacity = min(max(float(opacity), 0.0), 1.0)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        masks = _masks(result)
        if (
            masks is None
            or masks.shape[0] == 0
            or masks.shape[1:] != (image.height, image.width)
        ):
            return image
        union = masks.any(axis=0)
        if not union.any():
            return image
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        color = np.asarray(self.color, dtype=np.float32)
        outside = ~union
        base[outside] = (
            base[outside] * (1.0 - self.opacity) + color * self.opacity
        )
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
