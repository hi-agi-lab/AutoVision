# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task-aware image annotators built on Pillow.

The renderers are deliberately small and composable: each one consumes an
:class:`~autovision.results.InferenceResult` and draws one layer
(boxes, masks, polygons, keypoints, labels). ``annotate_result`` picks the
layers that match the populated result fields, so a pose/OBB/classification
record renders instead of raising.

All annotators mutate the image passed to them and return it; the public
entry points copy the input once, so callers never lose their original.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from autovision.errors import AutoVisionError
from autovision.results import InferenceResult, mask_to_bool
from autovision.visualize.skeletons import COCO_SKELETON, SKELETONS

__all__ = [
    "BaseAnnotator",
    "BoxAnnotator",
    "ClassificationAnnotator",
    "ColorPalette",
    "DepthAnnotator",
    "KeypointAnnotator",
    "LabelAnnotator",
    "MaskAnnotator",
    "PanopticAnnotator",
    "PolygonAnnotator",
    "SemanticMaskAnnotator",
    "annotate_result",
    "draw_result",
]

RGB = tuple[int, int, int]

#: 20-color categorical palette. Indexed by class id, so a class keeps its
#: color across boxes, masks, polygons and keypoints in one frame.
DEFAULT_PALETTE: tuple[RGB, ...] = (
    (163, 81, 251),
    (255, 64, 64),
    (255, 161, 160),
    (255, 118, 51),
    (255, 182, 51),
    (209, 212, 53),
    (76, 251, 18),
    (148, 207, 26),
    (64, 222, 138),
    (27, 150, 64),
    (0, 214, 193),
    (46, 156, 170),
    (0, 196, 255),
    (54, 71, 151),
    (102, 117, 255),
    (0, 25, 239),
    (134, 58, 255),
    (83, 0, 135),
    (205, 58, 255),
    (255, 151, 202),
)

#: COCO-17 skeleton, 1-based pairs (the official COCO keypoint order).
# Re-exported from ``autovision.visualize.skeletons`` for existing imports.


@dataclass(frozen=True)
class ColorPalette:
    """Indexed RGB palette; indexes wrap around."""

    colors: Sequence[RGB] = DEFAULT_PALETTE

    def by_index(self, index: int) -> RGB:
        if not self.colors:
            raise ValueError("ColorPalette must contain at least one color.")
        return tuple(self.colors[int(index) % len(self.colors)])  # type: ignore[return-value]


class BaseAnnotator(ABC):
    """Draw one result layer onto a Pillow image."""

    requires_mask: bool = False

    @abstractmethod
    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        """Draw onto ``image`` and return the same (mutated) image."""


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _class_index(
    result: InferenceResult, index: int, *, labels: np.ndarray | None = None
) -> int:
    if labels is None:
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
    if labels is None or index >= int(labels.shape[0]):
        return index
    return int(labels[index])


def _label_text(
    result: InferenceResult,
    index: int,
    class_names: Mapping[int, str] | None,
    *,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> str:
    label = _class_index(result, index, labels=labels)
    name = (
        class_names.get(label, str(label))
        if class_names is not None
        else str(label)
    )
    if scores is None:
        scores = (
            _as_numpy(result.scores).reshape(-1)
            if result.scores is not None
            else None
        )
    if scores is None or index >= int(scores.shape[0]):
        return name
    return f"{name} {float(scores[index]):.2f}"


def _clamped_box(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width - 1, int(round(x2))), min(height - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class BoxAnnotator(BaseAnnotator):
    """Draw axis-aligned boxes from ``result.bboxes``."""

    def __init__(
        self,
        color: RGB | ColorPalette | None = None,
        thickness: int = 2,
    ) -> None:
        self.color = color
        self.thickness = max(1, int(thickness))

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.bboxes is None:
            return image
        draw = ImageDraw.Draw(image)
        width, height = image.size
        palette = self.color if isinstance(self.color, ColorPalette) else ColorPalette((self.color,)) if self.color else ColorPalette()
        boxes = _as_numpy(result.bboxes).reshape(-1, 4)
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
        for index in range(boxes.shape[0]):
            clamped = _clamped_box(boxes[index].tolist(), width, height)
            if clamped is None:
                continue
            draw.rectangle(
                clamped,
                outline=palette.by_index(
                    _class_index(result, index, labels=labels)
                ),
                width=self.thickness,
            )
        return image


class LabelAnnotator(BaseAnnotator):
    """Draw a filled label background above each box/polygon/keypoint row."""

    def __init__(
        self,
        text_color: RGB = (255, 255, 255),
        background: RGB | ColorPalette | None = None,
        text_size: int = 12,
        padding: int = 4,
    ) -> None:
        self.text_color = text_color
        self.background = background
        self.text_size = max(6, int(text_size))
        self.padding = max(0, int(padding))

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        fill: RGB,
        image_size: tuple[int, int],
    ) -> None:
        font = _font(self.text_size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        width, height = image_size
        x = min(max(0, xy[0]), max(0, width - text_w - 2 * self.padding))
        y = xy[1] - text_h - 2 * self.padding
        if y < 0:
            y = min(max(0, xy[1]), max(0, height - text_h - 2 * self.padding))
        draw.rectangle(
            (x, y, x + text_w + 2 * self.padding, y + text_h + 2 * self.padding),
            fill=fill,
        )
        draw.text((x + self.padding, y + self.padding), text, fill=self.text_color, font=font)

    def annotate(
        self,
        image: Image.Image,
        result: InferenceResult,
        *,
        class_names: Mapping[int, str] | None = None,
    ) -> Image.Image:
        if result.labels is None and result.scores is None:
            return image
        draw = ImageDraw.Draw(image)
        boxes = (
            _as_numpy(result.bboxes).reshape(-1, 4)
            if result.bboxes is not None
            else None
        )
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
        scores = (
            _as_numpy(result.scores).reshape(-1)
            if result.scores is not None
            else None
        )
        keypoints = (
            _as_numpy(result.keypoints)
            if result.keypoints is not None
            else None
        )
        polygons = (
            _as_numpy(result.polygons).reshape(-1, 4, 2)
            if result.polygons is not None
            else None
        )
        masks = (
            mask_to_bool(result.masks) if result.has_instance_masks else None
        )
        for index in range(len(result)):
            if boxes is not None and index < boxes.shape[0]:
                clamped = _clamped_box(boxes[index].tolist(), *image.size)
                if clamped is None:
                    continue
                xy = (clamped[0], clamped[1])
            elif keypoints is not None:
                person = keypoints[index].reshape(-1, 3)
                visible = person[np.isfinite(person[:, 0]) & (person[:, 2] > 0)]
                if visible.size == 0:
                    continue
                xy = (int(visible[:, 0].min()), int(visible[:, 1].min()))
            elif polygons is not None:
                if index >= polygons.shape[0]:
                    continue
                xy = (
                    int(polygons[index][:, 0].min()),
                    int(polygons[index][:, 1].min()),
                )
            elif masks is not None and index < masks.shape[0]:
                rows, cols = np.nonzero(masks[index])
                if rows.size == 0:
                    continue
                xy = (int(cols.min()), int(rows.min()))
            else:
                continue
            background = self.background if isinstance(self.background, ColorPalette) else None
            fill = (
                background.by_index(
                    _class_index(result, index, labels=labels)
                )
                if background is not None
                else ColorPalette().by_index(
                    _class_index(result, index, labels=labels)
                )
            )
            self._draw_label(
                draw,
                xy,
                _label_text(
                    result,
                    index,
                    class_names,
                    labels=labels,
                    scores=scores,
                ),
                fill,
                image.size,
            )
        return image


class MaskAnnotator(BaseAnnotator):
    """Blend per-instance masks onto the image (alpha-composited)."""

    requires_mask = True

    def __init__(self, alpha: float = 0.45, palette: ColorPalette | None = None) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.palette = palette or ColorPalette()

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.masks is None or not result.has_instance_masks:
            return image
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        height, width = base.shape[:2]
        masks: np.ndarray = mask_to_bool(result.masks)
        boxes = (
            _as_numpy(result.bboxes).reshape(-1, 4)
            if result.bboxes is not None
            else None
        )
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
        for index in range(min(masks.shape[0], len(result))):
            mask = masks[index]
            if mask.shape != (height, width):
                continue
            clamped = None
            if boxes is not None and index < boxes.shape[0]:
                clamped = _clamped_box(boxes[index].tolist(), width, height)
            if clamped is None:
                rows, cols = np.nonzero(mask)
                if rows.size == 0:
                    continue
                clamped = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
            x1, y1, x2, y2 = clamped
            crop = mask[y1 : y2 + 1, x1 : x2 + 1]
            if not crop.any():
                continue
            color = np.asarray(
                self.palette.by_index(
                    _class_index(result, index, labels=labels)
                ),
                dtype=np.float32,
            )
            region = base[y1 : y2 + 1, x1 : x2 + 1]
            region[crop] = region[crop] * (1.0 - self.alpha) + color * self.alpha
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


class PolygonAnnotator(BaseAnnotator):
    """Draw oriented boxes from ``result.polygons``."""

    def __init__(self, color: RGB | ColorPalette | None = None, thickness: int = 2) -> None:
        self.color = color
        self.thickness = max(1, int(thickness))

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.polygons is None:
            return image
        points = _as_numpy(result.polygons).reshape(-1, 4, 2)
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
        draw = ImageDraw.Draw(image)
        palette = self.color if isinstance(self.color, ColorPalette) else ColorPalette((self.color,)) if self.color else ColorPalette()
        width, height = image.size
        for index in range(points.shape[0]):
            polygon = points[index]
            if not np.isfinite(polygon).all():
                continue
            polygon = polygon.copy()
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            xy = [tuple(float(value) for value in point) for point in polygon]
            color = palette.by_index(
                _class_index(result, index, labels=labels)
            )
            for start, end in zip(xy, xy[1:] + xy[:1]):
                draw.line([start, end], fill=color, width=self.thickness)
        return image


class KeypointAnnotator(BaseAnnotator):
    """Draw vertices and skeleton edges from ``result.keypoints``.

    The default is the COCO-17 skeleton. ``skeleton`` accepts a registry name
    (``"coco"`` / ``"ghum"``) or an explicit zero-based ``(start, end)`` edge
    sequence for custom pose models.
    """

    def __init__(
        self,
        vertex_radius: int = 3,
        edge_thickness: int = 2,
        palette: ColorPalette | None = None,
        visibility_threshold: float = 0.0,
        skeleton: str | Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if skeleton is None:
            self.skeleton = "coco"
            self.edges = tuple((start - 1, end - 1) for start, end in COCO_SKELETON)
        elif isinstance(skeleton, str):
            name = skeleton.lower()
            if name not in SKELETONS:
                raise AutoVisionError(
                    f"Unknown skeleton '{skeleton}'; available: "
                    f"{sorted(SKELETONS)}."
                )
            self.skeleton = name
            self.edges = SKELETONS[name]
        else:
            self.skeleton = "custom"
            self.edges = tuple((int(start), int(end)) for start, end in skeleton)
        for start, end in self.edges:
            if start < 0 or end < 0:
                raise AutoVisionError(
                    "Keypoint skeleton edges must be zero-based and non-negative."
                )
        self.vertex_radius = max(1, int(vertex_radius))
        self.edge_thickness = max(1, int(edge_thickness))
        self.palette = palette or ColorPalette()
        self.visibility_threshold = float(visibility_threshold)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.keypoints is None:
            return image
        points = _as_numpy(result.keypoints)
        if points.size == 0:
            return image
        if points.ndim == 2:
            if points.shape[1] % 3:
                return image
            points = points.reshape(points.shape[0], -1, 3)
        labels = (
            _as_numpy(result.labels).reshape(-1)
            if result.labels is not None
            else None
        )
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for index in range(points.shape[0]):
            person = points[index]
            if person.shape[0] < 2:
                continue
            visible = (
                np.isfinite(person[:, 0])
                & np.isfinite(person[:, 1])
                & (person[:, 2] > self.visibility_threshold)
            )
            color = self.palette.by_index(
                _class_index(result, index, labels=labels)
            )
            for a, b in self.edges:
                if a >= person.shape[0] or b >= person.shape[0]:
                    continue
                if not (visible[a] and visible[b]):
                    continue
                draw.line(
                    [
                        (float(person[a, 0]), float(person[a, 1])),
                        (float(person[b, 0]), float(person[b, 1])),
                    ],
                    fill=color,
                    width=self.edge_thickness,
                )
            for point_index in range(person.shape[0]):
                if not visible[point_index]:
                    continue
                x, y = float(person[point_index, 0]), float(person[point_index, 1])
                if not (0 <= x < width and 0 <= y < height):
                    continue
                radius = self.vertex_radius
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline=(255, 255, 255),
                )
        return image


class ClassificationAnnotator(BaseAnnotator):
    """Draw the top-1 class label for image-level classification results."""

    def __init__(
        self,
        text_color: RGB = (255, 255, 255),
        text_size: int = 14,
        padding: int = 6,
    ) -> None:
        self.text_color = text_color
        self.text_size = max(6, int(text_size))
        self.padding = max(0, int(padding))

    def annotate(
        self,
        image: Image.Image,
        result: InferenceResult,
        *,
        class_names: Mapping[int, str] | None = None,
    ) -> Image.Image:
        if result.labels is None or len(result) == 0:
            return image
        draw = ImageDraw.Draw(image)
        font = _font(self.text_size)
        text = _label_text(result, 0, class_names)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.rectangle(
            (0, 0, right - left + 2 * self.padding, bottom - top + 2 * self.padding),
            fill=(0, 0, 0),
        )
        draw.text((self.padding, self.padding), text, fill=self.text_color, font=font)
        return image


def _blend_label_map(
    image: Image.Image,
    labels: np.ndarray,
    alpha: float,
    palette: ColorPalette,
    valid: np.ndarray | None = None,
) -> Image.Image:
    """Alpha-blend a ``(H, W)`` integer label map with a categorical palette."""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = base.shape[:2]
    if labels.shape != (height, width):
        resized = Image.fromarray(labels.astype(np.int32), mode="I").resize(
            (width, height), Image.NEAREST
        )
        labels = np.asarray(resized)
        if valid is not None:
            valid = np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (width, height), Image.NEAREST
                )
            ) > 0
    colors = np.asarray(palette.colors, dtype=np.float32)
    keep = labels >= 0
    if valid is not None:
        keep = keep & valid
    mapped = colors[np.clip(labels, 0, len(colors) - 1) % len(colors)]
    base[keep] = base[keep] * (1.0 - alpha) + mapped[keep] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


class SemanticMaskAnnotator(BaseAnnotator):
    """Colorize a semantic label map (``result.mask``)."""

    requires_mask = True

    def __init__(self, alpha: float = 0.55, palette: ColorPalette | None = None) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.palette = palette or ColorPalette()

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.mask is None:
            return image
        labels = _as_numpy(result.mask)
        if labels.ndim != 2:
            return image
        return _blend_label_map(image, labels, self.alpha, self.palette)


class PanopticAnnotator(BaseAnnotator):
    """Colorize a panoptic ``(H, W, 2)`` map and outline segment boundaries."""

    requires_mask = True

    def __init__(self, alpha: float = 0.55, palette: ColorPalette | None = None) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.palette = palette or ColorPalette()

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.masks is None or not result.is_panoptic:
            return image
        masks = _as_numpy(result.masks)
        if masks.ndim != 3 or masks.shape[-1] != 2:
            return image
        canvas = _blend_label_map(
            image,
            masks[..., 0],
            self.alpha,
            self.palette,
            valid=masks[..., 1] >= 0,
        )
        segments = masks[..., 1]
        height, width = segments.shape
        edge = np.zeros((height, width), dtype=bool)
        edge[:, 1:] |= segments[:, 1:] != segments[:, :-1]
        edge[1:, :] |= segments[1:, :] != segments[:-1, :]
        if edge.any():
            base = np.asarray(canvas).copy()
            base[edge] = (255, 255, 255)
            canvas = Image.fromarray(base, mode="RGB")
        return canvas


class DepthAnnotator(BaseAnnotator):
    """Render a depth map with a blue-to-red gradient over the image."""

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)

    def annotate(self, image: Image.Image, result: InferenceResult) -> Image.Image:
        if result.depth is None:
            return image
        depth = _as_numpy(result.depth)
        if depth.ndim != 2:
            return image
        finite = np.isfinite(depth)
        if not finite.any():
            return image
        finite_values = depth[finite]
        low, high = float(finite_values.min()), float(finite_values.max())
        normalized = np.zeros_like(depth, dtype=np.float32)
        if high > low:
            normalized[finite] = (depth[finite] - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
        red = np.clip(1.5 - np.abs(normalized * 4 - 1), 0, 1)
        blue = np.clip(1.5 - np.abs(normalized * 4 - 3), 0, 1)
        green = np.clip(1.5 - np.abs(normalized * 4 - 2), 0, 1)
        colored = np.stack([red, green, blue], axis=-1) * 255.0
        height, width = image.size[1], image.size[0]
        if colored.shape[:2] != (height, width):
            colored = np.asarray(
                Image.fromarray(colored.astype(np.uint8), mode="RGB").resize(
                    (width, height), Image.BILINEAR
                ),
                dtype=np.float32,
            )
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        blended = base * (1.0 - self.alpha) + colored * self.alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def annotate_result(
    image: Image.Image,
    result: InferenceResult | Mapping[str, Any],
    *,
    class_names: Mapping[int, str] | None = None,
    min_score: float = 0.0,
    box_color: RGB | ColorPalette | None = None,
) -> Image.Image:
    """Render every populated layer of ``result`` onto a copy of ``image``."""
    if not isinstance(result, InferenceResult):
        result = InferenceResult.from_record(result)
    if min_score > 0:
        result = result.filter_scores(min_score)

    canvas = image.copy()
    if result.mask is not None:
        canvas = SemanticMaskAnnotator().annotate(canvas, result)
    if result.depth is not None:
        canvas = DepthAnnotator().annotate(canvas, result)
    if result.masks is not None:
        if result.is_panoptic:
            canvas = PanopticAnnotator().annotate(canvas, result)
        else:
            canvas = MaskAnnotator().annotate(canvas, result)
    if result.polygons is not None:
        canvas = PolygonAnnotator(color=box_color).annotate(canvas, result)
    if result.keypoints is not None:
        canvas = KeypointAnnotator().annotate(canvas, result)
    if result.bboxes is not None:
        canvas = BoxAnnotator(color=box_color).annotate(canvas, result)
    if result.labels is not None and (
        result.bboxes is not None
        or result.polygons is not None
        or result.keypoints is not None
        or result.has_instance_masks
    ):
        canvas = LabelAnnotator().annotate(
            canvas, result, class_names=class_names
        )
    elif (
        result.labels is not None
        and result.bboxes is None
        and result.polygons is None
        and result.keypoints is None
        and result.mask is None
        and result.masks is None
        and result.depth is None
    ):
        canvas = ClassificationAnnotator().annotate(
            canvas, result, class_names=class_names
        )
    return canvas


def draw_result(
    image: Image.Image,
    result: InferenceResult | Mapping[str, Any],
    *,
    class_names: Mapping[int, str] | None = None,
    min_score: float = 0.0,
    box_color: RGB | ColorPalette | None = None,
) -> Image.Image:
    """Alias for :func:`annotate_result` (public rendering entry point)."""
    return annotate_result(
        image,
        result,
        class_names=class_names,
        min_score=min_score,
        box_color=box_color,
    )
