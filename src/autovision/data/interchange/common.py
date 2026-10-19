# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared COCO assembly for annotation converters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from autovision.errors import AutoVisionError

__all__ = ["CocoBuilder", "polygon_area"]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace area (absolute value) for an ordered polygon."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


@dataclass
class CocoBuilder:
    """Accumulate images/annotations and emit a COCO payload."""

    images_dir: Path | None = None
    _images: list[dict[str, Any]] = field(default_factory=list)
    _annotations: list[dict[str, Any]] = field(default_factory=list)
    _categories: set[str] = field(default_factory=set)
    _image_ids: dict[str, int] = field(default_factory=dict)

    def _relative(self, file_name: str) -> str:
        path = Path(file_name)
        if self.images_dir is not None and path.is_absolute():
            try:
                return path.resolve().relative_to(
                    Path(self.images_dir).resolve()
                ).as_posix()
            except ValueError:
                return path.as_posix()
        return path.as_posix()

    def add_image(
        self, file_name: str, width: int, height: int
    ) -> int:
        key = self._relative(file_name)
        if key in self._image_ids:
            return self._image_ids[key]
        if width <= 0 or height <= 0:
            raise AutoVisionError(
                f"Image '{file_name}' has invalid size {width}x{height}."
            )
        image_id = len(self._images) + 1
        self._images.append(
            {"id": image_id, "file_name": key, "width": int(width), "height": int(height)}
        )
        self._image_ids[key] = image_id
        return image_id

    def add_category(self, name: str) -> None:
        """Register a class even if no image uses it (empty class)."""
        if name:
            self._categories.add(name)

    def add_annotation(
        self,
        image_id: int,
        category: str,
        bbox: Sequence[float],
        *,
        segmentation: Sequence[Any] | None = None,
        area: float | None = None,
        iscrowd: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> int:
        if not category:
            raise AutoVisionError("Annotation category must be non-empty.")
        values = [float(value) for value in bbox]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise AutoVisionError(f"Invalid bbox: {bbox!r}.")
        width, height = values[2], values[3]
        if width <= 0 or height <= 0:
            raise AutoVisionError(f"Invalid bbox size: {bbox!r}.")
        annotation_id = len(self._annotations) + 1
        payload: dict[str, Any] = {
            "id": annotation_id,
            "image_id": int(image_id),
            "category_id": category,  # resolved to an int in to_dict()
            "bbox": values,
            "area": float(area) if area is not None else width * height,
            "iscrowd": int(iscrowd),
        }
        if segmentation is not None:
            payload["segmentation"] = segmentation
        if extra:
            payload.update(extra)
        self._annotations.append(payload)
        self._categories.add(category)
        return annotation_id

    def to_dict(self) -> dict[str, Any]:
        names = sorted(self._categories)
        category_ids = {name: index + 1 for index, name in enumerate(names)}
        categories = [
            {"id": category_ids[name], "name": name}
            for name in names
        ]
        annotations = []
        for entry in self._annotations:
            payload = dict(entry)
            payload["category_id"] = category_ids[payload["category_id"]]
            annotations.append(payload)
        return {
            "images": [dict(image) for image in self._images],
            "annotations": annotations,
            "categories": categories,
        }

    def write(self, out_json: str | Path) -> Path:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
