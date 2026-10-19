# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""YOLO object detection dataset.

Ultralytics layout: ``<split>/images/...`` with labels mirrored under a
``labels`` directory (the first ``images`` path part is replaced). Labels
are per-image ``class cx cy w h`` text files in normalized cxcywh.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autovision.data.label_helpers import contiguous_class_id_map

__all__ = ["YOLOObjectDetectionDataArgs", "yolo_images_to_labels_path"]


def yolo_images_to_labels_path(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


class YOLOObjectDetectionDataArgs(BaseModel):
    format: str = "yolo"
    path: Path
    train: Path
    val: Path
    names: dict[int, str]  # required: no auto inference, like the baseline
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False
    data_config_file: Path | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(strict=False, extra="forbid")

    @model_validator(mode="after")
    def _validate_split_layout(self) -> "YOLOObjectDetectionDataArgs":
        for split in (self.train, self.val):
            if "images" not in split.parts:
                raise ValueError(
                    f"YOLO split path '{split}' must contain an 'images' part."
                )
        return self

    def resolve_data_paths(self, base_dir: Path | None = None) -> None:
        # Relative splits anchor to `path` (Ultralytics layout) even when
        # invoked with a dict payload (base_dir is only set for config
        # files, where it anchors `path` itself first).
        if base_dir is not None and not self.path.is_absolute():
            self.path = base_dir / self.path
        if not self.path.is_absolute():
            return
        for name in ("train", "val"):
            path: Path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, self.path / path)

    def classes(self) -> dict[int, str]:
        included = set(self.names) - set(self.ignore_classes or [])
        internal = contiguous_class_id_map(sorted(included))
        return {internal[class_id]: self.names[class_id] for class_id in sorted(included)}

    def _mmap_hash(self, split: Path) -> str:
        return str(
            (
                (self.path / split).resolve(),
                tuple(sorted(self.names.items())),
                sorted(self.ignore_classes or []),
                self.skip_if_label_file_missing,
            )
        )

    def train_data_mmap_hash(self) -> str:
        return self._mmap_hash(self.train)

    def val_data_mmap_hash(self) -> str:
        return self._mmap_hash(self.val)

    def list_image_info(self, split: str = "train") -> Iterator[dict[str, Any]]:
        images_dir = getattr(self, split)
        labels_dir = yolo_images_to_labels_path(images_dir)
        included = set(self.names) - set(self.ignore_classes or [])
        internal = contiguous_class_id_map(sorted(included))

        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue
            relative = image_path.relative_to(images_dir)
            label_path = (labels_dir / relative).with_suffix(".txt")
            boxes: list[list[float]] = []
            labels: list[int] = []
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    if class_id not in internal:
                        continue
                    cx, cy, w, h = (float(v) for v in parts[1:])
                    boxes.append([cx, cy, w, h])
                    labels.append(internal[class_id])
            elif self.skip_if_label_file_missing and split == "train":
                continue
            yield {
                "image_path": str(image_path),
                "bboxes": json.dumps(boxes),
                "class_labels": json.dumps(labels),
            }
