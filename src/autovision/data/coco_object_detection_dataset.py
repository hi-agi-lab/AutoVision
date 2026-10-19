# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO object detection dataset.

Rows are built once from the COCO json and serialized into the mmap cache;
the internal representation is normalized ``cxcywh``. The class list comes
from the **train** annotation file's ``categories`` (like the baseline).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autovision.data.label_helpers import contiguous_class_id_map
from autovision.errors import AutoVisionError

__all__ = ["COCOObjectDetectionDataArgs"]


class SplitArgs(BaseModel):
    annotations: Path
    images: Path | None = None

    model_config = ConfigDict(strict=False)


class COCOObjectDetectionDataArgs(BaseModel):
    format: str = "coco"
    train: SplitArgs
    val: SplitArgs
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False
    data_config_file: Path | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(strict=False, extra="forbid")

    def resolve_data_paths(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            return
        for split in (self.train, self.val):
            for name in ("annotations", "images"):
                path: Path | None = getattr(split, name)
                if path is not None and not path.is_absolute():
                    setattr(split, name, base_dir / path)

    def _classes(self) -> dict[int, str]:
        with open(self.train.annotations) as file:
            annotations = json.load(file)
        return {
            int(c["id"]): str(c["name"]) for c in annotations.get("categories", [])
        }

    def classes(self) -> dict[int, str]:
        """Class id -> name (internal contiguous ids)."""
        user_classes = self._classes()
        included = _included_class_ids(user_classes, self.ignore_classes)
        internal = contiguous_class_id_map(sorted(included))
        return {internal[class_id]: user_classes[class_id] for class_id in sorted(included)}

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                self.train.annotations.resolve(),
                self.train.annotations.stat().st_mtime,
                self.train.images,
                sorted(self.ignore_classes or []),
                self.skip_if_annotations_missing,
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                self.val.annotations.resolve(),
                self.val.annotations.stat().st_mtime,
                self.val.images,
                sorted(self.ignore_classes or []),
                self.skip_if_annotations_missing,
            )
        )

    def list_image_info(self, split: str = "train") -> Iterator[dict[str, Any]]:
        split_args: SplitArgs = getattr(self, split)
        with open(split_args.annotations) as file:
            annotations = json.load(file)
        images_dir = split_args.images or Path(split_args.annotations).parent

        image_id_to_file: dict[int, tuple[str, int, int]] = {}
        images_root = Path(images_dir).resolve()
        for image in annotations.get("images", []):
            candidate = (Path(images_dir) / str(image["file_name"])).resolve()
            if not candidate.is_relative_to(images_root):
                raise AutoVisionError(
                    "COCO image 'file_name' escapes the images directory: "
                    f"{image['file_name']!r} (resolved to '{candidate}')."
                )
            image_id_to_file[int(image["id"])] = (
                str(candidate),
                int(image["width"]),
                int(image["height"]),
            )
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in annotations.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
                annotation
            )

        user_classes = self._classes()
        included = _included_class_ids(user_classes, self.ignore_classes)
        internal = contiguous_class_id_map(sorted(included))

        for image_id, (image_path, width, height) in image_id_to_file.items():
            image_annotations = annotations_by_image.get(image_id, [])
            boxes: list[list[float]] = []
            labels: list[int] = []
            iscrowds: list[int] = []
            for annotation in image_annotations:
                class_id = int(annotation["category_id"])
                if class_id not in internal:
                    continue
                x, y, w, h = annotation["bbox"]  # absolute pixels xywh
                boxes.append(
                    [
                        (x + w / 2) / width,
                        (y + h / 2) / height,
                        w / width,
                        h / height,
                    ]
                )
                labels.append(internal[class_id])
                iscrowds.append(int(annotation.get("iscrowd", 0)))
            if not boxes and self.skip_if_annotations_missing and split == "train":
                continue
            yield {
                "image_path": image_path,
                "bboxes": json.dumps(boxes),
                "class_labels": json.dumps(labels),
                # Official-protocol ignore regions (consumed by the mAP
                # evaluator; training treats every box as a regular target).
                "iscrowds": json.dumps(iscrowds),
            }


def _included_class_ids(
    user_classes: dict[int, str], ignore_classes: set[int] | None
) -> set[int]:
    included = set(user_classes)
    if ignore_classes:
        included -= set(ignore_classes)
    return included
