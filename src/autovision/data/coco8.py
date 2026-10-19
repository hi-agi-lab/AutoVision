# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""``coco8``: deterministic synthetic smoke dataset in COCO format.

Eight 64x64 training images and two validation images with fixed rectangle
annotations, generated procedurally at first use (PIL only — no downloads,
no binary assets in the repository, no third-party licensing questions).
It powers the sub-2-minute end-to-end smoke path (train -> val -> predict
-> export); the content is deliberately simple so a few training steps move
the loss visibly, which is all a smoke run needs to demonstrate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["coco8_data_config", "ensure_coco8"]

_IMAGE_SIZE = 64
#: class_id -> (x, y, w, h) in absolute pixels; class ids are COCO 1-based.
_BOXES = {1: (8.0, 8.0, 32.0, 32.0), 2: (24.0, 24.0, 32.0, 32.0)}
_CATEGORIES = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
_TRAIN_IMAGES = 8
_VAL_IMAGES = 2


def _default_root() -> Path:
    cache = os.environ.get(
        "AUTOVISION_CACHE_DIR", str(Path.home() / ".cache" / "autovision")
    )
    return Path(cache) / "datasets" / "coco8"


def _draw_image(path: Path, index: int, category_id: int) -> None:
    from PIL import Image, ImageDraw

    level = 32 + (index * 29) % 160
    image = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), (level, level, level))
    draw = ImageDraw.Draw(image)
    x, y, w, h = _BOXES[category_id]
    color = [(220, 40, 40), (40, 180, 60), (40, 80, 220)][index % 3]
    draw.rectangle([x, y, x + w, y + h], fill=color)
    image.save(path)


def _split_payload(
    images_dir: Path, count: int, offset: int, annotation_start: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for i in range(count):
        index = offset + i
        category_id = 1 + (index % 2)
        file_name = f"img_{index:03d}.png"
        _draw_image(images_dir / file_name, index, category_id)
        images.append(
            {
                "id": index + 1,
                "file_name": file_name,
                "width": _IMAGE_SIZE,
                "height": _IMAGE_SIZE,
            }
        )
        x, y, w, h = _BOXES[category_id]
        annotations.append(
            {
                "id": annotation_start + i,
                "image_id": index + 1,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )
    return images, annotations


def coco8_data_config(root: Path | str) -> dict[str, Any]:
    """Data config dict consumable by ``train_object_detection(data=...)``."""
    root = Path(root)
    return {
        "format": "coco",
        "train": {"annotations": str(root / "train.json"), "images": str(root / "images")},
        "val": {"annotations": str(root / "val.json"), "images": str(root / "images")},
    }


def ensure_coco8(
    out_dir: Path | str | None = None, force: bool = False
) -> dict[str, Any]:
    """Generate the coco8 dataset once and return its data config.

    The result is cached under ``out_dir`` (default
    ``$AUTOVISION_CACHE_DIR/datasets/coco8``) with a marker file; repeated
    calls are no-ops unless ``force=True``. Deterministic: identical bytes
    on every machine for the same autovision version.
    """
    root = Path(out_dir) if out_dir is not None else _default_root()
    marker = root / "coco8.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    train_images, train_annotations = _split_payload(
        images_dir, _TRAIN_IMAGES, 0, annotation_start=1
    )
    # Validation reuses the first two rendered images (ids re-based so the
    # two annotation files stay independently consistent).
    val_images = [
        dict(image, id=i + 1) for i, image in enumerate(train_images[:_VAL_IMAGES])
    ]
    val_annotations = []
    for i, image in enumerate(val_images):
        category_id = 1 + (i % 2)
        x, y, w, h = _BOXES[category_id]
        val_annotations.append(
            {
                "id": i + 1,
                "image_id": image["id"],
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    (root / "train.json").write_text(
        json.dumps(
            {"images": train_images, "annotations": train_annotations, "categories": _CATEGORIES}
        )
    )
    (root / "val.json").write_text(
        json.dumps(
            {"images": val_images, "annotations": val_annotations, "categories": _CATEGORIES}
        )
    )
    config = coco8_data_config(root)
    marker.write_text(json.dumps(config, indent=2))
    return config
