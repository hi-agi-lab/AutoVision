# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO instance segmentation dataset (in-memory, eval-oriented).

Reads a COCO json (annotations carry ``segmentation`` polygons or RLE),
converts each instance to a binary mask via pycocotools, and yields
normalized boxes + per-instance masks. The internal class ids are
contiguous (insertion order of the train categories), matching the
detection dataset conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["COCOInstanceSegmentationDataset"]


class COCOInstanceSegmentationDataset(Dataset):
    """COCO instance dataset: (image, bboxes, classes, masks) per row.

    ``masks`` is a (K, H, W) float tensor with one binary mask per
    instance; ``bboxes`` is (K, 4) normalized cxcywh on the original frame.
    """

    def __init__(
        self,
        annotations: Path,
        images_dir: Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        from torchvision.transforms.v2 import functional as TF

        self._tf = TF
        with open(annotations) as file:
            payload = json.load(file)
        images_dir = images_dir or Path(annotations).parent
        categories = sorted(int(c["id"]) for c in payload.get("categories", []))
        self._user_to_internal = {c: i for i, c in enumerate(categories)}
        self._images_dir = images_dir
        self.transform = transform

        image_meta = {
            int(image["id"]): (image["file_name"], int(image["width"]), int(image["height"]))
            for image in payload.get("images", [])
        }
        rows: list[dict[str, Any]] = []
        for image_id, (file_name, width, height) in image_meta.items():
            boxes: list[list[float]] = []
            classes: list[int] = []
            masks: list[np.ndarray] = []
            for annotation in payload.get("annotations", []):
                if int(annotation["image_id"]) != image_id:
                    continue
                if int(annotation["category_id"]) not in self._user_to_internal:
                    continue
                boxes.append([float(v) for v in annotation["bbox"]])
                classes.append(self._user_to_internal[int(annotation["category_id"])])
                masks.append(self._decode_mask(annotation["segmentation"], height, width))
            rows.append(
                {
                    "image_path": str(images_dir / file_name),
                    "size": (width, height),
                    "bboxes": boxes,
                    "classes": classes,
                    "masks": masks,
                }
            )
        self._rows = rows

    @staticmethod
    def _decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
        from pycocotools import mask as mask_utils

        if isinstance(segmentation, dict):  # RLE
            rle = segmentation
            if isinstance(rle.get("counts"), str):
                rle = mask_utils.frPyObjects(rle, height, width)
            return mask_utils.decode(rle).astype(bool)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from autovision.data.file_helpers import open_image

        row = self._rows[index]
        width, height = row["size"]
        image = open_image(row["image_path"]).float() / 255.0  # (3, H, W)
        if image.shape[-2:] != (height, width):
            image = self._tf.resize(image, [height, width], antialias=True)
        masks = (
            torch.as_tensor(np.stack(row["masks"]), dtype=torch.float)
            if row["masks"]
            else torch.zeros(0, height, width)
        )
        boxes = torch.as_tensor(row["bboxes"], dtype=torch.float).reshape(-1, 4)
        boxes = torch.stack(
            [
                (boxes[:, 0] + boxes[:, 2] / 2) / width,
                (boxes[:, 1] + boxes[:, 3] / 2) / height,
                boxes[:, 2] / width,
                boxes[:, 3] / height,
            ],
            dim=-1,
        )
        classes = torch.as_tensor(row["classes"], dtype=torch.long)
        if self.transform is not None:
            image, masks, boxes, classes = self.transform(image, masks, boxes, classes)
        return {
            "image": image,
            "masks": masks,
            "bboxes": boxes,
            "classes": classes,
            "original_size": (int(height), int(width)),
        }
