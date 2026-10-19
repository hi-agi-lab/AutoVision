# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection dataset over mmap rows + task transform."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import torch
from torchvision import tv_tensors

from autovision.data.file_helpers import open_image
from autovision.data.task_dataset import TaskDataset
from autovision.types import PathLike

__all__ = ["ObjectDetectionDataset"]


class ObjectDetectionDataset(TaskDataset):
    """Yields detection items; boxes are normalized cxcywh on entry and
    converted to absolute-pixel XYXY ``tv_tensors.BoundingBoxes`` for the
    geometric transform pipeline."""

    def __init__(
        self,
        dataset_args: Any,
        image_info: Sequence[dict[str, Any]],
        transform: Any = None,
        num_channels: int = 3,
    ) -> None:
        super().__init__(dataset_args, image_info, transform)
        self.num_channels = num_channels

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.image_info[index]
        if isinstance(row, dict) and "image_path" in row:
            image_path: PathLike = row["image_path"]
            bboxes = json.loads(row["bboxes"])
            class_labels = json.loads(row["class_labels"])
        else:
            raise TypeError(f"Unsupported image info row: {type(row).__name__}")
        image = open_image(image_path, num_channels=self.num_channels)
        height, width = image.shape[-2:]
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(class_labels, dtype=torch.int64)
        if boxes.numel() > 0:
            # normalized cxcywh -> absolute xyxy for torchvision transforms.
            cx, cy, w, h = boxes.unbind(-1)
            boxes_xyxy = torch.stack(
                [(cx - w / 2) * width, (cy - h / 2) * height,
                 (cx + w / 2) * width, (cy + h / 2) * height],
                dim=-1,
            )
            boxes_tv = tv_tensors.BoundingBoxes(
                boxes_xyxy, format="XYXY", canvas_size=(height, width)
            )
        else:
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros(0, 4), format="XYXY", canvas_size=(height, width)
            )
        item = {
            "image_path": str(image_path),
            "image": image,
            "bboxes": boxes_tv,
            "classes": labels,
            "original_size": (width, height),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
