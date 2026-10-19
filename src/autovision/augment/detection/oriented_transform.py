# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented (rotated-box) detection transform + collate.

Mirrors the RT-DETR detection transform contract for 8-component corner
polygons (normalized xyxyxyxy in, normalized xyxyxyxy out). Geometry ops
are implemented explicitly because ``tv_tensors.BoundingBoxes`` cannot
represent rotated boxes:

- horizontal/vertical flips mirror every corner independently — a reflected
  ordered polygon stays an ordered polygon (the traversal merely reverses
  winding), so no re-ordering is required;
- resize scales image and corners with per-axis factors;
- photometric distortion reuses the torchvision v2 op on the image only;
- sanitization drops non-finite or sub-threshold-area polygons (shoelace
  area in px²) — the oriented analogue of ``SanitizeBoundingBoxes``.

IoU-crop / zoom-out augmentations are intentionally omitted: the
torchvision v2 implementations are axis-aligned-box-only.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torchvision.transforms import v2

from autovision.augment.detection.args import (
    OrientedObjectDetectionTransformArgs,
)

__all__ = [
    "OrientedObjectDetectionCollateFunction",
    "OrientedObjectDetectionTransform",
    "build_oriented_val_transform_args",
]


def _polygon_area_px(polys: torch.Tensor) -> torch.Tensor:
    """Shoelace area (px²) of (..., 4, 2) absolute-pixel polygons."""
    nxt = torch.roll(polys, -1, dims=-2)
    cross = polys[..., 0] * nxt[..., 1] - polys[..., 1] * nxt[..., 0]
    return (cross.sum(dim=-1) / 2.0).abs()


class OrientedObjectDetectionTransform:
    """Per-sample oriented detection transform (train/val via args)."""

    def __init__(self, transform_args: OrientedObjectDetectionTransformArgs) -> None:
        self.transform_args = transform_args
        self._photometric = (
            v2.RandomPhotometricDistort(p=transform_args.photometric_distort_prob)
            if transform_args.photometric_distort_prob
            else None
        )
        mean, std = transform_args.normalize.to_tensor(torch.device("cpu"))
        self._mean = mean
        self._std = std

    def set_step(self, step: int) -> None:  # contract parity with RT-DETR transform
        return None

    def requires_dataloader_reinitialization(self) -> bool:
        return False

    def mark_dataloader_as_reinitialized(self) -> None:
        return None

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        args = self.transform_args
        image: torch.Tensor = item["image"]  # (C, H, W), uint8 or float
        polys: torch.Tensor = item["bboxes"]  # (N, 8) absolute px
        classes: torch.Tensor = item["classes"]
        _, height, width = image.shape

        if self._photometric is not None:
            image = self._photometric(image.float() / 255.0)
        else:
            image = image.float() / 255.0

        if (
            args.horizontal_flip_prob > 0
            and random.random() < args.horizontal_flip_prob
        ):
            image = torch.flip(image, dims=[-1])
            polys = polys.clone()
            polys[:, 0::2] = width - polys[:, 0::2]
        if args.vertical_flip_prob > 0 and random.random() < args.vertical_flip_prob:
            image = torch.flip(image, dims=[-2])
            polys = polys.clone()
            polys[:, 1::2] = height - polys[:, 1::2]

        new_width, new_height = args.image_size
        if (new_width, new_height) != (width, height):
            image = torch.nn.functional.interpolate(
                image[None],
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            polys = polys.clone()
            polys[:, 0::2] *= new_width / width
            polys[:, 1::2] *= new_height / height

        # Sanitize: finite coordinates + minimum polygon area (px²).
        if polys.numel() > 0:
            areas = _polygon_area_px(polys.reshape(-1, 4, 2))
            keep = torch.isfinite(polys).all(dim=-1) & (areas >= args.min_bbox_size_px)
            polys = polys[keep]
            classes = classes[keep]
        polys = polys.reshape(-1, 8)
        polys[:, 0::2] /= new_width
        polys[:, 1::2] /= new_height

        image = (image - self._mean) / self._std
        return {
            "image_path": item["image_path"],
            "image": image,
            "bboxes": polys,
            "classes": classes,
            "original_size": item["original_size"],
        }


def build_oriented_val_transform_args(
    train_args: OrientedObjectDetectionTransformArgs,
) -> OrientedObjectDetectionTransformArgs:
    """Disable all stochastic augmentations for the val transform.

    The polygon-area filter is relaxed to 1 px²: at eval time dropping
    sub-threshold objects with the *training* threshold would silently
    delete tiny GT (dense aerial datasets) and distort the mAP.
    """
    val_args = train_args.model_copy(deep=True)
    val_args.photometric_distort_prob = None
    val_args.horizontal_flip_prob = 0.0
    val_args.vertical_flip_prob = 0.0
    val_args.min_bbox_size_px = 1.0
    return val_args


class OrientedObjectDetectionCollateFunction:
    """Collates oriented transform outputs into a batch.

    Output: ``{"image": (B,C,H,W) float normalized, "bboxes": list[(n,8)]
    normalized xyxyxyxy, "classes": list[(n,)], "image_path": list[str],
    "original_size": list[(w,h)]}``.
    """

    def __init__(
        self, transform_args: OrientedObjectDetectionTransformArgs, train: bool = True
    ) -> None:
        self.transform_args = transform_args
        self.train = train

    def set_step(self, step: int) -> None:
        return None

    def requires_dataloader_reinitialization(self) -> bool:
        return False

    def mark_dataloader_as_reinitialized(self) -> None:
        return None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        args = self.transform_args
        images: list[torch.Tensor] = []
        boxes_list: list[torch.Tensor] = []
        classes_list: list[torch.Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        input_size: tuple[int, int] | None = None

        for item in batch:
            image: torch.Tensor = item["image"]
            polys: torch.Tensor = item["bboxes"]
            classes: torch.Tensor = item["classes"]
            images.append(image)
            boxes_list.append(polys)
            classes_list.append(classes)
            original_sizes.append(tuple(item["original_size"]))

        input_size = (images[0].shape[-1], images[0].shape[-2])

        # Degenerate-polygon filtering on the normalized polygons.
        filtered = []
        for polys, classes in zip(boxes_list, classes_list):
            if polys.numel() > 0:
                quads = torch.stack(
                    [polys[:, 0::2], polys[:, 1::2]], dim=-1
                ) * torch.tensor([input_size[0], input_size[1]], device=polys.device)
                areas = _polygon_area_px(quads)
                keep = areas >= args.min_bbox_size_px
                polys, classes = polys[keep], classes[keep]
            filtered.append((polys.reshape(-1, 8), classes))
        boxes_list = [b for b, _ in filtered]
        classes_list = [c for _, c in filtered]

        return {
            "image": torch.stack(images),
            "bboxes": boxes_list,
            "classes": classes_list,
            "image_path": [item["image_path"] for item in batch],
            "original_size": original_sizes,
            "input_size": input_size,
        }
