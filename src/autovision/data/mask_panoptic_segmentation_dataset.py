# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Panoptic segmentation dataset over mask PNGs + a segments json.

Layout: an images directory and a masks directory of single-channel PNGs
whose pixel values are **segment ids** (0 is reserved for unlabeled
pixels by the COCO panoptic convention), plus a segments json::

    {
      "categories": [{"id": 1, "name": "car", "isthing": 1}, ...],
      "annotations": {"<image stem>": [
          {"id": 0, "category_id": 1, "iscrowd": 0}, ...]}
    }

Internal class ids are contiguous: thing classes first (ascending
category id), then stuff classes, matching the task model conventions.
The ignore class id (``num_classes``) is attributed to pixels whose
segment is unknown or unmapped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import open_image

__all__ = [
    "MaskPanopticSegmentationDataset",
    "list_panoptic_pairs",
    "load_panoptic_segments",
    "split_panoptic_classes",
]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_panoptic_segments(path: Path) -> tuple[dict[int, dict], dict[str, list[dict]]]:
    """Load the segments json → (categories, annotations-by-stem)."""
    payload = json.loads(Path(path).read_text())
    categories = {
        int(category["id"]): {
            "name": str(category.get("name", category["id"])),
            "isthing": bool(category.get("isthing", 0)),
        }
        for category in payload.get("categories", [])
    }
    annotations = {
        str(stem): list(segments)
        for stem, segments in payload.get("annotations", {}).items()
    }
    return categories, annotations


def split_panoptic_classes(
    categories: dict[int, dict],
) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    """Category ids → contiguous internal ids: things first, then stuffs.

    Returns ``(class_id_to_internal, thing_classes, stuff_classes)`` with
    the class dicts keyed by internal id.
    """
    thing_ids = sorted(c for c, info in categories.items() if info["isthing"])
    stuff_ids = sorted(c for c, info in categories.items() if not info["isthing"])
    mapping: dict[int, int] = {}
    for internal_id, class_id in enumerate(thing_ids + stuff_ids):
        mapping[class_id] = internal_id
    thing_classes = {mapping[c]: categories[c]["name"] for c in thing_ids}
    stuff_classes = {mapping[c]: categories[c]["name"] for c in stuff_ids}
    return mapping, thing_classes, stuff_classes


def list_panoptic_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files and segment-id mask PNGs by stem."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    mask_by_stem: dict[str, Path] = {}
    for path in sorted(masks_dir.rglob("*")):
        if path.is_file():
            mask_by_stem[path.stem] = path
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise ValueError(
            f"No image/mask pairs found ('{images_dir}' x '{masks_dir}'); "
            f"masks must share the image file stem."
        )
    return pairs


class MaskPanopticSegmentationDataset(Dataset):
    """Yields ``{"image", "masks", "binary_masks"}`` after the joint transform.

    ``masks`` is the (H, W, 2) long map with ``(class_id, segment_id)``
    per pixel (crowd regions carry class -1, unassigned pixels the
    ignore class); ``binary_masks`` mirrors the per-segment view consumed
    by the HF mask criterion: ``{"masks": (N, H, W) float, "labels": (N,),
    "iscrowd": (N,)}``.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        categories: dict[int, dict],
        segments_by_stem: dict[str, list[dict]],
        transform: Any = None,
        num_channels: int = 3,
        drop_crowd: bool = False,
    ) -> None:
        self.pairs = [(Path(image), Path(mask)) for image, mask in pairs]
        self.categories = dict(categories)
        self.segments_by_stem = segments_by_stem
        self.transform = transform
        self.num_channels = num_channels
        self.drop_crowd = drop_crowd
        (
            self._class_id_to_internal,
            self.thing_classes,
            self.stuff_classes,
        ) = split_panoptic_classes(categories)
        self.num_classes = len(self._class_id_to_internal)
        self.internal_ignore_class_id = self.num_classes

    def _get_binary_masks(
        self, segments: torch.Tensor, stem: str
    ) -> dict[str, torch.Tensor]:
        height, width = segments.shape
        segment_info = {
            int(segment["id"]): segment
            for segment in self.segments_by_stem.get(stem, [])
        }
        masks: list[torch.Tensor] = []
        labels: list[int] = []
        iscrowds: list[bool] = []
        for segment_id in segments.unique().tolist():
            segment = segment_info.get(int(segment_id))
            if segment is None:  # Unknown segment id (e.g. unlabeled 0).
                continue
            internal_id = self._class_id_to_internal.get(int(segment["category_id"]))
            if internal_id is None:  # Ignored class.
                continue
            is_crowd = bool(segment.get("iscrowd", 0))
            if self.drop_crowd and is_crowd:
                continue
            masks.append(segments == segment_id)
            labels.append(internal_id)
            iscrowds.append(is_crowd)
        return {
            "masks": (
                torch.stack(masks).float()
                if masks
                else torch.zeros(0, height, width)
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "iscrowd": torch.tensor(iscrowds, dtype=torch.bool),
        }

    @staticmethod
    def _get_masks(
        binary_masks: dict[str, torch.Tensor], ignore_class_id: int
    ) -> torch.Tensor:
        per_segment = binary_masks["masks"].bool()
        num_segments, height, width = per_segment.shape
        masks = torch.full((height, width, 2), -1, dtype=torch.long)
        masks[..., 0] = ignore_class_id
        labels = binary_masks["labels"].clone()
        labels[binary_masks["iscrowd"]] = -1  # crowd regions ignored in metrics
        for segment_id in range(num_segments):
            region = per_segment[segment_id]
            masks[..., 0] = torch.where(region, labels[segment_id], masks[..., 0])
            masks[..., 1] = torch.where(region, segment_id, masks[..., 1])
        return masks

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        from autovision.data.semantic_segmentation_dataset import _read_mask

        image = open_image(image_path, num_channels=self.num_channels)
        segments = _read_mask(mask_path)
        if self.transform is not None:
            # Retry so the random crop keeps at least one valid segment.
            for _ in range(20):
                image_t, segments_t = self.transform(image, segments)
                binary_masks = self._get_binary_masks(segments_t, image_path.stem)
                if binary_masks["labels"].numel() > 0:
                    break
        else:
            image_t, segments_t = image, segments
            binary_masks = self._get_binary_masks(segments_t, image_path.stem)
        masks = self._get_masks(binary_masks, self.internal_ignore_class_id)
        return {
            "image": image_t,
            "masks": masks,
            "binary_masks": binary_masks,
            "filename": image_path.name,
        }
