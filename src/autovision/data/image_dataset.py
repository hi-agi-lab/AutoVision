# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining dataset over plain images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from autovision.augment.method_transform import MethodTransform
from autovision.data.file_helpers import open_image, open_mask
from autovision.types import DatasetItem

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Multi-view pretraining dataset.

    ``image_filenames`` is any sequence of paths (a plain list, or a
    ``MemoryMappedSequence`` for zero-copy sharing across workers).

    When ``mask_dir`` is set, a region-id mask is loaded for every image
    (``<image stem>.png`` inside ``mask_dir``) and passed through the
    transform, which must emit a ``mask`` entry per view (e.g. the DetCon
    transforms). Masks enable region-contrastive pretraining objectives.
    """

    def __init__(
        self,
        image_dir: Path | None,
        image_filenames: Sequence[str | Path],
        transform: MethodTransform,
        num_channels: int = 3,
        mask_dir: Path | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.transform = transform
        self.num_channels = num_channels
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> DatasetItem:
        filename = self.image_filenames[index]
        filename_str = (
            filename["filenames"] if isinstance(filename, dict) else str(filename)
        )
        path = Path(filename_str)
        if not path.is_absolute() and self.image_dir is not None:
            path = self.image_dir / path
        image = open_image(path, num_channels=self.num_channels)
        transform_input: dict = {"image": image}
        if self.mask_dir is not None:
            transform_input["mask"] = open_mask(
                (self.mask_dir / path.name).with_suffix(".png")
            )
        transformed: list[dict[str, Any]] = self.transform(transform_input)
        item: DatasetItem = {
            "filename": path.name,
            "views": [view["image"] for view in transformed],
        }
        if transformed and all("mask" in view for view in transformed):
            item["masks"] = [view["mask"] for view in transformed]
        if transformed and all("geometry" in view for view in transformed):
            item["geometries"] = [view["geometry"] for view in transformed]
        return item
