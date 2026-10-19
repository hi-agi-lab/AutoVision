# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation dataset over paired image/mask directories.

Layout: an images directory and a masks directory containing files with
matching stems (``abc.jpg`` <-> ``abc.png``). Masks are single-channel
class-index images; label 255 marks ignored pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from autovision.data.file_helpers import open_image

__all__ = ["SemanticSegmentationDataset", "list_segmentation_pairs"]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_mask(path: Path) -> torch.Tensor:
    from torchvision.io import read_image
    from torchvision.io.image import ImageReadMode

    mask = read_image(str(path), mode=ImageReadMode.UNCHANGED)
    if mask.ndim == 3:
        # RGB(A) palette mask: any color-collapsed mask works as long as all
        # channels agree; otherwise the R channel is the class index (the
        # convention for 8-bit grayscale masks saved as PNG).
        mask = mask[0]
    return mask.long()


def list_segmentation_pairs(
    images_dir: Path, masks_dir: Path, ignore_index: int = 255
) -> tuple[list[tuple[Path, Path]], int]:
    """Match image/mask files by stem.

    Returns ``(pairs, max_label)`` where ``max_label`` is the highest class
    index found in any sampled mask (0 for an empty list). Pixels labeled
    ``ignore_index`` (255, the ignore convention) do not count towards the
    label range.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    mask_by_stem: dict[str, Path] = {}
    for path in sorted(masks_dir.rglob("*")):
        if path.is_file():
            mask_by_stem[path.stem] = path
    pairs: list[tuple[Path, Path]] = []
    max_label = 0
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        pairs.append((image_path, mask_path))
        mask = _read_mask(mask_path)
        valid = mask[mask != ignore_index]
        if valid.numel():
            max_label = max(max_label, int(valid.max().item()))
    if not pairs:
        raise ValueError(
            f"No image/mask pairs found ('{images_dir}' x '{masks_dir}'); "
            f"masks must share the image file stem."
        )
    return pairs, max_label + 1


class SemanticSegmentationDataset(Dataset):
    """Yields ``{"image", "mask"}`` after the joint transform."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        transform: Any = None,
        num_channels: int = 3,
    ) -> None:
        self.pairs = [(Path(image), Path(mask)) for image, mask in pairs]
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        image = open_image(image_path, num_channels=self.num_channels)
        mask = _read_mask(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name,
        }
