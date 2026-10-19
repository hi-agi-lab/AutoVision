# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Data helpers: image filename discovery and decoding."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image

__all__ = ["list_image_filenames_from_dir", "open_image", "open_mask"]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def list_image_filenames_from_dir(image_dir: Path) -> Iterator[Path]:
    """Recursively list image filenames under ``image_dir`` (sorted, stable)."""
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: '{image_dir}'")
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def open_image(filename: Path | str, num_channels: int = 3) -> torch.Tensor:
    """Decode an image to a uint8 (C, H, W) tensor.

    Fast path: ``torchvision.io.read_image``. Fallback: PIL for exotic
    formats; EXIF orientation is normalized so boxes and pixels agree, and
    truncated files raise instead of silently training on partial data.
    """
    mode = ImageReadMode.RGB if num_channels == 3 else ImageReadMode.UNCHANGED
    # torchvision's fast decoder does not normalize EXIF rotation. Route
    # oriented JPEGs through PIL so boxes and pixels share one frame.
    if Path(filename).suffix.lower() in {".jpg", ".jpeg"}:
        try:
            from PIL import Image

            with Image.open(filename) as image:
                if image.getexif().get(274, 1) not in (None, 1):
                    return _open_image_pil(filename, num_channels)
        except Exception:  # noqa: BLE001 - the normal decoder reports it
            pass
    try:
        return read_image(str(filename), mode=mode)
    except RuntimeError:
        return _open_image_pil(filename, num_channels)


def _open_image_pil(filename: Path | str, num_channels: int) -> torch.Tensor:
    from PIL import Image, ImageFile, ImageOps
    from torchvision.transforms import functional as TF

    # Truncated images must fail loudly: silently decoding a partial image
    # trains on pixels that no longer match its annotations.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(filename) as image:
            image = ImageOps.exif_transpose(image)
            if num_channels == 3:
                image = image.convert("RGB")
            elif num_channels == 1:
                image = image.convert("L")
            return TF.pil_to_tensor(image)
    except Exception as error:
        raise RuntimeError(
            f"Failed to open image '{filename}': {error}"
        ) from error


def open_mask(filename: Path | str) -> torch.Tensor:
    """Decode a segmentation mask to an integer (H, W) tensor.

    Masks store region ids (e.g. panoptic ids), so the pixel values must be
    preserved exactly: read without color conversion and never scale.
    Single-channel masks are squeezed from (1, H, W) to (H, W).
    """
    try:
        mask = read_image(str(filename), mode=ImageReadMode.UNCHANGED)
    except RuntimeError:
        mask = _open_mask_pil(filename)
    if mask.ndim == 2:
        return mask
    if mask.shape[0] == 1:
        return mask[0]
    raise ValueError(
        f"Mask '{filename}' has {mask.shape[0]} channels; single-channel "
        "masks (grayscale/palette PNG) are required."
    )


def _open_mask_pil(filename: Path | str) -> torch.Tensor:
    from PIL import Image, ImageFile
    from torchvision.transforms import functional as TF

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(filename) as image:
            return TF.pil_to_tensor(image)
    except Exception as error:
        raise RuntimeError(
            f"Failed to open mask '{filename}': {error}"
        ) from error
