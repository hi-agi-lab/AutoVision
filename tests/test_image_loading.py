# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image decoding: EXIF orientation and corrupt-file behavior."""

import pytest
from PIL import Image

from autovision.data.file_helpers import open_image


def test_exif_orientation_is_normalized(tmp_path):
    """An EXIF-rotated JPEG is decoded in the annotation's pixel frame."""
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise
    image.save(path, exif=exif)

    tensor = open_image(path)
    # Orientation 6 swaps width/height: 40x20 -> 20x40 (C, H, W).
    assert tensor.shape == (3, 40, 20)


def test_corrupt_image_fails_loudly(tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="Failed to open image"):
        open_image(path)
