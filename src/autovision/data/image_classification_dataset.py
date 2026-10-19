# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification dataset over an ImageFolder-style layout.

Layout: one subdirectory per class under the data root; the subdirectory
name is the class label (the same convention the kNN probe uses, so a
pretraining dataset doubles as a classification dataset).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from autovision.data.file_helpers import open_image

__all__ = [
    "ImageClassificationDataset",
    "list_classification_files",
]


def list_classification_files(
    root: Path,
) -> tuple[list[str], list[str], dict[int, str]]:
    """Scan an ImageFolder-style root.

    Returns ``(filenames, labels, classes)`` where ``classes`` maps internal
    indices to class names in sorted order.
    """
    root = Path(root)
    class_names = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if len(class_names) < 2:
        raise ValueError(
            f"The classification dataset at '{root}' needs at least two class "
            f"subdirectories, found {class_names!r}."
        )
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    filenames: list[str] = []
    labels: list[str] = []
    for class_name in class_names:
        class_dir = root / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                filenames.append(str(path))
                labels.append(class_name)
    if not filenames:
        raise ValueError(f"The classification dataset at '{root}' has no images.")
    classes = {index: name for index, name in enumerate(class_names)}
    return filenames, labels, classes


class ImageClassificationDataset(Dataset):
    """Single-view classification dataset yielding ``{"image", "label"}``."""

    def __init__(
        self,
        filenames: Sequence[str],
        labels: Sequence[str],
        classes: dict[int, str],
        transform: Any = None,
        num_channels: int = 3,
    ) -> None:
        self.filenames = list(filenames)
        self.labels = list(labels)
        self.classes = dict(classes)
        self.class_to_index = {
            name: index for index, name in self.classes.items()
        }
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = open_image(Path(self.filenames[index]), num_channels=self.num_channels)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": self.class_to_index[self.labels[index]],
            "filename": Path(self.filenames[index]).name,
        }
