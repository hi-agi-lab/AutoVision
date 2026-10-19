# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task dataset base class: transform injection and collate selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from torch.utils.data import Dataset

__all__ = ["TaskDataset"]


class TaskDataset(Dataset):
    """Base class for supervised task datasets.

    Rows come from a ``MemoryMappedSequence``; transforms are installed
    *after* dataloader construction (their resolution needs the loader
    length), and each dataset declares its collate function class.
    """

    batch_collate_fn_cls: ClassVar[type | None] = None

    def __init__(
        self,
        dataset_args: Any,
        image_info: Sequence[dict[str, Any]],
        transform: Any = None,
    ) -> None:
        self.dataset_args = dataset_args
        self.image_info = image_info
        self.transform = transform

    def set_transform(self, transform: Any) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_info)
