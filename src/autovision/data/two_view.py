# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Two-view dataset wrapper for the SSL ``two_crop`` objective (doc 14 §4.5)."""

from __future__ import annotations

from typing import Any, Tuple

from torch.utils.data import Dataset

__all__ = ["TwoViewObjectDetectionDataset"]


class TwoViewObjectDetectionDataset(Dataset[Tuple[Any, Any]]):
    """Yield ``(view1_item, view2_item)`` per source item.

    The wrapped dataset is indexed twice per item, so its stochastic
    per-call transform pipeline produces two genuinely different views of
    the same image (boxes follow each view's geometric transform through
    the wrapped dataset's own pipeline). The length matches the wrapped
    dataset — one item is one image pair — so loader length, step schedule
    and sampler epoch accounting are unchanged.

    The matching collate flattens the pairs into a view-major doubled batch
    (rows ``r`` and ``r + B`` are the same source image); the engine's
    gradient accumulation pulls whole loader batches, so a pair is never
    split across accumulation windows.
    """

    def __init__(self, dataset: Dataset[Any]) -> None:
        """Wrap a dataset whose items are transform-ready dicts."""
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.dataset[index], self.dataset[index]

    def set_transform(self, transform: Any) -> None:
        """Delegate: transforms are installed on the wrapped dataset."""
        self.dataset.set_transform(transform)

    @property
    def transform(self) -> Any:
        return self.dataset.transform
