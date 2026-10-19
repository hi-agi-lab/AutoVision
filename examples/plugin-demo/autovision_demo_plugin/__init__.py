# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Demo AutoVision plugin.

Install this package (``pip install -e examples/plugin-demo``) and run
``autovision-cli plugins action=list``; the entry point below is discovered
from the ``autovision.plugins`` group. Real plugins should register their
own uniquely named kinds/factories; this package only demonstrates the
contract with a metric and a dataset factory.
"""

from __future__ import annotations

from typing import Any

from autovision.plugins import PluginInfo


class DemoMetric:
    """Minimal top-k accuracy metric (no framework dependency)."""

    def __init__(self, k: int = 1) -> None:
        self.k = k
        self.total = 0
        self.correct = 0

    def update(self, logits: Any, targets: Any) -> None:
        import torch

        scores = torch.as_tensor(logits)
        labels = torch.as_tensor(targets).reshape(-1)
        top = scores.topk(min(self.k, scores.shape[-1]), dim=-1).indices
        self.total += int(labels.numel())
        self.correct += int((top == labels.unsqueeze(-1)).any(dim=-1).sum())

    def compute(self) -> dict[str, float]:
        return {
            f"top{self.k}": (
                self.correct / self.total if self.total else 0.0
            )
        }


class DemoDataset:
    """Tiny map-style dataset factory (replace with the real loader)."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def plugin() -> list[PluginInfo]:
    return [
        PluginInfo(
            name="autovision-demo-metric",
            version="0.1.0",
            kind="metric",
            factory=DemoMetric,
            description="Demo top-k metric plugin.",
        ),
        PluginInfo(
            name="autovision-demo-dataset",
            version="0.1.0",
            kind="dataset",
            factory=DemoDataset,
            description="Demo dataset factory plugin.",
        ),
    ]
