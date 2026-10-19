# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""JSONL metrics logger: one JSON object per line in ``<out>/metrics.jsonl``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["JSONLLogger"]


class JSONLLogger:
    """Minimal file logger for metrics and hyperparameters."""

    def __init__(self, save_dir: Path | str, flush_logs_every_n_steps: int = 100) -> None:
        self.save_dir = Path(save_dir)
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.save_dir / "metrics.jsonl"
        self._hyperparams_logged = False

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        record = {"ts": time.time(), "step": step, **metrics}
        with self.metrics_file.open("a") as file:
            file.write(json.dumps(record, default=str) + "\n")

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self._hyperparams_logged:
            return
        self._hyperparams_logged = True
        record = {"ts": time.time(), "step": -1, "hyperparams": params}
        with self.metrics_file.open("a") as file:
            file.write(json.dumps(record, default=str) + "\n")
