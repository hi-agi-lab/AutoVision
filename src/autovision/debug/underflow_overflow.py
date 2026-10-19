# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Per-module activation overflow tracing (HF DebugUnderflowOverflow style).

Forward hooks record abs-min/max statistics per module; on the first inf/nan
activation the accumulated frames are dumped as JSON and the run optionally
aborts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["DebugUnderflowOverflow"]


class DebugUnderflowOverflow:
    def __init__(
        self,
        model: nn.Module,
        out_dir: Path | str,
        max_frames_to_save: int = 21,
        trace_batch_nums: list[int] | None = None,
        abort_after_batch_num: int | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames_to_save = max_frames_to_save
        self.trace_batch_nums = set(trace_batch_nums or [])
        self.abort_after_batch_num = abort_after_batch_num
        self.batch_number = -1
        self.frames: list[dict[str, Any]] = []
        self.matched_frames: list[dict[str, Any]] = []
        self.aborted = False
        self._hooks = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in model.named_modules()
        ]

    def _make_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            tensors = []
            if torch.is_tensor(output):
                tensors = [output]
            elif isinstance(output, (list, tuple)):
                tensors = [o for o in output if torch.is_tensor(o)]
            stats = {}
            for index, tensor in enumerate(tensors):
                if not tensor.is_floating_point():
                    continue
                detached = tensor.detach()
                stats[f"absmax_{index}"] = float(detached.abs().max())
                stats[f"absmin_{index}"] = float(detached.abs().min())
                if not torch.isfinite(detached).all():
                    stats[f"anomaly_{index}"] = True
            if stats:
                self.frames.append(
                    {"batch": self.batch_number, "module": module_name, **stats}
                )
                # Unbounded growth guard: trim to the anomaly window.
                if len(self.frames) > self.max_frames_to_save * 10:
                    del self.frames[: self.max_frames_to_save]
            if any(k.startswith("anomaly") for k in stats):
                self.matched_frames.append(self.frames[-1])

        return hook

    def set_batch_number(self, batch_number: int) -> None:
        self.batch_number = batch_number
        if self.trace_batch_nums and batch_number in self.trace_batch_nums:
            self.dump_frames(suffix=f"_batch{batch_number}")
        if (
            self.abort_after_batch_num is not None
            and batch_number >= self.abort_after_batch_num
        ):
            self.dump_frames(suffix="_abort")
            self.remove_hooks()
            self.aborted = True

    def dump_frames(self, suffix: str = "") -> Path:
        path = self.out_dir / f"underflow_overflow{suffix}.json"
        payload = {
            "batch_number": self.batch_number,
            "frames": self.matched_frames or self.frames[-self.max_frames_to_save:],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
