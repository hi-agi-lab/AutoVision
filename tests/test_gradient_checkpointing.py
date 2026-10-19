# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Activation checkpointing helper: custom blocks and the HF native path."""

from __future__ import annotations

import torch
from torch import nn

from autovision.tasks.shared.gradient_checkpointing import (
    enable_gradient_checkpointing,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x).relu()


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_custom_blocks_are_checkpointed_and_trainable(monkeypatch):
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def counting(*args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting)
    model = _Stack()
    assert enable_gradient_checkpointing(model) == 3
    # Idempotent: a second call patches nothing.
    assert enable_gradient_checkpointing(model) == 0

    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    assert len(calls) == 3
    assert all(use_reentrant is False for use_reentrant in calls)
    assert model.blocks[0].linear.weight.grad is not None


def test_hf_native_api_takes_precedence():
    calls = []

    class _HfLike(nn.Module):
        def gradient_checkpointing_enable(self):
            calls.append("enabled")

    model = _HfLike()
    assert enable_gradient_checkpointing(model) == 1
    assert calls == ["enabled"]
