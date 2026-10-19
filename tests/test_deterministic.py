# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Determinism is an opt-in environment switch, not a default side effect."""

import os

import torch

from autovision.engine.engine import seed_everything


def _restore(old_deterministic: bool, old_cudnn: bool, old_benchmark: bool) -> None:
    torch.use_deterministic_algorithms(old_deterministic)
    torch.backends.cudnn.deterministic = old_cudnn
    torch.backends.cudnn.benchmark = old_benchmark


def _restore_cublas(old: str | None) -> None:
    if old is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = old


def test_deterministic_env_enables_torch_flags(monkeypatch):
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setenv("AUTOVISION_DETERMINISTIC", "1")
    try:
        seed_everything(123)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        _restore(*old)
        _restore_cublas(old_cublas)


def test_default_does_not_turn_determinism_on(monkeypatch):
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.delenv("AUTOVISION_DETERMINISTIC", raising=False)
    try:
        seed_everything(7)
        assert (
            torch.are_deterministic_algorithms_enabled()
            == old[0]
        )
        assert torch.backends.cudnn.deterministic == old[1]
        assert torch.backends.cudnn.benchmark == old[2]
    finally:
        _restore(*old)
