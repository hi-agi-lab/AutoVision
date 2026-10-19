# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Gradient-finiteness and DDP sampler guards."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from autovision.engine.engine import _assert_finite_gradients, _validate_ddp_sampler


def _optimizer_with_grad(value: float):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(value)
    return torch.optim.SGD([parameter], lr=0.1)


def test_non_finite_gradient_raises():
    """A finite loss with a NaN gradient must not reach optimizer.step()."""
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        _assert_finite_gradients(_optimizer_with_grad(float("nan")), None)


def test_infinite_clip_norm_raises():
    """The pre-clip norm from a clipping model is checked as well."""
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        _assert_finite_gradients(_optimizer_with_grad(float("inf")), None)
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        _assert_finite_gradients(_optimizer_with_grad(1.0), float("nan"))


def test_finite_gradient_passes():
    _assert_finite_gradients(_optimizer_with_grad(1.0), None)
    _assert_finite_gradients(_optimizer_with_grad(1.0), 0.5)


def test_ddp_rejects_weighted_sampler():
    """A weighted/hard-example sampler must fail instead of being replaced."""
    dataset = TensorDataset(torch.arange(4))
    sampler = WeightedRandomSampler([1.0, 1.0, 1.0, 1.0], num_samples=4)
    with pytest.raises(ValueError, match="Custom samplers"):
        _validate_ddp_sampler(DataLoader(dataset, sampler=sampler))


def test_ddp_accepts_default_sampler():
    _validate_ddp_sampler(DataLoader(TensorDataset(torch.arange(4)), batch_size=2))
