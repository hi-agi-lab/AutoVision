# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Regression tests: detection criteria must return graph-attached losses.

DDP averages gradients across ranks, so the criterion's training loss must
stay connected to the autograd graph. Reducing it with
``runtime.distributed.reduce_dict`` (which detaches) makes
``loss.backward()`` fail under ``world_size >= 2``.
"""

from __future__ import annotations

import torch

from autovision.tasks.detection.shared import (
    HungarianMatcher,
    RTDETRCriterionV2,
)
from autovision.tasks.detection.shared.dfine_criterion import DFINECriterionV2


def _fake_initialized_distributed(monkeypatch, world_size: int = 2) -> None:
    """Pretend a process group is initialized without spawning processes."""

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    # The criteria only all-reduce scalar bookkeeping and (before this fix)
    # the loss dictionary; no communication is needed for the unit test.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda *args, **kwargs: None
    )


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "labels": torch.tensor([0]),
        }
    ]


def _outputs() -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        {
            "pred_logits": torch.randn(1, 8, 3, requires_grad=True),
            "pred_boxes": torch.rand(1, 8, 4, requires_grad=True),
        }
    ]


def test_rtdetr_criterion_loss_is_graph_attached_under_ddp(monkeypatch):
    _fake_initialized_distributed(monkeypatch)
    criterion = RTDETRCriterionV2(HungarianMatcher())
    outputs = _outputs()

    losses = criterion(outputs, _targets(), world_size=2)

    loss = losses["loss"]
    assert loss.requires_grad, "criterion returned a detached training loss"
    assert loss.grad_fn is not None
    loss.backward()
    assert outputs[0]["pred_logits"].grad is not None
    assert outputs[0]["pred_boxes"].grad is not None


def test_dfine_criterion_loss_is_graph_attached_under_ddp(monkeypatch):
    _fake_initialized_distributed(monkeypatch)
    criterion = DFINECriterionV2(HungarianMatcher())
    outputs = _outputs()

    losses = criterion(outputs, _targets(), world_size=2)

    loss = losses["loss"]
    assert loss.requires_grad, "criterion returned a detached training loss"
    assert loss.grad_fn is not None
    loss.backward()
    assert outputs[0]["pred_logits"].grad is not None
    assert outputs[0]["pred_boxes"].grad is not None
