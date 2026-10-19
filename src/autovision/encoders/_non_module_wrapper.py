# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Base class for wrapping models that are not ``torch.nn.Module``\ s.

Some providers (ultralytics ``YOLO``, ...) expose model
objects that hold weights but cannot be registered as submodules. This base
class stores the object outside the module registry and delegates the
``nn.Module`` plumbing (parameters, state dict save/load hooks, device
moves, train/eval) to the inner ``nn.Module`` exposed by :attr:`inner_model`.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from autovision.encoders.model_wrapper import (
    FeatureMapsOut,
    PoolOut,
)

__all__ = ["NonModuleModelWrapper"]


class NonModuleModelWrapper(nn.Module):
    """Adapts a non-Module model object to the ``EncoderProtocol`` protocol."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        # Plain list attribute: a non-Module object must NOT be registered.
        self._model: list[Any] = [model]

    @property
    def inner_model(self) -> nn.Module:
        """The ``nn.Module`` that actually holds the weights."""
        raise NotImplementedError

    def raw_module(self) -> Any:
        return self._model[0]

    # ── nn.Module plumbing delegated to the inner model ────────────────
    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        inner_apply = self.inner_model._apply
        # Some providers override _apply with older signatures (e.g.
        # ultralytics BaseModel has no "recurse" parameter).
        if "recurse" in inspect.signature(inner_apply).parameters:
            inner_apply(fn, recurse=recurse)
        else:
            inner_apply(fn)
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True) -> "NonModuleModelWrapper":
        super().train(mode)
        self.inner_model.train(mode)
        return self

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.inner_model.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        return self.inner_model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        return self.inner_model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        return self.inner_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def _save_to_state_dict(self, destination, prefix, keep_vars):  # type: ignore[override]
        # Parent modules recurse into children via these hooks; the wrapper
        # has no registered parameters, so forward to the inner model.
        self.inner_model._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):  # type: ignore[override]
        self.inner_model._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def requires_grad_(self, requires_grad: bool = True):  # type: ignore[override]
        self.inner_model.requires_grad_(requires_grad)
        return self

    # ── shared EncoderProtocol pieces ──────────────────────────────────────
    def pool_features(self, x: FeatureMapsOut) -> PoolOut:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}
