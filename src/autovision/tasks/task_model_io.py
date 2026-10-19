# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Typed I/O contracts for exportable task models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, model_validator

__all__ = ["BaseModelOutput", "ModelInputSpec", "TensorSpec"]


class TensorSpec(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    shape: tuple[int, ...]  # without the batch dim
    dtype: torch.dtype
    is_batched: bool = True

    def example_tensor(self, batch_size: int | None = None) -> torch.Tensor:
        shape = tuple(self.shape) if self.is_batched is False else (batch_size or 1, *self.shape)
        return torch.zeros(shape, dtype=self.dtype)


class ModelInputSpec(BaseModel):
    input_specs: dict[str, TensorSpec]
    input_dynamic_shapes: dict[str, tuple[Any, ...]] = {}

    @model_validator(mode="after")
    def _validate(self) -> "ModelInputSpec":
        for name, spec in self.input_specs.items():
            rank = len(spec.shape) + int(spec.is_batched)
            dynamic = self.input_dynamic_shapes.get(name)
            if dynamic is not None and len(dynamic) != rank:
                raise ValueError(
                    f"Dynamic shape rank for '{name}' ({len(dynamic)}) does not "
                    f"match the tensor rank ({rank})."
                )
        return self

    def example_inputs(
        self, batch_size: int = 1, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {}
        for name, spec in self.input_specs.items():
            tensor = spec.example_tensor(batch_size)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            inputs[name] = tensor.to(device) if device is not None else tensor
        return inputs


@dataclass
class BaseModelOutput:
    """Typed model output; subclasses auto-register as pytree nodes so they
    can cross ``torch.export`` boundaries."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Note: torch >= 2.6 flattens dataclasses natively in pytree/dynamo,
        # so no custom registration is needed (and custom unflatten handlers
        # break dynamo's output reconstruction).


ModelOutputFlattenSpec = (BaseModelOutput,)
