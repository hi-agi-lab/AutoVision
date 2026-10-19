# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model cost measurement: parameters, MACs and FLOPs (design doc 08 §12).

Counting convention: PyTorch's ``FlopCounterMode`` reports a multiply-add
as two FLOPs for the ops in its registry (matmul, convolutions, scaled
dot-product attention, ...). Detection papers and vendor tables quote
"GFLOPs" in the MAC convention, so :class:`ModelCost` exposes both —
``flops`` exactly as the counter reports and ``macs = flops / 2`` for the
number comparable with published tables. Ops outside the counter's
registry are not counted (undercount, never overcount).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

__all__ = ["ModelCost", "count_parameters", "measure_model_cost"]


@dataclass(frozen=True)
class ModelCost:
    """Cost snapshot for one concrete input shape."""

    parameters: int
    trainable_parameters: int
    macs: float
    flops: float
    input_shape: tuple[int, ...]
    device: str

    @property
    def gmacs(self) -> float:
        """MACs in billions — the convention published tables quote as GFLOPs."""
        return self.macs / 1e9

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["gmacs"] = self.gmacs
        return payload


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """``(total, trainable)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


@torch.no_grad()
def measure_model_cost(
    model: torch.nn.Module,
    *,
    example_input: torch.Tensor | None = None,
    batch_size: int = 1,
    device: str | torch.device | None = None,
) -> ModelCost:
    """Parameters + MACs/FLOPs for one forward at the given input shape.

    The input defaults to the model's own ``model_input_spec`` example at
    ``batch_size`` (the same contract the export pipeline and the benchmark
    harness use), so the reading is the cost of the exported graph's input.
    The forward runs through ``forward_backend`` — the entry the latency
    harness times — under ``torch.utils.flop_counter.FlopCounterMode``.
    """
    from torch.utils.flop_counter import FlopCounterMode

    if device is not None:
        model = model.to(torch.device(device))
    was_training = model.training
    model.eval()
    try:
        if example_input is None:
            parameter = next(model.parameters(), None)
            target_device = parameter.device if parameter is not None else torch.device("cpu")
            example_input = next(
                iter(
                    model.model_input_spec.example_inputs(
                        batch_size=batch_size, device=target_device
                    ).values()
                )
            )
        parameters, trainable = count_parameters(model)
        # TaskNetworks expose ``forward_backend`` (the entry the latency
        # harness times); plain modules count through ``forward``.
        forward_fn = getattr(model, "forward_backend", None) or model.forward
        with FlopCounterMode(display=False) as counter:
            forward_fn(example_input)
        flops = float(counter.get_total_flops())
        return ModelCost(
            parameters=parameters,
            trainable_parameters=trainable,
            macs=flops / 2.0,
            flops=flops,
            input_shape=tuple(example_input.shape),
            device=str(example_input.device),
        )
    finally:
        if was_training:
            model.train()
