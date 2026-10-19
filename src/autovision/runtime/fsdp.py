# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""FSDP state-dict helpers (full, unsharded checkpoints).

The training engine owns the checkpoint/export *format* (TaskModel state +
EMA sidecar); FSDP owns the sharding. These helpers are the narrow bridge:
they gather a full state/optimizer dict on rank zero for writing and load a
full dict on every rank for resume.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = [
    "collect_full_state_dict",
    "is_fsdp_module",
    "load_full_state_dict",
    "wrap_fsdp",
]


def is_fsdp_module(module: Any) -> bool:
    """True when ``module`` is an FSDP wrapper (no hard import at module load)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:  # pragma: no cover - torch always ships FSDP
        return False
    return isinstance(module, FullyShardedDataParallel)


def wrap_fsdp(module: nn.Module, device: torch.device) -> nn.Module:
    """Wrap a task model for parameter sharding.

    ``use_orig_params=True`` keeps the original parameter objects so the
    model's existing optimizer builder and EMA implementation keep working.
    ``sync_module_states=True`` broadcasts rank-zero weights at startup.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if device.type != "cuda":
        raise ValueError(
            "FSDP requires a CUDA device (torch FSDP refuses CPU parameters); "
            f"got '{device}'."
        )
    return FSDP(
        module,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )


def collect_full_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    rank0_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gather the full (unsharded) model and optimizer state.

    Every rank must call this (FSDP gathers collectively). With
    ``rank0_only=True`` non-zero ranks receive empty dicts; only rank zero
    writes the checkpoint.
    """
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
    ):
        state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    return state, optim_state


def load_full_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    optim_state: dict[str, Any] | None,
) -> None:
    """Load a full checkpoint into an FSDP model on every rank."""
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(state)
    if optim_state:
        loaded = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
        optimizer.load_state_dict(loaded)
