# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Distributed rank detection without requiring an initialized process group.

Rank detection is based on environment variables so that helpers work before
torch.distributed is initialized (e.g. for rank-0-only file writes during
setup). Detection chains mirror the baseline and add Kubernetes variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

__all__ = [
    "get_global_rank",
    "get_local_rank",
    "get_node_rank",
    "is_global_rank_zero",
    "is_local_rank_zero",
    "is_node_rank_zero",
    "reduce_dict",
]

_GLOBAL_RANK_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
_LOCAL_RANK_VARS = ("LOCAL_RANK", "SLURM_LOCALID", "JSM_NAMESPACE_LOCAL_RANK")
_NODE_RANK_VARS = ("NODE_RANK", "GROUP_RANK", "SLURM_NODEID")


def _detect_from_env(
    vars_: tuple[str, ...], env: Mapping[str, str] | None = None
) -> int | None:
    source = os.environ if env is None else env
    for name in vars_:
        raw = source.get(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def get_global_rank(env: Mapping[str, str] | None = None) -> int | None:
    return _detect_from_env(_GLOBAL_RANK_VARS, env)


def get_local_rank(env: Mapping[str, str] | None = None) -> int | None:
    return _detect_from_env(_LOCAL_RANK_VARS, env)


def get_node_rank(env: Mapping[str, str] | None = None) -> int | None:
    return _detect_from_env(_NODE_RANK_VARS, env)


def is_node_rank_zero(env: Mapping[str, str] | None = None) -> bool:
    node_rank = get_node_rank(env)
    return node_rank == 0 or node_rank is None


def is_local_rank_zero(env: Mapping[str, str] | None = None) -> bool:
    """True if this is local rank zero.

    An unset local rank is treated as rank zero: in single-process runs every
    process must behave like rank zero.
    """
    local_rank = get_local_rank(env)
    return local_rank == 0 or local_rank is None


def is_global_rank_zero(env: Mapping[str, str] | None = None) -> bool:
    global_rank = get_global_rank(env)
    return global_rank == 0 or (global_rank is None and is_node_rank_zero(env))


def _distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def reduce_dict(data: dict[str, Any], average: bool = True) -> dict[str, Any]:
    """Reduce a dict of tensors across all ranks (mean or sum).

    No-op when torch.distributed is not initialized; tensors are averaged (or
    summed) keyed-consistently across ranks. Key sets may differ between
    ranks (e.g. denoising loss keys exist only where the batch contains GT
    boxes): the reduction runs over the union, with missing keys counted as
    zero, so the communication shape matches on every rank.
    """
    if not _distributed_available() or torch.distributed.get_world_size() < 2:
        return data
    world_size = torch.distributed.get_world_size()
    local_keys = sorted(data.keys())
    gathered_keys: list[list[str]] = [[] for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_keys, local_keys)
    all_keys = sorted({key for keys in gathered_keys for key in keys})
    device = next(iter(data.values())).device if data else torch.device("cpu")
    values = torch.stack(
        [
            data.get(key, torch.zeros((), device=device)).detach().clone()
            for key in all_keys
        ]
    )
    torch.distributed.all_reduce(values)
    if average:
        values /= world_size
    return dict(zip(all_keys, values, strict=True))
