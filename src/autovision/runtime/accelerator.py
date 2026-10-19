# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Accelerator detection: CUDA first, Ascend NPU (torch_npu) fallback.

Ascend support is a software-side adaptation only (design §12.4): device
resolution, DDP backend (hccl), bf16 capability and mixed-precision guards
accept ``npu``. Real-hardware validation stays an M4+ item — all NPU paths
are exercised via mocks in the test suite.

torch_npu exposes the ``torch.npu`` namespace once imported (its
``transfer_to_npu`` shims most ``torch.cuda`` entry points, but we resolve
devices explicitly instead of relying on global shimming).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

__all__ = [
    "accelerator_device_count",
    "accelerator_device_name",
    "accelerator_supports_bf16",
    "import_torch_npu",
    "npu_available",
]

logger = logging.getLogger("autovision")

_TORCH_NPU: Any = None
_NPU_IMPORT_WARNED = False


def import_torch_npu() -> Any | None:
    """Import torch_npu once; cache the module or ``None`` (soft dependency)."""
    global _TORCH_NPU, _NPU_IMPORT_WARNED
    if _TORCH_NPU is not None:
        return _TORCH_NPU
    if _NPU_IMPORT_WARNED:
        return None
    try:
        import torch_npu  # noqa: F401  (registers the torch.npu namespace)
    except ImportError:
        _NPU_IMPORT_WARNED = True
        return None
    _TORCH_NPU = torch_npu
    return torch_npu


def npu_available() -> bool:
    """True when torch_npu is installed and at least one NPU is visible."""
    if import_torch_npu() is None:
        return False
    try:
        return bool(torch.npu.is_available())
    except (AttributeError, RuntimeError):
        return False


def accelerator_device_name() -> str | None:
    """Preferred accelerator type: ``"cuda"`` > ``"npu"``; ``None`` = CPU only."""
    if torch.cuda.is_available():
        return "cuda"
    if npu_available():
        return "npu"
    return None


def accelerator_device_count() -> int:
    """Device count of the preferred accelerator (0 when CPU only)."""
    accelerator = accelerator_device_name()
    if accelerator == "cuda":
        return torch.cuda.device_count()
    if accelerator == "npu":
        return torch.npu.device_count()
    return 0


def accelerator_supports_bf16() -> bool:
    """bf16 capability of the preferred accelerator (precision "auto" picks
    ``bf16-mixed`` when this returns True)."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if npu_available():
        try:
            return torch.npu.is_bf16_supported()
        except (AttributeError, RuntimeError):
            return False
    return False


def ddp_backend_for(device_type: str) -> str:
    """Distributed backend: nccl on CUDA, hccl on NPU, gloo otherwise."""
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


def supports_amp(device_type: str) -> bool:
    """Mixed-precision guard: autocast accelerators are CUDA and NPU."""
    return device_type in ("cuda", "npu")
