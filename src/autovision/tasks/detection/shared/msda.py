# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Optional MultiScaleDeformableAttention CUDA op (opt-in, JIT-compiled).

Vendored from the Deformable-DETR / RT-DETR / Dome-DETR lineage
(``_msda_src/``, Apache-2.0, SenseTime), patched for torch 2.x
(``scalar_type()``/``data_ptr``) and extended to fp16/bf16 dispatch.

The op fuses the per-level bilinear sampling + weighting that
``rtdetrv2_decoder.deformable_attention_core_func_v2`` otherwise expresses
as ``F.grid_sample`` per level. Same math, one kernel per call — the
sampling sits on the critical path of every decoder layer (and its
backward), which is where the D-FINE/Dome step time goes.

Compilation is lazy (first call) and cached by torch's extension cache
keyed on the source hash; set ``AUTOVISION_MSDA_CROSS_ATTN=1`` to route
the decoders' cross-attention through it.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

__all__ = ["ms_deform_attn_cuda", "unload_msda"]

_LOCK = threading.Lock()
_MODULE: object | None = None
_LOAD_ERROR: str | None = None

_SOURCE_DIR = Path(__file__).resolve().parent / "_msda_src"
_SOURCES = [
    _SOURCE_DIR / "vision.cpp",
    _SOURCE_DIR / "ms_deform_attn_cpu.cpp",
    _SOURCE_DIR / "ms_deform_attn_cuda.cu",
]


def _load():
    global _MODULE, _LOAD_ERROR
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        from torch.utils.cpp_extension import load

        try:
            with_cuda = torch.cuda.is_available()
            _MODULE = load(
                name="autovision_ms_deformable_attention",
                sources=[str(path) for path in _SOURCES],
                # vision.cpp guards the CUDA dispatch behind WITH_CUDA —
                # setup.py normally defines it; cpp_extension.load does not.
                extra_cflags=["-O2", "-DWITH_CUDA"] if with_cuda else ["-O2"],
                extra_cuda_cflags=["-O2"],
                with_cuda=with_cuda,
                verbose=False,
            )
        except Exception as error:  # pragma: no cover — toolchain dependent
            _LOAD_ERROR = (
                "MSDA CUDA extension unavailable "
                f"(compile error: {error}); keep "
                "AUTOVISION_MSDA_CROSS_ATTN unset to use the grid_sample path."
            )
            raise RuntimeError(_LOAD_ERROR) from error
        return _MODULE


def unload_msda() -> None:
    """Drop the compiled module (test hook — reload picks up source edits)."""
    global _MODULE, _LOAD_ERROR
    with _LOCK:
        _MODULE = None
        _LOAD_ERROR = None


class _MSDADeformAttnFunction(Function):
    """Autograd shell around the raw bindings (official form)."""

    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        module = _load()
        ctx.im2col_step = im2col_step
        output = module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        module = _load()
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_cuda(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CUDA-op equivalent of ``deformable_attention_core_func_v2``.

    Args mirror the shared core: ``value`` (B, N_total, heads, C),
    ``value_spatial_shapes`` (L, 2), ``sampling_locations``
    (B, Q, heads, L·P, 2) in [0, 1], ``attention_weights``
    (B, Q, heads, L·P) softmax-normalized. Returns (B, Q, heads·C).
    """
    if level_start_index is None:
        level_start_index = torch.cat(
            [
                value_spatial_shapes.new_zeros(1),
                value_spatial_shapes.prod(-1).cumsum(0)[:-1],
            ]
        )
    return _MSDADeformAttnFunction.apply(
        value.contiguous(),
        value_spatial_shapes.contiguous(),
        level_start_index.contiguous(),
        sampling_locations.contiguous(),
        attention_weights.contiguous(),
        64,  # im2col_step — the op's simple path covers our batch sizes
    )
