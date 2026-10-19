# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Parameter grouping: weight-decay exclusion rules and optimizer building."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn.parameter import Parameter

from autovision.optim.trainable_modules import OptimizationTargets

__all__ = ["build_adamw", "get_optimizer", "get_weight_decay_parameters"]

logger = logging.getLogger("autovision")


def build_adamw(
    param_groups: list[dict],
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    fused: bool,
) -> torch.optim.AdamW:
    """AdamW with the optional fused single-kernel step (design doc 26 W1).

    ``fused=True`` collapses the whole per-step parameter update into one
    kernel launch — measured 8.70 → 0.71 ms/step on a 21M-parameter /
    140-tensor detector shape (doc 26 §10.1). The fused update reorders
    float ops relative to the foreach implementation, so runs are only
    bitwise-comparable within the same setting: the flag is off by
    default and callers flip it for fresh runs. Unavailability (CPU/NPU
    parameters, older torch) falls back to the foreach implementation
    with a visible warning — never a hard failure mid-launch.
    """
    if not fused:
        return torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, weight_decay=weight_decay
        )
    try:
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            fused=True,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.warning(
            "fused_optimizer=True is unavailable here (%s); falling back "
            "to the foreach AdamW implementation.",
            error,
        )
        return torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, weight_decay=weight_decay
        )

# Module types whose parameters never receive weight decay.
_NORM_LAYERS = (
    nn.modules.batchnorm._NormBase,
    nn.LayerNorm,
    nn.CrossMapLRN2d,
    nn.LocalResponseNorm,
    nn.GroupNorm,
)

# Substring rules: any occurrence in the parameter name excludes decay.
_SUBSTRING_NO_DECAY_KEYS = (
    "cls_token",
    "mask_token",
    "storage_token",
    "register_token",
    "queries",
    "query",
    "pos_embed",
    "position_embedding",
    "positional_embedding",
    "relative_position",
    "relative_pos",
    "rel_pos",
    "rpb",
    "bias_table",
)

# Whole-word rules: the name must contain the key as a dotted path component.
_WHOLE_WORD_NO_DECAY_KEYS = (
    "bias",
    "bias_v",
    "bias_k",
    "logit_scale",
    "temperature",
    "tau",
    "layer_scale",
    "layerscale",
    "res_scale",
    "rezero",
    "gamma",
)


def _contains_key(name: str, key: str) -> bool:
    """Whole-word match: ``key`` appears as a complete dotted-path component."""
    padded = f".{name}."
    return f".{key}." in padded


def _is_no_weight_decay(
    module: nn.Module,
    param_name: str,
    param: Parameter,
    extra_no_decay_keys: Sequence[str],
) -> bool:
    if isinstance(module, _NORM_LAYERS):
        return True
    if "Norm" in type(module).__name__:
        return True
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
        return True
    if param.ndim <= 1:
        return True
    if any(_contains_key(param_name, key) for key in _WHOLE_WORD_NO_DECAY_KEYS):
        return True
    if any(key in param_name for key in _SUBSTRING_NO_DECAY_KEYS):
        return True
    if any(key in param_name for key in extra_no_decay_keys):
        return True
    return False


def get_weight_decay_parameters(
    modules: Iterable[nn.Module],
    extra_no_decay_keys: Sequence[str] = (),
) -> tuple[list[Parameter], list[Parameter]]:
    """Split parameters into (with weight decay, without weight decay).

    A parameter is excluded from weight decay if any of:
    - its module is a normalization layer (or its class name contains "Norm"),
    - its module is an Embedding/EmbeddingBag,
    - it has ndim <= 1 (biases, norms scales, ...),
    - its name matches one of the token/position/logit-scale/layer-scale rules.

    Parameters are deduplicated across (shared) modules.
    """
    params_weight_decay: list[Parameter] = []
    params_no_weight_decay: list[Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for sub_module in module.modules():
            for name, param in sub_module.named_parameters(recurse=False):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if _is_no_weight_decay(sub_module, name, param, extra_no_decay_keys):
                    params_no_weight_decay.append(param)
                else:
                    params_weight_decay.append(param)
    return params_weight_decay, params_no_weight_decay


def get_optimizer(
    optim_args,
    trainable_modules: OptimizationTargets,
    lr_scale: float,
):
    """Build the optimizer with standard parameter groups.

    Groups: ``params`` (with decay) and ``params_no_weight_decay`` (only when
    non-empty). ``lr_scale`` is applied by the optimizer args themselves.
    """
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        modules=trainable_modules.targets
    )
    # Modules marked as no-weight-decay move ALL their parameters out of the
    # decay group (never duplicated across groups).
    forced_no_wd_ids = {
        id(p)
        for module in trainable_modules.decay_free
        for p in module.parameters()
    }
    if forced_no_wd_ids:
        params_weight_decay = [
            p for p in params_weight_decay if id(p) not in forced_no_wd_ids
        ]
        existing = {id(p) for p in params_no_weight_decay}
        for module in trainable_modules.decay_free:
            for param in module.parameters():
                if id(param) not in existing:
                    existing.add(id(param))
                    params_no_weight_decay.append(param)

    params: list[dict] = [{"name": "params", "params": params_weight_decay}]
    if params_no_weight_decay:
        params.append(
            {
                "name": "params_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            }
        )
    return optim_args.get_optimizer(params=params, lr_scale=lr_scale)
