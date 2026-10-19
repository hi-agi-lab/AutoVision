# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared model-layer helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch

__all__ = ["adapt_conv_input_channels", "load_state_dict_compat"]


def load_state_dict_compat(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Strictly load a state dict, tolerating missing ``mask_token`` keys.

    Newer dinov2 builds always carry ``embeddings.mask_token`` (HF
    zeros-initializes it); checkpoints exported by older versions predate
    that key. Missing ``mask_token`` keys are accepted as-is — the parameter
    keeps its zero init, matching the official DINOv2 semantics. Any other
    missing key, and any unexpected key, still raises ``RuntimeError`` so
    real architecture mismatches stay loud.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{type(model).__name__}: unexpected key(s) {sorted(unexpected)}."
        )
    real_missing = [key for key in missing if key.rpartition(".")[2] != "mask_token"]
    if real_missing:
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{type(model).__name__}: missing key(s) {real_missing} (only "
            "'mask_token' keys may be absent for checkpoints exported before "
            "the mask-token default)."
        )


def adapt_conv_input_channels(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    """Repeat or trim the input channel dimension of a conv weight.

    Used to adapt pretrained 3-channel backbones to arbitrary
    ``num_input_channels`` (grayscale, multispectral, ...): extra pretrained
    channels are trimmed, missing channels are repeated from the full set.
    """
    pretrained_chans = weight.shape[1]
    if pretrained_chans == in_chans:
        return weight
    if pretrained_chans > in_chans:
        return weight[:, :in_chans].clone()
    repeats = in_chans // pretrained_chans
    remainder = in_chans % pretrained_chans
    parts = [weight] * repeats
    if remainder:
        parts.append(weight[:, :remainder])
    return torch.cat(parts, dim=1)
