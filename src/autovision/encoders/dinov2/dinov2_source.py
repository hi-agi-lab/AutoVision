# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv2 package, backed by HuggingFace ``transformers``.

Model names mirror the baseline registry:

- ``dinov2/{vits14,vitb14,vitl14,vitg14}`` — with register tokens,
  pretrained via ``transformers`` (``facebook/dinov2-with-registers-*``;
  requires network access on first use).
- ``dinov2/{...}-noreg`` — without register tokens, pretrained
  (``facebook/dinov2-*``).
- ``dinov2/{...}-notpretrained`` — random initialization built locally from
  the config (works fully offline).

The TiPSv2 variants of the baseline have no transformers counterpart and
are not exposed.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch

from autovision.encoders._model_helpers import adapt_conv_input_channels
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.encoders.package import CatalogModelSource
from autovision.encoders.weight_verification import load_pretrained_verified

__all__ = ["DINOV2_SOURCE", "Dinov2Source"]

_PATCH_SIZE = 14
_IMAGE_SIZE = 224  # local untrained builds; divisible by the patch size

# model suffix -> (HF with-registers checkpoint, HF plain checkpoint, config)
_MODEL_INFO: dict[str, dict[str, Any]] = {
    "vits14": dict(
        hf="facebook/dinov2-with-registers-small",
        hf_noreg="facebook/dinov2-small",
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
    ),
    "vitb14": dict(
        hf="facebook/dinov2-with-registers-base",
        hf_noreg="facebook/dinov2-base",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "vitl14": dict(
        hf="facebook/dinov2-with-registers-large",
        hf_noreg="facebook/dinov2-large",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "vitg14": dict(
        hf="facebook/dinov2-with-registers-giant",
        hf_noreg="facebook/dinov2-giant",
        hidden_size=1536,
        num_hidden_layers=40,
        num_attention_heads=24,
    ),
}

_LICENSE_LABEL = "apache-2.0"


def _get_transformers() -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "The 'transformers' package is required for DINOv2 models. "
            "Install it with: pip install 'autovision[transformers]'."
        ) from error
    return transformers


def _parse_model_name(model_name: str) -> tuple[str, bool, bool]:
    """Split into (base size, use_registers, pretrained).

    Returns (size, registers, pretrained); ``-notpretrained`` names build
    with registers (the default architecture) from the local config.
    """
    if model_name.endswith("-notpretrained"):
        return model_name[: -len("-notpretrained")], True, False
    if model_name.endswith("-noreg"):
        return model_name[: -len("-noreg")], False, True
    return model_name, True, True


# ── Pretrained-weights license gate (CC-BY-NC-4.0) ──────────────────────
#
# The DINOv2 CODE — and its transformers port — is Apache-2.0, but the
# pretrained WEIGHTS Meta publishes on the HF hub (``facebook/dinov2-*``,
# ``facebook/dinov2-with-registers-*``) are CC-BY-NC 4.0, i.e.
# non-commercial. The framework never accepts those terms on anyone's
# behalf: fetching them requires an explicit per-environment opt-in.
# Config-built (``-notpretrained``) models never touch this gate — that
# plus user-provided weights (e.g. this framework's own self-supervised
# pretraining output) is the unrestricted path.

DINOV2_WEIGHTS_ENV = "AUTOVISION_DINOV2_WEIGHTS"
_TRUTHY = frozenset({"1", "yes", "true", "on"})


def dinov2_weights_opt_in() -> bool:
    """Whether the user opted in to fetching the CC-BY-NC DINOv2 weights."""
    return os.environ.get(DINOV2_WEIGHTS_ENV, "").strip().lower() in _TRUTHY


def _license_gate_error(model_name: str, repo_id: str) -> PermissionError:
    return PermissionError(
        f"DINOv2 pretrained weights are NOT fetched by default: Meta "
        f"publishes them under CC-BY-NC 4.0 (non-commercial — the DINOv2 "
        f"code is Apache-2.0, the weights are not). To load '{model_name}' "
        f"from '{repo_id}', set {DINOV2_WEIGHTS_ENV}=1 in the environment "
        f"after accepting the CC-BY-NC 4.0 terms yourself, or build the "
        f"model config-only ('{model_name}-notpretrained') and load "
        f"user-provided weights via backbone_weights."
    )


class Dinov2Source(CatalogModelSource):
    name = "dinov2"

    @classmethod
    def catalog_names(cls) -> list[str]:
        try:
            _get_transformers()
        except ImportError:
            # Degrade to [] when transformers is missing (matches the other
            # gated packages; see the module docstring).
            return []
        names: list[str] = []
        for size in _MODEL_INFO:
            names.extend(
                [
                    f"{cls.name}/{size}",
                    f"{cls.name}/{size}-noreg",
                    f"{cls.name}/{size}-notpretrained",
                ]
            )
        return sorted(names)

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        if isinstance(model, TransformersDinov2Wrapper):
            return True
        try:
            transformers = _get_transformers()
        except ImportError:
            return False
        return isinstance(
            model, (transformers.Dinov2Model, transformers.Dinov2WithRegistersModel)
        )

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        transformers = _get_transformers()
        size, use_registers, pretrained = _parse_model_name(model_name)
        if size not in _MODEL_INFO:
            supported = ", ".join(sorted(_MODEL_INFO))
            raise ValueError(
                f"Unknown dinov2 model: '{model_name}'. Supported sizes: {supported}."
            )
        info = _MODEL_INFO[size]
        config_kwargs = dict(model_args or {})
        # The mask token is part of the baseline DINOv2 architecture and the
        # iBOT method requires it; HF gates it off by default only for
        # checkpoint parity. Enable it unless explicitly disabled. Missing
        # mask_token keys in pretrained checkpoints are zeros-initialized by
        # transformers (matches the official implementation).
        config_kwargs.setdefault("use_mask_token", True)
        if pretrained and load_weights:
            hf_name = info["hf"] if use_registers else info["hf_noreg"]
            model_cls = (
                transformers.Dinov2WithRegistersModel
                if use_registers
                else transformers.Dinov2Model
            )
            if not dinov2_weights_opt_in():
                raise _license_gate_error(model_name, hf_name)
            model = load_pretrained_verified(model_cls, hf_name, config_kwargs)
        else:
            # Local (offline) build from the config.
            num_registers = config_kwargs.pop("num_register_tokens", None)
            if num_registers is None:
                num_registers = 4 if use_registers else 0
            if use_registers:
                config = transformers.Dinov2WithRegistersConfig(
                    hidden_size=info["hidden_size"],
                    num_hidden_layers=info["num_hidden_layers"],
                    num_attention_heads=info["num_attention_heads"],
                    image_size=_IMAGE_SIZE,
                    patch_size=_PATCH_SIZE,
                    num_register_tokens=num_registers,
                    **config_kwargs,
                )
                model = transformers.Dinov2WithRegistersModel(config)
            else:
                config = transformers.Dinov2Config(
                    hidden_size=info["hidden_size"],
                    num_hidden_layers=info["num_hidden_layers"],
                    num_attention_heads=info["num_attention_heads"],
                    image_size=_IMAGE_SIZE,
                    patch_size=_PATCH_SIZE,
                    num_register_tokens=num_registers,
                    **config_kwargs,
                )
                model = transformers.Dinov2Model(config)
        if num_input_channels != 3:
            cls._adapt_patch_embed_channels(model, num_input_channels)
        return model

    @classmethod
    def _adapt_patch_embed_channels(cls, model: Any, num_input_channels: int) -> None:
        projection = model.embeddings.patch_embeddings.projection
        if projection.in_channels == num_input_channels:
            return
        new_projection = torch.nn.Conv2d(
            num_input_channels,
            projection.out_channels,
            kernel_size=projection.kernel_size,
            stride=projection.stride,
            padding=projection.padding,
            bias=projection.bias is not None,
        )
        with torch.no_grad():
            new_projection.weight.copy_(
                adapt_conv_input_channels(projection.weight.data, num_input_channels)
            )
            if projection.bias is not None:
                new_projection.bias.copy_(projection.bias.data)
        model.embeddings.patch_embeddings.projection = new_projection
        # transformers validates channels against captured copies at several
        # levels (config, embeddings, patch_embeddings) depending on the
        # model variant — sync them all, defensively.
        model.config.num_channels = num_input_channels
        for owner in (model.embeddings, model.embeddings.patch_embeddings):
            if hasattr(owner, "num_channels"):
                owner.num_channels = num_input_channels

    @classmethod
    def as_encoder(cls, model: Any) -> TransformersDinov2Wrapper:
        if isinstance(model, TransformersDinov2Wrapper):
            return model
        return TransformersDinov2Wrapper(model)

    @classmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapper = cls.as_encoder(model)
        torch.save(wrapper.raw_module().state_dict(), out)
        if log_example:
            print(
                f"Exported DINOv2 model to '{out}'. Rebuild with "
                f"autovision.get_wrapped_model('<dinov2 model name>') and "
                f"load_state_dict(torch.load('{out}'))."
            )


DINOV2_SOURCE = Dinov2Source()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(DINOV2_SOURCE)
