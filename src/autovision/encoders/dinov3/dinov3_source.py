# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv3 package, backed by HuggingFace ``transformers``.

Per framework policy, pretrained DINOv3 usage prefers ``transformers`` over
vendored code. transformers gained DINOv3 support relatively recently, so
this package is **gated**: it auto-enables when the installed transformers
exposes the DINOv3 classes and degrades gracefully (empty model list,
informative error) otherwise — no vendored fallback.

Per decision D1-a (design doc 28 §6.1) the DINOv3 backbone line is
**user-held weights**: pretrained checkpoints are only fetched when the
environment sets ``AUTOVISION_DINOV3_WEIGHTS=1`` (the user's own
acceptance of Meta's DINOv3 License); ``-notpretrained`` config builds
and explicit ``weights_path`` loads are unaffected.

Model names mirror the baseline registry: ``vits16(+plus)``, ``vitb16``,
``vitl16(+plus alias vith16plus)``, ``vit7b16``, ``convnext-*``, SAT
variants, each with a ``-notpretrained`` offline variant.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autovision.encoders.dinov3.dinov3_wrapper import (
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)
from autovision.encoders.package import CatalogModelSource
from autovision.encoders.weight_verification import load_pretrained_verified

__all__ = [
    "DINOV3_SOURCE",
    "Dinov3Source",
    "dinov3_weights_opt_in",
]

_PATCH_SIZE = 16
_IMAGE_SIZE = 224

# model name -> (HF checkpoint, architecture, config kwargs)
#: DINOv3 ViT release geometry: every published ViT (LVD-1689M and the
#: vitt16/vitt16plus distillations) carries 4 storage (register) tokens.
_NUM_REGISTER_TOKENS = 4

_MODEL_INFO: dict[str, dict[str, Any]] = {
    "vitt16": dict(
        # Distilled tiny model with no public HF transformers checkpoint —
        # only the -notpretrained (config-built) name resolves; distilled
        # weights must come in through backbone_weights/detector loads.
        hf=None,
        arch="vit",
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vitt16plus": dict(
        hf=None,
        arch="vit",
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=1152,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vits16": dict(
        hf="facebook/dinov3-vits16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vits16plus": dict(
        hf="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        arch="vit",
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vitb16": dict(
        hf="facebook/dinov3-vitb16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vitl16": dict(
        hf="facebook/dinov3-vitl16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vith16plus": dict(
        hf="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        arch="vit",
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vit7b16": dict(
        hf="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        arch="vit",
        hidden_size=4096,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_register_tokens=_NUM_REGISTER_TOKENS,
    ),
    "vitl16-sat493m": dict(
        hf="facebook/dinov3-vitl16-pretrain-sat493m",
        arch="vit",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "vit7b16-sat493m": dict(
        hf="facebook/dinov3-vit7b16-pretrain-sat493m",
        arch="vit",
        hidden_size=4096,
        num_hidden_layers=40,
        num_attention_heads=16,
    ),
    # ``depths`` must be explicit: DINOv3ConvNextConfig defaults to the
    # tiny layout [3, 3, 9, 3] regardless of ``hidden_sizes``, which
    # silently builds small/base/large one-third deep (the published
    # small/base/large checkpoints are 3 + 3 + 27 + 3 blocks).
    "convnext-tiny": dict(
        hf="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
    ),
    "convnext-small": dict(
        hf="facebook/dinov3-convnext-small-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[96, 192, 384, 768],
        depths=[3, 3, 27, 3],
    ),
    "convnext-base": dict(
        hf="facebook/dinov3-convnext-base-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
    ),
    "convnext-large": dict(
        hf="facebook/dinov3-convnext-large-pretrain-lvd1689m",
        arch="convnext",
        hidden_sizes=[192, 384, 768, 1536],
        depths=[3, 3, 27, 3],
    ),
}


def _dinov3_classes() -> dict[str, Any] | None:
    """Live detection of DINOv3 support in the installed transformers.

    Returns the class map or ``None`` when transformers is missing or too
    old. Reads module attributes (not imports) so the check stays honest
    across transformers versions with differing class names.
    """
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        return None
    # transformers exports the DINOv3 classes as ``DINOv3*`` (both the 4.x
    # lazy exports and 5.x); the ``Dinov3*`` spellings are defensive
    # fallbacks for intermediate layouts.
    vit_config = (
        getattr(transformers, "DINOv3ViTConfig", None)
        or getattr(transformers, "Dinov3ViTConfig", None)
        or getattr(transformers, "Dinov3Config", None)
    )
    vit_model = (
        getattr(transformers, "DINOv3ViTModel", None)
        or getattr(transformers, "Dinov3ViTModel", None)
        or getattr(transformers, "Dinov3Model", None)
    )
    convnext_config = (
        getattr(transformers, "DINOv3ConvNextConfig", None)
        or getattr(transformers, "Dinov3ConvNextConfig", None)
    )
    convnext_model = (
        getattr(transformers, "DINOv3ConvNextModel", None)
        or getattr(transformers, "Dinov3ConvNextModel", None)
    )
    if None in (vit_config, vit_model, convnext_config, convnext_model):
        return None
    return {
        "vit_config": vit_config,
        "vit_model": vit_model,
        "convnext_config": convnext_config,
        "convnext_model": convnext_model,
    }


def _sync_num_channels(model: Any, num_input_channels: int) -> None:
    """Sync every captured num_channels copy (config/embeddings/stem)."""
    model.config.num_channels = num_input_channels
    embeddings = getattr(model, "embeddings", None)
    for owner in (embeddings, getattr(embeddings, "patch_embeddings", None)):
        if owner is not None and hasattr(owner, "num_channels"):
            owner.num_channels = num_input_channels


def _support_error(model_name: str) -> ImportError:
    return ImportError(
        f"DINOv3 model '{model_name}' requires a transformers version with "
        f"DINOv3 support (class Dinov3ViTModel/Dinov3ConvNextModel). "
        f"Upgrade: pip install -U 'autovision[transformers]'."
    )


# ── D1-a: user-held DINOv3 weights (design doc 28, section 6.1) ─────────
#
# DINOv3 backbone weights are published by Meta under the (non-Apache)
# DINOv3 License. Per decision D1-a the backbone line stays supported, but
# the framework never fetches these weights by default: the user must
# opt in per environment, asserting that they hold the weights under their
# own acceptance of Meta's license terms. Config-built (-notpretrained)
# and weights_path-supplied usage never touches this gate.

DINOV3_WEIGHTS_ENV = "AUTOVISION_DINOV3_WEIGHTS"
_TRUTHY = frozenset({"1", "yes", "true", "on"})


def dinov3_weights_opt_in() -> bool:
    """Whether the user opted in to fetching user-held DINOv3 weights."""
    return os.environ.get(DINOV3_WEIGHTS_ENV, "").strip().lower() in _TRUTHY


def _license_gate_error(model_name: str, repo_id: str) -> PermissionError:
    return PermissionError(
        f"DINOv3 pretrained weights are NOT fetched by default (decision "
        f"D1-a: the weights are user-held under Meta's DINOv3 License, "
        f"which this framework does not accept on anyone's behalf). To "
        f"load '{model_name}' from '{repo_id}', set "
        f"{DINOV3_WEIGHTS_ENV}=1 in the environment after accepting the "
        f"DINOv3 License yourself, or build the model config-only "
        f"('{model_name}-notpretrained') and load user-provided weights "
        f"via backbone_weights."
    )


class Dinov3Source(CatalogModelSource):
    name = "dinov3"

    @classmethod
    def catalog_names(cls) -> list[str]:
        if _dinov3_classes() is None:
            return []
        names: list[str] = []
        for model_name, info in _MODEL_INFO.items():
            # Entries without a public HF checkpoint only resolve as
            # -notpretrained; do not advertise a plain name that errors.
            names.append(f"{cls.name}/{model_name}-notpretrained")
            if info.get("hf") is not None:
                names.append(f"{cls.name}/{model_name}")
        return sorted(names)

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        if isinstance(
            model, (TransformersDinov3ViTWrapper, TransformersDinov3ConvNextWrapper)
        ):
            return True
        classes = _dinov3_classes()
        if classes is None:
            return False
        return isinstance(model, (classes["vit_model"], classes["convnext_model"]))

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        classes = _dinov3_classes()
        if classes is None:
            raise _support_error(model_name)
        base_name = model_name.removesuffix("-notpretrained")
        pretrained = model_name == base_name
        if base_name not in _MODEL_INFO:
            supported = ", ".join(sorted(_MODEL_INFO))
            raise ValueError(
                f"Unknown dinov3 model: '{model_name}'. Supported models: {supported}."
            )
        info = _MODEL_INFO[base_name]
        config_kwargs = dict(model_args or {})
        if pretrained and load_weights:
            if info.get("hf") is None:
                raise ValueError(
                    f"dinov3 model '{model_name}' has no public HF "
                    f"checkpoint; use '{base_name}-notpretrained' (config "
                    f"build) and load pretrained weights via "
                    f"backbone_weights instead."
                )
            model_cls = (
                classes["vit_model"] if info["arch"] == "vit" else classes["convnext_model"]
            )
            if not dinov3_weights_opt_in():
                raise _license_gate_error(model_name, info["hf"])
            model = load_pretrained_verified(model_cls, info["hf"], config_kwargs)
        elif info["arch"] == "vit":
            config = classes["vit_config"](
                hidden_size=info["hidden_size"],
                num_hidden_layers=info["num_hidden_layers"],
                num_attention_heads=info["num_attention_heads"],
                image_size=_IMAGE_SIZE,
                patch_size=_PATCH_SIZE,
                **{
                    key: info[key]
                    for key in ("intermediate_size", "num_register_tokens")
                    if key in info
                },
                **config_kwargs,
            )
            model = classes["vit_model"](config)
        else:
            config = classes["convnext_config"](
                hidden_sizes=info["hidden_sizes"],
                image_size=_IMAGE_SIZE,
                patch_size=_PATCH_SIZE,
                **{key: info[key] for key in ("depths",) if key in info},
                **config_kwargs,
            )
            model = classes["convnext_model"](config)
        if num_input_channels != 3:
            cls._adapt_input_channels(model, num_input_channels)
        return model

    @classmethod
    def _adapt_input_channels(cls, model: Any, num_input_channels: int) -> None:
        from autovision.encoders._model_helpers import adapt_conv_input_channels

        if hasattr(model.config, "hidden_sizes"):  # ConvNeXt
            stem = model.embeddings.patch_embeddings
            new_stem = nn.Conv2d(
                num_input_channels,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=stem.bias is not None,
            )
            with torch.no_grad():
                new_stem.weight.copy_(
                    adapt_conv_input_channels(stem.weight.data, num_input_channels)
                )
                if stem.bias is not None:
                    new_stem.bias.copy_(stem.bias.data)
            model.embeddings.patch_embeddings = new_stem
            _sync_num_channels(model, num_input_channels)
        else:  # ViT
            projection = model.embeddings.patch_embeddings.projection
            new_projection = nn.Conv2d(
                num_input_channels,
                projection.out_channels,
                kernel_size=projection.kernel_size,
                stride=projection.stride,
                padding=projection.padding,
                bias=projection.bias is not None,
            )
            with torch.no_grad():
                new_projection.weight.copy_(
                    adapt_conv_input_channels(
                        projection.weight.data, num_input_channels
                    )
                )
                if projection.bias is not None:
                    new_projection.bias.copy_(projection.bias.data)
            model.embeddings.patch_embeddings.projection = new_projection
            _sync_num_channels(model, num_input_channels)

    @classmethod
    def as_encoder(cls, model: Any) -> Any:
        if isinstance(model, TransformersDinov3ViTWrapper):
            return model
        if isinstance(model, TransformersDinov3ConvNextWrapper):
            return model
        if hasattr(getattr(model, "config", None), "hidden_sizes"):
            return TransformersDinov3ConvNextWrapper(model)
        return TransformersDinov3ViTWrapper(model)

    @classmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapper = cls.as_encoder(model)
        torch.save(wrapper.raw_module().state_dict(), out)
        if log_example:
            print(
                f"Exported DINOv3 model to '{out}'. Rebuild with "
                f"autovision.get_wrapped_model('<dinov3 model name>') and "
                f"load_state_dict(torch.load('{out}'))."
            )


DINOV3_SOURCE = Dinov3Source()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(DINOV3_SOURCE)
