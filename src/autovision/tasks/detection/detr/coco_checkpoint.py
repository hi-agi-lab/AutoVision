# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Loader for baseline DETR COCO detector checkpoints.

A baseline ``.pt`` (a user-held file or a registered D-FINE COCO asset)
stores the EMA weights of the baseline task model::

    {"train_model": {"ema_model.model.<key>": tensor},
     "model_init_args": ..., "model_class_path": ...}

The baseline state splits into backbone / wrapper-extras / encoder / decoder
groups. The raw-backbone key prefix identifies the checkpoint family, and
each family needs its own remap onto the current HF ``transformers`` layout:

ConvNeXt (``backbone.backbone.*``)
    The baseline ships the OLD ConvNeXt parameterization —
    ``downsample_layers.*`` at the top level, blocks named ``dwconv`` /
    ``norm`` / ``pwconv1`` / ``pwconv2``, the final norm under ``norm``, and
    ``norms.3`` as an ALIAS of that same final norm (baseline
    vendored ConvNeXt source: ``norms = [Identity×3, self.norm]``).
    Everything is remapped onto the current HF ``DINOv3ConvNextModel``
    layout (``stages.*.downsample_layers.*``, ``depthwise_conv``,
    ``layer_norm``, ...); the ``norms.3`` alias is dropped.

ViT (``backbone.dinov3.*``)
    The baseline ships the Meta DINOv3 ViT parameterization — fused
    ``blocks.{i}.attn.qkv`` with a position-independent ``bias_mask``
    channel gate, ``ls1/ls2.gamma`` layer scales, ``mlp.fc1/fc2``, and
    ``storage_tokens``. Everything is remapped onto the HF
    ``DINOv3ViTModel`` layout: the qkv weight/bias split into
    ``attention.{q,v}_proj`` (the masked k-bias must be identically zero —
    HF's attention has no k bias — and the mask itself is dropped as
    functional), ``ls{1,2}.gamma`` → ``layer_scale{1,2}.lambda1``,
    ``mlp.fc1/fc2`` → ``mlp.up_proj/down_proj``, ``storage_tokens`` →
    ``embeddings.register_tokens``, and ``mask_token`` gains its token
    axis. ``rope_embed.periods`` is not a state-dict entry on the HF side
    (HF recomputes the inverse frequencies from ``config.rope_theta``),
    but the trained buffer is authoritative and differs from the
    recomputation by ~1e-4 relative — it is transported to the HF rope
    module's ``inv_freq`` after the strict load (see
    ``_VIT_ROPE_INV_FREQ_KEY``).

Wrapper extras (``backbone.convs/norms.*``, ConvNeXt checkpoints only)
    The baseline ConvNeXt wrapper's 1x1 projections and output norms —
    trained during the COCO run, so they are loaded strictly into the
    target's identically-named ``ConvNeXtDetectionBackbone`` submodules.

STA-era ViT checkpoints (``backbone.sta.*``) are NOT supported: the
DEIMv2-derived spatial prior was removed for licensing. The loader
rejects them with an explicit error instead of silently dropping the
tensors; DINOv3 ViT targets use the no-STA adapter and require a
retrained/exported checkpoint.

Width and depth contract: the baseline couples a per-variant ENCODER width
(ConvNeXt: 384; ViT: 224/192/256 — shared with the decoder) with a FIXED
256 decoder width and a 6-layer decoder body for ConvNeXt. Load targets
must be built with ``decoder_hidden_dim=256`` AND ``dec_layers=6`` for
ConvNeXt checkpoints (the variant specs default to 3/4 layers — the strict
load refuses otherwise); ViT checkpoints share the spec width and depth, so
``decoder_hidden_dim=None`` plus the spec ``dec_layers`` is correct there.
The user-held store's per-asset build args are returned verbatim by
``registry.user_held_assets.UserHeldAsset.required_model_args``.

Class-count-dependent tensors (``dec_score_head`` / ``enc_score_head`` /
``denoising_class_embed``) are skipped when the target model's class count
differs from the checkpoint's (COCO 80) — standard warm-start practice;
everything else must match in name AND shape or the load raises.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

from autovision.registry.downloadable import (
    ASSET_MANIFEST,
    DownloadableModelError,
    resolve_downloadable,
)
from autovision.registry.user_held_assets import (
    USER_HELD_ASSETS,
    USER_HELD_LOADABLE_ASSETS,
    USER_HELD_ROOT,
    USER_HELD_UNSUPPORTED_ASSETS,
    verify_user_held_asset,
)

__all__ = [
    "load_baseline_coco_checkpoint",
    "resolve_detector_checkpoint",
]

logger = logging.getLogger("autovision")

#: Task-module suffixes that legacy payload ``model_class_path`` values end
#: with (design doc 28, section 4.3): four generations — the baseline's
#: original task module and its ``dinov3_``/``dinov2_`` renames, plus our
#: own pre-remediation and pre-rename package layouts. The remap matches on
#: the SUFFIX rather than a full prefix, which keeps every generation
#: loadable without spelling any producer package name in this source file:
#: the payloads themselves carry the original string, and only the module
#: part is retargeted (the class name is preserved).
LEGACY_TASK_MODEL_SUFFIXES: tuple[str, ...] = (
    ".ltdetr_object_detection.task_model.",
    ".dinov3_ltdetr_object_detection.task_model.",
    ".dinov2_ltdetr_object_detection.task_model.",
)

#: Module the legacy payloads' task-model classes live in today.
CURRENT_TASK_MODEL_MODULE = "autovision.tasks.detection.detr.task_model."


def remap_legacy_class_path(class_path: str) -> str:
    """Map a checkpoint payload's ``model_class_path`` onto the current tree.

    A payload records ``<producer package prefix><task module>.<Class>``;
    the module part selects the current module while the class name is
    preserved, so the log line still names the object the payload stored.
    Paths matching no known suffix pass through untouched.
    """
    for suffix in LEGACY_TASK_MODEL_SUFFIXES:
        index = class_path.rfind(suffix)
        if index <= 0:  # need a non-empty package prefix
            continue
        class_name = class_path[index + len(suffix) :]
        if not class_name or "." in class_name:
            continue  # not a task-model class path; leave it alone
        remapped = CURRENT_TASK_MODEL_MODULE + class_name
        logger.info("Checkpoint class path remapped: %s -> %s", class_path, remapped)
        return remapped
    return class_path


#: Old ConvNeXt block name -> current HF DINOv3ConvNextModel name.
_CONVNEXT_RENAME = {
    "dwconv": "depthwise_conv",
    "norm": "layer_norm",
    "pwconv1": "pointwise_conv1",
    "pwconv2": "pointwise_conv2",
}

#: Decoder tensors whose shape depends on the class count.
_CLS_DEPENDENT = ("dec_score_head", "enc_score_head", "denoising_class_embed")

#: Raw-backbone key prefixes per checkpoint family.
_CONVNEXT_RAW_PREFIX = "backbone.backbone."
_VIT_RAW_PREFIX = "backbone.dinov3."

#: Transport key for the ViT RoPE periods (a persistent baseline buffer
#: with no HF state-dict destination; ``inv_freq`` is non-persistent on the
#: HF side, so the key cannot collide with a real state-dict entry).
_VIT_ROPE_INV_FREQ_KEY = "rope_embeddings.inv_freq"

_BLOCK_RE = re.compile(r"blocks\.(\d+)\.(.+)")


def _strip_ema(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload["train_model"]
    return {
        k[len("ema_model.model."):]: v
        for k, v in state.items()
        if k.startswith("ema_model.model.")
    }


def _map_convnext_key(key: str) -> str | None:
    """Old baseline ConvNeXt layout -> current HF layout (relative keys).

    Returns ``None`` for the ``norms.3`` alias (a duplicate reference of
    the final ``norm`` kept by the baseline model — nothing to load).
    """
    parts = key.split(".")
    if parts[0] == "norms":
        return None
    if parts[0] == "downsample_layers":
        return f"stages.{parts[1]}.downsample_layers.{'.'.join(parts[2:])}"
    if parts[0] == "norm":
        return f"layer_norm.{'.'.join(parts[1:])}"
    if parts[0] == "stages":
        stage, block, module, *rest = parts[1:]
        module = _CONVNEXT_RENAME.get(module, module)
        return f"stages.{stage}.layers.{block}.{module}.{'.'.join(rest)}".rstrip(".")
    raise KeyError(f"unexpected backbone key: {key}")


def _map_vit_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Meta DINOv3 ViT layout -> HF ``DINOv3ViTModel`` layout.

    Blocks are grouped so ``attn.qkv`` can be split jointly with its
    ``bias_mask`` gate: the baseline gates the fused bias per channel
    group (q/k/v), and HF has no k bias at all, so the masked k bias must
    be identically zero for the checkpoint to be representable.
    """
    mapped: dict[str, torch.Tensor] = {}
    blocks: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        block_match = _BLOCK_RE.match(key)
        if block_match is None:
            mapped.update(_map_vit_nonblock_key(key, value))
        else:
            blocks.setdefault(int(block_match.group(1)), {})[
                block_match.group(2)
            ] = value

    for index, tensors in sorted(blocks.items()):
        prefix = f"model.layer.{index}."
        for src, dst in (
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("attn.proj.weight", "attention.o_proj.weight"),
            ("attn.proj.bias", "attention.o_proj.bias"),
            ("ls1.gamma", "layer_scale1.lambda1"),
            ("ls2.gamma", "layer_scale2.lambda1"),
            ("mlp.fc1.weight", "mlp.up_proj.weight"),
            ("mlp.fc1.bias", "mlp.up_proj.bias"),
            ("mlp.fc2.weight", "mlp.down_proj.weight"),
            ("mlp.fc2.bias", "mlp.down_proj.bias"),
        ):
            mapped[prefix + dst] = tensors[src]
        weight = tensors["attn.qkv.weight"]
        bias = tensors["attn.qkv.bias"]
        mask = tensors.get("attn.qkv.bias_mask")
        if mask is not None:
            if mask.shape != bias.shape:
                raise ValueError(
                    f"blocks.{index}.attn.qkv: bias_mask shape "
                    f"{tuple(mask.shape)} does not gate bias "
                    f"{tuple(bias.shape)}."
                )
            bias = bias * mask
        query, key_bias, value = bias.chunk(3)
        if bool(torch.any(key_bias != 0)):
            raise ValueError(
                f"blocks.{index}.attn.qkv: masked k bias is nonzero; the HF "
                "DINOv3ViTModel attention has no k bias, so this checkpoint "
                "is not representable in the current layout."
            )
        q_weight, k_weight, v_weight = weight.chunk(3)
        mapped[prefix + "attention.q_proj.weight"] = q_weight
        mapped[prefix + "attention.k_proj.weight"] = k_weight
        mapped[prefix + "attention.v_proj.weight"] = v_weight
        mapped[prefix + "attention.q_proj.bias"] = query
        mapped[prefix + "attention.v_proj.bias"] = value
        # attn.qkv.bias_mask: functional channel gate, absorbed above.
    return mapped


def _map_vit_nonblock_key(key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Top-level (non-block) Meta ViT keys -> HF layout."""
    if key == "cls_token":
        return {"embeddings.cls_token": value}
    if key == "mask_token":
        # Baseline (1, H) vs HF (1, 1, H): add the token axis.
        return {"embeddings.mask_token": value.reshape(1, 1, -1)}
    if key == "storage_tokens":
        return {"embeddings.register_tokens": value}
    if key == "rope_embed.periods":
        # Transported, not dropped: the baseline computes its rope angles
        # as ``2*pi*coords / periods`` from THIS buffer, while HF recomputes
        # inverse frequencies from ``config.rope_theta`` — the two agree
        # only to ~1e-4 relative (measured; not a plain fp16/bf16 rounding
        # of the same formula), which drifts the block outputs far above
        # port-level float noise. Carried under the reserved ``inv_freq``
        # key and applied to the HF rope module after the strict load.
        return {_VIT_ROPE_INV_FREQ_KEY: 1.0 / value.float()}
    if key.startswith("patch_embed.proj."):
        return {f"embeddings.patch_embeddings.{key[len('patch_embed.proj.'):]}": value}
    if key.startswith("norm."):
        # The final norm stays top-level in HF (unlike ConvNeXt's
        # ``layer_norm`` rename).
        return {key: value}
    raise KeyError(f"unexpected ViT backbone key: {key}")


def _partition_baseline_state(
    state: dict[str, torch.Tensor],
) -> tuple[dict, dict, dict, dict, str]:
    """Split the baseline state into (backbone, extras, encoder, decoder,
    family).

    The raw-backbone prefix selects the family (``backbone.backbone`` =
    ConvNeXt, ``backbone.dinov3`` = ViT); every other
    ``backbone.*`` key is a wrapper extra (convs/norms for ConvNeXt;
    ``sta.*`` keys mark an unsupported STA-era ViT checkpoint) loaded by
    name into the target's detection backbone wrapper.
    """
    has_convnext = any(k.startswith(_CONVNEXT_RAW_PREFIX) for k in state)
    has_vit = any(k.startswith(_VIT_RAW_PREFIX) for k in state)
    if has_convnext and has_vit:
        raise ValueError(
            "baseline checkpoint mixes backbone-family keys "
            f"(convnext={has_convnext}, vit={has_vit})."
        )
    if not has_convnext and not has_vit:
        raise ValueError(
            "baseline checkpoint has no recognizable raw-backbone keys "
            f"(expected '{_CONVNEXT_RAW_PREFIX}*' or '{_VIT_RAW_PREFIX}*')."
        )
    family = "convnext" if has_convnext else "vit"

    backbone: dict[str, torch.Tensor] = {}
    extras: dict[str, torch.Tensor] = {}
    encoder: dict[str, torch.Tensor] = {}
    decoder: dict[str, torch.Tensor] = {}
    vit_raw: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(_CONVNEXT_RAW_PREFIX):
            mapped = _map_convnext_key(key[len(_CONVNEXT_RAW_PREFIX):])
            if mapped is not None:
                backbone[mapped] = value
        elif key.startswith(_VIT_RAW_PREFIX):
            # Collected first: the qkv split needs each block's weight,
            # bias, and bias_mask together.
            vit_raw[key[len(_VIT_RAW_PREFIX):]] = value
        elif key.startswith("backbone."):
            extras[key[len("backbone."):]] = value
        elif key.startswith("encoder."):
            encoder[key[len("encoder."):]] = value
        elif key.startswith("decoder."):
            decoder[key[len("decoder."):]] = value
        else:
            raise KeyError(f"unroutable baseline key: {key}")
    if vit_raw:
        backbone.update(_map_vit_state(vit_raw))
    return backbone, extras, encoder, decoder, family


def _apply_rope_inv_freq(hf_model: torch.nn.Module, inv_freq: torch.Tensor) -> bool:
    """Install the checkpoint's RoPE inverse frequencies on the HF rope module.

    ``inv_freq`` is a non-persistent HF buffer, so it is assigned (not
    state-dict loaded) after the strict load. Layout drift (a renamed or
    absent rope module) degrades to a log — HF's computed table is a
    ~1e-4-relative approximation of the trained one, not a different
    semantic.
    """
    rope = getattr(hf_model, "rope_embeddings", None)
    current = getattr(rope, "inv_freq", None)
    if rope is None or not isinstance(current, torch.Tensor):
        logger.info(
            "baseline checkpoint carries rope_embed.periods but the HF "
            "model exposes no rope_embeddings.inv_freq buffer; keeping the "
            "recomputed RoPE table."
        )
        return False
    # Same registered buffer name: nn.Module.__setattr__ accepts a Tensor.
    rope.inv_freq = inv_freq.to(device=current.device, dtype=current.dtype)
    return True


def _load_into_hf_model(hf_model: torch.nn.Module, mapped: dict[str, torch.Tensor],
                        family: str) -> tuple[int, bool]:
    """Strict-load remapped raw-backbone tensors into the HF module.

    Adapts to the installed transformers' layout (5.19 nests the DINOv3
    weights under a ``model.`` prefix while embeddings / the final norm
    stay top-level). Returns ``(loaded_tensor_count, rope_periods_applied)``.
    """
    rope_inv_freq = mapped.pop(_VIT_ROPE_INV_FREQ_KEY, None)
    hf_keys = set(hf_model.state_dict().keys())
    adapted = {}
    for key, value in mapped.items():
        if key in hf_keys:
            adapted[key] = value
        elif f"model.{key}" in hf_keys:
            adapted[f"model.{key}"] = value
        else:
            raise KeyError(f"{family} backbone key '{key}' not found in the HF model")
    hf_model.load_state_dict(adapted, strict=True)
    rope_periods = (
        _apply_rope_inv_freq(hf_model, rope_inv_freq)
        if rope_inv_freq is not None
        else False
    )
    return len(adapted), rope_periods


def _load_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - reported below with context
        raise CheckpointPayloadError(
            f"baseline checkpoint '{path}' did not load under the safe "
            f"pickle-free loader (weights_only=True): {error}"
        ) from error


class CheckpointPayloadError(RuntimeError):
    pass


def load_baseline_coco_checkpoint(
    task_model: torch.nn.Module, path: str | Path
) -> dict[str, int]:
    """Load a published baseline LT-DETR detector checkpoint in place.

    Loads backbone (family-remapped), wrapper extras (ConvNeXt projections
    and norms), encoder, and decoder weights. Decoder tensors whose shape
    depends on the class count are skipped when they do not match the
    target model (i.e. warm-starting a non-COCO class count); every other
    tensor must match exactly or the call raises. STA-era ViT checkpoints
    are rejected: the DEIMv2-derived spatial prior was removed.
    """
    payload = _load_payload(path)
    if isinstance(payload.get("model_class_path"), str):
        # Touch the legacy remap so stale class paths are logged/remapped
        # for callers that inspect the payload (design doc 28 §4.3).
        payload["model_class_path"] = remap_legacy_class_path(
            payload["model_class_path"]
        )
    state = _strip_ema(payload)

    # STA-era ViT checkpoints are refused before any model access: the
    # DEIMv2-derived spatial prior was removed for licensing, and the
    # no-STA adapter cannot reproduce its features. Scan the raw state before
    # partitioning so the guard does not depend on how the removed tensors
    # would have been routed.
    sta_keys = sorted(k for k in state if k.startswith("backbone.sta."))
    if sta_keys:
        raise ValueError(
            f"baseline checkpoint carries {len(sta_keys)} backbone.sta.* "
            "tensor(s) from the DEIMv2-derived spatial prior, which was "
            "removed for licensing. The no-STA ViT adapter cannot load "
            "this checkpoint faithfully; retrain/export a no-STA checkpoint "
            "or use a loadable ConvNeXt asset "
            f"({', '.join(sorted(USER_HELD_LOADABLE_ASSETS))})."
        )

    backbone_sd, extras_sd, encoder_sd, decoder_sd, family = (
        _partition_baseline_state(state)
    )

    summary: dict[str, int] = {}

    # 1) backbone — strict: the remapped set IS the HF model's key set.
    hf_model = task_model.backbone_wrapper._model
    loaded_backbone, rope_periods = _load_into_hf_model(hf_model, backbone_sd, family)
    summary["backbone"] = loaded_backbone
    if rope_periods:
        summary["backbone_rope_periods"] = 1

    # 2) wrapper extras (ConvNeXt 1x1 projections / output norms) — every
    #    key must exist in the wrapper by name (they were trained during
    #    the COCO run); the rest of the wrapper (the raw HF model) was
    #    loaded in step 1.
    if extras_sd:
        wrapper = task_model.backbone
        wrapper_keys = set(wrapper.state_dict().keys())
        unknown = sorted(set(extras_sd) - wrapper_keys)
        if unknown:
            raise ValueError(
                f"baseline checkpoint carries wrapper extras that the "
                f"target backbone ({type(wrapper).__name__}) does not "
                f"define (first keys: {unknown[:5]}) — build the model "
                "with the checkpoint's wrapper recipe."
            )
        wrapper.load_state_dict(extras_sd, strict=False)
        summary["backbone_extras"] = len(extras_sd)
    summary["backbone_extras"] = summary.get("backbone_extras", 0)

    # 3) encoder — strict: the baseline encoder is operator-identical and
    #    name-identical; the target must be built at the checkpoint's
    #    encoder width.
    task_model.encoder.load_state_dict(encoder_sd, strict=True)
    summary["encoder"] = len(encoder_sd)

    # 4) decoder — class-count heads may be skipped; the rest is strict.
    own = task_model.decoder.state_dict()
    skipped: list[str] = []
    incoming: dict[str, torch.Tensor] = {}

    def _is_cls_dependent(key: str) -> bool:
        return any(key.startswith(pattern) for pattern in _CLS_DEPENDENT)

    def _is_ov_fresh(key: str) -> bool:
        """Own-side tensors legitimately absent from a baseline checkpoint:
        the renamed contrastive heads, the zero-init text cross-attention
        sublayers (bitwise identity until training moves them — the
        warm-start contract, design doc 27 §3.6) and the prompt-encoded
        text buffer (rebuilt from the run's vocabulary)."""
        return (
            _is_cls_dependent(key)
            or ".text_attn." in key
            or key.endswith("ov_text_embeds")
        )

    for key, value in decoder_sd.items():
        if key not in own:
            if _is_cls_dependent(key):
                # The open-vocab branch renames the score-head tensors
                # (proj/logit_scale/class_bias — design doc 27 §4): a
                # baseline Linear-head tensor has no destination and is
                # skipped like any other class-count-dependent head.
                skipped.append(key)
                continue
            raise ValueError(
                f"decoder key '{key}' from the checkpoint is unknown to "
                f"the target model."
            )
        if own[key].shape != value.shape:
            if _is_cls_dependent(key) or key.endswith("ov_text_embeds"):
                # A shape-mismatched text buffer (cross-vocabulary warm
                # start) keeps the values the constructor just encoded for
                # the TARGET vocabulary (design doc 27 §4).
                skipped.append(key)
                continue
            raise ValueError(
                f"decoder tensor '{key}' shape mismatch: checkpoint "
                f"{tuple(value.shape)} vs model {tuple(own[key].shape)} — "
                "build the model with the checkpoint's decoder width "
                "(decoder_hidden_dim=256)."
            )
        incoming[key] = value
    incompatible = task_model.decoder.load_state_dict(incoming, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected decoder keys from checkpoint: "
            f"{incompatible.unexpected_keys[:5]}"
        )
    # Missing own keys that are class-count-dependent (fresh-init heads —
    # under the open-vocab branch the contrastive head tensors are always
    # fresh), zero-init text-attention sublayers, or the prompt-encoded
    # text buffer are tolerated by design.
    tolerated_missing = {key for key in incompatible.missing_keys if _is_ov_fresh(key)}
    if set(incompatible.missing_keys) - set(skipped) - tolerated_missing:
        raise ValueError(
            f"decoder keys missing after load (outside the skipped set): "
            f"{sorted(set(incompatible.missing_keys) - set(skipped) - tolerated_missing)[:5]}"
        )
    summary["decoder"] = len(incoming)
    summary["decoder_skipped_cls"] = len(skipped)
    if skipped:
        logger.info(
            "baseline checkpoint: skipped %d class-count-dependent decoder "
            "tensors (fresh init): %s...",
            len(skipped), skipped[:3],
        )
    return summary


def resolve_detector_checkpoint(source: str | Path) -> Path:
    """Resolve a ``detector_checkpoint`` argument to a verified local file.

    An existing file passes through untouched; a registered user-held asset
    name (``registry/user_held_assets.py``, policy D1-a — e.g.
    ``"dinov3-vitt16"``) resolves to its store ``weights.pt``; otherwise the
    value is looked up in the downloadable registry (registered asset names
    like ``d-fine/dfine_s_coco`` download + sha256-verify into
    ``AUTOVISION_CACHE_DIR``). Anything else is an error naming both
    interpretations.
    """
    path = Path(source)
    if path.is_file():
        return path
    name = str(source)
    if name in ASSET_MANIFEST:
        return resolve_downloadable(name)
    user_held = USER_HELD_ASSETS.get(name)
    if user_held is not None:
        reason = user_held.unsupported_reason
        if reason is not None:
            raise DownloadableModelError(reason)
        weights = user_held.weights_path
        if not weights.is_file():
            raise DownloadableModelError(
                f"user-held detector checkpoint '{name}' is registered but "
                f"not present at '{weights}' (store root {USER_HELD_ROOT}; "
                f"AUTOVISION_USER_HELD_ROOT overrides)."
            )
        # Identity is pinned in the registry; verify before use. The digest
        # is cached per (path, size, mtime), so repeated resolves are cheap.
        verify_user_held_asset(user_held, files=("weights.pt",))
        return weights
    raise DownloadableModelError(
        f"detector_checkpoint '{source}' is neither an existing file nor a "
        f"loadable checkpoint name; loadable names: "
        f"{sorted(ASSET_MANIFEST) + sorted(USER_HELD_LOADABLE_ASSETS)}; "
        f"registered but unsupported: "
        f"{sorted(USER_HELD_UNSUPPORTED_ASSETS)}"
    )
