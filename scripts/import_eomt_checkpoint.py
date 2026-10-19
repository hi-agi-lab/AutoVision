#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Import third-party DINOv3 EoMT training checkpoints into AutoVision.

The source checkpoints use a reference training package's layout::

    {"model_class_path": "...",
     "model_init_args": {...},
     "train_model": {"model.backbone....": tensor, ...}}

``train_model`` is a flat state dict with fused ``attn.qkv`` projections and
the reference package's module names. AutoVision's DINOv3 task models wrap
the HF ``EomtDinov3ForUniversalSegmentation`` layout (separate q/k/v
projections, ``layers.N``, ``mask_head.fc1..3``), so the weights cannot be
loaded directly. This script performs the deterministic remap, transports the
trained RoPE periods through the reserved state-dict key, sets the inference
attention-mask probability, and writes a standard AutoVision export loadable
with :func:`autovision.load_task_model`.

Usage::

    python scripts/import_eomt_checkpoint.py \
        --checkpoint path/to/checkpoint.pt \
        --out path/to/export.pt

The source class/ID table is written next to the export as ``<out>.json``
(AutoVision task models store class counts, not names). Pass
``--layer-norm-eps 1e-5`` for checkpoints trained with the reference
``layernormbf16`` backbone (the published DINOv3 EoMT family), and
``--attn-mask-prob 0`` for inference parity (the reference package disables
the joint-block attention mask during evaluation).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autovision import load_task_model  # noqa: E402
from autovision.tasks.panoptic_segmentation.dinov3_eomt import (  # noqa: E402
    Dinov3EomtPanopticSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt import (  # noqa: E402
    Dinov3EomtSemanticSegmentation,
)
from autovision.tasks.shared.eomt.modeling import ROPE_PERIODS_KEY  # noqa: E402

_SEMANTIC_CLASS_SUFFIX = "dinov3_eomt_semantic_segmentation"
_PANOPTIC_CLASS_SUFFIX = "dinov3_eomt_panoptic_segmentation"

# The task models normalize with these exact ImageNet statistics; a source
# checkpoint with a different normalization cannot be represented.
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)

_MASK_HEAD_NAMES = {"0": "fc1", "2": "fc2", "4": "fc3"}
_BLOCK_SIMPLE_KEYS = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "attn.proj.weight": "attention.o_proj.weight",
    "attn.proj.bias": "attention.o_proj.bias",
    "ls1.gamma": "layer_scale1.lambda1",
    "ls2.gamma": "layer_scale2.lambda1",
    "mlp.fc1.weight": "mlp.up_proj.weight",
    "mlp.fc1.bias": "mlp.up_proj.bias",
    "mlp.fc2.weight": "mlp.down_proj.weight",
    "mlp.fc2.bias": "mlp.down_proj.bias",
}
_QKV_KEYS = {
    "attn.qkv.weight": "weight",
    "attn.qkv.bias": "bias",
    "attn.qkv.bias_mask": "bias_mask",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_default_permissions(path: Path) -> None:
    """Give the final export normal umask permissions.

    ``tempfile.NamedTemporaryFile`` creates 0600; ``os.replace`` preserves
    that mode, which would leave a converted checkpoint unreadable to other
    users. Best-effort: platforms without POSIX modes keep the temp mode.
    """
    try:
        umask = os.umask(0)
        os.umask(umask)
        path.chmod(0o666 & ~umask)
    except OSError:
        pass


def _class_entries(classes: Any, *, label: str) -> list[dict[str, Any]]:
    """Validate a source ``{id: name}`` table and preserve its order."""
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"{label} must be a non-empty dict, got {type(classes)}")
    entries: list[dict[str, Any]] = []
    for raw_id, name in classes.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} has a non-integer class id {raw_id!r}"
            ) from error
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}[{class_id}] must be a non-empty name")
        entries.append({"id": class_id, "name": name})
    return entries


def _detect_task(checkpoint: dict[str, Any]) -> str:
    init_args = checkpoint.get("model_init_args")
    if not isinstance(init_args, dict):
        raise ValueError("checkpoint has no model_init_args dict")
    has_semantic = "classes" in init_args
    has_panoptic = "thing_classes" in init_args or "stuff_classes" in init_args
    if has_semantic == has_panoptic:
        raise ValueError(
            "checkpoint must carry either 'classes' (semantic) or "
            "'thing_classes' + 'stuff_classes' (panoptic)"
        )
    task = "semantic" if has_semantic else "panoptic"
    class_path = str(checkpoint.get("model_class_path", "")).lower()
    suffix = (
        _SEMANTIC_CLASS_SUFFIX if task == "semantic" else _PANOPTIC_CLASS_SUFFIX
    )
    if suffix not in class_path:
        raise ValueError(
            f"checkpoint class path {class_path!r} is not a DINOv3 EoMT "
            f"{task} model (expected '{suffix}')"
        )
    return task


def _validate_normalize(init_args: dict[str, Any]) -> None:
    normalize = init_args.get("image_normalize")
    if not isinstance(normalize, dict):
        raise ValueError("checkpoint has no image_normalize dict")
    try:
        mean = tuple(float(value) for value in normalize["mean"])
        std = tuple(float(value) for value in normalize["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "image_normalize must contain numeric 'mean' and 'std' sequences"
        ) from error
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("image_normalize must contain three mean/std values")
    if any(abs(a - b) > 1e-6 for a, b in zip(mean, _IMAGE_MEAN)) or any(
        abs(a - b) > 1e-6 for a, b in zip(std, _IMAGE_STD)
    ):
        raise ValueError(
            "checkpoint normalization differs from the AutoVision EoMT task "
            f"model's ImageNet statistics: mean={mean}, std={std}"
        )


def _require_tensor(state: dict[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint state entry {key!r} must be a tensor")
    return value


def _derive_layout(checkpoint: dict[str, Any], task: str) -> dict[str, Any]:
    """Derive the task-model construction args from the checkpoint shapes."""
    init_args = checkpoint["model_init_args"]
    state = checkpoint.get("train_model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no train_model state dict")

    required = (
        "model.backbone.patch_embed.proj.weight",
        "model.backbone.storage_tokens",
        "model.backbone.blocks.0.mlp.fc1.weight",
        "model.backbone.rope_embed.periods",
        "model.queries.weight",
        "model.class_head.weight",
        "model.class_head.bias",
        "criterion.empty_weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint state dict is missing {missing}")

    patch_weight = state["model.backbone.patch_embed.proj.weight"]
    if (
        not isinstance(patch_weight, torch.Tensor)
        or patch_weight.ndim != 4
        or patch_weight.shape[1] != 3
        or patch_weight.shape[2] != patch_weight.shape[3]
    ):
        raise ValueError(
            "patch_embed.proj.weight must be (hidden, 3, patch, patch)"
        )
    hidden_size = int(patch_weight.shape[0])
    patch_size = int(patch_weight.shape[2])

    block_ids = sorted(
        {
            int(key.split(".")[3])
            for key in state
            if key.startswith("model.backbone.blocks.")
        }
    )
    if block_ids != list(range(len(block_ids))) or not block_ids:
        raise ValueError(f"backbone blocks are not contiguous from 0: {block_ids}")
    num_hidden_layers = len(block_ids)

    periods = state["model.backbone.rope_embed.periods"]
    if (
        not isinstance(periods, torch.Tensor)
        or periods.ndim != 1
        or periods.numel() == 0
        or not bool(torch.isfinite(periods).all())
        or bool(torch.any(periods <= 0))
    ):
        raise ValueError("rope_embed.periods must be a finite positive 1-D tensor")
    head_dim = int(periods.numel()) * 4
    if hidden_size % head_dim:
        raise ValueError(
            f"hidden_size {hidden_size} is not divisible by head_dim {head_dim} "
            "derived from the RoPE periods"
        )
    num_attention_heads = hidden_size // head_dim

    upscale_ids = sorted(
        {
            int(key.split(".")[2])
            for key in state
            if key.startswith("model.upscale.")
        }
    )
    if upscale_ids != list(range(len(upscale_ids))) or not upscale_ids:
        raise ValueError(f"upscale blocks are not contiguous from 0: {upscale_ids}")
    num_upscale_blocks = len(upscale_ids)

    intermediate_size = int(
        _require_tensor(
            state, "model.backbone.blocks.0.mlp.fc1.weight"
        ).shape[0]
    )
    if intermediate_size % hidden_size:
        raise ValueError(
            f"intermediate_size {intermediate_size} is not a multiple of "
            f"hidden_size {hidden_size}"
        )
    mlp_ratio = intermediate_size // hidden_size
    if any(".mlp.gate_proj." in key for key in state):
        raise ValueError("gated MLP checkpoints are not supported by this importer")

    try:
        num_queries = int(init_args["num_queries"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model_init_args.num_queries must be an integer") from error
    queries = _require_tensor(state, "model.queries.weight")
    if queries.shape != (num_queries, hidden_size):
        raise ValueError(
            f"queries.weight shape {tuple(queries.shape)} does not match "
            f"num_queries={num_queries}, hidden_size={hidden_size}"
        )

    try:
        num_blocks = int(init_args["num_joint_blocks"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "model_init_args.num_joint_blocks must be an integer"
        ) from error
    if not 1 <= num_blocks <= num_hidden_layers:
        raise ValueError(
            f"num_joint_blocks={num_blocks} must be in "
            f"[1, {num_hidden_layers}]"
        )

    image_size = init_args.get("image_size")
    if image_size is None:
        raise ValueError("model_init_args.image_size is required")
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError(f"image_size must have two values, got {image_size}")
        try:
            width, height = int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"image_size must be numeric, got {image_size}") from error
        if width != height:
            raise ValueError(
                "AutoVision EoMT task models require square inputs, got "
                f"image_size={image_size}"
            )
    else:
        try:
            width = height = int(image_size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"image_size must be numeric, got {image_size}") from error
    if width <= 0 or width % patch_size:
        raise ValueError(
            f"image_size={width} must be positive and divisible by "
            f"patch_size={patch_size}"
        )

    storage = state["model.backbone.storage_tokens"]
    if (
        not isinstance(storage, torch.Tensor)
        or storage.shape != (1, 4, hidden_size)
    ):
        raise ValueError(
            "checkpoint must carry exactly four storage tokens with shape "
            f"(1, 4, {hidden_size}); got "
            f"{tuple(storage.shape) if isinstance(storage, torch.Tensor) else storage}"
        )

    if task == "semantic":
        classes = _class_entries(init_args.get("classes"), label="classes")
        num_classes = len(classes)
        num_thing_classes = 0
        thing_classes: list[dict[str, Any]] = []
        stuff_classes: list[dict[str, Any]] = []
    else:
        thing_classes = _class_entries(
            init_args.get("thing_classes"), label="thing_classes"
        )
        stuff_classes = _class_entries(
            init_args.get("stuff_classes"), label="stuff_classes"
        )
        classes = []
        num_classes = len(thing_classes) + len(stuff_classes)
        num_thing_classes = len(thing_classes)

    class_weight = _require_tensor(state, "model.class_head.weight")
    class_bias = _require_tensor(state, "model.class_head.bias")
    if class_weight.shape != (num_classes + 1, hidden_size):
        raise ValueError(
            f"class_head.weight shape {tuple(class_weight.shape)} does not match "
            f"num_classes={num_classes} (+ null class), hidden={hidden_size}"
        )
    if class_bias.shape != (num_classes + 1,):
        raise ValueError(
            f"class_head.bias shape {tuple(class_bias.shape)} does not match "
            f"num_classes={num_classes} (+ null class)"
        )
    empty_weight = _require_tensor(state, "criterion.empty_weight")
    if empty_weight.numel() != num_classes + 1:
        raise ValueError(
            f"criterion.empty_weight has {empty_weight.numel()} entries, "
            f"expected {num_classes + 1}"
        )

    _validate_normalize(init_args)

    return {
        "task": task,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_queries": num_queries,
        "num_blocks": num_blocks,
        "num_upscale_blocks": num_upscale_blocks,
        "image_size": width,
        "patch_size": patch_size,
        "mlp_ratio": mlp_ratio,
        "num_classes": num_classes,
        "num_thing_classes": num_thing_classes,
        "classes": classes,
        "thing_classes": thing_classes,
        "stuff_classes": stuff_classes,
        "rope_periods": periods,
        "source_model_name": str(init_args.get("model_name", "")),
    }


def _map_state(
    state: dict[str, Any],
    layout: dict[str, Any],
    *,
    attn_mask_prob: float,
) -> dict[str, torch.Tensor]:
    """Remap the source state dict onto the HF EoMT task-model layout."""
    hidden_size = layout["hidden_size"]
    num_hidden_layers = layout["num_hidden_layers"]
    num_upscale_blocks = layout["num_upscale_blocks"]
    mapped: dict[str, torch.Tensor] = {}
    qkv: dict[tuple[int, str], torch.Tensor] = {}
    rope_periods: torch.Tensor | None = None

    def put(key: str, tensor: torch.Tensor) -> None:
        if key in mapped:
            raise ValueError(f"duplicate mapped key: {key}")
        mapped[key] = tensor

    def block_index(raw: str, *, what: str) -> int:
        try:
            index = int(raw)
        except ValueError as error:
            raise ValueError(f"invalid {what} block index {raw!r}") from error
        if not 0 <= index < num_hidden_layers:
            raise ValueError(f"{what} block index {index} out of range")
        return index

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state entry {key} is not a tensor")
        if key == "criterion.empty_weight":
            put("model.criterion.empty_weight", value)
            continue

        if key.startswith("model.mask_head."):
            raw_index, rest = key[len("model.mask_head.") :].split(".", 1)
            target = _MASK_HEAD_NAMES.get(raw_index)
            if target is None:
                raise ValueError(f"unexpected mask_head index {raw_index!r}")
            put(f"model.mask_head.{target}.{rest}", value)
            continue

        if key.startswith("model.queries."):
            put("model.query." + key[len("model.queries.") :], value)
            continue

        if key.startswith("model.class_head."):
            put("model.class_predictor." + key[len("model.class_head.") :], value)
            continue

        if key.startswith("model.upscale."):
            raw_index, rest = key[len("model.upscale.") :].split(".", 1)
            try:
                index = int(raw_index)
            except ValueError as error:
                raise ValueError(f"invalid upscale block index {raw_index!r}") from error
            if not 0 <= index < num_upscale_blocks:
                raise ValueError(f"upscale block index {index} out of range")
            rest = rest.replace("norm.", "layernorm2d.", 1)
            put(f"model.upscale_block.block.{index}.{rest}", value)
            continue

        if not key.startswith("model.backbone."):
            raise ValueError(f"unmapped state key: {key}")
        rest = key[len("model.backbone.") :]

        if rest == "cls_token":
            put("model.embeddings.cls_token", value)
        elif rest == "storage_tokens":
            put("model.embeddings.register_tokens", value)
        elif rest == "mask_token":
            # HF EoMT registers the mask token as None and never uses it.
            pass
        elif rest.startswith("patch_embed.proj."):
            put(
                "model.embeddings.patch_embeddings."
                + rest[len("patch_embed.proj.") :],
                value,
            )
        elif rest == "rope_embed.periods":
            if rope_periods is not None:
                raise ValueError("duplicate rope_embed.periods")
            rope_periods = value
        elif rest.startswith("norm."):
            put("model.layernorm." + rest[len("norm.") :], value)
        elif rest.startswith("blocks."):
            raw_index, sub = rest[len("blocks.") :].split(".", 1)
            index = block_index(raw_index, what="backbone")
            if sub in _BLOCK_SIMPLE_KEYS:
                put(
                    f"model.layers.{index}.{_BLOCK_SIMPLE_KEYS[sub]}",
                    value,
                )
            elif sub in _QKV_KEYS:
                kind = _QKV_KEYS[sub]
                if (index, kind) in qkv:
                    raise ValueError(
                        f"duplicate qkv {kind} for block {index}"
                    )
                qkv[(index, kind)] = value
            else:
                raise ValueError(f"unmapped backbone block key: {key}")
        else:
            raise ValueError(f"unmapped backbone key: {key}")

    if rope_periods is None:
        raise ValueError("checkpoint has no rope_embed.periods tensor")

    qkv_blocks = sorted({index for index, _ in qkv})
    if qkv_blocks != list(range(num_hidden_layers)):
        raise ValueError(
            f"attn.qkv tensors found for blocks {qkv_blocks}, expected "
            f"0..{num_hidden_layers - 1}"
        )
    for index in qkv_blocks:
        weight = qkv.get((index, "weight"))
        bias = qkv.get((index, "bias"))
        mask = qkv.get((index, "bias_mask"))
        if weight is None or bias is None:
            raise ValueError(f"block {index} is missing attn.qkv weight/bias")
        if weight.shape != (3 * hidden_size, hidden_size):
            raise ValueError(
                f"block {index} qkv.weight shape {tuple(weight.shape)} does not "
                f"match hidden_size={hidden_size}"
            )
        if bias.shape != (3 * hidden_size,):
            raise ValueError(
                f"block {index} qkv.bias shape {tuple(bias.shape)} does not "
                f"match hidden_size={hidden_size}"
            )
        masked_bias = bias
        if mask is not None:
            if mask.shape != bias.shape:
                raise ValueError(
                    f"block {index} qkv.bias_mask shape {tuple(mask.shape)} "
                    f"does not match bias {tuple(bias.shape)}"
                )
            if not bool(torch.isfinite(mask).all()):
                raise ValueError(f"block {index} qkv.bias_mask is not finite")
            masked_bias = bias * mask.to(device=bias.device, dtype=bias.dtype)
        if not bool(torch.isfinite(masked_bias).all()):
            raise ValueError(f"block {index} masked qkv bias is not finite")
        query_bias, key_bias, value_bias = masked_bias.chunk(3)
        if bool(torch.any(key_bias != 0)):
            raise ValueError(
                f"block {index} has a nonzero key bias; the HF EoMT attention "
                "has no key bias, so this checkpoint is not representable"
            )
        query_weight, key_weight, value_weight = weight.chunk(3)
        prefix = f"model.layers.{index}.attention."
        put(prefix + "q_proj.weight", query_weight)
        put(prefix + "k_proj.weight", key_weight)
        put(prefix + "v_proj.weight", value_weight)
        put(prefix + "q_proj.bias", query_bias)
        put(prefix + "v_proj.bias", value_bias)

    put(
        "model.attn_mask_probs",
        torch.full(
            (layout["num_blocks"],), float(attn_mask_prob), dtype=torch.float32
        ),
    )
    put(ROPE_PERIODS_KEY, rope_periods.float())
    return mapped


def _task_config(layout: dict[str, Any], *, layer_norm_eps: float) -> dict[str, Any]:
    config: dict[str, Any] = {
        "num_classes": layout["num_classes"],
        "image_size": layout["image_size"],
        "num_queries": layout["num_queries"],
        "num_blocks": layout["num_blocks"],
        "num_upscale_blocks": layout["num_upscale_blocks"],
        "hidden_size": layout["hidden_size"],
        "num_hidden_layers": layout["num_hidden_layers"],
        "num_attention_heads": layout["num_attention_heads"],
        "patch_size": layout["patch_size"],
        "mlp_ratio": layout["mlp_ratio"],
        "use_gated_mlp": False,
        "drop_path_rate": 0.0,
        "layer_norm_eps": layer_norm_eps,
    }
    if layout["task"] == "panoptic":
        # Inference thresholds are not stored in the checkpoint; use the
        # task model's documented defaults.
        config.update(
            num_thing_classes=layout["num_thing_classes"],
            threshold=0.8,
            mask_threshold=0.5,
            mask_overlap_threshold=0.8,
        )
    return config


def _build_task(layout: dict[str, Any], config: dict[str, Any]) -> Any:
    if layout["task"] == "semantic":
        return Dinov3EomtSemanticSegmentation(**config)
    return Dinov3EomtPanopticSegmentation(**config)


def _verify_export(
    export_path: Path,
    layout: dict[str, Any],
    *,
    mapped: dict[str, torch.Tensor],
    layer_norm_eps: float,
    attn_mask_prob: float,
    forward_check: bool,
) -> Any:
    expected_cls = (
        Dinov3EomtSemanticSegmentation
        if layout["task"] == "semantic"
        else Dinov3EomtPanopticSegmentation
    )
    model = load_task_model(export_path, device="cpu")
    if not isinstance(model, expected_cls):
        raise ValueError(
            f"reloaded model is {type(model).__name__}, expected "
            f"{expected_cls.__name__}"
        )
    hf_model = model.model
    norm_modules = [
        hf_model.layernorm,
        hf_model.layers[0].norm1,
        hf_model.layers[0].norm2,
    ]
    if len(hf_model.upscale_block.block):
        norm_modules.append(hf_model.upscale_block.block[0].layernorm2d)
    for module in norm_modules:
        if abs(float(module.eps) - layer_norm_eps) > 1e-12:
            raise ValueError(
                f"LayerNorm eps {module.eps} does not match {layer_norm_eps}"
            )

    periods = layout["rope_periods"].float()
    expected_inv = periods.reciprocal()
    for name in ("inv_freq", "original_inv_freq"):
        buffer = getattr(hf_model.rope_embeddings, name, None)
        if not isinstance(buffer, torch.Tensor):
            raise ValueError(f"reloaded model has no rope_embeddings.{name}")
        torch.testing.assert_close(
            buffer.detach().cpu(), expected_inv, rtol=1e-6, atol=1e-8
        )
    torch.testing.assert_close(
        hf_model.attn_mask_probs.detach().cpu(),
        torch.full((layout["num_blocks"],), float(attn_mask_prob)),
        rtol=0.0,
        atol=0.0,
    )

    # Strict loading proves names/shapes; sample the remapped values too so a
    # swapped/misrouted tensor cannot hide behind identical shapes.
    hf_state = hf_model.state_dict()
    for key in (
        "model.embeddings.patch_embeddings.weight",
        "model.layernorm.weight",
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.mask_head.fc1.weight",
        "model.upscale_block.block.0.conv1.weight",
        "model.query.weight",
        "model.class_predictor.weight",
        "model.criterion.empty_weight",
    ):
        if key not in mapped:
            raise ValueError(f"mapped checkpoint is missing {key}")
        hf_key = key[len("model.") :]
        if hf_key not in hf_state:
            raise ValueError(f"reloaded model is missing {hf_key}")
        torch.testing.assert_close(
            hf_state[hf_key].detach().cpu(),
            mapped[key].detach().float().cpu(),
            rtol=0.0,
            atol=0.0,
        )

    if forward_check:
        size = layout["image_size"]
        with torch.inference_mode():
            raw = model.forward_backend(
                model.preprocess_batch(torch.randn(1, 3, size, size))
            )
        masks, logits = raw
        if not bool(torch.isfinite(masks).all()) or not bool(
            torch.isfinite(logits).all()
        ):
            raise ValueError("reloaded model produced non-finite logits")
    return model


def _metadata(
    checkpoint_path: Path,
    source_sha256: str,
    layout: dict[str, Any],
    *,
    layer_norm_eps: float,
    attn_mask_prob: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": {
            "path": str(checkpoint_path.resolve()),
            "sha256": source_sha256,
            "model_name": layout["source_model_name"],
        },
        "task": layout["task"],
        "num_classes": layout["num_classes"],
        "image_size": layout["image_size"],
        "layer_norm_eps": layer_norm_eps,
        "attn_mask_prob": attn_mask_prob,
        "internal_class_order": "source dict insertion order",
    }
    if layout["task"] == "semantic":
        metadata["classes"] = layout["classes"]
    else:
        metadata["num_thing_classes"] = layout["num_thing_classes"]
        metadata["thing_classes"] = layout["thing_classes"]
        metadata["stuff_classes"] = layout["stuff_classes"]
    return metadata


def _run(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint)
    out = Path(args.out)
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    if out.suffix.lower() != ".pt":
        raise ValueError("--out must end with .pt")
    if not args.dry_run and out.exists() and not args.overwrite:
        raise ValueError(f"output already exists (pass --overwrite): {out}")
    if not math.isfinite(args.layer_norm_eps) or args.layer_norm_eps <= 0:
        raise ValueError("--layer-norm-eps must be a finite positive number")
    if not 0.0 <= args.attn_mask_prob <= 1.0:
        raise ValueError("--attn-mask-prob must be in [0, 1]")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a dict")
    task = _detect_task(checkpoint)
    layout = _derive_layout(checkpoint, task)
    mapped = _map_state(
        checkpoint["train_model"], layout, attn_mask_prob=args.attn_mask_prob
    )

    if args.dry_run:
        preview = {
            "checkpoint": str(checkpoint_path),
            "out": str(out),
            "task": task,
            "config": _task_config(layout, layer_norm_eps=args.layer_norm_eps),
            "num_mapped_keys": len(mapped),
            "num_classes": layout["num_classes"],
            "num_thing_classes": layout["num_thing_classes"],
            "class_order": (
                layout["classes"] if task == "semantic" else {
                    "thing": layout["thing_classes"],
                    "stuff": layout["stuff_classes"],
                }
            ),
        }
        print(json.dumps(preview, indent=2))
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    config = _task_config(layout, layer_norm_eps=args.layer_norm_eps)
    model = _build_task(layout, config)
    model.load_train_state_dict(mapped)
    model.eval()
    num_mapped_keys = len(mapped)
    payload = {
        "model_class_path": model.class_path,
        "model_init_args": model.init_args,
        "train_model": mapped,
    }

    source_sha256 = _sha256(checkpoint_path)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=out.name + ".", suffix=".tmp", dir=out.parent, delete=False
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        torch.save(payload, temp_path)
        del model, payload, checkpoint
        gc.collect()
        if not args.no_verify:
            loaded = _verify_export(
                temp_path,
                layout,
                mapped=mapped,
                layer_norm_eps=args.layer_norm_eps,
                attn_mask_prob=args.attn_mask_prob,
                forward_check=args.verify_forward,
            )
            del loaded
            gc.collect()
        del mapped
        gc.collect()
        os.replace(temp_path, out)
        _set_default_permissions(out)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            _metadata(
                checkpoint_path,
                source_sha256,
                layout,
                layer_norm_eps=args.layer_norm_eps,
                attn_mask_prob=args.attn_mask_prob,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "metadata": str(metadata_path),
                "source_sha256": source_sha256,
                "task": task,
                "num_mapped_keys": num_mapped_keys,
                "verified": not args.no_verify,
            },
            indent=2,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="third-party DINOv3 EoMT checkpoint (.pt)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output AutoVision export (.pt); class metadata goes to <out>.json",
    )
    parser.add_argument(
        "--layer-norm-eps",
        type=float,
        default=1e-5,
        help=(
            "LayerNorm epsilon used by the source backbone (default 1e-5, "
            "the published DINOv3 EoMT family)"
        ),
    )
    parser.add_argument(
        "--attn-mask-prob",
        type=float,
        default=0.0,
        help=(
            "joint-block attention-mask probability stored in the export "
            "(0 for inference parity, 1 for the training-time mask)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing export",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the strict reload + eps/RoPE/mask checks",
    )
    parser.add_argument(
        "--verify-forward",
        action="store_true",
        help="also run one forward pass at the checkpoint resolution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the derived layout without writing anything",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
