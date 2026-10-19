# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Third-party DINOv3 EoMT checkpoint importer.

The tests build a tiny source-layout checkpoint (fused qkv, reference module
names) and exercise the CLI end to end: remap, strict reload, eps/RoPE/mask
transport and forward. The 11 published files are validated separately by the
release-time converter run; these fixtures keep CI hermetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from autovision import load_task_model

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "import_eomt_checkpoint.py"

HIDDEN = 32
LAYERS = 4
HEADS = 4
QUERIES = 6
JOINT_BLOCKS = 2
UPSCALE_BLOCKS = 1
PATCH = 16
IMAGE = 64
NUM_CLASSES = 3
PERIODS = (1.0, 2.0)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _source_state(*, mlp_ratio: int = 4, storage_tokens: int = 4) -> dict[str, torch.Tensor]:
    intermediate = mlp_ratio * HIDDEN
    state: dict[str, torch.Tensor] = {
        "model.backbone.cls_token": torch.randn(1, 1, HIDDEN),
        "model.backbone.storage_tokens": torch.randn(1, storage_tokens, HIDDEN),
        "model.backbone.mask_token": torch.randn(1, HIDDEN),
        "model.backbone.patch_embed.proj.weight": torch.randn(
            HIDDEN, 3, PATCH, PATCH
        ),
        "model.backbone.patch_embed.proj.bias": torch.randn(HIDDEN),
        "model.backbone.rope_embed.periods": torch.tensor(PERIODS),
    }
    for index in range(LAYERS):
        prefix = f"model.backbone.blocks.{index}."
        state[prefix + "norm1.weight"] = torch.randn(HIDDEN)
        state[prefix + "norm1.bias"] = torch.randn(HIDDEN)
        state[prefix + "attn.qkv.weight"] = torch.randn(3 * HIDDEN, HIDDEN)
        # The reference model gates the fused bias per q/k/v group; k has no
        # bias in the HF attention, so its masked slice must be zero.
        state[prefix + "attn.qkv.bias"] = torch.cat(
            [torch.randn(HIDDEN), torch.zeros(HIDDEN), torch.randn(HIDDEN)]
        )
        state[prefix + "attn.qkv.bias_mask"] = torch.cat(
            [torch.ones(HIDDEN), torch.zeros(HIDDEN), torch.ones(HIDDEN)]
        )
        state[prefix + "attn.proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state[prefix + "attn.proj.bias"] = torch.randn(HIDDEN)
        state[prefix + "ls1.gamma"] = torch.randn(HIDDEN)
        state[prefix + "norm2.weight"] = torch.randn(HIDDEN)
        state[prefix + "norm2.bias"] = torch.randn(HIDDEN)
        state[prefix + "mlp.fc1.weight"] = torch.randn(intermediate, HIDDEN)
        state[prefix + "mlp.fc1.bias"] = torch.randn(intermediate)
        state[prefix + "mlp.fc2.weight"] = torch.randn(HIDDEN, intermediate)
        state[prefix + "mlp.fc2.bias"] = torch.randn(HIDDEN)
        state[prefix + "ls2.gamma"] = torch.randn(HIDDEN)
    state["model.backbone.norm.weight"] = torch.randn(HIDDEN)
    state["model.backbone.norm.bias"] = torch.randn(HIDDEN)
    state["model.queries.weight"] = torch.randn(QUERIES, HIDDEN)
    state["model.class_head.weight"] = torch.randn(NUM_CLASSES + 1, HIDDEN)
    state["model.class_head.bias"] = torch.randn(NUM_CLASSES + 1)
    for index in (0, 2, 4):
        state[f"model.mask_head.{index}.weight"] = torch.randn(HIDDEN, HIDDEN)
        state[f"model.mask_head.{index}.bias"] = torch.randn(HIDDEN)
    state["model.upscale.0.conv1.weight"] = torch.randn(HIDDEN, HIDDEN, 2, 2)
    state["model.upscale.0.conv1.bias"] = torch.randn(HIDDEN)
    state["model.upscale.0.conv2.weight"] = torch.randn(HIDDEN, 1, 3, 3)
    state["model.upscale.0.norm.weight"] = torch.randn(HIDDEN)
    state["model.upscale.0.norm.bias"] = torch.randn(HIDDEN)
    state["criterion.empty_weight"] = torch.ones(NUM_CLASSES + 1)
    return state


def _write_checkpoint(tmp_path: Path, task: str = "semantic") -> Path:
    torch.manual_seed(0)
    init_args: dict[str, Any] = {
        "model_name": "dinov3/_vittest16-eomt",
        "image_size": [IMAGE, IMAGE],
        "image_normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "num_queries": QUERIES,
        "num_joint_blocks": JOINT_BLOCKS,
        "backbone_args": None,
    }
    if task == "semantic":
        init_args.update(
            classes={1: "a", 2: "b", 3: "c"},
            class_ignore_index=-100,
        )
        class_path = (
            "reference._task_models.dinov3_eomt_semantic_segmentation."
            "task_model.DINOv3EoMTSemanticSegmentation"
        )
    else:
        init_args.update(
            thing_classes={1: "a", 2: "b"},
            stuff_classes={5: "c"},
        )
        class_path = (
            "reference._task_models.dinov3_eomt_panoptic_segmentation."
            "task_model.DINOv3EoMTPanopticSegmentation"
        )
    checkpoint = {
        "model_class_path": class_path,
        "model_init_args": init_args,
        "train_model": _source_state(),
    }
    path = tmp_path / f"source_{task}.pt"
    torch.save(checkpoint, path)
    return path


def _assert_expected_model(model: Any, *, task: str, mlp_ratio: int = 4) -> None:
    assert model.init_args["layer_norm_eps"] == 1e-5
    assert model.init_args["mlp_ratio"] == mlp_ratio
    assert model.init_args["num_queries"] == QUERIES
    assert model.init_args["num_blocks"] == JOINT_BLOCKS
    if task == "panoptic":
        assert model.init_args["num_classes"] == NUM_CLASSES
        assert model.init_args["num_thing_classes"] == 2
    assert model.model.layernorm.eps == 1e-5
    assert model.model.layers[0].norm1.eps == 1e-5
    assert model.model.upscale_block.block[0].layernorm2d.eps == 1e-5
    torch.testing.assert_close(
        model.model.rope_embeddings.inv_freq,
        torch.tensor([1.0 / value for value in PERIODS]),
    )
    torch.testing.assert_close(
        model.model.rope_embeddings.original_inv_freq,
        torch.tensor([1.0 / value for value in PERIODS]),
    )
    torch.testing.assert_close(
        model.model.attn_mask_probs, torch.zeros(JOINT_BLOCKS)
    )


def test_semantic_roundtrip(tmp_path):
    source = _write_checkpoint(tmp_path, "semantic")
    out = tmp_path / "export.pt"
    result = _run("--checkpoint", str(source), "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    metadata = json.loads((tmp_path / "export.json").read_text())
    assert metadata["task"] == "semantic"
    assert metadata["num_classes"] == NUM_CLASSES
    assert metadata["classes"] == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]
    assert metadata["layer_norm_eps"] == 1e-5
    assert metadata["attn_mask_prob"] == 0.0

    model = load_task_model(out, device="cpu")
    assert type(model).__name__ == "Dinov3EomtSemanticSegmentation"
    _assert_expected_model(model, task="semantic")
    model.eval()
    with torch.inference_mode():
        raw = model.forward_backend(
            model.preprocess_batch(torch.randn(1, 3, IMAGE, IMAGE))
        )
    masks, logits = raw
    assert masks.shape == (
        1,
        QUERIES,
        IMAGE // PATCH * 2**UPSCALE_BLOCKS,
        IMAGE // PATCH * 2**UPSCALE_BLOCKS,
    )
    assert logits.shape == (1, QUERIES, NUM_CLASSES + 1)
    assert torch.isfinite(masks).all() and torch.isfinite(logits).all()


def test_panoptic_roundtrip(tmp_path):
    source = _write_checkpoint(tmp_path, "panoptic")
    out = tmp_path / "export.pt"
    result = _run("--checkpoint", str(source), "--out", str(out))
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "export.json").read_text())
    assert metadata["task"] == "panoptic"
    assert metadata["num_thing_classes"] == 2
    assert metadata["thing_classes"] == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]
    assert metadata["stuff_classes"] == [{"id": 5, "name": "c"}]

    model = load_task_model(out, device="cpu")
    assert type(model).__name__ == "Dinov3EomtPanopticSegmentation"
    _assert_expected_model(model, task="panoptic")
    model.eval()
    with torch.inference_mode():
        raw = model.forward_backend(
            model.preprocess_batch(torch.randn(1, 3, IMAGE, IMAGE))
        )
    masks, logits = raw
    assert masks.shape == (
        1,
        QUERIES,
        IMAGE // PATCH * 2**UPSCALE_BLOCKS,
        IMAGE // PATCH * 2**UPSCALE_BLOCKS,
    )
    assert logits.shape == (1, QUERIES, NUM_CLASSES + 1)


def test_mlp_ratio_six_roundtrip(tmp_path):
    torch.manual_seed(0)
    source = _write_checkpoint(tmp_path, "semantic")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["train_model"] = _source_state(mlp_ratio=6)
    torch.save(checkpoint, source)
    out = tmp_path / "export.pt"
    result = _run("--checkpoint", str(source), "--out", str(out))
    assert result.returncode == 0, result.stderr
    model = load_task_model(out, device="cpu")
    _assert_expected_model(model, task="semantic", mlp_ratio=6)


def test_dry_run_writes_nothing(tmp_path):
    source = _write_checkpoint(tmp_path, "semantic")
    out = tmp_path / "export.pt"
    result = _run("--checkpoint", str(source), "--out", str(out), "--dry-run")
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["task"] == "semantic"
    assert preview["config"]["hidden_size"] == HIDDEN
    assert preview["config"]["num_hidden_layers"] == LAYERS
    assert preview["config"]["layer_norm_eps"] == 1e-5
    assert preview["num_classes"] == NUM_CLASSES
    assert not out.exists()
    assert not (tmp_path / "export.json").exists()


def test_masked_q_bias_is_transported(tmp_path):
    # The reference attention applies bias_mask on every forward, so a
    # masked-off q bias must be written as zero even when the raw bias is
    # nonzero.
    source = _write_checkpoint(tmp_path, "semantic")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["train_model"]["model.backbone.blocks.0.attn.qkv.bias_mask"][
        :HIDDEN
    ] = 0.0
    torch.save(checkpoint, source)
    out = tmp_path / "export.pt"
    result = _run("--checkpoint", str(source), "--out", str(out))
    assert result.returncode == 0, result.stderr
    model = load_task_model(out, device="cpu")
    torch.testing.assert_close(
        model.model.layers[0].attention.q_proj.bias, torch.zeros(HIDDEN)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("masked_k_bias", "key bias"),
        ("unknown_key", "unmapped"),
        ("storage_tokens", "four storage tokens"),
        ("normalize", "normalization"),
    ],
)
def test_rejects_unrepresentable_checkpoints(tmp_path, mutation, message):
    source = _write_checkpoint(tmp_path, "semantic")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["train_model"]
    if mutation == "masked_k_bias":
        # A nonzero k bias is only invisible when the mask zeroes it; make
        # the mask pass it through so the HF layout cannot represent it.
        state["model.backbone.blocks.0.attn.qkv.bias"][HIDDEN : 2 * HIDDEN] = 0.5
        state["model.backbone.blocks.0.attn.qkv.bias_mask"][
            HIDDEN : 2 * HIDDEN
        ] = 1.0
    elif mutation == "unknown_key":
        state["model.backbone.unexpected"] = torch.zeros(1)
    elif mutation == "storage_tokens":
        checkpoint["train_model"] = _source_state(storage_tokens=3)
    elif mutation == "normalize":
        checkpoint["model_init_args"]["image_normalize"]["mean"] = [0.5, 0.5, 0.5]
    torch.save(checkpoint, source)
    result = _run("--checkpoint", str(source), "--out", str(tmp_path / "out.pt"))
    assert result.returncode == 1
    assert message in result.stderr
