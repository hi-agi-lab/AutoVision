# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv3 EOMT segmentation variants: derivation contracts on top of the
DINOv2 packages (model strings, HF EomtDinov3 classes, args surfaces and
minimal end-to-end loops). Deep task logic (losses, metrics, decodes,
transforms) is covered by the dinov2 test suites and not repeated here."""

import json

import numpy as np
import pytest
import torch
from transformers import (
    EomtDinov3ForUniversalSegmentation,
    EomtForUniversalSegmentation,
)

from autovision import (
    load_task_model as load_model,
)
from autovision import (
    train_instance_segmentation,
    train_panoptic_segmentation,
    train_semantic_segmentation,
)
from autovision.tasks.instance_segmentation.dinov2_eomt import (
    Dinov2EomtInstanceSegmentation,
    Dinov2EomtInstanceSegmentationTrain,
)
from autovision.tasks.instance_segmentation.dinov3_eomt import (
    Dinov3EomtInstanceSegmentation,
    Dinov3EomtInstanceSegmentationTrain,
    Dinov3EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.panoptic_segmentation.dinov2_eomt import (
    Dinov2EomtPanopticSegmentation,
    Dinov2EomtPanopticSegmentationTrain,
)
from autovision.tasks.panoptic_segmentation.dinov3_eomt import (
    Dinov3EomtPanopticSegmentation,
    Dinov3EomtPanopticSegmentationTrain,
    Dinov3EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt import (
    Dinov2EomtSemanticSegmentation,
    Dinov2EomtSemanticSegmentationTrain,
    Dinov2EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt import (
    Dinov3EomtSemanticSegmentation,
    Dinov3EomtSemanticSegmentationTrain,
    Dinov3EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.shared.eomt.modeling import ROPE_PERIODS_KEY, build_eomt_model

# Small HF-compatible EOMT layout (hidden 16 / queries 4 / blocks 3 crash).
MODEL_KWARGS = {
    "image_size": 64,
    "num_queries": 6,
    "num_blocks": 2,
    "num_upscale_blocks": 1,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}


def make_model(**overrides) -> Dinov3EomtSemanticSegmentation:
    return Dinov3EomtSemanticSegmentation(
        num_classes=3, **{**MODEL_KWARGS, **overrides}
    )


def make_train_model(**overrides) -> Dinov3EomtSemanticSegmentationTrain:
    args = Dinov3EomtSemanticSegmentationTrainArgs(
        num_classes=3, lr_warmup_steps=2, **{**MODEL_KWARGS, **overrides}
    )
    return Dinov3EomtSemanticSegmentationTrain(args)


def make_batch(size: int = 64, images: int = 2):
    mask = torch.randint(0, 3, (images, size, size))
    return {
        "image": torch.randn(images, 3, size, size),
        "mask": mask,
    }


def write_coco(tmp_path, num_images: int = 3, size: int = 64):
    """Two square instances per image (classes 0 and 1) as polygons."""
    annotations = {
        "images": [
            {"id": i, "file_name": f"img_{i}.jpg", "width": size, "height": size}
            for i in range(num_images)
        ],
        "categories": [
            {"id": 1, "name": "square_a"},
            {"id": 2, "name": "square_b"},
        ],
        "annotations": [],
    }
    for i in range(num_images):
        annotations["annotations"].extend(
            [
                {
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [4, 4, size // 4, size // 4],
                    "segmentation": [
                        [4, 4, 4 + size // 4, 4, 4 + size // 4, 4 + size // 4, 4, 4 + size // 4]
                    ],
                    "area": (size // 4) ** 2,
                    "iscrowd": 0,
                },
                {
                    "image_id": i,
                    "category_id": 2,
                    "bbox": [size // 2, size // 2, size // 4, size // 4],
                    "segmentation": [
                        [
                            size // 2, size // 2,
                            size // 2 + size // 4, size // 2,
                            size // 2 + size // 4, size // 2 + size // 4,
                            size // 2, size // 2 + size // 4,
                        ]
                    ],
                    "area": (size // 4) ** 2,
                    "iscrowd": 0,
                },
            ]
        )
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations))
    return path


def write_panoptic_data(tmp_path, num_images: int = 2, size: int = 64):
    from PIL import Image

    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "masks").mkdir(parents=True)
        annotations = {}
        for i in range(num_images):
            stem = f"img_{split[0]}{i}"
            Image.fromarray(
                np.random.default_rng(hash(stem) % 2**32).integers(
                    0, 256, (size, size, 3), dtype=np.uint8
                )
            ).save(tmp_path / split / "images" / f"{stem}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[: size // 2, : size // 2] = 1  # thing segment
            mask[size // 2 :, size // 2 :] = 2  # stuff segment
            Image.fromarray(mask).save(tmp_path / split / "masks" / f"{stem}.png")
            annotations[stem] = [
                {"id": 1, "category_id": 1, "iscrowd": 0},
                {"id": 2, "category_id": 2, "iscrowd": 0},
            ]
        path = tmp_path / split / "segments.json"
        path.write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": 1, "name": "car", "isthing": 1},
                        {"id": 2, "name": "sky", "isthing": 0},
                    ],
                    "annotations": annotations,
                }
            )
        )
    return {
        "train": {
            "images": tmp_path / "train" / "images",
            "masks": tmp_path / "train" / "masks",
            "segments": tmp_path / "train" / "segments.json",
        },
        "val": {
            "images": tmp_path / "val" / "images",
            "masks": tmp_path / "val" / "masks",
            "segments": tmp_path / "val" / "segments.json",
        },
    }


class TestArgs:
    def test_inheritance_chain(self):
        assert issubclass(
            Dinov3EomtSemanticSegmentationTrainArgs,
            Dinov2EomtSemanticSegmentationTrainArgs,
        )
        assert issubclass(
            Dinov3EomtInstanceSegmentationTrainArgs,
            Dinov3EomtSemanticSegmentationTrainArgs,
        )
        assert issubclass(
            Dinov3EomtPanopticSegmentationTrainArgs,
            Dinov3EomtSemanticSegmentationTrainArgs,
        )

    def test_semantic_defaults_inherited(self):
        args = Dinov3EomtSemanticSegmentationTrainArgs(num_classes=3)
        assert args.image_size == 512
        assert args.num_queries == 100
        assert args.num_blocks == 4
        assert args.layer_decay == 0.9
        assert args.use_gated_mlp is False
        assert args.default_steps == 160_000

    def test_instance_and_panoptic_defaults(self):
        instance = Dinov3EomtInstanceSegmentationTrainArgs(num_classes=3)
        assert instance.default_steps == 360_000
        assert instance.score_threshold == 0.05
        panoptic = Dinov3EomtPanopticSegmentationTrainArgs(
            num_classes=2, num_thing_classes=1
        )
        assert panoptic.default_steps == 90_000
        assert panoptic.num_blocks == 3
        assert panoptic.scale_jitter == (0.1, 2.0)

    def test_instance_decode_args_forwarded_to_task_model(self):
        # Mirrors the DINOv2 fix: val decoding (model_args) and
        # predict/postprocess (task model attrs) share a single source.
        args = Dinov3EomtInstanceSegmentationTrainArgs(
            num_classes=3,
            score_threshold=0.42,
            mask_threshold=0.61,
            num_top_queries=7,
            **MODEL_KWARGS,
        )
        train = Dinov3EomtInstanceSegmentationTrain(args)
        assert train.model.score_threshold == 0.42
        assert train.model.mask_threshold == 0.61
        assert train.model.num_top_queries == 7

    def test_rejects_use_swiglu_ffn(self):
        # The DINOv3 HF config has no SwiGLU switch; fail loudly instead of
        # silently ignoring the knob.
        with pytest.raises(ValueError, match="use_gated_mlp"):
            Dinov3EomtSemanticSegmentationTrainArgs(
                num_classes=3, use_swiglu_ffn=True
            )

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="nonexistent_option"):
            Dinov3EomtSemanticSegmentationTrainArgs(
                num_classes=3, nonexistent_option=1
            )

    def test_layer_norm_eps_default_and_forwarded(self):
        default = Dinov3EomtSemanticSegmentationTrainArgs(num_classes=3)
        assert default.layer_norm_eps == 1e-6
        args = Dinov3EomtSemanticSegmentationTrainArgs(
            num_classes=3, layer_norm_eps=1e-5, **MODEL_KWARGS
        )
        train = Dinov3EomtSemanticSegmentationTrain(args)
        assert train.model.init_args["layer_norm_eps"] == 1e-5
        assert train.model.model.layernorm.eps == 1e-5
        assert train.model.model.layers[0].norm1.eps == 1e-5
        assert train.model.model.upscale_block.block[0].layernorm2d.eps == 1e-5

    def test_rejects_nonpositive_layer_norm_eps(self):
        with pytest.raises(ValueError):
            Dinov3EomtSemanticSegmentationTrainArgs(
                num_classes=3, layer_norm_eps=0.0
            )


class TestBuilder:
    def test_builds_hf_eomt_dinov3_model(self):
        model = build_eomt_model(3, variant="dinov3", **MODEL_KWARGS)
        assert isinstance(model, EomtDinov3ForUniversalSegmentation)
        assert model.config.num_labels == 3
        assert model.config.num_register_tokens == 4
        # mlp_ratio maps to the DINOv3 intermediate_size.
        assert model.config.intermediate_size == 4 * MODEL_KWARGS["hidden_size"]

    def test_layer_norm_eps_applied_to_scale_block(self):
        # HF constructs EomtDinov3ScaleLayer's LayerNorm2d with its own
        # default, so the builder must apply the requested eps everywhere.
        model = build_eomt_model(
            3, variant="dinov3", layer_norm_eps=1e-5, **MODEL_KWARGS
        )
        assert model.config.layer_norm_eps == 1e-5
        assert model.layernorm.eps == 1e-5
        assert model.layers[0].norm1.eps == 1e-5
        assert model.layers[0].norm2.eps == 1e-5
        assert model.upscale_block.block[0].layernorm2d.eps == 1e-5

    def test_gated_mlp_forwarded(self):
        model = build_eomt_model(
            3, variant="dinov3", use_gated_mlp=True, **MODEL_KWARGS
        )
        assert model.config.use_gated_mlp is True

    def test_dinov2_variant_regression(self):
        model = build_eomt_model(3, **MODEL_KWARGS)
        assert isinstance(model, EomtForUniversalSegmentation)

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValueError, match="variant"):
            build_eomt_model(3, variant="dinov4")

    def test_loss_contract_parity(self):
        model = build_eomt_model(3, variant="dinov3", **MODEL_KWARGS)
        x = torch.randn(2, 3, 64, 64)
        mask_labels = [
            torch.randint(0, 2, (2, 64, 64)).float(),
            torch.randint(0, 2, (1, 64, 64)).float(),
        ]
        class_labels = [torch.tensor([0, 1]), torch.tensor([1])]
        outputs = model(
            pixel_values=x, mask_labels=mask_labels, class_labels=class_labels
        )
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()
        assert outputs.masks_queries_logits.shape == (2, 6, 8, 8)
        assert outputs.class_queries_logits.shape == (2, 6, 4)


class TestTaskModel:
    def test_subclass_and_class_path(self):
        model = make_model()
        assert isinstance(model, Dinov2EomtSemanticSegmentation)
        assert model.class_path.endswith("Dinov3EomtSemanticSegmentation")

    def test_init_args_layout(self):
        model = make_model()
        assert model.init_args["num_classes"] == 3
        assert model.init_args["use_gated_mlp"] is False
        assert model.init_args["layer_norm_eps"] == 1e-6
        assert "use_swiglu_ffn" not in model.init_args
        assert "load_weights" not in model.init_args

    def test_layer_norm_eps_in_init_args(self):
        model = make_model(layer_norm_eps=1e-5)
        assert model.init_args["layer_norm_eps"] == 1e-5
        assert model.model.layernorm.eps == 1e-5
        assert model.model.upscale_block.block[0].layernorm2d.eps == 1e-5

    def test_rope_periods_transport(self):
        model = make_model()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state[ROPE_PERIODS_KEY] = torch.tensor([1.0, 2.0])
        loaded = make_model()
        loaded.load_train_state_dict(state)
        torch.testing.assert_close(
            loaded.model.rope_embeddings.inv_freq, torch.tensor([1.0, 0.5])
        )
        torch.testing.assert_close(
            loaded.model.rope_embeddings.original_inv_freq,
            torch.tensor([1.0, 0.5]),
        )

    def test_rope_periods_length_mismatch_rejected(self):
        model = make_model()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state[ROPE_PERIODS_KEY] = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="does not match"):
            model.load_train_state_dict(state)

    def test_attn_mask_probs_loaded(self):
        model = make_model()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["model.attn_mask_probs"] = torch.zeros(MODEL_KWARGS["num_blocks"])
        loaded = make_model()
        loaded.load_train_state_dict(state)
        torch.testing.assert_close(
            loaded.model.attn_mask_probs,
            torch.zeros(MODEL_KWARGS["num_blocks"]),
        )

    def test_panoptic_rope_periods_transport(self):
        # The mixin must be first in the panoptic MRO, not only in semantic.
        model = Dinov3EomtPanopticSegmentation(
            num_classes=4, num_thing_classes=2, **MODEL_KWARGS
        )
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state[ROPE_PERIODS_KEY] = torch.tensor([1.0, 2.0])
        loaded = Dinov3EomtPanopticSegmentation(
            num_classes=4, num_thing_classes=2, **MODEL_KWARGS
        )
        loaded.load_train_state_dict(state)
        torch.testing.assert_close(
            loaded.model.rope_embeddings.inv_freq, torch.tensor([1.0, 0.5])
        )

    def test_instance_rope_periods_transport(self):
        model = Dinov3EomtInstanceSegmentation(num_classes=3, **MODEL_KWARGS)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state[ROPE_PERIODS_KEY] = torch.tensor([1.0, 2.0])
        loaded = Dinov3EomtInstanceSegmentation(num_classes=3, **MODEL_KWARGS)
        loaded.load_train_state_dict(state)
        torch.testing.assert_close(
            loaded.model.rope_embeddings.inv_freq, torch.tensor([1.0, 0.5])
        )

    def test_three_stage_contract(self):
        model = make_model()
        model.eval()
        raw = model.forward_backend(
            model.preprocess_batch(torch.rand(1, 3, 64, 64))
        )
        masks_logits, class_logits = raw
        assert masks_logits.shape == (1, 6, 8, 8)
        assert class_logits.shape == (1, 6, 4)
        results = model.postprocess(raw, [{"orig_w": 96, "orig_h": 80}])
        assert results[0]["mask"].shape == (80, 96)
        assert results[0]["mask"].max().item() < 3

    def test_instance_model_decode(self):
        model = Dinov3EomtInstanceSegmentation(num_classes=3, **MODEL_KWARGS)
        assert isinstance(model, Dinov2EomtInstanceSegmentation)
        model.eval()
        raw = model.forward_backend(
            model.preprocess_batch(torch.rand(1, 3, 64, 64))
        )
        results = model.postprocess(raw, [{"orig_w": 64, "orig_h": 64}])
        assert set(results[0]) == {"masks", "labels", "scores"}

    def test_panoptic_model_decode(self):
        model = Dinov3EomtPanopticSegmentation(
            num_classes=4, num_thing_classes=2, **MODEL_KWARGS
        )
        assert isinstance(model, Dinov2EomtPanopticSegmentation)
        assert model.init_args["num_thing_classes"] == 2
        model.eval()
        raw = model.forward_backend(
            model.preprocess_batch(torch.rand(1, 3, 64, 64))
        )
        results = model.postprocess(raw, [{"orig_w": 64, "orig_h": 64}])
        assert results[0]["masks"].dtype == torch.long
        # User class ids only (ignore slot remapped to the last user class).
        assert results[0]["masks"].max() <= 3


class TestTrainModel:
    def test_wrappers_subclass_dinov2(self):
        assert issubclass(
            Dinov3EomtSemanticSegmentationTrain, Dinov2EomtSemanticSegmentationTrain
        )
        assert issubclass(
            Dinov3EomtInstanceSegmentationTrain, Dinov2EomtInstanceSegmentationTrain
        )
        assert issubclass(
            Dinov3EomtPanopticSegmentationTrain, Dinov2EomtPanopticSegmentationTrain
        )

    def test_task_attributes(self):
        assert Dinov3EomtSemanticSegmentationTrain.task == "semantic_segmentation"
        assert Dinov3EomtSemanticSegmentationTrain.watch_metric == "val_metric/miou"
        assert Dinov3EomtInstanceSegmentationTrain.task == "instance_segmentation"
        assert Dinov3EomtInstanceSegmentationTrain.watch_metric == "val_metric/mask_map"
        assert Dinov3EomtPanopticSegmentationTrain.task == "panoptic_segmentation"
        assert Dinov3EomtPanopticSegmentationTrain.watch_metric == "val_metric/pq"

    def test_training_step_backprop(self):
        train = make_train_model()
        result = train.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        head_grad = train.model.model.class_predictor.weight.grad
        assert head_grad is not None and head_grad.abs().sum() > 0

    def test_validation_updates_miou(self):
        train = make_train_model()
        train.model.eval()
        train.validation_step(make_batch(), step=0)
        metrics = train.compute_metrics("val")
        assert set(metrics) == {"val_metric/miou", "val_metric/pixel_accuracy"}

    def test_instance_and_panoptic_val_steps(self):
        instance_args = Dinov3EomtInstanceSegmentationTrainArgs(
            num_classes=3, lr_warmup_steps=2, **MODEL_KWARGS
        )
        instance = Dinov3EomtInstanceSegmentationTrain(instance_args)
        x = torch.randn(2, 3, 64, 64)
        batch = {
            "image": x,
            "masks": [
                torch.randint(0, 2, (2, 64, 64)).float(),
                torch.randint(0, 2, (1, 64, 64)).float(),
            ],
            "classes": [torch.tensor([0, 1]), torch.tensor([1])],
            "bboxes": [torch.rand(2, 4), torch.rand(1, 4)],
            "original_size": [(64, 64), (64, 64)],
        }
        instance.training_step(batch, step=0)
        instance.validation_step(batch, step=0)
        assert "val_metric/mask_map" in instance.compute_metrics("val")

        panoptic_args = Dinov3EomtPanopticSegmentationTrainArgs(
            num_classes=4, num_thing_classes=2, lr_warmup_steps=2, **MODEL_KWARGS
        )
        panoptic = Dinov3EomtPanopticSegmentationTrain(panoptic_args)
        binary_masks = [
            {"masks": torch.randint(0, 2, (2, 64, 64)).float(),
             "labels": torch.tensor([0, 1])},
            {"masks": torch.randint(0, 2, (1, 64, 64)).float(),
             "labels": torch.tensor([3])},
        ]
        batch = {"image": x, "binary_masks": binary_masks}
        panoptic.training_step(batch, step=0)
        batch["masks"] = torch.randint(0, 4, (2, 64, 64))
        panoptic.validation_step(batch, step=0)
        assert "val_metric/pq" in panoptic.compute_metrics("val")

    def test_optimizer_builds(self):
        train = make_train_model()
        optimizer, scheduler = train.get_optimizer(total_steps=10, global_batch_size=2)
        assert optimizer is not None and scheduler is not None
        assert len(optimizer.param_groups) > 0


class TestEndToEnd:
    def test_semantic_train_and_predict(self, tmp_path):
        from PIL import Image

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        rng = np.random.default_rng(0)
        for index in range(3):
            Image.fromarray(
                rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(images / f"seg_{index}.jpg")
            Image.fromarray(rng.integers(0, 3, (64, 64), dtype=np.uint8)).save(
                masks / f"seg_{index}.png"
            )
        result = train_semantic_segmentation(
            out=tmp_path / "run",
            data={"train": {"images": images, "masks": masks}},
            model="dinov3/eomt",
            num_classes=3,
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
        )
        assert result.exported_model_path is not None
        loaded = load_model(result.exported_model_path, device="cpu")
        assert type(loaded).__name__ == "Dinov3EomtSemanticSegmentation"
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["mask"].shape == (64, 64)
        assert prediction["mask"].max().item() < 3

    def test_instance_train_and_predict(self, tmp_path):
        from PIL import Image

        annotations = write_coco(tmp_path)
        for i in range(3):
            Image.fromarray(
                np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(tmp_path / f"img_{i}.jpg")
        result = train_instance_segmentation(
            out=tmp_path / "run",
            data={"train": {"annotations": annotations, "images": tmp_path}},
            model="dinov3/eomt",
            num_classes=2,
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
        )
        assert result.exported_model_path is not None
        loaded = load_model(result.exported_model_path, device="cpu")
        assert type(loaded).__name__ == "Dinov3EomtInstanceSegmentation"
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["masks"].shape[1:] == (64, 64)
        assert prediction["masks"].dtype == torch.bool

    def test_panoptic_train_and_predict(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        result = train_panoptic_segmentation(
            out=tmp_path / "run",
            data=data,
            model="dinov3/eomt",
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
        )
        assert result.best_metrics is not None
        assert 0.0 <= result.best_metrics["val_metric/pq"] <= 1.0
        assert result.classes == {0: "car", 1: "sky"}
        loaded = load_model(result.exported_model_path, device="cpu")
        assert type(loaded).__name__ == "Dinov3EomtPanopticSegmentation"
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["masks"].shape == (64, 64, 2)
        assert prediction["masks"][..., 0].max() <= 1  # user class ids only

    def test_semantic_training_is_deterministic(self, tmp_path):
        import random as _random

        from PIL import Image

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        rng = np.random.default_rng(0)
        for index in range(3):
            Image.fromarray(
                rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(images / f"seg_{index}.jpg")
            Image.fromarray(rng.integers(0, 3, (64, 64), dtype=np.uint8)).save(
                masks / f"seg_{index}.png"
            )
        data = {"train": {"images": images, "masks": masks}}
        runs = []
        for i in range(2):
            # Model weights are drawn at construction time inside the
            # command, so seed the process RNGs before each run.
            torch.manual_seed(0)
            _random.seed(0)
            np.random.seed(0)
            result = train_semantic_segmentation(
                out=tmp_path / f"run_{i}",
                data=data,
                model="dinov3/eomt",
                num_classes=3,
                steps=3,
                batch_size=2,
                num_workers=0,
                model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
            )
            runs.append(result.best_metrics)
        assert runs[0] == runs[1]
