"""Tests for the EOMT semantic segmentation family (HF-backed)."""

import pytest
import torch

from autovision import load_task_model as load_model
from autovision import train_semantic_segmentation
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.semantic_segmentation.dinov2_eomt import (
    Dinov2EomtSemanticSegmentation,
    Dinov2EomtSemanticSegmentationTrain,
    Dinov2EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.shared.eomt import (
    adapt_eomt_head,
    build_eomt_model,
    build_eomt_optimizer,
    semantic_masks_to_instances,
    semantic_segmentation_decode,
)

MODEL_KWARGS = {
    "image_size": 64,
    "num_queries": 6,
    "num_blocks": 2,
    "num_upscale_blocks": 1,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}


def make_model(**overrides) -> Dinov2EomtSemanticSegmentation:
    torch.manual_seed(0)
    return Dinov2EomtSemanticSegmentation(
        num_classes=3, **{**MODEL_KWARGS, **overrides}
    )


def make_train_model(**overrides) -> Dinov2EomtSemanticSegmentationTrain:
    args = Dinov2EomtSemanticSegmentationTrainArgs(
        num_classes=3, lr_warmup_steps=2, **{**MODEL_KWARGS, **overrides}
    )
    return Dinov2EomtSemanticSegmentationTrain(args)


def make_batch(size: int = 64, images: int = 2):
    mask0 = torch.zeros(size, size, dtype=torch.long)
    mask0[: size // 2] = 1
    mask1 = torch.full((size, size), 2, dtype=torch.long)
    mask1[size // 2 :] = 255  # ignored region only
    return {
        "image": torch.randn(images, 3, size, size),
        "mask": torch.stack([mask0, mask1]),
    }


class TestLabels:
    def test_one_instance_per_present_class(self):
        mask = torch.tensor([[0, 0, 1], [1, 2, 2]])
        mask_labels, class_labels = semantic_masks_to_instances(mask.unsqueeze(0))
        assert mask_labels[0].shape == (3, 2, 3)
        assert class_labels[0].tolist() == [0, 1, 2]
        assert mask_labels[0][2].sum().item() == 2.0  # class 2 area
        assert mask_labels[0][0].sum().item() == 2.0

    def test_ignore_index_spawns_no_instance(self):
        mask = torch.tensor([[0, 255], [255, 0]])
        mask_labels, class_labels = semantic_masks_to_instances(mask.unsqueeze(0))
        assert class_labels[0].tolist() == [0]
        # Ignored pixels are 0 inside the binary mask, not instances.
        assert mask_labels[0][0, 0, 1].item() == 0.0

    def test_all_ignored_gives_empty_instance_list(self):
        mask = torch.full((1, 2, 2), 255, dtype=torch.long)
        mask_labels, class_labels = semantic_masks_to_instances(mask)
        assert mask_labels[0].shape == (0, 2, 2)
        assert class_labels[0].numel() == 0


class TestModeling:
    def test_forward_training_contract(self):
        model = build_eomt_model(
            3, **MODEL_KWARGS,
        )  # type: ignore[arg-type]
        x = torch.randn(2, 3, 64, 64)
        mask_labels, class_labels = semantic_masks_to_instances(make_batch()["mask"])
        out = model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels)
        assert torch.isfinite(out.loss)
        assert out.loss.item() > 0
        # mask logits: grid 4x4 x 2^num_upscale_blocks
        assert out.masks_queries_logits.shape == (2, 6, 8, 8)
        assert out.class_queries_logits.shape == (2, 6, 4)  # num_labels + 1

    def test_gradients_reach_the_whole_stack(self):
        model = build_eomt_model(3, **MODEL_KWARGS)  # type: ignore[arg-type]
        x = torch.randn(1, 3, 64, 64)
        mask_labels, class_labels = semantic_masks_to_instances(make_batch()["mask"][:1])
        model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels).loss.backward()
        assert model.embeddings.patch_embeddings.projection.weight.grad is not None
        assert model.query.weight.grad is not None
        assert model.class_predictor.weight.grad is not None

    def test_adapt_head_rebuilds_layout(self):
        model = build_eomt_model(3, **MODEL_KWARGS)  # type: ignore[arg-type]
        adapt_eomt_head(model, num_labels=5, num_queries=8)
        assert model.class_predictor.out_features == 6
        assert model.query.weight.shape == (8, 32)
        assert model.criterion.empty_weight.shape == (6,)
        # The adapted head still runs with the new label count.
        x = torch.randn(1, 3, 64, 64)
        mask = torch.randint(0, 5, (1, 64, 64))
        mask_labels, class_labels = semantic_masks_to_instances(mask)
        out = model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels)
        assert out.class_queries_logits.shape[-1] == 6


class TestPostprocess:
    def test_decode_semantic_map(self):
        masks = torch.full((1, 2, 4, 4), -10.0)
        masks[0, 0, :2] = 10.0  # query 0: top half positive
        masks[0, 1, 2:] = 10.0  # query 1: bottom half positive
        classes = torch.full((1, 2, 3), -10.0)
        classes[0, 0, 0] = 10.0  # query 0 -> class 0
        classes[0, 1, 1] = 10.0  # query 1 -> class 1
        results = semantic_segmentation_decode(
            masks, classes, [{"orig_w": 4, "orig_h": 4}]
        )
        mask = results[0]["mask"]
        assert mask.shape == (4, 4)
        assert (mask[:2] == 0).all() and (mask[2:] == 1).all()

    def test_decode_resizes_to_original(self):
        masks = torch.zeros(1, 2, 4, 4)
        classes = torch.zeros(1, 2, 3)
        results = semantic_segmentation_decode(
            masks, classes, [{"orig_w": 6, "orig_h": 8}]
        )
        assert results[0]["mask"].shape == (8, 6)


class TestScheduler:
    def test_layerwise_decay_ladder(self):
        model = build_eomt_model(3, **MODEL_KWARGS)  # type: ignore[arg-type]
        optimizer = build_eomt_optimizer(model, lr=1e-4, layer_decay=0.9)
        # 4 blocks: layers.N -> 0.9^(4-N); parameters outside the stack
        # (embeddings, queries, scale/head blocks) train at the full lr.
        expected = sorted(
            1e-4 * factor for factor in (0.9**4, 0.9**3, 0.9**2, 0.9, 1.0)
        )
        lrs = sorted({group["lr"] for group in optimizer.param_groups})
        assert lrs == pytest.approx(expected)

    def test_norm_and_bias_skip_weight_decay(self):
        model = build_eomt_model(3, **MODEL_KWARGS)  # type: ignore[arg-type]
        optimizer = build_eomt_optimizer(model, lr=1e-4, weight_decay=0.01)
        bias_groups = [
            group
            for group in optimizer.param_groups
            if any(p.ndim <= 1 for p in group["params"])
        ]
        assert bias_groups and all(g["weight_decay"] == 0.0 for g in bias_groups)


class TestEomtTaskModel:
    def test_three_stage_contract(self):
        model = make_model()
        model.eval()
        image = torch.rand(3, 64, 64)
        tensor, meta = model.preprocess_image(image)
        assert tensor.shape == (3, 64, 64) and meta == {"orig_w": 64, "orig_h": 64}
        raw = model.forward_backend(model.preprocess_batch(tensor.unsqueeze(0)))
        masks_logits, class_logits = raw
        assert masks_logits.shape == (1, 6, 8, 8)
        assert class_logits.shape == (1, 6, 4)
        assert torch.isfinite(masks_logits).all() and torch.isfinite(class_logits).all()
        results = model.postprocess(raw, [{"orig_w": 96, "orig_h": 80}])
        assert results[0]["mask"].shape == (80, 96)
        assert results[0]["mask"].dtype == torch.long
        assert results[0]["mask"].max().item() < 3

    def test_init_args_layout(self):
        model = make_model()
        assert model.init_args["num_classes"] == 3
        assert model.init_args["image_size"] == 64
        assert "load_weights" not in model.init_args
        assert model.class_path.endswith("Dinov2EomtSemanticSegmentation")

    def test_unknown_kwargs_absorbed(self):
        # Task-model constructors absorb extra kwargs by contract
        # (forward-compatible loading of older checkpoints).
        model = make_model(nonexistent_option=1)
        assert "nonexistent_option" not in model.init_args


class TestEomtTrainModel:
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
        result = train.validation_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        metrics = train.compute_metrics("val")
        assert set(metrics) == {"val_metric/miou", "val_metric/pixel_accuracy"}
        assert 0.0 <= metrics["val_metric/miou"] <= 1.0

    def test_metric_state_roundtrip(self):
        train = make_train_model()
        train.model.eval()
        train.validation_step(make_batch(), step=0)
        state = train.export_metric_state("val")
        other = make_train_model()
        other.merge_metric_state("val", state)
        miou = other.compute_metrics("val")
        assert 0.0 <= miou["val_metric/miou"] <= 1.0

    def test_optimizer_recipe(self):
        train = make_train_model()
        optimizer, scheduler = train.get_optimizer(total_steps=10, global_batch_size=2)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert isinstance(scheduler, CosineWarmupScheduler)
        # The cosine scheduler overwrites group lrs on construction (step 0
        # of the warmup); the recipe is checked via the base lrs.
        base_lrs = {base_lr for base_lr in scheduler.base_lrs}
        assert max(base_lrs) == pytest.approx(train.model_args.lr)
        assert min(base_lrs) < train.model_args.lr  # layer decay present

    def test_class_var_defaults(self):
        assert Dinov2EomtSemanticSegmentationTrainArgs.default_batch_size == 16
        assert Dinov2EomtSemanticSegmentationTrainArgs.default_steps == 160_000


class TestEomtEndToEnd:
    def test_train_and_predict(self, tmp_path):
        import numpy as np
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
            mask = rng.integers(0, 3, (64, 64), dtype=np.uint8)
            Image.fromarray(mask).save(masks / f"seg_{index}.png")
        result = train_semantic_segmentation(
            out=tmp_path / "run",
            data={"train": {"images": images, "masks": masks}},
            model="dinov2/eomt",
            num_classes=3,
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={
                **MODEL_KWARGS,
                "lr_warmup_steps": 2,
                "scale_jitter": [0.5, 2.0],
            },
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()

        loaded = load_model(result.exported_model_path, device="cpu")
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["mask"].shape == (64, 64)
        assert prediction["mask"].max().item() < 3


class TestEomtExportAndWeights:
    def test_forward_matches_backend(self):
        model = make_model()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            direct = model(x)
            backend = model.forward_backend(x)
        for left, right in zip(direct, backend):
            torch.testing.assert_close(left, right)

    def test_onnx_export_uses_forward(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        model = make_model()
        out = model.export_onnx(
            tmp_path / "eomt.onnx",
            batch_size=1,
            dynamic_batch_size=False,
            simplify=False,
            verify=True,
        )
        assert out.is_file() and out.stat().st_size > 0

    def test_hub_weights_require_license_opt_in(self, monkeypatch):
        monkeypatch.delenv("AUTOVISION_DINOV2_WEIGHTS", raising=False)
        with pytest.raises(PermissionError, match="CC-BY-NC"):
            build_eomt_model(3, load_weights=True, **MODEL_KWARGS)
        monkeypatch.delenv("AUTOVISION_DINOV3_WEIGHTS", raising=False)
        with pytest.raises(PermissionError, match="DINOv3"):
            build_eomt_model(
                3, variant="dinov3", load_weights=True, **MODEL_KWARGS
            )

    def test_unpinned_hub_weights_fail_closed(self, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DINOV2_WEIGHTS", "1")
        monkeypatch.delenv("AUTOVISION_ALLOW_UNPINNED_WEIGHTS", raising=False)
        with pytest.raises(PermissionError, match="no pinned sha256"):
            build_eomt_model(3, load_weights=True, **MODEL_KWARGS)

    def test_class_head_transfers_from_checkpoint_layout(self, monkeypatch):
        from transformers import EomtConfig, EomtForUniversalSegmentation

        checkpoint_model = EomtForUniversalSegmentation(
            EomtConfig(num_labels=2, num_queries=4)
        )
        monkeypatch.setattr(
            EomtForUniversalSegmentation,
            "from_pretrained",
            classmethod(
                lambda cls, source, config=None, **kwargs: checkpoint_model
            ),
        )
        monkeypatch.setenv("AUTOVISION_DINOV2_WEIGHTS", "1")
        monkeypatch.setenv("AUTOVISION_ALLOW_UNPINNED_WEIGHTS", "1")
        model = build_eomt_model(
            5, load_weights=True, **{**MODEL_KWARGS, "num_queries": 8}
        )
        assert model.class_predictor.out_features == 6
        assert model.query.num_embeddings == 8
        assert model.config.num_labels == 5
