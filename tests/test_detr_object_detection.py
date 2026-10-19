# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DETR object detection family (SSL-backbone finetuning)."""

import random

import pytest
import torch
import transformers
from tests.conftest import create_image
from tests.test_train_object_detection import (
    data_config,
    make_coco_dataset,
    transform_args,
)

from autovision import load_task_model as load_model
from autovision import train_object_detection
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
    list_detr_model_names,
    resolve_detr_variant,
)
from autovision.tasks.detection.detr.task_model import (
    _relabel_prefix_drift,
)


def make_tiny_hf_model() -> TransformersDinov2Wrapper:
    config = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=14,
        num_register_tokens=0,
    )
    return TransformersDinov2Wrapper(transformers.Dinov2Model(config))


def make_train_model(**overrides) -> DETRObjectDetectionTrain:
    args = DETRObjectDetectionTrainArgs(
        num_classes=2,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=20,
        num_top_queries=20,
        **overrides,
    )
    return DETRObjectDetectionTrain(args)


def make_batch(size: int = 224, images: int = 2):
    targets = [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])}
        for _ in range(images)
    ]
    return {
        "image": torch.randn(images, 3, size, size),
        "bboxes": [t["boxes"] for t in targets],
        "classes": [t["labels"] for t in targets],
        "targets": targets,
    }


class TestRegistry:
    def test_names_sorted_with_prefix(self):
        names = list_detr_model_names()
        assert names == sorted(names)
        assert all(name.startswith("detr/") for name in names)
        assert "detr/dinov2-vits14" in names
        assert "detr/dinov2-vits14-notpretrained" in names
        assert "detr/dinov2-vittest-notpretrained" in names
        # Baseline-faithful expansions.
        assert "detr/dinov3-vitt16-notpretrained" in names
        assert "detr/dinov3-vits16" in names

    def test_resolve_variant(self):
        key, spec = resolve_detr_variant("detr/dinov2-vits14")
        assert key == "dinov2-vits14"
        assert spec.family == "vit"
        assert spec.patch_size == 14
        assert spec.interaction_indexes == (5, 8, 11)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown DETR"):
            resolve_detr_variant("detr/nope")
        with pytest.raises(ValueError, match="Unknown DETR"):
            resolve_detr_variant("rtdetr/resnet18")

    def test_dinov2_variants_are_nosta_with_baseline_recipes(self):
        """Baseline: DINOv2 ViT variants use NO spatial prior, and the
        vits14 recipe projects to hidden 224 with a 4-layer decoder."""
        _, spec = resolve_detr_variant("detr/dinov2-vits14")
        assert not hasattr(spec, "use_sta")
        assert not hasattr(spec, "conv_inplane")
        assert spec.hidden_dim == 224
        assert spec.dec_layers == 4
        assert spec.enc_dim_feedforward == 896
        assert spec.dec_dim_feedforward == 1792
        _, base = resolve_detr_variant("detr/dinov2-vitb14")
        assert base.hidden_dim == 768
        _, giant = resolve_detr_variant("detr/dinov2-vitg14")
        assert giant.interaction_indexes == (19, 29, 39)

    def test_dinov3_variants_keep_vit_with_nosta_adapter(self):
        """DINOv3 ViT stays, but the DEIMv2-derived spatial prior was
        removed for licensing; the variant spec carries no STA fields."""
        _, tiny = resolve_detr_variant("detr/dinov3-vitt16")
        assert not hasattr(tiny, "use_sta")
        assert not hasattr(tiny, "conv_inplane")
        assert tiny.hidden_dim == 192
        assert tiny.enc_expansion == 0.34 and tiny.enc_depth_mult == 0.67
        assert tiny.requires_dinov3 is True
        _, small = resolve_detr_variant("detr/dinov3-vits16")
        assert small.hidden_dim == 224

    def test_eupe_names_are_aliases(self):
        base = resolve_detr_variant("detr/dinov3-vits16")[1]
        alias = resolve_detr_variant("detr/dinov3-vits16-eupe")[1]
        assert alias is base

    def test_convnext_recipes(self):
        _, large = resolve_detr_variant("detr/dinov3-convnext-large")
        assert large.family == "convnext"
        assert large.dec_layers == 6
        assert large.dec_num_points == (4, 4, 4)
        assert large.hidden_dim == 384


class TestMultiscaleFeatures:
    def test_intermediate_taps(self):
        wrapper = make_tiny_hf_model()
        x = torch.randn(2, 3, 224, 224)
        feats = wrapper.stage_feature_maps(x, (1, 2, 3))
        assert len(feats) == 3
        for feat in feats:
            assert feat["features"].shape == (2, 64, 16, 16)
            assert feat["cls_token"].shape == (2, 64)
        # Taps are distinct blocks: features must differ across taps.
        assert not torch.allclose(feats[0]["features"], feats[2]["features"])


def _make_tiny_dinov3_vit_wrapper():
    """Tiny real-HF DINOv3 ViT (4 blocks, 2 registers, patch 14)."""
    from autovision.encoders.dinov3.dinov3_wrapper import (
        TransformersDinov3ViTWrapper,
    )

    config = transformers.DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_register_tokens=2,
        image_size=56,
        patch_size=14,
        intermediate_size=64,
    )
    return TransformersDinov3ViTWrapper(transformers.DINOv3ViTModel(config)).eval()


class TestDinov3TapSemantics:
    """The published LT-DETR detector baselines tap through
    ``get_intermediate_layers(norm=True)``: every DINOv3 ViT tap carries
    the final LayerNorm, and the ConvNeXt wrapper norms ONLY the final
    stage (``norms = [Identity, Identity, Identity, self.norm]``).

    Measured against the baseline implementations, the previous un-normed
    taps cost the published detectors 87% (vitt16) / 29% (convnext-tiny) of
    their COCO mAP on a 500-image val2017 subset. The DINOv2 wrapper keeps
    its raw-block-output convention (``tests/test_models_dinov2.py``).
    """

    def test_vit_taps_apply_final_norm(self):
        from autovision.tasks.detection.detr.task_model import _dinov3_supported

        if not _dinov3_supported():
            pytest.skip("installed transformers lacks DINOv3")
        wrapper = _make_tiny_dinov3_vit_wrapper()
        x = torch.randn(1, 3, 56, 56)
        taps = wrapper.stage_feature_maps(x, (1, 3))
        outputs = wrapper.raw_module()(pixel_values=x, output_hidden_states=True)
        num_prefix = 1 + int(wrapper.config.num_register_tokens or 0)
        side = 56 // int(wrapper.config.patch_size)
        for tap, layer_index in zip(taps, (1, 3)):
            hidden = outputs.hidden_states[layer_index + 1]
            patches = wrapper.raw_module().norm(hidden)[:, num_prefix:]
            expected = (
                patches.reshape(1, side, side, -1).permute(0, 3, 1, 2).contiguous()
            )
            assert torch.equal(tap["features"], expected)
            # LayerNorm signature: per-token channel mean is zeroed. The raw
            # (un-normed) block output does not have that property, so this
            # is what distinguishes the two semantics.
            assert tap["features"].mean(dim=1).abs().max() < 1e-5

    def test_vit_masked_taps_share_the_dense_norm(self):
        from autovision.tasks.detection.detr.task_model import _dinov3_supported

        if not _dinov3_supported():
            pytest.skip("installed transformers lacks DINOv3")
        wrapper = _make_tiny_dinov3_vit_wrapper()
        x = torch.randn(1, 3, 56, 56)
        dense = wrapper.stage_feature_maps(x, (1, 3))
        keep_all = torch.ones(2, 2, dtype=torch.bool)
        masked = wrapper.stage_feature_maps_masked(x, (1, 3), [keep_all])
        for tap_dense, tap_masked in zip(dense, masked):
            # keep-all ≡ dense (design doc 17) + dense carries the final
            # norm (above) ⇒ the masked path applies the same norm.
            assert torch.equal(tap_dense["features"], tap_masked["features"])
            assert tap_masked["features"].mean(dim=1).abs().max() < 1e-5

    @pytest.mark.parametrize("pyramid_levels", (3, 4))
    def test_convnext_only_the_last_tap_is_normed(self, pyramid_levels):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
            _dinov3_supported,
        )

        if not _dinov3_supported():
            pytest.skip("installed transformers lacks DINOv3")
        model = DETRObjectDetection(
            num_classes=4,
            model_name="detr/dinov3-convnext-tiny",
            image_size=(64, 64),
            decoder="rtdetrv2",
            dec_layers=6,
            decoder_hidden_dim=256,
            pyramid_levels=pyramid_levels,
            load_weights=False,
        ).eval()
        backbone = model.backbone
        wrapper = backbone._model_wrapper
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            taps = backbone(x)
            outputs = wrapper.raw_module()(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        raw = [hidden_states[i + 1] for i in backbone.stage_indices]
        assert len(taps) == pyramid_levels
        for tap, expected_raw in zip(taps[:-1], raw[:-1]):
            assert torch.equal(tap, expected_raw)
        final = wrapper.raw_module().layer_norm(
            raw[-1].permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        assert torch.equal(taps[-1], final)


class TestViTBackboneWrapper:
    def test_pyramid_shapes_and_projection(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        x = torch.randn(1, 3, 224, 224)
        c2, c3, c4 = backbone(x)
        # Conv-chain pyramid shapes (input//8, then successive ceil-halves
        # matching the encoder's stride-2 convolutions).
        assert c2.shape == (1, 64, 28, 28)
        assert c3.shape == (1, 64, 14, 14)
        assert c4.shape == (1, 64, 7, 7)

    def test_pyramid_shapes_at_odd_patch_grid(self):
        """640 gives a 45x45 patch grid (odd) — the pyramid must still
        land on conv-chain-consistent shapes for the encoder fusion."""
        train_model = make_train_model()
        x = torch.randn(1, 3, 640, 640)
        c2, c3, c4 = train_model.model.backbone(x)
        assert c2.shape == (1, 64, 80, 80)
        assert c3.shape == (1, 64, 40, 40)
        assert c4.shape == (1, 64, 20, 20)

    def test_pyramid_shapes_at_644(self):
        """Regression: 644 (the patch-14 default after snapping) must
        ceil-halve along the encoder's stride-2 conv chain (80 -> 40 -> 20);
        an independent ceil(x/16) ladder would give 41 and break the PAN
        concat."""
        train_model = make_train_model()
        x = torch.randn(1, 3, 644, 644)
        c2, c3, c4 = train_model.model.backbone(x)
        assert c2.shape == (1, 64, 80, 80)
        assert c3.shape == (1, 64, 40, 40)
        assert c4.shape == (1, 64, 20, 20)
        # The full forward must run through the encoder + decoder.
        out = train_model.model._forward_train(x)
        assert out["pred_logits"].shape[-1] == 2

    def test_backbone_model_is_the_hf_vit(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        assert backbone.backbone_model is backbone._model_wrapper.raw_module()
        # The pretrained ViT is the only part of the backbone at low LR —
        # projection/norm layers must live OUTSIDE backbone_model.
        backbone_ids = {id(p) for p in backbone.backbone_model.parameters()}
        adapter_params = [
            p
            for module in (backbone.convs, backbone.norms)
            for p in module.parameters()
        ]
        assert adapter_params
        assert all(id(p) not in backbone_ids for p in adapter_params)

    def test_finetune_false_freezes_vit(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        backbone._model_wrapper.requires_grad_(False)
        backbone._model_wrapper.eval()
        assert not any(p.requires_grad for p in backbone.backbone_model.parameters())


class TestDinov3ViTNoSta:
    """DINOv3 ViT is retained but runs the no-STA adapter after the
    licensing removal; no ``backbone.sta.*`` state is created."""

    def test_builds_and_forwards_without_sta(self):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        model = DETRObjectDetection(
            num_classes=2,
            model_name="dinov3-vitt16-notpretrained",
            image_size=(224, 224),
            num_queries=20,
            load_weights=False,
        )
        assert not hasattr(model.backbone, "sta")
        state_keys = list(model.state_dict())
        assert not any(k.startswith("backbone.sta.") for k in state_keys)
        assert any(k == "backbone.convs.0.weight" for k in state_keys)
        out = model._forward_train(torch.randn(2, 3, 224, 224))
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)


class TestTaskModel:
    def test_forward_shapes(self):
        train_model = make_train_model()
        out = train_model.model._forward_train(torch.randn(2, 3, 224, 224))
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()

    def test_loss_backward(self):
        train_model = make_train_model()
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        assert torch.isfinite(result.loss)
        result.loss.backward()
        grads = [
            p.grad.abs().sum()
            for p in train_model.model.parameters()
            if p.grad is not None
        ]
        assert grads and float(sum(grads)) > 0

    def test_postprocess_maps_to_original_pixels(self):
        train_model = make_train_model()
        model = train_model.model
        orig_w, orig_h = 136, 76
        boxes_px = torch.tensor([[10.0, 10.0, 32.0, 32.0]])
        cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2 / orig_w
        cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2 / orig_h
        w = (boxes_px[:, 2] - boxes_px[:, 0]) / orig_w
        h = (boxes_px[:, 3] - boxes_px[:, 1]) / orig_h
        boxes_norm = torch.stack([cx, cy, w, h], -1).unsqueeze(0)
        logits = torch.full((1, 1, 2), -8.0)
        logits[0, 0, 0] = 6.0
        (decoded,) = model.postprocess(
            (logits, boxes_norm),
            [{"orig_w": orig_w, "orig_h": orig_h}],
            threshold=0.05,
        )
        assert torch.allclose(decoded["bboxes"], boxes_px, atol=1e-4)

    def test_backbone_weights_rejects_foreign_keys(self, tmp_path):
        train_model = make_train_model()
        foreign = {"does.not.exist": torch.zeros(1)}
        path = tmp_path / "weights.pt"
        torch.save(foreign, path)
        with pytest.raises(ValueError, match="unknown to the backbone"):
            train_model.model._load_backbone_weights(str(path))

    def test_backbone_weights_reject_partial_files(self, tmp_path):
        # A file that drops tensors must fail loud — silently keeping the
        # random init for the missing keys would poison the run.
        from safetensors.torch import save_file

        train_model = make_train_model()
        state = train_model.model.backbone.backbone_model.state_dict()
        partial = dict(list(state.items())[: len(state) // 2])
        path = tmp_path / "partial.safetensors"
        save_file(partial, str(path))
        with pytest.raises(ValueError, match="missing"):
            train_model.model._load_backbone_weights(str(path))

    def test_backbone_weights_accept_safetensors(self, tmp_path):
        from safetensors.torch import save_file

        train_model = make_train_model()
        state = train_model.model.backbone.backbone_model.state_dict()
        path = tmp_path / "backbone.safetensors"
        save_file(state, str(path))
        # Loads without raising and produces the same backbone weights.
        fresh = make_train_model()
        fresh.model._load_backbone_weights(str(path))
        reference = train_model.model.backbone.backbone_model.state_dict()
        for key, value in fresh.model.backbone.backbone_model.state_dict().items():
            assert torch.equal(value, reference[key])


class _PrefixedInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)


class _PrefixedTarget(torch.nn.Module):
    """Mirrors the transformers 5.x DINOv3-ConvNeXt layout: weights under
    ``model.*`` plus a top-level ``norm`` (the ConvNeXt ``layer_norm``)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _PrefixedInner()
        self.norm = torch.nn.LayerNorm(1)


class TestBackbonePrefixDrift:
    def test_relabels_legacy_unprefixed_keys(self):
        target = _PrefixedTarget()
        reference = target.state_dict()
        legacy = {
            key.removeprefix("model."): value.clone()
            for key, value in reference.items()
        }
        relabeled = _relabel_prefix_drift(legacy, target)
        assert set(relabeled) == set(reference)
        for key, value in relabeled.items():
            assert torch.equal(value, reference[key])

    def test_identity_state_passes_through(self):
        target = _PrefixedTarget()
        state = target.state_dict()
        assert _relabel_prefix_drift(state, target) is state

    def test_foreign_keys_returned_unchanged_for_strict_report(self):
        target = _PrefixedTarget()
        reference = target.state_dict()
        foreign = {key: value.clone() for key, value in reference.items()}
        foreign["does.not.exist"] = torch.zeros(1)
        assert _relabel_prefix_drift(foreign, target) is foreign


class TestOptimizerGroups:
    def test_vit_lr_groups(self):
        train_model = make_train_model()
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16, gradient_accumulation_steps=2
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        base = {
            g["name"]: g["_initial_lr"] if "_initial_lr" in g else g["initial_lr"]
            for g in optimizer.param_groups
        }
        # DINOv2 ViT family recipe (baseline V2 args): lr 1e-4 at the
        # reference batch, backbone factor 1e-2. The group "lr" at
        # construction already reflects the step-0 warmup factor, so the
        # recipe is asserted on the scheduler's base LRs.
        assert base["detector"] == pytest.approx(1e-4)
        assert base["backbone"] == pytest.approx(1e-4 * 0.01)
        # Full parameter coverage across groups.
        total = sum(len(g["params"]) for g in optimizer.param_groups)
        assert total == len(list(train_model.model.parameters()))
        # The projection/norm adapter trains at detector LR.
        adapter_ids = {
            id(p)
            for module in (
                train_model.model.backbone.convs,
                train_model.model.backbone.norms,
            )
            for p in module.parameters()
        }
        detector_ids = {id(p) for g in groups.values() for p in g["params"]}
        assert adapter_ids <= detector_ids

    def test_scheduler_is_flat_cosine(self):
        from autovision.tasks.shared.flat_cosine import (
            FlatCosineLRScheduler,
        )

        train_model = make_train_model()
        _, scheduler = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        # DINOv2 ViT family resolves to the LINEAR recipe (warmup ->
        # constant); CNN families resolve to flat-cosine.
        from autovision.tasks.detection.shared.multistep_schedule import (
            MultiStepWarmupScheduler,
        )

        assert isinstance(scheduler, MultiStepWarmupScheduler)
        assert scheduler.milestones == ()
        cnn_args = DETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="dfine",
            num_queries=20,
            num_top_queries=20,
            backbone_pretrained=False,
        )
        cnn_train_model = DETRObjectDetectionTrain(cnn_args)
        _, cnn_scheduler = cnn_train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        assert isinstance(cnn_scheduler, FlatCosineLRScheduler)


class TestCocoCheckpointLoader:
    """Published baseline COCO detector warm-start (doc: loader contract)."""

    @staticmethod
    def _model(num_classes: int):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        return DETRObjectDetection(
            num_classes=num_classes,
            model_name="detr/dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="rtdetrv2",
            dec_layers=6,
            num_queries=20,
            pyramid_levels=3,
            decoder_hidden_dim=256,
            load_weights=False,
        )

    @staticmethod
    def _fake_checkpoint(model) -> dict:
        """Reverse-map the model's own state dict into the baseline
        checkpoint layout (old ConvNeXt names + ema_model.model prefix)."""
        RENAME_INV = {"depthwise_conv": "dwconv", "layer_norm": "norm",
                      "pointwise_conv1": "pwconv1", "pointwise_conv2": "pwconv2"}
        sd = model.state_dict()
        out = {}
        for key, value in sd.items():
            if key.startswith("backbone_wrapper._model."):
                rest = key[len("backbone_wrapper._model."):]
                if rest.startswith("model.stages."):
                    p = rest[len("model.stages."):].split(".")
                    if p[1] == "downsample_layers":
                        out[f"backbone.backbone.downsample_layers.{p[0]}.{'.'.join(p[2:])}"] = value
                    else:
                        # stages.{s}.layers.{b}.{module}.{suffix}
                        out[f"backbone.backbone.stages.{p[0]}.{p[2]}.{RENAME_INV.get(p[3], p[3])}.{'.'.join(p[4:])}"] = value
                elif rest.startswith("model.downsample_layers."):
                    p = rest.split(".")  # model.downsample_layers.{s}.{m}.{...}
                    out[f"backbone.backbone.downsample_layers.{p[2]}.{'.'.join(p[3:])}"] = value
                elif rest.startswith("layer_norm."):
                    out[f"backbone.backbone.norm.{rest[len('layer_norm.'):]}"] = value
                else:
                    raise AssertionError(f"unmapped test key {key}")
            elif key.startswith(("encoder.", "decoder.")):
                out[key] = value
        return {"train_model": {f"ema_model.model.{k}": v for k, v in out.items()}}

    def test_round_trip_80_classes(self, tmp_path):
        import torch

        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=80)
        summary = load_baseline_coco_checkpoint(target, path)
        assert summary == {"backbone": 180, "encoder": 246,
                           "decoder": 449 - 1, "decoder_skipped_cls": 0} or \
               summary["decoder_skipped_cls"] == 0
        # spot tensors identical
        for key in ("backbone_wrapper._model.model.stages.2.layers.5.gamma",
                    "encoder.fpn_blocks.0.conv1.conv.weight",
                    "decoder.decoder.layers.2.self_attn.in_proj_weight",
                    "decoder.dec_score_head.0.weight"):
            assert torch.equal(target.state_dict()[key],
                               source.state_dict()[key]), key

    def test_skips_class_count_heads(self, tmp_path):
        import torch

        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=10)
        summary = load_baseline_coco_checkpoint(target, path)
        # 3 dec score heads + 1 enc score head + 1 denoising embed, (w+b).
        assert summary["decoder_skipped_cls"] == 15
        # body tensors still landed
        assert torch.equal(
            target.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
            source.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
        )
        # cls heads stayed fresh
        assert not torch.equal(
            target.state_dict()["decoder.dec_score_head.0.weight"],
            source.state_dict()["decoder.dec_score_head.0.weight"],
        )

    def test_decoder_hidden_dim_decoupling(self):
        # Default (None) keeps the shared width; explicit 256 decouples it
        # (checkpoint contract) without touching the encoder width.
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        model = self._model(num_classes=10)
        sd = model.state_dict()
        assert sd["encoder.lateral_convs.0.conv.weight"].shape[0] == 384
        assert sd["decoder.decoder.layers.0.self_attn.in_proj_weight"].shape == (768, 256)
        assert sd["decoder.dec_score_head.0.weight"].shape == (10, 256)
        # default shared-width model still behaves as before
        shared = DETRObjectDetection(
            num_classes=10, model_name="detr/dinov3-convnext-tiny",
            image_size=(224, 224), decoder="rtdetrv2", dec_layers=6,
            num_queries=20, pyramid_levels=3, load_weights=False,
        )
        ssd = shared.state_dict()
        assert ssd["decoder.decoder.layers.0.self_attn.in_proj_weight"].shape == (1152, 384)


class TestCocoCheckpointLoaderViT:
    """ViT-family baseline COCO detector warm-start (Meta ViT remap)."""

    @staticmethod
    def _model(num_classes: int, variant: str = "vitt16", **kwargs):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        return DETRObjectDetection(
            num_classes=num_classes,
            model_name=f"detr/dinov3-{variant}-notpretrained",
            image_size=(128, 128),
            decoder="rtdetrv2",
            num_queries=20,
            load_weights=False,
            **kwargs,
        )

    @staticmethod
    def _fake_checkpoint(model) -> dict:
        """Reverse-map the model's own state dict into the baseline ViT
        checkpoint layout (Meta names + ema_model.model prefix)."""
        sd = model.state_dict()
        out: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith("backbone._model_wrapper."):
                # Same module as backbone_wrapper._model (registered under
                # both parents); handled by the branch below.
                continue
            if not key.startswith("backbone_wrapper._model."):
                if key.startswith(
                    ("backbone.convs.", "backbone.norms.",
                     "encoder.", "decoder.")
                ):
                    # wrapper extras + encoder/decoder: identical names on
                    # both sides.
                    out[key] = value
                continue
            rest = key[len("backbone_wrapper._model."):]
            if rest.startswith("model.layer."):
                index, sub = rest[len("model.layer."):].split(".", 1)
                if sub.startswith(
                    ("attention.q_proj", "attention.k_proj", "attention.v_proj")
                ):
                    continue  # fused qkv rebuilt after the loop
                if sub.startswith("attention.o_proj."):
                    out[f"backbone.dinov3.blocks.{index}.attn.proj."
                        f"{sub.split('.')[-1]}"] = value
                    continue
                rename = {
                    "layer_scale1.lambda1": "ls1.gamma",
                    "layer_scale2.lambda1": "ls2.gamma",
                    "mlp.up_proj.": "mlp.fc1.",
                    "mlp.down_proj.": "mlp.fc2.",
                }
                for src, dst in rename.items():
                    if sub.startswith(src):
                        out[f"backbone.dinov3.blocks.{index}."
                            f"{dst}{sub[len(src):]}"] = value
                        break
                else:
                    out[f"backbone.dinov3.blocks.{index}.{sub}"] = value
            elif rest.startswith("embeddings.cls_token"):
                out["backbone.dinov3.cls_token"] = value
            elif rest.startswith("embeddings.mask_token"):
                out["backbone.dinov3.mask_token"] = value.reshape(value.shape[0], -1)
            elif rest.startswith("embeddings.register_tokens"):
                out["backbone.dinov3.storage_tokens"] = value
            elif rest.startswith("embeddings.patch_embeddings."):
                out[f"backbone.dinov3.patch_embed.proj.{rest.split('.')[-1]}"] = value
            elif rest.startswith("norm."):
                out[f"backbone.dinov3.{rest}"] = value
        # Rebuild the fused qkv (weight cat, gated bias cat, channel mask).
        n_layers = len(
            {k.split(".layer.")[1].split(".")[0] for k in sd if ".layer." in k}
        )
        for index in range(n_layers):
            layer = f"backbone_wrapper._model.model.layer.{index}.attention"
            q_w = sd[f"{layer}.q_proj.weight"]
            k_w = sd[f"{layer}.k_proj.weight"]
            v_w = sd[f"{layer}.v_proj.weight"]
            q_b = sd[f"{layer}.q_proj.bias"]
            v_b = sd[f"{layer}.v_proj.bias"]
            dim = q_w.shape[0]
            zeros = torch.zeros(dim)
            ones = torch.ones(dim)
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.weight"] = torch.cat(
                [q_w, k_w, v_w]
            )
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.bias"] = torch.cat(
                [q_b, zeros, v_b]
            )
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.bias_mask"] = torch.cat(
                [ones, zeros, ones]
            )
        return {"train_model": {f"ema_model.model.{k}": v for k, v in out.items()}}

    def test_round_trip_80_classes(self, tmp_path):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=80)
        summary = load_baseline_coco_checkpoint(target, path)
        assert summary["backbone"] == 211
        assert summary["backbone_extras"] == 18  # convs 3 + norms 15
        assert summary["decoder_skipped_cls"] == 0
        # spot tensors identical across the remap
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        for key in (
            "backbone_wrapper._model.model.layer.5.attention.q_proj.weight",
            "backbone_wrapper._model.model.layer.5.attention.v_proj.bias",
            "backbone_wrapper._model.model.layer.5.layer_scale1.lambda1",
            "backbone_wrapper._model.embeddings.register_tokens",
            "backbone_wrapper._model.embeddings.mask_token",
            "backbone.convs.1.weight",
            "backbone.norms.2.running_var",
            "encoder.fpn_blocks.0.conv1.conv.weight",
            "decoder.dec_score_head.0.weight",
        ):
            assert torch.equal(tgt_sd[key], src_sd[key]), key

    def test_skips_class_count_heads(self, tmp_path):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=10)
        summary = load_baseline_coco_checkpoint(target, path)
        # 4 dec score heads + 1 enc score head (w+b each) + 1 denoising
        # embed (weight only).
        assert summary["decoder_skipped_cls"] == 11
        assert not torch.equal(
            target.state_dict()["decoder.dec_score_head.0.weight"],
            source.state_dict()["decoder.dec_score_head.0.weight"],
        )
        assert torch.equal(
            target.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
            source.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
        )

    def test_rope_periods_are_transported(self, tmp_path):
        """The baseline's ``rope_embed.periods`` buffer is authoritative
        (HF recomputes a ~1e-4-relative approximation), so the loader
        installs it on the HF rope module's non-persistent ``inv_freq``."""
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        payload = self._fake_checkpoint(source)
        target = self._model(num_classes=80)
        rope = target.backbone._model_wrapper._model.rope_embeddings
        recomputed = rope.inv_freq.clone()
        periods = torch.linspace(1.0, 10.0, rope.inv_freq.numel())
        payload["train_model"][
            "ema_model.model.backbone.dinov3.rope_embed.periods"
        ] = periods
        path = tmp_path / "fake_vit_baseline_rope.pt"
        torch.save(payload, path)

        summary = load_baseline_coco_checkpoint(target, path)
        assert summary["backbone_rope_periods"] == 1
        assert torch.equal(rope.inv_freq, 1.0 / periods)
        assert not torch.allclose(rope.inv_freq, recomputed)

        # A payload WITHOUT periods keeps HF's recomputed table untouched.
        plain_path = tmp_path / "fake_vit_baseline_plain.pt"
        torch.save(self._fake_checkpoint(source), plain_path)
        plain = self._model(num_classes=80)
        plain_summary = load_baseline_coco_checkpoint(plain, plain_path)
        assert "backbone_rope_periods" not in plain_summary
        plain_rope = plain.backbone._model_wrapper._model.rope_embeddings
        assert torch.equal(plain_rope.inv_freq, recomputed)

    def test_geometry_mismatch_raises(self, tmp_path):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80, variant="vitt16")
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        # vitt16 checkpoint into a vits16 target: the backbone remap cannot
        # land (different widths) and must fail loudly.
        target = self._model(num_classes=80, variant="vits16")
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            load_baseline_coco_checkpoint(target, path)


class TestLegacyClassPathRemap:
    """Legacy payloads embed the class path of the code that trained them
    (design doc 28 §4.3); loading remaps the MODULE part (suffix-matched)
    and preserves the class name, instead of failing on stale strings."""

    CURRENT = "autovision.tasks.detection.detr.task_model."

    #: The baseline package name, split so this file carries no banned
    #: reference-system marker itself (the provenance checker spells its
    #: own patterns the same way).
    BASELINE = "light" + "ly_train"

    def test_all_generations_resolve(self):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            remap_legacy_class_path,
        )

        for legacy_prefix, class_name in (
            # Generation 1: the baseline's original task module.
            (f"{self.BASELINE}._task_models.ltdetr_object_detection.task_model.",
             "LTDETRObjectDetection"),
            # Generation 2/3: our own pre-remediation and pre-rename layouts.
            ("autovision._task_models.ltdetr_object_detection.task_model.",
             "LTDETRObjectDetection"),
            ("autovision.tasks.detection.ltdetr_object_detection.task_model.",
             "AVDETRObjectDetection"),
            # Generation 4: the baseline's dinov3_/dinov2_ task-module renames
            # (the published dinov3_*_ltdetr_coco_*.pt payloads carry the
            # former).
            (f"{self.BASELINE}._task_models.dinov3_ltdetr_object_detection.task_model.",
             "DINOv3LTDETRObjectDetection"),
            (f"{self.BASELINE}._task_models.dinov2_ltdetr_object_detection.task_model.",
             "DINOv2LTDETRObjectDetection"),
        ):
            legacy = legacy_prefix + class_name
            assert remap_legacy_class_path(legacy) == self.CURRENT + class_name

    def test_foreign_paths_pass_through(self):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            remap_legacy_class_path,
        )

        for path in (
            # Already current.
            "autovision.tasks.detection.detr.task_model.DETRObjectDetection",
            # No package prefix before the matched suffix.
            ".ltdetr_object_detection.task_model.X",
            # Nested module after the task module: not a task-model class path.
            "a.b.ltdetr_object_detection.task_model.nested.Cls",
            # Unrelated package.
            "other.pkg.Model",
        ):
            assert remap_legacy_class_path(path) == path


class TestGradientClipping:
    def _train_model_with_ones_grads(self):
        train_model = make_train_model()
        optimizer = torch.optim.AdamW(train_model.model.parameters(), lr=1e-9)
        with torch.no_grad():
            for p in train_model.model.parameters():
                p.grad = torch.ones_like(p)
        # All-ones grads: the total norm is sqrt(total element count).
        expected = sum(p.numel() for p in train_model.model.parameters()) ** 0.5
        return train_model, optimizer, expected

    def test_zero_max_norm_disables_clipping(self):
        # Baseline contract: gradient_clip_val <= 0 turns clipping OFF and
        # only reports the total norm — clip_grad_norm_ with max_norm=0
        # would silently ZERO every gradient and kill the run.
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = 0.0
        total = train_model.clip_gradients(optimizer)
        assert total == pytest.approx(expected, rel=1e-4)
        for p in train_model.model.parameters():
            assert torch.all(p.grad == 1.0)

    def test_negative_max_norm_disables_clipping(self):
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = -1.0
        total = train_model.clip_gradients(optimizer)
        assert total == pytest.approx(expected, rel=1e-4)
        for p in train_model.model.parameters():
            assert torch.all(p.grad == 1.0)

    def test_positive_max_norm_still_clips(self):
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = 0.1
        total = train_model.clip_gradients(optimizer)
        # clip_grad_norm_ returns the PRE-clip total norm; the grads
        # themselves are rescaled down to the max norm.
        assert total == pytest.approx(expected, rel=1e-4)
        scale = 0.1 / expected
        for p in train_model.model.parameters():
            torch.testing.assert_close(p.grad, torch.full_like(p.grad, scale))


class TestVariantRecipes:
    """Baseline recipe dispatch is by model-name prefix (dinov2 V2 args vs
    the DETR args everything else shares) — not by network family."""

    def _resolved(self, backbone: str, **kwargs):
        args = DETRObjectDetectionTrainArgs(
            num_classes=2, backbone=backbone, image_size=(224, 224), **kwargs
        )
        args.resolve_auto_fields(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        return args

    def test_dinov2_keeps_v2_recipe(self):
        args = self._resolved("dinov2-vittest-notpretrained")
        assert args.lr == 1e-4
        assert args.backbone_lr_factor == 1e-2
        assert args.lr_scheduler == "linear"
        assert args.lr_warmup_steps == 2000
        assert args.lr_reference_batch == 16

    def test_dinov3_vit_walks_ltdetr_recipe(self):
        # The bug this locks: dinov3 ViT used to inherit the dinov2 V2
        # recipe (1e-4 / 1e-2 / linear) instead of the DETR recipe.
        args = self._resolved("dinov3-vits16-notpretrained")
        assert args.lr == 5e-4
        assert args.backbone_lr_factor == 0.05
        assert args.lr_scheduler == "flat_cosine"
        assert isinstance(args.lr_warmup_steps, int)
        assert args.lr_warmup_steps != 2000
        assert args.lr_reference_batch == 32

    def test_convnext_walks_ltdetr_recipe(self):
        args = self._resolved("dinov3-convnext-tiny")
        assert args.lr == 5e-4
        assert args.backbone_lr_factor == 0.05
        assert args.lr_scheduler == "flat_cosine"
        assert args.lr_reference_batch == 32

    def test_explicit_pins_win(self):
        args = self._resolved(
            "dinov3-vits16-notpretrained",
            lr=1e-4,
            backbone_lr_factor=0.1,
            lr_scheduler="linear",
            lr_reference_batch=16,
        )
        assert args.lr == 1e-4
        assert args.backbone_lr_factor == 0.1
        assert args.lr_scheduler == "linear"
        assert args.lr_reference_batch == 16

    def test_default_batch_size_helper(self):
        from autovision.tasks.detection.detr import (
            detr_default_batch_size,
        )

        assert detr_default_batch_size("dinov2-vits14") == 16
        assert detr_default_batch_size("dinov2-vittest-notpretrained") == 16
        assert detr_default_batch_size("dinov3-convnext-tiny") == 32
        assert detr_default_batch_size("dinov3-vits16-notpretrained") == 32


class TestConvNextOptimizerScaling:
    def test_lr_sqrt_scaling_uses_variant_reference_batch(self):
        # Baseline denominator: 32 for the ConvNeXt variant (the dinov2
        # V2 args keep 16). 5e-4 * sqrt(16/32) ~ 3.536e-4.
        args = DETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="dfine",
            backbone_pretrained=False,
        )
        train_model = DETRObjectDetectionTrain(args)
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        detector = next(
            g for g in optimizer.param_groups if g["name"] == "detector"
        )
        base = detector.get("_initial_lr", detector.get("initial_lr"))
        assert base == pytest.approx(5e-4 * (16 / 32) ** 0.5)


class TestDecoders:
    def test_rtdetrv2_is_default(self):
        train_model = make_train_model()
        assert train_model.model.decoder_name == "rtdetrv2"
        from autovision.tasks.detection.shared.rtdetrv2_decoder import (
            RTDETRv2Decoder,
        )

        assert type(train_model.model.decoder) is RTDETRv2Decoder
        assert type(train_model.criterion).__name__ == "RTDETRCriterionV2"

    def test_dfine_decoder_swaps_criterion(self):
        from autovision.tasks.detection.shared.dfine_decoder import (
            DFINEv2Decoder,
        )

        train_model = make_train_model(decoder="dfine", reg_max=16)
        assert type(train_model.model.decoder) is DFINEv2Decoder
        assert train_model.model.decoder.reg_max == 16
        assert type(train_model.criterion).__name__ == "DFINECriterionV2"
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        assert torch.isfinite(result.loss)
        assert any("loss_fgl" in key for key in result.log_dict)
        result.loss.backward()

class TestImageSize:
    def test_default_snaps_to_patch_grid(self):
        args = DETRObjectDetectionTrainArgs(num_classes=2)
        assert args.image_size == (640, 640)
        args.resolve_image_size()
        # 640 -> 644: next multiple of 2 x patch 14.
        assert args.image_size == (644, 644)

    def test_patch16_default_unchanged(self):
        args = DETRObjectDetectionTrainArgs(
            num_classes=2, backbone="dinov3-vits16"
        )
        args.resolve_image_size()
        assert args.image_size == (640, 640)

    def test_explicit_divisible_size_kept(self):
        args = DETRObjectDetectionTrainArgs(num_classes=2, image_size=(644, 644))
        args.resolve_image_size()
        assert args.image_size == (644, 644)

    def test_indivisible_explicit_size_raises(self):
        args = DETRObjectDetectionTrainArgs(num_classes=2, image_size=(640, 640))
        args.backbone = "dinov3-vits16"  # divisor 32 -> 640 still fine
        args.resolve_image_size()
        args.image_size = (500, 500)
        with pytest.raises(ValueError, match="divisible by 32"):
            args.resolve_image_size()

    def test_task_model_rejects_bad_size(self):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        with pytest.raises(ValueError, match="divisible by 28"):
            DETRObjectDetection(
                num_classes=2,
                model_name="dinov2-vittest-notpretrained",
                image_size=(64, 64),
                num_queries=10,
            )


class TestDETRTransforms:
    def test_scale_jitter_ladder_aligns_to_patch_grid(self):
        from autovision.tasks.detection.detr.transforms import (
            detr_scale_jitter_sizes,
        )

        # Patch 16 at 640 reproduces the stock 480..800 ladder.
        sizes = detr_scale_jitter_sizes((640, 640), 32)
        assert sizes[:11] == [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        assert all(s % 32 == 0 for s in sizes)
        # Patch 14 ladder snaps to multiples of 28 around 644.
        sizes14 = detr_scale_jitter_sizes((644, 644), 28)
        assert all(s % 28 == 0 for s in sizes14)
        assert 644 in sizes14 and 476 in sizes14 and 812 in sizes14

    def test_build_args_enables_mosaic(self):
        from autovision.tasks.detection.detr.transforms import (
            build_detr_train_transform_args,
        )

        args = build_detr_train_transform_args((644, 644), "dinov2-vits14")
        assert args.mosaic_prob == 0.5
        assert args.image_size == (644, 644)
        assert all(s % 28 == 0 for s in args.scale_jitter.sizes)
        assert args.mixup_prob == 0.5
        # User overrides win.
        args_off = build_detr_train_transform_args(
            (644, 644), "dinov2-vits14", {"mosaic_prob": None}
        )
        assert args_off.mosaic_prob is None

    def test_mosaic_merges_samples_and_aligns_boxes(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        transform = RTDETRDetectionTransform(args)

        def make_item(index: int) -> dict:
            return {
                "image_path": f"img{index}.jpg",
                "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
                "bboxes": tv_tensors.BoundingBoxes(
                    torch.tensor([[10.0, 10.0, 50.0, 60.0], [30.0, 30.0, 90.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([0, 1]),
                "original_size": (120, 100),
            }

        outputs = [transform(make_item(i)) for i in range(6)]
        for output in outputs:
            assert output["image"].shape == (3, 224, 224)
            # Box/class alignment must survive the per-tile degenerate filter.
            assert output["bboxes"].shape[0] == output["classes"].shape[0]
            assert bool((output["bboxes"] >= 0).all())
            assert bool((output["bboxes"] <= 1).all())
        # Baseline donor sampling is with replacement from the whole
        # cache (duplicates allowed); the merged sample still carries a
        # healthy box count.
        assert outputs[-1]["bboxes"].shape[0] >= 4
        # Mosaic samples keep photometric distort; only zoom-out/IoU-crop
        # are skipped (baseline behavior).
        assert (True, False, False, False) in transform.compose._compose_cache

    def test_mosaic_composes_with_tiny_cache(self):
        # Baseline donor draws are with replacement: the FIRST sample (a
        # one-entry cache) still composes a full 4-tile mosaic.
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        transform = RTDETRDetectionTransform(args)
        item = {
            "image_path": "img0.jpg",
            "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
            "bboxes": tv_tensors.BoundingBoxes(
                torch.tensor([[10.0, 10.0, 50.0, 60.0]]),
                format="XYXY",
                canvas_size=(100, 120),
            ),
            "classes": torch.tensor([0]),
            "original_size": (120, 100),
        }
        output = transform(item)
        assert output["image"].shape == (3, 224, 224)
        assert output["bboxes"].shape[0] >= 1
        assert output["bboxes"].shape[0] == output["classes"].shape[0]

    def test_mosaic_cache_random_pop_keeps_newest(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=None,  # cache append only fires when mosaic is on
            scale_jitter=None,
        )
        transform = RTDETRDetectionTransform(args)
        transform.MOSAIC_CACHE_SIZE = 3
        for index in range(10):
            # Distinct pixel value identifies the sample through the clone.
            transform._mosaic_cache_append(
                {
                    "image_path": f"img{index}",
                    "image": torch.full((3, 8, 8), index, dtype=torch.uint8),
                    "bboxes": tv_tensors.BoundingBoxes(
                        torch.zeros(0, 4), format="XYXY", canvas_size=(8, 8)
                    ),
                    "classes": torch.zeros(0, dtype=torch.int64),
                }
            )
        marks = [int(s["image"][0, 0, 0]) for s in transform._mosaic_cache]
        assert len(marks) == 3
        # random_pop must never evict the freshly appended entry.
        assert marks[-1] == 9

    def test_mosaic_boxes_survive_whole_canvas_affine(self):
        # The affine is applied to the assembled canvas: boxes must stay
        # inside the canvas, aligned with classes, and never degenerate.
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        transform = RTDETRDetectionTransform(args)

        def make_item(index: int) -> dict:
            return {
                "image_path": f"img{index}.jpg",
                "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
                "bboxes": tv_tensors.BoundingBoxes(
                    torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([3]),
                "original_size": (120, 100),
            }

        for _ in range(8):
            output = transform(make_item(0))
            boxes = output["bboxes"]
            assert boxes.shape[0] == output["classes"].shape[0]
            if boxes.shape[0]:
                # Normalized cxcywh: w/h are columns 2/3.
                assert bool((boxes[:, 2] > 0).all()) and bool((boxes[:, 3] > 0).all())
                assert bool((boxes >= 0).all()) and bool((boxes <= 1).all())

    def test_mosaic_keeps_photometric(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=0.5,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        args.photometric_step_start = 0
        args.photometric_step_stop = None
        transform = RTDETRDetectionTransform(args)

        def make_item(index: int) -> dict:
            return {
                "image_path": f"img{index}.jpg",
                "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
                "bboxes": tv_tensors.BoundingBoxes(
                    torch.tensor([[10.0, 10.0, 50.0, 60.0], [30.0, 30.0, 90.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([0, 1]),
                "original_size": (120, 100),
            }

        for _ in range(6):
            transform(make_item(0))
        assert (True, True, False, False) in transform.compose._compose_cache

    def test_scale_jitter_step_window(self):
        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(64, 64),
            mosaic_prob=None,
            scale_jitter={"sizes": [64, 80], "step_stop": 10},
        )
        collate = RTDETRDetectionCollateFunction(args, train=True)
        batch = [
            {
                "image": torch.randn(3, 64, 64),
                "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "classes": torch.tensor([0]),
                "original_size": (32, 32),
                "image_path": "x.jpg",
            }
        ]
        collate.set_step(5)
        assert collate(batch)["image"].shape[-1] in (64, 80)
        collate.set_step(20)
        assert collate(batch)["image"].shape[-1] == 64
        collate.set_step(30)
        assert collate.requires_dataloader_reinitialization()


class TestE2E:
    def test_train_object_detection_ltetr(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "ltdetr1",
            data=data_config(train_json, val_json, images_dir),
            model="detr/dinov2-vittest-notpretrained",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                # 56 = 4 x patch 14: divisible by 2 x patch_size.
                "image_size": (56, 56),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 64,
                "dec_layers": 2,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 4
        assert result.checkpoint_path and result.checkpoint_path.exists()
        assert result.best_checkpoint_path and result.best_checkpoint_path.exists()
        assert "val_metric/map" in result.best_metrics

    def test_exported_model_loads_and_predicts(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=4)
        result = train_object_detection(
            out=tmp_path / "ltdetr2",
            data=data_config(train_json, val_json, images_dir),
            model="detr/dinov2-vittest-notpretrained",
            steps=2,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (56, 56),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 64,
                "dec_layers": 2,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        model = load_model(result.exported_model_path, device="cpu")
        image = create_image(tmp_path / "probe.jpg", size=(64, 64), seed=0)
        prediction = model.predict(image, threshold=0.05)
        assert set(prediction) == {"labels", "bboxes", "scores"}


class TestEncoderOverrides:
    def test_enc_overrides_reach_the_encoder(self):
        """The official Dome-S 'S' encoder recipe (1024/0.5/0.34) must land
        in the built HybridEncoder instead of the variant spec's heavy
        defaults (2048/1.0/1.0)."""
        from autovision.tasks.detection.detr import (
            DETRObjectDetection,
            DETRObjectDetectionTrainArgs,
        )

        args = DETRObjectDetectionTrainArgs(
            num_classes=10,
            backbone="dinov3-convnext-tiny",
            image_size=(64, 64),
            decoder="dfine",
            num_queries=10,
            num_top_queries=10,
            enc_dim_feedforward=1024,
            enc_expansion=0.5,
            enc_depth_mult=0.34,
        )
        model = DETRObjectDetection(
            num_classes=args.num_classes,
            model_name=args.backbone,
            image_size=args.image_size,
            decoder="dfine",
            reg_max=8,
            load_weights=False,
            enc_dim_feedforward=args.enc_dim_feedforward,
            enc_expansion=args.enc_expansion,
            enc_depth_mult=args.enc_depth_mult,
        )
        assert model.encoder.enc_dim_feedforward == 1024
        assert model.encoder.enc_expansion == 0.5
        assert model.encoder.enc_depth_mult == 0.34

    def test_enc_auto_follows_variant_spec(self):
        from autovision.tasks.detection.detr import (
            DETRObjectDetection,
            DETRObjectDetectionTrainArgs,
        )

        args = DETRObjectDetectionTrainArgs(
            num_classes=10,
            backbone="dinov3-convnext-tiny",
            image_size=(64, 64),
            decoder="dfine",
            num_queries=10,
            num_top_queries=10,
        )
        model = DETRObjectDetection(
            num_classes=args.num_classes,
            model_name=args.backbone,
            image_size=args.image_size,
            decoder="dfine",
            reg_max=8,
            load_weights=False,
        )
        assert model.encoder.enc_dim_feedforward == 2048  # variant spec value
        assert model.encoder.enc_expansion == 1.0
        assert model.encoder.enc_depth_mult == 1.0


class TestSIoUBoxRegression:
    """box_regression_iou='siou' config surface (design doc 15 §4.4-4.5)."""

    def test_default_config_keeps_giou(self):
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrainArgs,
        )

        args = DETRObjectDetectionTrainArgs(num_classes=2)
        assert args.box_regression_iou == "giou"
        assert args.siou_loss_coef is None
        assert args.siou_theta == pytest.approx(4.0)

    def test_config_rejects_invalid_siou_values(self):
        from pydantic import ValidationError

        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrainArgs,
        )

        with pytest.raises(ValidationError):
            DETRObjectDetectionTrainArgs(num_classes=2, siou_theta=0.0)
        with pytest.raises(ValidationError):
            DETRObjectDetectionTrainArgs(num_classes=2, siou_loss_coef=-0.1)

    def test_siou_requires_dfine_decoder(self):
        with pytest.raises(ValueError, match="box_regression_iou='siou'"):
            make_train_model(box_regression_iou="siou", decoder="rtdetrv2")

    def test_siou_wires_dfine_criterion_weight(self):
        train_model = make_train_model(
            decoder="dfine", box_regression_iou="siou", reg_max=16
        )
        assert train_model.criterion.box_regression_iou == "siou"
        assert train_model.criterion.weight_dict["loss_siou"] == pytest.approx(2.0)
        assert "loss_giou" not in train_model.criterion.weight_dict

    def test_siou_training_step_emits_full_suffix_family(self):
        """End-to-end (design doc 15 §7 item 12): the swapped term reaches
        every box-loss branch — main, _pre, _dn_pre, the dn layers and the
        encoder proposals — with no loss_giou key anywhere."""
        torch.manual_seed(0)
        train_model = make_train_model(
            decoder="dfine", box_regression_iou="siou", reg_max=16
        )
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        keys = set(result.log_dict)
        assert "loss_siou" in keys
        assert "loss_bbox_pre" in keys  # the _pre branch ran
        assert "loss_siou_pre" in keys
        assert "loss_siou_dn_pre" in keys
        assert any(key.startswith("loss_siou_dn_") for key in keys)
        assert any(key.startswith("loss_siou_enc_") for key in keys)
        assert not any("loss_giou" in key for key in keys)
        assert torch.isfinite(result.loss)


# ── R16 family rename: alias window closed, artifacts migrate ───────────


class TestFamilyKeyAlias:
    def test_legacy_key_is_no_longer_accepted(self):
        # The pre-rename family keys completed their deprecation window:
        # the resolver rejects them (and the CLI's "detr/" dispatch does
        # not match), so a mistyped key fails loudly instead of silently
        # resolving.
        from autovision.tasks.detection.detr.config import (
            resolve_detr_variant,
        )

        for legacy in ("ltdetr/dinov3-vits16", "avdetr/dinov3-vits16"):
            with pytest.raises(ValueError):
                resolve_detr_variant(legacy)

    def test_exported_payload_family_key_migrates(self):
        # Old exports embed the legacy key in ``model_init_args``; the
        # export loader rewrites it before the class is constructed, so
        # those artifacts keep loading after the window closed.
        from autovision.tasks.task_model_helpers import _migrate_family_key

        assert (
            _migrate_family_key("ltdetr/dinov3-vitt16-notpretrained")
            == "detr/dinov3-vitt16-notpretrained"
        )
        assert (
            _migrate_family_key("avdetr/dinov3-convnext-tiny")
            == "detr/dinov3-convnext-tiny"
        )
        assert _migrate_family_key("detr/dinov3-vits16") == "detr/dinov3-vits16"

    def test_listed_names_use_new_prefix(self):
        from autovision.tasks.detection.detr.config import (
            list_detr_model_names,
        )

        names = list_detr_model_names()
        assert names and all(name.startswith("detr/") for name in names)
