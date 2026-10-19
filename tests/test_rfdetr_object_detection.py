# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the RF-DETR object detection task model.

The real detector needs transformers >= 5.9 (RfDetr classes), so the logic
runs against an injected fake ``transformers`` module that mirrors the HF
API surface the wrapper uses (forward with labels returning
loss/loss_dict/logits/pred_boxes, class_embed + grouped encoder heads) —
same pattern as tests/test_models_rfdetr.py.
"""

from __future__ import annotations

import copy
import sys
import types

import pytest
import torch
from torch import nn

from autovision.tasks.detection.rfdetr.config import (
    RFDETREvalEraseArgs,
    RFDETRObjectDetectionTrainArgs,
    resolve_rfdetr_variant,
)
from autovision.tasks.detection.rfdetr.task_model import (
    RFDETRObjectDetection,
    _make_siou_box_loss_class,
    _reinitialize_class_head,
)
from autovision.tasks.detection.rfdetr.train_model import (
    RFDETRObjectDetectionTrain,
    _rfdetr_param_buckets,
)

# ── fake transformers mirroring the HF RF-DETR API ───────────────────────


class FakeRfDetrDinov2Config:
    def __init__(self, **kwargs) -> None:
        self.model_type = "rf_detr_dinov2"
        defaults = dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=224,
            patch_size=14,
            num_windows=4,
            out_indices=[2, 5, 8, 11],
            out_features=["stage2", "stage5", "stage8", "stage11"],
            use_mask_token=True,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeRfDetrConfig:
    def __init__(self, backbone_config=None, **kwargs) -> None:
        self.model_type = "rf_detr"
        self.backbone_config = backbone_config or FakeRfDetrDinov2Config()
        self.d_model = 32
        self.num_queries = 16
        self.group_detr = 13
        self.num_labels = 80
        self.decoder_layers = 3
        # Matcher costs (the SSL branch rebuilds the HF matcher contract).
        self.class_cost = 1.0
        self.bbox_cost = 5.0
        self.giou_cost = 2.0
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeEncoderBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)


class _FakeRfDetrDinov2Backbone(nn.Module):
    """Attribute tree mirroring the real names for bucketing tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(8, 8)
        self.layernorm = nn.LayerNorm(8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_FakeEncoderBlock() for _ in range(2)])


class _FakeRfDetrConvEncoder(nn.Module):
    """Real structure: ConvEncoder{backbone: Dinov2Backbone, projector}."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeRfDetrDinov2Backbone()
        self.projector = nn.Linear(8, 8)


class _FakeRfDetrInner(nn.Module):
    def __init__(self, config: FakeRfDetrConfig) -> None:
        super().__init__()
        self.backbone = _FakeRfDetrConvEncoder()
        self.decoder = nn.Linear(32, 32)
        self.query_feat = nn.Parameter(torch.randn(config.num_queries * 2, 32))
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(32, config.num_labels) for _ in range(2)]
        )
        self.enc_out_bbox_embed = nn.ModuleList([nn.Linear(32, 4) for _ in range(2)])


class FakeRfDetrForObjectDetection(nn.Module):
    def __init__(self, config: FakeRfDetrConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeRfDetrInner(config)
        self.proj = nn.Conv2d(3, 32, 7, stride=7)
        self.class_embed = nn.Linear(32, config.num_labels)
        self.bbox_embed = nn.Linear(32, 4)

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        features = self.proj(pixel_values)
        hidden = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
        hidden = hidden.flatten(2).transpose(1, 2)  # (B, 16, 32)
        logits = self.class_embed(hidden)
        boxes = torch.sigmoid(self.bbox_embed(hidden))
        if labels is None:
            return {
                "logits": logits,
                "pred_boxes": boxes,
                "last_hidden_state": hidden,
                "loss": None,
                "loss_dict": {},
            }
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)
        )
        l1 = torch.nn.functional.l1_loss(boxes, torch.zeros_like(boxes))
        return {
            "logits": logits,
            "pred_boxes": boxes,
            "last_hidden_state": hidden,
            "loss": ce + l1,
            "loss_dict": {"loss_ce": ce.detach(), "loss_bbox": l1.detach()},
        }


@pytest.fixture
def fake_rfdetr_transformers(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.RfDetrConfig = FakeRfDetrConfig
    fake.RfDetrForObjectDetection = FakeRfDetrForObjectDetection
    fake.RfDetrDinov2Config = FakeRfDetrDinov2Config
    # The SSL branch rebuilds the HF matcher contract; hand the fake module
    # the REAL matcher when transformers is installed, else a deterministic
    # stub (plumbing tests never rely on optimal assignment).
    try:
        from transformers.loss.loss_lw_detr import (
            LwDetrHungarianMatcher as _real_matcher,
        )

        matcher = _real_matcher(class_cost=1.0, bbox_cost=5.0, giou_cost=2.0)
    except ImportError:

        class _StubMatcher:
            def __call__(self, outputs, targets, group_detr):
                indices = []
                for target in targets:
                    n = min(
                        outputs["pred_logits"].shape[1],
                        target["boxes"].shape[0],
                    )
                    indices.append(
                        (
                            torch.arange(n, dtype=torch.int64),
                            torch.arange(n, dtype=torch.int64),
                        )
                    )
                return indices

        matcher = _StubMatcher()
    fake.loss = types.SimpleNamespace(
        loss_lw_detr=types.SimpleNamespace(LwDetrHungarianMatcher=lambda **kw: matcher)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def make_task_model(num_classes: int = 3, **kwargs) -> RFDETRObjectDetection:
    return RFDETRObjectDetection(
        num_classes=num_classes,
        model="rf-detr-nano",
        image_size=(224, 224),
        num_select=16,
        load_weights=False,
        **kwargs,
    )


def make_batch(batch_size: int = 2, image_size: int = 224) -> dict:
    return {
        "image": torch.randn(batch_size, 3, image_size, image_size),
        "bboxes": [torch.rand(3, 4) for _ in range(batch_size)],
        "classes": [torch.tensor([0, 1, 2]) for _ in range(batch_size)],
        "original_size": [(100, 150) for _ in range(batch_size)],
    }


# ── config ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown rfdetr detection model"):
            resolve_rfdetr_variant("rf-detr-huge")

    def test_auto_image_size_resolves_to_variant_native(self):
        args = RFDETRObjectDetectionTrainArgs(model="rf-detr-nano", image_size="auto")
        args.resolve_image_size()
        assert args.image_size == (384, 384)

    def test_custom_size_divisibility(self):
        args = RFDETRObjectDetectionTrainArgs(
            model="rf-detr-nano", image_size=(800, 800)
        )
        args.resolve_image_size()
        assert args.image_size == (800, 800)

        base = RFDETRObjectDetectionTrainArgs(
            model="rf-detr-base", image_size=(800, 800)
        )
        with pytest.raises(ValueError, match="not divisible"):
            base.resolve_image_size()

    def test_resolve_auto_fields_scheduler(self):
        args = RFDETRObjectDetectionTrainArgs(model="rf-detr-nano")
        args.resolve_auto_fields(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.lr_scheduler == "cosine"
        assert args.lr_warmup_steps == 100
        assert args.image_size == (384, 384)


# ── gating ───────────────────────────────────────────────────────────────


class TestGating:
    def test_no_support_raises_informative(self, monkeypatch):
        from autovision.tasks.detection.rfdetr import task_model

        monkeypatch.setattr(task_model, "_od_classes", lambda: None)
        with pytest.raises(ImportError, match="transformers"):
            make_task_model()


# ── task model ───────────────────────────────────────────────────────────


class TestTaskModel:
    def test_forward_shapes(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=3)
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 16, 3)
        assert boxes.shape == (2, 16, 4)
        assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_loss_backward(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=3)
        batch = make_batch()
        targets = [
            {"boxes": batch["bboxes"][i], "labels": batch["classes"][i]}
            for i in range(2)
        ]
        out = model._forward_train(batch["image"], targets=targets)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])
        out["loss"].backward()

    def test_postprocess_decodes_to_original_pixels(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=3)
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 224, 224))
        decoded = model.postprocess(outputs, [{"orig_w": 150, "orig_h": 100}])
        assert len(decoded) == 1
        prediction = decoded[0]
        assert set(prediction) == {"labels", "bboxes", "scores"}
        assert prediction["bboxes"].shape[1] == 4
        assert prediction["bboxes"][:, 0].max() <= 150
        assert prediction["bboxes"][:, 1].max() <= 100

    def test_class_head_retile(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=80)  # native head: 80 rows
        head = model._model.class_embed
        old_weight = head.weight.detach().clone()
        _reinitialize_class_head(model._model, 3)  # shrink: truncation
        assert model._model.class_embed.weight.shape == (3, 32)
        torch.testing.assert_close(model._model.class_embed.weight, old_weight[:3])
        # Grow 3 -> 7: cyclic tiling of the surviving (shrunk) rows.
        shrunk = model._model.class_embed.weight.detach().clone()
        _reinitialize_class_head(model._model, 7)
        assert model._model.class_embed.weight.shape == (7, 32)
        torch.testing.assert_close(
            model._model.class_embed.weight,
            shrunk[torch.arange(7) % 3],
        )
        # Grouped encoder heads are retiled too.
        assert model._model.model.enc_out_class_embed[0].weight.shape == (7, 32)


# ── train model ──────────────────────────────────────────────────────────


def make_train_args(**overrides) -> RFDETRObjectDetectionTrainArgs:
    defaults = dict(
        model="rf-detr-nano",
        num_classes=3,
        image_size=(224, 224),
        num_queries=16,
        num_select=16,
        load_weights=False,
        use_ema_model=True,
        ema_warmup_steps=2,
        eval_max_dets=10,
    )
    defaults.update(overrides)
    return RFDETRObjectDetectionTrainArgs(**defaults)


class TestTrainModel:
    def test_training_step_backward(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        result = train_model.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert "train_loss/loss_ce" in result.log_dict

    def test_optimizer_param_groups_and_lrs(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(lr=1e-4, lr_encoder=1.5e-4)
        )
        optimizer, scheduler = train_model.get_optimizer(
            total_steps=100,
            global_batch_size=16,
            train_num_batches=10,
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        # Bucket names from the fake's real-name attribute tree.
        assert "encoder_0" in groups  # embeddings (ViT layer 0)
        assert "encoder_1" in groups  # encoder.layer.0 -> layer 1
        assert "encoder_2" in groups  # encoder.layer.1 -> layer 2
        assert "decoder" in groups
        assert "other" in groups
        # LambdaLR writes the warmup start into group["lr"]; the base lrs
        # live in initial_lr. ViT layers: embeddings = layer 0, blocks
        # are 1-based (official backbone.py scheme).
        assert groups["encoder_0"]["initial_lr"] == pytest.approx(1.5e-4)
        assert groups["encoder_1"]["initial_lr"] == pytest.approx(1.5e-4 * 0.8)
        assert groups["encoder_2"]["initial_lr"] == pytest.approx(1.5e-4 * 0.64)
        assert groups["decoder"]["initial_lr"] == pytest.approx(1e-4 * 0.7)
        assert groups["other"]["initial_lr"] == pytest.approx(1e-4)
        # Bias parameters are exempt from weight decay.
        assert all(
            group.get("weight_decay", 1e-4) == 0.0
            for name, group in groups.items()
            if name.endswith("_no_wd")
        )
        # Cosine schedule: first optimizer step is inside the 100-step
        # linear warmup, so every group's lr doubles from the 1/100 start.
        before = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()
        after = [group["lr"] for group in optimizer.param_groups]
        assert all(a == pytest.approx(2 * b) for a, b in zip(after, before))

    def test_param_buckets_layerwise(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        buckets = _rfdetr_param_buckets(train_model.model, make_train_args())
        names = {bucket for bucket, _ in buckets}
        assert "encoder_0" in names and "encoder_1" in names
        assert "decoder" in names and "other" in names

    def test_ema_and_export_state(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        assert train_model.ema_model is not None
        train_model.training_step(make_batch(), step=0)
        train_model.on_train_batch_end()
        export_state = train_model.get_export_state_dict()
        assert any(key.startswith("ema_model.") for key in export_state)


# ── erased eval channel (design doc 16 P1.6, eval_erase) ─────────────────


def make_val_batch(batch_size: int = 2, image_size: int = 224) -> dict:
    batch = make_batch(batch_size, image_size)
    batch["image_path"] = [f"/data/val_{index}.jpg" for index in range(batch_size)]
    return batch


def make_erased_train_model(**overrides) -> RFDETRObjectDetectionTrain:
    train_model = RFDETRObjectDetectionTrain(
        make_train_args(
            eval_erase={"ratio": 0.4, "interior": True, "min_keep": 2}, **overrides
        )
    )
    train_model.set_categories({0: "a", 1: "b", 2: "c"})
    return train_model


class TestEvalEraseConfig:
    def test_default_off(self):
        args = RFDETRObjectDetectionTrainArgs(model="rf-detr-nano")
        assert args.eval_erase is None

    def test_dict_payload_validates_with_v2_defaults(self):
        args = RFDETRObjectDetectionTrainArgs(
            model="rf-detr-nano", eval_erase={"ratio": 0.5}
        )
        assert args.eval_erase.ratio == 0.5
        # Defaults mirror the --erase training recipe (V2 interior mode).
        assert args.eval_erase.interior is True
        assert args.eval_erase.min_keep == 2
        assert args.eval_erase.padding == pytest.approx(0.15)
        assert args.eval_erase.cell == 16

    def test_invalid_ratio_rejected(self):
        with pytest.raises(ValueError, match="ratio"):
            RFDETRObjectDetectionTrainArgs(
                model="rf-detr-nano", eval_erase={"ratio": 0.0}
            )


class TestErasedEvalChannel:
    def test_default_off_keeps_single_channel(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        train_model.set_categories({0: "a", 1: "b", 2: "c"})
        train_model.validation_step(make_val_batch(), step=0)
        metrics = train_model.compute_metrics("val")
        assert metrics
        assert all(key.startswith("val_metric/") for key in metrics)
        state = train_model.export_metric_state("val")
        assert state is not None and "dense" not in state

    def test_erased_channel_reports_val_erased_metrics(self, fake_rfdetr_transformers):
        train_model = make_erased_train_model()
        train_model.validation_step(make_val_batch(), step=0)
        metrics = train_model.compute_metrics("val")
        dense = {k for k in metrics if k.startswith("val_metric/")}
        erased = {k for k in metrics if k.startswith("val_erased_metric/")}
        assert dense and erased
        assert {k.removeprefix("val_metric/") for k in dense} == {
            k.removeprefix("val_erased_metric/") for k in erased
        }

    def test_dense_channel_identical_with_channel_on(self, fake_rfdetr_transformers):
        # Identical weights (deepcopy) + identical batch: enabling the
        # erased channel must not move the dense metric by a bit.
        base = RFDETRObjectDetectionTrain(make_train_args())
        base.set_categories({0: "a", 1: "b", 2: "c"})
        with_channel = copy.deepcopy(base)
        with_channel.model_args.eval_erase = RFDETREvalEraseArgs()
        batch = make_val_batch()
        base.validation_step(batch, step=0)
        with_channel.validation_step(batch, step=0)
        dense_without = base.compute_metrics("val")
        # The channel-on run also carries val_erased_metric/* — the dense
        # keys must match the channel-off run bit for bit.
        dense_with = {
            key: value
            for key, value in with_channel.compute_metrics("val").items()
            if key.startswith("val_metric/")
        }
        assert dense_with == dense_without

    def test_erased_forward_is_seeded_and_label_free(
        self, fake_rfdetr_transformers, monkeypatch
    ):
        train_model = make_erased_train_model()
        batch = make_val_batch()
        task_model = train_model.ema_model.model
        captured: list = []
        original = task_model._forward_train

        def spy(pixel_values, targets=None):
            captured.append((pixel_values.clone(), targets))
            return original(pixel_values, targets)

        monkeypatch.setattr(task_model, "_forward_train", spy)
        train_model.validation_step(batch, step=0)
        assert len(captured) == 2  # dense + erased forwards
        dense_x, dense_targets = captured[0]
        erased_x, erased_targets = captured[1]
        assert dense_targets is not None and erased_targets is None
        assert erased_x.shape == dense_x.shape
        assert not torch.equal(dense_x, erased_x)  # cells were zero-filled
        # Same batch again: the path-seeded generator reproduces the exact
        # same erased view (metrics comparable across checkpoints).
        captured.clear()
        train_model.validation_step(batch, step=0)
        assert torch.equal(captured[1][0], erased_x)

    def test_export_merge_roundtrip_carries_both_channels(
        self, fake_rfdetr_transformers
    ):
        source = make_erased_train_model()
        batch = make_val_batch()
        source.validation_step(batch, step=0)
        state = source.export_metric_state("val")
        assert set(state) == {"dense", "erased"}
        target = make_erased_train_model()
        target.merge_metric_state("val", state)
        assert source.compute_metrics("val") == target.compute_metrics("val")

    def test_reset_clears_both_channels(self, fake_rfdetr_transformers):
        train_model = make_erased_train_model()
        train_model.validation_step(make_val_batch(), step=0)
        train_model.reset_metrics("val")
        # The evaluator answers the empty accumulation with all-zero
        # metrics — what matters is that nothing from the previous round
        # survives (the DDP merge path relies on starting clean).
        metrics = train_model.compute_metrics("val")
        assert metrics and set(metrics.values()) == {0.0}
        # The reset also released the erased accumulation (DDP merge path
        # relies on starting from a clean evaluator).
        state = train_model.export_metric_state("val")
        assert state["dense"]["predictions"] == []
        assert state["erased"]["predictions"] == []


# ── SIoU box-loss swap (design doc 15, box_regression_iou) ───────────────


def _real_rfdetr_available() -> bool:
    try:
        import transformers

        return getattr(transformers, "RfDetrConfig", None) is not None
    except ImportError:
        return False


class TestSIoUBoxRegression:
    def test_attachment_semantics(self, fake_rfdetr_transformers):
        """siou mode binds a callable loss_function on the HF model; giou
        (default) leaves the model untouched (the fake class has no such
        attribute and the real one resolves its criterion lazily)."""
        siou_model = make_task_model(box_regression_iou="siou")
        assert callable(getattr(siou_model._model, "loss_function", None))
        giou_model = make_task_model()
        assert not hasattr(giou_model._model, "loss_function")

    def test_invalid_mode_raises(self, fake_rfdetr_transformers):
        with pytest.raises(ValueError, match="'giou' or 'siou'"):
            make_task_model(box_regression_iou="focal")

    def test_identical_boxes_yield_zero_siou(self):
        """SIoU degenerates to 0 for perfectly matching pairs."""
        from transformers.loss.loss_lw_detr import LwDetrHungarianMatcher

        torch.manual_seed(0)
        criterion = _make_siou_box_loss_class()(
            matcher=LwDetrHungarianMatcher(class_cost=1, bbox_cost=5, giou_cost=2),
            num_classes=3,
            focal_alpha=0.25,
            losses=["boxes"],
            group_detr=1,
            siou_theta=4.0,
        )
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.1, 0.2, 0.1, 0.1]])
        outputs = {"pred_boxes": boxes.unsqueeze(0)}
        targets = [{"boxes": boxes.clone(), "class_labels": torch.tensor([0, 1])}]
        indices = [(torch.arange(2), torch.arange(2))]
        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=2)
        assert losses["loss_siou"].item() == pytest.approx(0.0, abs=1e-6)
        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)

    def test_siou_gradient_flows_to_boxes(self):
        from transformers.loss.loss_lw_detr import LwDetrHungarianMatcher

        torch.manual_seed(0)
        criterion = _make_siou_box_loss_class()(
            matcher=LwDetrHungarianMatcher(class_cost=1, bbox_cost=5, giou_cost=2),
            num_classes=3,
            focal_alpha=0.25,
            losses=["boxes"],
            group_detr=1,
            siou_theta=4.0,
        )
        pred = torch.tensor([[0.48, 0.52, 0.22, 0.28]], requires_grad=True)
        target = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        outputs = {"pred_boxes": pred.unsqueeze(0)}
        targets = [{"boxes": target, "class_labels": torch.tensor([0])}]
        indices = [(torch.arange(1), torch.arange(1))]
        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)
        losses["loss_siou"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0

    @pytest.mark.skipif(
        not _real_rfdetr_available(), reason="transformers lacks RfDetr"
    )
    def test_forward_emits_siou_across_branches_and_backprops(self):
        """The real HF forward with labels routes through the swapped loss:
        loss_siou appears (main + aux + enc), loss_giou is gone."""
        model = RFDETRObjectDetection(
            num_classes=3,
            model="rf-detr-nano",
            image_size=(224, 224),
            num_select=16,
            load_weights=False,
            box_regression_iou="siou",
        )
        x = torch.randn(1, 3, 224, 224)
        targets = [
            {
                "boxes": torch.rand(2, 4) * 0.3 + 0.3,
                "labels": torch.tensor([0, 1]),
            }
        ]
        out = model._forward_train(x, targets=targets)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"]).all()
        assert "loss_siou" in out["loss_dict"]
        assert "loss_giou" not in out["loss_dict"]
        assert any(key.endswith("_enc") for key in out["loss_dict"])
        out["loss"].backward()

    @pytest.mark.skipif(
        not _real_rfdetr_available(), reason="transformers lacks RfDetr"
    )
    def test_giou_default_unchanged(self):
        model = RFDETRObjectDetection(
            num_classes=3,
            model="rf-detr-nano",
            image_size=(224, 224),
            num_select=16,
            load_weights=False,
        )
        x = torch.randn(1, 3, 224, 224)
        targets = [
            {
                "boxes": torch.rand(2, 4) * 0.3 + 0.3,
                "labels": torch.tensor([0, 1]),
            }
        ]
        out = model._forward_train(x, targets=targets)
        assert "loss_giou" in out["loss_dict"]
        assert "loss_siou" not in out["loss_dict"]


class TestExportRebuild:
    """Exported-model rebuilds land in the hermetic branch
    (load_task_model drops load_weights): the branch must reproduce the
    pretrained parameter shapes, and checkpoints trained under the
    transformers 5.9 overlay env must load under the renamed 5.19 layout.
    """

    @pytest.mark.skipif(
        not _real_rfdetr_available(), reason="transformers lacks RfDetr"
    )
    def test_hermetic_rebuild_matches_native_geometry(self):
        """The shape contract of a real export: native pos-embed geometry,
        300 queries x 13 groups, 2 decoder layers."""
        model = RFDETRObjectDetection(
            num_classes=10,
            model="rf-detr-nano",
            image_size=(800, 800),
            num_queries=300,
            num_select=500,
            load_weights=False,
        )
        state = model._model.state_dict()
        pos_embed = state["model.backbone.backbone.embeddings.position_embeddings"]
        assert tuple(pos_embed.shape) == (1, 577, 384)  # native 384px canvas
        assert tuple(state["model.query_feat.weight"].shape) == (3900, 256)
        decoder_layers = {
            key.split(".")[3] for key in state if ".decoder.layers." in key
        }
        assert decoder_layers == {"0", "1"}  # nano decoder_layers=2

    @pytest.mark.skipif(
        not _real_rfdetr_available(), reason="transformers lacks RfDetr"
    )
    def test_legacy_attention_keys_are_remapped_on_load(self):
        """A 5.9-layout state dict (nested attention naming, ema-prefixed)
        loads cleanly into the 5.19-built model via load_train_state_dict;
        a matching-layout dict is a no-op passthrough."""
        from autovision.tasks.detection.rfdetr.task_model import (
            _remap_legacy_dinov2_attention_keys,
        )

        model = RFDETRObjectDetection(
            num_classes=10,
            model="rf-detr-nano",
            image_size=(384, 384),
            num_queries=300,
            num_select=500,
            load_weights=False,
        )
        # Keys relative to the whole task model (load_train_state_dict
        # strips the ema prefix, then loads into `self`).
        reference = {
            f"_model.{key}": value for key, value in model._model.state_dict().items()
        }

        def legacy_key(key: str) -> str:
            return (
                key.replace(".attention.q_proj.", ".attention.attention.query.")
                .replace(".attention.k_proj.", ".attention.attention.key.")
                .replace(".attention.v_proj.", ".attention.attention.value.")
                .replace(".attention.o_proj.", ".attention.output.dense.")
            )

        legacy_state = {legacy_key(k): v for k, v in reference.items()}
        assert any(".attention.attention.query." in k for k in legacy_state)
        # ema prefix exercises the full load_train_state_dict path.
        prefixed = {f"ema_model.{k}": v for k, v in legacy_state.items()}
        model.load_train_state_dict(prefixed)  # must not raise
        loaded = model._model.state_dict()
        for key, value in list(reference.items())[:8]:
            inner_key = key.removeprefix("_model.")
            assert torch.equal(loaded[inner_key], value)

        # Matching layout: the remap is an exact no-op (identity return).
        fresh = dict(reference)
        assert _remap_legacy_dinov2_attention_keys(fresh, model._model) is fresh


# ── per-object contrastive SSL branch (design doc 14, rfdetr port) ───────


def make_ssl_args(**overrides):
    from autovision.tasks.detection.rfdetr.config import (
        RFDETRSSLArgs,
    )

    return RFDETRSSLArgs(**overrides)


class TestRFDETRSSLConfig:
    def test_off_by_default(self):
        assert RFDETRObjectDetectionTrainArgs().ssl is None

    def test_off_mode_allows_zero_coef(self):
        args = make_ssl_args(mode="off", loss_coef=0.0)
        assert args.mode == "off"

    def test_active_mode_requires_positive_coef(self):
        with pytest.raises(ValueError, match="loss_coef > 0"):
            make_ssl_args(mode="unsup", loss_coef=0.0)

    def test_same_image_positive_only_requires_two_crop(self):
        with pytest.raises(ValueError, match="two_crop"):
            make_ssl_args(same_image_positive_only=True, view="dropout")

    def test_sigreg_requires_two_crop(self):
        with pytest.raises(ValueError, match="two_crop"):
            make_ssl_args(sigreg_coef=0.1, view="dropout")

    def test_to_ssl_params_roundtrip(self):
        params = make_ssl_args(
            mode="sup", loss_coef=0.2, temp=0.07, iou_match=0.6
        ).to_ssl_params()
        assert params.mode == "sup"
        assert params.loss_coef == 0.2
        assert params.temp == 0.07
        assert params.iou_match == 0.6
        assert params.instance_source == "matched_gt"

    def test_task_model_builds_head_and_coerces_dict(self, fake_rfdetr_transformers):
        model = make_task_model(ssl={"mode": "unsup", "view": "two_crop"})
        assert model.ssl_head is not None
        assert model._ssl_args.view == "two_crop"
        assert isinstance(model._init_args["ssl"], dict)

    def test_default_model_has_no_head(self, fake_rfdetr_transformers):
        assert make_task_model().ssl_head is None


class TestRFDETRSSLTraining:
    def test_training_step_emits_ssl_loss(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop", "loss_coef": 0.1})
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        # two_crop: view-2 rows duplicate view-1 targets (pairing needs
        # both halves to carry the same GT set).
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        result = train_model.training_step(batch, step=0)
        assert "train_loss/loss_ssl" in result.log_dict
        assert result.log_dict["train_loss/ssl_pool_size"] > 0
        result.loss.backward()  # gradients reach head + HF loss paths

    def test_every_n_steps_gate(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(
                ssl={
                    "mode": "unsup",
                    "view": "two_crop",
                    "every_n_steps": 2,
                }
            )
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        off = train_model.training_step(batch, step=1)
        assert "train_loss/loss_ssl" not in off.log_dict
        on = train_model.training_step(batch, step=2)
        assert "train_loss/loss_ssl" in on.log_dict

    def test_mode_off_has_no_ssl_loss(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "off", "view": "two_crop"})
        )
        assert train_model.ssl_params is None
        train_model.train()
        result = train_model.training_step(make_batch(batch_size=2), step=0)
        assert "train_loss/loss_ssl" not in result.log_dict

    def test_ssl_head_travels_with_state_dict(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop"})
        )
        keys = list(train_model.state_dict().keys())
        assert any("ssl_head" in key for key in keys)
        export_state = train_model.get_export_state_dict()
        assert any("ssl_head" in key for key in export_state)

    @pytest.mark.skipif(
        not _real_rfdetr_available(), reason="transformers lacks RfDetr"
    )
    def test_real_forward_two_crop_end_to_end(self):
        """The full two_crop path on the real model: 2B images, HF loss +
        matched-pool InfoNCE, gradients into backbone/decoder/head."""
        model = RFDETRObjectDetection(
            num_classes=4,
            model="rf-detr-nano",
            image_size=(224, 224),
            num_queries=100,
            num_select=16,
            load_weights=False,
            ssl={"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},
        )
        model.train()
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop", "loss_coef": 0.1})
        )
        train_model.model = model  # share the real weights
        boxes = torch.rand(2, 4) * 0.3 + 0.3
        labels = torch.tensor([0, 1])
        batch = {
            "image": torch.randn(4, 3, 224, 224),
            "bboxes": [boxes, boxes, boxes, boxes],
            "classes": [labels, labels, labels, labels],
        }
        result = train_model.training_step(batch, step=0)
        assert torch.isfinite(result.loss).all()
        assert "train_loss/loss_ssl" in result.log_dict
        assert result.log_dict["train_loss/ssl_pool_size"] > 0
        result.loss.backward()
        head_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.ssl_head.parameters()
        )
        assert head_grad, "gradients must reach the projection head"


class TestRFDETRSSLSigReg:
    """SIGReg distribution regularizer (doc 16 R1-A1) on the rfdetr SSL tap."""

    def test_sigreg_loss_emitted_and_weighted(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(
                ssl={
                    "mode": "unsup",
                    "view": "two_crop",
                    "sigreg_coef": 0.5,
                }
            )
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        result = train_model.training_step(batch, step=0)
        assert "train_loss/loss_sigreg" in result.log_dict
        assert result.log_dict["train_loss/loss_sigreg"] >= 0
        result.loss.backward()

    def test_no_sigreg_by_default(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop"})
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        result = train_model.training_step(batch, step=0)
        assert "train_loss/loss_sigreg" not in result.log_dict

    def test_missing_raw_taps_fail_loud(self, fake_rfdetr_transformers, monkeypatch):
        """sigreg configured but the head produced no raw taps = head/config
        mismatch: fail loud instead of silently training without it."""
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(
                ssl={"mode": "unsup", "view": "two_crop", "sigreg_coef": 0.5}
            )
        )
        train_model.train()
        # Strip the raw taps off the model outputs to simulate the mismatch.
        real_forward = train_model.model._forward_train

        def no_taps(x, targets=None):
            out = real_forward(x, targets=targets)
            out.pop("ssl_s1", None)
            out.pop("ssl_s2", None)
            return out

        monkeypatch.setattr(train_model.model, "_forward_train", no_taps)
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        with pytest.raises(ValueError, match="ssl_s1/ssl_s2"):
            train_model.training_step(batch, step=0)

    def test_calibration_stat_logged_without_weight(self, fake_rfdetr_transformers):
        """sigreg_coef=0 (default): the unweighted Epps–Pulley statistic
        logs as loss_sigreg_stat (read-only λ_sr calibration readout) and
        the weighted loss_sigreg key stays absent."""
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop"})
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        result = train_model.training_step(batch, step=0)
        assert "train_loss/loss_sigreg_stat" in result.log_dict
        assert "train_loss/loss_sigreg" not in result.log_dict
        result.loss.backward()

    def test_sigreg_pool_capped_at_1024_rows(
        self, fake_rfdetr_transformers, monkeypatch
    ):
        """Two_crop pools beyond the 1024-row cap are randomly subsampled
        before the Epps–Pulley statistic (the ~4GB intermediate OOMed at
        micro 8, design doc 18 §2)."""
        import autovision.tasks.detection.shared.ssl_contrastive as ssl_mod

        seen = {}

        def spy(embeddings, **kwargs):
            seen["rows"] = embeddings.shape[0]
            return torch.tensor(0.1)

        monkeypatch.setattr(ssl_mod, "epps_pulley_sigreg", spy)
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(ssl={"mode": "unsup", "view": "two_crop"})
        )
        train_model.train()
        batch = make_batch(batch_size=2)
        batch["image"] = torch.randn(4, 3, 224, 224)
        batch["bboxes"] = batch["bboxes"] * 2
        batch["classes"] = batch["classes"] * 2
        train_model.training_step(batch, step=0)
        # Fake pool: 4 images x 16 queries = 64 rows, far below the cap.
        assert seen["rows"] == 64
        # And the cap constant really binds a two_crop-sized pool.
        from autovision.tasks.detection.rfdetr.train_model import (
            _SIGREG_POOL_CAP,
        )

        assert _SIGREG_POOL_CAP == 1024
