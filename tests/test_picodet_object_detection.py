"""Tests for the PicoDet object detection family (ESNet + CSP-PAN + dense head)."""

import pytest
import torch
from tests.conftest import create_image
from tests.test_train_object_detection import (
    data_config,
    make_coco_dataset,
    transform_args,
)

from autovision import load_task_model as load_model
from autovision import train_object_detection
from autovision.tasks.detection.picodet import (
    PicoDetObjectDetection,
    PicoDetObjectDetectionTrain,
    PicoDetObjectDetectionTrainArgs,
)
from autovision.tasks.detection.picodet.csp_pan import CSPPAN
from autovision.tasks.detection.picodet.esnet import (
    ESNet,
    channel_shuffle,
    resolve_backbone_scale,
)
from autovision.tasks.detection.picodet.losses import (
    PicoCriterion,
    distribution_focal_loss,
    varifocal_loss,
)
from autovision.tasks.detection.picodet.pico_head import PicoHead
from autovision.tasks.detection.picodet.postprocessor import (
    decode_pico_outputs,
)
from autovision.tasks.detection.picodet.sim_ota_assigner import (
    SimOTAAssigner,
    TaskAlignedAssigner,
)

STRIDES = (8, 16, 32)
HEAD_CHANNELS = (96, 96, 96)
# 64px canvas: 8x8 + 4x4 + 2x2 anchor points.
N_ANCHORS = 84


def make_feats(batch: int = 2):
    return [
        torch.randn(batch, 96, 8, 8),
        torch.randn(batch, 96, 4, 4),
        torch.randn(batch, 96, 2, 2),
    ]


def make_targets(batch: int = 2):
    torch.manual_seed(1)
    return [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])}
        for _ in range(batch)
    ]


def make_head(**overrides) -> PicoHead:
    torch.manual_seed(0)
    return PicoHead(
        num_classes=2, in_channels=HEAD_CHANNELS, strides=STRIDES, **overrides
    )


def make_anchor_grid(stride: int = 8, n: int = 8) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(n), torch.arange(n), indexing="ij"
    )
    return (
        torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], dim=-1)
        .reshape(-1, 2)
        .float()
    )


def make_assigner_data(topk_scores: bool = True):
    """64px canvas, stride-8 anchor grid; two GT boxes with in-box anchors
    predicting the GT box exactly (IoU 1) and a high class score."""
    anchors = make_anchor_grid()
    gt_xyxy = torch.tensor([[8.0, 8.0, 24.0, 24.0], [42.0, 42.0, 54.0, 54.0]])
    gt_labels = torch.tensor([0, 1])
    in1 = (
        (anchors[:, 0] >= 8) & (anchors[:, 0] <= 24)
        & (anchors[:, 1] >= 8) & (anchors[:, 1] <= 24)
    )
    in2 = (
        (anchors[:, 0] >= 42) & (anchors[:, 0] <= 54)
        & (anchors[:, 1] >= 42) & (anchors[:, 1] <= 54)
    )
    pred_boxes = torch.full((64, 4), 1.0)  # degenerate far boxes
    pred_boxes[in1] = gt_xyxy[0]
    pred_boxes[in2] = gt_xyxy[1]
    pred_logits = torch.full((64, 2), -4.595)  # ≈ 0.01 probability
    if topk_scores:
        pred_logits[in1, 0] = 2.197  # ≈ 0.9
        pred_logits[in2, 1] = 2.197
    return pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels, in1, in2


class TestESNet:
    def test_pyramid_strides_and_channels(self):
        model = ESNet(scale=1.0)
        c2, c3, c4 = model(torch.randn(1, 3, 64, 64))
        assert c2.shape == (1, 128, 8, 8)
        assert c3.shape == (1, 256, 4, 4)
        assert c4.shape == (1, 512, 2, 2)
        assert model.out_channels == (128, 256, 512)

    def test_scaled_channels(self):
        model = ESNet(scale=0.75)
        assert model.out_channels == (96, 192, 384)

    def test_scale_aliases(self):
        assert resolve_backbone_scale("s") == 1.0
        assert resolve_backbone_scale("xs") == 0.75
        assert resolve_backbone_scale("m") == 1.25
        assert resolve_backbone_scale("l") == 1.5
        assert resolve_backbone_scale("1.5") == pytest.approx(1.5)
        with pytest.raises(ValueError):
            resolve_backbone_scale("resnet18")

    def test_channel_shuffle(self):
        x = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        shuffled = channel_shuffle(x, groups=2)
        assert shuffled.flatten().tolist() == [0.0, 2.0, 1.0, 3.0]

    def test_gradient_flow(self):
        model = ESNet(scale=0.75)
        c2, c3, c4 = model(torch.randn(1, 3, 64, 64))
        (c2.sum() + c3.sum() + c4.sum()).backward()
        assert model.stage4.branch2[0][0].weight.grad is not None


class TestCSPan:
    def test_output_shapes_and_shared_width(self):
        neck = CSPPAN((128, 256, 512), hidden_channels=96)
        out = neck(
            [torch.randn(1, 128, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 512, 2, 2)]
        )
        assert [o.shape for o in out] == [
            (1, 96, 8, 8), (1, 96, 4, 4), (1, 96, 2, 2),
        ]
        assert neck.out_channels == (96, 96, 96)

    def test_gradient_flow(self):
        neck = CSPPAN((128, 256, 512), hidden_channels=32)
        out = neck(
            [torch.randn(1, 128, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 512, 2, 2)]
        )
        sum(o.sum() for o in out).backward()
        assert neck.lateral[2][0].weight.grad is not None


class TestPicoHead:
    def test_forward_contract(self):
        head = make_head()
        out = head(make_feats())
        assert out["pred_logits"].shape == (2, N_ANCHORS, 2)
        assert out["pred_boxes"].shape == (2, N_ANCHORS, 4)
        assert out["pred_dist"].shape == (2, N_ANCHORS, 4, 8)
        assert out["anchor_points"].shape == (N_ANCHORS, 2)
        assert out["anchor_strides"].shape == (N_ANCHORS,)
        assert torch.isfinite(out["pred_boxes"]).all()
        assert (out["pred_boxes"][..., 2:] >= 0).all()

    def test_anchor_grid_layout(self):
        head = make_head()
        out = head(make_feats(1))
        anchors = out["anchor_points"]
        assert anchors[0].tolist() == [4.0, 4.0]  # first stride-8 cell center
        assert anchors[8].tolist() == [4.0, 12.0]  # row-major ordering
        assert out["anchor_strides"][:64].unique().item() == 8.0
        assert out["anchor_strides"][64:80].unique().item() == 16.0
        assert out["anchor_strides"][80:].unique().item() == 32.0

    def test_zero_distance_decodes_to_anchor(self):
        head = make_head()
        with torch.no_grad():
            for conv in head.reg_convs:
                torch.nn.init.zeros_(conv.weight)
                torch.nn.init.zeros_(conv.bias)
                conv.bias[0::head.reg_max + 1] = 40.0  # exact one-hot bin 0 per side
        out = head(make_feats(1))
        torch.testing.assert_close(
            out["pred_boxes"][0, ..., 0] * 64, out["anchor_points"][:, 0],
            atol=1e-4, rtol=1e-3,
        )
        torch.testing.assert_close(
            out["pred_boxes"][0, ..., 1] * 64, out["anchor_points"][:, 1],
            atol=1e-4, rtol=1e-3,
        )
        torch.testing.assert_close(
            out["pred_boxes"][0, ..., 2:], torch.zeros_like(out["pred_boxes"][0, ..., 2:]),
            atol=1e-5, rtol=1e-3,
        )

    def test_gradient_flow(self):
        head = make_head()
        out = head(make_feats())
        (out["pred_logits"].sum() + out["pred_boxes"].sum()).backward()
        assert head.stems[0][0][0].weight.grad is not None
        assert head.cls_convs[0].weight.grad is not None


class TestLosses:
    def test_varifocal_perfect_logits_near_zero(self):
        target = torch.zeros(3, 2)
        target[0, 0] = 1.0  # one positive with quality 1
        pred_logits = torch.full((3, 2), -12.0)  # ≈ 0 everywhere
        pred_logits[0, 0] = 12.0  # ≈ 1
        loss = varifocal_loss(pred_logits, target)
        assert loss.sum().item() == pytest.approx(0.0, abs=1e-4)

    def test_varifocal_wrong_logits_positive(self):
        target = torch.zeros(2, 2)
        target[0, 0] = 1.0
        pred_logits = torch.zeros(2, 2)  # p = 0.5 everywhere
        loss = varifocal_loss(pred_logits, target)
        assert loss.sum().item() > 0.1

    def test_varifocal_quality_weights_positive_term(self):
        # Same (wrong) prediction, higher-quality target -> larger loss.
        target_hi = torch.zeros(1, 2)
        target_lo = torch.zeros(1, 2)
        target_hi[0, 0] = 1.0
        target_lo[0, 0] = 0.3
        pred_logits = torch.logit(torch.full((1, 2), 0.2))
        assert varifocal_loss(pred_logits, target_hi).sum() > varifocal_loss(
            pred_logits, target_lo
        ).sum()

    def test_dfl_matches_soft_target(self):
        import math

        # One-hot at bin 1 with target 1.0 -> zero loss.
        dist = torch.full((1, 4, 8), -20.0)
        dist[:, :, 1] = 20.0
        loss = distribution_focal_loss(dist, torch.full((1, 4), 1.0))
        assert loss.sum().item() == pytest.approx(0.0, abs=1e-5)
        # 0.5/0.5 between bins 1 and 2 with target 1.5: the cross-entropy
        # floor of a fractional target is ln(2) per side (standard GFL DFL —
        # constant offset, zero gradient at the optimum).
        dist = torch.full((1, 4, 8), -20.0)
        dist[:, :, 1] = torch.log(torch.tensor(0.5))
        dist[:, :, 2] = torch.log(torch.tensor(0.5))
        loss = distribution_focal_loss(dist, torch.full((1, 4), 1.5))
        assert loss.sum().item() == pytest.approx(4 * math.log(2.0), abs=1e-4)
        # A one-hot distribution on a fractional target is strictly worse.
        dist_onehot = torch.full((1, 4, 8), -20.0)
        dist_onehot[:, :, 1] = 20.0
        assert distribution_focal_loss(
            dist_onehot, torch.full((1, 4), 1.5)
        ).sum().item() > 4 * math.log(2.0)

    def test_dfl_out_of_range_saturates_last_bin(self):
        dist = torch.full((1, 4, 8), -20.0)
        dist[:, :, 7] = 20.0
        loss = distribution_focal_loss(dist, torch.full((1, 4), 100.0))
        assert loss.sum().item() == pytest.approx(0.0, abs=1e-5)


class TestCriterion:
    def make_criterion(self, **overrides) -> PicoCriterion:
        return PicoCriterion(num_classes=2, **overrides)

    def test_loss_keys(self):
        head = make_head()
        preds = head(make_feats())
        losses = self.make_criterion()(preds, make_targets(), image_size=(64, 64))
        for key in ("loss_vfl", "loss_giou", "loss_dfl", "loss"):
            assert key in losses, key
        assert torch.isfinite(losses["loss"]).item()

    def test_empty_gt_background_only(self):
        head = make_head()
        preds = head(make_feats(1))
        empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
        losses = self.make_criterion()(preds, empty, image_size=(64, 64))
        assert torch.isfinite(losses["loss"]).item()
        assert losses["loss"].item() > 0.0
        assert losses["loss_giou"].item() == pytest.approx(0.0)
        assert losses["loss_dfl"].item() == pytest.approx(0.0)

    def test_taskaligned_path(self):
        head = make_head()
        preds = head(make_feats())
        losses = self.make_criterion()(preds, make_targets(), image_size=(64, 64))
        assert torch.isfinite(losses["loss"]).item()

    def test_unknown_assigner_raises(self):
        with pytest.raises(ValueError):
            PicoCriterion(num_classes=2, assigner="atss")

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(0)
        head = make_head()
        criterion = self.make_criterion()
        targets = make_targets()
        feats = make_feats()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

        def step_loss():
            optimizer.zero_grad()
            preds = head(feats)
            losses = criterion(preds, targets, image_size=(64, 64))
            losses["loss"].backward()
            optimizer.step()
            return float(losses["loss"].detach())

        first = sum(step_loss() for _ in range(5)) / 5
        last = sum(step_loss() for _ in range(5)) / 5
        assert last < first


class TestAssigners:
    def test_simota_assigns_in_box_anchors(self):
        pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels, in1, in2 = make_assigner_data()
        result = SimOTAAssigner(topk=10).assign(
            pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels
        )
        assert result.fg_mask.sum().item() == 8  # in-box candidates only
        assert (result.fg_mask & ~(in1 | in2)).sum().item() == 0
        idx9 = (anchors[:, 0] == 12.0) & (anchors[:, 1] == 12.0)
        assert result.assigned_gt_idx[idx9].item() == 0
        assert result.assigned_quality[idx9].item() == pytest.approx(1.0)
        assert result.loss_weight[idx9].item() == pytest.approx(1.0)

    def test_simota_dynamic_k_respects_topk(self):
        pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels, *_ = make_assigner_data()
        result = SimOTAAssigner(topk=2).assign(
            pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels
        )
        assert 2 <= result.fg_mask.sum().item() <= 4

    def test_simota_empty_gt(self):
        pred_logits, pred_boxes, anchors, *_ = make_assigner_data()
        result = SimOTAAssigner().assign(
            pred_logits, pred_boxes, anchors,
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
        )
        assert result.fg_mask.sum().item() == 0
        assert (result.assigned_gt_idx == -1).all()

    def test_taskaligned_assigns_perfect_anchors(self):
        pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels, *_ = make_assigner_data()
        result = TaskAlignedAssigner(topk=3).assign(
            pred_logits, pred_boxes, anchors, gt_xyxy, gt_labels
        )
        assert result.fg_mask.sum().item() == 6  # top-3 per GT
        idx9 = (anchors[:, 0] == 12.0) & (anchors[:, 1] == 12.0)
        assert result.assigned_gt_idx[idx9].item() == 0
        assert result.assigned_quality[idx9].item() == pytest.approx(1.0)
        assert result.loss_weight[idx9].item() == pytest.approx(1.0)

    def test_taskaligned_empty_gt(self):
        pred_logits, pred_boxes, anchors, *_ = make_assigner_data()
        result = TaskAlignedAssigner().assign(
            pred_logits, pred_boxes, anchors,
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
        )
        assert result.fg_mask.sum().item() == 0


class TestPicoDetTaskModel:
    def make_task_model(self, **overrides) -> PicoDetObjectDetection:
        return PicoDetObjectDetection(
            num_classes=2, image_size=(64, 64),
            neck_channels=32, reg_max=7, num_top_queries=20,
            **{"backbone": "s", **overrides},
        )

    def test_forward_shapes(self):
        model = self.make_task_model()
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, N_ANCHORS, 2)
        assert boxes.shape == (1, N_ANCHORS, 4)
        assert torch.isfinite(boxes).all()

    def test_init_args_record_config(self):
        model = self.make_task_model()
        assert model.init_args["reg_max"] == 7
        assert model.init_args["backbone"] == "s"

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError):
            self.make_task_model(backbone="resnet18")

    def test_gradient_flow_through_backward(self):
        model = self.make_task_model()
        model.train()
        logits, boxes = model(torch.randn(1, 3, 64, 64))
        (logits.sum() + boxes.sum()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads

    def test_postprocess_maps_to_original_pixels(self):
        # A known normalized cxcywh box must map to exact original pixels
        # (single rescale, class-offset NMS keeps the single detection).
        model = self.make_task_model()
        model.eval()
        raw = (
            torch.tensor([[[8.0]]]),
            torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
        )
        results = model.postprocess(raw, [{"orig_w": 128, "orig_h": 96}], threshold=0.0)
        xyxy = results[0]["bboxes"]
        torch.testing.assert_close(
            xyxy, torch.tensor([[32.0, 24.0, 96.0, 72.0]])
        )
        assert results[0]["labels"].item() == 0
        assert results[0]["scores"].item() == pytest.approx(1.0, abs=1e-3)

    def test_nms_suppresses_duplicates(self):
        # Two identical same-class detections must collapse to one.
        raw = (
            torch.tensor([[[8.0, -8.0], [8.0, -8.0]]]),
            torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]]),
        )
        results = decode_pico_outputs(
            raw, [{"orig_w": 64, "orig_h": 64}],
            num_top_queries=4, threshold=0.1, nms_iou_threshold=0.5,
        )
        assert results[0]["bboxes"].shape[0] == 1


class TestPicoDetTrainModel:
    def make_train_model(self, **overrides) -> PicoDetObjectDetectionTrain:
        args = PicoDetObjectDetectionTrainArgs(
            num_classes=2, image_size=(64, 64), neck_channels=32, **overrides
        )
        return PicoDetObjectDetectionTrain(args)

    def make_batch(self, size: int = 64, images: int = 2):
        targets = make_targets(images)
        return {
            "image": torch.randn(images, 3, size, size),
            "bboxes": [t["boxes"] for t in targets],
            "classes": [t["labels"] for t in targets],
            "image_path": [f"img_{i}.jpg" for i in range(images)],
            "original_size": [(size, size)] * images,
        }

    def test_training_step_backward(self):
        train_model = self.make_train_model()
        train_model.train()
        result = train_model.training_step(self.make_batch(), step=0)
        assert torch.isfinite(result.loss).item()
        result.loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in train_model.model.head.parameters()
        )

    def test_loss_log_contains_pico_terms(self):
        train_model = self.make_train_model()
        train_model.train()
        result = train_model.training_step(self.make_batch(), step=0)
        for key in ("loss_vfl", "loss_giou", "loss_dfl"):
            assert key in result.log_dict

    def test_is_train_model_with_watch_metric(self):
        train_model = self.make_train_model()
        assert train_model.watch_metric == "val_metric/map"
        assert train_model.task == "object_detection"

    def test_validation_step(self):
        train_model = self.make_train_model()
        train_model.set_categories({0: "a", 1: "b"})
        train_model.eval()
        with torch.no_grad():
            result = train_model.validation_step(self.make_batch(), step=0)
        assert torch.isfinite(result.loss).item()

    def test_get_optimizer_sgd_linear_lr_scaling(self):
        train_model = self.make_train_model()
        optimizer, scheduler = train_model.get_optimizer(
            total_steps=10, global_batch_size=64
        )
        assert isinstance(optimizer, torch.optim.SGD)
        # 0.2 * (64 / 32) linear scaling.
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.4)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-4)
        assert any(g["weight_decay"] == 0.0 for g in optimizer.param_groups)
        assert scheduler is not None

    def test_defaults_follow_port_recipe(self):
        args = PicoDetObjectDetectionTrainArgs(num_classes=2)
        assert args.assigner == "simota"
        assert args.reg_max == 7
        assert args.loss_cls_weight == 1.0
        assert args.loss_giou_weight == 2.0
        assert args.loss_dfl_weight == 0.25
        assert args.nms_iou_threshold == 0.6
        assert PicoDetObjectDetectionTrainArgs.default_batch_size == 32
        assert PicoDetObjectDetectionTrainArgs.default_steps == 12_000


class TestE2E:
    def test_train_object_detection_picodet(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "picodet1",
            data=data_config(train_json, val_json, images_dir),
            model="picodet/s",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={"image_size": (64, 64), "neck_channels": 32},
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
            out=tmp_path / "picodet2",
            data=data_config(train_json, val_json, images_dir),
            model="picodet/s",
            steps=2,
            batch_size=2,
            num_workers=0,
            model_args={"image_size": (64, 64), "neck_channels": 32},
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        model = load_model(result.exported_model_path, device="cpu")
        image = create_image(tmp_path / "probe.jpg", size=(64, 64), seed=0)
        prediction = model.predict(image, threshold=0.05)
        assert set(prediction) == {"labels", "bboxes", "scores"}
