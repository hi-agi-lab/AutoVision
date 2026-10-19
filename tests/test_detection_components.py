# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for the object detection components."""

import pytest
import torch
from torch import nn

from autovision.geometry import (
    box_cxcywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
)
from autovision.geometry.box_ops import (
    elementwise_box_iou,
    elementwise_siou,
)
from autovision.tasks.detection.shared import (
    HungarianMatcher,
    HybridEncoder,
    RTDETRCriterionV2,
    RTDETRv2Decoder,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.multistep_schedule import (
    MultiStepWarmupScheduler,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.shared.flat_cosine import FlatCosineLRScheduler


def fake_targets(batch_size: int = 2, device="cpu"):
    return [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.2, 0.3, 0.2, 0.2]], device=device),
            "labels": torch.tensor([0, 0], device=device),
        },
        {
            "boxes": torch.tensor([[0.7, 0.7, 0.3, 0.3]], device=device),
            "labels": torch.tensor([0], device=device),
        } if batch_size > 1 else torch.empty(0),
    ][:batch_size]


class TestBoxOps:
    def test_roundtrip(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
        xyxy = box_cxcywh_norm_to_xyxy(boxes, size=(100, 50))
        assert torch.allclose(
            xyxy, torch.tensor([[30.0, 10.0, 70.0, 40.0]])
        )
        back = box_xyxy_to_cxcywh_norm(xyxy, size=(100, 50))
        assert torch.allclose(back, boxes, atol=1e-6)

    def test_giou_self_zero(self):
        from autovision.geometry import generalized_box_iou_loss

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        loss = generalized_box_iou_loss(boxes, boxes)
        assert loss.max() == pytest.approx(0.0, abs=1e-6)


class TestHungarianMatcher:
    def test_perfect_match(self):
        matcher = HungarianMatcher()
        targets = fake_targets()
        targets[0]["labels"] = torch.tensor([0, 1])
        targets[1]["labels"] = torch.tensor([0])
        outputs = {
            "pred_logits": torch.randn(2, 10, 2),
            "pred_boxes": torch.zeros(2, 10, 4),
        }
        # Put target boxes in the first queries so the matcher must find them.
        outputs["pred_boxes"][0, :2] = targets[0]["boxes"]
        outputs["pred_boxes"][1, :1] = targets[1]["boxes"]
        # Make the classification cost agree with the boxes, so the joint
        # minimum is unambiguous.
        for b, target in enumerate(targets):
            for q, label in zip(range(target["boxes"].shape[0]), target["labels"]):
                outputs["pred_logits"][b, q] = -5.0
                outputs["pred_logits"][b, q, label] = 5.0
        indices = matcher(outputs, targets)
        assert indices[0][0].tolist() == [0, 1]
        assert indices[1][0].tolist() == [0]

    def test_empty_target(self):
        matcher = HungarianMatcher()
        outputs = {
            "pred_logits": torch.randn(1, 5, 1),
            "pred_boxes": torch.rand(1, 5, 4),
        }
        indices = matcher(outputs, [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}])
        assert indices[0][0].numel() == 0


class TestModelEMA:
    def test_convergence_to_weights(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.99, warmup_steps=0)
        with torch.no_grad():
            model.weight.fill_(2.0)
        for _ in range(2000):
            ema.update(model)
        assert ema.model.weight[0, 0].item() == pytest.approx(2.0, abs=1e-3)

    def test_buffers_copied(self):
        model = nn.BatchNorm2d(4)
        ema = ModelEMA(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            model.running_mean.fill_(5.0)
        ema.update(model)
        assert ema.model.running_mean[0].item() > 0

    def test_state_roundtrip(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.99, warmup_steps=0)
        ema.update(model)
        ema2 = ModelEMA(model, decay=0.99, warmup_steps=0)
        ema2.load_state_dict(ema.state_dict())
        assert ema2.updates == ema.updates
        assert torch.allclose(ema2.model.weight, ema.model.weight)

    def test_aliased_tensors_update_once(self):
        # The AV-DETR task model exposes the backbone under two state_dict
        # key paths (`backbone_wrapper.*` and `backbone.*`) sharing one
        # storage. The foreach EMA update must land ONCE per unique
        # tensor: applied per key, an aliased tensor follows
        # t <- d^2*t + 2(1-d)*s and inflates uniformly toward 2/(1+d) times
        # the model — invisible at decay 0.995 (factor 1.000025), fatal at
        # 0.9999 whose long warmup lingers at d ~ 0.3-0.7 (factor up to
        # ~1.2). A frozen model must converge to itself exactly.
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.backbone = self.inner.linear  # aliased exposure

        torch.manual_seed(0)
        model = Outer()
        aliased_keys = [k for k in model.state_dict() if k.startswith("backbone.")]
        assert len(aliased_keys) == 2, "expected the alias to show in state_dict"

        ema = ModelEMA(model, decay=0.9, warmup_steps=2)
        for _ in range(50):
            ema.update(model)
        state = model.state_dict()
        ema_state = ema.model.state_dict()
        for key, value in state.items():
            torch.testing.assert_close(
                ema_state[key], value, msg=f"EMA diverged on '{key}'"
            )


class TestFlatCosine:
    def _sched(self, **kwargs):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        return FlatCosineLRScheduler(optimizer, **kwargs), optimizer

    def _lr_at(self, scheduler, optimizer, step):
        """Advance to the ABSOLUTE scheduler step and read the LR."""
        while scheduler.last_epoch < step:
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_warmup_then_flat_then_decay(self):
        """Baseline semantics: quadratic warmup -> flat -> cosine decaying to
        min_factor (0.5) -> tail HOLDS min_factor (no restart)."""
        scheduler, optimizer = self._sched(
            total_steps=100, warmup_steps=10, flat_steps=40, no_aug_steps=10
        )
        # Quadratic warmup: at absolute step 5 lr = (5/10)^2; step 10 -> 1.0.
        assert self._lr_at(scheduler, optimizer, 5) == pytest.approx(0.25)
        assert self._lr_at(scheduler, optimizer, 10) == pytest.approx(1.0)
        # flat plateau up to (exclusive) cosine_start = 50.
        assert self._lr_at(scheduler, optimizer, 49) == pytest.approx(1.0)
        # cosine midpoint (step 70, progress 0.5) -> halfway to min_factor.
        assert self._lr_at(scheduler, optimizer, 70) == pytest.approx(0.75)
        # cosine end floors at min_factor and the tail HOLDS it.
        assert self._lr_at(scheduler, optimizer, 90) == pytest.approx(0.5)
        assert self._lr_at(scheduler, optimizer, 99) == pytest.approx(0.5)

    def test_validation(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="total_steps"):
            FlatCosineLRScheduler(
                torch.optim.SGD(model.parameters(), lr=1.0), total_steps=0
            )


class TestHybridEncoder:
    def test_shapes(self):
        encoder = HybridEncoder(in_channels=[64, 128, 256], hidden_dim=32)
        features = [
            torch.randn(2, 64, 56, 56),
            torch.randn(2, 128, 28, 28),
            torch.randn(2, 256, 14, 14),
        ]
        outputs = encoder(features)
        assert len(outputs) == 3
        assert outputs[0].shape == (2, 32, 56, 56)
        assert outputs[1].shape == (2, 32, 28, 28)
        assert outputs[2].shape == (2, 32, 14, 14)
        # AIFI lives on the deepest level.
        assert any("encoder" in name for name, _ in encoder.named_modules())

    def test_level_count_validated(self):
        encoder = HybridEncoder(in_channels=[64, 128])
        with pytest.raises(AssertionError):
            encoder([torch.randn(2, 64, 8, 8)])


class TestDecoderAndCriterion:
    def _make_decoder(self):
        return RTDETRv2Decoder(
            num_classes=2, hidden_dim=32, num_queries=20,
            feat_channels=[32, 32, 32], feat_strides=[8, 16, 32],
            num_levels=3, nhead=4, num_layers=2, num_denoising=0,
        )

    def _feats(self, batch_size: int = 1):
        return [
            torch.randn(batch_size, 32, 14, 14),
            torch.randn(batch_size, 32, 7, 7),
            torch.randn(batch_size, 32, 4, 4),
        ]

    def _model_and_targets(self):
        torch.manual_seed(0)
        decoder = self._make_decoder()
        out = decoder(self._feats())
        assert "dn_meta" not in out
        assert len(out["aux_outputs"]) == 1  # num_layers - 1
        final = {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        outputs = out["aux_outputs"] + [final]
        assert len(outputs) == 2
        assert outputs[-1]["pred_logits"].shape == (1, 20, 2)
        assert outputs[-1]["pred_boxes"].shape == (1, 20, 4)
        assert outputs[-1]["pred_boxes"].dtype == torch.float32
        assert outputs[0]["pred_boxes"].min() >= 0
        return outputs

    def test_criterion_backward(self):
        torch.manual_seed(0)
        decoder = self._make_decoder()
        feats = [
            torch.randn(1, 32, 14, 14, requires_grad=True),
            torch.randn(1, 32, 7, 7),
            torch.randn(1, 32, 4, 4),
        ]
        out = decoder(feats)
        criterion = RTDETRCriterionV2(HungarianMatcher())
        targets = fake_targets(batch_size=1)
        outputs_list = out.get("aux_outputs", []) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = criterion(outputs_list, targets, world_size=1)
        assert "loss" in losses
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        assert feats[0].grad is not None

    def test_criterion_all_loss_names(self):
        decoder_out = {
            "pred_logits": torch.randn(2, 10, 2, requires_grad=True),
            "pred_boxes": torch.rand(2, 10, 4),
        }
        criterion = RTDETRCriterionV2(HungarianMatcher())
        losses = criterion([decoder_out], fake_targets(), world_size=1)
        for name in ("loss_vfl", "loss_bbox", "loss_giou", "loss"):
            assert name in losses
        expected = (
            losses["loss_vfl"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        ).detach()
        assert losses["loss"].detach() == pytest.approx(expected, rel=1e-5)

    def test_vfl_survives_degenerate_pred_boxes(self):
        # Baseline sanitizes the predicted side before the quality IoU: a
        # degenerate (NaN) box must yield a computed VFL term, not a NaN
        # that the trailing nan_to_num zeroes out.
        decoder_out = {
            "pred_logits": torch.randn(2, 10, 2, requires_grad=True),
            "pred_boxes": torch.rand(2, 10, 4),
        }
        targets = fake_targets()
        criterion = RTDETRCriterionV2(HungarianMatcher())
        healthy = criterion(
            [decoder_out], targets, world_size=1
        )["loss_vfl"]
        assert torch.isfinite(healthy).all()

        poisoned_boxes = torch.rand(2, 10, 4)
        poisoned_boxes[0, 0, 0] = float("nan")  # a matched degenerate box
        poisoned = {"pred_logits": decoder_out["pred_logits"], "pred_boxes": poisoned_boxes}
        degenerate = criterion([poisoned], targets, world_size=1)
        assert torch.isfinite(degenerate["loss_vfl"])
        assert degenerate["loss_vfl"].item() > 0

    def test_matched_pair_iou_sanitizes_degenerate_boxes(self):
        from autovision.tasks.detection.shared.dfine_utils import (
            matched_pair_iou,
        )

        pred = torch.tensor([[float("nan"), 0.5, 0.5, 0.5],
                             [float("inf"), 2.0, -1.0, 0.5],
                             [0.5, 0.5, 0.5, 0.5]])
        target = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 3)
        ious = matched_pair_iou(pred, target)
        assert torch.isfinite(ious).all()
        assert bool((ious >= 0).all()) and bool((ious <= 1).all())

    def test_aux_layers_get_box_deep_supervision(self):
        # Baseline rtdetrv2 aux loop (share_matched_indices defaults False):
        # every intermediate layer re-runs the full loss set (VFL + L1 +
        # GIoU) on its OWN Hungarian match — the v2 family has no union
        # set. Lock the per-aux box keys and the weighted total they feed.
        out = self._make_decoder()(self._feats(batch_size=2))
        targets = fake_targets(batch_size=2)
        criterion = RTDETRCriterionV2(HungarianMatcher())
        outputs_list = out["aux_outputs"] + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = criterion(outputs_list, targets, world_size=1)
        for name in ("loss_vfl_aux_0", "loss_bbox_aux_0", "loss_giou_aux_0"):
            assert name in losses, name
            assert torch.isfinite(losses[name])
        rebuilt = sum(
            value.detach() * criterion.weight_dict[key.replace("_aux_0", "")]
            for key, value in losses.items()
            if key != "loss"
        )
        assert losses["loss"].detach() == pytest.approx(rebuilt, rel=1e-5)


class TestTrainWindow:
    def test_means_over_window(self):
        from autovision.engine.engine import _TrainWindow

        window = _TrainWindow()
        for value in (2.0, 4.0, 6.0):
            window.update({"loss": torch.tensor(value), "tag": "not-a-tensor"})
        means = window.means()
        assert means["loss"] == pytest.approx(4.0)
        assert window.count == 3
        window.reset()
        assert window.count == 0
        assert window.means() == {}

    def test_seconds_per_step_positive(self):
        import time

        from autovision.engine.engine import _TrainWindow

        window = _TrainWindow()
        window.update({"loss": torch.tensor(1.0)})
        time.sleep(0.02)
        assert window.seconds_per_step() > 0.0


class TestValidationReportTables:
    def test_detection_report_renders_overall_and_per_class(self):
        from autovision.metrics.report import format_detection_validation_report

        metrics = {
            "map": 0.0625, "map_50": 0.1030, "map_75": 0.05,
            "mar": 0.20, "f1": 0.24, "precision": 0.25, "recall": 0.27,
        }
        per_class = [
            {"name": "car", "ap": 0.29, "ar": 0.47, "f1": 0.49,
             "precision": 0.37, "recall": 0.69},
            {"name": "bus", "ap": 0.08, "ar": 0.25, "f1": 0.28,
             "precision": 0.36, "recall": 0.22},
        ]
        report = format_detection_validation_report(
            "Validation (EMA weights)", metrics, per_class, max_dets=500,
            footer="note: val_loss covers VFL/bbox/GIoU only",
        )
        assert "Validation (EMA weights)" in report
        assert "mAR @500" in report
        assert "car" in report and "bus" in report
        assert "0.0625" in report and "0.1030" in report
        assert "val_loss covers VFL/bbox/GIoU only" in report
        # Box-drawing frame lines enclose the table body.
        lines = report.splitlines()
        assert lines[1].startswith("┏") and lines[-2].startswith("└")

    def test_metric_table_first_column_left_aligned(self):
        from autovision.metrics.report import format_metric_table

        table = format_metric_table(
            "T", ("Name", "Val"), [["longer-name", "1.0"], ["x", "22"]]
        )
        name_line = next(line for line in table.splitlines() if "longer-name" in line)
        assert "longer-name┃" in name_line.replace(" ", "")


class TestMultiStepWarmup:
    def _sched(self, **kwargs):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        return MultiStepWarmupScheduler(optimizer, **kwargs), optimizer

    def _lr_at(self, scheduler, optimizer, step):
        """Advance to the ABSOLUTE scheduler step and read the LR."""
        while scheduler.last_epoch < step:
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_warmup_then_constant_with_drops(self):
        scheduler, optimizer = self._sched(
            total_steps=100, warmup_steps=10, milestones=(30, 60), gamma=0.8
        )
        assert self._lr_at(scheduler, optimizer, 10) == pytest.approx(1.0)
        assert self._lr_at(scheduler, optimizer, 29) == pytest.approx(1.0)
        assert self._lr_at(scheduler, optimizer, 30) == pytest.approx(0.8)
        assert self._lr_at(scheduler, optimizer, 60) == pytest.approx(0.64)
        assert self._lr_at(scheduler, optimizer, 99) == pytest.approx(0.64)

    def test_total_steps_validated(self):
        with pytest.raises(ValueError, match="total_steps"):
            MultiStepWarmupScheduler(
                torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0),
                total_steps=0,
            )


def _random_xyxy_boxes(n: int, seed: int) -> torch.Tensor:
    """Random non-degenerate xyxy boxes in a ~[0, 2] canvas."""
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=generator)
    wh = torch.rand(n, 2, generator=generator) + 0.05
    return torch.cat([xy, xy + wh], dim=-1)


class TestElementwiseSIoU:
    """Scylla-IoU math (design doc 15 §4.1; cases mirrored from the reference
    implementation's validated suite)."""

    def test_identical_boxes_give_exact_zero_loss(self):
        boxes = _random_xyxy_boxes(16, seed=7)

        loss = elementwise_siou(boxes, boxes)

        assert torch.equal(loss, torch.zeros_like(loss))

    def test_is_symmetric(self):
        boxes1 = _random_xyxy_boxes(32, seed=11)
        boxes2 = _random_xyxy_boxes(32, seed=12)

        torch.testing.assert_close(
            elementwise_siou(boxes1, boxes2), elementwise_siou(boxes2, boxes1)
        )

    def test_is_bounded_and_finite(self):
        pairs1 = torch.cat(
            [_random_xyxy_boxes(64, seed=13), torch.tensor([[10.0, 10.0, 10.0, 10.0]])]
        )
        pairs2 = torch.cat(
            [_random_xyxy_boxes(64, seed=14), torch.tensor([[10.0, 10.0, 10.0, 10.0]])]
        )

        loss = elementwise_siou(pairs1, pairs2)

        assert torch.isfinite(loss).all()
        assert (loss >= 0).all()
        assert (loss < 3).all()

    def test_is_axis_symmetric(self):
        """A pure x-axis center offset and the equal pure y-axis offset give the same loss."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        gt_x = torch.tensor([[3.0, 0.0, 5.0, 2.0]])
        gt_y = torch.tensor([[0.0, 3.0, 2.0, 5.0]])

        torch.testing.assert_close(elementwise_siou(pred, gt_x), elementwise_siou(pred, gt_y))

    def test_angle_contribution_extinguishes_for_shared_center(self):
        """Same-center pairs reduce exactly to 1 - IoU + shape/2 (self-extinguish)."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])  # center (1, 1), w = h = 2
        gt = torch.tensor([[0.5, -0.5, 1.5, 2.5]])  # center (1, 1), w = 1, h = 3

        loss = elementwise_siou(pred, gt)
        iou, _ = elementwise_box_iou(pred, gt)
        expected = 1 - iou + (
            (1 - torch.exp(torch.tensor(-0.5))) ** 4
            + (1 - torch.exp(torch.tensor(-1.0 / 3.0))) ** 4
        ) / 2

        torch.testing.assert_close(loss, expected)

    def test_known_values_lock_the_formula(self):
        """Hand-derived values pin gamma = 2 - Lambda, the outer 1/2, and the
        per-axis enclosure normalization.

        Case A: x-only center offset (Lambda=0 -> gamma=2), disjoint boxes,
        equal shapes, enclosure 5x2 -> 1 + (1 - exp(-2*(3/5)^2))/2.
        Case B: 45-degree offset (Lambda=1 -> gamma=1), enclosure 4x4
        -> 2 - exp(-0.25); a gamma=2 implementation fails this case.
        """
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
        gt = torch.tensor([[3.0, 0.0, 5.0, 2.0], [2.0, 2.0, 4.0, 4.0]])

        loss = elementwise_siou(pred, gt)

        expected_a = 1 + (1 - torch.exp(torch.tensor(-2 * (3.0 / 5.0) ** 2))) / 2
        expected_b = 2 - torch.exp(torch.tensor(-0.25))
        torch.testing.assert_close(loss[0], expected_a)
        torch.testing.assert_close(loss[1], expected_b)

    def test_gradients_are_finite(self):
        """Mixed batch (aligned, diagonal, disjoint, degenerate) keeps grads finite."""
        boxes1 = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            requires_grad=True,
        )
        boxes2 = torch.tensor(
            [
                [3.0, 0.0, 5.0, 2.0],
                [2.0, 2.0, 4.0, 4.0],
                [10.0, 10.0, 12.0, 12.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            requires_grad=True,
        )

        elementwise_siou(boxes1, boxes2).sum().backward()

        assert torch.isfinite(boxes1.grad).all()
        assert torch.isfinite(boxes2.grad).all()

    def test_forward_is_finite_under_cpu_bfloat16_autocast(self):
        boxes1 = _random_xyxy_boxes(32, seed=15)
        boxes2 = _random_xyxy_boxes(32, seed=16)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = elementwise_siou(boxes1, boxes2)

        assert torch.isfinite(loss).all()

    def test_empty_input_returns_empty(self):
        empty = torch.empty(0, 4)

        assert elementwise_siou(empty, empty).shape == (0,)

    def test_rejects_unequal_length(self):
        with pytest.raises(ValueError, match="same length"):
            elementwise_siou(_random_xyxy_boxes(3, seed=4), _random_xyxy_boxes(1, seed=5))

    def test_theta_only_affects_shape_mismatched_pairs(self):
        """theta=2 vs theta=4 differ when shapes mismatch and coincide when equal."""
        same_shape = (torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([[1.0, 0.0, 3.0, 2.0]]))
        mismatched = (torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([[0.5, 0.0, 1.5, 3.0]]))

        for boxes1, boxes2 in (same_shape, mismatched):
            loss_low = elementwise_siou(boxes1, boxes2, theta=2.0)
            loss_high = elementwise_siou(boxes1, boxes2, theta=4.0)
            if boxes2 is mismatched[1]:
                assert not torch.allclose(loss_low, loss_high)
            else:
                torch.testing.assert_close(loss_low, loss_high)


class TestDFINESIoUCriterion:
    """box_regression_iou='siou' wiring (design doc 15 §4.3)."""

    def _outputs(self, layers: int = 1, grad: bool = False):
        torch.manual_seed(0)
        outputs = []
        for _ in range(layers):
            boxes = torch.rand(2, 10, 4) * 0.4 + 0.3
            if grad:
                boxes = boxes.detach().requires_grad_(True)
            outputs.append(
                {
                    "pred_logits": torch.randn(2, 10, 3),
                    "pred_boxes": boxes,
                }
            )
        return outputs

    def _criterion(self, **kwargs):
        return DFINECriterionV2(HungarianMatcher(), **kwargs)

    def test_default_mode_emits_giou_not_siou(self):
        losses = self._criterion()(self._outputs(), fake_targets(), world_size=1)

        assert "loss_giou" in losses
        assert "loss_siou" not in losses

    def test_siou_mode_emits_siou_not_giou(self):
        criterion = self._criterion(box_regression_iou="siou")

        losses = criterion(self._outputs(layers=2), fake_targets(), world_size=1)

        assert "loss_siou" in losses
        assert "loss_bbox" in losses
        assert not any("loss_giou" in key for key in losses)
        # Every box-loss branch carries the swapped IoU term (docstring:
        # main, _pre, _dn_pre, and the copied _aux_i/_dn_i/_enc_i), while
        # the aux VFL keeps its own suffix shape.
        assert "loss_vfl_aux_0" in losses
        assert "loss_siou_aux_0" in losses

    def test_siou_value_matches_box_ops_on_matched_pairs(self):
        from autovision.tasks.detection.shared.rtdetrv2_criterion import (
            _cxcywh_to_xyxy_unit,
        )

        outputs = self._outputs()
        targets = fake_targets()
        criterion = self._criterion(box_regression_iou="siou")
        losses = criterion(outputs, targets, world_size=1)

        indices = criterion.matcher(outputs[-1], targets)
        manual = sum(
            elementwise_siou(
                _cxcywh_to_xyxy_unit(outputs[-1]["pred_boxes"][b, src].detach()),
                _cxcywh_to_xyxy_unit(targets[b]["boxes"][dst]),
            ).sum()
            for b, (src, dst) in enumerate(indices)
            if src.numel()
        )
        matched = sum(int(src.numel()) for src, _ in indices)
        torch.testing.assert_close(
            losses["loss_siou"] * max(matched, 1), manual, check_dtype=False
        )

    def test_siou_gradient_flows_to_pred_boxes(self):
        outputs = self._outputs(grad=True)
        criterion = self._criterion(box_regression_iou="siou")

        losses = criterion(outputs, fake_targets(), world_size=1)
        losses["loss_siou"].backward()

        grad = outputs[-1]["pred_boxes"].grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_weight_inherits_giou_and_explicit_value_wins(self):
        inherited = self._criterion(box_regression_iou="siou")
        assert inherited.weight_dict["loss_siou"] == pytest.approx(2.0)
        assert "loss_giou" not in inherited.weight_dict

        explicit = self._criterion(box_regression_iou="siou", siou_loss_coef=1.5)
        assert explicit.weight_dict["loss_siou"] == pytest.approx(1.5)

        supplied = self._criterion(
            weight_dict={"loss_giou": 3.0}, box_regression_iou="siou"
        )
        assert supplied.weight_dict["loss_siou"] == pytest.approx(3.0)
        assert "loss_giou" not in supplied.weight_dict

    def test_pre_branch_weight_key_follows_the_active_iou_term(self):
        """The hardcoded _pre/_dn_pre weight lines must track the active term
        (design doc 15 E4) — a missing weight would silently drop the branch."""
        outputs = self._outputs(layers=2)
        pre_outputs = self._outputs()[-1]
        targets = fake_targets()

        siou = self._criterion(box_regression_iou="siou")
        losses_siou = siou(outputs, targets, world_size=1, pre_outputs=pre_outputs)
        assert "loss_siou_pre" in losses_siou
        assert "loss_bbox_pre" in losses_siou
        assert torch.isfinite(losses_siou["loss"]).all()

        giou = self._criterion()
        losses_giou = giou(outputs, targets, world_size=1, pre_outputs=pre_outputs)
        assert "loss_giou_pre" in losses_giou

    def test_pre_branch_siou_weights_are_actually_applied(self):
        """Zeroing the loss_siou weights must shift the total by exactly
        Σ siou_loss × weight (design doc 15 E4). A stale _pre weight line
        would shrink the difference by the silently-unweighted term."""
        outputs = self._outputs(layers=2)
        pre_outputs = self._outputs()[-1]
        targets = fake_targets()
        criterion = self._criterion(box_regression_iou="siou")
        weights = dict(criterion.weight_dict)

        base = criterion(outputs, targets, world_size=1, pre_outputs=pre_outputs)
        for key in weights:
            if key.startswith("loss_siou"):
                criterion.weight_dict[key] = 0.0
        zeroed = criterion(outputs, targets, world_size=1, pre_outputs=pre_outputs)

        siou_weights = {
            key: weight
            for key, weight in weights.items()
            if key.startswith("loss_siou")
        }
        expected = sum(
            base[key] * weight for key, weight in siou_weights.items() if key in base
        )
        # The _pre weight is a forward-local copy of the main term's weight
        # (the E4 line) — it must carry the same factor. So is the aux-loop
        # expansion (weights[f"{key}_aux_{i}"]) for the deep-supervised
        # aux box branch.
        if "loss_siou_pre" in base:
            expected = expected + base["loss_siou_pre"] * siou_weights["loss_siou"]
        if "loss_siou_aux_0" in base:
            expected = expected + base["loss_siou_aux_0"] * siou_weights["loss_siou"]
        torch.testing.assert_close(base["loss"] - zeroed["loss"], expected)

    def test_ctor_rejects_unknown_iou_term(self):
        with pytest.raises(ValueError, match="box_regression_iou"):
            self._criterion(box_regression_iou="diou")
