"""Tests for the D-FINE object detection family (FGL + DDF)."""

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
from autovision.tasks.detection.dfine import (
    DFINEObjectDetection,
    DFINEObjectDetectionTrain,
    DFINEObjectDetectionTrainArgs,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision.tasks.detection.shared.dfine_utils import (
    bbox2distance,
    ddf_loss,
    distance2bbox,
    fg_loss,
    integrate_corners,
    translate_gt,
    weighting_function,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)

REG_MAX = 8  # small bin layout keeps tests fast


def make_decoder(num_layers: int = 2, num_queries: int = 20) -> DFINEv2Decoder:
    torch.manual_seed(0)
    return DFINEv2Decoder(
        num_classes=2,
        hidden_dim=32,
        num_queries=num_queries,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        nhead=4,
        num_layers=num_layers,
        num_denoising=50,
        reg_max=REG_MAX,
    )


def make_feats(batch: int = 2):
    return [
        torch.randn(batch, 32, 8, 8),
        torch.randn(batch, 32, 4, 4),
        torch.randn(batch, 32, 2, 2),
    ]


def make_targets(batch: int = 2):
    torch.manual_seed(1)
    return [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])}
        for _ in range(batch)
    ]


def make_train_model(**overrides) -> DFINEObjectDetectionTrain:
    args = DFINEObjectDetectionTrainArgs(
        num_classes=2,
        image_size=(64, 64),
        num_queries=10,
        num_top_queries=10,
        hidden_dim=32,
        dec_layers=2,
        reg_max=REG_MAX,
        **overrides,
    )
    return DFINEObjectDetectionTrain(args)


def make_batch(size: int = 64, images: int = 2):
    targets = make_targets(images)
    return {
        "image": torch.randn(images, 3, size, size),
        "bboxes": [t["boxes"] for t in targets],
        "classes": [t["labels"] for t in targets],
        "image_path": [f"img_{i}.jpg" for i in range(images)],
        "original_size": [(size, size)] * images,
    }


class TestDFineUtils:
    def test_weighting_function_geometry(self):
        """Official W: reg_max+1 values, zero-centered, monotone, symmetric,
        reach ±2·up·reg_scale = ±4 (up=0.5, reg_scale=4)."""
        weights = weighting_function(REG_MAX, up=0.5, reg_scale=4.0)
        assert weights.shape == (REG_MAX + 1,)
        assert weights[REG_MAX // 2].item() == pytest.approx(0.0, abs=1e-7)
        assert weights[0].item() == pytest.approx(-4.0)
        assert weights[-1].item() == pytest.approx(4.0)
        assert (weights[1:] >= weights[:-1]).all()  # monotone non-decreasing
        assert (weights[: REG_MAX // 2] <= 0).all()
        assert (weights[REG_MAX // 2 :] >= 0).all()
        assert torch.allclose(weights[: REG_MAX // 2], -weights[REG_MAX // 2 + 1 :].flip(0))

    def test_distance_bbox_roundtrip(self):
        """Continuous roundtrip in the official parameterization: distance
        units are w/reg_scale with a 0.5·reg_scale offset; zero distance
        reproduces the reference exactly."""
        ref = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.7, 0.1, 0.1]])
        distances = torch.tensor([[0.0, -1.0, 2.0, 0.5], [0.3, 0.0, -0.2, 1.0]])
        rebuilt = distance2bbox(ref, distances, reg_scale=4.0)
        cx, cy, w, h = rebuilt.unbind(-1)
        # x1 = cx - (0.5·rs + d_l)·(w/rs): check the left edge explicitly.
        expected_x1 = ref[:, 0] - (0.5 * 4.0 + distances[:, 0]) * (ref[:, 2] / 4.0)
        torch.testing.assert_close(cx - w / 2, expected_x1)
        # Zero distance must reproduce the reference box.
        zero = distance2bbox(ref, torch.zeros_like(distances), reg_scale=4.0)
        torch.testing.assert_close(zero, ref)

    def test_uniform_distribution_decodes_to_reference(self):
        # Uniform logits → uniform distribution → distance 0 on every side →
        # the decoded box IS the reference box.
        ref = torch.rand(5, 4) * 0.4 + 0.2
        ref[:, 2:] = ref[:, 2:].clamp(min=0.05)
        corners = torch.zeros(5, 4 * (REG_MAX + 1))
        weights = weighting_function(REG_MAX, up=0.5, reg_scale=4.0)
        decoded = distance2bbox(ref, integrate_corners(corners, weights), reg_scale=4.0)
        torch.testing.assert_close(decoded, ref)

    def test_translate_gt_interpolates_between_bins(self):
        weights = weighting_function(REG_MAX, up=0.5, reg_scale=4.0)
        # A target exactly midway between W[10] and W[11] splits the mass
        # evenly across the two bins.
        k = REG_MAX // 2
        gt = torch.full((1, 4), (weights[k] + weights[k + 1]) / 2)
        label, wr, wl = translate_gt(gt, REG_MAX, reg_scale=4.0, up=0.5)
        assert (label == k).all()
        torch.testing.assert_close(wl, torch.full_like(wl, 0.5))
        torch.testing.assert_close(wr, torch.full_like(wr, 0.5))
        # The two-bin interpolation reproduces the value exactly.
        assert torch.allclose(wl * weights[label.long()] + wr * weights[label.long() + 1], gt)

    def test_translate_gt_clamps_out_of_range(self):
        gt = torch.tensor([-100.0, 0.0, 100.0])
        label, wr, wl = translate_gt(gt, REG_MAX, reg_scale=4.0, up=0.5)
        # Far below W[0]: all mass on bin 0; far above W[-1]: all mass on
        # the last bin.
        assert label[0].item() == 0.0 and wl[0].item() == 1.0 and wr[0].item() == 0.0
        assert label[2].item() == pytest.approx(REG_MAX - 0.1)
        assert wr[2].item() == 1.0 and wl[2].item() == 0.0
        # Center value lands on the zero axis with its exact interpolation.
        weights = weighting_function(REG_MAX, up=0.5, reg_scale=4.0)
        assert label[1].item() == pytest.approx(REG_MAX // 2)
        assert torch.allclose(
            wl[1] * weights[label[1].long()] + wr[1] * weights[label[1].long() + 1],
            torch.tensor(0.0),
        )


class TestDFineDecoder:
    def test_training_output_contract(self):
        model = make_decoder(num_layers=3)
        model.train()
        out = model(make_feats(), targets=make_targets())
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert len(out["aux_outputs"]) == 2
        assert len(out["pred_corners_list"]) == 3
        assert out["pred_corners_list"][0].shape == (2, 20, 4 * (REG_MAX + 1))
        assert len(out["ref_boxes_list"]) == 3
        assert out["ref_boxes_list"][0].shape == (2, 20, 4)
        # dn segment split off
        dn = out["dn_meta"]
        assert out["dn_pred_corners_list"][0].shape[1] == dn["dn_num_split"][0]
        assert out["dn_ref_boxes_list"][-1].shape[1] == dn["dn_num_split"][0]

    def test_ref_boxes_are_detached(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        for ref in out["ref_boxes_list"] + out["dn_ref_boxes_list"]:
            assert not ref.requires_grad

    def test_corners_carry_gradients(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        loss = out["pred_boxes"].sum() + sum(
            c.sum() for c in out["pred_corners_list"]
        )
        loss.backward()
        for head in model.dec_bbox_head:
            assert head.layers[-1].weight.grad is not None
            assert head.layers[-1].weight.grad.abs().sum() > 0

    def test_zero_init_starts_from_reference(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        torch.testing.assert_close(
            out["ref_boxes_list"][0], out["pred_boxes"], atol=1e-4, rtol=1e-3
        )

    def test_eval_uses_last_layer_only(self):
        model = make_decoder(num_layers=3)
        model.eval()
        with torch.no_grad():
            out = model(make_feats())
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert "aux_outputs" not in out
        assert "pred_corners_list" not in out

    def test_empty_gt_skips_denoising(self):
        model = make_decoder()
        model.train()
        empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
        out = model(make_feats(1), targets=empty * 1)
        assert "dn_meta" not in out
        assert out["pred_boxes"].shape == (1, 20, 4)

    def test_boxes_stay_finite(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        assert torch.isfinite(out["pred_boxes"]).all()
        assert (out["pred_boxes"][:, :, 2:] >= 0).all()


class TestFGL:
    def _layer_inputs(self, quality: float):
        ref = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])  # (B=1, Q=1, 4)
        gt_cxcywh = torch.tensor([[0.5, 0.5, 0.4 * quality, 0.4 * quality]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        targets = [{"boxes": gt_cxcywh, "labels": torch.tensor([0])}]
        return ref, gt_cxcywh, indices, targets

    def _corners_for(self, ref, gt_cxcywh):
        """Corner logits whose softmax IS the FGL soft target distribution."""
        gt_xyxy = torch.stack([
            gt_cxcywh[0, 0] - gt_cxcywh[0, 2] / 2,
            gt_cxcywh[0, 1] - gt_cxcywh[0, 3] / 2,
            gt_cxcywh[0, 0] + gt_cxcywh[0, 2] / 2,
            gt_cxcywh[0, 1] + gt_cxcywh[0, 3] / 2,
        ]).unsqueeze(0)
        label, wr, wl = bbox2distance(ref[0], gt_xyxy, REG_MAX, reg_scale=4.0)
        probs = torch.full((1, 1, 4, REG_MAX + 1), 1e-12)
        for side in range(4):
            probs[0, 0, side, label[0, side].long()] += wl[0, side]
            probs[0, 0, side, label[0, side].long() + 1] += wr[0, side]
        return probs.log().reshape(1, 1, 4 * (REG_MAX + 1))

    def test_perfect_distribution_near_zero(self):
        ref, gt, indices, targets = self._layer_inputs(quality=1.0)
        corners = self._corners_for(ref, gt)
        loss = fg_loss(
            corners, ref, gt.view(1, 1, 4), targets, indices,
            torch.tensor(1.0), REG_MAX, reg_scale=4.0,
        )
        # The two-bin CE against its own target distribution leaves only
        # the target entropy (~≤0.7 per side at worst) — well below the
        # uniform-distribution baseline.
        assert loss.item() < 1.0

    def test_wrong_distribution_positive(self):
        ref, gt, indices, targets = self._layer_inputs(quality=1.0)
        corners = torch.zeros(1, 1, 4 * (REG_MAX + 1))
        loss = fg_loss(
            corners, ref, gt.view(1, 1, 4), targets, indices,
            torch.tensor(1.0), REG_MAX, reg_scale=4.0,
        )
        assert loss.item() > 0.1

    def test_exact_bin_target_is_exactly_zero(self):
        """A GT landing exactly ON a weighting-function value has all target
        mass on one bin — the matched one-hot distribution scores 0."""
        ref = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])  # (B=1, Q=1, 4)
        weights = weighting_function(REG_MAX, up=0.5, reg_scale=4.0)
        d = weights[REG_MAX // 2 + 2].item()  # a small positive distance
        # Move every edge outward by d·(w or h)/reg_scale ⇒ all sides at d.
        gt_cxcywh = torch.tensor([[
            ref[0, 0, 0], ref[0, 0, 1],
            ref[0, 0, 2] + 2 * d * ref[0, 0, 2] / 4.0,
            ref[0, 0, 3] + 2 * d * ref[0, 0, 3] / 4.0,
        ]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        targets = [{"boxes": gt_cxcywh, "labels": torch.tensor([0])}]
        corners = torch.zeros(1, 1, 4 * (REG_MAX + 1))
        for side in range(4):
            corners[0, 0, side * (REG_MAX + 1) + REG_MAX // 2 + 2] = 12.0
        loss = fg_loss(
            corners, ref, gt_cxcywh.view(1, 1, 4), targets, indices,
            torch.tensor(1.0), REG_MAX, reg_scale=4.0,
        )
        # Analytically zero (target lands exactly on bin 6 → wr=0, matched
        # one-hot scores 0); the 1e-4 residual is float32 quantization of
        # the expanded GT drifting the label to the neighbouring bin with
        # ~4e-7 of left mass, plus the one-hot's finite softmax sharpness.
        assert loss.item() < 1e-3

    def test_iou_weight_scales_loss(self):
        # Same distribution, lower-quality target (smaller GT → smaller IoU)
        # must produce a smaller weighted loss.
        ref, gt_hi, indices, targets_hi = self._layer_inputs(quality=1.0)
        _, gt_lo, _, targets_lo = self._layer_inputs(quality=0.5)
        corners = torch.zeros(1, 1, 4 * (REG_MAX + 1))
        pred_boxes = torch.zeros(1, 1, 4)  # placeholder, IoU uses pred vs gt
        pred_boxes[0, 0] = gt_hi[0]
        loss_hi = fg_loss(corners, ref, pred_boxes, targets_hi, indices,
                          torch.tensor(1.0), REG_MAX, reg_scale=4.0)
        loss_lo = fg_loss(corners, ref, pred_boxes, targets_lo, indices,
                          torch.tensor(1.0), REG_MAX, reg_scale=4.0)
        assert loss_lo.item() < loss_hi.item()

    def test_empty_indices_zero_with_graph(self):
        corners = torch.zeros(1, 1, 4 * (REG_MAX + 1), requires_grad=True)
        ref = torch.zeros(1, 4)
        loss = fg_loss(
            corners, ref, corners.new_zeros(1, 1, 4),
            [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}],
            [(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))],
            torch.tensor(1.0), REG_MAX, reg_scale=4.0,
        )
        loss.backward()
        assert loss.item() == 0.0
        assert corners.grad is not None


class TestDDF:
    def _stacked(self, corners_bq_side_bin):
        return corners_bq_side_bin.reshape(
            corners_bq_side_bin.shape[0], corners_bq_side_bin.shape[1],
            4 * (REG_MAX + 1),
        )

    def test_identical_distributions_self_extinguish(self):
        corners = self._stacked(torch.randn(2, 3, 4, REG_MAX + 1))
        boxes = torch.rand(2, 3, 4) * 0.3 + 0.3
        indices = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)
        ]
        logits = torch.rand(2, 3, 2)
        loss, _, _ = ddf_loss(corners, corners.clone(), logits, logits.clone(),
                              boxes, boxes.clone(), indices, REG_MAX)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_divergent_distributions_positive(self):
        torch.manual_seed(3)
        student = self._stacked(torch.randn(2, 3, 4, REG_MAX + 1) * 4)
        teacher = self._stacked(torch.randn(2, 3, 4, REG_MAX + 1) * 4)
        boxes = torch.rand(2, 3, 4) * 0.3 + 0.3
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)]
        logits = torch.rand(2, 3, 2)
        loss, _, _ = ddf_loss(student, teacher, logits, logits, boxes, boxes,
                              indices, REG_MAX)
        assert loss.item() > 0.0

    def test_teacher_is_detached(self):
        student = self._stacked(torch.randn(1, 2, 4, REG_MAX + 1))
        teacher = self._stacked(torch.randn(1, 2, 4, REG_MAX + 1))
        student.requires_grad_(True)
        boxes = torch.rand(1, 2, 4) * 0.3 + 0.3
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        logits = torch.rand(1, 2, 2)
        loss, _, _ = ddf_loss(student, teacher, logits, logits, boxes, boxes,
                              indices, REG_MAX)
        loss.backward()
        assert teacher.grad is None
        assert student.grad is not None
        assert student.grad.abs().sum() > 0


class TestDFineCriterion:
    def make_criterion(self) -> DFINECriterionV2:
        return DFINECriterionV2(
            HungarianMatcher(), reg_max=REG_MAX,
            fgl_loss_weight=0.15, ddf_loss_weight=1.5,
        )

    def test_loss_keys_and_weights(self):
        model = make_decoder(num_layers=2)
        model.train()
        out = model(make_feats(), targets=make_targets())
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(
            outputs_list,
            make_targets(),
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            dn_pre_outputs=out["dn_pre_outputs"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        for key in (
            "loss_vfl", "loss_bbox", "loss_giou", "loss_fgl",
            "loss_vfl_aux_0", "loss_bbox_aux_0", "loss_giou_aux_0",
            "loss_fgl_aux_0", "loss_ddf_aux_0",
            "loss_vfl_dn_0", "loss_fgl_dn_0", "loss_ddf_dn_0", "loss",
        ):
            assert key in losses, key
        assert torch.isfinite(losses["loss"]).item()

    def test_aux_box_losses_ride_union_assignment(self):
        # Baseline aux loop: each intermediate layer's L1/GIoU uses the
        # union assignment and num_boxes_go normalization (official
        # ``indices_in = indices_go if loss in ["boxes", ...]``), NOT the
        # layer's own Hungarian match — and is weighted like the final
        # layer's box loss.
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        targets = make_targets()
        out = model(make_feats(), targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(
            outputs_list,
            targets,
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
        )
        assert torch.isfinite(losses["loss_bbox_aux_0"]).item()
        assert torch.isfinite(losses["loss_giou_aux_0"]).item()
        # The weighted total includes the aux box terms at the base
        # weights: rebuild the total from the individual losses (the only
        # suffix at num_layers=2 is "_aux_0") and compare.
        criterion = self.make_criterion()
        rebuilt = sum(
            value.detach() * criterion.weight_dict[key.replace("_aux_0", "")]
            for key, value in losses.items()
            if key != "loss"
        )
        assert losses["loss"].detach() == pytest.approx(rebuilt, rel=1e-5)

    def test_dn_pre_outputs_supervised(self):
        # The denoising segment of the pre-output head is supervised like
        # the matched one (baseline dn_pre_outputs contract): VFL + L1/GIoU
        # over the CDN assignment, weighted like the base losses.
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        targets = make_targets()
        out = model(make_feats(), targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        kwargs = dict(
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        without = self.make_criterion()(outputs_list, targets, **kwargs)
        with_pre = self.make_criterion()(
            outputs_list, targets, dn_pre_outputs=out["dn_pre_outputs"], **kwargs
        )
        for key in ("loss_vfl_dn_pre", "loss_bbox_dn_pre", "loss_giou_dn_pre"):
            assert key in with_pre, key
            assert key not in without, key
            assert torch.isfinite(with_pre[key]).item()
        assert "loss" in with_pre and torch.isfinite(with_pre["loss"]).item()

    def test_pre_heads_supervised_like_baseline(self):
        # Baseline parity lock: pre_outputs / dn_pre_outputs carry no
        # pred_corners (the baseline dfine_decoder assembles pred_logits +
        # pred_boxes only), so the baseline loss_local early-outs on the
        # missing key — NO loss_fgl_pre / loss_ddf_pre (nor dn_pre FGL/DDF)
        # exists upstream. VFL + L1/GIoU are the only pre-head terms, and
        # they must stay that way for baseline equivalence.
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        targets = make_targets()
        out = model(make_feats(), targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(
            outputs_list,
            targets,
            pre_outputs=out["pre_outputs"],
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            dn_pre_outputs=out["dn_pre_outputs"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        for name in (
            "loss_vfl_pre", "loss_bbox_pre", "loss_giou_pre",
            "loss_vfl_dn_pre", "loss_bbox_dn_pre", "loss_giou_dn_pre",
        ):
            assert name in losses, name
        for name in (
            "loss_fgl_pre", "loss_ddf_pre",
            "loss_fgl_dn_pre", "loss_ddf_dn_pre",
        ):
            assert name not in losses, name
        # The decoder-side contract that makes the equivalence hold.
        assert "pred_corners" not in out["pre_outputs"]
        assert "pred_corners" not in out["dn_pre_outputs"]

    def test_dn_fgl_weight_override_reweights_only_dn(self):
        # The dense-batch knob: dn_fgl_loss_weight replaces the main FGL
        # weight on _dn_* keys and touches nothing else.
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        targets = make_targets()
        feats = make_feats()
        out = model(feats, targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        kwargs = dict(
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        baseline = self.make_criterion()(outputs_list, targets, **kwargs)
        overridden = DFINECriterionV2(
            HungarianMatcher(), reg_max=REG_MAX,
            fgl_loss_weight=0.15, dn_fgl_loss_weight=0.05, ddf_loss_weight=1.5,
        )(outputs_list, targets, **kwargs)

        dn_fgl_keys = [key for key in baseline if key.startswith("loss_fgl_dn")]
        assert dn_fgl_keys, "expected denoising FGL terms in the loss set"
        expected = float(baseline["loss"]) + sum(
            (0.05 - 0.15) * float(baseline[key]) for key in dn_fgl_keys
        )
        assert float(overridden["loss"]) == pytest.approx(expected, rel=1e-6)
        # Non-dn losses are bit-identical: same inputs, same weights.
        for key in ("loss_vfl", "loss_fgl", "loss_vfl_dn_0"):
            assert baseline[key].item() == overridden[key].item()

    def test_dn_fgl_weight_default_follows_main_weight(self):
        # Default None keeps baseline parity: the dn weight equals the
        # main FGL weight, so the total is unchanged vs make_criterion().
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        targets = make_targets()
        feats = make_feats()
        out = model(feats, targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        kwargs = dict(
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        baseline = self.make_criterion()(outputs_list, targets, **kwargs)
        explicit = DFINECriterionV2(
            HungarianMatcher(), reg_max=REG_MAX,
            fgl_loss_weight=0.15, dn_fgl_loss_weight=None, ddf_loss_weight=1.5,
        )(outputs_list, targets, **kwargs)
        assert baseline["loss"].item() == explicit["loss"].item()

    def test_matches_baseline_when_corners_absent(self):
        # Without distributional outputs the criterion must degrade to the
        # plain RT-DETRv2 loss set (validation path).
        model = make_decoder(num_layers=2)
        model.train()
        out = model(make_feats(), targets=make_targets())
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(outputs_list, make_targets())
        assert "loss_fgl" not in losses and "loss_ddf" not in losses
        assert "loss" in losses

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        criterion = self.make_criterion()
        targets = make_targets()
        feats = make_feats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step_loss():
            optimizer.zero_grad()
            out = model(feats, targets=targets)
            outputs_list = list(out["aux_outputs"]) + [
                {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
            ]
            losses = criterion(
                outputs_list, targets,
                dn_outputs=out["dn_aux_outputs"], dn_meta=out["dn_meta"],
                corners_list=out["pred_corners_list"],
                ref_boxes_list=out["ref_boxes_list"],
                dn_corners_list=out["dn_pred_corners_list"],
                dn_ref_boxes_list=out["dn_ref_boxes_list"],
            )
            losses["loss"].backward()
            optimizer.step()
            return float(losses["loss"].detach())

        first = sum(step_loss() for _ in range(5)) / 5
        last = sum(step_loss() for _ in range(5)) / 5
        assert last < first


class TestDFineTaskModel:
    def make_task_model(self, **overrides) -> DFINEObjectDetection:
        return DFINEObjectDetection(
            num_classes=2, image_size=(64, 64), hidden_dim=32,
            num_queries=10, dec_layers=2, num_heads=4, num_top_queries=10,
            reg_max=REG_MAX, **overrides,
        )

    def test_forward_shapes(self):
        model = self.make_task_model()
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, 10, 2)
        assert boxes.shape == (1, 10, 4)

    def test_init_args_record_reg_max(self):
        model = self.make_task_model()
        assert model.init_args["reg_max"] == REG_MAX

    def test_gradient_flow_through_backward(self):
        model = self.make_task_model()
        model.train()
        logits, boxes = model(torch.randn(1, 3, 64, 64))
        (logits.sum() + boxes.sum()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads

    def test_postprocess_maps_to_original_pixels(self):
        model = self.make_task_model()
        model.eval()
        raw = model(torch.randn(1, 3, 64, 64))
        results = model.postprocess(raw, [{"orig_w": 128, "orig_h": 96}], threshold=0.0)
        xyxy = results[0]["bboxes"]
        assert xyxy.shape[0] > 0
        assert (xyxy[:, [0, 2]] <= 128.0 + 1e-3).all()
        assert (xyxy[:, [1, 3]] <= 96.0 + 1e-3).all()


class TestDFineTrainModel:
    def test_training_step_backward(self):
        train_model = make_train_model()
        train_model.train()
        result = train_model.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss).item()
        result.loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in train_model.model.decoder.dec_bbox_head.parameters()
        )

    def test_loss_log_contains_dfine_terms(self):
        train_model = make_train_model()
        train_model.train()
        result = train_model.training_step(make_batch(), step=0)
        assert "loss_fgl" in result.log_dict
        assert "loss_ddf_aux_0" in result.log_dict
        assert "loss_fgl_dn_0" in result.log_dict

    def test_is_train_model_with_watch_metric(self):
        train_model = make_train_model()
        assert train_model.watch_metric == "val_metric/map"
        assert train_model.task == "object_detection"

    def test_validation_step_plain_rtdetr_path(self):
        train_model = make_train_model()
        train_model.set_categories({0: "a", 1: "b"})
        train_model.eval()
        with torch.no_grad():
            result = train_model.validation_step(make_batch(), step=0)
        assert torch.isfinite(result.loss).item()

    def test_default_weights_follow_paper(self):
        args = DFINEObjectDetectionTrainArgs(num_classes=2)
        assert args.fgl_loss_weight == 0.15
        assert args.ddf_loss_weight == 1.5
        assert args.reg_max == 32

    def test_default_batch_and_steps_inherited(self):
        assert DFINEObjectDetectionTrainArgs.default_batch_size == 16
        assert DFINEObjectDetectionTrainArgs.default_steps == 12_000


class TestE2E:
    def test_train_object_detection_dfine(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "dfine1",
            data=data_config(train_json, val_json, images_dir),
            model="dfine/resnet18",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 32,
                "dec_layers": 2,
                "reg_max": REG_MAX,
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
            out=tmp_path / "dfine2",
            data=data_config(train_json, val_json, images_dir),
            model="dfine/resnet18",
            steps=2,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 32,
                "dec_layers": 2,
                "reg_max": REG_MAX,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        model = load_model(result.exported_model_path, device="cpu")
        image = create_image(tmp_path / "probe.jpg", size=(64, 64), seed=0)
        prediction = model.predict(image, threshold=0.05)
        assert set(prediction) == {"labels", "bboxes", "scores"}


class TestLQE:
    def test_zero_init_is_identity_on_scores(self):
        """At init the LQE's last layer is zeroed — the quality scalar is 0,
        so the module must return the scores unchanged (official no-op)."""
        from autovision.tasks.detection.shared.dfine_utils import LQE

        torch.manual_seed(0)
        lqe = LQE(k=4, hidden_dim=64, num_layers=2, reg_max=REG_MAX)
        scores = torch.randn(2, 20, 2)
        corners = torch.randn(2, 20, 4 * (REG_MAX + 1))
        torch.testing.assert_close(lqe(scores, corners), scores)

    def test_trained_offset_shifts_scores(self):
        from autovision.tasks.detection.shared.dfine_utils import LQE

        lqe = LQE(k=4, hidden_dim=64, num_layers=2, reg_max=REG_MAX)
        with torch.no_grad():
            lqe.reg_conf.layers[-1].bias.fill_(0.5)
        scores = torch.zeros(2, 20, 2)
        corners = torch.randn(2, 20, 4 * (REG_MAX + 1))
        shifted = lqe(scores, corners)
        torch.testing.assert_close(shifted, torch.full_like(scores, 0.5))
