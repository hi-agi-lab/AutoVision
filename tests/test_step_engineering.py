# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the training-step engineering items (design doc 26).

W1: fused AdamW flag (plumbing, CPU fallback, state-dict round trip).
W2: criterion vectorized match-index paths — the flat FGL/DDF paths must
match the per-image loops within fp32 tolerance (bitwise on small inputs),
and both flags must be off by default.
"""

import logging

import pytest
import torch
from tests.test_detr_ssl import VARIANT_ARGS

from autovision.geometry.box_ops import (
    elementwise_giou_loss,
    generalized_box_iou_loss,
)
from autovision.optim import build_adamw
from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.dfine_utils import (
    ddf_loss,
    fg_loss,
    flatten_match_indices,
    matched_pair_iou,
    matched_pair_iou_flat,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)

REG_MAX = 8
B, Q, C = 2, 20, 2


# ── W1: build_adamw ─────────────────────────────────────────────────────
class TestBuildAdamW:
    def _param_groups(self):
        params = [torch.randn(8, 8, requires_grad=True), torch.randn(4, requires_grad=True)]
        return [{"params": params[:1], "lr": 1e-4}, {"params": params[1:], "lr": 1e-5, "weight_decay": 0.0}]

    def test_off_is_foreach(self):
        optimizer = build_adamw(
            self._param_groups(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=False
        )
        # torch normalizes an explicit False to its auto sentinel (None);
        # what matters is that the fused kernel is not enabled.
        assert optimizer.defaults["fused"] is not True

    def test_fused_falls_back_on_cpu(self, caplog):
        with caplog.at_level(logging.WARNING):
            optimizer = build_adamw(
                self._param_groups(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=True
            )
        # CPU params: either torch accepts fused and CPU-fused runs (newer
        # torch supports fused on CPU for AdamW) or it raises and we fall
        # back — either way construction succeeds and a fallback warning is
        # emitted when the fused request could not be honored.
        if optimizer.defaults["fused"] is not True:
            assert any("fused_optimizer" in record.message for record in caplog.records)

    def test_state_dict_round_trip_fused_foreach(self):
        groups_a = self._param_groups()
        groups_b = [
            {"params": [p.detach().clone().requires_grad_(True) for p in g["params"]], **{k: v for k, v in g.items() if k != "params"}}
            for g in groups_a
        ]
        opt_a = build_adamw(groups_a, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=True)
        opt_b = build_adamw(groups_b, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=False)
        for group in opt_a.param_groups + opt_b.param_groups:
            for p in group["params"]:
                p.grad = torch.randn_like(p)
        opt_a.step()
        opt_b.step()
        # Cross-loading optimizer state must work (resume across the flag).
        opt_b.load_state_dict(opt_a.state_dict())

    def test_per_step_group_lr_mutation(self):
        """FlatCosine/MultiStep mutate group['lr'] (python float) per step;
        the fused implementation must honor that (doc 26 §10.1)."""
        groups = self._param_groups()
        optimizer = build_adamw(groups, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=True)
        for group in optimizer.param_groups:
            for p in group["params"]:
                p.grad = torch.randn_like(p)
        for _ in range(3):
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * 0.5
        assert optimizer.param_groups[0]["lr"] < 1e-4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
    def test_fused_on_cuda(self):
        params = [torch.randn(8, 8, device="cuda", requires_grad=True)]
        optimizer = build_adamw(
            [{"params": params, "lr": 1e-4}],
            lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, fused=True,
        )
        assert optimizer.defaults["fused"] is True


class TestLtdetrFusedFlag:
    def test_flag_off_by_default(self):
        args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS)
        assert args.fused_optimizer is False
        assert args.criterion_flat_losses is False

    def test_optimizer_plumbing(self):
        args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS, fused_optimizer=True)
        wrapper = DETRObjectDetectionTrain(args)
        optimizer, scheduler = wrapper.get_optimizer(total_steps=100, global_batch_size=8)
        # CPU test env: the fused request falls back — construction and the
        # scheduler interplay must work either way (warning visible).
        optimizer.step()
        scheduler.step()
        for group in optimizer.param_groups:
            assert group["lr"] > 0


# ── W2: flatten helpers ─────────────────────────────────────────────────
class TestFlattenMatchIndices:
    def test_offsets_and_batch_ids(self):
        device = torch.device("cpu")
        indices = [
            (torch.tensor([1, 3]), torch.tensor([0, 2])),
            (torch.tensor([]), torch.tensor([])),
            (torch.tensor([2]), torch.tensor([1])),
        ]
        flat_q, flat_gt, batch_idx = flatten_match_indices(indices, num_queries=10, device=device)
        assert flat_q.tolist() == [1, 3, 22]  # +0, +0, +2*10
        assert flat_gt.tolist() == [0, 2, 1]
        assert batch_idx.tolist() == [0, 0, 2]

    def test_all_empty(self):
        empty = (torch.tensor([]), torch.tensor([]))
        flat_q, flat_gt, batch_idx = flatten_match_indices([empty, empty], 5, torch.device("cpu"))
        assert flat_q.numel() == flat_gt.numel() == batch_idx.numel() == 0


class TestMatchedPairIouFlat:
    def test_parity_with_pairwise_diag(self):
        torch.manual_seed(0)
        pred = torch.rand(50, 4) * 0.6 + 0.2
        target = torch.rand(50, 4) * 0.6 + 0.2
        flat = matched_pair_iou_flat(pred, target)
        loop = matched_pair_iou(pred, target)
        torch.testing.assert_close(flat, loop, rtol=1e-6, atol=1e-7)

    def test_identical_boxes_iou_one(self):
        boxes = torch.rand(10, 4) * 0.4 + 0.3
        flat = matched_pair_iou_flat(boxes, boxes)
        assert torch.allclose(flat, torch.ones(10), atol=1e-6)

    def test_degenerate_boxes_stay_finite(self):
        zero = torch.zeros(3, 4)
        flat = matched_pair_iou_flat(zero, zero)
        assert torch.isfinite(flat).all()


class TestElementwiseGiou:
    def test_parity_with_pairwise_diag(self):
        torch.manual_seed(1)
        pred = torch.rand(40, 4) * 0.6 + 0.2
        target = torch.rand(40, 4) * 0.6 + 0.2
        from autovision.tasks.detection.shared.denoising import (
            box_cxcywh_to_xyxy,
        )

        pred_xyxy = box_cxcywh_to_xyxy(pred)
        target_xyxy = box_cxcywh_to_xyxy(target)
        flat = elementwise_giou_loss(pred_xyxy, target_xyxy)
        loop = torch.diag(generalized_box_iou_loss(pred_xyxy, target_xyxy))
        torch.testing.assert_close(flat, loop, rtol=1e-6, atol=1e-7)

    def test_identical_boxes_zero(self):
        # Valid xyxy (x2 > x1, y2 > y1) — the GIoU of identical boxes is 0.
        top_left = torch.rand(5, 2) * 0.3
        size = torch.rand(5, 2) * 0.3 + 0.1
        boxes = torch.cat([top_left, top_left + size], dim=1)
        loss = elementwise_giou_loss(boxes, boxes)
        assert loss.max().item() == pytest.approx(0.0, abs=1e-6)


# ── W2: criterion flat paths ────────────────────────────────────────────
def _harness(seed: int):
    torch.manual_seed(seed)
    targets = [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])},
        {"boxes": torch.rand(2, 4) * 0.3 + 0.3, "labels": torch.tensor([1, 0])},
    ]
    outputs = [
        {"pred_logits": torch.randn(B, Q, C), "pred_boxes": torch.rand(B, Q, 4) * 0.6 + 0.2}
        for _ in range(2)
    ]
    corners = [torch.randn(B, Q, 4 * (REG_MAX + 1)) for _ in range(2)]
    refs = [torch.rand(B, Q, 4) * 0.6 + 0.2 for _ in range(2)]
    return targets, outputs, corners, refs


def _run_criterion(harness, flat: bool, distill_params=None, distill_targets=None):
    targets, outputs, corners, refs = harness
    criterion = DFINECriterionV2(
        HungarianMatcher(),
        reg_max=REG_MAX,
        flat_losses=flat,
        distill_params=distill_params,
    )
    criterion.train()
    kwargs = {"distill_targets": distill_targets} if distill_params is not None else {}
    return criterion(
        outputs, targets, world_size=1,
        corners_list=corners, ref_boxes_list=refs, **kwargs,
    )


class TestCriterionFlatPaths:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fgl_ddf_flat_equivalence(self, seed):
        harness = _harness(seed)
        off = _run_criterion(harness, flat=False)
        on = _run_criterion(harness, flat=True)
        for key in off:
            if "fgl" in key or "ddf" in key:
                torch.testing.assert_close(off[key], on[key], rtol=1e-5, atol=1e-6)

    def test_dn_fg_loss_flat_equivalence(self):
        targets, outputs, corners, refs = _harness(7)
        indices = [(torch.tensor([0, 2]), torch.tensor([0, 1])), (torch.tensor([1]), torch.tensor([0]))]
        num_boxes = torch.tensor(3.0)
        loop = fg_loss(
            corners[0], refs[0], outputs[0]["pred_boxes"], targets, indices,
            num_boxes, REG_MAX, flat_indexing=False,
        )
        flat = fg_loss(
            corners[0], refs[0], outputs[0]["pred_boxes"], targets, indices,
            num_boxes, REG_MAX, flat_indexing=True,
        )
        torch.testing.assert_close(loop, flat, rtol=1e-5, atol=1e-6)

    def test_ddf_loss_flat_equivalence(self):
        targets, outputs, corners, refs = _harness(8)
        indices = [(torch.tensor([0, 2]), torch.tensor([0, 1])), (torch.tensor([1]), torch.tensor([0]))]
        kwargs = dict(
            student_corners=corners[0], teacher_corners=corners[-1],
            student_logits=outputs[0]["pred_logits"], teacher_logits=outputs[-1]["pred_logits"],
            student_boxes=outputs[0]["pred_boxes"], teacher_boxes=outputs[-1]["pred_boxes"],
            indices=indices, reg_max=REG_MAX, targets=targets,
        )
        loop = ddf_loss(**kwargs, flat_indexing=False)
        flat = ddf_loss(**kwargs, flat_indexing=True)
        torch.testing.assert_close(loop[0], flat[0], rtol=1e-5, atol=1e-6)

    def test_off_by_default(self):
        criterion = DFINECriterionV2(HungarianMatcher(), reg_max=REG_MAX)
        assert criterion.flat_losses is False
        assert criterion.distill_params is None
        assert "loss_distill_cls" not in criterion.weight_dict


class TestLtdetrFlatFlag:
    def test_wrapper_plumbs_flag(self):
        args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS, criterion_flat_losses=True)
        wrapper = DETRObjectDetectionTrain(args)
        assert wrapper.criterion.flat_losses is True


# ── W2: flat box/VFL loss paths (loss_boxes + loss_labels_vfl) ─────────


class TestFlatBoxAndVFLLosses:
    """The flat paths must reproduce the legacy per-image loops to
    < 1e-6 (doc 26 W2 convention): fp32 summation order may differ."""

    def _case(self, seed: int = 0, batch: int = 4, queries: int = 30, classes: int = 5):
        torch.manual_seed(seed)
        logits = torch.randn(batch, queries, classes)
        boxes = torch.rand(batch, queries, 4) * 0.6 + 0.05
        targets, indices = [], []
        counts = [0, 3, 7, 12, 1, 5][:batch]
        for n in counts:
            targets.append(
                {
                    "boxes": torch.rand(n, 4) * 0.4 + 0.1,
                    "labels": torch.randint(0, classes, (n,)),
                }
            )
            if n:
                indices.append(
                    (torch.randperm(queries)[:n], torch.arange(n))
                )
            else:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                    )
                )
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        num_boxes = torch.tensor(float(sum(n for n in counts)))
        return outputs, targets, indices, num_boxes

    def test_loss_boxes_flat_matches_legacy(self):
        from autovision.tasks.detection.shared.rtdetrv2_criterion import loss_boxes

        outputs, targets, indices, num_boxes = self._case()
        for iou_term in ("giou", "siou"):
            legacy = loss_boxes(outputs, targets, indices, num_boxes, iou_term=iou_term)
            flat = loss_boxes(
                outputs, targets, indices, num_boxes,
                iou_term=iou_term, flat_indexing=True,
            )
            assert legacy.keys() == flat.keys()
            for key in legacy:
                torch.testing.assert_close(flat[key], legacy[key], rtol=1e-6, atol=1e-7)

    def test_vfl_flat_matches_legacy(self):
        from autovision.tasks.detection.shared.rtdetrv2_criterion import (
            loss_labels_vfl,
        )

        outputs, targets, indices, num_boxes = self._case(seed=1)
        mask = torch.ones(outputs["pred_logits"].shape[:2], dtype=torch.bool)
        mask[0, 20:] = False
        for query_mask in (None, mask):
            legacy = loss_labels_vfl(
                outputs, targets, indices, num_boxes, query_mask=query_mask
            )
            flat = loss_labels_vfl(
                outputs, targets, indices, num_boxes,
                query_mask=query_mask, flat_indexing=True,
            )
            torch.testing.assert_close(flat, legacy, rtol=1e-6, atol=1e-7)

    def test_flat_paths_with_no_matches_at_all(self):
        from autovision.tasks.detection.shared.rtdetrv2_criterion import (
            loss_boxes,
            loss_labels_vfl,
        )

        outputs, targets, indices, num_boxes = self._case(seed=2)
        empty_indices = [
            (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
            for _ in indices
        ]
        legacy = loss_boxes(outputs, targets, empty_indices, num_boxes)
        flat = loss_boxes(outputs, targets, empty_indices, num_boxes, flat_indexing=True)
        for key in legacy:
            torch.testing.assert_close(flat[key], legacy[key], rtol=1e-6, atol=1e-7)
        vfl = loss_labels_vfl(outputs, targets, empty_indices, num_boxes)
        vfl_flat = loss_labels_vfl(
            outputs, targets, empty_indices, num_boxes, flat_indexing=True
        )
        torch.testing.assert_close(vfl_flat, vfl, rtol=1e-6, atol=1e-7)

    def test_criterion_flat_flag_end_to_end(self, tmp_path):
        """The flag threads through the full criterion and training loop."""
        from autovision import train_object_detection
        from autovision.data.coco8 import ensure_coco8

        data = ensure_coco8(tmp_path / "coco8")
        result = train_object_detection(
            out=tmp_path / "run",
            data=data,
            model="detr/n",
            steps=2,
            batch_size=4,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
                "criterion_flat_losses": True,
            },
            transform_args={
                "mosaic_prob": 0.0,
                "scale_jitter": None,
                "photometric_distort_prob": None,
                "random_zoom_out_prob": None,
                "random_iou_crop_prob": None,
            },
            trainer_args={"val_every_num_steps": 100_000},
        )
        assert result.steps_run == 2
        assert "val_metric/map" in result.best_metrics or result.best_metrics == {}
