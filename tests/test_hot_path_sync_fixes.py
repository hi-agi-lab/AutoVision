# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Equivalence tests for the hot-path sync-removal refactors.

Each test pins the refactored implementation against a verbatim copy of
the pre-refactor code (the "reference" functions below), so a behavior
drift — not just a crash — fails the suite. All cases run on CPU; the
refactors only changed WHERE tensors live during host round-trips, so
CPU equality implies GPU equality for the same values.

Covered refactors:

- ``HungarianMatcher``: per-image ``.cpu()`` transfers batched into one
  bulk transfer (assignments must be bit-identical — scipy consumes the
  same cost matrix);
- ``DFINECriterionV2._get_go_indices``: sequential ``.item()`` dict loop
  replaced by stable sort + group-firsts (exact order parity on
  unambiguous rows; ties resolve deterministically to the
  lexicographically-first max-count pair, where the old unstable
  ``argsort`` pick was unspecified);
- ``per_class_dynamic_nms``: per-class GPU partitioning moved to the
  host (the greedy scan itself is unchanged);
- the Dome dynamic-query selection loop: per-image GPU boolean indexing
  replaced by one host transfer + host-side selection + index gathers
  (``dome_decoder._get_decoder_input_dynamic``);
- ``ddf_loss``: scalar counts moved on-device (identical arithmetic);
- ``_pair_two_crop``: four host transfers packed into two.
"""

import importlib.util
import math

import pytest
import torch

from autovision.tasks.detection.detr.dome_decoder import (
    DomeDFINEv2Decoder,
)
from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.dfine_utils import (
    ddf_loss,
)
from autovision.tasks.detection.shared.dynamic_nms import (
    per_class_dynamic_nms,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
    forward_match,
)
from autovision.tasks.detection.shared.mwas import (
    adaptive_window_mask,
)

REG_MAX = 8


# ── reference implementations (verbatim pre-refactor code) ──────────────


def _reference_match_per_image(matcher, outputs, targets):
    """Per-image reference for the batched-padded matcher.

    Same costs (GIoU on normalized xyxy — the official D-FINE/RT-DETR
    matcher convention), same guards, one host transfer + one scipy solve
    per image: the batched implementation must produce identical
    assignments.
    """
    from scipy.optimize import linear_sum_assignment

    from autovision.geometry.box_ops import (
        generalized_box_iou_loss,
    )

    def to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    batch_logits = outputs["pred_logits"]
    batch_boxes = torch.nan_to_num(
        outputs["pred_boxes"], nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(min=0.0, max=1.0)
    indices = []
    for batch_index in range(batch_logits.shape[0]):
        target = targets[batch_index]
        if target["boxes"].numel() == 0:
            indices.append(
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
            )
            continue
        target_boxes = target["boxes"].to(batch_boxes.device)
        target_labels = target["labels"].to(batch_logits.device)
        pred_logits = batch_logits[batch_index]
        pred_boxes = batch_boxes[batch_index]
        pred_scores = pred_logits.sigmoid()
        tgt_scores = pred_scores[:, target_labels]
        neg_cost = (
            (1 - matcher.alpha)
            * tgt_scores.pow(matcher.gamma)
            * (-(1 - tgt_scores + 1e-8).log())
        )
        pos_cost = (
            matcher.alpha
            * (1 - tgt_scores).pow(matcher.gamma)
            * (-(tgt_scores + 1e-8).log())
        )
        class_cost = pos_cost - neg_cost
        l1_cost = torch.cdist(pred_boxes, target_boxes, p=1)
        giou_cost = generalized_box_iou_loss(
            to_xyxy(pred_boxes), to_xyxy(target_boxes)
        )
        cost = (
            matcher.cost_class * class_cost
            + matcher.cost_bbox * l1_cost
            + matcher.cost_giou * giou_cost
        )
        cost = torch.nan_to_num(cost, nan=1.0)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append(
            (
                torch.as_tensor(rows, dtype=torch.int64),
                torch.as_tensor(cols, dtype=torch.int64),
            )
        )
    return indices


def _reference_go_indices_loop(ind):
    """Pre-refactor per-image body of _get_go_indices (sequential loop)."""
    unique, counts = torch.unique(ind, return_counts=True, dim=0)
    count_sort_indices = torch.argsort(counts, descending=True)
    unique_sorted = unique[count_sort_indices]
    column_to_row: dict[int, int] = {}
    for idx in unique_sorted:
        row_idx, col_idx = idx[0].item(), idx[1].item()
        if row_idx not in column_to_row:
            column_to_row[row_idx] = col_idx
    return (
        torch.tensor(list(column_to_row.keys())).long(),
        torch.tensor(list(column_to_row.values())).long(),
    )


def _reference_per_class_dynamic_nms(boxes, scores, class_ids, iou_thresholds):
    """Pre-refactor per-class loop (GPU-tensor form; CPU inputs == same
    semantics, which is the point of the equivalence check)."""
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    keep_mask = torch.zeros(boxes.shape[0], dtype=torch.bool)
    for class_id in class_ids.unique():
        class_mask = class_ids == class_id
        keep_class = _reference_nms_single_class(
            boxes[class_mask], scores[class_mask], iou_thresholds[class_mask]
        )
        original_indices = torch.nonzero(class_mask, as_tuple=True)[0]
        keep_mask[original_indices[keep_class]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _reference_nms_single_class(boxes, scores, iou_thresholds):
    """Pre-refactor _nms_single_class (torch ops, no host shortcut)."""
    import numpy as np
    from torchvision.ops import box_iou

    num = boxes.shape[0]
    order = scores.argsort(descending=True)
    boxes_cpu = boxes[order].detach()
    iou_matrix = box_iou(boxes_cpu, boxes_cpu).numpy()
    thresholds = iou_thresholds[order].detach().cpu().numpy()
    keep_flags = np.ones(num, dtype=bool)
    keep: list[int] = []
    for i in range(num):
        if not keep_flags[i]:
            continue
        keep.append(i)
        if i < num - 1:
            suppressed = iou_matrix[i, i + 1 :] >= thresholds[i]
            keep_flags[i + 1 :] &= ~suppressed
    index = torch.tensor(keep, dtype=torch.long)
    return order[index]


def _reference_ddf_loss(
    student_corners,
    teacher_corners,
    student_logits,
    teacher_logits,
    student_boxes,
    teacher_boxes,
    indices,
    reg_max,
    targets=None,
    temperature=5.0,
    counts=None,
):
    """Pre-refactor ddf_loss (float() scalars + .any() guards)."""
    from torch import nn

    from autovision.tasks.detection.shared.dfine_utils import (
        matched_pair_iou,
    )

    num_bins = reg_max + 1
    b, q = student_boxes.shape[:2]
    teacher_probs = torch.softmax(
        teacher_corners.reshape(b, q, 4, num_bins) / temperature, dim=-1
    ).detach()
    student_log_probs = torch.log_softmax(
        student_corners.reshape(b, q, 4, num_bins) / temperature, dim=-1
    )
    weight_targets_local = teacher_logits.detach().sigmoid().max(dim=-1).values
    mask = torch.zeros(b, q, dtype=torch.bool)
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        mask[batch_index, src] = True
        if targets is not None:
            reference_boxes = targets[batch_index]["boxes"][dst]
        else:
            reference_boxes = teacher_boxes[batch_index, src].detach()
        ious = matched_pair_iou(
            student_boxes[batch_index, src].detach(), reference_boxes
        )
        weight_targets_local[batch_index, src] = ious.to(weight_targets_local.dtype)
    weight_targets_local = weight_targets_local.detach()
    elem_weight = weight_targets_local.unsqueeze(-1).expand(-1, -1, 4).reshape(-1)
    kl = nn.KLDivLoss(reduction="none")(
        student_log_probs.reshape(-1, num_bins),
        teacher_probs.reshape(-1, num_bins),
    ).sum(-1)
    loss_match_local = elem_weight * (temperature**2) * kl
    query_mask_flat = mask.reshape(-1)
    elem_mask = query_mask_flat.unsqueeze(-1).expand(-1, 4).reshape(-1)
    if counts is None:
        batch_size = max(b, 1)
        num_pos = max(float((query_mask_flat.sum() * 8.0 / batch_size) ** 0.5), 1e-6)
        num_neg = max(
            float(((~query_mask_flat).sum() * 8.0 / batch_size) ** 0.5), 1e-6
        )
    else:
        num_pos, num_neg = counts
    m1 = (
        loss_match_local[elem_mask].mean()
        if mask.any()
        else loss_match_local.sum() * 0.0
    )
    m2 = (
        loss_match_local[~elem_mask].mean()
        if (~mask).any()
        else loss_match_local.sum() * 0.0
    )
    loss = (m1 * num_pos + m2 * num_neg) / (num_pos + num_neg)
    return loss, num_pos, num_neg


# ── matcher ─────────────────────────────────────────────────────────────


class TestMatcherBatchedTransfer:
    def _random_batch(self, generator):
        batch, queries, classes = 5, 24, 6
        logits = torch.randn(batch, queries, classes, generator=generator)
        boxes = torch.rand(batch, queries, 4, generator=generator)
        targets = []
        for i in range(batch):
            n = int(torch.randint(0, 7, (1,), generator=generator))
            targets.append(
                {
                    "boxes": torch.rand(n, 4, generator=generator) * 0.4 + 0.1,
                    "labels": torch.randint(0, classes, (n,), generator=generator),
                }
            )
        return {"pred_logits": logits, "pred_boxes": boxes}, targets

    def test_assignments_identical_to_per_image_transfer(self):
        generator = torch.Generator().manual_seed(11)
        outputs, targets = self._random_batch(generator)
        matcher = HungarianMatcher()
        got = forward_match(matcher, outputs, targets)
        want = _reference_match_per_image(matcher, outputs, targets)
        assert len(got) == len(want)
        for (g_src, g_dst), (w_src, w_dst) in zip(got, want):
            assert g_src.dtype == torch.int64 and g_dst.dtype == torch.int64
            assert torch.equal(g_src, w_src)
            assert torch.equal(g_dst, w_dst)

    def test_all_empty_targets(self):
        outputs = {
            "pred_logits": torch.randn(3, 10, 4),
            "pred_boxes": torch.rand(3, 10, 4),
        }
        targets = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(3)
        ]
        got = forward_match(HungarianMatcher(), outputs, targets)
        assert all(src.numel() == 0 and dst.numel() == 0 for src, dst in got)

    def test_single_query_single_target(self):
        outputs = {
            "pred_logits": torch.randn(1, 1, 2),
            "pred_boxes": torch.rand(1, 1, 4),
        }
        targets = [{"boxes": torch.rand(1, 4), "labels": torch.tensor([1])}]
        (src, dst), = forward_match(HungarianMatcher(), outputs, targets)
        assert src.tolist() == [0] and dst.tolist() == [0]

    def test_mixed_empty_and_uneven_counts(self):
        """Empty-GT images keep the short path; uneven counts pad correctly."""
        generator = torch.Generator().manual_seed(23)
        outputs, targets = self._random_batch(generator)
        targets[2] = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        matcher = HungarianMatcher()
        got = forward_match(matcher, outputs, targets)
        want = _reference_match_per_image(matcher, outputs, targets)
        for (g_src, g_dst), (w_src, w_dst) in zip(got, want):
            assert torch.equal(g_src, w_src) and torch.equal(g_dst, w_dst)
        assert got[2][0].numel() == 0 and got[2][1].numel() == 0

    def test_more_targets_than_queries(self):
        generator = torch.Generator().manual_seed(5)
        outputs = {
            "pred_logits": torch.randn(2, 8, 3, generator=generator),
            "pred_boxes": torch.rand(2, 8, 4, generator=generator),
        }
        targets = [
            {
                "boxes": torch.rand(25, 4, generator=generator) * 0.4 + 0.1,
                "labels": torch.randint(0, 3, (25,), generator=generator),
            }
            for _ in range(2)
        ]
        matcher = HungarianMatcher()
        got = forward_match(matcher, outputs, targets)
        want = _reference_match_per_image(matcher, outputs, targets)
        for (g_src, g_dst), (w_src, w_dst) in zip(got, want):
            assert torch.equal(g_src, w_src) and torch.equal(g_dst, w_dst)
        assert got[0][0].numel() == 8  # every query matched exactly once

    def test_nan_pred_boxes_guarded(self):
        generator = torch.Generator().manual_seed(7)
        outputs, targets = self._random_batch(generator)
        outputs["pred_boxes"][0, :3] = float("nan")
        outputs["pred_boxes"][0, 3] = float("inf")
        matcher = HungarianMatcher()
        got = forward_match(matcher, outputs, targets)
        want = _reference_match_per_image(matcher, outputs, targets)
        for (g_src, g_dst), (w_src, w_dst) in zip(got, want):
            assert torch.equal(g_src, w_src) and torch.equal(g_dst, w_dst)

    def test_deterministic(self):
        generator = torch.Generator().manual_seed(13)
        outputs, targets = self._random_batch(generator)
        matcher = HungarianMatcher()
        first = forward_match(matcher, outputs, targets)
        second = forward_match(matcher, outputs, targets)
        assert all(
            torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            for a, b in zip(first, second)
        )

    def test_multi_seed_parity(self):
        matcher = HungarianMatcher()
        for seed in range(1, 40):
            generator = torch.Generator().manual_seed(seed)
            outputs, targets = self._random_batch(generator)
            got = forward_match(matcher, outputs, targets)
            want = _reference_match_per_image(matcher, outputs, targets)
            for (g_src, g_dst), (w_src, w_dst) in zip(got, want):
                assert torch.equal(g_src, w_src), f"rows differ at seed {seed}"
                assert torch.equal(g_dst, w_dst), f"cols differ at seed {seed}"

    def test_giou_batched_matches_pairwise_exact(self):
        """The vmap'd per-image GIoU is bit-identical to the pairwise loop
        (locks the single-source-of-truth math)."""
        from autovision.geometry.box_ops import generalized_box_iou_loss
        from autovision.tasks.detection.shared.matcher import _giou_batched

        generator = torch.Generator().manual_seed(3)
        pred = torch.rand(4, 20, 4, generator=generator)
        tgt = torch.rand(4, 7, 4, generator=generator)
        batched = _giou_batched(pred, tgt)
        reference = torch.stack(
            [generalized_box_iou_loss(pred[i], tgt[i]) for i in range(4)]
        )
        assert torch.equal(batched, reference)

    def test_xyxy_conversion_helper(self):
        from autovision.tasks.detection.shared.matcher import _cxcywh_to_xyxy_norm

        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]])
        assert torch.allclose(_cxcywh_to_xyxy_norm(boxes), expected)


# ── _get_go_indices ─────────────────────────────────────────────────────


class TestGoIndicesVectorized:
    @staticmethod
    def _layered_pairs(batch, queries, num_gts, layers, generator):
        """Simulate per-layer Hungarian assignments: each layer matches each
        image's queries to distinct GTs, so repeated pairs carry real
        multi-layer counts."""
        per_image = []
        for _ in range(batch):
            union = torch.zeros(0, 2, dtype=torch.int64)
            for _ in range(layers):
                n = min(int(torch.randint(1, num_gts + 1, (1,), generator=generator)), queries)
                rows = torch.randperm(queries, generator=generator)[:n]
                cols = torch.randperm(num_gts, generator=generator)[:n]
                union = torch.cat([union, torch.stack([rows, cols], 1)])
            per_image.append(union)
        return per_image

    def test_exact_parity_with_sequential_loop(self):
        generator = torch.Generator().manual_seed(23)
        criterion = DFINECriterionV2(HungarianMatcher(), reg_max=REG_MAX)
        for _ in range(40):
            batch = int(torch.randint(1, 4, (1,), generator=generator))
            per_image = self._layered_pairs(batch, 16, 9, layers=3, generator=generator)
            indices = [
                (img[:, 0], img[:, 1]) for img in per_image
            ]
            indices_aux = [
                [
                    (torch.randperm(16, generator=generator)[:6],
                     torch.randint(0, 9, (6,), generator=generator))
                    for _ in range(batch)
                ]
            ]
            got = criterion._get_go_indices(indices, indices_aux)
            # reference: the old accumulation + loop
            accumulated = indices
            for indices_aux_round in indices_aux:
                accumulated = [
                    (
                        torch.cat([a[0], b[0]]),
                        torch.cat([a[1], b[1]]),
                    )
                    for a, b in zip(accumulated, indices_aux_round)
                ]
            for (src, dst), union in zip(got, accumulated):
                ind = torch.cat([union[0][:, None], union[1][:, None]], 1)
                want_src, want_dst = _reference_go_indices_loop(ind)
                # Same rows in both. Per row: the picked pair must carry the
                # row's max count; when that maximum is UNIQUE the pick must
                # match the reference exactly. On ties the old pick was
                # unspecified (unstable argsort) — only the max-count
                # property holds there.
                counts: dict[tuple[int, int], int] = {}
                for row, col in ind.tolist():
                    counts[(row, col)] = counts.get((row, col), 0) + 1
                assert set(src.tolist()) == set(want_src.tolist())
                max_count = {}
                for (row, _), count in counts.items():
                    max_count[row] = max(max_count.get(row, 0), count)
                row_picks = dict(zip(src.tolist(), dst.tolist()))
                for row, col in row_picks.items():
                    assert counts[(row, col)] == max_count[row]
                    ties = [
                        c
                        for (r, c), count in counts.items()
                        if r == row and count == max_count[row]
                    ]
                    if len(ties) == 1:
                        assert col == want_dst[
                            want_src.tolist().index(row)
                        ], f"row {row}: pick {col} != reference"
                assert src.dtype == torch.int64 and dst.dtype == torch.int64

    def test_picked_pair_has_max_count(self):
        generator = torch.Generator().manual_seed(5)
        criterion = DFINECriterionV2(HungarianMatcher(), reg_max=REG_MAX)
        for _ in range(30):
            per_image = self._layered_pairs(1, 12, 6, layers=4, generator=generator)[0]
            (src, dst), = criterion._get_go_indices(
                [(per_image[:, 0], per_image[:, 1])], []
            )
            counts: dict[tuple[int, int], int] = {}
            for row, col in per_image.tolist():
                counts[(row, col)] = counts.get((row, col), 0) + 1
            row_max: dict[int, int] = {}
            for (row, col), count in counts.items():
                row_max[row] = max(row_max.get(row, 0), count)
            for src_i, dst_i in zip(src.tolist(), dst.tolist()):
                assert counts[(src_i, dst_i)] == row_max[src_i]

    def test_empty_image(self):
        criterion = DFINECriterionV2(HungarianMatcher(), reg_max=REG_MAX)
        (src, dst), = criterion._get_go_indices(
            [(torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))],
            [],
        )
        assert src.numel() == 0 and dst.numel() == 0
        assert src.dtype == torch.int64 and dst.dtype == torch.int64


# ── per-class dynamic NMS ───────────────────────────────────────────────


class TestPerClassDynamicNmsHostPath:
    def _random_candidates(self, n, classes, generator):
        boxes = torch.rand(n, 2, generator=generator) * 0.5
        wh = torch.rand(n, 2, generator=generator) * 0.2 + 0.02
        boxes_xyxy = torch.cat(
            [boxes, (boxes + wh).clamp(max=1.0)], dim=-1
        )
        scores = torch.rand(n, generator=generator)
        class_ids = torch.randint(0, classes, (n,), generator=generator)
        thresholds = torch.rand(n, generator=generator) * 0.5 + 0.1
        return boxes_xyxy, scores, class_ids, thresholds

    def test_identical_to_gpu_loop_semantics(self):
        generator = torch.Generator().manual_seed(31)
        for _ in range(30):
            n = int(torch.randint(1, 60, (1,), generator=generator))
            boxes, scores, class_ids, thresholds = self._random_candidates(
                n, classes=4, generator=generator
            )
            got = per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
            want = _reference_per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
            assert torch.equal(got, want)
            assert got.device.type == boxes.device.type
            assert got.dtype == torch.int64

    def test_empty_input(self):
        empty = torch.zeros(0, 4)
        got = per_class_dynamic_nms(
            empty, torch.zeros(0), torch.zeros(0, dtype=torch.int64), torch.zeros(0)
        )
        assert got.numel() == 0 and got.dtype == torch.int64

    def test_single_class_overlap(self):
        # two heavily overlapping boxes, tight threshold -> later one dies
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 1.0, 1.0]])
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.zeros(2, dtype=torch.int64)
        thresholds = torch.tensor([0.3, 0.3])
        got = per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
        assert got.tolist() == [0]

    def test_disjoint_classes_never_suppress_each_other(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.tensor([0, 1], dtype=torch.int64)
        thresholds = torch.tensor([0.3, 0.3])
        got = per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
        assert got.tolist() == [0, 1]


# ── Dome dynamic-query selection ────────────────────────────────────────


class _TinyDomeDecoder:
    """The TestDomeDecoderUnit decoder recipe (kept local to avoid
    importing test helpers across files)."""

    @staticmethod
    def build(**overrides) -> DomeDFINEv2Decoder:
        kwargs = dict(
            num_classes=2,
            hidden_dim=64,
            num_queries=10,
            feat_channels=[64, 64, 64],
            feat_strides=[8, 16, 32],
            num_levels=3,
            nhead=4,
            num_layers=2,
            num_denoising=10,
            reg_max=REG_MAX,
            min_num_select=10,
            max_num_select=30,
        )
        kwargs.update(overrides)
        decoder = DomeDFINEv2Decoder(**kwargs)
        decoder.eval()
        return decoder


class TestDynamicSelectionEquivalence:
    @staticmethod
    def _inputs(decoder, batch, generator, sparse_density=True):
        spatial_shapes = [(5, 8), (4, 8), (2, 8)]
        total_cells = sum(h * w for h, w in spatial_shapes)
        memory = torch.randn(batch, total_cells, 64, generator=generator)
        pooled = (
            (torch.rand(batch, 1, 3, 3, generator=generator) < 0.4).float()
            if sparse_density
            else torch.ones(batch, 1, 3, 3)
        )
        info = {
            "density_pooled": pooled,
            "window_mask": adaptive_window_mask(pooled),
            "reg_value": torch.zeros(batch, 1),
        }
        return memory, spatial_shapes, info

    @staticmethod
    def _reference_selection(decoder, memory, spatial_shapes, defe_info):
        """The pre-refactor selection loop, re-derived through the
        decoder's own modules (candidate scoring/gating is unchanged by
        the refactor — the equivalence target is the per-image loop)."""
        anchors, valid_mask = decoder._generate_anchors(
            spatial_shapes, device=memory.device, dtype=memory.dtype
        )
        if anchors.shape[0] == 1 and memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        mem_in = valid_mask.to(memory.dtype) * memory
        output_memory = decoder.enc_output(mem_in)
        enc_logits = decoder.enc_score_head(output_memory)
        enc_coords = decoder.enc_bbox_head(output_memory) + anchors
        topk = min(decoder.max_num_select, enc_logits.shape[1])
        enc_scores, enc_class_ids = enc_logits.max(-1)
        topk_ind = enc_scores.topk(topk, dim=-1).indices

        def _gather(t):
            return t.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

        mem_k = _gather(output_memory)
        logits_k = _gather(enc_logits)
        scores_k = enc_scores.gather(1, topk_ind)
        class_k = enc_class_ids.gather(1, topk_ind)
        anchors_k = _gather(anchors)
        coords_k = _gather(enc_coords)

        window_mask = defe_info["window_mask"]
        pooled = defe_info["density_pooled"]
        n_h, n_w = window_mask.shape[1], window_mask.shape[2]
        # Gating keys on the ANCHOR centers (baseline semantics), not the
        # predicted box centers.
        anchor_x = anchors_k[..., 0].sigmoid()
        anchor_y = anchors_k[..., 1].sigmoid()
        window_y = (anchor_y * n_h).long().clamp(0, n_h - 1)
        window_x = (anchor_x * n_w).long().clamp(0, n_w - 1)
        batch_index = torch.arange(mem_k.shape[0])
        gate = window_mask[batch_index[:, None], window_y, window_x]
        min_num = min(decoder.min_num_select, topk)
        gate[:, :min_num] = True

        pooled_h, pooled_w = pooled.shape[-2:]
        kept_memory, kept_coords, kept_logits, queries_num = [], [], [], []
        for b in range(mem_k.shape[0]):
            rest = gate[b, min_num:]
            mem_i = torch.cat([mem_k[b, :min_num], mem_k[b, min_num:][rest]], dim=0)
            scores_i = torch.cat(
                [scores_k[b, :min_num], scores_k[b, min_num:][rest]], dim=0
            )
            class_i = torch.cat(
                [class_k[b, :min_num], class_k[b, min_num:][rest]], dim=0
            )
            coords_i = torch.cat(
                [coords_k[b, :min_num], coords_k[b, min_num:][rest]], dim=0
            )
            logits_i = torch.cat(
                [logits_k[b, :min_num], logits_k[b, min_num:][rest]], dim=0
            )
            proposals = coords_i.sigmoid()
            if proposals.shape[0] > min_num:
                center_y = proposals[:, 1]
                center_x = proposals[:, 0]
                density_y = (center_y * pooled_h).long().clamp(0, pooled_h - 1)
                density_x = (center_x * pooled_w).long().clamp(0, pooled_w - 1)
                density = pooled[b, 0, density_y, density_x].detach()
                iou_thresholds = (
                    decoder.nms_iou_base + decoder.nms_iou_density_scale * density
                )
                keep = per_class_dynamic_nms(
                    box_cxcywh_to_xyxy(proposals).detach(),
                    scores_i.detach(),
                    class_i.detach(),
                    iou_thresholds.detach(),
                )
                keep = keep[keep >= min_num]
                keep = torch.cat([torch.arange(min_num), keep])
                mem_i = mem_i[keep]
                coords_i = coords_i[keep]
                logits_i = logits_i[keep]
            kept_memory.append(mem_i)
            kept_coords.append(coords_i)
            kept_logits.append(logits_i)
            queries_num.append(int(mem_i.shape[0]))
        return kept_memory, kept_coords, kept_logits, queries_num

    def test_selection_matches_reference_loop(self):
        generator = torch.Generator().manual_seed(47)
        for sparse in (True, False):
            decoder = _TinyDomeDecoder.build()
            batch = 3
            memory, spatial_shapes, info = self._inputs(
                decoder, batch, generator, sparse_density=sparse
            )
            with torch.no_grad():
                got = decoder._get_decoder_input_dynamic(memory, spatial_shapes, info)
                (
                    ref_mem,
                    ref_coords,
                    ref_logits,
                    ref_queries,
                ) = self._reference_selection(decoder, memory, spatial_shapes, info)
            content, coords, queries, valid, enc_logits, enc_boxes = got
            assert queries == ref_queries
            padded = max(ref_queries)
            assert content.shape == (batch, padded, 64)
            for b in range(batch):
                n_b = ref_queries[b]
                assert torch.equal(content[b, :n_b], ref_mem[b])
                assert torch.equal(coords[b, :n_b], ref_coords[b])
                assert torch.equal(enc_logits[b, :n_b], ref_logits[b])
                assert torch.equal(enc_boxes[b, :n_b], ref_coords[b].sigmoid())
                assert bool(valid[b, :n_b].all()) and not bool(valid[b, n_b:].any())
                # D5: padded rows buried
                if n_b < padded:
                    assert bool((enc_logits[b, n_b:] == -1e4).all())

    def test_queries_within_bounds(self):
        generator = torch.Generator().manual_seed(59)
        decoder = _TinyDomeDecoder.build()
        memory, spatial_shapes, info = self._inputs(decoder, 2, generator)
        with torch.no_grad():
            _, _, queries, valid, _, _ = decoder._get_decoder_input_dynamic(
                memory, spatial_shapes, info
            )
        total_cells = sum(h * w for h, w in spatial_shapes)
        for n_b in queries:
            assert decoder.min_num_select <= n_b <= min(
                decoder.max_num_select, total_cells
            )


# ── ddf_loss on-device scalars ──────────────────────────────────────────


class TestDDFOnDeviceScalars:
    @staticmethod
    def _stacked(t):
        return t.reshape(t.shape[0], t.shape[1], 4 * (REG_MAX + 1))

    def test_matches_reference_on_random_inputs(self):
        generator = torch.Generator().manual_seed(71)
        for empty_mask in (False, True):
            student = self._stacked(torch.randn(2, 5, 4, REG_MAX + 1, generator=generator))
            teacher = self._stacked(torch.randn(2, 5, 4, REG_MAX + 1, generator=generator))
            boxes = torch.rand(2, 5, 4, generator=generator) * 0.3 + 0.3
            logits = torch.rand(2, 5, 2, generator=generator)
            if empty_mask:
                indices = [
                    (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
                    for _ in range(2)
                ]
            else:
                indices = [
                    (torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2])),
                    (torch.tensor([1, 3]), torch.tensor([0, 2])),
                ]
            got_loss, got_pos, got_neg = ddf_loss(
                student, teacher, logits, logits, boxes, boxes, indices, REG_MAX
            )
            want_loss, want_pos, want_neg = _reference_ddf_loss(
                student, teacher, logits, logits, boxes, boxes, indices, REG_MAX
            )
            assert torch.allclose(got_loss, want_loss, atol=1e-6)
            # the scalar counts move from float64 Python floats to float32
            # tensors — equal to 1e-6 relative (the clamp floor rounds)
            assert math.isclose(float(got_pos), want_pos, rel_tol=1e-5)
            assert math.isclose(float(got_neg), want_neg, rel_tol=1e-5)
            assert torch.is_tensor(got_pos) and torch.is_tensor(got_neg)

    def test_counts_round_trip_through_dn_branch(self):
        # the criterion caches (num_pos, num_neg) tensors and passes them
        # back as `counts` for the dn pass — the tensor tuple must work
        student = self._stacked(torch.randn(1, 4, 4, REG_MAX + 1))
        teacher = self._stacked(torch.randn(1, 4, 4, REG_MAX + 1))
        boxes = torch.rand(1, 4, 4) * 0.3 + 0.3
        logits = torch.rand(1, 4, 2)
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        _, num_pos, num_neg = ddf_loss(
            student, teacher, logits, logits, boxes, boxes, indices, REG_MAX
        )
        loss_with_counts, _, _ = ddf_loss(
            student,
            teacher,
            logits,
            logits,
            boxes,
            boxes,
            indices,
            REG_MAX,
            counts=(num_pos, num_neg),
        )
        recomputed, _, _ = ddf_loss(
            student, teacher, logits, logits, boxes, boxes, indices, REG_MAX
        )
        assert torch.allclose(loss_with_counts, recomputed, atol=1e-6)


# ── _pair_two_crop packed transfer ──────────────────────────────────────


class TestPairTwoCropPacking:
    def test_pairs_and_unpaired_unchanged(self):
        from autovision.tasks.detection.shared.ssl_contrastive import (
            _pair_two_crop,
        )

        # 2 images per view; view-1 rows 0/1, view-2 rows 2/3.
        # Image 0: one instance perfectly aligned across views.
        # Image 1: view-1 instance with NO view-2 counterpart (unpaired).
        batch_sel = torch.tensor([0, 1, 2, 2])
        src_sel = torch.tensor([3, 5, 3, 7])
        labels_all = torch.tensor([1, 0, 1, 0])
        boxes_xyxy = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],  # img0 view1
                [0.6, 0.6, 0.9, 0.9],  # img1 view1 (unpaired)
                [0.1, 0.1, 0.4, 0.4],  # img0 view2 (row 2 = 0 + images_per_view)
                [0.1, 0.1, 0.2, 0.2],  # img1 view2: no IoU with img1 view1
            ]
        )
        paired = _pair_two_crop(
            batch_sel,
            src_sel,
            labels_all,
            boxes_xyxy,
            images_per_view=2,
            iou_match=0.55,
            flip_invariant=False,
        )
        assert paired is not None
        rows1, srcs1, rows2, srcs2, ious, unpaired = paired
        # image 0 pair: pool positions 0 (view1) and 2 (view2)
        assert rows1.tolist() == [0]
        assert rows2.tolist() == [2]
        assert srcs1.tolist() == [3]
        assert srcs2.tolist() == [3]
        assert ious.numel() == 1 and ious[0] > 0.55
        # image 1's view-1 instance (pool position 1) is unpaired
        assert unpaired.tolist() == [1] or set(unpaired.tolist()) == {1, 3}


# ── translate_gt where-rewrite ──────────────────────────────────────────


class TestTranslateGtWhereRewrite:
    @staticmethod
    def _reference_translate_gt(gt, reg_max, reg_scale, up):
        """Pre-refactor body (boolean-mask writes)."""
        from autovision.tasks.detection.shared.dfine_utils import (
            weighting_function,
        )

        gt = gt.reshape(-1)
        function_values = weighting_function(reg_max, up, reg_scale).to(gt.device)

        diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
        mask = diffs <= 0
        closest_left_indices = torch.sum(mask, dim=1) - 1

        indices = closest_left_indices.float()
        weight_right = torch.zeros_like(indices)
        weight_left = torch.zeros_like(indices)

        valid_idx_mask = (indices >= 0) & (indices < reg_max)
        valid_indices = indices[valid_idx_mask].long()
        left_values = function_values[valid_indices]
        right_values = function_values[valid_indices + 1]
        left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
        right_diffs = torch.abs(right_values - gt[valid_idx_mask])
        weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
        weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

        invalid_idx_mask_neg = indices < 0
        weight_right[invalid_idx_mask_neg] = 0.0
        weight_left[invalid_idx_mask_neg] = 1.0
        indices[invalid_idx_mask_neg] = 0.0

        invalid_idx_mask_pos = indices >= reg_max
        weight_right[invalid_idx_mask_pos] = 1.0
        weight_left[invalid_idx_mask_pos] = 0.0
        indices[invalid_idx_mask_pos] = reg_max - 0.1
        return indices, weight_right, weight_left

    def test_value_identical_on_all_lane_classes(self):
        from autovision.tasks.detection.shared.dfine_utils import (
            translate_gt,
        )

        generator = torch.Generator().manual_seed(83)
        for reg_max, reg_scale in ((8, 4.0), (32, 4.0), (16, 2.0)):
            # spread across: below W[0], inside the span, above the top,
            # and EXACTLY on knots (historical NaN path)
            from autovision.tasks.detection.shared.dfine_utils import (
                weighting_function,
            )

            w = weighting_function(reg_max, 0.5, reg_scale)
            values = torch.cat(
                [
                    w - 0.37,
                    w,
                    w + 0.29,
                    torch.tensor([-1e3, 1e3, 0.0], dtype=torch.float32),
                ]
            )
            values = values[torch.randperm(values.numel(), generator=generator)]
            got = translate_gt(values, reg_max, reg_scale, 0.5)
            want = self._reference_translate_gt(values, reg_max, reg_scale, 0.5)
            for g, w_ in zip(got, want):
                # NaNs (exact-knot lanes) must appear in BOTH or neither
                assert torch.equal(torch.isnan(g), torch.isnan(w_))
                assert torch.equal(g, w_)

    def test_cpu_values_sane(self):
        from autovision.tasks.detection.shared.dfine_utils import (
            translate_gt,
        )

        values = torch.linspace(-5, 5, 97)
        indices, weight_right, weight_left = translate_gt(values, 32, 4.0, 0.5)
        # every lane's weights sum to 1 (NaN-knot lanes excepted)
        total = weight_right + weight_left
        finite = ~torch.isnan(total)
        assert torch.allclose(total[finite], torch.ones_like(total[finite]))
        assert bool(((indices >= 0) & (indices <= 31.9)).all())


# ── MWAS batched window scatter ─────────────────────────────────────────


class TestMwasWindowScatter:
    def test_matches_per_window_slice_add(self):
        import torch.nn.functional as F

        from autovision.tasks.detection.shared.mwas import (
            WindowProcessor,
            window_grid_size,
        )

        torch.manual_seed(97)
        batch, channels, height, width = 3, 32, 40, 48
        window_size = 8
        n_h, n_w = window_grid_size(height, width, window_size)
        processor = WindowProcessor(embed_dim=channels, dim_feedforward=64)
        processor.eval()  # dropout off — deterministic encoder

        memory = torch.randn(batch, channels, height, width)
        # sparse-but-realistic masks, incl. one all-false (defensive) row
        window_mask = torch.rand(batch, n_h, n_w) < 0.3
        window_mask[2] = False

        glob_pos_embed = torch.randn(channels, height, width)
        with torch.no_grad():
            got = processor(memory, window_mask, window_size, glob_pos_embed)

            # reference: the pre-refactor per-window slice-add loop, re-run
            # through the module's own encoder weights
            pad_h = n_h * window_size - height
            pad_w = n_w * window_size - width
            padded = F.pad(memory, (0, pad_w, 0, pad_h))
            enhanced = padded.clone()
            windows = padded.view(
                batch, channels, n_h, window_size, n_w, window_size
            ).permute(0, 2, 4, 1, 3, 5)
            glob_padded = F.pad(glob_pos_embed, (0, pad_w, 0, pad_h))
            glob_grid = glob_padded.view(
                channels, n_h, window_size, n_w, window_size
            ).permute(1, 3, 0, 2, 4)
            rel_pos = processor._relative_embedding(window_size, memory.device).to(
                memory.dtype
            )
            seq_len = window_size * window_size
            rel_pos_flat = rel_pos.reshape(channels, seq_len).permute(1, 0).unsqueeze(0)
            nonempty = window_mask.flatten(1).any(dim=1).tolist()
            for b in range(batch):
                if not nonempty[b]:
                    continue
                selected = window_mask[b]
                window_feats = windows[b][selected]
                window_pos = glob_grid[selected].to(memory.dtype)
                num_win = window_feats.shape[0]
                feats_flat = window_feats.reshape(
                    num_win, channels, seq_len
                ).permute(0, 2, 1)
                pos_flat = window_pos.reshape(num_win, channels, seq_len).permute(
                    0, 2, 1
                )
                encoded = processor.window_encoder(
                    feats_flat, pos_embed=rel_pos_flat, glob_pos_embeds=pos_flat
                )
                encoded = encoded.permute(0, 2, 1).reshape(
                    num_win, channels, window_size, window_size
                )
                ys, xs = selected.nonzero(as_tuple=True)
                for k in range(num_win):
                    y0 = int(ys[k]) * window_size
                    x0 = int(xs[k]) * window_size
                    enhanced[
                        b, :, y0 : y0 + window_size, x0 : x0 + window_size
                    ] += encoded[k]
            want = enhanced[:, :, :height, :width]

        assert torch.equal(got, want)


# ── DN group max_gt_cap ─────────────────────────────────────────────────


class TestDenoisingMaxGtCap:
    """The crash guard: capping per-image DN slots must leave sub-cap
    batches untouched and keep dn_meta consistent with the targets on
    capped ones."""

    @staticmethod
    def _build(num_gt_per_image, max_gt_cap=None):
        from autovision.tasks.detection.shared.denoising import (
            get_contrastive_denoising_training_group,
        )

        targets = [
            {
                "boxes": torch.rand(n, 4) * 0.5 + 0.1,
                "labels": torch.randint(0, 5, (n,)),
            }
            for n in num_gt_per_image
        ]
        class_embed = torch.nn.Embedding(6, 16)
        return get_contrastive_denoising_training_group(
            targets,
            num_classes=5,
            num_queries=10,
            class_embed=class_embed,
            num_denoising=100,
            max_gt_cap=max_gt_cap,
        )

    def test_uncapped_batches_identical_to_capped_below_cap(self):
        counts = [7, 3, 50]  # max 50 < cap
        torch.manual_seed(11)
        uncapped = self._build(counts)
        torch.manual_seed(11)  # same label/box noise draws
        capped = self._build(counts, max_gt_cap=300)
        for a, b in zip(uncapped, capped):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
        assert uncapped[3]["dn_num_split"] == capped[3]["dn_num_split"]

    def test_cap_bounds_query_count_on_extreme_batches(self):
        counts = [900, 4, 4]
        dn_content, dn_boxes, attn_mask, dn_meta = self._build(counts, max_gt_cap=300)
        group_size = dn_meta["dn_group_size"]
        assert group_size == 300
        num_dn = dn_meta["dn_num_split"][0]
        assert num_dn == 300 * 2 * dn_meta["dn_num_group"]
        assert dn_content.shape[1] == num_dn
        assert attn_mask.shape[0] == num_dn + 10
        # matched indices stay consistent with the truncated targets
        from autovision.tasks.detection.shared.denoising import (
            get_cdn_matched_indices,
        )

        targets = [
            {
                "boxes": torch.rand(n, 4) * 0.5 + 0.1,
                "labels": torch.randint(0, 5, (n,)),
            }
            for n in counts
        ]
        indices = get_cdn_matched_indices(dn_meta, targets)
        for (src, dst), n in zip(indices, counts):
            assert dst.numel() == min(n, 300) * dn_meta["dn_num_group"]
            assert int(dst.max()) < n  # target indices stay in range

    def test_group_count_floors_to_one_with_or_without_cap(self):
        counts = [900, 4]
        _, _, _, meta = self._build(counts)
        _, _, _, meta_capped = self._build(counts, max_gt_cap=300)
        assert meta["dn_num_group"] == 1
        assert meta_capped["dn_num_group"] == 1


# ── dataloader worker hygiene (ultralytics port, item 4) ────────────────


class _ThreadProbeDataset(torch.utils.data.Dataset):
    """One item per index; carries the worker's torch thread count."""

    def __init__(self, length: int = 4) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(torch.get_num_threads(), dtype=torch.int64)


class TestDataloaderWorkerHygiene:
    def test_worker_cap_by_batch_count(self):
        from autovision.cli.common_helpers import build_task_dataloader

        # 2 batches with 8 requested workers -> capped to the batch count.
        loader = build_task_dataloader(
            _ThreadProbeDataset(4), batch_size=2, num_workers=8, train=True
        )
        assert loader.num_workers == 2
        # A single-batch loader runs in-process (no persistent pool).
        loader = build_task_dataloader(
            _ThreadProbeDataset(2), batch_size=8, num_workers=8, train=True
        )
        assert loader.num_workers == 0
        assert loader.prefetch_factor is None


# ── clip disabled path: foreach norm (ultralytics port, item 2) ─────────


class TestClipDisabledPathForeach:
    def _stub(self, max_norm: float):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_args=SimpleNamespace(gradient_clip_val=max_norm)
        )

    def _optimizer(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        params = [
            torch.nn.Parameter(torch.randn(7, 5, generator=generator)),
            torch.nn.Parameter(torch.randn(11, generator=generator)),
        ]
        for p in params:
            p.grad = torch.randn_like(p, generator=generator)
        return torch.optim.SGD(params, lr=0.1)

    def test_disabled_path_matches_reference_and_no_mutation(self):
        from autovision.tasks.detection.rtdetr.train_model import (
            RTDETRObjectDetectionTrain,
        )

        optimizer = self._optimizer()
        before = [p.grad.clone() for g in optimizer.param_groups for p in g["params"]]
        got = RTDETRObjectDetectionTrain.clip_gradients(
            self._stub(0.0), optimizer
        )
        # Reference: per-tensor norms then the outer norm (pre-foreach code).
        grads = [p.grad for g in optimizer.param_groups for p in g["params"]]
        reference = float(
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in grads])
            )
        )
        assert got == pytest.approx(reference, rel=1e-6)
        after = [p.grad for g in optimizer.param_groups for p in g["params"]]
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_enabled_path_smoke(self):
        from autovision.tasks.detection.rtdetr.train_model import (
            RTDETRObjectDetectionTrain,
        )

        optimizer = self._optimizer(seed=3)
        total = RTDETRObjectDetectionTrain.clip_gradients(self._stub(0.05), optimizer)
        assert math.isfinite(total)
        grads = [p.grad for g in optimizer.param_groups for p in g["params"]]
        assert float(
            torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
        ) <= 0.05 + 1e-5


# ── nominal-batch auto accumulation (ultralytics port, item 3) ──────────


class TestGradientAccumulationResolution:
    def _resolve(self, accum, nominal, batch):
        from autovision.cli.common_helpers import resolve_gradient_accumulation

        return resolve_gradient_accumulation(accum, nominal, batch)

    def test_auto_from_nominal_batch(self):
        assert self._resolve("auto", 64, 16) == 4
        assert self._resolve("auto", 64, 32) == 2
        # Larger per-step batch than nominal -> no accumulation.
        assert self._resolve("auto", 64, 128) == 1
        # round(0.5) = 0 would disable training steps entirely.
        assert self._resolve("auto", 64, 192) == 1

    def test_explicit_passthrough_and_validation(self):
        assert self._resolve(3, 64, 16) == 3
        with pytest.raises(ValueError, match=">= 1"):
            self._resolve(0, 64, 16)
        with pytest.raises(ValueError, match="auto"):
            self._resolve("sometimes", 64, 16)

    @pytest.mark.skipif(
        importlib.util.find_spec("timm") is None,
        reason="detr/n+s backbone needs the timm extra (pip install -e '.[timm]')",
    )
    def test_auto_accumulation_end_to_end(self, tmp_path):
        """accum=round(8/2)=4 over the 8-image coco8 smoke set."""
        from autovision import train_object_detection
        from autovision.data.coco8 import ensure_coco8

        data = ensure_coco8(tmp_path / "coco8")
        result = train_object_detection(
            out=tmp_path / "run",
            data=data,
            model="detr/n",
            steps=2,
            batch_size=2,
            nominal_batch_size=8,
            gradient_accumulation_steps="auto",
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
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


class TestMatcherL1CostEquivalence:
    def test_elementwise_l1_matches_cdist(self):
        """The matcher's L1 cost uses elementwise abs-sum (cdist is 6-12x
        slower at these shapes); values must agree with cdist up to fp32
        rounding on shapes the matcher actually sees."""
        generator = torch.Generator().manual_seed(29)
        pred = torch.rand(6, 60, 4, generator=generator)
        tgt = torch.rand(6, 21, 4, generator=generator)
        elementwise = (pred[:, :, None, :] - tgt[:, None, :, :]).abs().sum(-1)
        cdist = torch.cdist(pred, tgt, p=1)
        assert torch.allclose(elementwise, cdist, atol=1e-5)
        assert torch.equal(elementwise, cdist) or (
            elementwise - cdist
        ).abs().max() < 1e-5


# ── consolidated matcher: many-groups solve + optional GPU LSAP ─────────


class TestConsolidatedMatching:
    def _groups(self, seed: int, num_groups: int = 5, batch: int = 5, queries: int = 40):
        generator = torch.Generator().manual_seed(seed)
        groups = []
        for _ in range(num_groups):
            targets = []
            for _ in range(batch):
                count = int(torch.randint(0, 7, (1,), generator=generator))
                targets.append(
                    {
                        "boxes": torch.rand(count, 4, generator=generator) * 0.4 + 0.1,
                        "labels": torch.randint(0, 6, (count,), generator=generator),
                    }
                )
            outputs = {
                "pred_logits": torch.randn(batch, queries, 6, generator=generator),
                "pred_boxes": torch.rand(batch, queries, 4, generator=generator),
            }
            groups.append((outputs, targets))
        return groups

    def _force_scipy(self, monkeypatch):
        import autovision.tasks.detection.shared.matcher as matcher_module

        monkeypatch.setenv("AUTOVISION_LSAP", "scipy")
        matcher_module._LSAP["tried"] = False
        matcher_module._LSAP["fn"] = None

    def _restore_auto(self, monkeypatch):
        import autovision.tasks.detection.shared.matcher as matcher_module

        monkeypatch.setenv("AUTOVISION_LSAP", "auto")
        matcher_module._LSAP["tried"] = False
        matcher_module._LSAP["fn"] = None

    def test_many_matches_sequential_scipy(self, monkeypatch):
        from autovision.tasks.detection.shared.matcher import (
            forward_match,
            forward_match_many,
        )

        self._force_scipy(monkeypatch)
        matcher = HungarianMatcher()
        for seed in (3, 11, 29):
            groups = self._groups(seed)
            got = forward_match_many(matcher, groups)
            for (outputs, targets), group_result in zip(groups, got):
                want = forward_match(matcher, outputs, targets)
                for (g_src, g_dst), (w_src, w_dst) in zip(group_result, want):
                    assert torch.equal(g_src, w_src) and torch.equal(g_dst, w_dst)

    def test_gpu_path_matches_scipy(self, monkeypatch):
        pytest.importorskip("torch_linear_assignment")
        if not torch.cuda.is_available():
            pytest.skip("GPU LSAP needs CUDA")
        from autovision.tasks.detection.shared.matcher import forward_match_many

        matcher = HungarianMatcher()
        for seed in (5, 17):
            groups = self._groups(seed)
            self._force_scipy(monkeypatch)
            reference = forward_match_many(matcher, groups)
            self._restore_auto(monkeypatch)
            gpu = forward_match_many(matcher, groups)
            for ref_group, gpu_group in zip(reference, gpu):
                for (r_src, r_dst), (g_src, g_dst) in zip(ref_group, gpu_group):
                    assert torch.equal(r_src, g_src.cpu())
                    assert torch.equal(r_dst, g_dst.cpu())

    def test_kill_switch_forces_scipy(self, monkeypatch):
        import autovision.tasks.detection.shared.matcher as matcher_module

        self._force_scipy(monkeypatch)
        assert matcher_module._load_lsap() is None

    def test_more_targets_than_queries_falls_back(self, monkeypatch):
        """Batched GPU kernels require full column coverage; >Q targets
        must stay on the scipy path and still solve exactly."""
        from autovision.tasks.detection.shared.matcher import forward_match_many

        generator = torch.Generator().manual_seed(2)
        outputs = {
            "pred_logits": torch.randn(2, 8, 3, generator=generator),
            "pred_boxes": torch.rand(2, 8, 4, generator=generator),
        }
        targets = [
            {
                "boxes": torch.rand(13, 4, generator=generator) * 0.4 + 0.1,
                "labels": torch.randint(0, 3, (13,), generator=generator),
            }
            for _ in range(2)
        ]
        matcher = HungarianMatcher()
        self._force_scipy(monkeypatch)
        reference = forward_match_many(matcher, [(outputs, targets)])
        self._restore_auto(monkeypatch)
        got = forward_match_many(matcher, [(outputs, targets)])
        for (r_src, r_dst), (g_src, g_dst) in zip(reference[0], got[0]):
            assert torch.equal(r_src, g_src.cpu())
            assert torch.equal(r_dst, g_dst.cpu())

    def test_all_empty_group_among_solved_groups(self, monkeypatch):
        from autovision.tasks.detection.shared.matcher import forward_match_many

        self._force_scipy(monkeypatch)
        groups = self._groups(7, num_groups=3)
        empty_targets = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(groups[0][0]["pred_logits"].shape[0])
        ]
        groups.append((groups[0][0], empty_targets))
        got = forward_match_many(HungarianMatcher(), groups)
        assert got[0]  # non-empty groups still solved
        assert all(src.numel() == 0 and dst.numel() == 0 for src, dst in got[-1])
