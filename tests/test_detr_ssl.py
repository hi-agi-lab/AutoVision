# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the per-object contrastive SSL branch (design doc 14 §7).

All cases run offline against the tiny ``dinov2-vittest-notpretrained``
variant (hermetic, CPU-friendly), mirroring the conventions of
``tests/test_detr_dome.py``.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torchvision.transforms.v2.functional as TF
from torchvision import tv_tensors
from torchvision.ops import box_iou

from autovision.augment.detection import (
    RTDETRDetectionCollateFunction,
    RTDETRDetectionTransformArgs,
    TwoViewDetectionCollateFunction,
)
from autovision.config.validate import pydantic_model_validate
from autovision.data.two_view import TwoViewObjectDetectionDataset
from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
    DETRSSLArgs,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLParams,
    SSLProjectionHead,
    cosine_logits,
    gather_distributed,
    gather_distributed_padded,
    match_cross_view,
    positive_logit_bias,
    simcse_infonce_loss,
    ssl_contrastive_loss,
    supervised_contrastive_loss,
)

VARIANT_ARGS = dict(
    num_classes=2,
    backbone="dinov2-vittest-notpretrained",
    image_size=(224, 224),
    num_queries=20,
    num_top_queries=20,
    decoder="dfine",
)


# ── fixtures ────────────────────────────────────────────────────────────
def make_ssl_train_model(**ssl_overrides) -> DETRObjectDetectionTrain:
    ssl = dict(mode="unsup", view="two_crop", loss_coef=0.1)
    ssl.update(ssl_overrides)
    args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS, ssl=ssl)
    return DETRObjectDetectionTrain(args)


def make_baseline_train_model() -> DETRObjectDetectionTrain:
    args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS)
    return DETRObjectDetectionTrain(args)


@pytest.fixture(scope="module")
def ssl_train_model() -> DETRObjectDetectionTrain:
    return make_ssl_train_model()


@pytest.fixture(scope="module")
def baseline_train_model() -> DETRObjectDetectionTrain:
    return make_baseline_train_model()


def make_two_crop_batch(pairs: int = 2, matching: bool = True, seed: int = 0) -> dict:
    """A view-major doubled batch (rows ``r`` and ``r + B`` share the source
    image — the collate's contract); with ``matching`` the two views of
    every image carry identical GT boxes (cross-view IoU 1 → all pairs
    survive)."""
    generator = torch.Generator().manual_seed(seed)
    images = [[], []]
    boxes = [[], []]
    classes = [[], []]
    for _ in range(pairs):
        n = int(torch.randint(1, 4, (1,), generator=generator))
        b = torch.rand(n, 4, generator=generator) * 0.3 + 0.3
        c = torch.randint(0, 2, (n,), generator=generator)
        for view in (0, 1):
            if view == 0 or matching:
                view_boxes, view_classes = b.clone(), c.clone()
            else:
                view_boxes = torch.rand(n, 4, generator=generator) * 0.3 + 0.3
                view_classes = torch.randint(0, 2, (n,), generator=generator)
            images[view].append(torch.randn(3, 224, 224, generator=generator))
            boxes[view].append(view_boxes)
            classes[view].append(view_classes)
    return {
        "image": torch.stack(images[0] + images[1]),
        "bboxes": boxes[0] + boxes[1],
        "classes": classes[0] + classes[1],
    }


# ── projection head ─────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["simcse", "mlp2"])
def test_projection_head_contract(kind):
    head = SSLProjectionHead(hidden_dim=32, projection_dim=16, kind=kind, dropout=0.5)
    features = torch.randn(7, 32)
    head.train()
    out = head(features)
    assert out.shape == (7, 16)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(7))
    # Two draws differ under dropout (the dropout view's noise source).
    assert not torch.equal(head(features), head(features))
    head.eval()
    torch.testing.assert_close(head(features), head(features))
    # bf16 autocast keeps the output float32 (AMP-stable cosine).
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert head(features).dtype == torch.float32


def test_projection_head_rejects_bad_args():
    with pytest.raises(ValueError, match="kind"):
        SSLProjectionHead(32, 16, kind="bogus")
    with pytest.raises(ValueError, match="dropout"):
        SSLProjectionHead(32, 16, dropout=1.0)


# ── distributed gather (single-process contract) ───────────────────────
def test_gather_distributed_passthrough_keeps_grad():
    tensor = torch.randn(3, 4, requires_grad=True)
    gathered = gather_distributed(tensor)
    assert gathered is tensor
    gathered.sum().backward()
    assert tensor.grad is not None


def test_gather_distributed_padded_passthrough_mask():
    tensor = torch.randn(3, 4)
    result = gather_distributed_padded(tensor)
    assert result.tensor is tensor
    assert result.valid.shape == (3,)
    assert bool(result.valid.all())


# ── losses ──────────────────────────────────────────────────────────────
def test_infonce_matches_hand_computation():
    generator = torch.Generator().manual_seed(0)
    z1 = torch.randn(5, 8, generator=generator)
    z2 = torch.randn(5, 8, generator=generator)
    loss = simcse_infonce_loss(z1, z2, temp=0.07)
    expected = torch.nn.functional.cross_entropy(
        cosine_logits(z1, z2, 0.07),
        torch.arange(5),
    )
    torch.testing.assert_close(loss, expected)


def test_infonce_single_instance_raises():
    with pytest.raises(ValueError, match="2"):
        simcse_infonce_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_infonce_valid_masks_exclude_pad_rows_and_columns():
    generator = torch.Generator().manual_seed(1)
    z1 = torch.randn(4, 6, generator=generator)
    z2 = torch.randn(4, 6, generator=generator)
    # Row/col 3 is a pad slot: anchors come from valid rows only and pad
    # columns leave the softmax denominator.
    valid = torch.tensor([True, True, True, False])
    loss = simcse_infonce_loss(z1, z2, temp=0.05, valid_rows=valid, valid_cols=valid)
    logits = cosine_logits(z1, z2, 0.05)
    logits[:, 3] = float("-inf")
    expected = torch.nn.functional.cross_entropy(logits[[0, 1, 2]], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(loss, expected)


def test_supcon_distinct_labels_equals_infonce():
    generator = torch.Generator().manual_seed(2)
    z1 = torch.randn(4, 8, generator=generator)
    z2 = torch.randn(4, 8, generator=generator)
    labels = torch.tensor([0, 1, 2, 3])
    torch.testing.assert_close(
        supervised_contrastive_loss(z1, z2, labels, labels.clone(), temp=0.05),
        simcse_infonce_loss(z1, z2, temp=0.05),
    )


def test_supcon_multi_positive_hand_computed():
    generator = torch.Generator().manual_seed(3)
    z1 = torch.randn(3, 8, generator=generator)
    z2 = torch.randn(3, 8, generator=generator)
    labels1 = torch.tensor([0, 1, 0])
    labels2 = torch.tensor([0, 0, 1])
    loss = supervised_contrastive_loss(z1, z2, labels1, labels2, temp=0.05)
    logits = cosine_logits(z1, z2, 0.05)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Per-anchor positives: label equality ∪ forced diagonal.
    # row 0 (label 0) → cols {0, 1}; row 1 (label 1) → {1 (forced), 2};
    # row 2 (label 0) → {0, 1, 2 (forced)}.
    masks = [
        torch.tensor([True, True, False]),
        torch.tensor([False, True, True]),
        torch.tensor([True, True, True]),
    ]
    expected = torch.stack(
        [
            -log_probs[i][mask].sum() / int(mask.sum())
            for i, mask in enumerate(masks)
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_supcon_same_image_requires_ids():
    labels = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="image_ids"):
        supervised_contrastive_loss(
            torch.randn(2, 4), torch.randn(2, 4), labels, labels.clone(),
            same_image_positive_only=True,
        )


def test_supcon_pad_columns_never_positives_for_real_class_zero():
    generator = torch.Generator().manual_seed(4)
    z1 = torch.randn(3, 8, generator=generator)
    z2 = torch.randn(3, 8, generator=generator)
    # Real class-0 anchor (row 0) and a pad row (row 2) whose label 0 must
    # NOT turn column 2 into a positive for the class-0 anchor.
    labels1 = torch.tensor([0, 1, 0])
    labels2 = torch.tensor([0, 1, 0])
    valid = torch.tensor([True, True, False])
    loss = supervised_contrastive_loss(
        z1, z2, labels1, labels2, temp=0.05, valid_rows=valid, valid_cols=valid
    )
    assert torch.isfinite(loss)
    # Manual: anchors {0, 1}; row 0 positives = valid columns with label 0
    # plus the diagonal → col {0} only (col 2 is pad).
    logits = cosine_logits(z1, z2, 0.05)
    logits[:, 2] = float("-inf")
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    expected = (-log_probs[0][[0]].sum() / 1 - log_probs[1][[1]].sum() / 1) / 2
    torch.testing.assert_close(loss, expected)


# ── IoU-aligned positive logit bias (A1 semantics) ──────────────────────
def test_positive_logit_bias_untouched_for_unmarked():
    logits = torch.zeros(3, 3)
    mask = torch.eye(3, dtype=torch.bool)
    biased = positive_logit_bias(logits, mask, iou=torch.full((3, 3), 0.5))
    off_diagonal = biased[~mask]
    assert torch.all(off_diagonal == 0)  # log(1) == 0


def test_positive_logit_bias_monotone_in_iou():
    logits = torch.zeros(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    iou = torch.tensor([[0.2, 0.5, 0.9]])
    biased = positive_logit_bias(logits, mask, iou=iou)
    assert biased[0, 0] < biased[0, 1] < biased[0, 2]


def test_positive_logit_bias_clamps_zero_iou():
    logits = torch.zeros(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    biased = positive_logit_bias(logits, mask, iou=torch.zeros(1, 2))
    assert torch.isfinite(biased).all()
    assert biased[0, 0] == pytest.approx(0.5 * math.log(1e-12))


def test_positive_logit_bias_none_iou_is_identity():
    logits = torch.randn(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    torch.testing.assert_close(
        positive_logit_bias(logits, mask, iou=None), logits
    )


# ── cross-view matching ─────────────────────────────────────────────────
def test_match_cross_view_identity_boxes():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
    labels = torch.tensor([0, 1])
    match = match_cross_view(boxes, boxes.clone(), labels, labels.clone(), iou_threshold=0.55)
    assert match.row_idx.tolist() == [0, 1]
    assert match.col_idx.tolist() == [0, 1]
    torch.testing.assert_close(match.iou, torch.ones(2))


def test_match_cross_view_optimal_under_permutation():
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
    boxes2 = boxes1.flip(0)
    labels = torch.tensor([0, 1])
    match = match_cross_view(boxes1, boxes2, labels, labels.flip(0), iou_threshold=0.55)
    assert sorted(zip(match.row_idx.tolist(), match.col_idx.tolist())) == [(0, 1), (1, 0)]


def test_match_cross_view_drops_low_iou_and_label_mismatch():
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
    # View 2: box 0 overlaps a little (IoU < 0.55); box 1 overlaps fully but
    # carries a different label.
    boxes2 = torch.tensor([[0.9, 0.9, 1.9, 1.9], [2.0, 2.0, 3.0, 3.0]])
    labels1 = torch.tensor([0, 1])
    labels2 = torch.tensor([0, 5])
    match = match_cross_view(boxes1, boxes2, labels1, labels2, iou_threshold=0.55)
    assert match.row_idx.numel() == 0


def test_match_cross_view_empty_inputs():
    empty = torch.zeros(0, 4)
    match = match_cross_view(empty, empty, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert match.row_idx.numel() == 0


# ── orchestration: ssl_contrastive_loss ─────────────────────────────────
def _pool_fixture(num_views: int = 1, matching: bool = False):
    """Hand-built match: image 0 → queries {1, 3}, image 1 → query {0}.

    With ``num_views=2`` the view-2 targets are fresh copies. When
    ``matching`` they duplicate the view-1 boxes exactly (cross-view IoU 1 →
    every pair survives); otherwise the boxes are mirrored into the RIGHT
    half of the frame — disjoint from every view-1 box (all in the LEFT
    half), so cross-view IoU is exactly 0.
    """
    def make_image0() -> dict:
        return {
            "boxes": torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.3, 0.3, 0.1, 0.1]]),
            "labels": torch.tensor([0, 1]),
        }

    def make_image1() -> dict:
        return {"boxes": torch.tensor([[0.25, 0.7, 0.1, 0.1]]), "labels": torch.tensor([1])}

    def make_view2(source: dict) -> dict:
        if matching:
            return {"boxes": source["boxes"].clone(), "labels": source["labels"].clone()}
        # Tiny boxes pinned to the far top-left corner: disjoint from every
        # view-1 box (cx 0.2–0.3, cy 0.3–0.7) under BOTH alignments, so
        # flip-invariant pairing cannot accidentally recover them.
        boxes = torch.full_like(source["boxes"], 0.08)
        boxes[:, 2:] = 0.04
        return {"boxes": boxes, "labels": source["labels"].clone()}

    targets = [make_image0(), make_image1()]
    if num_views == 2:
        targets += [make_view2(targets[0]), make_view2(targets[1])]
    indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
    ] * num_views
    return indices, targets


def test_ssl_loss_pool_respects_query_mask():
    indices, targets = _pool_fixture()
    z1 = torch.randn(2, 6, 16, requires_grad=True)
    z2 = torch.randn(2, 6, 16, requires_grad=True)
    params = SSLParams(mode="unsup", view="dropout")
    # Query 3 of image 0 is masked out (dynamic-query padding defence).
    query_mask = torch.ones(2, 6, dtype=torch.bool)
    query_mask[0, 3] = False
    stats: dict[str, float] = {}
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(2, 6, 3)}, targets, indices, params,
        query_mask=query_mask, stats_sink=stats,
    )
    assert stats["pool_size"] == 2.0  # 3 matched instances − 1 masked
    assert loss.requires_grad


def test_ssl_loss_single_instance_is_connected_zero():
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{"boxes": torch.rand(1, 4), "labels": torch.tensor([0])}]
    z1 = torch.randn(1, 4, 16, requires_grad=True)
    z2 = torch.randn(1, 4, 16, requires_grad=True)
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(1, 4, 3)}, targets, indices,
        SSLParams(mode="unsup", view="dropout"),
    )
    assert loss.item() == 0.0
    assert loss.requires_grad  # connected zero keeps DDP gradients alive
    loss.backward()
    assert z1.grad is not None


def test_ssl_loss_two_crop_pairs_identical_boxes():
    # Two images × two views with duplicated GT boxes → every matched
    # instance finds its cross-view counterpart (IoU 1).
    boxes0 = torch.rand(2, 4) * 0.3 + 0.3
    boxes1 = torch.rand(1, 4) * 0.3 + 0.3
    targets = [
        {"boxes": boxes0, "labels": torch.tensor([0, 1])},
        {"boxes": boxes1, "labels": torch.tensor([1])},
        {"boxes": boxes0.clone(), "labels": torch.tensor([0, 1])},
        {"boxes": boxes1.clone(), "labels": torch.tensor([1])},
    ]
    # Hand-picked matcher indices (the real matcher output shape — one
    # (query, GT) pair per matched instance, per image per view).
    indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    stats: dict[str, float] = {}
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
        SSLParams(mode="unsup", view="two_crop"), stats_sink=stats,
    )
    assert stats["pool_size"] == 6.0
    assert stats["paired_size"] == 3.0  # 2 + 1 validated cross-view pairs
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None


def test_ssl_loss_two_crop_all_pairs_dropped_is_connected_zero():
    indices, targets = _pool_fixture(num_views=2, matching=False)
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
        SSLParams(mode="unsup", view="two_crop"),
    )
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_ssl_loss_sup_and_both_modes():
    indices, targets = _pool_fixture(num_views=2, matching=True)
    torch.manual_seed(0)
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    outputs = {"pred_logits": torch.zeros(4, 6, 3)}
    sup = ssl_contrastive_loss(z1, z2, outputs, targets, indices, SSLParams(mode="sup", view="two_crop"))
    both = ssl_contrastive_loss(z1, z2, outputs, targets, indices, SSLParams(mode="both", view="two_crop"))
    unsup = ssl_contrastive_loss(z1, z2, outputs, targets, indices, SSLParams(mode="unsup", view="two_crop"))
    assert torch.isfinite(sup) and torch.isfinite(both) and torch.isfinite(unsup)


def test_ssl_both_mode_sup_leg_is_unbiased():
    """R11: in "both" mode the two objectives consume SEPARATE logit
    matrices — the IoU alignment bias stays on the InfoNCE leg (its diagonal
    positives), while the SupCon leg gets plain cosine logits so same-label
    soft positives on low-IoU pairs are not suppressed. With the bias zeroed
    the biased and unbiased matrices coincide, so the mode decomposition must
    be exact: both == unsup + sup."""
    indices, targets = _pool_fixture(num_views=2, matching=True)
    torch.manual_seed(0)
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    outputs = {"pred_logits": torch.zeros(4, 6, 3)}

    def run(mode: str, iou_bias: float):
        return ssl_contrastive_loss(
            z1, z2, outputs, targets, indices,
            SSLParams(mode=mode, view="two_crop", iou_bias=iou_bias),
        )

    # sup-ALONE keeps its biased-logits contract (documented behavior).
    assert not torch.equal(run("sup", 0.5), run("sup", 0.0))
    # With the bias zeroed, "both" decomposes exactly into its two legs.
    torch.testing.assert_close(
        run("both", 0.0), run("unsup", 0.0) + run("sup", 0.0)
    )
    # With the bias active the legs no longer share one matrix: "both" is
    # NOT sup-alone plus a re-run of the shared-bias InfoNCE.
    assert not torch.equal(run("both", 0.5), run("sup", 0.5) + run("unsup", 0.5))


def test_ssl_loss_sup_same_image_positive_only():
    indices, targets = _pool_fixture(num_views=2, matching=True)
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    params = SSLParams(
        mode="sup", view="two_crop", same_image_positive_only=True,
    )
    loss = ssl_contrastive_loss(z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices, params)
    assert torch.isfinite(loss)


# ── criterion wiring ────────────────────────────────────────────────────
def _criterion_outputs(batch: int = 2, queries: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    outputs = {
        "pred_logits": torch.randn(batch, queries, 3, generator=generator),
        "pred_boxes": torch.rand(batch, queries, 4, generator=generator) * 0.4 + 0.3,
    }
    targets = [
        {"boxes": torch.rand(2, 4, generator=generator) * 0.3 + 0.3, "labels": torch.tensor([0, 1])}
        for _ in range(batch)
    ]
    return outputs, targets


def test_criterion_weight_dict_carries_ssl_coef():
    criterion = DFINECriterionV2(HungarianMatcher(), ssl_params=SSLParams(loss_coef=0.3))
    assert criterion.weight_dict["loss_ssl"] == 0.3
    baseline = DFINECriterionV2(HungarianMatcher())
    assert "loss_ssl" not in baseline.weight_dict


def test_criterion_emits_loss_ssl_only_at_final_layer():
    criterion = DFINECriterionV2(HungarianMatcher(), ssl_params=SSLParams(mode="unsup", view="dropout"))
    criterion.train()
    outputs, targets = _criterion_outputs()
    outputs_list = [dict(outputs), dict(outputs)]  # aux + final
    z1 = torch.randn(2, 6, 16)
    z2 = torch.randn(2, 6, 16)
    losses = criterion(
        outputs_list, targets, ssl_z1=z1, ssl_z2=z2,
    )
    assert "loss_ssl" in losses
    assert not any(key.startswith("loss_ssl_aux") or key.startswith("loss_ssl_dn") for key in losses)
    # Replicate the criterion's own weight-duplication + filter aggregation.
    weights = dict(criterion.weight_dict)
    for output_index in range(len(outputs_list) - 1):
        for key, value in criterion.weight_dict.items():
            weights[f"{key}_aux_{output_index}"] = value
    total = sum(losses[key] * weight for key, weight in weights.items() if key in losses)
    torch.testing.assert_close(losses["loss"], total)


def test_criterion_ssl_absent_without_features_stays_silent_in_eval():
    """Eval mode: absent SSL features are the norm (validation never emits
    them) — no raise, no loss_ssl key. The training-mode mismatch raises
    (see test_criterion_missing_features_raise_loud_in_training)."""
    criterion = DFINECriterionV2(HungarianMatcher(), ssl_params=SSLParams())
    criterion.eval()
    outputs, targets = _criterion_outputs()
    losses = criterion([dict(outputs), dict(outputs)], targets)
    assert "loss_ssl" not in losses


def test_criterion_missing_features_raise_loud_in_training():
    """Head/criterion mismatch must fail loud (design doc 14 §7), not
    silently train a whole run without the SSL objective."""
    criterion = DFINECriterionV2(HungarianMatcher(), ssl_params=SSLParams())
    criterion.train()
    outputs, targets = _criterion_outputs()
    with pytest.raises(ValueError, match="ssl_z1"):
        criterion([dict(outputs), dict(outputs)], targets)


def test_ssl_loss_rejects_same_image_with_dropout_view():
    indices, targets = _pool_fixture()
    z1 = torch.randn(2, 6, 16, requires_grad=True)
    z2 = torch.randn(2, 6, 16, requires_grad=True)
    params = SSLParams(mode="sup", view="dropout", same_image_positive_only=True)
    with pytest.raises(ValueError, match="two_crop"):
        ssl_contrastive_loss(
            z1, z2, {"pred_logits": torch.zeros(2, 6, 3)}, targets, indices, params
        )


def test_degenerate_boxes_do_not_nan_the_bias():
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.0]])  # zero-area GT box
    labels = torch.tensor([0])
    match = match_cross_view(boxes, boxes.clone(), labels, labels.clone())
    assert match.iou.numel() == 0 or torch.isfinite(match.iou).all()
    logits = torch.zeros(1, 1)
    biased = positive_logit_bias(
        logits,
        positive_mask=torch.ones(1, 1, dtype=torch.bool),
        iou=box_iou(boxes, boxes).nan_to_num(0.0),
    )
    assert torch.isfinite(biased).all()


def test_criterion_eval_never_enters_ssl_path():
    criterion = DFINECriterionV2(HungarianMatcher(), ssl_params=SSLParams())
    criterion.eval()
    outputs, targets = _criterion_outputs()
    z1 = torch.randn(2, 6, 16)
    losses = criterion([dict(outputs)], targets, ssl_z1=z1, ssl_z2=z1)
    assert "loss_ssl" not in losses


def test_criterion_stride_gate():
    criterion = DFINECriterionV2(
        HungarianMatcher(), ssl_params=SSLParams(mode="unsup", view="dropout", every_n_steps=2)
    )
    criterion.train()
    outputs, targets = _criterion_outputs()
    z1 = torch.randn(2, 6, 16)
    z2 = torch.randn(2, 6, 16)
    criterion.ssl_step = 1
    assert "loss_ssl" not in criterion([dict(outputs)], targets, ssl_z1=z1, ssl_z2=z2)
    criterion.ssl_step = 2
    assert "loss_ssl" in criterion([dict(outputs)], targets, ssl_z1=z1, ssl_z2=z2)


# ── model-level integration ─────────────────────────────────────────────
def test_ssl_off_is_baseline(baseline_train_model, ssl_train_model):
    baseline_keys = set(baseline_train_model.state_dict().keys())
    ssl_keys = {
        key for key in ssl_train_model.state_dict().keys() if ".ssl_head." not in key
    }
    assert baseline_keys == ssl_keys
    assert getattr(baseline_train_model.model, "ssl_head", None) is None


def test_eval_forward_matches_baseline_bitwise():
    torch.manual_seed(7)
    baseline = make_baseline_train_model()
    torch.manual_seed(7)
    ssl_model = make_ssl_train_model()
    baseline.model.eval()
    ssl_model.model.eval()
    batch = make_two_crop_batch(pairs=1)
    x = batch["image"][:1]
    with torch.no_grad():
        out_baseline = baseline.model._forward_train(x)
        out_ssl = ssl_model.model._forward_train(x)
    # Bitwise: the head-last construction keeps the RNG stream identical,
    # so the eval graphs must agree element for element.
    assert torch.equal(out_baseline["pred_logits"], out_ssl["pred_logits"])
    assert torch.equal(out_baseline["pred_boxes"], out_ssl["pred_boxes"])
    assert "query_features" not in out_ssl
    assert "ssl_z1" not in out_ssl


def test_training_forward_emits_ssl_features(ssl_train_model):
    ssl_train_model.model.train()
    batch = make_two_crop_batch(pairs=2)
    out = ssl_train_model.model._forward_train(
        batch["image"], targets=[
            {"boxes": b, "labels": c} for b, c in zip(batch["bboxes"], batch["classes"])
        ]
    )
    assert out["query_features"].shape[0] == 4  # 2B view-major rows
    assert out["ssl_z1"].shape == out["ssl_z2"].shape
    assert out["ssl_z1"].requires_grad and out["ssl_z2"].requires_grad
    # Two independent dropout draws differ.
    assert not torch.equal(out["ssl_z1"], out["ssl_z2"])


def test_two_crop_training_step_pairs_and_backprops(ssl_train_model):
    batch = make_two_crop_batch(pairs=2, matching=True, seed=3)
    result = ssl_train_model.training_step(batch, step=0)
    assert "loss_ssl" in result.log_dict
    assert torch.isfinite(result.loss)
    # Duplicated GT boxes: every view-1 instance finds its cross-view
    # counterpart → paired ratio 1.0 (denominator = pool/2 for two_crop).
    stats = ssl_train_model.criterion.last_ssl_stats
    assert stats is not None
    assert stats["paired_size"] == stats["pool_size"] / 2
    assert result.log_dict["ssl_paired_ratio"] == pytest.approx(1.0)
    result.loss.backward()
    grads = [p.grad for p in ssl_train_model.model.ssl_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_ssl_group_in_param_groups():
    model = make_ssl_train_model()
    optimizer, _ = model.get_optimizer(total_steps=100, global_batch_size=4)
    names = [group.get("name") for group in optimizer.param_groups]
    assert "ssl" in names
    head_param_ids = {id(p) for p in model.model.ssl_head.parameters()}
    ssl_params = [
        p for group in optimizer.param_groups if group.get("name", "").startswith("ssl")
        for p in group["params"]
    ]
    assert {id(p) for p in ssl_params} == head_param_ids
    # No double assignment across groups.
    all_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(all_ids) == len(set(all_ids))


def test_checkpoint_load_tolerates_missing_ssl_head():
    ssl_model = make_ssl_train_model()
    baseline_state = {
        key: value
        for key, value in ssl_model.state_dict().items()
        if ".ssl_head." not in key
    }
    ssl_model.load_state_dict(baseline_state)  # strict — must not raise
    # Unexpected keys still raise loudly.
    with pytest.raises(RuntimeError):
        ssl_model.load_state_dict({**baseline_state, "model.bogus.weight": torch.zeros(1)})


def test_export_state_dict_keeps_ssl_head_for_roundtrip():
    model = make_ssl_train_model()
    export_state = model.get_export_state_dict()
    assert any(".ssl_head." in key for key in export_state)
    # The inference-facing model (built from the same init args) can load it.
    task_model = model.get_task_model()
    task_model.load_train_state_dict(export_state)


def test_ssl_off_dropout_view_integration():
    model = make_ssl_train_model(view="dropout", mode="unsup")
    batch = make_two_crop_batch(pairs=2)
    single_view = {
        "image": batch["image"][:2],
        "bboxes": batch["bboxes"][:2],
        "classes": batch["classes"][:2],
    }
    result = model.training_step(single_view, step=0)
    assert "loss_ssl" in result.log_dict
    assert model.criterion.last_ssl_stats["paired_size"] == model.criterion.last_ssl_stats["pool_size"]


def test_dome_and_ssl_combined():
    args = DETRObjectDetectionTrainArgs(
        **VARIANT_ARGS,
        density={"max_num_select": 50, "mwas_window_size": 5},
        ssl={"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},
    )
    model = DETRObjectDetectionTrain(args)
    batch = make_two_crop_batch(pairs=2, matching=True, seed=5)
    result = model.training_step(batch, step=0)
    assert torch.isfinite(result.loss)
    assert "loss_ssl" in result.log_dict
    assert "defe_density_loss" in result.log_dict


# ── config validation ───────────────────────────────────────────────────
def test_config_zero_coef_with_active_mode_raises():
    with pytest.raises(ValueError, match="loss_coef"):
        DETRSSLArgs(mode="unsup", loss_coef=0.0)


def test_config_zero_coef_with_off_mode_is_the_baseline():
    args = DETRSSLArgs(mode="off", loss_coef=0.0, view="two_crop")
    assert args.view == "two_crop"


def test_config_same_image_requires_two_crop():
    with pytest.raises(ValueError, match="two_crop"):
        DETRSSLArgs(mode="sup", view="dropout", same_image_positive_only=True)


def test_config_route_b_requires_unsup_and_dropout():
    """Route B (encoder_candidates) is label-free: unsup + dropout only."""
    args = DETRSSLArgs(
        mode="unsup", view="dropout", instance_source="encoder_candidates"
    )
    assert args.instance_source == "encoder_candidates"
    assert args.to_ssl_params().instance_source == "encoder_candidates"
    with pytest.raises(ValueError, match="unsup"):
        DETRSSLArgs(mode="sup", view="dropout", instance_source="encoder_candidates")
    with pytest.raises(ValueError, match="dropout"):
        DETRSSLArgs(mode="unsup", view="two_crop", instance_source="encoder_candidates")
    # The stride gate interleaves SSL with the detection losses — there is
    # nothing to interleave with in SSL-only mode, so reject a silent no-op.
    with pytest.raises(ValueError, match="every_n_steps"):
        DETRSSLArgs(
            mode="unsup", view="dropout", instance_source="encoder_candidates",
            every_n_steps=3,
        )


def test_config_unknown_key_rejected():
    with pytest.raises(Exception):
        pydantic_model_validate(DETRSSLArgs, {"mode": "unsup", "bogus": 1})


def test_task_model_rejects_ssl_with_rtdetrv2_decoder():
    from autovision.tasks.detection.detr.task_model import (
        DETRObjectDetection,
    )

    with pytest.raises(ValueError, match="D-FINE"):
        DETRObjectDetection(
            num_classes=2,
            model_name="dinov2-vittest-notpretrained",
            image_size=(224, 224),
            decoder="rtdetrv2",
            ssl={"mode": "unsup", "view": "two_crop"},
        )


# ── two-view data path ──────────────────────────────────────────────────
class _StubDetectionDataset(torch.utils.data.Dataset):
    """Minimal detection dataset with transform injection (task contract)."""

    def __init__(self, count: int = 4) -> None:
        self.transform = None
        self._items = [self._make_item(index) for index in range(count)]

    @staticmethod
    def _make_item(index: int) -> dict:
        image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[4.0, 4.0, 20.0, 20.0]]), format="XYXY", canvas_size=(32, 32)
        )
        return {
            "image_path": f"img_{index}.jpg",
            "image": image,
            "bboxes": boxes,
            "classes": torch.tensor([0]),
            "original_size": (32, 32),
        }

    def set_transform(self, transform) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict:
        item = self._items[index]
        if self.transform is not None:
            item = self.transform(item)
        return item


def _random_flip_transform(item: dict) -> dict:
    """Per-call random geometry: boxes must follow the image flip."""
    if torch.rand(()).item() < 0.5:
        return {**item, "image": TF.hflip(item["image"]), "bboxes": TF.hflip(item["bboxes"])}
    return item


def test_two_view_dataset_two_draws_and_transform_delegation():
    dataset = TwoViewObjectDetectionDataset(_StubDetectionDataset())
    assert len(dataset) == 4
    item1, item2 = dataset[0]
    assert item1["image_path"] == item2["image_path"] == "img_0.jpg"
    assert torch.equal(item1["image"], item2["image"])  # no transform → identical
    dataset.set_transform(_random_flip_transform)
    assert dataset.transform is not None
    flip_states = set()
    for _ in range(16):
        first, _second = dataset[0]
        # Horizontal-flip on a box whose x-center ≠ 16: two states exist.
        flip_states.add(tuple(first["bboxes"][0].tolist()))
    assert len(flip_states) == 2  # both flip states occur across draws


def test_two_view_dataset_boxes_follow_flip():
    dataset = TwoViewObjectDetectionDataset(_StubDetectionDataset())
    dataset.set_transform(_random_flip_transform)
    for _ in range(8):
        item, _second = dataset[0]
        boxes = item["bboxes"]
        # XYXY boxes stay inside the canvas regardless of the flip draw.
        assert boxes[:, 0].min() >= 0 and boxes[:, 2].max() <= 32


def test_two_view_collate_order_and_contract():
    base = RTDETRDetectionCollateFunction(
        pydantic_model_validate(
            RTDETRDetectionTransformArgs,
            {"image_size": (64, 64), "scale_jitter": None, "mixup_prob": None},
        ),
        train=True,
    )
    collate = TwoViewDetectionCollateFunction(base)
    dataset = TwoViewObjectDetectionDataset(_StubDetectionDataset())
    items = [dataset[i] for i in range(2)]  # [(v1_0, v2_0), (v1_1, v2_1)]
    batch = collate(items)
    assert batch["image"].shape == (4, 3, 32, 32)
    # View-major: row r and row r + B are the same source image.
    assert batch["image_path"] == ["img_0.jpg", "img_1.jpg", "img_0.jpg", "img_1.jpg"]
    assert len(batch["bboxes"]) == len(batch["classes"]) == 4
    assert batch["input_size"] == (32, 32)


def test_two_view_collate_delegates_step_hooks():
    base = RTDETRDetectionCollateFunction(
        pydantic_model_validate(
            RTDETRDetectionTransformArgs,
            {"image_size": (64, 64), "scale_jitter": None, "mixup_prob": None},
        ),
        train=True,
    )
    collate = TwoViewDetectionCollateFunction(base)
    collate.set_step(3)
    assert collate.step == 3
    assert collate.requires_dataloader_reinitialization() is False


def test_two_view_collate_preserves_pairing_against_accumulation():
    """Each loader batch must contain complete (r, r+B) pairs — the engine
    treats one loader batch as one gradient-accumulation microbatch."""
    base = RTDETRDetectionCollateFunction(
        pydantic_model_validate(
            RTDETRDetectionTransformArgs,
            {"image_size": (64, 64), "scale_jitter": None, "mixup_prob": None},
        ),
        train=True,
    )
    collate = TwoViewDetectionCollateFunction(base)
    dataset = TwoViewObjectDetectionDataset(_StubDetectionDataset(count=3))
    for offset in range(0, 3, 2):
        items = [dataset[i] for i in range(offset, min(offset + 2, 3))]
        batch = collate(items)
        paths = batch["image_path"]
        count = len(items)
        for row in range(count):
            assert paths[row] == paths[row + count]


# ── distributed behaviour (real gloo groups) ────────────────────────────
def test_gather_distributed_single_rank_gloo_keeps_grad():
    import datetime

    store = dist.HashStore()
    dist.init_process_group(
        backend="gloo", store=store, rank=0, world_size=1,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        tensor = torch.randn(3, 4, requires_grad=True)
        gathered = gather_distributed(tensor)
        assert gathered.shape == (3, 4)
        assert gathered.grad_fn is not None  # slot replacement keeps autograd
        gathered.sum().backward()
        assert tensor.grad is not None
        assert bool((tensor.grad == 1).all())
        # Padded gather equal-size fast path.
        padded = gather_distributed_padded(torch.randn(2, 4, requires_grad=True))
        assert padded.valid.shape == (2,)
        assert bool(padded.valid.all())
    finally:
        dist.destroy_process_group()


def _ssl_ddp_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    import datetime

    from autovision.tasks.detection.shared.ssl_contrastive import (
        SSLParams as WorkerSSLParams,
    )
    from autovision.tasks.detection.shared.ssl_contrastive import (
        ssl_contrastive_loss as worker_ssl_contrastive_loss,
    )

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        torch.manual_seed(100 + rank)
        box0 = torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.3, 0.3, 0.1, 0.1]])
        box1 = torch.tensor([[0.25, 0.7, 0.1, 0.1]])
        labels0 = torch.tensor([0, 1])
        labels1 = torch.tensor([1])
        if rank == 0:
            # Unbalanced on purpose: 6 matched instances here vs 4 on rank 1.
            indices = [
                (torch.tensor([1, 3]), torch.tensor([0, 1])),
                (torch.tensor([0]), torch.tensor([0])),
            ] * 2
            targets = [
                {"boxes": box0, "labels": labels0},
                {"boxes": box1, "labels": labels1},
            ] * 2
        else:
            # Two image pairs, one matched instance per row — pool 4 (vs 6
            # on rank 0), 2 validated pairs.
            indices = [
                (torch.tensor([2]), torch.tensor([0])),
                (torch.tensor([2]), torch.tensor([0])),
                (torch.tensor([2]), torch.tensor([0])),
                (torch.tensor([2]), torch.tensor([0])),
            ]
            targets = [
                {"boxes": box1, "labels": labels1},
                {"boxes": box1, "labels": labels1},
                {"boxes": box1.clone(), "labels": labels1.clone()},
                {"boxes": box1.clone(), "labels": labels1.clone()},
            ]
        z1 = torch.randn(4, 6, 16, requires_grad=True)
        z2 = torch.randn(4, 6, 16, requires_grad=True)
        stats: dict[str, float] = {}
        loss = worker_ssl_contrastive_loss(
            z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
            WorkerSSLParams(mode="unsup", view="two_crop"), stats_sink=stats,
        )
        loss.backward()
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "ok": bool(torch.isfinite(loss).item()) and z1.grad is not None,
                    "stats": stats,
                    "loss": float(loss.detach()),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_ddp_unbalanced_pools_consensus_and_padded_gather(tmp_path):
    """2-rank gloo: different local pool sizes (6 vs 4) exercise the global
    enter/skip consensus, the padded gather and slot replacement — both
    ranks must finish with finite, backward-able losses (no deadlock)."""
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "ddp_store")
    mp.spawn(
        _ssl_ddp_worker,
        args=(2, init_file, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = json.loads((tmp_path / "rank0.json").read_text())
    rank1 = json.loads((tmp_path / "rank1.json").read_text())
    assert rank0["ok"] and rank1["ok"]
    assert rank0["stats"]["pool_size"] == 6.0 and rank1["stats"]["pool_size"] == 4.0
    # paired_size is the PRE-gather local count (rank 0: 2+1 pairs, rank 1: 2).
    assert rank0["stats"]["paired_size"] == 3.0
    assert rank1["stats"]["paired_size"] == 2.0


def _ssl_extras_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """2-rank gloo with ASYMMETRIC D14 extras (rank 0: none, rank 1: two).

    Embeddings encode object identity in disjoint basis vectors, so the
    correctly aligned InfoNCE diagonal is the only high-logit column: the
    loss must be ~0. Under the pre-fix layout (extras folded into each
    rank's candidate block before the padded gather) the bias matrix shape
    breaks loudly or the diagonal points at a wrong/pad column, which shows
    up as a large loss.
    """
    import datetime

    from autovision.tasks.detection.shared.ssl_contrastive import (
        SSLParams as WorkerSSLParams,
    )
    from autovision.tasks.detection.shared.ssl_contrastive import (
        ssl_contrastive_loss as worker_ssl_contrastive_loss,
    )

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        q, d = 4, 32
        eye = torch.eye(4, d) * 10.0

        # Both draws carry BOTH batch rows (the loss pulls view-1 rows from
        # z1 and view-2 rows from z2); rows 0/1 are the view-1/view-2 image
        # of the single source image. Embeddings encode OBJECT identity in
        # disjoint basis vectors, so a correctly aligned diagonal is the only
        # high-logit column pair.
        def identity_batch(objects):
            # objects: [(query, object_id), ...] — identical content on both
            # batch rows (view-1 and view-2 image of the same source image).
            full = torch.zeros(2, q, d)
            for row in range(2):
                for query, object_id in objects:
                    full[row, query] = eye[object_id]
            return full

        # A (label 1) pairs across views; B (label 2) fails cross-view
        # validation (its view-2 box is mirrored into the far corner).
        box_a = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        box_b1 = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
        box_b2 = torch.tensor([[0.9, 0.9, 0.2, 0.2]])

        if rank == 0:
            z1 = identity_batch([(0, 0)])
            z2 = identity_batch([(0, 0)])
            targets = [
                {"boxes": box_a, "labels": torch.tensor([1])},
                {"boxes": box_a.clone(), "labels": torch.tensor([1])},
            ]
            indices = [
                (torch.tensor([0]), torch.tensor([0])),
                (torch.tensor([0]), torch.tensor([0])),
            ]
        else:
            z1 = identity_batch([(0, 1), (1, 2)])
            z2 = identity_batch([(0, 1), (1, 2)])
            targets = [
                {"boxes": torch.cat([box_a, box_b1]), "labels": torch.tensor([1, 2])},
                {"boxes": torch.cat([box_a, box_b2]), "labels": torch.tensor([1, 2])},
            ]
            indices = [
                (torch.tensor([0, 1]), torch.tensor([0, 1])),
                (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ]

        results = {}
        for use_extras in (True, False):
            params = WorkerSSLParams(
                mode="unsup",
                view="two_crop",
                temp=0.05,
                iou_bias=0.0,  # isolate the pairing geometry from the bias
                use_unpaired_as_negatives=use_extras,
            )
            stats: dict[str, float] = {}
            z1 = z1.detach().requires_grad_(True)
            z2 = z2.detach().requires_grad_(True)
            loss = worker_ssl_contrastive_loss(
                z1,
                z2,
                {"pred_logits": torch.zeros(2, q, 3)},
                targets,
                indices,
                params,
                stats_sink=stats,
            )
            loss.backward()
            results[f"extras={use_extras}"] = {
                "ok": bool(torch.isfinite(loss).item()) and z1.grad is not None,
                # Correct pairing: each anchor's ONLY high-logit column is its
                # own partner → loss ≈ log(1 + k·e^-20) ≈ 0. Any misaligned
                # diagonal lands on a zero logit → loss ≈ 19+.
                "loss": float(loss.detach()),
                **stats,
            }
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(results)
        )
    finally:
        dist.destroy_process_group()


def test_ddp_asymmetric_extras_keep_diagonal_pairing(tmp_path):
    """P0 regression (design doc 14 D14 × DDP): extras must be padded-gathered
    SEPARATELY from the paired candidates so the paired block stays
    block-aligned with the anchor rows. Pre-concatenating them per rank broke
    the arange diagonal as soon as any rank had extras (loud shape error in
    the bias matrix, or a silently wrong InfoNCE target)."""
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "extras_store")
    mp.spawn(
        _ssl_extras_worker,
        args=(2, init_file, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in (0, 1):
        results = json.loads((tmp_path / f"rank{rank}.json").read_text())
        with_extras = results["extras=True"]
        without = results["extras=False"]
        assert with_extras["ok"] and without["ok"]
        assert with_extras["loss"] < 0.5, with_extras
        assert without["loss"] < 0.5, without
    # Rank 1 owns the unpaired instances; its candidate column count grows
    # from 1 paired to 1 + 2 extras.
    rank1 = json.loads((tmp_path / "rank1.json").read_text())
    assert rank1["extras=True"]["candidate_size"] == 3.0
    assert rank1["extras=False"]["candidate_size"] == 1.0


# ── EMA round-trip ──────────────────────────────────────────────────────
def test_ema_checkpoint_roundtrip_covers_ssl_head():
    model = make_ssl_train_model()
    checkpoint: dict = {}
    model.on_save_checkpoint(checkpoint)
    ema_weights = checkpoint["ema"]["model"]
    # EMA state dict keys are bare (deepcopy of the task model).
    assert any(key.startswith("ssl_head.") for key in ema_weights)
    reference = model.ema_model.model.ssl_head.projector[0].weight

    fresh = make_ssl_train_model()
    fresh.on_load_checkpoint(checkpoint)
    torch.testing.assert_close(
        fresh.ema_model.model.ssl_head.projector[0].weight, reference
    )

    # A pre-SSL EMA checkpoint loads too: the head keeps its fresh init
    # (the fill provides current values for the strict loader).
    stripped = {
        key: value
        for key, value in ema_weights.items()
        if not (key.startswith("ssl_head.") or ".ssl_head." in key)
    }
    ema_meta = {key: value for key, value in checkpoint["ema"].items() if key != "model"}
    legacy = make_ssl_train_model()
    fresh_head = legacy.ema_model.model.ssl_head.projector[0].weight.clone()
    legacy.on_load_checkpoint({"ema": {"model": stripped, **ema_meta}})
    torch.testing.assert_close(
        legacy.ema_model.model.ssl_head.projector[0].weight, fresh_head
    )


# ── Route B: label-free SSL-only pretraining (design doc 14 §4.10) ──────
def make_route_b_train_model() -> DETRObjectDetectionTrain:
    args = DETRObjectDetectionTrainArgs(
        **VARIANT_ARGS,
        ssl={
            "mode": "unsup",
            "view": "dropout",
            "loss_coef": 1.0,
            "instance_source": "encoder_candidates",
        },
    )
    return DETRObjectDetectionTrain(args)


def test_route_b_component_pool_is_all_candidates():
    """The candidate pool is every (row, candidate) pair — no matching, no
    targets; the loss is plain InfoNCE and backprops into the features."""
    z1 = torch.randn(2, 5, 16, requires_grad=True)
    z2 = torch.randn(2, 5, 16, requires_grad=True)
    params = SSLParams(mode="unsup", view="dropout", instance_source="encoder_candidates")
    stats: dict[str, float] = {}
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(2, 5, 3)},
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}] * 2,
        [],  # no matcher output on the Route B path
        params, stats_sink=stats,
    )
    assert stats["pool_size"] == 10.0 and stats["paired_size"] == 10.0
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None and z1.grad.abs().sum() > 0


def test_route_b_component_rejects_sup_and_two_crop():
    z1 = torch.randn(2, 5, 16)
    z2 = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="unsup"):
        ssl_contrastive_loss(
            z1, z2, {"pred_logits": torch.zeros(2, 5, 3)}, [], [],
            SSLParams(mode="sup", view="dropout", instance_source="encoder_candidates"),
        )
    with pytest.raises(ValueError, match="dropout"):
        ssl_contrastive_loss(
            z1, z2, {"pred_logits": torch.zeros(2, 5, 3)}, [], [],
            SSLParams(mode="unsup", view="two_crop", instance_source="encoder_candidates"),
        )


def test_route_b_training_only_ssl_loss():
    """SSL-only: the loss set has NO detection terms, and gradients reach
    the encoder but not the randomly-initialized classification head."""
    model = make_route_b_train_model()
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "bboxes": [torch.rand(3, 4) * 0.3 + 0.3] * 2,
        "classes": [torch.tensor([0, 1, 0])] * 2,
    }
    result = model.training_step(batch, step=0)
    assert set(result.log_dict.keys()) == {"loss", "loss_ssl", "ssl_pool_size", "ssl_paired_ratio"}
    assert result.log_dict["ssl_pool_size"] == 40.0  # 2 rows × 20 candidates
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.model.encoder.parameters()
    )
    assert not any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.model.decoder.dec_score_head.parameters()
    )


def test_route_b_eval_emits_candidates_and_val_loss_runs():
    model = make_route_b_train_model()
    model.model.eval()
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "bboxes": [torch.rand(2, 4) * 0.3 + 0.3] * 2,
        "classes": [torch.tensor([0, 1])] * 2,
        "image_path": ["a.jpg", "b.jpg"],
        "original_size": [(224, 224), (224, 224)],
    }
    with torch.no_grad():
        raw = model.model._forward_train(batch["image"])
    assert "ssl_candidate_features" in raw
    assert raw["ssl_z1"].requires_grad is False  # no_grad eval, but PRESENT
    val_result = model.validation_step(batch, step=0)
    assert torch.isfinite(val_result.loss)
    assert set(val_result.log_dict.keys()) == {"loss", "loss_ssl"}


def test_route_b_watch_metric_switches():
    assert make_route_b_train_model().watch_metric == "val_ssl_loss"
    assert make_ssl_train_model().watch_metric == "val_metric/map"
    assert make_baseline_train_model().watch_metric == "val_metric/map"


def test_route_b_validation_keeps_head_dropout_active():
    """In eval mode the projection head's dropout is disabled and the two
    draws coincide exactly, degrading the held-out loss to a uniformity
    probe instead of the train objective's two-view estimate. The val step
    must run the forward with the head in train mode (restored right after)
    and the two stochastic draws must actually differ."""
    args = DETRObjectDetectionTrainArgs(
        **VARIANT_ARGS,
        ssl={
            "mode": "unsup",
            "view": "dropout",
            "loss_coef": 1.0,
            "instance_source": "encoder_candidates",
            "projector_dropout": 0.5,  # identical-draw probability 0.5^64 ≈ 0
        },
    )
    model = DETRObjectDetectionTrain(args)
    model.model.eval()
    val_model = model.ema_model.model if model.ema_model is not None else model.model
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "bboxes": [torch.rand(2, 4) * 0.3 + 0.3] * 2,
        "classes": [torch.tensor([0, 1])] * 2,
        "image_path": ["a.jpg", "b.jpg"],
        "original_size": [(224, 224), (224, 224)],
    }
    with torch.no_grad():
        first = model.validation_step(batch, step=0)
        second = model.validation_step(batch, step=0)
    assert math.isfinite(first.log_dict["loss_ssl"])
    assert not math.isclose(
        first.log_dict["loss_ssl"],
        second.log_dict["loss_ssl"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
    # The head is restored to eval after the forward — no state leaks into
    # EMA evaluation or export paths.
    assert not val_model.ssl_head.training


def test_route_b_stage2_load_drops_pretraining_head():
    """Stage 2 (supervised finetune without the head) loads a stage-1
    checkpoint: the pretraining-only ssl_head keys are dropped, everything
    else loads strictly."""
    stage1 = make_route_b_train_model()
    stage2 = make_baseline_train_model()
    stage2.load_state_dict(stage1.state_dict())  # strict — must not raise
    assert getattr(stage2.model, "ssl_head", None) is None
    # Non-SSL keys really did load (spot check the decoder).
    torch.testing.assert_close(
        stage2.model.decoder.proj,
        stage1.model.decoder.proj,
    )


def test_route_b_rejects_density():
    from autovision.tasks.detection.detr import (
        DETRObjectDetectionTrainArgs as Args,
    )

    with pytest.raises(ValueError, match="density"):
        DETRObjectDetectionTrain(
            Args(
                **VARIANT_ARGS,
                density={"max_num_select": 50, "mwas_window_size": 5},
                ssl={
                    "mode": "unsup",
                    "view": "dropout",
                    "instance_source": "encoder_candidates",
                },
            )
        )


# ── flip-invariant cross-view pairing (design doc 14 D13) ───────────────
def _mirror_xyxy_test(boxes):
    out = boxes.clone()
    out[:, 0] = 1.0 - boxes[:, 2]
    out[:, 2] = 1.0 - boxes[:, 0]
    return out


def _mirror_cxcywh_test(boxes):
    out = boxes.clone()
    out[:, 0] = 1.0 - boxes[:, 0]
    return out


def test_match_cross_view_recovers_mirrored_objects():
    """The two views flip independently: a mirrored object's raw normalized
    IoU is ~0, but it IS the same object — flip-invariant pairing recovers
    it; identity-only pairing drops it."""
    boxes1 = torch.tensor([[0.1, 0.4, 0.3, 0.6], [0.7, 0.1, 0.9, 0.3]])  # left + right
    boxes2 = _mirror_xyxy_test(boxes1)[[1, 0]]  # view 2 flipped horizontally, order swapped
    labels = torch.tensor([0, 1])
    labels2 = labels[[1, 0]]  # labels follow their boxes through the mirror+swap
    identity = match_cross_view(boxes1, boxes2, labels, labels2, iou_threshold=0.55)
    assert identity.row_idx.numel() == 0  # mirrored boxes never overlap raw
    flip = match_cross_view(
        boxes1, boxes2, labels, labels2, iou_threshold=0.55, flip_invariant=True
    )
    assert sorted(zip(flip.row_idx.tolist(), flip.col_idx.tolist())) == [(0, 1), (1, 0)]
    assert (flip.iou >= 0.55).all()


def test_two_crop_flip_recovery_in_loss():
    """End-to-end: mirrored view-2 boxes pair under the default config and
    are dropped when flip-invariant pairing is disabled."""
    boxes0 = torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.7, 0.3, 0.1, 0.1]])
    boxes1 = torch.tensor([[0.25, 0.7, 0.1, 0.1]])
    mirrored = torch.cat([_mirror_cxcywh_test(boxes0), _mirror_cxcywh_test(boxes1)])
    targets = [
        {"boxes": boxes0, "labels": torch.tensor([0, 1])},
        {"boxes": boxes1, "labels": torch.tensor([1])},
        {"boxes": mirrored[:2], "labels": torch.tensor([0, 1])},  # mirrored view 2
        {"boxes": mirrored[2:], "labels": torch.tensor([1])},
    ]
    indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    stats: dict[str, float] = {}
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
        SSLParams(mode="unsup", view="two_crop"),  # flip_invariant_pairing=True
        stats_sink=stats,
    )
    assert stats["paired_size"] == 3.0  # all mirrored pairs recovered
    assert torch.isfinite(loss)
    loss.backward()

    stats_off: dict[str, float] = {}
    loss_off = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
        SSLParams(mode="unsup", view="two_crop", flip_invariant_pairing=False),
        stats_sink=stats_off,
    )
    assert stats_off["paired_size"] == 0.0  # identity-only pairing drops them
    assert loss_off.item() == 0.0


# ── D14: unpaired instances as extra candidate columns ──────────────────
def test_infonce_rectangular_extra_negatives_hand_computed():
    """z2 may carry extras beyond the paired columns: target arange(N) still
    points at the paired diagonal, extras only join the softmax denominator."""
    generator = torch.Generator().manual_seed(5)
    z1 = torch.randn(2, 8, generator=generator)
    z2_paired = torch.randn(2, 8, generator=generator)
    extra = torch.randn(1, 8, generator=generator)
    z2_all = torch.cat([z2_paired, extra])
    loss = simcse_infonce_loss(z1, z2_all, temp=0.05)
    logits = cosine_logits(z1, z2_all, 0.05)  # (2, 3)
    assert logits.shape == (2, 3)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    torch.testing.assert_close(loss, expected)


def test_supcon_rectangular_same_label_extra_becomes_positive():
    generator = torch.Generator().manual_seed(6)
    z1 = torch.randn(2, 8, generator=generator)
    z2 = torch.cat([torch.randn(2, 8, generator=generator),
                    torch.randn(1, 8, generator=generator)])
    labels1 = torch.tensor([0, 1])
    labels2 = torch.tensor([0, 1, 0])  # the extra candidate shares anchor 0's label
    loss = supervised_contrastive_loss(z1, z2, labels1, labels2, temp=0.05)
    logits = cosine_logits(z1, z2, 0.05)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # anchor 0 positives: cols {0 (pair), 2 (same-label extra)}; anchor 1: {1}.
    expected = (-log_probs[0][[0, 2]].sum() / 2 - log_probs[1][[1]].sum() / 1) / 2
    torch.testing.assert_close(loss, expected)


def test_two_crop_unpaired_instances_reclaimed_as_candidates():
    """Image 0: A pairs, B unpaired; image 1: C pairs, D unpaired. With D14
    the unpaired B/D join as extra candidates; candidate_size = 4 > 2."""
    box_a = torch.tensor([[0.2, 0.5, 0.1, 0.2]])
    box_b = torch.tensor([[0.7, 0.3, 0.1, 0.1]])
    box_b_lost = torch.tensor([[0.08, 0.08, 0.04, 0.04]])  # B's counterpart vanished
    targets = [
        {"boxes": torch.cat([box_a, box_b]), "labels": torch.tensor([0, 1])},
        {"boxes": box_a.clone(), "labels": torch.tensor([0])},
        {"boxes": torch.cat([box_a.clone(), box_b_lost]), "labels": torch.tensor([0, 1])},
        {"boxes": box_a.clone(), "labels": torch.tensor([0])},
    ]
    indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    outputs = {"pred_logits": torch.zeros(4, 6, 3)}
    stats_on: dict[str, float] = {}
    loss_on = ssl_contrastive_loss(
        z1, z2, outputs, targets, indices,
        SSLParams(mode="unsup", view="two_crop"), stats_sink=stats_on,
    )
    assert stats_on["paired_size"] == 2.0
    assert stats_on["candidate_size"] == 4.0  # 2 paired + 2 unpaired extras
    assert torch.isfinite(loss_on)
    loss_on.backward()
    # Disabling D14 restores the strict paired pool.
    stats_off: dict[str, float] = {}
    ssl_contrastive_loss(
        z1, z2, outputs, targets, indices,
        SSLParams(mode="unsup", view="two_crop", use_unpaired_as_negatives=False),
        stats_sink=stats_off,
    )
    assert stats_off["candidate_size"] == 2.0


def test_two_crop_unpaired_negatives_keep_ratio_semantics():
    """Asymmetric fixture: image 0 loses object B in view 2, image 1 gains an
    unmatched object D in view 2 — both rejoin as extras; paired_ratio keeps
    its paired/(pool/2) semantics."""
    box_a = torch.tensor([[0.2, 0.5, 0.1, 0.2]])   # image 0 object (both views)
    box_b = torch.tensor([[0.7, 0.3, 0.1, 0.1]])   # image 0 object, view-2 counterpart gone
    box_c = torch.tensor([[0.25, 0.7, 0.1, 0.1]])  # image 1 object (both views)
    box_d = torch.tensor([[0.6, 0.6, 0.08, 0.08]]) # image 1 object, view-2 only
    # View-major rows: image 0 = rows (0, 2), image 1 = rows (1, 3).
    targets = [
        {"boxes": torch.cat([box_a, box_b]), "labels": torch.tensor([0, 1])},
        {"boxes": box_c, "labels": torch.tensor([0])},
        {"boxes": box_a.clone(), "labels": torch.tensor([0])},
        {"boxes": torch.cat([box_c.clone(), box_d]), "labels": torch.tensor([0, 1])},
    ]
    indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([0])),
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
    ]
    z1 = torch.randn(4, 6, 16, requires_grad=True)
    z2 = torch.randn(4, 6, 16, requires_grad=True)
    stats: dict[str, float] = {}
    loss = ssl_contrastive_loss(
        z1, z2, {"pred_logits": torch.zeros(4, 6, 3)}, targets, indices,
        SSLParams(mode="unsup", view="two_crop"), stats_sink=stats,
    )
    assert stats["pool_size"] == 6.0
    assert stats["paired_size"] == 2.0  # anchors: A and C
    assert stats["candidate_size"] == 4.0  # A', C' paired + B, D unpaired extras
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None


def test_sigreg_training_step_emits_and_backprops():
    """R1-A1 (design doc 16): loss_sigreg joins the total and reaches the head."""
    model = make_ssl_train_model(sigreg_coef=0.2, projector="mlp2")
    batch = make_two_crop_batch(pairs=2, matching=True, seed=3)
    result = model.training_step(batch, step=0)
    assert "loss_sigreg" in result.log_dict
    assert torch.isfinite(result.log_dict["loss_sigreg"])
    assert result.log_dict["loss_sigreg"] > 0.0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in model.model.ssl_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_sigreg_disabled_by_default_matches_baseline():
    """Without sigreg_coef no raw taps, no loss_sigreg key — baseline intact."""
    model = make_ssl_train_model()
    batch = make_two_crop_batch(pairs=2, matching=True, seed=3)
    result = model.training_step(batch, step=0)
    assert "loss_sigreg" not in result.log_dict
    raw = model.model._forward_train(
        batch["image"],
        targets=[
            {"boxes": b, "labels": c} for b, c in zip(batch["bboxes"], batch["classes"])
        ],
    )
    assert "ssl_s1" not in raw and "ssl_s2" not in raw
