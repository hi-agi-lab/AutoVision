# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Cross-model distillation losses for the D-FINE criterion (design doc 23).

The teacher never enters the criterion as a module: the training wrapper
runs it (frozen, no grad) and hands the criterion a ``distill_targets``
dict of raw tensors plus a GT→teacher-query map. The join with the
student's union match happens at criterion level — that is where
``indices_go`` lives — inside :func:`cross_model_distill_losses`: for
every union pair (student query, GT) whose GT the teacher also matched,
the corresponding teacher prediction becomes a soft target for three legs:

- ``loss_distill_cls``: BCE of the student's GT-label logit against the
  teacher's confidence for the same label — calibration transfer, not a
  replacement for VFL's hard supervision;
- ``loss_distill_box``: L1 + ``box_giou_coef`` × (1 − GIoU) against the
  teacher's refined box — a smooth regression target; the GT box losses
  stay the primary supervision;
- ``loss_distill_ddf``: a matched-only variant of the official DDF. The
  self-distillation's unmatched-query term (teacher max-prob weight) is
  meaningless across models — query spaces do not correspond — so only
  the IoU-weighted matched pairs are distilled, in the shared W(n) bin
  space (the wrapper enforces teacher/student ``reg_max``/``reg_scale``
  agreement before the leg is enabled).

Union pairs whose GT the teacher missed (teacher false negatives) are
skipped by every leg — the GT hard supervision covers them. All legs run
in fp32 (the framework's fp32-island convention) and normalize per
union-matched pair (cls/box: ``num_boxes_go``, same denominator as the
box losses; DDF: covered elements).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from autovision.geometry.box_ops import (
    clip_normalize_boxes,
    elementwise_box_iou,
    elementwise_giou_loss,
)
from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
)
from autovision.tasks.detection.shared.dfine_utils import (
    flatten_match_indices,
)

__all__ = [
    "DistillParams",
    "cross_model_ddf_loss",
    "cross_model_distill_losses",
]


@dataclass(frozen=True)
class DistillParams:
    """Shared-component distillation contract (design doc 23 §4.1).

    Built from the task-family config (``DETRDistillArgs``) by the train
    wrapper, keeping the components layer free of detr imports — the
    ``SSLParams`` precedent. Coefficients ride the criterion's
    ``weight_dict`` (applied once); the legs are computed at the final
    decoder layer only.
    """

    cls_coef: float = 1.0
    box_coef: float = 1.0
    box_giou_coef: float = 2.0
    ddf_coef: float = 1.5
    ddf_temperature: float = 5.0


def cross_model_ddf_loss(
    student_corners: torch.Tensor,
    teacher_corners: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Matched-only KL of per-side distributions, IoU-weighted (doc 23 §3.3).

    ``student_corners``/``teacher_corners``: (P, 4 × bins) raw logits in
    the shared W(n) bin space; ``quality``: (P,) per-pair IoU weight
    (student box vs its assigned GT — the official ``weight_targets``
    semantics with ``targets=``). Returns the T²-scaled mean over all
    4 × P elements.
    """
    num_bins = student_corners.shape[-1] // 4
    student_log_probs = torch.log_softmax(
        student_corners.float().reshape(-1, 4, num_bins) / temperature, dim=-1
    )
    teacher_probs = torch.softmax(
        teacher_corners.float().reshape(-1, 4, num_bins) / temperature, dim=-1
    )
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    weighted = (temperature**2) * kl * quality.float().unsqueeze(-1)
    return weighted.sum() / max(quality.numel() * 4, 1)


def cross_model_distill_losses(
    final_outputs: dict[str, torch.Tensor],
    final_corners: torch.Tensor | None,
    targets: list[dict[str, torch.Tensor]],
    indices_go: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes_go: torch.Tensor,
    distill_targets: dict,
    params: DistillParams,
    stats_sink: dict,
) -> dict[str, torch.Tensor]:
    """The three distillation legs over the student's union match.

    ``distill_targets`` carries (all batch-major, from the training
    wrapper): ``teacher_logits`` (B, Q_t, C), ``teacher_boxes``
    (B, Q_t, 4), optional ``teacher_corners`` (B, Q_t, 4 × bins) and the
    ``gt_to_teacher_query`` map (B, G) int64 with −1 for GTs the teacher
    did not match. ``stats_sink`` receives ``covered``/``union`` counts.
    """
    device = final_outputs["pred_logits"].device
    _, num_queries = final_outputs["pred_logits"].shape[:2]
    flat_q, flat_gt, batch_idx = flatten_match_indices(indices_go, num_queries, device)
    connected_zero = final_outputs["pred_logits"].sum() * 0.0
    stats_sink["union"] = int(flat_q.numel())

    legs_cls = params.cls_coef > 0
    legs_box = params.box_coef > 0
    legs_ddf = (
        params.ddf_coef > 0
        and final_corners is not None
        and distill_targets.get("teacher_corners") is not None
    )
    if flat_q.numel() == 0:
        stats_sink["covered"] = 0
        losses = {}
        if legs_cls:
            losses["loss_distill_cls"] = connected_zero
        if legs_box:
            losses["loss_distill_box"] = connected_zero
        if legs_ddf:
            losses["loss_distill_ddf"] = connected_zero
        return losses

    # GT→teacher join: a union pair is covered when the teacher also
    # matched its GT (−1 entries = teacher false negatives, skipped).
    teacher_query = distill_targets["gt_to_teacher_query"][batch_idx, flat_gt]
    covered = teacher_query >= 0
    stats_sink["covered"] = int(covered.sum())
    if not bool(covered.any()):
        losses = {}
        if legs_cls:
            losses["loss_distill_cls"] = connected_zero
        if legs_box:
            losses["loss_distill_box"] = connected_zero
        if legs_ddf:
            losses["loss_distill_ddf"] = connected_zero
        return losses

    batch_c = batch_idx[covered]
    query_c = teacher_query[covered]
    gt_boxes = torch.cat(
        [
            targets[batch_index]["boxes"][dst.to(device)]
            for batch_index, (_, dst) in enumerate(indices_go)
            if dst.numel()
        ]
    )[covered]
    labels = torch.cat(
        [
            targets[batch_index]["labels"][dst.to(device)]
            for batch_index, (_, dst) in enumerate(indices_go)
            if dst.numel()
        ]
    )[covered]

    student_logits = (
        final_outputs["pred_logits"].reshape(-1, final_outputs["pred_logits"].shape[-1])[flat_q][covered].float()
    )
    student_boxes = final_outputs["pred_boxes"].reshape(-1, 4)[flat_q][covered]
    teacher_logits = distill_targets["teacher_logits"][batch_c, query_c].float()
    teacher_boxes = distill_targets["teacher_boxes"][batch_c, query_c]

    losses: dict[str, torch.Tensor] = {}
    if legs_cls:
        # Teacher confidence for the assigned GT label as the BCE target
        # of the student's logit — ε-clamped so a saturated teacher side
        # cannot push an infinite BCE gradient.
        target_prob = (
            torch.sigmoid(teacher_logits.gather(1, labels.unsqueeze(1)))
            .squeeze(1)
            .clamp(0.01, 0.99)
        )
        student_score = student_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        losses["loss_distill_cls"] = F.binary_cross_entropy_with_logits(
            student_score, target_prob, reduction="sum"
        ) / num_boxes_go.clamp(min=1)
    if legs_box:
        l1 = (student_boxes.float() - teacher_boxes.float()).abs().sum(dim=-1)
        giou = elementwise_giou_loss(
            box_cxcywh_to_xyxy(student_boxes), box_cxcywh_to_xyxy(teacher_boxes)
        )
        losses["loss_distill_box"] = (
            l1 + params.box_giou_coef * giou
        ).sum() / num_boxes_go.clamp(min=1)
    if legs_ddf:
        quality, _ = elementwise_box_iou(
            box_cxcywh_to_xyxy(clip_normalize_boxes(student_boxes)),
            box_cxcywh_to_xyxy(gt_boxes),
        )
        losses["loss_distill_ddf"] = cross_model_ddf_loss(
            final_corners.reshape(-1, final_corners.shape[-1])[flat_q][covered],
            distill_targets["teacher_corners"][batch_c, query_c],
            quality.clamp(min=0.0),
            params.ddf_temperature,
        )
    return losses
