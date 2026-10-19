# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dynamic label assigners for the dense PicoDet head.

- :class:`SimOTAAssigner` — YOLOX-style cost-based dynamic-k matching:
  candidates are anchor centers inside the GT box or its central
  half-extent region, cost = BCE class + log-IoU regression with a giant
  penalty for non-candidates, and each GT pulls its dynamic-k
  (top-10 IoU sum) cheapest anchors; conflicts keep the cheapest GT.
- :class:`TaskAlignedAssigner` — TOOD-style alignment metric
  (score^alpha * iou^beta) with per-GT top-k selection and normalized
  alignment weights.

Both return the same :class:`AssignResult`; the VFL/GIoU/DFL losses use
``assigned_quality`` as the positive classification target and
``loss_weight`` as the positive loss weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

__all__ = ["AssignResult", "SimOTAAssigner", "TaskAlignedAssigner"]


@dataclass
class AssignResult:
    """Assignment for one image: per-anchor records over all levels."""

    assigned_gt_idx: torch.Tensor  # (N,) gt index, -1 = background
    fg_mask: torch.Tensor  # (N,) bool
    assigned_quality: torch.Tensor  # (N,) IoU of the assigned pair
    loss_weight: torch.Tensor  # (N,) positive loss weight


def _empty_result(n: int, device: torch.device) -> AssignResult:
    return AssignResult(
        assigned_gt_idx=torch.full((n,), -1, dtype=torch.long, device=device),
        fg_mask=torch.zeros(n, dtype=torch.bool, device=device),
        assigned_quality=torch.zeros(n, device=device),
        loss_weight=torch.zeros(n, device=device),
    )


def _centers_in_gt(
    anchor_points: torch.Tensor, gt_xyxy: torch.Tensor
) -> torch.Tensor:
    """(N, 2) anchor centers inside the (M, 4) xyxy GT boxes -> (N, M) bool."""
    left_top = anchor_points[:, None, :] >= gt_xyxy[None, :, :2]
    right_bottom = anchor_points[:, None, :] <= gt_xyxy[None, :, 2:]
    return (left_top & right_bottom).all(dim=-1)


class SimOTAAssigner:
    """YOLOX-style SimOTA dynamic-k assignment."""

    def __init__(
        self,
        topk: int = 10,
        cls_cost_weight: float = 1.0,
        bbox_cost_weight: float = 3.0,
    ) -> None:
        self.topk = topk
        self.cls_cost_weight = cls_cost_weight
        self.bbox_cost_weight = bbox_cost_weight

    @torch.no_grad()
    def assign(
        self,
        pred_logits: torch.Tensor,  # (N, C) raw class logits
        pred_boxes_xyxy: torch.Tensor,  # (N, 4) pixels
        anchor_points: torch.Tensor,  # (N, 2) pixels
        gt_xyxy: torch.Tensor,  # (M, 4) pixels
        gt_labels: torch.Tensor,  # (M,)
    ) -> AssignResult:
        n, m = pred_logits.shape[0], gt_xyxy.shape[0]
        if m == 0 or n == 0:
            return _empty_result(n, pred_logits.device)
        iou = box_iou(pred_boxes_xyxy, gt_xyxy)  # (N, M)

        # Candidates: center inside the GT box, or inside its central
        # half-extent region (w/2 x h/2 box around the GT center).
        in_box = _centers_in_gt(anchor_points, gt_xyxy)
        center = (gt_xyxy[:, :2] + gt_xyxy[:, 2:]) / 2
        half_extent = (gt_xyxy[:, 2:] - gt_xyxy[:, :2]) / 4
        in_center = (
            (anchor_points[:, None, :] - center[None]).abs() <= half_extent[None]
        ).all(dim=-1)
        candidate = in_box | in_center

        # Cost: BCE of the score vs the one-hot GT class + log-IoU loss,
        # with a giant penalty for anchors outside the candidate set.
        num_classes = pred_logits.shape[1]
        gt_onehot = F.one_hot(gt_labels, num_classes=num_classes).float()
        cls_cost = F.binary_cross_entropy_with_logits(
            pred_logits.unsqueeze(1).expand(n, m, num_classes),
            gt_onehot.unsqueeze(0).expand(n, m, num_classes),
            reduction="none",
        ).sum(-1)
        iou_cost = -torch.log(iou.clamp(min=1e-8))
        cost = (
            self.cls_cost_weight * cls_cost
            + self.bbox_cost_weight * iou_cost
            + (~candidate).float() * 100000.0
        )

        # Dynamic k per GT: sum of the top-10 IoUs, floored to >= 1. The
        # top-k runs over the anchor dimension, bounded by the anchor count.
        topk_ious, _ = torch.topk(iou, min(self.topk, n), dim=0)
        dynamic_ks = (
            topk_ious.sum(0).floor().to(torch.long).clamp(min=1, max=self.topk)
        )
        pos_mask = torch.zeros(n, m, dtype=torch.bool, device=iou.device)
        for j in range(m):
            _, indices = torch.topk(cost[:, j], dynamic_ks[j], largest=False)
            pos_mask[indices, j] = True

        # An anchor claimed by several GTs keeps the cheapest one.
        cost_masked = cost.masked_fill(~pos_mask, float("inf"))
        gt_for_anchor = cost_masked.argmin(dim=1)
        fg_mask = pos_mask.any(dim=1)
        assigned = torch.where(
            fg_mask, gt_for_anchor, torch.full_like(gt_for_anchor, -1)
        )
        quality = iou.gather(1, assigned.clamp(min=0).unsqueeze(1)).squeeze(1)
        quality = torch.where(fg_mask, quality, torch.zeros_like(quality))
        return AssignResult(assigned, fg_mask, quality, quality.clone())


class TaskAlignedAssigner:
    """TOOD-style alignment-metric assignment."""

    def __init__(self, topk: int = 10, alpha: float = 1.0, beta: float = 6.0) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def assign(
        self,
        pred_logits: torch.Tensor,  # (N, C) raw class logits
        pred_boxes_xyxy: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_xyxy: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> AssignResult:
        n, m = pred_logits.shape[0], gt_xyxy.shape[0]
        if m == 0 or n == 0:
            return _empty_result(n, pred_logits.device)
        iou = box_iou(pred_boxes_xyxy, gt_xyxy)  # (N, M)
        cls_score = pred_logits.sigmoid()[:, gt_labels]  # (N, M)
        align = cls_score.pow(self.alpha) * iou.pow(self.beta)
        align = align * _centers_in_gt(anchor_points, gt_xyxy).float()

        # Top-k anchors per GT by the alignment metric.
        k = min(self.topk, n)
        topk_align, topk_idx = align.topk(k, dim=0)  # (k, M)
        columns = torch.arange(m, device=align.device).expand(k, m)
        pos_mask = torch.zeros(n, m, dtype=torch.bool, device=align.device)
        pos_mask[topk_idx, columns] = topk_align > 0

        align_masked = align * pos_mask.float()
        fg_mask = pos_mask.any(dim=1)
        assigned = torch.where(
            fg_mask,
            align_masked.argmax(dim=1),
            torch.full((n,), -1, dtype=torch.long, device=align.device),
        )
        quality = iou.gather(1, assigned.clamp(min=0).unsqueeze(1)).squeeze(1)
        quality = torch.where(fg_mask, quality, torch.zeros_like(quality))
        # Positive weight: alignment normalized per GT, scaled by IoU.
        max_per_gt = align_masked.max(dim=0).values
        norm_align = align_masked / max_per_gt.clamp(min=1e-8)
        weight = norm_align.gather(
            1, assigned.clamp(min=0).unsqueeze(1)
        ).squeeze(1) * quality
        weight = torch.where(fg_mask, weight, torch.zeros_like(weight))
        return AssignResult(assigned, fg_mask, quality, weight)
