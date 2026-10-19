# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented RT-DETRv2 criterion + matcher: VFL + 5-D L1 + ProbIoU.

Losses reuse the baseline weight dict (``loss_vfl 1.0 + loss_bbox 5.0 +
loss_giou 2.0``) and loss names so the parent weighting/reduction logic
applies unchanged. Differences from the axis-aligned parent:

- box regression works on 5-D normalized ``xywhr`` (cx, cy, w, h, θ) —
  the L1 term covers the angle, and GIoU is replaced by the
  differentiable ProbIoU similarity (``1 - probiou``);
- the VFL positive-target quality score is the ProbIoU instead of the
  axis-aligned IoU (a 45°-rotated thin box would score ~0 there);
- the Hungarian matcher costs are the oriented analogues with the same
  weights (``2.0 focal + 5.0 L1 + 2.0 (1 - ProbIoU)``).
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from autovision.geometry.obb import probiou_matrix
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.shared.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)

__all__ = [
    "OrientedHungarianMatcher",
    "OrientedRTDETRCriterion",
    "loss_boxes_oriented",
    "loss_labels_vfl_oriented",
]


def _xywhr(
    boxes: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """Concatenate (…, 4) cxcywh and (…, 1) angle into (…, 5) xywhr."""
    return torch.cat([boxes, angles], dim=-1)


def _matched_probiou(pred_xywhr: torch.Tensor, target_xywhr: torch.Tensor) -> torch.Tensor:
    """ProbIoU of matched pairs (rows aligned), clamped like the parent IoU."""
    return torch.diag(probiou_matrix(pred_xywhr, target_xywhr)).clamp(min=0.0)


def loss_labels_vfl_oriented(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Varifocal loss with ProbIoU-scored targets on the matched queries."""
    logits = outputs["pred_logits"]  # (B, Q, C)
    # fp32 targets: IoU scores are fp32, and under bf16 autocast the
    # index_put_ above would otherwise clash with a bf16 destination.
    target_scores = torch.zeros_like(logits, dtype=torch.float32)
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred_xywhr = _xywhr(
            outputs["pred_boxes"][batch_index, src].detach(),
            outputs["pred_angles"][batch_index, src].detach(),
        )
        target_xywhr = _xywhr(
            targets[batch_index]["boxes"][dst].to(outputs["pred_boxes"].device),
            targets[batch_index]["angles"][dst].to(outputs["pred_boxes"].device),
        )
        ious = _matched_probiou(pred_xywhr, target_xywhr)
        classes = targets[batch_index]["labels"][dst].to(logits.device)
        target_scores[batch_index, src, classes] = ious
    pred_scores = logits.sigmoid()
    weight = alpha * pred_scores.detach().pow(gamma) * (1 - target_scores) + target_scores
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target_scores, reduction="none"
    )
    return (loss * weight).sum() / num_boxes.clamp(min=1)


def loss_boxes_oriented(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """5-D L1 + (1 - ProbIoU) on matched pairs, normalized by num_boxes.

    Loss names (``loss_bbox``/``loss_giou``) match the baseline weight dict.
    """
    l1_total = outputs["pred_boxes"].sum() * 0.0
    probiou_total = outputs["pred_boxes"].sum() * 0.0
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred_xywhr = _xywhr(
            outputs["pred_boxes"][batch_index, src],
            outputs["pred_angles"][batch_index, src],
        )
        target_xywhr = _xywhr(
            targets[batch_index]["boxes"][dst].to(pred_xywhr.device),
            targets[batch_index]["angles"][dst].to(pred_xywhr.device),
        )
        l1_total = l1_total + (pred_xywhr - target_xywhr).abs().sum()
        probiou_total = probiou_total + (1 - probiou_matrix(pred_xywhr, target_xywhr)).diagonal().sum()
    return {
        "loss_bbox": l1_total / num_boxes.clamp(min=1),
        "loss_giou": probiou_total / num_boxes.clamp(min=1),
    }


class OrientedRTDETRCriterion(RTDETRCriterionV2):
    """Oriented criterion: overrides the two box-quality seams of the parent.

    ``loss_labels`` accepts the parent's ``query_mask`` seam (density-
    guided dynamic queries) but ignores it — oriented detection keeps the
    fixed-proposal contract, so there are never padded queries to mask.
    """

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return loss_labels_vfl_oriented(outputs, targets, indices, num_boxes)

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return loss_boxes_oriented(outputs, targets, indices, num_boxes)


class OrientedHungarianMatcher(HungarianMatcher):
    """Hungarian matcher with 5-D L1 + (1 - ProbIoU) box costs."""

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch_logits = outputs["pred_logits"]
        batch_boxes = outputs["pred_boxes"]
        batch_angles = outputs["pred_angles"]
        num_queries = batch_logits.shape[1]

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
            target_xywhr = _xywhr(
                target["boxes"].to(batch_boxes.device),
                target["angles"].to(batch_angles.device),
            )
            target_labels = target["labels"].to(batch_logits.device)
            pred_logits = batch_logits[batch_index]  # (Q, C)
            pred_xywhr = _xywhr(batch_boxes[batch_index], batch_angles[batch_index])

            # Focal-style classification cost (baseline alpha/gamma).
            pred_scores = pred_logits.sigmoid()
            negative_cost = (1 - self.alpha) * pred_scores.pow(self.gamma)
            positive_cost = self.alpha * (1 - pred_scores).pow(self.gamma)
            class_cost = torch.gather(
                positive_cost - negative_cost,
                1,
                target_labels[None].repeat(num_queries, 1),
            )  # (Q, n)

            l1_cost = torch.cdist(pred_xywhr, target_xywhr, p=1)  # (Q, n)
            probiou_cost = 1.0 - probiou_matrix(pred_xywhr, target_xywhr)

            cost = (
                self.cost_class * class_cost
                + self.cost_bbox * l1_cost
                + self.cost_giou * probiou_cost
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
