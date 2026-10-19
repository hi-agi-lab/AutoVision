# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PicoDet losses (GFL family): Varifocal classification, DFL box
distribution regression, GIoU box regression (shared ``box_ops``), plus the
:class:`PicoCriterion` that ties them to the head outputs via a dynamic
label assigner.

Loss weights follow the GFL-family port defaults: classification 1.0,
GIoU 2.0, DFL 0.25. Classification is normalized by the positive anchor
count; box losses by the sum of positive loss weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
    generalized_box_iou_loss,
)
from autovision.tasks.detection.picodet.sim_ota_assigner import (
    SimOTAAssigner,
    TaskAlignedAssigner,
)

__all__ = ["PicoCriterion", "distribution_focal_loss", "varifocal_loss"]


def varifocal_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0
) -> torch.Tensor:
    """Varifocal loss (VFNet) on raw logits (autocast-safe).

    ``pred_logits``: (M, C) raw scores; ``target``: (M, C) soft quality
    (IoU on the assigned class, 0 elsewhere). Positives weight by their
    quality (non-uniform), negatives by ``alpha * p^gamma`` with
    ``p = sigmoid(logits)``. Returns the unnormalized (M, C) loss.
    """
    pred_scores = pred_logits.sigmoid()
    weight = torch.where(target > 0, target, alpha * pred_scores.pow(gamma))
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, target.clamp(min=0.0, max=1.0), reduction="none"
    )
    return loss * weight


def distribution_focal_loss(
    pred_dist: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """DFL (GFL) between the per-side bin distribution and a continuous
    target: bilinear cross-entropy over the two adjacent bins.

    ``pred_dist``: (P, 4, bins) logits; ``target``: (P, 4) distances in
    stride units. Out-of-range targets saturate on the last bin. Returns
    the unnormalized (P, 4) loss.
    """
    bins = pred_dist.shape[-1]
    left = target.floor().clamp(0, bins - 1)
    right = (left + 1).clamp(0, bins - 1)
    weight_left = (right.to(target.dtype) - target).clamp(0.0, 1.0)
    weight_right = 1.0 - weight_left
    log_prob = pred_dist.log_softmax(dim=-1)
    loss = -(
        log_prob.gather(-1, left.to(torch.long).unsqueeze(-1)).squeeze(-1)
        * weight_left
        + log_prob.gather(-1, right.to(torch.long).unsqueeze(-1)).squeeze(-1)
        * weight_right
    )
    return loss


class PicoCriterion(nn.Module):
    """Dense-head criterion: assign -> VFL + GIoU + DFL."""

    def __init__(
        self,
        num_classes: int,
        assigner: str = "simota",
        assigner_topk: int = 10,
        loss_cls_weight: float = 1.0,
        loss_giou_weight: float = 2.0,
        loss_dfl_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if assigner == "simota":
            self.assigner = SimOTAAssigner(topk=assigner_topk)
        elif assigner == "taskaligned":
            self.assigner = TaskAlignedAssigner(topk=assigner_topk)
        else:
            raise ValueError(
                f"Unknown assigner '{assigner}'; supported: simota, taskaligned."
            )
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_giou_weight = loss_giou_weight
        self.loss_dfl_weight = loss_dfl_weight

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Compute the loss dict over a batch.

        ``preds``: the head output dict (``pred_logits`` (B, N, C) raw,
        ``pred_xyxy`` (B, N, 4) pixels, ``pred_dist`` (B, N, 4, bins),
        ``anchor_points`` (N, 2), ``anchor_strides`` (N,)).
        ``targets``: per-image ``{"boxes": (m, 4) normalized cxcywh,
        "labels": (m,)}``. ``image_size``: network canvas (height, width).
        """
        device = preds["pred_logits"].device
        height, width = image_size
        pred_logits = preds["pred_logits"]
        pred_xyxy = preds["pred_xyxy"]
        pred_dist = preds["pred_dist"]
        anchors = preds["anchor_points"]
        strides = preds["anchor_strides"]
        batch, num_anchors, num_classes = pred_logits.shape

        target_quality = torch.zeros(batch, num_anchors, device=device)
        pos_class = torch.full(
            (batch, num_anchors), -1, dtype=torch.long, device=device
        )
        pos_weight = torch.zeros(batch, num_anchors, device=device)
        total_giou = torch.zeros((), device=device)
        total_dfl = torch.zeros((), device=device)

        for i in range(batch):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            if gt_boxes.numel() == 0:
                continue
            gt_xyxy = box_cxcywh_norm_to_xyxy(gt_boxes, (width, height))
            result = self.assigner.assign(
                pred_logits[i], pred_xyxy[i], anchors, gt_xyxy, gt_labels
            )
            fg = result.fg_mask.nonzero(as_tuple=True)[0]
            target_quality[i, fg] = result.assigned_quality[fg]
            pos_class[i, fg] = gt_labels[result.assigned_gt_idx[fg]]
            pos_weight[i] = result.loss_weight
            if fg.numel() == 0:
                continue
            gt_assigned = gt_xyxy[result.assigned_gt_idx[fg]]

            giou_matrix = generalized_box_iou_loss(
                pred_xyxy[i], gt_xyxy
            )  # (N, M)
            assigned_giou = giou_matrix[fg, result.assigned_gt_idx[fg]]
            total_giou = total_giou + (assigned_giou * result.loss_weight[fg]).sum()

            # DFL targets: per-side distances in stride units.
            stride_fg = strides[fg]
            ltrb = torch.stack(
                [
                    (anchors[fg, 0] - gt_assigned[:, 0]) / stride_fg,
                    (anchors[fg, 1] - gt_assigned[:, 1]) / stride_fg,
                    (gt_assigned[:, 2] - anchors[fg, 0]) / stride_fg,
                    (gt_assigned[:, 3] - anchors[fg, 1]) / stride_fg,
                ],
                dim=-1,
            )
            dfl = distribution_focal_loss(pred_dist[i][fg], ltrb)  # (P, 4)
            total_dfl = total_dfl + (dfl.sum(-1) * result.loss_weight[fg]).sum()

        # Varifocal classification over every anchor (positives carry their
        # IoU quality on the assigned class; negatives the alpha*p^gamma term).
        valid = pos_class >= 0
        target_cls = (
            F.one_hot(pos_class.clamp(min=0), num_classes=num_classes).float()
            * target_quality.unsqueeze(-1)
            * valid.unsqueeze(-1)
        )
        vfl = varifocal_loss(
            pred_logits.reshape(-1, num_classes),
            target_cls.reshape(-1, num_classes),
        )
        num_pos = max(1, int(valid.sum()))
        avg_factor = max(float(pos_weight.sum().item()), 1.0)
        loss_vfl = vfl.sum() / num_pos
        loss_giou = total_giou / avg_factor
        loss_dfl = total_dfl / avg_factor
        loss = (
            self.loss_cls_weight * loss_vfl
            + self.loss_giou_weight * loss_giou
            + self.loss_dfl_weight * loss_dfl
        )
        return {
            "loss_vfl": loss_vfl,
            "loss_giou": loss_giou,
            "loss_dfl": loss_dfl,
            "loss": loss,
        }
