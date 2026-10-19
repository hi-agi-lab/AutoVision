# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETRv2 criterion: Varifocal (VFL) classification + L1 + GIoU boxes.

Losses (baseline weight dict): ``loss_vfl 1.0 + loss_bbox 5.0 + loss_giou

Source lineage: RT-DETRv2 criterion (``lyuwenyu/RT-DETR`` @ ``df2a96c``, Apache-2.0),
restructured by AutoVision; provenance ledger entry (P1-4).
2.0``; the box losses are normalized by the total number of target boxes
(averaged across ranks with ``all_reduce``). Auxiliary decoder layers are
weighted with factor 1.0 through the same loss names (``*_aux_{i}``), like
the baseline deep supervision.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from autovision.geometry.box_ops import (
    elementwise_box_iou,
    elementwise_giou_loss,
    elementwise_siou,
    generalized_box_iou_loss,
)
from autovision.tasks.detection.shared.denoising import (
    get_cdn_matched_indices,
)
from autovision.tasks.detection.shared.dfine_utils import (
    flatten_match_indices,
)
from autovision.tasks.detection.shared.matcher import (
    indices_to_device,
)

__all__ = ["RTDETRCriterionV2", "VFL_LOSS_WEIGHT_DICT"]

VFL_LOSS_WEIGHT_DICT = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def _clip_normalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Baseline guard: NaN/inf → 0 and clamp to the unit box before any
    L1/IoU/GIoU consumption of predicted boxes."""
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    return boxes.clamp(min=0.0, max=1.0)


def loss_labels_vfl(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    query_mask: torch.Tensor | None = None,
    flat_indexing: bool = False,
) -> torch.Tensor:
    """Varifocal loss with IoU-scored targets on the matched queries.

    Positives: BCE against the matched IoU score, weighted by
    ``target_score^gamma``. Negatives: BCE against 0, weighted by
    ``pred_score^gamma`` (the quality-focal formulation).

    ``query_mask``: optional (B, Q) bool marking *valid* queries. With
    density-guided dynamic queries the padded tail of every image is
    excluded from the classification loss (baseline Dome semantics —
    box losses need no mask, they only touch matched pairs).
    """
    logits = outputs["pred_logits"]  # (B, Q, C)
    batch_size, num_queries, num_classes = logits.shape
    # fp32 targets: IoU scores are fp32, and under bf16 autocast the
    # index_put_ above would otherwise clash with a bf16 destination.
    target_scores = torch.zeros_like(logits, dtype=torch.float32)
    target = torch.zeros_like(target_scores)  # one-hot class gate (baseline)
    # Baseline sanitizes the predicted side before the quality IoU (NaN
    # boxes would zero the whole VFL term via the trailing nan_to_num).
    pred_boxes = _clip_normalize_boxes(outputs["pred_boxes"])
    if flat_indexing:
        # W2 flat path: one row-aligned IoU over the concatenated matched
        # pairs + a single scattered write, instead of per-image loop
        # iterations with a pairwise IoU-diagonal each. Same math; the
        # fp32 summation order of the trailing BCE is unchanged.
        device = logits.device
        flat_q, _, _ = flatten_match_indices(indices, num_queries, device)
        if flat_q.numel():
            gt_matched = torch.cat(
                [
                    targets[batch_index]["boxes"][dst.to(device)].to(
                        device=device, dtype=pred_boxes.dtype
                    )
                    for batch_index, (_, dst) in enumerate(indices)
                    if dst.numel()
                ]
            )
            classes_matched = torch.cat(
                [
                    targets[batch_index]["labels"][dst.to(device)].to(device=device)
                    for batch_index, (_, dst) in enumerate(indices)
                    if dst.numel()
                ]
            )
            ious, _ = elementwise_box_iou(
                _cxcywh_to_xyxy_unit(pred_boxes.reshape(-1, 4)[flat_q].detach()),
                _cxcywh_to_xyxy_unit(gt_matched),
            )
            target_scores.reshape(-1, num_classes)[flat_q, classes_matched] = (
                ious.clamp(min=0.0)
            )
            target.reshape(-1, num_classes)[flat_q, classes_matched] = 1.0
    else:
        for batch_index, (src, dst) in enumerate(indices):
            if src.numel() == 0:
                continue
            ious = _matched_iou(
                pred_boxes[batch_index, src].detach(),
                targets[batch_index]["boxes"][dst].to(pred_boxes.device),
            )
            classes = targets[batch_index]["labels"][dst].to(logits.device)
            target_scores[batch_index, src, classes] = ious
            target[batch_index, src, classes] = 1.0
    pred_scores = logits.sigmoid().detach()
    # Baseline RT-DETRv2 VFL: the (1 - .) term is gated by the ONE-HOT
    # class indicator, NOT the continuous IoU scores — positives pay no
    # negative-class penalty on their own class channel.
    weight = alpha * pred_scores.pow(gamma) * (1 - target) + target_scores
    loss = F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none")
    loss = loss * weight
    if query_mask is not None:
        loss = loss * query_mask.to(loss.dtype).unsqueeze(-1)
    return loss.sum() / num_boxes.clamp(min=1)


def _matched_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """IoU of matched pairs (rows aligned)."""
    from torchvision.ops import box_iou

    pred_xyxy = _cxcywh_to_xyxy_unit(pred_boxes)
    target_xyxy = _cxcywh_to_xyxy_unit(target_boxes)
    ious = torch.diag(box_iou(pred_xyxy, target_xyxy))
    return ious.clamp(min=0.0)


def _cxcywh_to_xyxy_unit(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou

    return box_iou(_cxcywh_to_xyxy_unit(boxes1), _cxcywh_to_xyxy_unit(boxes2))


def loss_boxes(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    *,
    iou_term: Literal["giou", "siou"] = "giou",
    siou_theta: float = 4.0,
    flat_indexing: bool = False,
) -> dict[str, torch.Tensor]:
    """L1 + a matched-pair IoU loss, normalized by num_boxes (rank-reduced).

    ``iou_term="giou"`` (default) emits ``loss_giou`` (1 - GIoU);
    ``iou_term="siou"`` emits ``loss_siou`` (Scylla-IoU, arXiv:2205.12740)
    instead — the emissions are mutually exclusive, so every box-loss
    consumer switches together (design doc 15 §4.2).

    ``flat_indexing`` (design doc 26 W2): one flat gather over the
    concatenated matched pairs with the row-aligned
    :func:`~autovision.geometry.box_ops.elementwise_giou_loss` instead of
    the per-image loop of pairwise-matrix diagonals; numerically
    equivalent up to float summation order (tested < 1e-6).
    """
    l1_total = outputs["pred_boxes"].sum() * 0.0
    iou_total = outputs["pred_boxes"].sum() * 0.0
    pred_boxes_sanitized = _clip_normalize_boxes(outputs["pred_boxes"])
    if flat_indexing:
        device = pred_boxes_sanitized.device
        num_queries = pred_boxes_sanitized.shape[1]
        flat_q, _, _ = flatten_match_indices(indices, num_queries, device)
        if flat_q.numel():
            gt_matched = torch.cat(
                [
                    targets[batch_index]["boxes"][dst.to(device)].to(
                        device=device, dtype=pred_boxes_sanitized.dtype
                    )
                    for batch_index, (_, dst) in enumerate(indices)
                    if dst.numel()
                ]
            )
            pred = pred_boxes_sanitized.reshape(-1, 4)[flat_q]
            l1_total = l1_total + (pred - gt_matched).abs().sum()
            pred_xyxy = _cxcywh_to_xyxy_unit(pred)
            target_xyxy = _cxcywh_to_xyxy_unit(gt_matched)
            if iou_term == "siou":
                iou_total = iou_total + elementwise_siou(
                    pred_xyxy, target_xyxy, theta=siou_theta
                ).sum()
            else:
                iou_total = iou_total + elementwise_giou_loss(
                    pred_xyxy, target_xyxy
                ).sum()
        losses = {
            "loss_bbox": l1_total / num_boxes.clamp(min=1),
        }
        if iou_term == "siou":
            losses["loss_siou"] = iou_total / num_boxes.clamp(min=1)
        else:
            losses["loss_giou"] = iou_total / num_boxes.clamp(min=1)
        return losses
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred = pred_boxes_sanitized[batch_index, src]
        target = targets[batch_index]["boxes"][dst]
        l1_total = l1_total + (pred - target).abs().sum()
        # generalized_box_iou_loss is PAIRWISE (N, M): taking .sum() over the
        # full matrix would add every pred-vs-every-other-GT cross term. The
        # matched pairs live on the diagonal.
        pred_xyxy = _cxcywh_to_xyxy_unit(pred)
        target_xyxy = _cxcywh_to_xyxy_unit(target)
        if iou_term == "siou":
            iou_total = iou_total + elementwise_siou(
                pred_xyxy, target_xyxy, theta=siou_theta
            ).sum()
        else:
            giou_matrix = generalized_box_iou_loss(pred_xyxy, target_xyxy)
            iou_total = iou_total + giou_matrix.diagonal().sum()
    losses = {
        "loss_bbox": l1_total / num_boxes.clamp(min=1),
    }
    if iou_term == "siou":
        losses["loss_siou"] = iou_total / num_boxes.clamp(min=1)
    else:
        losses["loss_giou"] = iou_total / num_boxes.clamp(min=1)
    return losses


class RTDETRCriterionV2(nn.Module):
    """Matches + computes the weighted RT-DETRv2 loss over all decoder outputs.

    When ``dn_outputs``/``dn_meta`` are provided (contrastive denoising
    training), each denoising layer is supervised against its *known*
    assignment (positive noised query ↔ its source GT; no matching needed),
    normalized by ``num_boxes × dn_num_group`` — baseline semantics.
    """

    def __init__(
        self, matcher: nn.Module, weight_dict: dict[str, float] | None = None
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = dict(weight_dict or VFL_LOSS_WEIGHT_DICT)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to the module-level VFL loss (subclass override seam)."""
        return loss_labels_vfl(outputs, targets, indices, num_boxes, query_mask=query_mask)

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Delegate to the module-level box loss (subclass override seam)."""
        return loss_boxes(outputs, targets, indices, num_boxes)

    def forward(
        self,
        outputs_list: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        world_size: int = 1,
        dn_outputs: list[dict[str, torch.Tensor]] | None = None,
        dn_meta: dict | None = None,
        query_mask: torch.Tensor | None = None,
        enc_aux_outputs: list[dict[str, torch.Tensor]] | None = None,
        enc_meta: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        num_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
        device = outputs_list[-1]["pred_logits"].device
        num_boxes = torch.tensor(float(num_boxes), device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes /= torch.distributed.get_world_size()

        losses: dict[str, torch.Tensor] = {}
        num_entries = len(outputs_list)
        for output_index, outputs in enumerate(outputs_list):
            indices = indices_to_device(self.matcher(outputs, targets), device)
            suffix = "" if output_index == num_entries - 1 else f"_aux_{output_index}"
            losses[f"loss_vfl{suffix}"] = self.loss_labels(
                outputs, targets, indices, num_boxes, query_mask=query_mask
            )
            for name, value in self.loss_boxes(outputs, targets, indices, num_boxes).items():
                losses[f"{name}{suffix}"] = value

        if dn_outputs and dn_meta is not None:
            dn_indices = get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = torch.clamp(
                num_boxes * dn_meta["dn_num_group"], min=1
            )
            for dn_index, dn_layer in enumerate(dn_outputs):
                suffix = f"_dn_{dn_index}"
                losses[f"loss_vfl{suffix}"] = self.loss_labels(
                    dn_layer, targets, dn_indices, dn_num_boxes
                )
                for name, value in self.loss_boxes(
                    dn_layer, targets, dn_indices, dn_num_boxes
                ).items():
                    losses[f"{name}{suffix}"] = value

        # Encoder-proposal supervision (baseline RT-DETRv2 enc_aux chain):
        # the enc top-k selection is non-differentiable and the query
        # content is detached, so without this branch the proposal heads
        # never receive a gradient. Full VFL+box pass with its own match.
        if enc_aux_outputs:
            class_agnostic = bool(enc_meta.get("class_agnostic", False))
            if class_agnostic:
                enc_targets = [
                    {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
                    for t in targets
                ]
            else:
                enc_targets = targets
            for enc_index, enc_outputs in enumerate(enc_aux_outputs):
                enc_indices = indices_to_device(
                    self.matcher(enc_outputs, enc_targets), device
                )
                suffix = f"_enc_{enc_index}"
                losses[f"loss_vfl{suffix}"] = self.loss_labels(
                    enc_outputs, enc_targets, enc_indices, num_boxes
                )
                for name, value in self.loss_boxes(
                    enc_outputs, enc_targets, enc_indices, num_boxes
                ).items():
                    losses[f"{name}{suffix}"] = value

        weights = dict(self.weight_dict)
        for output_index in range(num_entries - 1):
            for key, value in self.weight_dict.items():
                weights[f"{key}_aux_{output_index}"] = value
        for dn_index in range(len(dn_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_dn_{dn_index}"] = value
        for enc_index in range(len(enc_aux_outputs or [])):
            for key, value in self.weight_dict.items():
                weights[f"{key}_enc_{enc_index}"] = value
        total = sum(losses[key] * weight for key, weight in weights.items() if key in losses)
        losses["loss"] = total
        # Keep the loss graph-attached: DDP averages gradients across ranks,
        # so the training loss must not be all-reduced/detached here.
        # ``reduce_dict`` is for logging only and would break backward.
        return losses
