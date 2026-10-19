# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Bounding box utilities for detection tasks.

Internal representation throughout the framework: normalized
``(x_center, y_center, w, h)`` in [0, 1] (albumentations/YOLO convention).
"""

from __future__ import annotations

import torch
from torchvision.ops import box_convert

__all__ = [
    "box_cxcywh_norm_to_xyxy",
    "box_xyxy_to_cxcywh_norm",
    "elementwise_box_iou",
    "elementwise_giou_loss",
    "elementwise_siou",
    "generalized_box_iou_loss",
    "clip_normalize_boxes",
]


def clip_normalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Baseline guard before any IoU consumption of predicted boxes.

    NaN/inf -> 0/1 and clamp to the unit box (baseline ``box_ops``
    semantics: the IoU stays finite and computed instead of poisoning the
    whole loss term into a ``nan_to_num`` zero). For finite in-range boxes
    this is an exact identity.
    """
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    return boxes.clamp(min=0.0, max=1.0)


def box_cxcywh_norm_to_xyxy(
    boxes: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Normalized (cx, cy, w, h) -> absolute (x1, y1, x2, y2) pixels.

    ``size`` is (width, height) — the multiplication order matters.
    """
    w, h = size
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        dim=-1,
    )


def box_xyxy_to_cxcywh_norm(
    boxes: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Absolute (x1, y1, x2, y2) pixels -> normalized (cx, cy, w, h)."""
    w, h = size
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
    boxes = boxes / torch.tensor([w, h, w, h], device=boxes.device)
    return boxes


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Standard GIoU (Gilmer et al., CVPR 2019) on absolute xyxy boxes.

    Self-implemented: torchvision's version assumes integer-pixel boxes
    (+1 conventions) and returns -inf for identical unit-scale boxes.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-9)

    elt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    erb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    ewh = (erb - elt).clamp(min=0)
    area_c = ewh[..., 0] * ewh[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-9)


def generalized_box_iou_loss(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Pairwise GIoU loss (1 - GIoU, range [0, 2]) on absolute xyxy boxes.

    No upper clamp: GIoU's defining property is its gradient for
    non-overlapping pairs (1-GIoU saturating at 1 would zero it).
    """
    giou = _generalized_box_iou(boxes1, boxes2)
    loss = 1 - giou
    return torch.nan_to_num(loss, nan=1.0, posinf=2.0, neginf=-1.0)


def elementwise_giou_loss(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Row-aligned GIoU loss (1 - GIoU) for pre-matched pairs.

    The O(N) counterpart of :func:`generalized_box_iou_loss` for
    row-aligned (pre-matched) inputs: identical per-pair math — including
    the 1e-9 union/enclosing-area floors and the ``nan_to_num`` guards —
    without materializing the N×N pairwise matrix of which only the
    diagonal is consumed (design doc 23's box soft-target leg).
    """
    _assert_equal_length(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-9)
    elt = torch.min(boxes1[:, :2], boxes2[:, :2])
    erb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    ewh = (erb - elt).clamp(min=0)
    area_c = ewh[:, 0] * ewh[:, 1]
    giou = iou - (area_c - union) / area_c.clamp(min=1e-9)
    return torch.nan_to_num(1 - giou, nan=1.0, posinf=2.0, neginf=-1.0)


def _assert_equal_length(boxes1: torch.Tensor, boxes2: torch.Tensor) -> None:
    """Reject unequal-length operands so a length-1 side cannot silently broadcast."""
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            "elementwise box ops expect boxes1 and boxes2 to have the same length, "
            f"got {boxes1.shape[0]} and {boxes2.shape[0]}"
        )


def elementwise_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """IoU and union for pre-matched box pairs (rows aligned, absolute xyxy).

    Unlike ``torchvision.ops.box_iou`` this avoids materializing the full
    N×N pairwise matrix of which only the diagonal is consumed for matched
    pairs — O(N) memory instead of O(N²). The union clamps at ``eps=1e-7``
    only in the degenerate (union == 0) case, so identical non-degenerate
    boxes yield IoU == 1.0 exactly.

    Returns:
        (iou, union): per-pair ``[N]`` tensors.
    """
    _assert_equal_length(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def elementwise_siou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    theta: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Scylla-IoU (SIoU) loss for pre-matched box pairs (arXiv:2205.12740).

    ``L = 1 - IoU + (distance + shape) / 2`` where the distance term is
    modulated by an angle cost through ``gamma = 2 - Lambda`` and
    normalized anisotropically by the smallest enclosing box (c_w, c_h).
    The angle cost uses the paper's form ``1 - 2*sin^2(arcsin(s) - pi/4)``
    rewritten as ``Lambda = 2*s*sqrt(1-s^2)`` with ``s = min(dx, dy)/sigma``;
    since ``s <= 1/sqrt(2)`` by construction, this form has no arcsin domain
    or 45-degree gradient singularity. At equal centers ``s = 0`` and both
    rhos vanish, so the angle machinery contributes nothing — the loss
    degrades smoothly to ``1 - IoU + shape/2``.

    Must not run under pure fp16: the ``eps**2`` summand underflows there
    and a zero center distance would divide by zero (AMP fp32/bf16 safe).

    Args:
        boxes1: ``[N, 4]`` predicted boxes in ``[x0, y0, x1, y1]`` format,
            matched order.
        boxes2: ``[N, 4]`` target boxes in ``[x0, y0, x1, y1]`` format,
            matched order.
        theta: shape-cost exponent; the paper searches per dataset in
            ``[2, 6]`` and reports values near 4.
        eps: numerical floor for denominator clamps.

    Returns:
        A ``[N]`` tensor of non-negative per-pair SIoU loss values; zero
        for identical boxes and below 3 in all cases.

    Raises:
        ValueError: If ``boxes1`` and ``boxes2`` have unequal length.
    """
    _assert_equal_length(boxes1, boxes2)

    c1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    dx = (c1[:, 0] - c2[:, 0]).abs()
    dy = (c1[:, 1] - c2[:, 1]).abs()
    # The eps^2 summand keeps sqrt away from the singular-at-zero backward (a
    # clamp after sqrt would turn its infinite gradient into NaN via 0 * inf).
    s = torch.minimum(dx, dy) / torch.sqrt(dx * dx + dy * dy + eps * eps)
    angle = 2.0 * s * torch.sqrt(1.0 - s * s)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    cw = (rb[:, 0] - lt[:, 0]).clamp(min=eps)
    ch = (rb[:, 1] - lt[:, 1]).clamp(min=eps)
    gamma = 2.0 - angle
    distance = (1 - torch.exp(-gamma * (dx / cw) ** 2)) + (
        1 - torch.exp(-gamma * (dy / ch) ** 2)
    )

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    omega_w = (w1 - w2).abs() / torch.max(w1, w2).clamp(min=eps)
    omega_h = (h1 - h2).abs() / torch.max(h1, h2).clamp(min=eps)
    shape = (1 - torch.exp(-omega_w)) ** theta + (1 - torch.exp(-omega_h)) ** theta

    iou, _ = elementwise_box_iou(boxes1, boxes2)
    return 1 - iou + (distance + shape) * 0.5
