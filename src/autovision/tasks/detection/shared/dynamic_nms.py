# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Density-modulated per-class NMS for dynamic query selection.

Port of the baseline ``src/zoo/dome/dynamic_nms.py`` (Dome-DETR,
Apache-2.0), ``dynamic_nms_fast`` vectorized variant. Every candidate
carries its own IoU threshold derived from the object density at its
center (``base + scale * density``, wired by the caller): in dense
clusters the threshold rises so more mutually-overlapping candidates
survive (the scene may genuinely contain that many objects), in sparse
regions it stays low so duplicates are pruned aggressively.

The first ``min_num_select`` candidates are exempt from suppression by
construction — the caller force-keeps them (baseline semantics).

Performance note: the greedy scan runs on CPU numpy over a single
transferred IoU matrix. Semantics are identical to the baseline's GPU
scan, but the per-iteration GPU kernel launches + implicit syncs of the
torch version cost ~0.5-1 s/step at VisDrone query counts (the loop
runs per class per image every step).
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou

__all__ = ["per_class_dynamic_nms"]


def _nms_single_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Greedy NMS with per-box IoU thresholds; returns original-space indices.

    Vectorized port of the baseline: all pairwise IoUs are precomputed
    once, then a single score-descending scan suppresses later boxes whose
    IoU with a kept box exceeds *that kept box's* threshold. Everything
    runs on CPU (one bulk transfer of the small inputs) — the greedy scan
    must not issue per-iteration GPU round-trips.
    """
    num = boxes.shape[0]
    order = scores.argsort(descending=True).cpu()
    boxes_cpu = boxes[order].detach().cpu()
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
    # `order` lives on the host after the transfer; index it in place and
    # move the (tiny) result back to the caller's device.
    index = torch.tensor(keep, dtype=torch.long, device=order.device)
    return order[index].to(boxes.device)


def per_class_dynamic_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Per-class greedy NMS with per-candidate IoU thresholds.

    Args:
        boxes: (N, 4) absolute xyxy.
        scores: (N,) confidence used for the descending scan order.
        class_ids: (N,) integer class per candidate.
        iou_thresholds: (N,) per-candidate IoU threshold (indexed by the
            *kept* box).

    Returns:
        Indices into the original (N,) arrays that survive NMS, in
        ascending scan order. Empty input returns an empty index tensor.

    The four inputs move to the host in one bulk transfer up front and
    every per-class partition below runs on CPU tensors: the previous
    GPU-tensor loop (``class_ids.unique()``, boolean indexing and
    ``nonzero`` per class) paid ~6 pipeline syncs per class — hundreds of
    syncs per training step at VisDrone query counts.
    """
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    device = boxes.device
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    class_cpu = class_ids.detach().cpu()
    thresholds_cpu = iou_thresholds.detach().cpu()

    keep_mask = torch.zeros(boxes.shape[0], dtype=torch.bool)
    for class_id in class_cpu.unique():
        class_mask = class_cpu == class_id
        keep_class = _nms_single_class(
            boxes_cpu[class_mask],
            scores_cpu[class_mask],
            thresholds_cpu[class_mask],
        )
        original_indices = torch.nonzero(class_mask, as_tuple=True)[0]
        keep_mask[original_indices[keep_class]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0].to(device)
