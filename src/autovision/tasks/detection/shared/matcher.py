# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hungarian matcher for DETR-family detectors.

Source lineage: the RT-DETRv2/D-FINE matching-cost design (Apache-2.0,
``Peterande/D-FINE`` @ ``956d170``), re-implemented for this framework;
provenance ledger entry (P1-4). The batched padded cost
construction and the consolidated multi-group solve follow the design
used by RF-DETR's matcher (Apache-2.0) and are implemented independently
here.

Cost (baseline RT-DETRv2): ``2.0 * focal_cls + 5.0 * L1 + 2.0 * GIoU``.
L1 is measured in normalized cxcywh (official); GIoU is evaluated on
normalized **xyxy** boxes, converted here — both reference matchers
(D-FINE official, RF-DETR) convert before the IoU, and the criterion's
own GIoU losses work on xyxy. The cost matrix is guarded with
``nan_to_num`` for AMP stability.

Solver: per-image scipy by default. With the optional ``lsap`` extra
(``torch-linear-assignment``, Apache-2.0) and merged problem batches, one
batched Triton GPU solve replaces the whole step's scipy loop — measured
on the dev 4090 at real COCO target counts (Q=300, gt 0-26): 80 merged
problems in 0.45 ms vs 2.01 ms scipy (4.5x), 0/845 assignment mismatches
(padded target columns carry a 1e4 sentinel, so they can only absorb
otherwise-unused queries — the same construction RF-DETR ships). Any
import/kernel failure falls back to scipy automatically, and the
cross-image pad construction costs padding compute, not semantics.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from autovision.geometry.box_ops import (
    clip_normalize_boxes,
    generalized_box_iou_loss,
)

__all__ = ["HungarianMatcher", "forward_match", "forward_match_many", "indices_to_device"]

logger = logging.getLogger("autovision")

#: Per-image pairwise GIoU over a batch dim (batch-exact with the per-pair
#: ``generalized_box_iou_loss`` — asserted in the matcher tests).
_giou_batched = torch.vmap(generalized_box_iou_loss, in_dims=(0, 0))

#: Cost handed to padded (non-real) target columns on the GPU-solver path
#: so the assignment parks unused queries on them: larger than any real
#: cost (class+L1+GIoU stay far below 1e2 at these weights) and finite so
#: the solver stays well conditioned. On the scipy path padded columns are
#: sliced away before solving and never see this value.
_PADDED_TARGET_COST = 1.0e4

_LSAP: dict[str, object] = {"fn": None, "tried": False}


def _load_lsap():
    """Lazily import the optional batched GPU solver (extra ``lsap``).

    ``AUTOVISION_LSAP`` (see ``runtime/env.py``): "auto" (default) uses
    the solver when importable; "scipy"/"0"/"off" force the scipy loop.
    Failures are sticky — one attempt per process, then scipy forever.
    """
    if _LSAP["tried"]:
        return _LSAP["fn"]
    _LSAP["tried"] = True
    from autovision.runtime.env import Env

    if str(Env.LSAP.value).lower() in {"0", "off", "false", "scipy"}:
        return None
    try:
        from torch_linear_assignment import batch_linear_assignment

        _LSAP["fn"] = batch_linear_assignment
    except Exception as error:  # pragma: no cover - depends on the extra
        logger.info("Batched GPU LSAP unavailable, using scipy: %s", error)
        _LSAP["fn"] = None
    return _LSAP["fn"]


def _cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    """Normalized cxcywh -> normalized xyxy (scale-free, matching the
    official matchers' ``box_cxcywh_to_xyxy`` convention)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _empty_indices(device: torch.device | None = None):
    return (
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
    )


@torch.no_grad()
def _build_cost(
    matcher: "HungarianMatcher",
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor | None, list[int]]:
    """Batched (B, Q, Nmax) cost matrix and per-image target counts.

    Returns ``(None, counts)`` when the batch has no targets at all.
    """
    batch_logits = outputs["pred_logits"]
    batch_boxes = outputs["pred_boxes"]
    # Baseline guard: NaN/inf → 0 and clamp to the unit box before costs.
    batch_boxes = clip_normalize_boxes(batch_boxes)
    batch_size, num_queries = batch_logits.shape[0], batch_logits.shape[1]
    device = batch_logits.device

    num_targets = [int(t["boxes"].shape[0]) for t in targets]
    max_targets = max(num_targets, default=0)
    if max_targets == 0:
        return None, num_targets

    # Padded target buffers: filled rows carry real targets, pad rows are
    # excluded from the solve (per-image slices / a sentinel column cost).
    tgt_boxes = torch.zeros(
        (batch_size, max_targets, 4), dtype=batch_boxes.dtype, device=device
    )
    tgt_labels = torch.zeros((batch_size, max_targets), dtype=torch.int64, device=device)
    for index, target in enumerate(targets):
        count = num_targets[index]
        if count:
            tgt_boxes[index, :count] = target["boxes"].to(
                device=device, dtype=batch_boxes.dtype
            )
            tgt_labels[index, :count] = target["labels"].to(device=device)

    # Focal-style classification cost (baseline formula: the alpha/gamma
    # powers each scale a -log term — dropping the logs, as an earlier
    # revision did, changes the Hungarian assignment).
    pred_scores = batch_logits.sigmoid()
    tgt_scores = pred_scores.gather(
        2, tgt_labels.unsqueeze(1).expand(-1, num_queries, -1)
    )
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
    class_cost = pos_cost - neg_cost  # (B, Q, Nmax)

    # L1 in cxcywh (official) + GIoU on normalized xyxy (official).
    # Elementwise abs-sum instead of torch.cdist(p=1): measured 6-12x
    # faster at matcher shapes (0.03 vs 0.19-0.37 ms per call on the 4090)
    # and numerically equal up to fp32 rounding (allclose-asserted in the
    # matcher tests) — same substitution GTR/D-FINE ship as
    # ``GTR_FAST_L1_COST`` (default on, "much faster than torch.cdist(p=1)
    # on CUDA").
    l1_cost = (
        batch_boxes[:, :, None, :] - tgt_boxes[:, None, :, :]
    ).abs().sum(-1)  # (B, Q, Nmax)
    giou_cost = _giou_batched(
        _cxcywh_to_xyxy_norm(batch_boxes), _cxcywh_to_xyxy_norm(tgt_boxes)
    )

    cost = (
        matcher.cost_class * class_cost
        + matcher.cost_bbox * l1_cost
        + matcher.cost_giou * giou_cost
    )
    mask_cost = _mask_cost(matcher, outputs, targets, num_targets, max_targets, device)
    if mask_cost is not None:
        cost = cost + mask_cost
    return torch.nan_to_num(cost, nan=1.0), num_targets


# ── mask costs (GTR seg lineage, design doc 31 §11 S2) ──────────────────
# Point-sampled pairwise CE/Dice, re-ported from GTR's engine/gtr/box_ops.py
# (MIT; those helpers are RF-DETR's, Apache-2.0) — provenance ledger entry
# provenance ledger entry. Only active when the matcher was constructed
# with mask costs and the outputs/targets carry masks.
def _batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def _batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    return loss / hw


def _mask_cost(
    matcher: "HungarianMatcher", outputs, targets, num_targets, max_targets, device
):
    """(B, Q, Nmax) point-sampled mask CE+Dice cost, or ``None``.

    Official semantics (GTR matcher): one shared random point grid per
    call at ``h*w // mask_point_sample_ratio`` points; predictions sampled
    bilinearly, target masks nearest; pairwise costs normalized per pair.
    Target masks are resized (nearest) to the prediction mask grid so a
    batch of mixed-resolution masks stays one batched op — per-image
    slices in the solve make the extra off-diagonal pairs irrelevant.
    """
    if not (matcher.cost_mask_ce or matcher.cost_mask_dice):
        return None
    if matcher.mask_point_sample_ratio is None:
        return None
    out_masks = outputs.get("pred_masks")
    if out_masks is None or not targets:
        return None
    if "masks" not in targets[0] or targets[0]["masks"] is None:
        return None

    from autovision.tasks.detection.shared.gtr.segmentation_head import point_sample

    batch_size, num_queries = out_masks.shape[0], out_masks.shape[1]
    mask_h, mask_w = out_masks.shape[-2:]
    num_points = max(mask_h * mask_w // matcher.mask_point_sample_ratio, 1)

    tgt_masks = torch.zeros(
        (batch_size, max_targets, mask_h, mask_w),
        dtype=out_masks.dtype,
        device=device,
    )
    for index, target in enumerate(targets):
        count = num_targets[index]
        if not count:
            continue
        masks = target["masks"].to(device=device, dtype=out_masks.dtype)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if tuple(masks.shape[-2:]) != (mask_h, mask_w):
            masks = F.interpolate(
                masks.unsqueeze(1), size=(mask_h, mask_w), mode="nearest"
            ).squeeze(1)
        tgt_masks[index, :count] = masks

    flat_pred = out_masks.flatten(0, 1)
    flat_tgt = tgt_masks.flatten(0, 1)
    point_coords = torch.rand(1, num_points, 2, device=device)
    pred_points = point_sample(
        flat_pred.unsqueeze(1), point_coords.repeat(flat_pred.shape[0], 1, 1),
        align_corners=False,
    ).squeeze(1)
    tgt_points = point_sample(
        flat_tgt.unsqueeze(1), point_coords.repeat(flat_tgt.shape[0], 1, 1),
        align_corners=False, mode="nearest",
    ).squeeze(1)

    ce = _batch_sigmoid_ce_loss(pred_points, tgt_points)
    dice = _batch_dice_loss(pred_points, tgt_points)
    # Diagonal blocks only: (B*Q, B*N) -> (B, Q, B, N) -> per-image (B, Q, N).
    ce = ce.view(batch_size, num_queries, batch_size, max_targets)
    dice = dice.view(batch_size, num_queries, batch_size, max_targets)
    diagonal = torch.arange(batch_size, device=device)
    return (
        matcher.cost_mask_ce * ce[diagonal, :, diagonal, :]
        + matcher.cost_mask_dice * dice[diagonal, :, diagonal, :]
    )


def _scipy_solve(cost_cpu: torch.Tensor, count: int):
    rows, cols = linear_sum_assignment(cost_cpu[:, :count].numpy())
    return (
        torch.as_tensor(rows, dtype=torch.int64),
        torch.as_tensor(cols, dtype=torch.int64),
    )


@torch.no_grad()
def forward_match_many(
    matcher: "HungarianMatcher",
    groups: list[tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]],
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Solve every matcher problem of one training step in one pass.

    ``groups`` is a list of ``(outputs, targets)`` pairs (the criterion's
    decoder layers, pre-output and encoder proposals). All cost matrices
    are built batched (see :func:`_build_cost`), stacked, and solved once:

    - scipy path: ONE host transfer for the whole step, then the per-image
      solves on exact slices (identical assignments to per-call scipy);
    - GPU path (optional ``lsap`` extra, ``AUTOVISION_LSAP`` not "scipy"):
      no host transfer at all — padded target columns carry a 1e4 sentinel
      so the batched solve cannot perturb real-column assignments, and the
      index tensors come back on-device (ascending rows, scipy layout).
      Guard: falls back to scipy when any image has more targets than
      queries (the batched kernel requires full column coverage).

    Per-group results keep the ``forward_match`` contract:
    ``result[g][i] = (query_indices, target_indices)`` per image, with the
    tensors on the accelerator for the GPU path and on the host otherwise
    (callers hoist once via :func:`indices_to_device` either way).
    """
    if not groups:
        return []
    batch_sizes = {outputs["pred_logits"].shape[0] for outputs, _ in groups}
    if len(batch_sizes) != 1:
        raise ValueError(
            f"forward_match_many expects one batch size across groups, got {batch_sizes}."
        )
    batch_size = batch_sizes.pop()

    built: list[tuple[torch.Tensor | None, list[int]]] = [
        _build_cost(matcher, outputs, targets) for outputs, targets in groups
    ]
    results: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = [
        None
    ] * len(built)
    non_empty = [index for index, (cost, _) in enumerate(built) if cost is not None]
    if not non_empty:
        return [
            [_empty_indices() for _ in range(batch_size)] for _ in range(len(built))
        ]
    for index, (cost, counts) in enumerate(built):
        if cost is None:
            results[index] = [_empty_indices() for _ in range(batch_size)]

    shapes = {built[index][0].shape[1:] for index in non_empty}
    device = built[non_empty[0]][0].device
    solver = None
    if len(shapes) == 1:
        solver = _load_lsap()
    if solver is not None:
        num_queries, max_targets = built[non_empty[0]][0].shape[1:]
        counts_fit = all(
            count <= num_queries for index in non_empty for count in built[index][1]
        )
        if not counts_fit:
            solver = None

    if solver is not None:
        try:
            stacked = torch.stack([built[index][0] for index in non_empty])
            flat = stacked.reshape(len(non_empty) * batch_size, *stacked.shape[2:])
            counts = built[non_empty[0]][1]
            mask = torch.zeros(
                (len(non_empty) * batch_size, max_targets),
                dtype=torch.bool,
                device=device,
            )
            for image_index, count in enumerate(counts):
                # Rows are group-major (g * B + i): one strided write per image.
                mask[image_index::batch_size, count:] = True
            flat = flat.masked_fill(mask.unsqueeze(1), _PADDED_TARGET_COST)
            assignments = solver(flat.contiguous())
            for position, index in enumerate(non_empty):
                group_results = []
                for image_index, count in enumerate(counts):
                    if count == 0:
                        group_results.append(_empty_indices(device))
                        continue
                    row = assignments[position * batch_size + image_index]
                    keep = (row >= 0) & (row < count)
                    group_results.append(
                        (
                            torch.nonzero(keep, as_tuple=False).squeeze(1).to(torch.int64),
                            row[keep].to(torch.int64),
                        )
                    )
                results[index] = group_results
            return results  # type: ignore[return-value]
        except Exception as error:  # pragma: no cover - kernel/driver issues
            logger.warning(
                "Batched GPU LSAP failed (%s); falling back to scipy for this step.",
                error,
            )

    # scipy path: one host transfer for all groups of this step.
    if len(shapes) == 1:
        stacked = torch.stack([built[index][0] for index in non_empty])
        flat_cpu = stacked.reshape(len(non_empty) * batch_size, *stacked.shape[2:]).cpu()
        counts = built[non_empty[0]][1]
        for position, index in enumerate(non_empty):
            group_results = []
            for image_index, count in enumerate(counts):
                if count == 0:
                    group_results.append(_empty_indices())
                    continue
                group_results.append(
                    _scipy_solve(flat_cpu[position * batch_size + image_index], count)
                )
            results[index] = group_results
        return results  # type: ignore[return-value]

    # Heterogeneous group shapes (rare): per-group scipy with its own transfer.
    for index in non_empty:
        cost, counts = built[index]
        cost_cpu = cost.cpu()
        results[index] = [
            _empty_indices() if count == 0 else _scipy_solve(cost_cpu[i], count)
            for i, count in enumerate(counts)
        ]
    return results  # type: ignore[return-value]


@torch.no_grad()
def forward_match(
    matcher: "HungarianMatcher",
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute one-to-one assignments for a single output group.

    Convenience wrapper over :func:`forward_match_many`; the per-image
    semantics are documented there (padding, sentinel, solver selection).
    """
    return forward_match_many(matcher, [(outputs, targets)])[0]


def indices_to_device(
    indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Move per-image match indices to *device* (host→device, once).

    The scipy path returns CPU tensors; the GPU path returns on-device
    tensors already (this becomes a no-op). Downstream losses index CUDA
    tensors with them — every such advanced-indexing op would otherwise
    pay a synchronous host→device copy per consumer.
    """
    return [
        (src.to(device, non_blocking=True), dst.to(device, non_blocking=True))
        for src, dst in indices
    ]


class HungarianMatcher(nn.Module):
    """One-to-one query-to-ground-truth assignment (DETR lineage)."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        cost_mask_ce: float = 0.0,
        cost_mask_dice: float = 0.0,
        mask_point_sample_ratio: int | None = None,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        #: Mask costs (GTR seg lineage): active only with a sampling ratio
        #: and non-zero weights; ``outputs['pred_masks']`` + target
        #: ``masks`` must be present or they are silently skipped.
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self.mask_point_sample_ratio = mask_point_sample_ratio

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return forward_match(self, outputs, targets)
