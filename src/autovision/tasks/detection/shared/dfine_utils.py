# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE utilities: distributional box regression + FGL/DDF loss math.

Source lineage: D-FINE ``dfine_utils.py`` (Apache-2.0, ``Peterande/D-FINE`` @ ``956d170``),
re-expressed and extended by AutoVision; provenance ledger entry (P1-3).

D-FINE (Peng et al., "D-FINE: Redefine Regression Task in DETRs as
Fine-grained Distribution Refinement", arXiv:2410.13842) replaces the box
delta head of RT-DETR with a *distributional* head: every decoder layer
predicts, per query and per box side, a categorical distribution over
``reg_max + 1`` bins, and the box is obtained by integrating the
distribution over the **non-uniform weighting function** ``W(n)``
(:func:`weighting_function`). Later layers predict a residual
distribution relative to the previous layer's decoded box — the
fine-grained distribution refinement (FDR) loop.

Distance parameterization (aligned with the official D-FINE/Dome
implementation — ``reg_scale=4`` controls the curvature of ``W``):

- bins encode distances in units of ``w / reg_scale`` (the reference
  box's own width scaled by ``1 / reg_scale``) with a ``0.5 · reg_scale``
  offset, so distance ``0`` reproduces the reference box exactly (the
  zero-initialized head makes refinement start at the reference);
- ``W`` spans ``±2 · up · reg_scale`` units (``up = 0.5`` ⇒ ±4 units
  ⇒ ±16 × w of reach) with geometric spacing — fine resolution near the
  reference (≈ w/50 per bin) and coarse far away, so distant targets keep
  informative gradients instead of saturating an edge bin;
- :func:`distance2bbox` maps (reference, distances) → cxcywh,
  :func:`bbox2distance` is its inverse (FGL target build);
- targets and decode share the same ``W`` — the FGL soft-label
  interpolation (:func:`translate_gt`) and the integration in the decoder
  are exact inverses of each other on representable boxes.

This replaces an earlier linear ``2 · reg_max + 1``-bin parameterization
that clamped at ±1 reference width and stalled FGL on dense tiny-object
benchmarks (VisDrone).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from autovision.geometry.box_ops import (
    clip_normalize_boxes,
    elementwise_box_iou,
)
from autovision.tasks.detection.shared.denoising import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)

__all__ = [
    "LQE",
    "bbox2distance",
    "ddf_loss",
    "distance2bbox",
    "fg_loss",
    "fg_loss_from_targets",
    "flatten_match_indices",
    "integrate_corners",
    "matched_pair_iou",
    "matched_pair_iou_flat",
    "translate_gt",
    "unimodal_distribution_focal_loss",
    "weighting_function",
]

#: Upper-bound parameter of ``W`` — the official models hardcode 0.5
#: (``self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)``).
WEIGHTING_UP = 0.5

#: Curvature of ``W`` — official ``DomeTransformer.reg_scale`` default.
DEFAULT_REG_SCALE = 4.0


def weighting_function(reg_max: int, up: float, reg_scale: float) -> torch.Tensor:
    """Non-uniform weighting function ``W(n)``, ``reg_max + 1`` values.

    ``W`` is geometric away from the center axis ``W(reg_max/2) = 0`` and
    clamped at ``±2 · up · reg_scale`` at both ends. ``reg_scale`` controls
    the curvature (larger = flatter near the center, steeper at the ends);
    ``up`` sets the reach in ``w / reg_scale`` units. Ported 1:1 from the
    official implementation (non-deploy branch).
    """
    reg_scale = abs(reg_scale)
    upper_bound1 = abs(up) * reg_scale
    upper_bound2 = abs(up) * reg_scale * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [0.0]
        + right_values
        + [upper_bound2]
    )
    return torch.tensor(values, dtype=torch.float32)


#: Per-device cache of the weighting function — ``translate_gt`` runs once
#: per matched-pair batch on every FGL site; rebuilding the (reg_max + 1)
#: tensor on CPU and copying it per call paid a pipeline sync each time.
_WEIGHTING_CACHE: dict[tuple[int, float, float, torch.device], torch.Tensor] = {}


def _weighting_function_cached(
    reg_max: int, reg_scale: float, up: float, device: torch.device
) -> torch.Tensor:
    key = (reg_max, float(reg_scale), float(up), device)
    values = _WEIGHTING_CACHE.get(key)
    if values is None:
        values = weighting_function(reg_max, up, reg_scale).to(device)
        _WEIGHTING_CACHE[key] = values
    return values


def translate_gt(
    gt: torch.Tensor, reg_max: int, reg_scale: float, up: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous GT distances → (bin indices, right/left interpolation weights).

    ``gt`` (N,) values in ``w / reg_scale`` units. Returns ``(label,
    weight_right, weight_left)`` each (N,): the target distribution puts
    ``weight_left`` on bin ``label`` and ``weight_right`` on bin
    ``label + 1`` (linear interpolation between adjacent ``W`` values).
    Out-of-range values clamp to the end bins with all mass on the edge.
    Ported 1:1 from the official implementation, then rewritten from
    boolean-mask writes (each one a GPU pipeline sync) to ``torch.where``
    selection — value-identical on every lane, including the historical
    NaN when ``gt`` coincides exactly with a ``W`` knot.
    """
    gt = gt.reshape(-1)
    function_values = _weighting_function_cached(reg_max, reg_scale, up, gt.device)

    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    closest_left_indices = torch.sum(diffs <= 0, dim=1) - 1

    indices = closest_left_indices.float()
    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    invalid_neg = indices < 0
    invalid_pos = indices >= reg_max

    # Gather the bracketing W values for every lane (clamped); invalid
    # lanes compute garbage ratios that the ``where`` selects away — the
    # valid lanes see exactly the old arithmetic.
    gather_idx = indices.clamp(0, reg_max - 1).long()
    left_values = function_values[gather_idx]
    right_values = function_values[(gather_idx + 1).clamp(max=reg_max)]
    left_diffs = (gt - left_values).abs()
    right_diffs = (right_values - gt).abs()
    ratio = left_diffs / (left_diffs + right_diffs)

    weight_right = torch.where(valid_idx_mask, ratio, torch.zeros_like(indices))
    weight_left = torch.where(valid_idx_mask, 1.0 - ratio, torch.ones_like(indices))
    weight_right = torch.where(invalid_pos, torch.ones_like(indices), weight_right)
    weight_left = torch.where(invalid_pos, torch.zeros_like(indices), weight_left)
    indices = torch.where(
        invalid_pos,
        torch.full_like(indices, reg_max - 0.1),
        torch.where(invalid_neg, torch.zeros_like(indices), indices),
    )

    return indices, weight_right, weight_left


def distance2bbox(
    points: torch.Tensor, distance: torch.Tensor, reg_scale: float = DEFAULT_REG_SCALE
) -> torch.Tensor:
    """Reference (…, 4) cxcywh + side distances (…, 4) in ``w / reg_scale``
    units → cxcywh. Zero distances reproduce the reference box exactly."""
    reg_scale = abs(reg_scale)
    cx, cy, w, h = points.unbind(-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (w / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (h / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (w / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (h / reg_scale)
    return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], dim=-1))


def bbox2distance(
    points: torch.Tensor,
    bbox_xyxy: torch.Tensor,
    reg_max: int,
    reg_scale: float = DEFAULT_REG_SCALE,
    up: float = WEIGHTING_UP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`distance2bbox` (FGL target build).

    ``points`` (N, 4) cxcywh reference, ``bbox_xyxy`` (N, 4) absolute xyxy.
    Returns ``(label, weight_right, weight_left)`` each (N, 4) — the
    ``translate_gt`` bin targets per side, in ``w / reg_scale`` units.
    """
    reg_scale = abs(reg_scale)
    px, py, w, h = points.unbind(-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    x1, y1, x2, y2 = bbox_xyxy.unbind(-1)
    left = (px - x1) / (w / reg_scale) - 0.5 * reg_scale
    top = (py - y1) / (h / reg_scale) - 0.5 * reg_scale
    right = (x2 - px) / (w / reg_scale) - 0.5 * reg_scale
    bottom = (y2 - py) / (h / reg_scale) - 0.5 * reg_scale
    distances = torch.stack([left, top, right, bottom], dim=-1)
    label, weight_right, weight_left = translate_gt(
        distances, reg_max, reg_scale, up
    )
    return label.reshape(distances.shape), weight_right.reshape(
        distances.shape
    ), weight_left.reshape(distances.shape)


def integrate_corners(corners: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Integrate per-side distributions into distances.

    ``corners``: (…, 4 × num_bins) raw logits, ``weights``: (num_bins,)
    (the weighting function ``W``). Returns (…, 4) distances in
    ``w / reg_scale`` units.
    """
    logits = corners.reshape(*corners.shape[:-1], 4, weights.shape[-1])
    probs = torch.softmax(logits, dim=-1)
    return (probs * weights).sum(dim=-1)


def unimodal_distribution_focal_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight_right: torch.Tensor,
    weight_left: torch.Tensor,
    weight: torch.Tensor | None = None,
    avg_factor: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Two-bin soft-label cross-entropy (official ``unimodal`` FGL form).

    ``pred`` (N, reg_max + 1) raw logits; ``label`` (N,) float bin indices;
    the target puts ``weight_left`` on ``floor(label)`` and
    ``weight_right`` on ``floor(label) + 1``; ``weight`` (N,) is the
    per-target quality modulation (matched IoU).
    """
    dis_left = label.long()
    dis_right = dis_left + 1

    loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
        -1
    ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

    if weight is not None:
        loss = loss * weight.float()

    if avg_factor is not None:
        loss = loss.sum() / avg_factor
    else:
        loss = loss.sum()
    return loss


def matched_pair_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Row-aligned IoU of matched pairs (normalized cxcywh inputs).

    The predicted side is sanitized first (baseline: degenerate NaN/inf
    boxes yield a clamped IoU target instead of NaN-poisoning the FGL/DDF
    terms into a ``nan_to_num`` zero).
    """
    from torchvision.ops import box_iou

    ious = torch.diag(
        box_iou(
            box_cxcywh_to_xyxy(clip_normalize_boxes(pred_boxes)),
            box_cxcywh_to_xyxy(target_boxes),
        )
    )
    return ious.clamp(min=0.0)


def matched_pair_iou_flat(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> torch.Tensor:
    """Row-aligned IoU without the O(N²) ``box_iou`` matrix (doc 26 W2).

    The ``elementwise_box_iou`` variant on sanitized cxcywh inputs: same
    per-pair arithmetic as the diagonal of the pairwise matrix, O(N)
    memory instead of O(N²). Differs from :func:`matched_pair_iou` only
    in the degenerate both-zero-area case (union floored at 1e-7 → IoU 0
    instead of a NaN the losses' ``nan_to_num`` would zero anyway).
    """
    iou, _ = elementwise_box_iou(
        box_cxcywh_to_xyxy(clip_normalize_boxes(pred_boxes)),
        box_cxcywh_to_xyxy(target_boxes),
    )
    return iou.clamp(min=0.0)


def flatten_match_indices(
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_queries: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-image match pairs into flat batch-space indices.

    Returns ``(flat_q, flat_gt, batch_idx)``: ``flat_q`` indexes the
    ``(B × num_queries, …)`` batch-major view of any per-query tensor,
    ``flat_gt`` indexes each image's ground-truth list, and ``batch_idx``
    repeats the image id once per pair (for gathering batch-major teacher
    tensors at per-image indices). Empty images contribute nothing; with
    no pairs at all three tensors come back empty. (design doc 26 W2)
    """
    srcs: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    batches: list[torch.Tensor] = []
    for batch_index, (src, dst) in enumerate(indices):
        src = src.to(device)
        dst = dst.to(device)
        srcs.append(src + batch_index * num_queries)
        gts.append(dst)
        batches.append(
            torch.full((src.shape[0],), batch_index, dtype=torch.long, device=device)
        )
    if not srcs:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone(), empty.clone()
    return torch.cat(srcs), torch.cat(gts), torch.cat(batches)


class LQE(nn.Module):
    """Layer Quality Estimator (official D-FINE module).

    Calibrates a decoder layer's class logits with a quality scalar read
    off the layer's own corner distributions: the top-``k`` probabilities
    per side plus their mean (``4 × (k + 1)`` features) go through a small
    zero-initialized MLP whose output is ADDED to the scores — at init the
    estimator is a no-op, and it learns to dampen (or boost) confidence
    where the side distributions look unsure. One instance per decoder
    layer; at eval only the ``eval_idx`` layer's estimator is exercised.
    """

    def __init__(self, k: int, hidden_dim: int, num_layers: int, reg_max: int) -> None:
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        from autovision.tasks.detection.shared.rtdetrv2_decoder import (
            MLP,
        )

        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        b, q, _ = pred_corners.size()
        prob = F.softmax(
            pred_corners.reshape(b, q, 4, self.reg_max + 1), dim=-1
        )
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(b, q, -1))
        return scores + quality_score


def fg_loss_from_targets(
    pred_corners: torch.Tensor,
    label: torch.Tensor,
    weight_right: torch.Tensor,
    weight_left: torch.Tensor,
    quality: torch.Tensor,
    avg_factor: torch.Tensor | float,
) -> torch.Tensor:
    """FGL over precomputed targets: (N, 4, bins) logits + (N, 4) targets."""
    flat_pred = pred_corners.reshape(-1, pred_corners.shape[-1])
    flat_quality = quality.unsqueeze(-1).expand(-1, 4).reshape(-1)
    return unimodal_distribution_focal_loss(
        flat_pred,
        label.reshape(-1),
        weight_right.reshape(-1),
        weight_left.reshape(-1),
        weight=flat_quality,
        avg_factor=avg_factor,
    )


def fg_loss(
    corners: torch.Tensor,
    ref_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    reg_max: int,
    reg_scale: float = DEFAULT_REG_SCALE,
    flat_indexing: bool = False,
) -> torch.Tensor:
    """Fine-grained Localization loss (FGL) for one decoder layer.

    Unimodal two-bin cross-entropy over the four side distributions in the
    ``W``-curved space, with IoU-aware weighting: each matched query
    contributes its side CE scaled by the matched IoU (high-quality boxes
    dominate — the paper's quality-aware target). Targets are computed
    relative to THIS layer's reference box (the layer-0 anchor lock of the
    paper falls out of the refinement loop: layer 0's reference is the
    encoder proposal).

    ``corners``: (B, Q, 4 × (reg_max + 1)) raw logits; ``ref_boxes``:
    (B, Q, 4) the reference each layer refined from (detached);
    ``pred_boxes``: (B, Q, 4) this layer's decoded boxes (IoU weighting).

    ``flat_indexing`` (design doc 26 W2) runs the per-image loops as three
    batched calls (one ``bbox2distance``, one IoU, one CE) over flattened
    match indices — same math and normalization, sums reordered.
    """
    up = WEIGHTING_UP
    if flat_indexing:
        device = ref_boxes.device
        flat_q, flat_gt, _ = flatten_match_indices(indices, ref_boxes.shape[1], device)
        denominator = num_boxes.clamp(min=1)
        if flat_q.numel() == 0:
            return corners.sum() * 0.0
        gt_cxcywh = torch.cat(
            [
                targets[batch_index]["boxes"][dst.to(device)]
                for batch_index, (_, dst) in enumerate(indices)
                if dst.numel()
            ]
        )
        reference = ref_boxes.reshape(-1, 4)[flat_q].detach()
        label, weight_right, weight_left = bbox2distance(
            reference, box_cxcywh_to_xyxy(gt_cxcywh), reg_max, reg_scale, up
        )
        layer_logits = corners.reshape(-1, 4 * (reg_max + 1))[flat_q].reshape(
            -1, 4, reg_max + 1
        )
        quality = matched_pair_iou_flat(
            pred_boxes.reshape(-1, 4)[flat_q].detach(), gt_cxcywh
        )
        return fg_loss_from_targets(
            layer_logits, label, weight_right, weight_left, quality,
            avg_factor=denominator,
        )
    loss = corners.sum() * 0.0
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        gt_xyxy = box_cxcywh_to_xyxy(
            targets[batch_index]["boxes"][dst].to(ref_boxes.device)
        )
        reference = ref_boxes[batch_index, src].detach()
        label, weight_right, weight_left = bbox2distance(
            reference, gt_xyxy, reg_max, reg_scale, up
        )
        layer_logits = corners[batch_index, src].reshape(
            src.numel(), 4, reg_max + 1
        )
        quality = matched_pair_iou(
            pred_boxes[batch_index, src].detach(),
            targets[batch_index]["boxes"][dst].to(pred_boxes.device),
        )
        loss = loss + fg_loss_from_targets(
            layer_logits, label, weight_right, weight_left, quality, avg_factor=1.0
        )
    return loss / num_boxes.clamp(min=1)


def ddf_loss(
    student_corners: torch.Tensor,
    teacher_corners: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    reg_max: int,
    targets: list[dict[str, torch.Tensor]] | None = None,
    temperature: float = 5.0,
    counts: tuple[torch.Tensor, torch.Tensor] | None = None,
    flat_indexing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decoupled Distillation Focal loss (DDF) — official full form.

    Distills EVERY query's corner distribution toward the final layer's
    (teacher's) with a temperature-``T`` KL scaled by ``T²``; matched
    queries are weighted by their IoU with the assigned GT box (official
    ``weight_targets_local[idx] = ious`` — pass ``targets``; without it
    the fallback is the teacher box, the pre-parity behavior), unmatched
    queries by the teacher's max class probability. Normalization is the
    official mean form with ``num_pos/num_neg = sqrt(count × 8 / batch)``
    (the counts are computed on the non-dn pass and reused for dn). The
    moving-target amplification that previously required stability brakes
    was traced to a missing zero-init on the anchor head — with the anchor
    fixed, the official form is numerically self-scaling (means over
    bounded per-element KL).

    ``flat_indexing`` (design doc 26 W2) builds the matched weights with
    one flattened gather + row-aligned IoU instead of the per-image loop —
    identical assignments, reordered sums.
    """
    num_bins = reg_max + 1
    b, q = student_boxes.shape[:2]
    device = student_boxes.device

    teacher_probs = torch.softmax(
        teacher_corners.reshape(b, q, 4, num_bins) / temperature, dim=-1
    ).detach()
    student_log_probs = torch.log_softmax(
        student_corners.reshape(b, q, 4, num_bins) / temperature, dim=-1
    )

    # Per-query weights: teacher max class probability everywhere, replaced
    # by the matched IoU on union-assigned queries (official).
    weight_targets_local = teacher_logits.detach().sigmoid().max(dim=-1).values
    mask = torch.zeros(b, q, dtype=torch.bool, device=device)
    if flat_indexing:
        flat_q, flat_gt, _ = flatten_match_indices(indices, q, device)
        if flat_q.numel():
            mask.reshape(-1)[flat_q] = True
            if targets is not None:
                reference_boxes = torch.cat(
                    [
                        targets[batch_index]["boxes"][dst.to(device)]
                        for batch_index, (_, dst) in enumerate(indices)
                        if dst.numel()
                    ]
                )
            else:
                reference_boxes = teacher_boxes.reshape(-1, 4)[flat_q].detach()
            ious = matched_pair_iou_flat(
                student_boxes.reshape(-1, 4)[flat_q].detach(), reference_boxes
            )
            weight_targets_local.reshape(-1)[flat_q] = ious.to(
                weight_targets_local.dtype
            )
    else:
        for batch_index, (src, dst) in enumerate(indices):
            if src.numel() == 0:
                continue
            mask[batch_index, src] = True
            if targets is not None:
                reference_boxes = targets[batch_index]["boxes"][dst].to(device)
            else:
                reference_boxes = teacher_boxes[batch_index, src].detach()
            ious = matched_pair_iou(
                student_boxes[batch_index, src].detach(),
                reference_boxes,
            )
            weight_targets_local[batch_index, src] = ious.to(
                weight_targets_local.dtype
            )
    weight_targets_local = weight_targets_local.detach()
    elem_weight = weight_targets_local.unsqueeze(-1).expand(-1, -1, 4).reshape(-1)

    kl = (
        nn.KLDivLoss(reduction="none")(
            student_log_probs.reshape(-1, num_bins),
            teacher_probs.reshape(-1, num_bins),
        )
    ).sum(-1)
    loss_match_local = elem_weight * (temperature**2) * kl

    # Element-level mask (4 sides per query) for the matched/unmatched split.
    query_mask_flat = mask.reshape(-1)
    elem_mask = query_mask_flat.unsqueeze(-1).expand(-1, 4).reshape(-1)

    # All scalars stay on-device: float()/bool() here would sync the GPU
    # twice per pass (the criterion runs one pass per aux layer, every
    # step). An empty selection normalizes to exactly 0 through the
    # clamped sum (0 / 1), matching the previous `if mask.any()` guard.
    if counts is None:
        batch_size = max(b, 1)
        num_pos = torch.clamp((query_mask_flat.sum() * 8.0 / batch_size) ** 0.5, min=1e-6)
        num_neg = torch.clamp(
            ((~query_mask_flat).sum() * 8.0 / batch_size) ** 0.5, min=1e-6
        )
    else:
        num_pos, num_neg = counts

    # Element-level normalization: 4 sides per query, so the divisor is the
    # element count, not the query count (mean == sum / count).
    matched_elements = elem_mask.sum()
    unmatched_elements = elem_mask.numel() - matched_elements
    m1 = loss_match_local[elem_mask].sum() / matched_elements.clamp(min=1)
    m2 = loss_match_local[~elem_mask].sum() / unmatched_elements.clamp(min=1)
    loss = (m1 * num_pos + m2 * num_neg) / (num_pos + num_neg)
    return loss, num_pos, num_neg
