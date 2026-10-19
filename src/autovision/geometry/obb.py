# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented bounding box (OBB) geometry: conversions, rotated IoU, NMS.

Self-contained torch implementation (no shapely/cv2/mmcv dependency).

Conventions:

- ``xywhr``: normalized-or-absolute (cx, cy, w, h, θ) with θ ∈
  ``[-π/4, π/4]`` measured from the box's *width* axis (the more
  horizontal edge is the width edge); the conversion re-anchors by ±π/2
  and swaps w/h as needed — every re-anchor preserves the box region.
- ``xyxyxyxy``: ordered 4-corner polygon (p1→p2→p3→p4), as produced by
  YOLO-OBB labels.
- Rotated IoU is EXACT: convex-quad × convex-quad intersection computed as
  the shoelace area of the candidate-vertex set (true edge intersections +
  vertices inside the other box) sorted by polar angle; the math runs in
  float64 so near-degenerate boundary cases (e.g. identical boxes) stay
  stable. OBB regression losses of future heads operate in xywhr space.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "obb_nms",
    "polygon_area",
    "probiou_matrix",
    "rotated_iou_matrix",
    "xywhr_to_xyxyxyxy",
    "xyxyxyxy_to_xywhr",
]


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """2D cross product over the last dimension."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def polygon_area(polys: torch.Tensor) -> torch.Tensor:
    """Shoelace area of (..., 4+, 2) polygons (signed, CCW positive)."""
    nxt = torch.roll(polys, -1, dims=-2)
    return _cross(polys, nxt).sum(dim=-1) / 2.0


def _ensure_ccw(polys: torch.Tensor) -> torch.Tensor:
    """Flip vertex order where the winding is clockwise."""
    flipped = torch.flip(polys, dims=[-2])
    return torch.where(
        (polygon_area(polys) < 0).unsqueeze(-1).unsqueeze(-1), flipped, polys
    )


def xywhr_to_xyxyxyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 5) xywhr → (N, 4, 2) corners (winding fixed to CCW)."""
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    u = torch.stack([cos, sin], dim=-1)  # width axis
    v = torch.stack([-sin, cos], dim=-1)  # height axis
    dx, dy = w / 2, h / 2
    corners = torch.stack(
        [
            -u * dx.unsqueeze(-1) - v * dy.unsqueeze(-1),
            u * dx.unsqueeze(-1) - v * dy.unsqueeze(-1),
            u * dx.unsqueeze(-1) + v * dy.unsqueeze(-1),
            -u * dx.unsqueeze(-1) + v * dy.unsqueeze(-1),
        ],
        dim=-2,
    )
    corners = corners + torch.stack([cx, cy], dim=-1).unsqueeze(-2)
    return _ensure_ccw(corners)


def xyxyxyxy_to_xywhr(polys: torch.Tensor) -> torch.Tensor:
    """(N, 4, 2) or (N, 8) ordered corners → (N, 5) xywhr (canonical form).

    The width edge is re-anchored into ``[-π/4, π/4]``: each ±π/2 re-anchor
    (``θ → θ ∓ π/2``) swaps the w/h roles — the resulting xywhr spans the
    same region. w/h are averaged over opposite edges (robust to jitter).
    """
    if polys.shape[-1] == 8:
        polys = polys.reshape(*polys.shape[:-1], 4, 2)
    p1, p2, p3, p4 = polys.unbind(-2)
    center = polys.mean(dim=-2)
    width_edge = (p2 - p1 + p3 - p4) / 2
    height_edge = (p3 - p2 + p4 - p1) / 2
    w = width_edge.norm(dim=-1)
    h = height_edge.norm(dim=-1)
    angle = torch.atan2(width_edge[..., 1], width_edge[..., 0])
    # Re-anchor: n quarter-turns bring the width edge into [-pi/4, pi/4];
    # odd n means the height edge became the width edge (w/h swap).
    n = torch.round(angle / (math.pi / 2))
    angle = angle - n * (math.pi / 2)
    swap = n.remainder(2) != 0
    w, h = torch.where(swap, h, w), torch.where(swap, w, h)
    return torch.stack([center[..., 0], center[..., 1], w, h, angle], dim=-1)


def _inside_flags(pts: torch.Tensor, poly_start: torch.Tensor, poly_end: torch.Tensor) -> torch.Tensor:
    """Whether each point of ``pts`` lies inside (or on) the convex polygon.

    ``pts``: (..., V, 1, 2); ``poly_start``/``poly_end``: (..., 1, E, 2)
    edge endpoints. Returns (..., V) — a point is inside when its signed
    distance to every CCW edge is ≥ a small relative tolerance (float32
    inputs carry ~1e-7 relative noise, so boundary points must not be
    rejected).
    """
    edge = poly_end - poly_start
    dist = _cross(edge, pts - poly_start) / edge.norm(dim=-1).clamp(min=1e-30)
    return (dist >= -1e-6).all(dim=-1)


def rotated_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise rotated IoU between two xywhr sets: (N, 5) × (M, 5) → (N, M).

    Exact convex intersection: the intersection polygon's vertices are all
    among (a) subject-edge × clip-edge segment intersections and (b) the
    vertices of one box lying inside the other — all of them lie on the
    intersection's boundary, so sorting the valid candidates by polar angle
    around their centroid and taking the shoelace area is exact (duplicate
    or collinear candidates contribute zero area). The math runs in float64
    so near-degenerate boundary cases (e.g. duplicate boxes in NMS) are
    stable.
    """
    out_dtype = boxes1.dtype
    c1 = xywhr_to_xyxyxyxy(boxes1.to(torch.float64))  # (N, 4, 2)
    c2 = xywhr_to_xyxyxyxy(boxes2.to(torch.float64))  # (M, 4, 2)
    N, M = c1.shape[0], c2.shape[0]
    K = 24  # 16 edge intersections + 4 + 4 inside vertices

    # Broadcasting layout: subject-edge index i lives on dim 2, clip-edge
    # index j on dim 3, so every pairwise quantity is a true i×j outer
    # grid (aligning both edge axes on the same dim would zip them
    # element-wise instead of pairing all i with all j).
    r1 = torch.roll(c1, -1, dims=-2)
    r2 = torch.roll(c2, -1, dims=-2)
    a_i1 = c1.unsqueeze(1).unsqueeze(3)  # (N, 1, 4i, 1, 2)
    a_i2 = r1.unsqueeze(1).unsqueeze(3)
    b_j1 = c2.unsqueeze(0).unsqueeze(2)  # (1, M, 1, 4j, 2)
    b_j2 = r2.unsqueeze(0).unsqueeze(2)
    a_e1 = c1.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, 4e, 2)
    a_e2 = r1.unsqueeze(1).unsqueeze(2)

    # True segment intersections: subject edge i x clip edge j. Near-parallel
    # edge pairs never cross transversally (the relative sine threshold
    # rejects the phantom mid-segment "crossings" of overlapping collinear
    # edges of nearly-identical boxes).
    d1 = a_i2 - a_i1  # (N, 1, 4, 1, 2)
    d2 = b_j2 - b_j1  # (1, M, 1, 4, 2)
    denom = _cross(d1, d2)  # (N, M, 4, 4)
    scale = d1.norm(dim=-1) * d2.norm(dim=-1)  # (N, M, 4, 4)
    ok = denom.abs() > 1e-6 * scale
    t = _cross(b_j1 - a_i1, d2) / torch.where(ok, denom, torch.ones_like(denom))
    s = _cross(b_j1 - a_i1, d1) / torch.where(ok, denom, torch.ones_like(denom))
    eps = 1e-6  # relative param tolerance, matches float32 input noise
    valid = ok & (t >= -eps) & (t <= 1 + eps) & (s >= -eps) & (s <= 1 + eps)
    inter = torch.where(
        valid.unsqueeze(-1), a_i1 + t.unsqueeze(-1) * d1, torch.zeros_like(a_i1)
    )  # (N, M, 4, 4, 2)

    # Vertices of one box inside the other.
    a_in = _inside_flags(
        c1.unsqueeze(1).unsqueeze(3), b_j1, b_j2
    )  # (N, M, 4)
    b_in = _inside_flags(
        c2.unsqueeze(0).unsqueeze(3), a_e1, a_e2
    )  # (N, M, 4)

    cand = torch.cat(
        [
            inter.reshape(N, M, 16, 2),
            c1.unsqueeze(1).expand(N, M, 4, 2),
            c2.unsqueeze(0).expand(N, M, 4, 2),
        ],
        dim=2,
    )  # (N, M, K, 2)
    mask = torch.cat(
        [valid.reshape(N, M, 16), a_in, b_in], dim=2
    )  # (N, M, K)

    # Compact valid candidates to the front; pad with the last valid one
    # (duplicate vertices contribute zero area).
    cum = torch.cumsum(mask.to(torch.int64), dim=-1)
    slots = torch.arange(1, K + 1, device=cand.device)
    match = cum.unsqueeze(-1) == slots.view(*([1] * cum.dim()), K)
    pos = match.to(torch.int64).argmax(dim=-2)  # (N, M, K)
    compact = cand.gather(2, pos.unsqueeze(-1).expand(N, M, K, 2))
    count = mask.sum(dim=-1)  # (N, M) — 0 means no overlap at all
    count_c = count.clamp(min=1, max=K)
    last = compact.gather(
        2, (count_c - 1).unsqueeze(-1).unsqueeze(-1).expand(N, M, 1, 2)
    )
    has = (slots <= count.unsqueeze(-1)).unsqueeze(-1)
    compact = torch.where(has, compact, last)

    centroid = (compact * has).sum(dim=2) / count_c.unsqueeze(-1)
    rel = compact - centroid.unsqueeze(2)
    order = torch.argsort(torch.atan2(rel[..., 1], rel[..., 0]), dim=-1)
    sorted_pts = compact.gather(2, order.unsqueeze(-1).expand(N, M, K, 2))
    area = polygon_area(sorted_pts).abs()
    area = torch.where(count > 0, area, torch.zeros_like(area))

    area1 = polygon_area(c1).abs().unsqueeze(-1)  # (N, 1)
    area2 = polygon_area(c2).abs().unsqueeze(0)  # (1, M)
    union = (area1 + area2 - area).clamp(min=1e-12)
    return (area / union).clamp(min=0.0, max=1.0).to(out_dtype)


def obb_nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Greedy NMS on xywhr boxes; returns kept indices (score-descending)."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    ious = rotated_iou_matrix(boxes[order], boxes[order])
    keep_mask = torch.ones(order.shape[0], dtype=torch.bool, device=boxes.device)
    for i in range(order.shape[0]):
        if not keep_mask[i]:
            continue
        keep_mask[i + 1:] &= ious[i, i + 1:] <= iou_threshold
    return order[keep_mask]


def probiou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Differentiable rotated-IoU *similarity* between xywhr sets: (N, 5) x
    (M, 5) -> (N, M) in (0, 1].

    Probabilistic IoU (Jiang et al.): each rotated rectangle is modeled by
    its uniform-density second-moment matrix ``Σ = R diag(w²/12, h²/12) Rᵀ``
    and the Gaussian Bhattacharyya coefficient between the two densities
    becomes the similarity ``exp(-B)`` — smooth everywhere (unlike the
    exact polygon IoU, whose vertex sorting kills gradients), equal to 1
    for identical boxes and decaying monotonically with overlap. Used by
    the OBB regression loss and matcher; exact ``rotated_iou_matrix``
    stays the eval/NMS tool.
    """
    dtype = boxes1.dtype

    def _sigma(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w2 = (boxes[..., 2] / 2) ** 2 / 3.0  # (w/2)²/3 == w²/12
        h2 = (boxes[..., 3] / 2) ** 2 / 3.0
        cos = torch.cos(boxes[..., 4])
        sin = torch.sin(boxes[..., 4])
        s11 = w2 * cos * cos + h2 * sin * sin
        s12 = (w2 - h2) * cos * sin
        s22 = w2 * sin * sin + h2 * cos * cos
        return torch.stack([s11, s12, s12, s22], dim=-1).reshape(
            *boxes.shape[:-1], 2, 2
        ), boxes[..., :2]

    sigma1, mu1 = _sigma(boxes1)
    sigma2, mu2 = _sigma(boxes2)
    sigma_bar = (sigma1.unsqueeze(1) + sigma2.unsqueeze(0)) / 2.0
    delta = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # (N, M, 2)

    det1 = torch.det(sigma1).clamp(min=1e-12)
    det2 = torch.det(sigma2).clamp(min=1e-12)
    det_bar = torch.det(sigma_bar).clamp(min=1e-12)
    d1 = delta.unsqueeze(-2) @ torch.linalg.inv(sigma_bar) @ delta.unsqueeze(-1)
    # Bhattacharyya distance between the two Gaussian moment models.
    b = 0.125 * d1.squeeze(-1).squeeze(-1) + 0.5 * torch.log(
        det_bar / (det1.unsqueeze(-1) * det2.unsqueeze(0)).sqrt()
    )
    return torch.exp(-b.clamp(min=0.0)).to(dtype)
