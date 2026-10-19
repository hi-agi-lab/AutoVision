# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Rotated-box math for GTR's oriented detection (OBB) line.

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/obb/rbox_ops.py``,
MIT License, Copyright (c) 2026 The GTR Authors), adapted by AutoVision
per MIT sections 2-4 (typing, docstring placement); provenance ledger entry (GTR multi-task S3, roadmap design doc 31
§11).

Conventions (official):
* Normalized rbox ("sigmoid domain"): ``(cx, cy, w, h, a)`` all in [0, 1]
  with ``a = theta / pi`` in the long-edge definition (w the longer side,
  theta in [0, pi)); square inputs keep the geometry isotropic.
* ADR parameterization ("Angle Distribution Refinement"): the external
  axis-aligned rectangle ``(Wr, Hr)`` plus two vertex offsets
  ``(eps, eta)``; 90-degree periodic, computed on the canonical
  ``t in [0, pi/2)`` representative.
"""

from __future__ import annotations

import math

import torch

from autovision.tasks.detection.shared.dfine_utils import translate_gt

__all__ = [
    "HALF_PI",
    "PI",
    "chamfer_cost_pairwise",
    "distance2rbox",
    "extbox_to_rbox",
    "kld_cost_pairwise",
    "kld_loss_paired",
    "rbox2distance",
    "rbox_norm_to_rad",
    "rbox_rad_to_canonical",
    "rbox_rad_to_longedge",
    "rbox_to_corners",
    "rbox_to_extbox",
    "rbox_to_hbox_xyxy",
]

PI = math.pi
HALF_PI = math.pi / 2


def rbox_to_corners(rbox: torch.Tensor) -> torch.Tensor:
    """(..., 5) rbox with theta in radians -> (..., 4, 2) corner points."""
    c = rbox[..., :2]
    w = rbox[..., 2:3]
    h = rbox[..., 3:4]
    t = rbox[..., 4:5]
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    dw = torch.cat([cos_t, sin_t], dim=-1) * w * 0.5
    dh = torch.cat([-sin_t, cos_t], dim=-1) * h * 0.5
    return torch.stack(
        [c + dw + dh, c + dw - dh, c - dw - dh, c - dw + dh], dim=-2
    )


def rbox_norm_to_rad(rbox: torch.Tensor) -> torch.Tensor:
    """Sigmoid-domain rbox (a = theta/pi) -> same tensor with radians."""
    return torch.cat([rbox[..., :4], rbox[..., 4:5] * PI], dim=-1)


def rbox_rad_to_canonical(rbox: torch.Tensor) -> torch.Tensor:
    """Any-angle rbox -> representative with theta in [0, pi/2)."""
    w, h, t = rbox[..., 2], rbox[..., 3], rbox[..., 4]
    t_mod = torch.remainder(t, HALF_PI)
    swap = torch.remainder(torch.floor(t / HALF_PI), 2) == 1
    w_c = torch.where(swap, h, w)
    h_c = torch.where(swap, w, h)
    return torch.stack([rbox[..., 0], rbox[..., 1], w_c, h_c, t_mod], dim=-1)


def rbox_rad_to_longedge(rbox: torch.Tensor) -> torch.Tensor:
    """Any-angle rbox -> long-edge representation: w >= h, theta in [0, pi)."""
    w, h, t = rbox[..., 2], rbox[..., 3], rbox[..., 4]
    swap = w < h
    w_l = torch.where(swap, h, w)
    h_l = torch.where(swap, w, h)
    t_l = torch.remainder(torch.where(swap, t + HALF_PI, t), PI)
    return torch.stack([rbox[..., 0], rbox[..., 1], w_l, h_l, t_l], dim=-1)


def rbox_to_extbox(rbox: torch.Tensor):
    """(..., 5) rbox (radians) -> (cx, cy, Wr, Hr, eps, eta), each (...)."""
    rbox = rbox_rad_to_canonical(rbox)
    w, h, t = rbox[..., 2], rbox[..., 3], rbox[..., 4]
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    Wr = w * cos_t + h * sin_t
    Hr = w * sin_t + h * cos_t
    eps = w * cos_t
    eta = h * cos_t
    return rbox[..., 0], rbox[..., 1], Wr, Hr, eps, eta


def _safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``atan2`` with a finite, zero-gradient value for an undefined direction.

    ADR predictions can collapse both components of a side to zero;
    ``atan2(0, 0)`` is finite in the forward but its backward is NaN, and
    the orientation of a degenerate side is undefined — use angle zero
    until either component becomes non-degenerate.
    """
    degenerate = x.abs() + y.abs() <= eps
    safe_x = torch.where(degenerate, torch.ones_like(x), x)
    safe_y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.atan2(safe_y, safe_x)


def extbox_to_rbox(cx, cy, Wr, Hr, eps, eta) -> torch.Tensor:
    """Inverse of :func:`rbox_to_extbox`; tolerates inconsistent params."""
    eps = eps.clamp(min=0)
    eta = eta.clamp(min=0)
    eps = torch.minimum(eps, Wr.clamp(min=0))
    eta = torch.minimum(eta, Hr.clamp(min=0))
    sw = (Hr - eta).clamp(min=0)  # = w*sin(t)
    sh = (Wr - eps).clamp(min=0)  # = h*sin(t)
    w = torch.sqrt(eps * eps + sw * sw + 1e-16)
    h = torch.sqrt(eta * eta + sh * sh + 1e-16)
    t = 0.5 * (_safe_atan2(sw, eps) + _safe_atan2(sh, eta))
    return torch.stack([cx, cy, w, h, t], dim=-1)


def rbox_to_hbox_xyxy(rbox: torch.Tensor) -> torch.Tensor:
    """(..., 5) rbox -> enclosing axis-aligned box (x1,y1,x2,y2)."""
    cx, cy, Wr, Hr, _, _ = rbox_to_extbox(rbox)
    return torch.stack(
        [cx - Wr / 2, cy - Hr / 2, cx + Wr / 2, cy + Hr / 2], dim=-1
    )


def distance2rbox(
    ref_rbox: torch.Tensor, distance: torch.Tensor, reg_scale
) -> torch.Tensor:
    """Decode 6 integral distances into a refined sigmoid-domain rbox (ADR).

    ``distance``: (..., 6) integral results for
    (left, top, right, bottom, eps, eta).
    """
    reg_scale = abs(reg_scale)
    ref = rbox_norm_to_rad(ref_rbox)
    cx, cy, Wr, Hr, eps0, eta0 = rbox_to_extbox(ref)
    su, sv = Wr / reg_scale, Hr / reg_scale
    x1 = cx - (0.5 * reg_scale + distance[..., 0]) * su
    y1 = cy - (0.5 * reg_scale + distance[..., 1]) * sv
    x2 = cx + (0.5 * reg_scale + distance[..., 2]) * su
    y2 = cy + (0.5 * reg_scale + distance[..., 3]) * sv
    eps = eps0 + distance[..., 4] * su
    eta = eta0 + distance[..., 5] * sv
    W_new = (x2 - x1).clamp(min=0)
    H_new = (y2 - y1).clamp(min=0)
    out = extbox_to_rbox((x1 + x2) / 2, (y1 + y2) / 2, W_new, H_new, eps, eta)
    out = rbox_rad_to_longedge(out)
    return torch.cat([out[..., :4], out[..., 4:5] / PI], dim=-1)


def rbox2distance(
    ref_rbox: torch.Tensor,
    gt_rbox: torch.Tensor,
    reg_max,
    reg_scale,
    up,
    eps_clip: float = 0.1,
):
    """Encode gt rboxes as 6 binned distance targets relative to the reference.

    Returns ``(indices, weight_right, weight_left)``, all flattened to
    ``(N*6,)`` and detached (FGL training targets; S3b).
    """
    reg_scale = abs(reg_scale)
    ref = rbox_norm_to_rad(ref_rbox)
    gt = rbox_norm_to_rad(gt_rbox)
    cx, cy, Wr, Hr, eps0, eta0 = rbox_to_extbox(ref)
    gcx, gcy, gWr, gHr, geps, geta = rbox_to_extbox(gt)
    su = Wr / reg_scale + 1e-16
    sv = Hr / reg_scale + 1e-16
    d0 = (cx - (gcx - gWr / 2)) / su - 0.5 * reg_scale
    d1 = (cy - (gcy - gHr / 2)) / sv - 0.5 * reg_scale
    d2 = ((gcx + gWr / 2) - cx) / su - 0.5 * reg_scale
    d3 = ((gcy + gHr / 2) - cy) / sv - 0.5 * reg_scale
    d4 = (geps - eps0) / su
    d5 = (geta - eta0) / sv
    six = torch.stack([d0, d1, d2, d3, d4, d5], dim=-1)
    six, weight_right, weight_left = translate_gt(six, reg_max, reg_scale, up)
    if reg_max is not None:
        six = six.clamp(min=0, max=reg_max - eps_clip)
    return six.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def xy_wh_r_2_xy_sigma(rbox: torch.Tensor):
    """(..., 5) rbox (radians) -> gaussian (xy (...,2), sigma (...,2,2))."""
    shape = rbox.shape[:-1]
    xy = rbox[..., :2]
    wh = rbox[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = rbox[..., 4].reshape(-1)
    cos_r, sin_r = torch.cos(r), torch.sin(r)
    rotation = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rotation.bmm(scale.square()).bmm(rotation.permute(0, 2, 1))
    return xy, sigma.reshape(shape + (2, 2))


def _kld_distance(xy_p, sigma_p, xy_t, sigma_t, alpha: float = 1.0) -> torch.Tensor:
    """KL divergence D(N_t || N_p) between broadcastable gaussians."""
    det_p = sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] * sigma_p[..., 1, 0]
    det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] * sigma_t[..., 1, 0]
    det_p = det_p.clamp(min=1e-14)
    det_t = det_t.clamp(min=1e-14)
    inv_p_00 = sigma_p[..., 1, 1] / det_p
    inv_p_11 = sigma_p[..., 0, 0] / det_p
    inv_p_01 = -sigma_p[..., 0, 1] / det_p
    dx = xy_p[..., 0] - xy_t[..., 0]
    dy = xy_p[..., 1] - xy_t[..., 1]
    xy_distance = 0.5 * (inv_p_00 * dx * dx + 2 * inv_p_01 * dx * dy + inv_p_11 * dy * dy)
    whr_distance = 0.5 * (
        inv_p_00 * sigma_t[..., 0, 0]
        + 2 * inv_p_01 * sigma_t[..., 0, 1]
        + inv_p_11 * sigma_t[..., 1, 1]
    )
    whr_distance = whr_distance + 0.5 * (det_p.log() - det_t.log()) - 1
    return xy_distance / (alpha * alpha) + whr_distance


def kld_loss_paired(pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Elementwise KLD loss (mmrotate GDLoss kld/log1p): (N,5)x(N,5) -> (N,) in [0,1)."""
    xy_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
    xy_t, sigma_t = xy_wh_r_2_xy_sigma(target)
    distance = _kld_distance(xy_p, sigma_p, xy_t, sigma_t).clamp(min=1e-7)
    return 1 - 1 / (tau + torch.log1p(distance))


def kld_cost_pairwise(pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Pairwise KLD cost: (N,5)x(M,5) -> (N,M) in [0,1)."""
    xy_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
    xy_t, sigma_t = xy_wh_r_2_xy_sigma(target)
    distance = _kld_distance(
        xy_p[:, None], sigma_p[:, None], xy_t[None, :], sigma_t[None, :]
    ).clamp(min=1e-7)
    return 1 - 1 / (tau + torch.log1p(distance))


def chamfer_cost_pairwise(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bidirectional nearest-corner Chamfer cost: (N,5)x(M,5) -> (N,M)."""
    c1 = rbox_to_corners(pred)
    c2 = rbox_to_corners(target)
    dist = torch.norm(c1[:, None, :, None, :] - c2[None, :, None, :, :], dim=-1)
    return dist.min(dim=-1)[0].mean(dim=-1) + dist.min(dim=-2)[0].mean(dim=-1)
