# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PaKA (Patch Kernel Alignment) building blocks.

- Linear CKA loss between student and teacher dense token features
  (PaKA, arXiv 2509.05606): ``1 - CKA`` over the valid tokens of each
  image pair, averaged over the batch.
- ``roi_resample_to_grid``: bilinear resampling of a feature map over
  arbitrary boxes onto a common output grid (grid_sample-based).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["_linear_cka_loss", "_roi_resample_to_grid"]


def _linear_cka_loss(
    student: Tensor,
    teacher: Tensor,
    valid: Tensor,
    max_tokens: int = 512,
) -> Tensor:
    """``mean(1 - linear CKA)`` over per-image student/teacher token pairs.

    Args:
        student: ``(M, N, D)`` student tokens (grad flows through this).
        teacher: ``(M, N, D)`` teacher tokens (expected detached).
        valid: ``(M, N)`` bool mask; pairs with fewer than 2 valid tokens
            are skipped entirely.
        max_tokens: per-pair token subsample before the O(N^2) kernel.

    Returns a scalar loss; ``0`` (with a zero grad path) when no pair is
    valid.
    """
    losses: list[Tensor] = []
    for index in range(student.shape[0]):
        mask = valid[index]
        if int(mask.sum()) < 2:
            continue
        s = student[index][mask]
        t = teacher[index][mask]
        count = s.shape[0]
        if count > max_tokens:
            pick = torch.randperm(count, device=s.device)[:max_tokens]
            s = s[pick]
            t = t[pick]
        s = s - s.mean(dim=0, keepdim=True)
        t = t - t.mean(dim=0, keepdim=True)
        cross = (s.transpose(0, 1) @ t).norm()
        cka = cross**2 / (
            (s.transpose(0, 1) @ s).norm() * (t.transpose(0, 1) @ t).norm() + 1e-8
        )
        losses.append(1.0 - cka)
    if not losses:
        return student.sum() * 0.0
    return torch.stack(losses).mean()


def _roi_resample_to_grid(
    feat: Tensor, boxes: Tensor, out_h: int, out_w: int
) -> Tensor:
    """Resample ``(B, C, H, W)`` features over ``(B, 4)`` xyxy boxes.

    Boxes are in feature-map pixel coordinates; each box is resampled onto
    an ``(out_h, out_w)`` grid with bilinear interpolation (`align_corners=False`).
    """
    height, width = feat.shape[-2:]
    device = feat.device
    dtype = feat.dtype
    # Sample at the CENTER of each output cell: (i + 0.5) / out maps an
    # identity box onto the input pixel centers (exact identity resample).
    idx = torch.arange(out_w, device=device, dtype=dtype)
    grid_x = (idx + 0.5) / out_w
    idx_y = torch.arange(out_h, device=device, dtype=dtype)
    grid_y = (idx_y + 0.5) / out_h
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    x0 = boxes[:, 0][:, None, None]
    y0 = boxes[:, 1][:, None, None]
    x1 = boxes[:, 2][:, None, None]
    y1 = boxes[:, 3][:, None, None]
    x = x0 + grid_x * (x1 - x0)
    y = y0 + grid_y * (y1 - y0)
    # Boxes use continuous corner semantics ([0, W] spans the full map);
    # align_corners=False maps [-1, 1] to exactly that range.
    grid = torch.stack([2.0 * x / width - 1.0, 2.0 * y / height - 1.0], dim=-1)
    return F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
