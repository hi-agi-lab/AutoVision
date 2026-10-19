# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared mosaic geometry: canvas paste clamp + one whole-canvas affine.

Extracted from the detection transform so the keypoint-aware pose mosaic
(roadmap C4) reuses the exact same math — box/keypoint consistency across
families is a contract, not a coincidence. Torch port of the baseline
``_get_affine_matrix`` (inverted=False): the returned 2x3 maps BOX/point
coordinates into the affine-transformed canvas.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "apply_affine_to_boxes",
    "apply_affine_to_keypoints",
    "mosaic_affine_matrix",
    "mosaic_canvas_box_clamp",
]


def mosaic_canvas_box_clamp(
    boxes: torch.Tensor, canvas_size: tuple[int, int]
) -> torch.Tensor:
    """Clamp xyxy boxes to the mosaic canvas (baseline paste semantics)."""
    height, width = canvas_size
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(width))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(height))
    return boxes


def mosaic_affine_matrix(
    canvas_size: tuple[int, int],
    angle: float,
    translate: tuple[int, int],
    scale: float,
) -> torch.Tensor:
    """Pillow-compatible 2x3 affine matrix matching torchvision geometry."""
    rot = math.radians(angle)
    cos, sin = math.cos(rot), math.sin(rot)
    height, width = canvas_size
    cx, cy = width * 0.5, height * 0.5
    tx, ty = translate
    a, b = cos * scale, -sin * scale
    c, d = sin * scale, cos * scale
    tx_out = a * (-cx) + b * (-cy) + cx + tx
    ty_out = c * (-cx) + d * (-cy) + cy + ty
    return torch.tensor(
        [[a, b, tx_out], [c, d, ty_out]], dtype=torch.float64
    )


def apply_affine_to_boxes(
    boxes: torch.Tensor,
    canvas_size: tuple[int, int],
    angle: float,
    translate: tuple[int, int],
    scale: float,
) -> torch.Tensor:
    """Transform xyxy boxes through the mosaic affine (baseline math):
    map the four corners, re-form min/max, clip to the canvas, KEEP
    degenerate boxes (filtered by the caller)."""
    if boxes.numel() == 0:
        return boxes
    matrix = mosaic_affine_matrix(canvas_size, angle, translate, scale)
    corners = boxes.to(torch.float64)[:, [[0, 1], [2, 1], [2, 3], [0, 3]]]
    corners = corners.reshape(-1, 2)
    ones = torch.ones(corners.shape[0], 1, dtype=torch.float64)
    points = torch.cat([corners, ones], dim=1) @ matrix.T
    points = points.reshape(-1, 4, 2)
    mins = points.min(dim=1).values
    maxs = points.max(dim=1).values
    out = torch.cat([mins, maxs], dim=1)
    height, width = canvas_size
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(width))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(height))
    return out.to(torch.float32)


def apply_affine_to_keypoints(
    keypoints: torch.Tensor,
    canvas_size: tuple[int, int],
    angle: float,
    translate: tuple[int, int],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform ``(N, K, 2)`` keypoint coordinates through the mosaic
    affine; returns ``(points, inside)`` where ``inside`` marks points that
    stayed within the canvas (callers unlabel the rest — a cropped-out
    keypoint carries no supervision)."""
    if keypoints.numel() == 0:
        return keypoints, torch.zeros(
            keypoints.shape[:-1], dtype=torch.bool
        )
    matrix = mosaic_affine_matrix(canvas_size, angle, translate, scale)
    flat = keypoints.reshape(-1, 2).to(torch.float64)
    ones = torch.ones(flat.shape[0], 1, dtype=torch.float64)
    points = torch.cat([flat, ones], dim=1) @ matrix.T
    points = points.reshape(keypoints.shape).to(torch.float32)
    height, width = canvas_size
    inside = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] < float(width))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < float(height))
    )
    return points, inside
