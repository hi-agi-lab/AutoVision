# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DeFE density head: Density-Focal Extractor (Dome-DETR port, doc 13 §4.1).

Port of the baseline ``src/zoo/dome/defe.py`` (Dome-DETR, ACM MM 2025,
Apache-2.0), with two deliberate deviations (design doc 13 §5):

- D1: the head consumes the encoder's *stride-8* projected feature (the
  AV-DETR pyramid has no stride-4 level) and emits the density map at
  *stride 2* — the module's net 4x upsampling after the internal 2x pool
  is preserved;
- D2: the channel width is parameterized (the baseline hardcodes 256)
  because AV-DETR hidden dims vary per variant.

Outputs per image:

- ``density`` (1, H/2, W/2): object-density map in [0, 1], per-image max
  normalized. Inference-time consumers: MWAS window gating and the
  density-modulated NMS thresholds (both read window-pooled copies).
- ``reg_value`` (B, 1): normalized object-count estimate in [0, 1],
  supervised against the GT count squeezed into
  [min_num_select, max_num_select]. Auxiliary-training only — the query
  gating itself reads the density map, not this scalar.

Supervision primitives:

- :class:`GaussHeatmapGenerator`: per-box anisotropic Gaussian kernels
  (sigma = ``sigma_ratio`` x box wh) summed into one heatmap, whole-map
  max normalized. Vectorized in chunks over boxes (the baseline loops in
  Python); kernels are truncated to +/-3 sigma and sum-normalized per
  box exactly like the baseline's cropped kernels.
- :func:`defe_density_loss`: MSE that penalizes underestimation of
  high-density regions extra (recall-oriented).
- :func:`defe_count_loss`: asymmetric MSE, underestimation penalized
  more (predicting too few objects costs recall; too many only costs
  precision).

Both losses and the heatmap run in fp32 explicitly: the engine wraps the
whole training step in bf16 autocast, and this is one of the designated
fp32 islands (design doc 13 §4.9 / D6).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DeFETrunk",
    "DepthwiseSeparableConv",
    "GaussHeatmapGenerator",
    "LiteDeFE",
    "LightweightAttention",
    "defe_count_loss",
    "defe_density_loss",
]


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise (dilated) + 1x1 pointwise + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)))


class LightweightAttention(nn.Module):
    """Squeeze-and-excitation channel attention (reduction 8)."""

    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        att = self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)
        return x * att.expand_as(x)


class DeFETrunk(nn.Module):
    """Five dilated depthwise-separable convs with attention after the third.

    Dilation pattern (1, 2, 3, 1, 1) gives the cheap multi-scale receptive
    field the density map needs; channels stay constant throughout.
    """

    _DILATIONS = (1, 2, 3, 1, 1)

    def __init__(self, channels: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for index, dilation in enumerate(self._DILATIONS):
            blocks.append(DepthwiseSeparableConv(channels, channels, dilation))
            blocks.append(nn.BatchNorm2d(channels))
            if index == 2:
                blocks.append(LightweightAttention(channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LiteDeFE(nn.Module):
    """Density-Focal Extractor: stride-8 feature in, stride-2 density out.

    Layout (net 4x upsampling after the internal 2x pool, per the
    baseline): 1x1 conv + AvgPool(2) -> :class:`DeFETrunk` -> density head
    (3x3, 4x upsample, 1x1, sigmoid, 2x upsample) and a GAP count head.
    The density map is max-normalized per image (max > 0 guard for empty
    scenes); ``reg_value`` is a sigmoid scalar in [0, 1].
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
        )
        self.trunk = DeFETrunk(channels)
        self.density_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(features))
        # fp32 from here on: the density map feeds normalization, gating,
        # NMS thresholds and the losses (fp32 island, design doc 13 D6).
        density = F.interpolate(
            self.density_head(x), scale_factor=2, mode="bilinear",
            align_corners=False,
        ).float()
        max_value = density.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        density = density / max_value
        reg_value = self.count_head(x).float()
        return density, reg_value


class GaussHeatmapGenerator:
    """Sum of per-box Gaussian kernels as the density-map ground truth.

    Baseline semantics (Dome ``defe.py`` GaussHeatmapGenerator): for each
    GT box (normalized cxcywh) place an anisotropic Gaussian centered at
    the box center with sigma = ``sigma_ratio`` x (box w, box h) in
    pixels, truncated to +/-3 sigma, per-kernel sum-normalized, kernels
    accumulated by addition, and the whole map max-normalized at the end.

    Implementation note: vectorized in chunks of boxes over a full-map
    distance grid instead of the baseline's Python per-kernel paste loop
    (same math, one GPU op per chunk). ``boxes`` is an (N, 4) normalized
    cxcywh tensor on any device; the result is fp32 on the same device.
    """

    _CHUNK = 64

    def __init__(self, img_size: Sequence[int], sigma_ratio: float = 1.2) -> None:
        self.height = int(img_size[0])
        self.width = int(img_size[1])
        self.sigma_ratio = float(sigma_ratio)

    def __call__(self, boxes: torch.Tensor) -> torch.Tensor:
        heatmap = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=boxes.device
        )
        if boxes.numel() == 0:
            return heatmap.unsqueeze(0)

        boxes = boxes.detach().float()
        cx = boxes[:, 0] * self.width
        cy = boxes[:, 1] * self.height
        # sigma >= 1 px per the baseline floor; 3-sigma support crop.
        sigma_x = (boxes[:, 2] * self.width * self.sigma_ratio).clamp(min=1.0)
        sigma_y = (boxes[:, 3] * self.height * self.sigma_ratio).clamp(min=1.0)

        ys = torch.arange(self.height, device=boxes.device, dtype=torch.float32)
        xs = torch.arange(self.width, device=boxes.device, dtype=torch.float32)
        # Explicit reshapes: a bare (H,) vector would broadcast onto the
        # LAST axis, silently collapsing the map to a 1-D profile.
        grid_y = (ys.view(1, -1, 1) - cy.view(-1, 1, 1)) ** 2  # (N, H, 1)
        grid_x = (xs.view(1, 1, -1) - cx.view(-1, 1, 1)) ** 2  # (N, 1, W)

        for start in range(0, boxes.shape[0], self._CHUNK):
            stop = min(start + self._CHUNK, boxes.shape[0])
            gy = grid_y[start:stop] / (2 * sigma_y[start:stop].view(-1, 1, 1) ** 2)
            gx = grid_x[start:stop] / (2 * sigma_x[start:stop].view(-1, 1, 1) ** 2)
            kernel = torch.exp(-(gx + gy))
            support = (gx <= 4.5) & (gy <= 4.5)  # |dx| <= 3 sigma, |dy| <= 3 sigma
            kernel = kernel * support
            kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            heatmap += kernel.sum(dim=0)

        # Branch-free guard: `if peak > 0` would sync the GPU every step;
        # boxes.numel() > 0 implies peak > 0, and the clamp is a no-op then.
        return (heatmap / heatmap.max().clamp(min=1e-6)).unsqueeze(0)


def defe_density_loss(
    density: torch.Tensor,
    gt_heatmap: torch.Tensor,
    *,
    weight: float = 1.0,
    underestimation_penalty: float = 0.3,
) -> torch.Tensor:
    """Weighted MSE between predicted and GT density maps.

    Underestimation of high-GT-density regions is penalized extra
    (``1 + penalty * gt * 1[pred < gt]``) — the downstream query gating
    fails recall when density is missed, while over-coverage is cheap.
    """
    pred = density.float()
    target = gt_heatmap.float().to(pred.device)
    diff = pred - target
    underestimation = (pred < target).float()
    penalty = 1 + underestimation_penalty * target * underestimation
    return (penalty * diff.pow(2)).mean() * float(weight)


def defe_count_loss(
    reg_value: torch.Tensor,
    gt_counts: Sequence[int],
    *,
    min_num_select: int,
    max_num_select: int,
    underestimation_weight: float = 2.0,
) -> torch.Tensor:
    """Asymmetric MSE on the normalized object-count estimate.

    Target per image: ``(clamp(n_gt, min, max) - min) / (max - min)``,
    matching the quantity the dynamic query selector cares about. Note
    the baseline computes the same target but stores it into an int64
    tensor, truncating it to 0 for nearly every image (upstream bug); the
    fp32 target here is the intended paper formula (design doc 13 §5/D11).

    ``reg_value``: (B, 1) or (B,) sigmoid outputs; ``gt_counts``: per-image
    GT counts.
    """
    if len(gt_counts) == 0:
        return reg_value.sum() * 0.0
    span = max(max_num_select - min_num_select, 1)
    target = torch.tensor(
        [
            (min(max(int(n), min_num_select), max_num_select) - min_num_select) / span
            for n in gt_counts
        ],
        dtype=torch.float32,
        device=reg_value.device,
    )
    pred = reg_value.float().reshape(-1)
    diff = pred - target
    penalty = torch.where(diff < 0, underestimation_weight, 1.0)
    return (penalty * diff.pow(2)).mean()


def density_heatmap_size(image_size: Sequence[int]) -> tuple[int, int]:
    """Density-map spatial size for a given network input size.

    The head nets out to stride 2 (design doc 13 D1): (H/2, W/2).
    Exposed so the encoder and tests agree on the resolution without
    round-tripping a forward pass.
    """
    return (-(-int(image_size[0]) // 2), -(-int(image_size[1]) // 2))
