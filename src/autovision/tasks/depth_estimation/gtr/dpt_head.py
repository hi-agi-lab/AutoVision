# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DPT-style monocular metric-depth head (GTR line).

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/depth/gtrdepth.py``,
MIT License, Copyright (c) 2026 The GTR Authors; that file is itself
ported from Depth-Anything-V2's metric-depth DPT
(https://github.com/DepthAnything/Depth-Anything-V2, Apache-2.0) and
adapted to the 3-level GTR encoder pyramid), adapted by AutoVision per
MIT/Apache-2.0 terms (registration cleanup, typing); provenance ledger entry (GTR multi-task S4, roadmap design doc 31
§11).

Module names (``layer_rn``/``refinenet{1,2,3}``/``output_conv{1,2}``) are
kept verbatim so the official ``decoder.*`` checkpoint keys load
unchanged. Decode convention: with ``log_depth=True`` (the released
weights) the final conv emits raw logits and depth = exp(clamp(logit,
-4, 5)) — the Sigmoid variant multiplies a bounded map by ``max_depth``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DPTDepthHead", "FeatureFusionBlock", "ResidualConvUnit"]


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, use_bn: bool = False) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = nn.ReLU(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT RefineNet-style fusion: add skip branch, refine, upsample."""

    def __init__(
        self, features: int, use_bn: bool = False, align_corners: bool = True
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, bias=True)
        self.res_conv_unit1 = ResidualConvUnit(features, use_bn)
        self.res_conv_unit2 = ResidualConvUnit(features, use_bn)

    def forward(self, *xs: torch.Tensor, size=None) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.res_conv_unit1(xs[1])
        output = self.res_conv_unit2(output)
        modifier = {"scale_factor": 2} if size is None else {"size": size}
        output = F.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        return self.out_conv(output)


class DPTDepthHead(nn.Module):
    """Fuse the encoder pyramid [stride 8, 16, 32] into a metric depth map.

    Per-level 3x3 projection → top-down RefineNet fusion (each step
    upsamples 2x, ending at stride 4) → two output convs. ``log_depth``
    selects the unbounded exp decode (see the module docstring).
    """

    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...] = (192, 192, 192),
        features: int = 128,
        use_bn: bool = False,
        max_depth: float = 10.0,
        log_depth: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(
                f"DPTDepthHead expects 3 pyramid levels, got {len(in_channels)}"
            )
        self.max_depth = max_depth
        self.log_depth = log_depth
        self.layer_rn = nn.ModuleList(
            [
                nn.Conv2d(ch, features, kernel_size=3, stride=1, padding=1, bias=False)
                for ch in in_channels
            ]
        )
        self.refinenet3 = FeatureFusionBlock(features, use_bn)  # stride 32 -> 16
        self.refinenet2 = FeatureFusionBlock(features, use_bn)  # stride 16 -> 8
        self.refinenet1 = FeatureFusionBlock(features, use_bn)  # stride 8  -> 4
        self.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1
        )
        out_layers: list[nn.Module] = [
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        ]
        if log_depth:
            # exp(0.182) ~ 1.2 m so early exp() outputs stay well
            # conditioned (the YOLO26 log-depth head init).
            out_layers[-1].bias.data.fill_(0.182)
        else:
            out_layers.append(nn.Sigmoid())
        self.output_conv2 = nn.Sequential(*out_layers)

    def forward(
        self, feats: tuple[torch.Tensor, ...], out_hw: tuple[int, int]
    ) -> torch.Tensor:
        # feats ordered highest resolution first: [stride8, stride16, stride32]
        layer_1, layer_2, layer_3 = [rn(f) for rn, f in zip(self.layer_rn, feats)]
        path_3 = self.refinenet3(layer_3, size=layer_2.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1)
        out = self.output_conv1(path_1)
        out = F.interpolate(out, size=out_hw, mode="bilinear", align_corners=True)
        out = self.output_conv2(out)
        if self.log_depth:
            return torch.exp(out.clamp(-4.0, 5.0)).squeeze(1)
        return out.squeeze(1) * self.max_depth
