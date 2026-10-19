# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CSP-PAN neck (PP-PicoDet).

Lateral 1x1 convs to a shared width, top-down path with nearest upsampling,
bottom-up path with depthwise stride-2 downsampling, CSP blocks at every
level. All three output levels share the same channel width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from autovision.tasks.detection.picodet.esnet import ConvBNAct

__all__ = ["CSPPAN"]


class CSBlock(nn.Module):
    """CSP block: 1x1 mix, channel split, one half through a depthwise
    bottleneck, concat, 1x1 fuse."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 5) -> None:
        super().__init__()
        self.mix = ConvBNAct(in_channels, in_channels, 1)
        half = in_channels // 2
        self.bottleneck = ConvBNAct(half, half, kernel, dw=True)
        self.fuse = ConvBNAct(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mix(x)
        a, b = x.chunk(2, dim=1)
        return self.fuse(torch.cat([a, self.bottleneck(b)], dim=1))


class CSPPAN(nn.Module):
    """Three-level CSP-PAN (strides 8/16/32 inputs -> same-stride outputs)."""

    def __init__(
        self,
        in_channels: tuple[int, ...] | list[int] = (128, 256, 512),
        hidden_channels: int = 96,
        kernel: int = 5,
    ) -> None:
        super().__init__()
        levels = len(in_channels)
        self.lateral = nn.ModuleList(
            [ConvBNAct(c, hidden_channels, 1) for c in in_channels]
        )
        # Separate blocks for the top-down (fpn) and bottom-up (pan) paths.
        self.fpn_blocks = nn.ModuleList(
            [CSBlock(hidden_channels, hidden_channels, kernel) for _ in range(levels)]
        )
        self.pan_blocks = nn.ModuleList(
            [CSBlock(hidden_channels, hidden_channels, kernel) for _ in range(levels)]
        )
        self.downsample = nn.ModuleList(
            [
                ConvBNAct(hidden_channels, hidden_channels, 3, stride=2, dw=True)
                for _ in range(levels - 1)
            ]
        )
        self.out_convs = nn.ModuleList(
            [ConvBNAct(hidden_channels, hidden_channels, 1) for _ in range(levels)]
        )
        self.out_channels = tuple([hidden_channels] * levels)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        lat = [lateral(f) for lateral, f in zip(self.lateral, feats)]
        # Top-down: finest level first, merging coarser levels by upsampling.
        top = [None] * len(lat)
        top[-1] = self.fpn_blocks[-1](lat[-1])
        for i in range(len(lat) - 2, -1, -1):
            merged = lat[i] + F.interpolate(top[i + 1], scale_factor=2.0, mode="nearest")
            top[i] = self.fpn_blocks[i](merged)
        # Bottom-up: finest level passes down through stride-2 convs.
        out = [None] * len(lat)
        out[0] = top[0]
        for i in range(1, len(lat)):
            merged = top[i] + self.downsample[i - 1](out[i - 1])
            out[i] = self.pan_blocks[i](merged)
        return [out_conv(o) for out_conv, o in zip(self.out_convs, out)]
