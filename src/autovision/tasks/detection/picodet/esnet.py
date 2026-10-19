# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ESNet backbone (PP-PicoDet).

Enhanced ShuffleNet: dual-branch ES bottlenecks with the cheap hinge
activation (``min(max(x, 0), 2)``) and channel shuffle. Channel widths
follow the PicoDet-S layout (stem 24, stages 64/128/256/512 at scale 1.0),
scaled by a width multiplier rounded to a multiple of 8. The feature
pyramid for the neck is stages 2-4 at strides (8, 16, 32).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ESNet", "Hinge", "channel_shuffle", "resolve_backbone_scale"]

_STAGE_CHANNELS = (64, 128, 256, 512)

# Model-string scale aliases (PicoDet-XS/S/M/L width multipliers).
BACKBONE_SCALES: dict[str, float] = {"xs": 0.75, "s": 1.0, "m": 1.25, "l": 1.5}


def resolve_backbone_scale(backbone: str) -> float:
    """Resolve the ``picodet/<name>`` model string tail to a width scale.

    Accepts the aliases ``xs``/``s``/``m``/``l`` or a raw float string
    (e.g. ``"1.25"``).
    """
    name = backbone.strip().lower()
    if name in BACKBONE_SCALES:
        return BACKBONE_SCALES[name]
    try:
        return float(name)
    except ValueError as error:
        raise ValueError(
            f"Unknown PicoDet backbone scale '{backbone}'; supported: "
            f"{', '.join(BACKBONE_SCALES)} or a float string."
        ) from error


class Hinge(nn.Module):
    """Hinge activation: ``min(max(x, 0), 2)`` (ESNet paper)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0, max=2.0)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Standard channel shuffle over dim 1."""
    b, c, h, w = x.shape
    return (
        x.view(b, groups, c // groups, h, w)
        .transpose(1, 2)
        .contiguous()
        .view(b, c, h, w)
    )


def _round_channels(channels: int, scale: float) -> int:
    return max(8, int(round(channels * scale / 8)) * 8)


class ConvBNAct(nn.Sequential):
    """Conv + BN (+ hinge). ``act=None`` keeps the conv linear; ``dw=True``
    makes it depthwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        act: str | None = "hinge",
        dw: bool = False,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=kernel // 2,
                groups=in_channels if dw else 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act is not None:
            layers.append(Hinge())
        super().__init__(*layers)


class ESBottleneck(nn.Module):
    """Dual-branch ES bottleneck + channel shuffle.

    stride 1: identity | PW hinge -> DW hinge -> PW linear.
    stride 2: DW stride-2 -> PW | PW -> DW stride-2 -> PW linear.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, kernel: int = 3
    ) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("ES bottleneck requires an even channel count.")
        half = out_channels // 2
        if stride == 1:
            if in_channels != out_channels:
                raise ValueError(
                    "stride-1 ES bottleneck keeps the channel count fixed."
                )
            self.branch1: nn.Module = nn.Identity()
            self.branch2 = nn.Sequential(
                ConvBNAct(half, half, 1),
                ConvBNAct(half, half, kernel, dw=True),
                ConvBNAct(half, half, 1, act=None),
            )
        else:
            self.branch1 = nn.Sequential(
                ConvBNAct(in_channels, in_channels, kernel, stride=2, dw=True),
                ConvBNAct(in_channels, half, 1),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(in_channels, half, 1),
                ConvBNAct(half, half, kernel, stride=2, dw=True),
                ConvBNAct(half, half, 1, act=None),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ESNet(nn.Module):
    """ESNet backbone: /2 stem + four stride-2 ES stages.

    ``forward`` returns the pyramid (c2, c3, c4) at strides (8, 16, 32).
    Each stage is a single downsample bottleneck (minimal PicoDet-S-style
    depth); kernel sizes are configurable per stage.
    """

    def __init__(self, scale: float = 1.0, kernels: tuple[int, ...] = (3, 3, 3, 3)) -> None:
        super().__init__()
        if not 0.1 <= scale <= 2.0:
            raise ValueError(f"backbone scale must be in [0.1, 2.0], got {scale}")
        channels = [_round_channels(c, scale) for c in _STAGE_CHANNELS]
        self.out_channels = tuple(channels[1:])
        stem_channels = _round_channels(24, scale)
        self.conv1 = ConvBNAct(3, stem_channels, 3, stride=2)
        self.stage1 = ESBottleneck(stem_channels, channels[0], stride=2, kernel=kernels[0])
        self.stage2 = ESBottleneck(channels[0], channels[1], stride=2, kernel=kernels[1])
        self.stage3 = ESBottleneck(channels[1], channels[2], stride=2, kernel=kernels[2])
        self.stage4 = ESBottleneck(channels[2], channels[3], stride=2, kernel=kernels[3])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c2, c3, c4
