# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""FDConv — Fourier-Disjoint Weight convolution (FSDC-DETR paper §3.1).

DEVIATION (G4-class, design doc 22 §4.4): the official FDConv sources
(github.com/Linwei-Chen/FDConv, CVPR 2025) were unreachable from this
environment at implementation time, so this is a *semantics* port of the
two components the FSDC-DETR paper relies on, not a line-faithful one:

- **FDW (Fourier-Disjoint Weight)** — the k×k kernel positions are
  partitioned into ``num_bands`` disjoint bands by circular-frequency
  radius (a position's frequency is its distance to the spectrum corner
  (0,0) under circular shift — the DFT convention). Each band owns a
  fixed 0/1 mask; the learnable kernel is band-masked so every band's
  sub-kernel lives in its own frequency range ("disjoint"), and the
  effective conv kernel is the FBL-weighted sum over bands.
- **FBL (Frequency Band Learning)** — one learnable per-band gate
  (sigmoid-initialized open) allocating amplitude across the bands.
- **KSM (Kernel Spatial Modulation)** — optional per-kernel-position
  spatial gate (``use_ksm``; off by default, doc 22 §4.4).

``fdconv_lite=True`` is the documented negative-control arm (plain 3×3
conv) that isolates "frequency modeling" from raw capacity in ablations.
The allclose test against a vendored official snapshot is deliberately
NOT implemented here (no snapshot available); replace this module with
the official port when the snapshot can be vendored — the class surface
(in_channels, out_channels, forward) is the contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FDConv", "band_masks"]


def band_masks(kernel_size: int, num_bands: int) -> Tensor:
    """Disjoint per-band masks over the k×k kernel grid (G, k, k) float.

    Positions are ranked by circular-frequency radius
    ``min(u, k-u)² + min(v, k-v)²`` (distance from DC), then split into
    ``num_bands`` equal bands — band 0 holds the low-frequency positions
    (including DC), the last band the highest frequencies.
    """
    positions = torch.arange(kernel_size)
    grid_u, grid_v = torch.meshgrid(positions, positions, indexing="ij")
    radius = torch.minimum(grid_u, kernel_size - grid_u) ** 2 + torch.minimum(
        grid_v, kernel_size - grid_v
    ) ** 2
    order = radius.argsort().argsort()  # rank 0..k²-1, 0 = lowest frequency
    band_index = (order * num_bands) // (kernel_size * kernel_size)
    return F.one_hot(band_index, num_bands).permute(2, 0, 1).float()


class FDConv(nn.Module):
    """Frequency-band-disjoint convolution (semantics port, see module doc)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_bands: int = 4,
        use_ksm: bool = False,
        lite: bool = False,
    ) -> None:
        super().__init__()
        self.lite = lite
        if lite:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, 1, kernel_size // 2
            )
            return
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        masks = band_masks(kernel_size, num_bands)  # (G, k, k)
        self.register_buffer("band_masks", masks)
        # One shared kernel whose band-disjoint sub-kernels are the masked
        # views — a single conv's worth of parameters plus G band gates.
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.band_gate = nn.Parameter(torch.ones(num_bands))
        self.use_ksm = use_ksm
        if use_ksm:
            self.ksm = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))

    def effective_kernel(self) -> Tensor:
        """FBL-weighted sum of the band-disjoint sub-kernels (out, in, k, k)."""
        g = self.band_masks.shape[0]
        masked = self.weight.unsqueeze(0) * self.band_masks.view(
            g, 1, 1, self.kernel_size, self.kernel_size
        )  # (G, out, in, k, k)
        gates = torch.sigmoid(self.band_gate).view(-1, 1, 1, 1, 1)
        kernel = (masked * gates).sum(dim=0)
        if self.use_ksm:
            kernel = kernel * torch.sigmoid(self.ksm)
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        if self.lite:
            return self.conv(x)
        return F.conv2d(x, self.effective_kernel(), padding=self.kernel_size // 2)
