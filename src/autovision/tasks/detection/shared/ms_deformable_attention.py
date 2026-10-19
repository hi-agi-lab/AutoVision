# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-scale deformable attention (MSDeformableAttention).

Pure-PyTorch implementation (value-sampling style, ``grid_sample`` based)
compatible with ``torch.export`` traces — no custom CUDA ops.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MSDeformableAttention"]


class MSDeformableAttention(nn.Module):
    """Deformable attention over multi-scale feature maps.

    Args follow the RT-DETRv2 usage: query tokens attend to ``num_points``
    sampled locations per level around a learned reference point per query.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Per-head radial base grid as the offset bias: sampling starts spread
        # around the reference point instead of collapsing onto it.
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # (heads, 2)
        offsets = (
            torch.arange(self.num_points, dtype=torch.float32)[:, None]
            / self.num_points
        )  # (points, 1)
        base = grid[:, None, :] * offsets[None, :, :]  # (heads, points, 2)
        bias = base.unsqueeze(1).expand(-1, self.num_levels, -1, -1).reshape(-1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,  # (B, Q, C)
        reference_points: torch.Tensor,  # (B, Q, num_levels, 2) in [0, 1]
        value: torch.Tensor,  # (B, sum(HW), C)
        spatial_shapes: torch.Tensor,  # (num_levels, 2) as (H, W)
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        device = query.device
        value = self.value_proj(value)
        _, sum_hw, channels = value.shape

        offsets: torch.Tensor = self.sampling_offsets(query).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        # Offsets in normalized units: x relative to level width, y to height.
        level_sizes = spatial_shapes.to(offsets.dtype).to(device)  # (L, 2) (H, W)
        scale = torch.stack([level_sizes[:, 1], level_sizes[:, 0]], dim=-1)
        offsets = offsets / scale[
            None, None, None, :, None, :
        ]  # (1, 1, 1, L, 1, 2)
        sampling_locations = (
            reference_points[:, :, None, :, None, :] + offsets
        )  # (B, Q, heads, L, points, 2)

        attention_weights = self.attention_weights(query).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )
        # Joint softmax over (level, point): cross-level competition.
        attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Per-level value maps and sampling; weighted sum over levels/points.
        # grid_sample requires 4D input/grid, so fold heads into the batch.
        head_channels = channels // self.num_heads
        output = torch.zeros(
            batch_size,
            self.num_heads,
            head_channels,
            num_queries,
            device=device,
            dtype=value.dtype,
        )
        start = 0
        for level in range(self.num_levels):
            height, width = int(spatial_shapes[level][0]), int(spatial_shapes[level][1])
            level_value = value[:, start : start + height * width].reshape(
                batch_size, height, width, self.num_heads, head_channels
            )
            level_value = level_value.permute(0, 3, 4, 1, 2).flatten(0, 1)
            # (B * heads, C', H, W)
            grid = sampling_locations[:, :, :, level]  # (B, Q, heads, points, 2)
            grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B * heads, Q, points, 2)
            grid = grid * 2.0 - 1.0  # normalize to [-1, 1]
            sampled = F.grid_sample(
                level_value,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (B * heads, C', Q, points)
            sampled = sampled.unflatten(0, (batch_size, self.num_heads))
            level_weights = attention_weights[:, :, :, level].permute(
                0, 2, 1, 3
            )  # (B, heads, Q, points)
            output = output + (sampled * level_weights.unsqueeze(2)).sum(dim=-1)
            start += height * width

        output = output.permute(0, 3, 1, 2).reshape(batch_size, num_queries, channels)
        return self.output_proj(output)
