# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Instance-segmentation mask head + point-sampling utilities (GTR line).

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/det/segmentation_head.py``,
MIT License, Copyright (c) 2026 The GTR Authors; that file is itself
"Modified from RF-DETR (https://github.com/roboflow/rf-detr), Apache-2.0";
``point_sample`` / ``get_uncertain_point_coords_with_randomness`` follow
PointRend's detectron2 implementation), adapted by AutoVision per
MIT/Apache-2.0 terms (registration cleanup, typing; the ``export()``
runtime-swap path is dropped — our deploy story lives in ONNX/TRT
exporters, not a Python-side method swap); provenance ledger entry (GTR multi-task S2, roadmap design doc 31
§11).

Contract: ``SegmentationHead(spatial_features, query_features)`` maps
encoder-level-0 features (B, C, H, W) + per-layer decoder query features
[(B, N, C), ...] to per-layer mask logits [(B, N, h, w), ...] at
``image_size / downsample_ratio``, where ``h = w`` fixed by
``image_size`` (the official segmentation finetune runs fixed-shape
inputs; the head's task-specific masked-softmax loss is its sibling in
``criterion``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DepthwiseConvBlock",
    "MLPBlock",
    "SegmentationHead",
    "get_uncertain_point_coords_with_randomness",
    "point_sample",
]


class DepthwiseConvBlock(nn.Module):
    """Simplified ConvNeXt block without the MLP subnet."""

    def __init__(self, dim: int, layer_scale_init_value: float = 0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class MLPBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 0) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + shortcut


class SegmentationHead(nn.Module):
    """Per-query mask logits from spatial features × query embeddings.

    ``mask_logits_l`` = einsum over the layer-``l`` query embedding and the
    spatial features after ``l`` residual depthwise blocks (one block per
    decoder layer — the head shares the decoder's depth).
    """

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        bottleneck_ratio: int = 1,
        downsample_ratio: int = 4,
        image_size: tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )
        self.blocks = nn.ModuleList([DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )
        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Linear(in_dim, self.interaction_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.image_size = tuple(image_size)

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor],
        skip_blocks: bool = False,
    ) -> list[torch.Tensor]:
        # spatial_features: (B, C, H, W); query_features: [(B, N, C)] per
        # decoder layer; output: [(B, N, h, w)] per layer.
        target_size = (
            self.image_size[0] // self.downsample_ratio,
            self.image_size[1] // self.downsample_ratio,
        )
        spatial_features = F.interpolate(
            spatial_features, size=target_size, mode="bilinear", align_corners=False
        )

        mask_logits: list[torch.Tensor] = []
        if skip_blocks:
            assert len(query_features) == 1, (
                "skip_blocks is only supported for length 1 query features"
            )
            qf = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(
                torch.einsum("bchw,bnc->bnhw", spatial_features, qf) + self.bias
            )
        elif len(query_features) == 1:
            # Eval path: run the FULL block stack so the spatial features
            # see the same depth as training's final mask (zip() against a
            # single query feature would only run block[0]).
            for block in self.blocks:
                spatial_features = block(spatial_features)
            spatial_proj = self.spatial_features_proj(spatial_features)
            qf = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(
                torch.einsum("bchw,bnc->bnhw", spatial_proj, qf) + self.bias
            )
        else:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))
                mask_logits.append(
                    torch.einsum("bchw,bnc->bnhw", spatial_proj, qf) + self.bias
                )
        return mask_logits


def point_sample(
    input: torch.Tensor, point_coords: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Bilinear sample of ``input`` at normalized ``[0, 1]`` coords."""
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: int = 3,
    importance_sample_ratio: float = 0.75,
) -> torch.Tensor:
    """PointRend uncertainty sampling (see the PointRend paper).

    Implementation follows detectron2's reference: uncertainty is computed
    on the SAMPLED point logits (computing it on the coarse map first and
    sampling that leads to incorrect scores when a sample lands between
    opposite-sign coarse predictions).
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords
