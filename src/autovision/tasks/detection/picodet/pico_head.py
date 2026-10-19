# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dense PicoHead: anchor-point detection head (GFL-style regression).

Per FPN level: a short depthwise-separable stem, then two parallel 1x1
convs — sigmoid classification and DFL box regression (``4 x (reg_max+1)``
distribution channels per location). Boxes decode as the anchor point
minus/plus the softmax-integrated per-side distances (in stride units).
The exported graph emits raw logits + normalized cxcywh boxes, matching
the DETR-family output contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from autovision.tasks.detection.picodet.esnet import Hinge

__all__ = ["PicoHead"]


class _DPConv(nn.Sequential):
    """Depthwise 3x3 + pointwise 1x1, both with BN + hinge."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.BatchNorm2d(channels),
            Hinge(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Hinge(),
        )


class PicoHead(nn.Module):
    """Anchor-point head over the CSP-PAN pyramid (strides 8/16/32)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: tuple[int, ...] | list[int],
        strides: tuple[int, ...] | list[int],
        reg_max: int = 7,
        num_stem_convs: int = 2,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError("in_channels and strides must have the same length")
        self.num_classes = num_classes
        self.strides = tuple(strides)
        self.reg_max = reg_max
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for channels in in_channels:
            self.stems.append(
                nn.Sequential(*[_DPConv(channels) for _ in range(num_stem_convs)])
            )
            self.cls_convs.append(nn.Conv2d(channels, num_classes, 1))
            self.reg_convs.append(nn.Conv2d(channels, 4 * (reg_max + 1), 1))
        self.register_buffer(
            "project", torch.arange(reg_max + 1, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        prior_prob = 0.01
        for conv in self.cls_convs:
            nn.init.constant_(conv.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        cls_flat, reg_flat, anchors, stride_list = [], [], [], []
        for feat, stem, cls_conv, reg_conv, stride in zip(
            feats, self.stems, self.cls_convs, self.reg_convs, self.strides
        ):
            _, _, h, w = feat.shape
            x = stem(feat)
            cls_flat.append(cls_conv(x).flatten(2).transpose(1, 2))
            reg_flat.append(reg_conv(x).flatten(2).transpose(1, 2))
            ys, xs = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=feat.dtype),
                torch.arange(w, device=feat.device, dtype=feat.dtype),
                indexing="ij",
            )
            anchors.append(
                torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], dim=-1)
                .reshape(-1, 2)
            )
            stride_list.append(
                torch.full((h * w,), float(stride), device=feat.device, dtype=feat.dtype)
            )
        pred_logits = torch.cat(cls_flat, dim=1)  # (B, N, C)
        pred_dist = torch.cat(reg_flat, dim=1)  # (B, N, 4*(R+1))
        anchor_points = torch.cat(anchors, dim=0)  # (N, 2)
        anchor_strides = torch.cat(stride_list, dim=0)  # (N,)

        batch, num_anchors = pred_logits.shape[:2]
        dist = pred_dist.reshape(batch, num_anchors, 4, self.reg_max + 1)
        # Integral: expected per-side distance in stride units.
        ltrb = (dist.softmax(dim=-1) * self.project.to(dist.dtype)).sum(dim=-1)
        lt_rb = ltrb * anchor_strides.view(1, num_anchors, 1)
        center = anchor_points.view(1, num_anchors, 2)
        pred_xyxy = torch.cat(
            [center - lt_rb[..., :2], center + lt_rb[..., 2:]], dim=-1
        )
        # Normalized cxcywh (family output contract).
        canvas_w = feats[0].shape[3] * self.strides[0]
        canvas_h = feats[0].shape[2] * self.strides[0]
        cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2 / canvas_w
        cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2 / canvas_h
        bw = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) / canvas_w
        bh = (pred_xyxy[..., 3] - pred_xyxy[..., 1]) / canvas_h
        pred_boxes = torch.stack([cx, cy, bw, bh], dim=-1)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_dist": dist,
            "pred_xyxy": pred_xyxy,
            "anchor_points": anchor_points,
            "anchor_strides": anchor_strides,
        }
