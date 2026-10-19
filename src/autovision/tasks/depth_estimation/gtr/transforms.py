# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR metric-depth: joint (image, depth) transforms (doc 31 §11 S4)."""

from __future__ import annotations

import random

import torch
from torch import nn

from autovision.tasks.depth_estimation.gtr.config import (
    GTRDepthEstimationTrainArgs,
)

__all__ = ["build_depth_transform"]


class _DepthTransform(nn.Module):
    """Resize image (bilinear, antialias) + depth (nearest), sync flip.

    Depths stay in meters — only the image is normalized.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        normalize_mean: tuple[float, ...],
        normalize_std: tuple[float, ...],
        train: bool,
        horizontal_flip_prob: float,
    ) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.train_mode = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.register_buffer("mean", torch.tensor(normalize_mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(normalize_std).view(-1, 1, 1))

    def forward(self, item: dict) -> dict:
        from torchvision.transforms.v2 import functional as TF

        image = item["image"]
        depth = item["depth"]
        height, width = self.image_size
        image = TF.resize(image, [height, width], antialias=True)
        depth = TF.resize(
            depth.unsqueeze(0), [height, width],
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(0)
        if self.train_mode and random.random() < self.horizontal_flip_prob:
            image = TF.horizontal_flip(image)
            depth = TF.horizontal_flip(depth)
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        image = (image - self.mean) / self.std
        return {**item, "image": image, "depth": depth.float()}


def build_depth_transform(
    args: GTRDepthEstimationTrainArgs, train: bool
) -> nn.Module:
    return _DepthTransform(
        image_size=args.image_size,
        normalize_mean=args.normalize.mean,
        normalize_std=args.normalize.std,
        train=train,
        horizontal_flip_prob=args.horizontal_flip_prob if train else 0.0,
    )
