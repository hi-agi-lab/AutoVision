# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Linear semantic segmentation: joint (image, mask) transforms."""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from autovision.tasks.semantic_segmentation.linear.config import (
    LinearSemanticSegmentationTrainArgs,
)

__all__ = ["build_segmentation_transform"]


class _SegmentationTransform(nn.Module):
    """Joint geometric + photometric transform with shared randomness.

    Image: bilinear resize + normalize. Mask: nearest resize (class indices
    must survive). Train mode adds a synchronized horizontal flip.
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
        self.image_size = image_size
        self.train_mode = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.register_buffer("mean", torch.tensor(normalize_mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(normalize_std).view(-1, 1, 1))

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        from torchvision.transforms.v2 import functional as TF

        height, width = self.image_size
        image = TF.resize(image, [height, width], antialias=True)
        mask = TF.resize(
            mask.unsqueeze(0), [height, width],
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(0)
        if self.train_mode and random.random() < self.horizontal_flip_prob:
            image = TF.horizontal_flip(image)
            mask = TF.horizontal_flip(mask)
        image = image.float() / 255.0
        image = (image - self.mean) / self.std
        return image, mask.long()


def build_segmentation_transform(
    args: LinearSemanticSegmentationTrainArgs, train: bool
) -> nn.Module:
    return _SegmentationTransform(
        image_size=args.image_size,
        normalize_mean=args.normalize.mean,
        normalize_std=args.normalize.std,
        train=train,
        horizontal_flip_prob=args.horizontal_flip_prob if train else 0.0,
    )
