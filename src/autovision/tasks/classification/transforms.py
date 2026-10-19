# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Classification transforms (train recipe + evaluation recipe)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2

from autovision.tasks.classification.config import (
    ImageClassificationTrainArgs,
)

__all__ = ["build_classification_transform"]


def build_classification_transform(
    args: ImageClassificationTrainArgs, train: bool
) -> nn.Module:
    """Build the train/eval input pipeline.

    Train: RandomResizedCrop -> horizontal flip -> normalize.
    Eval: Resize (shorter side) -> CenterCrop -> normalize.
    Input: uint8 (C, H, W) tensor; output: float32 normalized tensor.
    """
    height, width = args.image_size
    mean, std = args.normalize.mean, args.normalize.std
    if train:
        ops: list[nn.Module] = [
            v2.RandomResizedCrop(
                size=(height, width), scale=(0.08, 1.0), antialias=True
            ),
            v2.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            v2.Resize(size=height, antialias=True),
            v2.CenterCrop(size=(height, width)),
        ]
    ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.append(v2.Normalize(mean=list(mean), std=list(std)))
    return v2.Compose(ops)
