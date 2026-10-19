# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT segmentation: joint (image, mask) transforms.

Extends the shared resize/flip/normalize pipeline with the EOMT recipe's
random-scale crop on the training side: the image rescales by a random
factor, then a fixed-size crop is taken (zero-padded with the ignore
label when the scaled image is smaller than the crop window).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

__all__ = ["build_eomt_segmentation_transform", "build_eomt_instance_transform"]


class _EomtSegmentationTransform(nn.Module):
    """Joint geometric + photometric transform with shared randomness."""

    def __init__(
        self,
        image_size: tuple[int, int],
        normalize_mean: tuple[float, ...],
        normalize_std: tuple[float, ...],
        train: bool,
        horizontal_flip_prob: float,
        scale_range: tuple[float, float],
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.train_mode = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.scale_range = scale_range
        self.ignore_index = ignore_index
        self.register_buffer("mean", torch.tensor(normalize_mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(normalize_std).view(-1, 1, 1))

    def _scale_crop(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        from torchvision.transforms.v2 import functional as TF

        height, width = self.image_size
        scale = random.uniform(*self.scale_range)
        source_h, source_w = image.shape[-2:]
        scaled_h = max(1, int(round(source_h * scale)))
        scaled_w = max(1, int(round(source_w * scale)))
        image = TF.resize(image, [scaled_h, scaled_w], antialias=True)
        mask = TF.resize(
            mask, [scaled_h, scaled_w],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        pad_h, pad_w = max(0, height - scaled_h), max(0, width - scaled_w)
        if pad_h or pad_w:
            image = TF.pad(image, [0, 0, pad_w, pad_h])
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)
        top = random.randint(0, max(0, scaled_h + pad_h - height))
        left = random.randint(0, max(0, scaled_w + pad_w - width))
        return (
            image[..., top : top + height, left : left + width],
            mask[..., top : top + height, left : left + width],
        )

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        from torchvision.transforms.v2 import functional as TF

        # Masks arrive as (H, W); geometry ops need a channel dimension.
        squeeze_back = mask.ndim == 2
        if squeeze_back:
            mask = mask.unsqueeze(0)
        if self.train_mode:
            image, mask = self._scale_crop(image, mask)
        else:
            height, width = self.image_size
            image = TF.resize(image, [height, width], antialias=True)
            mask = TF.resize(
                mask, [height, width],
                interpolation=TF.InterpolationMode.NEAREST,
            )
        if self.train_mode and random.random() < self.horizontal_flip_prob:
            image = TF.horizontal_flip(image)
            mask = TF.horizontal_flip(mask)
        if squeeze_back:
            mask = mask.squeeze(0)
        image = image.float() / 255.0
        image = (image - self.mean) / self.std
        return image, mask.long()


def build_eomt_segmentation_transform(
    image_size: tuple[int, int],
    normalize_mean: tuple[float, ...],
    normalize_std: tuple[float, ...],
    train: bool,
    horizontal_flip_prob: float = 0.5,
    scale_range: tuple[float, float] = (0.5, 2.0),
    ignore_index: int = 255,
) -> nn.Module:
    return _EomtSegmentationTransform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        train=train,
        horizontal_flip_prob=horizontal_flip_prob,
        scale_range=scale_range,
        ignore_index=ignore_index,
    )


class _EomtInstanceTransform(nn.Module):
    """Joint (image, masks, boxes) geometry: fixed resize + optional flip.

    Normalized cxcywh boxes are invariant to the uniform resize; the flip
    mirrors masks and box centers together.
    """

    def __init__(
        self,
        image_size: int,
        normalize_mean: tuple[float, ...],
        normalize_std: tuple[float, ...],
        train: bool,
        horizontal_flip_prob: float,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.train_mode = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.register_buffer("mean", torch.tensor(normalize_mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(normalize_std).view(-1, 1, 1))

    def forward(
        self, image: Tensor, masks: Tensor, boxes: Tensor, classes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from torchvision.transforms.v2 import functional as TF

        size = (self.image_size, self.image_size)
        if image.shape[-2:] != size:
            image = TF.resize(image, size, antialias=True)
            if masks.numel():
                masks = TF.resize(
                    masks, size, interpolation=TF.InterpolationMode.NEAREST
                )
        if self.train_mode and random.random() < self.horizontal_flip_prob:
            image = TF.horizontal_flip(image)
            masks = TF.horizontal_flip(masks)
            if boxes.numel():
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
        image = (image.float() / 255.0 if image.max() > 1.5 else image.float())
        image = (image - self.mean) / self.std
        return image, masks.float(), boxes, classes


def build_eomt_instance_transform(
    image_size: int,
    normalize_mean: tuple[float, ...],
    normalize_std: tuple[float, ...],
    train: bool,
    horizontal_flip_prob: float = 0.5,
) -> nn.Module:
    return _EomtInstanceTransform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        train=train,
        horizontal_flip_prob=horizontal_flip_prob,
    )
