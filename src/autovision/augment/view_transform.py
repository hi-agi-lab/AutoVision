# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Single-view transform pipeline built on torchvision transforms v2.

Op order (fixed, mirrors the baseline ViewTransform):
channel-agnostic RandomResizedCrop -> flips -> photometric (color jitter ->
grayscale -> gaussian blur -> solarize) -> to float + normalize.

Input: uint8 (C, H, W) tensor. Output: float32 (C, H, W) normalized tensor.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from autovision.augment.transform_args import (
    GaussianBlurArgs,
    MethodTransformArgs,
    SolarizeArgs,
)

__all__ = ["ViewTransform"]


class RandomGaussianBlur(nn.Module):
    """Gaussian blur with sigma sampled as a fraction of the image size."""

    def __init__(self, args: GaussianBlurArgs) -> None:
        super().__init__()
        self.prob = args.prob
        self.sigmas = args.sigmas

    def forward(self, img: Tensor) -> Tensor:
        if random.random() > self.prob:
            return img
        size = float(img.shape[-1])
        sigma = random.uniform(*self.sigmas) * size
        kernel = int(2 * max(1.0, round(3 * sigma)) + 1)
        kernel = min(kernel, int(size) // 2 * 2 - 1)  # odd and <= size
        kernel = max(kernel, 3)
        return TF.gaussian_blur(img, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


class RandomSolarize(nn.Module):
    def __init__(self, args: SolarizeArgs) -> None:
        super().__init__()
        self.prob = args.prob
        self.threshold = args.threshold

    def forward(self, img: Tensor) -> Tensor:
        if random.random() > self.prob:
            return img
        max_value = 1.0 if img.is_floating_point() else 255.0
        return TF.solarize(img, threshold=self.threshold * max_value)


class ViewTransform:
    """Builds and applies one view of the augmentation pipeline."""

    def __init__(self, transform_args: MethodTransformArgs) -> None:
        self.transform_args = transform_args
        height, width = transform_args.image_size
        scale = transform_args.random_resize
        ops: list[nn.Module] = [
            v2.RandomResizedCrop(
                size=(height, width),
                scale=(scale.min_scale, scale.max_scale),
                antialias=True,
            )
        ]
        flip = transform_args.random_flip
        if flip is not None:
            if flip.horizontal_prob > 0:
                ops.append(v2.RandomHorizontalFlip(p=flip.horizontal_prob))
            if flip.vertical_prob > 0:
                ops.append(v2.RandomVerticalFlip(p=flip.vertical_prob))
        jitter = transform_args.color_jitter
        if jitter is not None:
            color_jitter = v2.ColorJitter(
                brightness=jitter.strength * jitter.brightness,
                contrast=jitter.strength * jitter.contrast,
                saturation=jitter.strength * jitter.saturation,
                hue=jitter.strength * jitter.hue,
            )
            ops.append(v2.RandomApply([color_jitter], p=jitter.prob))
        if transform_args.random_gray_scale is not None:
            ops.append(v2.RandomGrayscale(p=transform_args.random_gray_scale))
        if transform_args.gaussian_blur is not None:
            ops.append(RandomGaussianBlur(transform_args.gaussian_blur))
        if transform_args.solarize is not None:
            ops.append(RandomSolarize(transform_args.solarize))
        mean, std = transform_args.normalize.mean, transform_args.normalize.std
        ops.append(v2.ToDtype(torch.float32, scale=True))
        ops.append(v2.Normalize(mean=list(mean), std=list(std)))
        self.transform = v2.Compose(ops)

    def __call__(self, image: Tensor) -> Tensor:
        return self.transform(image)
