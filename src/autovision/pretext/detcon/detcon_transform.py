# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DetCon transforms: two views with image + region mask jointly augmented.

The region masks must follow the exact geometry of the image views (crop /
flips), so the geometric ops run on an ``{"image", "mask"}`` dict sample:
torchvision v2 shares the sampled parameters across all dict members and
forces NEAREST interpolation for masks, keeping region ids intact. The
photometric ops (jitter / grayscale / blur / solarize) and the normalization
run on the image only — masks pass through untouched.

Divergences from the DetCon paper reference configuration:

- Blur strength follows this codebase's sigma-as-fraction-of-size convention
  (the DeepMind reference samples absolute sigmas with a 23 px blur limit).
- The reference implementation relies on albumentations' implicit ``mask``
  target support; here the joint geometry is explicit through the
  torchvision v2 dict dispatch (verified: identical crop params, NEAREST
  for masks).
"""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors
from torch import Tensor

from autovision.augment.transform_args import (
    ColorJitterArgs,
    GaussianBlurArgs,
    MethodTransformArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    SolarizeArgs,
)
from autovision.augment.view_transform import RandomGaussianBlur, RandomSolarize

__all__ = [
    "DetConBTransform",
    "DetConBTransformArgs",
    "DetConSTransform",
    "DetConSTransformArgs",
]


class DetConSTransformArgs(MethodTransformArgs):
    """Two-view DetConS recipe (baseline defaults)."""

    image_size: tuple[int, int] = (224, 224)
    random_resize: RandomResizeArgs = RandomResizeArgs()
    random_flip: RandomFlipArgs | None = RandomFlipArgs()
    color_jitter: ColorJitterArgs | None = ColorJitterArgs(
        prob=0.8, strength=1.0, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
    )
    random_gray_scale: float | None = 0.2
    gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(prob=0.5, sigmas=(0.1, 2.0))
    solarize: SolarizeArgs | None = None
    normalize: NormalizeArgs | None = NormalizeArgs()
    # View 1 skips the blur (baseline DetConSView1TransformArgs).
    view_1_gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(prob=0.0)
    view_1_solarize: SolarizeArgs | None = None


class DetConBTransformArgs(MethodTransformArgs):
    """Two-view DetConB recipe (baseline defaults)."""

    image_size: tuple[int, int] = (224, 224)
    random_resize: RandomResizeArgs = RandomResizeArgs()
    random_flip: RandomFlipArgs | None = RandomFlipArgs()
    color_jitter: ColorJitterArgs | None = ColorJitterArgs(
        prob=0.8, strength=1.0, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    )
    random_gray_scale: float | None = 0.2
    gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(prob=1.0, sigmas=(0.1, 2.0))
    solarize: SolarizeArgs | None = None
    normalize: NormalizeArgs | None = NormalizeArgs()
    # View 1: light blur (p=0.1) plus solarize (p=0.2, threshold 0.5).
    view_1_gaussian_blur: GaussianBlurArgs | None = GaussianBlurArgs(prob=0.1, sigmas=(0.1, 2.0))
    view_1_solarize: SolarizeArgs | None = SolarizeArgs(prob=0.2, threshold=0.5)


class _MaskedViewTransform:
    """One augmented view of an ``{"image", "mask"}`` sample.

    Geometry (crop + flips) applies to image and mask jointly; photometric
    ops and normalization apply to the image only. The mask stays an
    integer tensor of shape (H, W) with region ids preserved.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        random_resize: RandomResizeArgs,
        random_flip: RandomFlipArgs | None,
        color_jitter: ColorJitterArgs | None,
        random_gray_scale: float | None,
        gaussian_blur: GaussianBlurArgs | None,
        solarize: SolarizeArgs | None,
        normalize: NormalizeArgs,
    ) -> None:
        height, width = image_size
        geometry: list[v2.Transform] = [
            v2.RandomResizedCrop(
                size=(height, width),
                scale=(random_resize.min_scale, random_resize.max_scale),
                antialias=True,
            )
        ]
        if random_flip is not None:
            if random_flip.horizontal_prob > 0:
                geometry.append(
                    v2.RandomHorizontalFlip(p=random_flip.horizontal_prob)
                )
            if random_flip.vertical_prob > 0:
                geometry.append(v2.RandomVerticalFlip(p=random_flip.vertical_prob))
        self.geometry = v2.Compose(geometry)

        photometric: list = []
        if color_jitter is not None:
            jitter = v2.ColorJitter(
                brightness=color_jitter.strength * color_jitter.brightness,
                contrast=color_jitter.strength * color_jitter.contrast,
                saturation=color_jitter.strength * color_jitter.saturation,
                hue=color_jitter.strength * color_jitter.hue,
            )
            photometric.append(v2.RandomApply([jitter], p=color_jitter.prob))
        if random_gray_scale is not None:
            photometric.append(v2.RandomGrayscale(p=random_gray_scale))
        if gaussian_blur is not None:
            photometric.append(RandomGaussianBlur(gaussian_blur))
        if solarize is not None:
            photometric.append(RandomSolarize(solarize))
        self.photometric = v2.Compose(photometric) if photometric else None

        if normalize is not None:
            self.postprocess = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=list(normalize.mean), std=list(normalize.std)),
                ]
            )
        else:
            self.postprocess = None

    def __call__(self, input: dict[str, Any]) -> dict[str, Tensor]:
        sample = {
            "image": input["image"],
            "mask": tv_tensors.Mask(input["mask"]),
        }
        out = self.geometry(sample)
        image = out["image"]
        if self.photometric is not None:
            image = self.photometric(image)
        if self.postprocess is not None:
            image = self.postprocess(image)
        # Return a plain tensor (not the tv_tensors.Mask subclass) so the
        # collated batch holds regular integer tensors.
        mask = out["mask"].as_subclass(torch.Tensor)
        return {"image": image, "mask": mask}


class _DetConTransform:
    """Shared two-view plumbing for the DetCon transforms."""

    transform_args: Any

    def __init__(self, transform_args: Any) -> None:
        self.transform_args = transform_args
        view_0 = _MaskedViewTransform(
            image_size=transform_args.image_size,
            random_resize=transform_args.random_resize,
            random_flip=transform_args.random_flip,
            color_jitter=transform_args.color_jitter,
            random_gray_scale=transform_args.random_gray_scale,
            gaussian_blur=transform_args.gaussian_blur,
            solarize=transform_args.solarize,
            normalize=transform_args.normalize,
        )
        view_1 = _MaskedViewTransform(
            image_size=transform_args.image_size,
            random_resize=transform_args.random_resize,
            random_flip=transform_args.random_flip,
            color_jitter=transform_args.color_jitter,
            random_gray_scale=transform_args.random_gray_scale,
            gaussian_blur=transform_args.view_1_gaussian_blur,
            solarize=transform_args.view_1_solarize,
            normalize=transform_args.normalize,
        )
        self.views = [view_0, view_1]

    def __call__(self, input: dict[str, Any]) -> list[dict[str, Tensor]]:
        return [view(input) for view in self.views]


class DetConSTransform(_DetConTransform):
    """Two-view DetConS transform emitting image + mask per view."""

    @staticmethod
    def transform_args_cls() -> type[DetConSTransformArgs]:
        return DetConSTransformArgs


class DetConBTransform(_DetConTransform):
    """Two-view DetConB transform emitting image + mask per view."""

    @staticmethod
    def transform_args_cls() -> type[DetConBTransformArgs]:
        return DetConBTransformArgs
