# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Transform argument models (shared building blocks).

The base :class:`MethodTransformArgs` has no field defaults — every method's
subclass supplies its own recipe defaults (mirroring the baseline design).
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field

from autovision.config.base import PydanticConfig
from autovision.types import ImageSizeTuple

__all__ = [
    "ColorJitterArgs",
    "GaussianBlurArgs",
    "MethodTransformArgs",
    "NormalizeArgs",
    "RandomFlipArgs",
    "RandomResizeArgs",
    "SolarizeArgs",
    "IMAGENET_NORMALIZE_MEAN",
    "IMAGENET_NORMALIZE_STD",
]

IMAGENET_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZE_STD = (0.229, 0.224, 0.225)


class NormalizeArgs(PydanticConfig):
    mean: tuple[float, ...] = Field(default=IMAGENET_NORMALIZE_MEAN, strict=False)
    std: tuple[float, ...] = Field(default=IMAGENET_NORMALIZE_STD, strict=False)

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizeArgs":
        return cls(mean=tuple(data["mean"]), std=tuple(data["std"]))

    def to_tensor(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.mean, dtype=torch.float32, device=device).view(-1, 1, 1),
            torch.tensor(self.std, dtype=torch.float32, device=device).view(-1, 1, 1),
        )


class RandomResizeArgs(PydanticConfig):
    min_scale: float = 0.08
    max_scale: float = 1.0


class RandomFlipArgs(PydanticConfig):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class ColorJitterArgs(PydanticConfig):
    prob: float = 0.8
    strength: float = 0.5
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.4
    hue: float = 0.2


class GaussianBlurArgs(PydanticConfig):
    """Gaussian blur with the sigma given as a *fraction of the image size*
    (DINO-style blur strength), applied with probability ``prob``."""

    prob: float = 1.0
    sigmas: tuple[float, float] = Field(default=(0.1, 2.0), strict=False)


class SolarizeArgs(PydanticConfig):
    prob: float = 0.2
    threshold: float = 0.5  # fraction of the value range


class MethodTransformArgs(PydanticConfig):
    """Base transform args. Concrete recipes come from method subclasses."""

    image_size: ImageSizeTuple
    num_channels: int | Literal["auto"] = "auto"
    random_resize: RandomResizeArgs
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
    random_gray_scale: float | None = None
    gaussian_blur: GaussianBlurArgs | None = None
    solarize: SolarizeArgs | None = None
    normalize: NormalizeArgs

    def resolve_auto_fields(self) -> None:
        if self.num_channels == "auto":
            self.num_channels = len(self.normalize.mean)

    def resolve_incompatible(self) -> None:
        num_channels: int = self.num_channels
        if num_channels != len(self.normalize.mean):
            repeats = num_channels // len(self.normalize.mean)
            remainder = num_channels % len(self.normalize.mean)
            mean = list(self.normalize.mean) * repeats + list(self.normalize.mean)[:remainder]
            std = list(self.normalize.std) * repeats + list(self.normalize.std)[:remainder]
            self.normalize = NormalizeArgs(mean=tuple(mean), std=tuple(std))
        if num_channels != 3:
            self.color_jitter = None
            self.random_gray_scale = None
            self.solarize = None
