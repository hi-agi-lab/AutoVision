# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Geometry-coupled crop helpers for the PaKA dense relational loss.

These are data-augmentation utilities (not SSL math): they build extra views
whose crop geometry depends on another view's crop, which the standard
stateless ``ViewTransform`` cannot express. Views carry an 8-element
``geometry`` tensor ``[x0, y0, x1, y1, image_w, image_h, hflip, vflip]`` in
original-image pixels so the PaKA loss can ROI-align crops against each
other.

- ``_GeometryViewTransform``: a single view (random resized crop + flips +
  photometric + normalize) that records its crop geometry.
- ``render_clean_global``: augmentation-free re-render of a recorded global
  crop (the PaKA clean teacher).
- ``sample_high_overlap_box``: small local box fully inside a parent global
  crop (containment = 1.0 satisfies the PaKA overlap constraint).
- ``render_augmented_local``: photometric-augmented render of such a box.
"""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor, nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from autovision.augment.transform_args import (
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    SolarizeArgs,
)
from autovision.augment.view_transform import RandomGaussianBlur, RandomSolarize

__all__ = [
    "_GeometryViewTransform",
    "render_clean_global",
    "render_augmented_local",
    "sample_high_overlap_box",
]

# Local-crop scale relative to the parent (global) crop area: PaKA student
# locals are small sub-regions of a global view, DINO local-crop style.
_LOCAL_SCALE_RANGE = (0.05, 0.40)
#: Aspect ratio (w/h) sampling range, drawn log-uniformly (the
#: RandomResizedCrop convention, torchvision BSD-3 family math).
_ASPECT_RANGE = (3.0 / 4.0, 4.0 / 3.0)


def _build_photometric_ops(
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
) -> nn.Module:
    """Photometric chain (jitter -> grayscale -> blur -> solarize), no geometry."""
    ops: list[nn.Module] = []
    if color_jitter is not None:
        jitter = v2.ColorJitter(
            brightness=color_jitter.strength * color_jitter.brightness,
            contrast=color_jitter.strength * color_jitter.contrast,
            saturation=color_jitter.strength * color_jitter.saturation,
            hue=color_jitter.strength * color_jitter.hue,
        )
        ops.append(v2.RandomApply([jitter], p=color_jitter.prob))
    if random_gray_scale is not None:
        ops.append(v2.RandomGrayscale(p=random_gray_scale))
    if gaussian_blur is not None:
        ops.append(RandomGaussianBlur(gaussian_blur))
    if solarize is not None:
        ops.append(RandomSolarize(solarize))
    if not ops:
        return nn.Identity()
    return v2.Compose(ops)


class _GeometryViewTransform:
    """One augmented view that records its crop geometry.

    Op order mirrors ``ViewTransform``: random resized crop -> flips ->
    photometric -> to float + normalize. The sampled crop box and flips are
    returned as the 8-element geometry tensor so other views can be aligned
    against this one.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        random_resize: tuple[float, float],
        random_flip_h: float,
        random_flip_v: float,
        color_jitter: ColorJitterArgs | None,
        random_gray_scale: float | None,
        gaussian_blur: GaussianBlurArgs | None,
        solarize: SolarizeArgs | None,
        normalize: NormalizeArgs,
    ) -> None:
        self.size = tuple(image_size)
        self.scale = tuple(random_resize)
        self.flip_h_prob = random_flip_h
        self.flip_v_prob = random_flip_v
        self.photometric = _build_photometric_ops(
            color_jitter, random_gray_scale, gaussian_blur, solarize
        )
        self.mean = list(normalize.mean)
        self.std = list(normalize.std)

    def __call__(self, image: Tensor) -> dict[str, Tensor]:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        top, left, crop_h, crop_w = v2.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        view = TF.resized_crop(
            image, top, left, crop_h, crop_w, list(self.size), antialias=True
        )
        hflip = random.random() < self.flip_h_prob
        vflip = random.random() < self.flip_v_prob
        if hflip:
            view = TF.hflip(view)
        if vflip:
            view = TF.vflip(view)
        view = self.photometric(view)
        view = TF.to_dtype(view, torch.float32, scale=True)
        view = TF.normalize(view, mean=self.mean, std=self.std)
        geometry = torch.tensor(
            [
                float(left),
                float(top),
                float(left + crop_w),
                float(top + crop_h),
                float(width),
                float(height),
                1.0 if hflip else 0.0,
                1.0 if vflip else 0.0,
            ],
            dtype=torch.float32,
        )
        return {"image": view, "geometry": geometry}


def _crop_resize_flips(
    image: Tensor,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    hflip: bool,
    vflip: bool,
) -> Tensor:
    x0, y0, x1, y1 = box
    view = TF.crop(image, y0, x0, y1 - y0, x1 - x0)
    view = TF.resize(view, list(size), antialias=True)
    if hflip:
        view = TF.hflip(view)
    if vflip:
        view = TF.vflip(view)
    return view


def _box_and_flips_from_geometry(
    geometry: Tensor,
) -> tuple[tuple[int, int, int, int], bool, bool]:
    """Integer crop box + flip flags decoded from a geometry row.

    The 8-float layout ``[x0, y0, x1, y1, w, h, hflip, vflip]`` is the
    cross-module interface; box edges round to nearest int and the flip
    flags threshold at 0.5.
    """
    x0, y0, x1, y1 = (int(round(float(edge))) for edge in geometry[:4])
    flips = geometry[6:8] > 0.5
    return (x0, y0, x1, y1), bool(flips[0]), bool(flips[1])


def render_clean_global(
    image: Tensor,
    geometry: Tensor,
    size: tuple[int, int],
    normalize: NormalizeArgs,
) -> dict[str, Tensor]:
    """Render a global crop with NO photometric augmentation (clean teacher).

    Crops ``image`` to the box stored in ``geometry``, resizes to ``size``,
    re-applies the recorded flips, and normalizes. The returned view re-uses
    ``geometry`` verbatim: the PaKA ROI alignment can treat the clean and the
    augmented global as the same region.
    """
    box, hflip, vflip = _box_and_flips_from_geometry(geometry)
    view = _crop_resize_flips(image, box, tuple(size), hflip, vflip)
    view = TF.normalize(
        TF.to_dtype(view, torch.float32, scale=True),
        mean=list(normalize.mean),
        std=list(normalize.std),
    )
    return {"image": view, "geometry": geometry.clone()}


def sample_high_overlap_box(
    parent_box: tuple[int, int, int, int],
    min_iou: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Sample a local crop fully contained in ``parent_box``.

    Containment satisfies the PaKA overlap constraint by construction (a
    local inside its parent global has overlap ratio 1.0), so ``min_iou``
    is accepted for API symmetry and unused. The local's area is drawn
    uniformly from ``_LOCAL_SCALE_RANGE`` of the parent area, its aspect
    ratio log-uniformly from ``_ASPECT_RANGE``, and its position uniformly
    among legal offsets. Spec: provenance spec S1.
    """
    del min_iou  # containment (ratio 1.0) always satisfies the constraint
    px0, py0, px1, py1 = parent_box
    parent_w = px1 - px0
    parent_h = py1 - py0
    if parent_w <= 1 or parent_h <= 1:
        return parent_box

    area = rng.uniform(*_LOCAL_SCALE_RANGE) * parent_w * parent_h
    aspect = math.exp(
        rng.uniform(math.log(_ASPECT_RANGE[0]), math.log(_ASPECT_RANGE[1]))
    )
    width = min(parent_w, max(1, round(math.sqrt(area * aspect))))
    height = min(parent_h, max(1, round(math.sqrt(area / aspect))))
    left = px0 + rng.randint(0, parent_w - width)
    top = py0 + rng.randint(0, parent_h - height)
    return left, top, left + width, top + height


def render_augmented_local(
    image: Tensor,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    hflip: bool,
    vflip: bool,
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
    normalize: NormalizeArgs,
) -> dict[str, Tensor]:
    """Render an augmented student local crop from a pre-sampled box.

    The crop box is decided by the caller (so geometry + containment are
    known); the render applies crop -> resize -> flips -> standard DINO
    photometric -> normalize, and records the box + flips as geometry.
    """
    image_h, image_w = int(image.shape[-2]), int(image.shape[-1])
    view = _crop_resize_flips(image, box, tuple(size), hflip, vflip)
    view = _build_photometric_ops(
        color_jitter, random_gray_scale, gaussian_blur, solarize
    )(view)
    view = TF.normalize(
        TF.to_dtype(view, torch.float32, scale=True),
        mean=list(normalize.mean),
        std=list(normalize.std),
    )
    geometry = torch.tensor(
        [
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(image_w),
            float(image_h),
            1.0 if hflip else 0.0,
            1.0 if vflip else 0.0,
        ],
        dtype=torch.float32,
    )
    return {"image": view, "geometry": geometry}
