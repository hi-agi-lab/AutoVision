# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose: joint (image, keypoint, box) transforms (doc 31 §11 S5b).

Ports the official val/train conversion pieces:

- ``PoseResize`` semantics: plain (aspect-distorting) resize to
  ``image_size`` — the postprocess inverse (fraction × original size)
  makes the whole protocol self-consistent;
- ``PoseRandomHorizontalFlip`` / ``hflip``: mirror ``x -> w - x - 1`` for
  visible keypoints (invisible get ``x = 0``), swap the COCO left/right
  pairs; boxes mirror on x;
- ``PoseToTensor`` + ``PoseNormalize``: zero invisible keypoint coords,
  map labelled visibility ``2 -> 1``, divide keypoints by the resized
  frame, boxes to normalized cxcywh, area to frame-normalized units.

The augmented official train pipeline gains its first two pieces here
(roadmap C4): keypoint-aware **mosaic** (2x2 canvas from the sample plus
three cache donors, one whole-canvas affine shared by the box and
keypoint geometry) and **mixup** (beta blend of the current sample with a
cached donor at the current frame, annotations merged). Both are OFF by
default (``mosaic_prob`` / ``mixup_prob`` = ``None``) — the resize + flip
pipeline stays bit-for-bit — and honor optional step windows (``start``
inclusive, ``stop`` exclusive, detection-family semantics). The official
recipe's zoom-out / color-jitter remain unported.

Geometry contract: person rows stay aligned across ``keypoints`` /
``bboxes`` / ``classes`` / ``area`` (the dataset builds them in one
pass); a person whose box degenerates after an affine is dropped with its
keypoints, and a keypoint pushed off the canvas is unlabeled (``v = 0``).
"""

from __future__ import annotations

import random

import torch
from torch import nn

from autovision.augment.mosaic_geometry import (
    apply_affine_to_boxes,
    apply_affine_to_keypoints,
    mosaic_canvas_box_clamp,
)
from autovision.tasks.pose_estimation.gtr.config import (
    GTRPoseEstimationTrainArgs,
)

__all__ = ["build_pose_transform"]

#: COCO person left/right keypoint index pairs (official ``hflip``).
COCO_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
)


class _PoseTransform(nn.Module):
    """Resize + normalize (both splits) with optional train-time augmentation:
    horizontal flip, keypoint-aware mosaic, and mixup (roadmap C4)."""

    #: Rolling cache depth for the mosaic/mixup donors (detection-family value).
    CACHE_SIZE = 50

    def __init__(
        self,
        image_size: tuple[int, int],
        normalize_mean: tuple[float, ...],
        normalize_std: tuple[float, ...],
        train: bool,
        horizontal_flip_prob: float,
        num_body_points: int,
        *,
        mosaic_prob: float | None = None,
        mosaic_output_size: int = 640,
        mosaic_step_start: int | None = None,
        mosaic_step_stop: int | None = None,
        mixup_prob: float | None = None,
        mixup_step_start: int | None = None,
        mixup_step_stop: int | None = None,
    ) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.train_mode = bool(train)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.num_body_points = int(num_body_points)
        self.mosaic_prob = mosaic_prob
        self.mosaic_output_size = int(mosaic_output_size)
        self.mosaic_step_start = mosaic_step_start
        self.mosaic_step_stop = mosaic_step_stop
        self.mixup_prob = mixup_prob
        self.mixup_step_start = mixup_step_start
        self.mixup_step_stop = mixup_step_stop
        self._step = 0
        self._cache: list[dict] = []
        self._reinit_pending = False
        self.register_buffer("mean", torch.tensor(normalize_mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(normalize_std).view(-1, 1, 1))

    # ── step scheduling (engine step hooks) ─────────────────────────────
    def set_step(self, step: int) -> None:
        """Track the step; a window edge crossing flags a dataloader rebuild.

        Workers hold PICKLED copies of this transform, so the only way a
        step window reaches them is the engine's reinit handshake
        (``requires_dataloader_reinitialization`` +
        ``mark_dataloader_as_reinitialized``): the rebuild re-pickles the
        current ``_step`` into fresh workers — same contract as the
        detection family's ``StepScheduledCompose``.
        """
        previous = self._windows_open()
        self._step = int(step)
        if self._windows_open() != previous:
            self._reinit_pending = True

    def requires_dataloader_reinitialization(self) -> bool:
        return self._reinit_pending

    def mark_dataloader_as_reinitialized(self) -> None:
        self._reinit_pending = False

    def _in_window(self, prob, start, stop) -> bool:
        """``start`` inclusive / ``stop`` exclusive / both optional."""
        if not self.train_mode or prob is None:
            return False
        if start is not None and self._step < int(start):
            return False
        if stop is not None and self._step >= int(stop):
            return False
        return True

    def _windows_open(self) -> tuple[bool, bool]:
        return (
            self._in_window(self.mosaic_prob, self.mosaic_step_start, self.mosaic_step_stop),
            self._in_window(self.mixup_prob, self.mixup_step_start, self.mixup_step_stop),
        )

    def _window_active(
        self, prob: float | None, start: int | None, stop: int | None
    ) -> bool:
        return self._in_window(prob, start, stop) and random.random() < float(prob)

    # ── rolling cache of raw samples (mosaic + mixup donors) ────────────
    @staticmethod
    def _clone_sample(item: dict) -> dict:
        return {
            "image": item["image"].clone(),
            "keypoints": item["keypoints"].clone(),
            "bboxes": item["bboxes"].clone(),
            "classes": item["classes"].clone(),
        }

    def _cache_append(self, item: dict) -> None:
        self._cache.append(self._clone_sample(item))
        if len(self._cache) > self.CACHE_SIZE:
            # Detection-family eviction: a random index excluding the fresh one.
            del self._cache[random.randint(0, len(self._cache) - 2)]

    # ── mosaic ──────────────────────────────────────────────────────────
    def _mosaic(self, item: dict) -> dict:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as TF

        donors = random.choices(self._cache, k=3)
        tiles = [item, *donors]
        tile = int(self.mosaic_output_size)
        # Shorter side -> output size, aspect kept (detection-family tile resize).
        resized = [TF.resize(sample["image"], [tile], antialias=True) for sample in tiles]
        max_h = max(image.shape[1] for image in resized)
        max_w = max(image.shape[2] for image in resized)
        canvas = torch.zeros((3, 2 * max_h, 2 * max_w), dtype=resized[0].dtype)
        keypoints_out, boxes_out, classes_out = [], [], []
        for index, (sample, image) in enumerate(zip(tiles, resized)):
            row, col = divmod(index, 2)
            y0, x0 = row * max_h, col * max_w
            _, height, width = image.shape
            canvas[:, y0 : y0 + height, x0 : x0 + width] = image
            ratio_x = width / sample["image"].shape[2]
            ratio_y = height / sample["image"].shape[1]
            keypoints = sample["keypoints"].clone().float()
            if keypoints.numel():
                keypoints[..., 0] = keypoints[..., 0] * ratio_x + x0
                keypoints[..., 1] = keypoints[..., 1] * ratio_y + y0
            boxes = sample["bboxes"].clone().float()
            if boxes.numel():
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_x + x0
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_y + y0
            keypoints_out.append(keypoints)
            boxes_out.append(boxes)
            classes_out.append(sample["classes"])
        keypoints = (
            torch.cat(keypoints_out, dim=0)
            if keypoints_out
            else item["keypoints"].new_zeros(0, self.num_body_points, 3)
        )
        boxes = (
            torch.cat(boxes_out, dim=0) if boxes_out else item["bboxes"].new_zeros(0, 4)
        )
        classes = (
            torch.cat(classes_out, dim=0)
            if classes_out
            else item["classes"].new_zeros(0)
        )
        canvas_size = (2 * max_h, 2 * max_w)
        boxes = mosaic_canvas_box_clamp(boxes, canvas_size)

        # One whole-canvas affine for the image AND both annotation kinds.
        angle, (tx, ty), scale_factor = self._sample_affine_params(*canvas_size)
        fill = 0 if canvas.dtype == torch.uint8 else [0.0] * canvas.shape[0]
        canvas = TF.affine(
            canvas,
            angle=angle,
            translate=[tx, ty],
            scale=scale_factor,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=fill,
        )
        if boxes.numel():
            boxes = apply_affine_to_boxes(
                boxes, canvas_size, angle, (tx, ty), scale_factor
            )
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            classes = classes[keep]
        else:
            keep = torch.ones(0, dtype=torch.bool)
        if keypoints.numel():
            points, inside = apply_affine_to_keypoints(
                keypoints[..., :2], canvas_size, angle, (tx, ty), scale_factor
            )
            visibility = keypoints[..., 2]
            # Cropped-out keypoints carry no supervision: v = 0 (their
            # coordinates are zeroed by the normalize stage).
            visibility = torch.where(
                inside, visibility, torch.zeros_like(visibility)
            )
            keypoints = torch.cat([points, visibility[..., None]], dim=-1)
            keypoints = keypoints[keep]
        return {
            **item,
            "image": canvas,
            "keypoints": keypoints,
            "bboxes": boxes,
            "classes": classes,
            "area": torch.zeros(boxes.shape[0], dtype=torch.float32),
            "original_size": (canvas.shape[2], canvas.shape[1]),
        }

    def _sample_affine_params(
        self, canvas_height: int, canvas_width: int
    ) -> tuple[float, tuple[int, int], float]:
        """Detection-family whole-canvas affine ranges."""
        angle = torch.empty(1).uniform_(-10.0, 10.0).item()
        tx = int(round(torch.empty(1).uniform_(-0.1 * canvas_width, 0.1 * canvas_width).item()))
        ty = int(round(torch.empty(1).uniform_(-0.1 * canvas_height, 0.1 * canvas_height).item()))
        scale_factor = torch.empty(1).uniform_(0.5, 1.5).item()
        return angle, (tx, ty), scale_factor

    # ── mixup ───────────────────────────────────────────────────────────
    def _mixup(self, item: dict) -> dict:
        from torchvision.transforms.v2 import functional as TF

        if not self._cache:
            return item
        donor = random.choice(self._cache)
        height, width = item["image"].shape[-2:]
        donor_image = TF.resize(donor["image"], [height, width], antialias=True)
        ratio_x = width / donor["image"].shape[-1]
        ratio_y = height / donor["image"].shape[-2]
        donor_keypoints = donor["keypoints"].clone().float()
        if donor_keypoints.numel():
            donor_keypoints[..., 0] *= ratio_x
            donor_keypoints[..., 1] *= ratio_y
        donor_boxes = donor["bboxes"].clone().float()
        if donor_boxes.numel():
            donor_boxes[:, [0, 2]] *= ratio_x
            donor_boxes[:, [1, 3]] *= ratio_y
        # Detection-family beta window.
        beta = random.uniform(0.45, 0.55)
        image = beta * item["image"].float() + (1.0 - beta) * donor_image.float()
        keypoints = torch.cat([item["keypoints"].float(), donor_keypoints], dim=0)
        boxes = torch.cat([item["bboxes"].float(), donor_boxes], dim=0)
        classes = torch.cat([item["classes"], donor["classes"]], dim=0)
        return {
            **item,
            "image": image,
            "keypoints": keypoints,
            "bboxes": boxes,
            "classes": classes,
            "area": torch.zeros(boxes.shape[0], dtype=torch.float32),
        }

    def forward(self, item: dict) -> dict:
        from torchvision.transforms.v2 import functional as TF

        if self.train_mode and (
            self.mosaic_prob is not None or self.mixup_prob is not None
        ):
            self._cache_append(item)
            if self._window_active(
                self.mosaic_prob, self.mosaic_step_start, self.mosaic_step_stop
            ):
                item = self._mosaic(item)
            if self._window_active(
                self.mixup_prob, self.mixup_step_start, self.mixup_step_stop
            ):
                item = self._mixup(item)
        image = item["image"]
        keypoints = item["keypoints"]
        boxes = item["bboxes"]
        orig_height, orig_width = image.shape[-2:]

        if self.train_mode and random.random() < self.horizontal_flip_prob:
            image = TF.horizontal_flip(image)
            if keypoints.numel():
                keypoints = keypoints.clone()
                visible = keypoints[..., 2:3] != 0
                mirrored = torch.where(
                    visible,
                    orig_width - keypoints[..., 0:1] - 1,
                    torch.zeros_like(keypoints[..., 0:1]),
                )
                keypoints[..., 0:1] = mirrored
                for left, right in COCO_FLIP_PAIRS:
                    keypoints[:, [left, right]] = keypoints[:, [right, left]]
            if boxes.numel():
                x1, y1, x2, y2 = boxes.unbind(-1)
                boxes = torch.stack(
                    [orig_width - x2 - 1, y1, orig_width - x1 - 1, y2], dim=-1
                )

        height, width = self.image_size
        ratio_w = width / orig_width
        ratio_h = height / orig_height
        image = TF.resize(image, [height, width], antialias=True)

        # ── keypoints: PoseNormalize order ──────────────────────────────
        # Official layout is (N, 3K) = [xy block (2K), visibility block (K)]
        # (``cat([Z, V])``), not interleaved [x, y, v] triplets.
        if keypoints.numel():
            keypoints = keypoints.clone()
            keypoints[..., 0] = keypoints[..., 0] * ratio_w / width
            keypoints[..., 1] = keypoints[..., 1] * ratio_h / height
            visible = keypoints[..., 2:3] != 0
            keypoints[..., 0:2] = torch.where(
                visible, keypoints[..., 0:2], torch.zeros_like(keypoints[..., 0:2])
            )
            visibility = (keypoints[..., 2:] > 0).to(keypoints.dtype)
            keypoints = torch.cat(
                [
                    keypoints[..., 0:2].reshape(keypoints.shape[0], -1),
                    visibility.reshape(visibility.shape[0], -1),
                ],
                dim=-1,
            )  # (N, 2K + K)
        keypoints = keypoints.reshape(
            keypoints.shape[0], 3 * self.num_body_points
        )  # (0, 3K) when the image has no persons

        # ── boxes: normalized cxcywh + frame-normalized area ────────────
        if boxes.numel():
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) / 2.0 * ratio_w / width
            cy = (y1 + y2) / 2.0 * ratio_h / height
            w_norm = (x2 - x1) * ratio_w / width
            h_norm = (y2 - y1) * ratio_h / height
            boxes = torch.stack([cx, cy, w_norm, h_norm], dim=-1)
            area = w_norm * h_norm
        else:
            area = item["area"]

        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        image = (image - self.mean) / self.std

        return {
            **item,
            "image": image,
            "keypoints": keypoints,
            "bboxes": boxes,
            "area": area,
        }


def build_pose_transform(
    args: GTRPoseEstimationTrainArgs, train: bool
) -> nn.Module:
    return _PoseTransform(
        image_size=args.image_size,
        normalize_mean=args.normalize.mean,
        normalize_std=args.normalize.std,
        train=train,
        horizontal_flip_prob=args.horizontal_flip_prob if train else 0.0,
        num_body_points=args.num_body_points,
        mosaic_prob=args.mosaic_prob if train else None,
        mosaic_output_size=args.mosaic_output_size,
        mosaic_step_start=args.mosaic_step_start,
        mosaic_step_stop=args.mosaic_step_stop,
        mixup_prob=args.mixup_prob if train else None,
        mixup_step_start=args.mixup_step_start,
        mixup_step_stop=args.mixup_step_stop,
    )
