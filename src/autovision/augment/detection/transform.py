# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Step-scheduled detection transform pipeline (torchvision v2 based).

The active augmentation set is recomputed per step from the step windows;
each distinct activation tuple gets its own cached v2 Compose, so stepping
is free. Because active state changes must reach already-spawned dataloader
workers, the transform reports when the dataloader must be reinitialized.

Mosaic (opt-in via ``mosaic_prob``; the AV-DETR family enables it) pastes
the current sample plus three cache draws onto a 2x2 canvas at the sample
level and applies ONE whole-canvas affine (baseline MosaicTransform
defaults: ±10 deg rotation, ±0.1 translation, 0.5-1.5 scaling); a mosaic
sample then skips the zoom-out and IoU-crop ops (baseline behavior) but
keeps photometric distort, flips, and the resize.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as TF

from autovision.augment.detection.box_aware_eraser import (
    erase_background,
)
from autovision.augment.mosaic_geometry import (
    apply_affine_to_boxes as _apply_affine_to_boxes,
)
from autovision.augment.mosaic_geometry import (
    mosaic_canvas_box_clamp as _mosaic_canvas_box_clamp,
)

__all__ = ["RTDETRDetectionTransform", "StepScheduledCompose"]


class StepScheduledCompose:
    """Caches one v2 Compose per activation tuple and switches per step.

    The step-state tuple is ``(mosaic, photometric, zoom_out, iou_crop,
    erase)`` — the window gates of every step-scheduled augmentation. A
    tuple change marks the dataloader for reinitialization so fresh worker
    processes observe the new gates (workers hold pickled copies of this
    object; in-place step updates would never reach them). The COMPOSE
    cache key is ``(mosaic_sample, photometric, zoom_out, iou_crop)`` —
    erasure is applied after the Compose, not inside it, so it takes no
    key slot; ``mosaic_sample=True`` means THIS sample came from the
    mosaic canvas and must skip zoom-out / IoU-crop.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.step = 0
        self._compose_cache: dict[tuple[bool, bool, bool, bool], v2.Compose] = {}
        self._active: tuple[bool, bool, bool, bool, bool] | None = None
        self._reinit_pending = False

    def _build(self, active: tuple[bool, bool, bool, bool]) -> v2.Compose:
        args = self.args
        _, photometric, zoom_out, iou_crop = active
        ops: list[Any] = []
        if photometric:
            ops.append(v2.RandomPhotometricDistort(p=args.photometric_distort_prob))
        if zoom_out:
            ops.append(v2.RandomZoomOut(fill=0.0, p=args.random_zoom_out_prob))
        if iou_crop:
            # torchvision's RandomIoUCrop has no probability parameter.
            ops.append(
                v2.RandomApply([v2.RandomIoUCrop()], p=args.random_iou_crop_prob)
            )
        if args.horizontal_flip_prob > 0:
            ops.append(v2.RandomHorizontalFlip(p=args.horizontal_flip_prob))
        if args.vertical_flip_prob > 0:
            ops.append(v2.RandomVerticalFlip(p=args.vertical_flip_prob))
        ops.append(v2.Resize(args.image_size, antialias=True))
        ops.append(v2.SanitizeBoundingBoxes(min_size=args.min_bbox_size_px))
        ops.append(v2.ToDtype(torch.float32, scale=True))
        ops.append(
            v2.Normalize(mean=list(args.normalize.mean), std=list(args.normalize.std))
        )
        return v2.Compose(ops)

    def _active_at(self, step: int) -> tuple[bool, bool, bool, bool, bool]:
        args = self.args

        def is_active(prob: float | None, start, stop) -> bool:
            return (
                prob is not None
                and prob > 0
                and start != "auto"
                and step >= int(start)
                and (stop is None or stop == "auto" or step < int(stop))
            )

        erase_prob = getattr(args, "random_erase_prob", None)
        erase_stop = getattr(args, "random_erase_step_stop", None)
        # Same window shape as the other ops but with no start: an
        # unresolved "auto" or None keeps the constant ratio, a resolved
        # int closes the window (Dome aug-stop cadence).
        erase_active = erase_prob is not None and (
            not isinstance(erase_stop, int) or step < int(erase_stop)
        )
        return (
            is_active(
                getattr(args, "mosaic_prob", None),
                getattr(args, "mosaic_step_start", None),
                getattr(args, "mosaic_step_stop", None),
            ),
            is_active(
                args.photometric_distort_prob,
                args.photometric_step_start,
                args.photometric_step_stop,
            ),
            is_active(
                args.random_zoom_out_prob,
                args.zoom_out_step_start,
                args.zoom_out_step_stop,
            ),
            is_active(
                args.random_iou_crop_prob,
                args.iou_crop_step_start,
                args.iou_crop_step_stop,
            ),
            erase_active,
        )

    def set_step(self, step: int) -> None:
        self.step = step
        active = self._active_at(step)
        if self._active is not None and active != self._active:
            self._reinit_pending = True
        self._active = active

    def compose_for(self, mosaic_sample: bool) -> v2.Compose:
        if self._active is None:
            self._active = self._active_at(self.step)
        _, photometric, zoom_out, iou_crop, _ = self._active
        if mosaic_sample:
            # Mosaic samples keep photometric distort and flips; only the
            # zoom-out / IoU-crop pair is skipped (baseline behavior).
            key = (True, photometric, False, False)
        else:
            key = (False, photometric, zoom_out, iou_crop)
        if key not in self._compose_cache:
            self._compose_cache[key] = self._build(key)
        return self._compose_cache[key]

    def mosaic_active(self) -> bool:
        """Whether the mosaic window is on at the current step (the per-
        sample coin flip happens in the transform)."""
        if self._active is None:
            self._active = self._active_at(self.step)
        return self._active[0]

    def erase_active(self) -> bool:
        """Whether the box-aware erasure window is on at the current step.

        Baked into the activation tuple (not compared against
        ``self.step`` at call time): dataloader workers hold copies of this
        object from the last reinitialization, so a call-time step read
        would see a stale value for any window that does not coincide with
        another op's boundary. As a tuple member, the window flips trigger
        the engine's dataloader reinitialization and every worker starts
        with the correct gate.
        """
        if self._active is None:
            self._active = self._active_at(self.step)
        return self._active[4]

    def requires_dataloader_reinitialization(self) -> bool:
        return self._reinit_pending

    def mark_dataloader_as_reinitialized(self) -> None:
        self._reinit_pending = False





class RTDETRDetectionTransform:
    """Per-sample detection transform with step scheduling and mosaic.

    Mosaic keeps a rolling cache of recent raw samples (per dataloader
    worker, mirroring the baseline's in-transform cache) and, on a coin
    flip inside its step window, pastes the current sample plus three
    cached ones onto a 2x2 ``mosaic_output_size``-tile canvas.
    """

    #: Rolling cache depth (baseline MosaicArgs.max_cached_images).
    MOSAIC_CACHE_SIZE = 50
    #: Whole-canvas affine ranges (baseline MosaicArgs defaults).
    MOSAIC_ROTATION_RANGE = 10.0
    MOSAIC_TRANSLATION_RANGE = (0.1, 0.1)
    MOSAIC_SCALING_RANGE = (0.5, 1.5)

    def __init__(self, transform_args) -> None:
        self.transform_args = transform_args
        self.compose = StepScheduledCompose(transform_args)
        self._mosaic_cache: list[dict[str, Any]] = []

    def _mosaic_cache_append(self, item: dict[str, Any]) -> None:
        self._mosaic_cache.append(self._clone_sample(item))
        if len(self._mosaic_cache) > self.MOSAIC_CACHE_SIZE:
            # Baseline random_pop: evict a RANDOM index, excluding the
            # freshly appended last entry (randint is inclusive).
            del self._mosaic_cache[random.randint(0, len(self._mosaic_cache) - 2)]

    def set_step(self, step: int) -> None:
        self.compose.set_step(step)

    def requires_dataloader_reinitialization(self) -> bool:
        return self.compose.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self.compose.mark_dataloader_as_reinitialized()

    # ── mosaic ──────────────────────────────────────────────────────────
    def _mosaic_hit(self) -> bool:
        args = self.transform_args
        prob = getattr(args, "mosaic_prob", None)
        # Baseline composes even when the cache holds fewer than four
        # samples (donor draws are with replacement).
        return self.compose.mosaic_active() and random.random() < (prob or 0.0)

    def _build_mosaic(self, item: dict[str, Any]) -> dict[str, Any]:
        # Baseline order: the current sample entered the cache in
        # __call__ BEFORE the donors are drawn (every sample is cached,
        # mosaic hit or not); the three donors are WITH-replacement draws
        # from the whole cache (duplicates allowed). The current sample
        # is the first tile, matching the baseline tile order.
        donors = random.choices(self._mosaic_cache, k=3)
        tiles = [item, *donors]

        tile = int(self.transform_args.mosaic_output_size)
        # Baseline tile resize: shorter side -> output_size, aspect kept.
        resized = [TF.resize(s["image"], [tile], antialias=True) for s in tiles]
        max_h = max(image.shape[1] for image in resized)
        max_w = max(image.shape[2] for image in resized)
        canvas = torch.zeros(
            (3, 2 * max_h, 2 * max_w), dtype=resized[0].dtype
        )
        boxes_out: list[torch.Tensor] = []
        classes_out: list[torch.Tensor] = []
        for index, (sample, image) in enumerate(zip(tiles, resized)):
            row, col = divmod(index, 2)
            y0, x0 = row * max_h, col * max_w
            _, height, width = image.shape
            canvas[:, y0 : y0 + height, x0 : x0 + width] = image
            boxes: torch.Tensor = tv_tensors.BoundingBoxes.as_subclass(
                sample["bboxes"], torch.Tensor
            )
            if boxes.numel() == 0:
                continue
            scaled = boxes.clone().float()
            scaled[:, [0, 2]] *= width / sample["image"].shape[2]
            scaled[:, [1, 3]] *= height / sample["image"].shape[1]
            scaled[:, [0, 2]] += x0
            scaled[:, [1, 3]] += y0
            boxes_out.append(scaled)
            classes_out.append(sample["classes"])
        merged = (
            torch.cat(boxes_out, dim=0)
            if boxes_out
            else torch.zeros(0, 4)
        )
        merged_classes = (
            torch.cat(classes_out, dim=0)
            if classes_out
            else torch.zeros(0, dtype=torch.int64)
        )
        merged = _mosaic_canvas_box_clamp(
            merged, (2 * max_h, 2 * max_w)
        )

        # One whole-canvas affine (baseline _sample_affine_params +
        # _apply_affine_to_image/_to_boxes on the pasted canvas).
        angle, (tx, ty), scale_factor = self._sample_affine_params(
            canvas.shape[1], canvas.shape[2]
        )
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
        merged = _apply_affine_to_boxes(
            merged, (canvas.shape[1], canvas.shape[2]), angle, (tx, ty), scale_factor
        )
        # Torchvision clips affine boxes to the canvas but keeps
        # degenerate ones — drop them here (the pipeline's sanitize
        # filters any remainder).
        keep = (merged[:, 2] > merged[:, 0]) & (merged[:, 3] > merged[:, 1])
        merged = merged[keep]
        merged_classes = merged_classes[keep]

        merged_boxes = tv_tensors.BoundingBoxes(
            merged, format="XYXY", canvas_size=(canvas.shape[1], canvas.shape[2])
        )
        return {
            "image_path": item["image_path"],
            "image": canvas,
            "bboxes": merged_boxes,
            "classes": merged_classes,
            "original_size": item["original_size"],
        }

    def _sample_affine_params(
        self, canvas_height: int, canvas_width: int
    ) -> tuple[float, tuple[int, int], float]:
        """Baseline _sample_affine_params (torch RNG, canvas-sized)."""
        angle = torch.empty(1).uniform_(
            -self.MOSAIC_ROTATION_RANGE, self.MOSAIC_ROTATION_RANGE
        ).item()
        max_dx = self.MOSAIC_TRANSLATION_RANGE[0] * canvas_width
        max_dy = self.MOSAIC_TRANSLATION_RANGE[1] * canvas_height
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        scale_factor = torch.empty(1).uniform_(*self.MOSAIC_SCALING_RANGE).item()
        return angle, (tx, ty), scale_factor

    @staticmethod
    def _clone_sample(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": item["image"].clone(),
            "bboxes": item["bboxes"].clone(),
            "classes": item["classes"].clone(),
        }

    # ── pipeline ────────────────────────────────────────────────────────
    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if getattr(self.transform_args, "mosaic_prob", None) is not None:
            # Keep a rolling cache of raw samples for the mosaic canvas;
            # skipped entirely when mosaic is off (no per-sample copies).
            self._mosaic_cache_append(item)
        mosaic_sample = self._mosaic_hit()
        if mosaic_sample:
            item = self._build_mosaic(item)
        image: torch.Tensor = item["image"]
        boxes: tv_tensors.BoundingBoxes = item["bboxes"]
        transformed = self.compose.compose_for(mosaic_sample)(
            {"image": image, "boxes": boxes, "labels": item["classes"]}
        )
        image_out = transformed["image"]
        boxes_out: tv_tensors.BoundingBoxes = transformed["boxes"]
        labels_out = transformed["labels"]
        # Back to normalized cxcywh for the criterion.
        _, height, width = image_out.shape
        boxes_out = boxes_out.to(torch.float32)
        cx = (boxes_out[:, 0] + boxes_out[:, 2]) / 2 / width
        cy = (boxes_out[:, 1] + boxes_out[:, 3]) / 2 / height
        w = (boxes_out[:, 2] - boxes_out[:, 0]) / width
        h = (boxes_out[:, 3] - boxes_out[:, 1]) / height
        bboxes_norm = torch.stack([cx, cy, w, h], dim=-1)

        # Box-aware background erasure (train views only — the val recipe
        # clears random_erase_prob): per-view coin flip, matching the
        # rf-detr-ssl-v2 dataset wrapper's per-view sampling semantics. The
        # window gate rides the step activation tuple (see
        # StepScheduledCompose.erase_active), so a stop boundary flips the
        # gate and triggers the dataloader reinitialization that carries it
        # to the workers.
        erase_prob = getattr(self.transform_args, "random_erase_prob", None)
        if (
            erase_prob is not None
            and self.compose.erase_active()
            and random.random() < erase_prob
        ):
            image_out = erase_background(
                image_out,
                bboxes_norm,
                self.transform_args.random_erase_ratio,
                padding=self.transform_args.random_erase_padding,
                cell=self.transform_args.random_erase_cell,
                interior=self.transform_args.random_erase_interior,
                min_keep=self.transform_args.random_erase_min_keep,
            )
        return {
            "image_path": item["image_path"],
            "image": image_out,
            "bboxes": bboxes_norm,
            "classes": labels_out,
            "original_size": item["original_size"],
        }


def build_val_transform_args(train_args):
    """Disable all scheduled augmentations for the val transform.

    The small-box filter is also relaxed to the torchvision minimum: at
    eval time, dropping sub-threshold objects with the *training* threshold
    would silently delete tiny GT (dense aerial datasets) and distort the
    mAP; only sub-pixel degenerate boxes are removed.
    """
    val_args = train_args.model_copy(deep=True)
    val_args.photometric_distort_prob = None
    val_args.random_zoom_out_prob = None
    val_args.random_iou_crop_prob = None
    val_args.mosaic_prob = None
    val_args.random_erase_prob = None
    val_args.horizontal_flip_prob = 0.0
    val_args.vertical_flip_prob = 0.0
    val_args.scale_jitter = None
    val_args.min_bbox_size_px = 1.0
    return val_args
