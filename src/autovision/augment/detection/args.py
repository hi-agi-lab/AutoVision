# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection transform arguments with step-window policies."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autovision.augment.task_transform import TaskTransformArgs
from autovision.augment.transform_args import NormalizeArgs
from autovision.config.base import PydanticConfig
from autovision.types import ImageSizeTuple

__all__ = [
    "OrientedObjectDetectionTransformArgs",
    "RTDETRDetectionTransformArgs",
    "ScaleJitterArgs",
]


class ScaleJitterArgs(PydanticConfig):
    sizes: list[int]
    prob: float = 1.0
    #: Step at which batch scale jitter turns OFF (the baseline disables it
    #: for the final no-augmentation window). ``None`` keeps it on for the
    #: whole run; ``"auto"`` resolves to the shared step schedule.
    step_stop: int | None | str = None


class OrientedObjectDetectionTransformArgs(TaskTransformArgs):
    """Oriented (YOLO-OBB) detection recipe.

    Rotated-box pipeline: flips mirror corners, resize scales corners,
    polygons are sanitized by px² shoelace area. IoU-crop / zoom-out /
    scale-jitter / mixup are omitted — the torchvision v2 implementations
    behind them are axis-aligned-box-only.
    """

    image_size: ImageSizeTuple = (640, 640)
    photometric_distort_prob: float | None = 0.5
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    min_bbox_size_px: float = 4.0


class RTDETRDetectionTransformArgs(TaskTransformArgs):
    """Detection recipe (train defaults; the val recipe disables them)."""

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort_prob: float | None = 0.5
    random_zoom_out_prob: float | None = 0.5
    random_iou_crop_prob: float | None = 0.8
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=lambda: ScaleJitterArgs(
            sizes=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        )
    )
    mixup_prob: float | None = 0.5
    #: Copy-paste blending (baseline AV-DETR CopyBlend, XOR with mixup in
    #: the collate): paste up to ``num_objects`` object crops sampled from
    #: the batch pool into each image, beta-blended. Default off; the
    #: VisDrone/official recipes disable both mixup and copyblend.
    copyblend_prob: float | None = None
    copyblend_area_threshold: int = 100
    copyblend_num_objects: int = 3
    copyblend_expand_ratios: tuple[float, float] = (0.1, 0.25)
    #: Step at which copy-paste blending turns ON (baseline CopyBlend
    #: window is ``[step_start, step_stop)`` with the start shared with
    #: the photometric family). An unresolved ``"auto"`` keeps it OFF —
    #: the baseline's fail-closed gate.
    copyblend_step_start: int | str = "auto"
    copyblend_step_stop: int | None | str = "auto"
    mixup_step_start: int | str = "auto"
    mixup_step_stop: int | None | str = "auto"
    #: Mosaic (4-tile sample-level paste; off by default — the AV-DETR
    #: family enables it per the baseline recipe). Mosaic samples skip the
    #: zoom-out and IoU-crop ops for that step.
    mosaic_prob: float | None = None
    mosaic_output_size: int = 320
    mosaic_step_start: int | str = "auto"
    mosaic_step_stop: int | None | str = "auto"
    #: Box-aware background erasure (rf-detr-ssl-v2 design doc 8): erase a
    #: uniform random subset of ``random_erase_cell``-px background cells per
    #: view while protecting every margin-dilated GT box (any-overlap), so
    #: Hungarian matching never sees an evidence-free box. None = off.
    random_erase_prob: float | None = None
    random_erase_ratio: float = Field(default=0.4, gt=0.0, le=1.0)
    random_erase_padding: float = Field(default=0.15, ge=0.0)
    #: Cell edge in px. On a ViT backbone this aligns with the patch size;
    #: on ConvNeXt (stride-4 stages) 16px is an empirical granularity
    #: choice, not a structural alignment.
    random_erase_cell: int = Field(default=16, ge=1)
    #: Constrained foreground erasure (design doc V2): box interiors become
    #: erasable behind a per-box reservation, so every object keeps at least
    #: ``random_erase_min_keep`` patches (boxes with at most that many cells
    #: stay fully protected).
    random_erase_interior: bool = False
    random_erase_min_keep: int = Field(default=2, ge=1)
    #: Step at which erasure turns OFF (the Dome aug-stop cadence: training
    #: ends on the dense distribution it deploys with). "auto" resolves
    #: to the shared step schedule; None keeps the constant ratio for
    #: the whole run (the reference-recommended first setting).
    random_erase_step_stop: int | None | str = "auto"
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    min_bbox_size_px: float = 4.0

    # Step windows (auto: resolved from the training schedule).
    photometric_step_start: int | str = "auto"
    photometric_step_stop: int | None | str = "auto"
    zoom_out_step_start: int | str = "auto"
    zoom_out_step_stop: int | None | str = "auto"
    iou_crop_step_start: int | str = "auto"
    iou_crop_step_stop: int | None | str = "auto"

    def resolve_auto_fields(self, **kwargs) -> None:
        num_channels = kwargs.get("num_channels")
        if self.image_size == "auto":
            default = kwargs.get("default_image_size", (640, 640))
            self.image_size = tuple(default)  # type: ignore[assignment]
        if self.num_channels == "auto":
            self.num_channels = (
                num_channels if num_channels is not None else len(self.normalize.mean)
            )

    def resolve_step_schedule(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        from autovision.augment.detection.schedule import (
            resolve_detection_step_schedule,
        )

        steps_per_epoch = max(
            1, train_num_batches // max(1, gradient_accumulation_steps)
        )
        schedule = resolve_detection_step_schedule(total_steps, steps_per_epoch)
        for op, start_attr, stop_attr in (
            ("photometric", "photometric_step_start", "photometric_step_stop"),
            ("zoom_out", "zoom_out_step_start", "zoom_out_step_stop"),
            ("iou_crop", "iou_crop_step_start", "iou_crop_step_stop"),
            ("mixup", "mixup_step_start", "mixup_step_stop"),
            ("mosaic", "mosaic_step_start", "mosaic_step_stop"),
        ):
            start, stop = schedule.window(op)
            if getattr(self, start_attr) == "auto":
                setattr(self, start_attr, start)
            if getattr(self, stop_attr) == "auto":
                setattr(self, stop_attr, stop)
        if self.scale_jitter is not None and self.scale_jitter.step_stop == "auto":
            self.scale_jitter.step_stop = schedule.stop
        if self.copyblend_step_start == "auto":
            # Baseline CopyBlend shares the photometric family's start.
            self.copyblend_step_start = schedule.start
        if self.copyblend_step_stop == "auto":
            # Same photometric-family stop window as the baseline; leaving
            # "auto" unresolved made the collate treat it as None and kept
            # copyblend on for the WHOLE run.
            self.copyblend_step_stop = schedule.stop
        if self.random_erase_step_stop == "auto":
            self.random_erase_step_stop = schedule.stop

    def resolve_incompatible(self) -> None:
        if self.num_channels != 3:
            self.photometric_distort_prob = None
