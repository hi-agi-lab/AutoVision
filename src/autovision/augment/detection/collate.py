# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection batch collation: batch scale jitter, mixup, box filtering.

Contract: consumes the **output of the per-sample transform** — float images
already normalized once, boxes as normalized cxcywh tensors — and produces
the training batch. Batch scale jitter resizes the whole batch to one shared
size (normalized coordinates are invariant under full-frame scaling), then
mixup may blend adjacent samples within its step window.
"""

from __future__ import annotations

import random
from typing import Any

import torch

__all__ = ["RTDETRDetectionCollateFunction", "TwoViewDetectionCollateFunction"]


class RTDETRDetectionCollateFunction:
    """Collates transform outputs into a training batch.

    1. batch scale jitter (one size for the whole batch; normalized boxes
       are unchanged by full-frame scaling),
    2. mixup within its step window (>= 2 samples),
    3. degenerate-box filtering and stacking.

    Output: ``{"image": (B,C,H,W) float normalized, "bboxes": list[(n,4)]
    normalized cxcywh, "classes": list[(n,)], "image_path": list[str],
    "original_size": list[(w,h)]}`` — original_size is the TRUE original
    image size (pre-resize), used by postprocess to map boxes back.
    """

    def __init__(self, transform_args, train: bool = True) -> None:
        self.transform_args = transform_args
        self.train = train
        self.step = 0
        self._reinit_pending = False
        self._mixup_active: bool | None = None
        self._copyblend_active: bool = False
        self._scale_jitter_active: bool | None = None

    def _scale_jitter_on_at(self, step: int) -> bool:
        scale_jitter = self.transform_args.scale_jitter
        if scale_jitter is None:
            return False
        stop = getattr(scale_jitter, "step_stop", None)
        stop = None if stop in (None, "auto") else int(stop)
        return stop is None or step < stop

    def set_step(self, step: int) -> None:
        self.step = step
        scale_jitter_active = self.train and self._scale_jitter_on_at(step)
        if not self.train:
            self._mixup_active = False
            if self._scale_jitter_active is not None and (
                scale_jitter_active != self._scale_jitter_active
            ):
                self._reinit_pending = True
            self._scale_jitter_active = scale_jitter_active
            return
        start = getattr(self.transform_args, "mixup_step_start", 0)
        stop = getattr(self.transform_args, "mixup_step_stop", None)
        start = None if start == "auto" else int(start)
        stop = None if stop in (None, "auto") else int(stop)
        active = (
            self.transform_args.mixup_prob is not None
            and start is not None
            and step >= start
            and (stop is None or step < stop)
        )
        if self._mixup_active is not None and active != self._mixup_active:
            self._reinit_pending = True
        self._mixup_active = active
        cb_stop = getattr(self.transform_args, "copyblend_step_stop", None)
        cb_stop = None if cb_stop in (None, "auto") else int(cb_stop)
        cb_start = getattr(self.transform_args, "copyblend_step_start", None)
        # Baseline CopyBlend gate is [step_start, step_stop) and FAILS
        # CLOSED: an unresolved start keeps the op OFF (the train pipeline
        # resolves "auto" in resolve_step_schedule).
        cb_start = None if cb_start in (None, "auto") else int(cb_start)
        self._copyblend_active = bool(
            self.transform_args.copyblend_prob is not None
            and cb_start is not None
            and step >= cb_start
            and (cb_stop is None or step < cb_stop)
        )
        if self._scale_jitter_active is not None and (
            scale_jitter_active != self._scale_jitter_active
        ):
            self._reinit_pending = True
        self._scale_jitter_active = scale_jitter_active

    def requires_dataloader_reinitialization(self) -> bool:
        return bool(self._reinit_pending)

    def mark_dataloader_as_reinitialized(self) -> None:
        self._reinit_pending = False

    def _copyblend_batch(
        self,
        images: list[torch.Tensor],
        boxes_list: list[torch.Tensor],
        classes_list: list[torch.Tensor],
    ) -> None:
        """Copy-paste blending (baseline CopyBlend, tensor form).

        Object pool from the whole batch (boxes above the pixel-area
        threshold); pastes up to ``num_objects`` crops into each image with
        a random expand ratio and beta blend; appends the source boxes
        remapped to the pasted placement. Simplification vs the baseline:
        random placement without collision avoidance.
        """
        args = self.transform_args
        h, w = images[0].shape[-2], images[0].shape[-1]
        pool = [
            (index, i)
            for index, boxes in enumerate(boxes_list)
            for i in range(boxes.shape[0])
            if boxes[i, 2] * boxes[i, 3] * w * h >= args.copyblend_area_threshold
        ]
        if not pool:
            return
        beta = random.uniform(0.45, 0.55)
        for image_index, image in enumerate(images):
            for _ in range(args.copyblend_num_objects):
                src_index, box_index = random.choice(pool)
                cx, cy, bw, bh = boxes_list[src_index][box_index].tolist()
                ratio = random.uniform(*args.copyblend_expand_ratios)
                bw, bh = bw * (1 + ratio), bh * (1 + ratio)
                x1 = int(max((cx - bw / 2) * w, 0.0))
                y1 = int(max((cy - bh / 2) * h, 0.0))
                x2 = int(min((cx + bw / 2) * w, float(w)))
                y2 = int(min((cy + bh / 2) * h, float(h)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                src = images[src_index]
                images[image_index][:, y1:y2, x1:x2] = (
                    beta * image[:, y1:y2, x1:x2] + (1.0 - beta) * src[:, y1:y2, x1:x2]
                )
                boxes_list[image_index] = torch.cat(
                    [
                        boxes_list[image_index],
                        torch.tensor(
                            [[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh]],
                            dtype=boxes_list[image_index].dtype,
                        ),
                    ],
                    dim=0,
                )
                classes_list[image_index] = torch.cat(
                    [
                        classes_list[image_index],
                        classes_list[src_index][box_index : box_index + 1],
                    ],
                    dim=0,
                )

    def _filter_small_boxes(
        self, boxes: torch.Tensor, classes: torch.Tensor, width: int, height: int
    ):
        if boxes.numel() == 0:
            return boxes, classes
        keep = (boxes[:, 2] * width >= self.transform_args.min_bbox_size_px) & (
            boxes[:, 3] * height >= self.transform_args.min_bbox_size_px
        )
        return boxes[keep], classes[keep]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        args = self.transform_args
        images, boxes_list, classes_list = [], [], []
        original_sizes = []
        input_size: tuple[int, int] | None = None

        if self._scale_jitter_active is None:
            # Direct use without set_step: keep the always-on legacy behavior.
            self._scale_jitter_active = self.train and self._scale_jitter_on_at(
                self.step
            )
        scale_jitter = self._scale_jitter_active
        jitter_size = (
            random.choice(self.transform_args.scale_jitter.sizes)
            if scale_jitter
            else None
        )

        for item in batch:
            image: torch.Tensor = item["image"]  # float, normalized, (C,H,W)
            boxes: torch.Tensor = item["bboxes"]  # normalized cxcywh
            classes: torch.Tensor = item["classes"]
            if jitter_size is not None:
                image = torch.nn.functional.interpolate(
                    image[None],
                    size=(jitter_size, jitter_size),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                # Normalized coordinates are invariant under full-frame scaling.
            images.append(image)
            boxes_list.append(boxes)
            classes_list.append(classes)
            original_sizes.append(tuple(item["original_size"]))

        input_size = (images[0].shape[-1], images[0].shape[-2])

        if (
            self._mixup_active
            and len(batch) >= 2
            and args.mixup_prob is not None
            and random.random() < args.mixup_prob
        ):
            beta = random.uniform(0.45, 0.55)
            stacked = torch.stack(images)
            mixed = beta * stacked + (1.0 - beta) * stacked.roll(1, 0)
            images = [m for m in mixed]
            boxes_list = [
                torch.cat([boxes_list[i], boxes_list[i - 1]], dim=0)
                for i in range(len(boxes_list))
            ]
            classes_list = [
                torch.cat([classes_list[i], classes_list[i - 1]], dim=0)
                for i in range(len(classes_list))
            ]

        elif (
            self._copyblend_active
            and len(batch) >= 1
            and args.copyblend_prob is not None
            and random.random() < args.copyblend_prob
        ):
            self._copyblend_batch(images, boxes_list, classes_list)

        filtered = [
            self._filter_small_boxes(b, c, input_size[0], input_size[1])
            for b, c in zip(boxes_list, classes_list)
        ]
        boxes_list = [b for b, _ in filtered]
        classes_list = [c for _, c in filtered]

        return {
            "image": torch.stack(images),
            "bboxes": boxes_list,
            "classes": classes_list,
            "image_path": [item["image_path"] for item in batch],
            "original_size": original_sizes,
            "input_size": input_size,
        }


class TwoViewDetectionCollateFunction:
    """Collate ``(view1, view2)`` item pairs into one view-major doubled batch.

    Wraps a base detection collate for the SSL two_crop pipeline (design
    doc 14 §4.5): the pairs are flattened view-major — rows ``r`` and
    ``r + B`` are the two augmented views of the same source image — and
    handed to the base collate in a SINGLE call. One call means one shared
    batch scale-jitter draw and one joint batch: splitting the views into
    two collate calls would need joint padding across different scales,
    which would break the normalized-cxcywh coordinate contract.

    Mixup / mosaic / small-box filtering keep their base behavior: mixed-in
    neighbor content simply fails the cross-view IoU+label pairing and is
    dropped from the contrastive pool (diagnosed via ``ssl_paired_ratio``).

    The step-scheduling hooks are delegated so the wrapper can take the
    base collate's place in the engine's ``step_hooks`` list.
    """

    def __init__(self, base: RTDETRDetectionCollateFunction) -> None:
        self.base = base

    def set_step(self, step: int) -> None:
        self.base.set_step(step)

    def requires_dataloader_reinitialization(self) -> bool:
        return self.base.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self.base.mark_dataloader_as_reinitialized()

    @property
    def step(self) -> int:
        return self.base.step

    def __call__(
        self, batch: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> dict[str, Any]:
        # View-major flattening (SimCSE convention): all view-1 items first,
        # then all view-2 items — row r and row r + B share the source
        # image, which is the boundary the SSL criterion splits on.
        views1 = [pair[0] for pair in batch]
        views2 = [pair[1] for pair in batch]
        return self.base(views1 + views2)
