# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT inference decode (query class x query mask -> semantic map).

Self-contained re-implementation of the HF image-processor semantic
post-processing so the three-stage task model does not depend on an
``EomtImageProcessor`` instance.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "semantic_segmentation_decode",
    "instance_segmentation_decode",
    "panoptic_segmentation_decode",
]


def semantic_segmentation_decode(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode EOMT queries to per-image semantic maps in original size.

    ``masks_queries_logits``: (B, Q, h, w) raw mask logits; ``class_queries_logits``:
    (B, Q, L+1) raw class logits (the last entry is the null class).
    ``metadata``: per-image ``{"orig_w", "orig_h"}``. Returns per-image
    ``{"mask": (H, W) long class ids, "input_h", "input_w"}``.
    """
    input_h, input_w = masks_queries_logits.shape[-2:]
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()
    segmentation_logits = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
    results = []
    for index, meta in enumerate(metadata):
        orig_h, orig_w = int(meta["orig_h"]), int(meta["orig_w"])
        logits = F.interpolate(
            segmentation_logits[index][None],
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        results.append(
            {
                "mask": logits.argmax(dim=0).long(),
                "input_h": input_h,
                "input_w": input_w,
            }
        )
    return results


def instance_segmentation_decode(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    metadata: list[dict[str, Any]],
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    num_top_queries: int = 100,
) -> list[dict[str, Any]]:
    """Decode EOMT queries to per-image instance masks in original size.

    Per query the non-null class probability is the instance score; the
    mask probability map is resized to the original frame and binarized.
    Returns per-image ``{"masks": (N,H,W) bool, "labels", "scores"}``
    sorted by descending score.
    """
    results = []
    class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
    for index, meta in enumerate(metadata):
        orig_h, orig_w = int(meta["orig_h"]), int(meta["orig_w"])
        best_score, best_label = class_probs[index].max(dim=-1)
        keep = best_score > score_threshold
        if keep.any():
            scores, order = best_score[keep].sort(descending=True)
            labels = best_label[keep][order]
            logits = masks_queries_logits[index][keep][order][:num_top_queries]
            scores = scores[:num_top_queries]
            labels = labels[:num_top_queries]
            probs = logits.sigmoid()
            probs = F.interpolate(
                probs[None], size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )[0]
            masks = probs >= mask_threshold
        else:
            scores = best_score.new_zeros(0)
            labels = best_label.new_zeros(0, dtype=torch.long)
            masks = torch.zeros(
                0, orig_h, orig_w, dtype=torch.bool, device=best_score.device
            )
        results.append({"masks": masks, "labels": labels, "scores": scores})
    return results


def panoptic_segmentation_decode(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    metadata: list[dict[str, Any]],
    num_labels: int,
    num_thing_classes: int,
    threshold: float = 0.8,
    mask_threshold: float = 0.5,
    mask_overlap_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Decode EOMT queries to per-image panoptic maps in original size.

    Mirrors the baseline ``get_image_masks_segment_ids_scores``: a query is
    kept when its best class is not the null class (last logit slot) and its
    confidence passes ``threshold``; per pixel the winning kept query claims
    the pixel if its mask probability passes ``mask_threshold`` and the
    claimed area retains ``mask_overlap_threshold`` of the query mask; same
    stuff-class segments merge into one segment id. Unassigned pixels carry
    the ignore label ``num_labels`` with segment id -1.

    Returns per-image ``{"masks": (H, W, 2) long (class_id, segment_id),
    "segment_ids": (S,), "scores": (S,)}`` with internal class ids.
    """
    results = []
    class_probs = class_queries_logits.softmax(dim=-1)
    for index, meta in enumerate(metadata):
        orig_h, orig_w = int(meta["orig_h"]), int(meta["orig_w"])
        class_logits = class_probs[index]  # (Q, K+1)
        score, label = class_logits.max(dim=-1)  # (Q,)
        keep = (label != num_labels) & (score > threshold)
        keep_indices = keep.nonzero(as_tuple=False).flatten()
        num_keep = keep_indices.numel()
        if num_keep:
            scores = score[keep_indices]
            labels = label[keep_indices]
            probs = F.interpolate(
                masks_queries_logits[index][keep_indices].sigmoid()[None],
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0]  # (n, H, W)
            # Winner per pixel: highest score-weighted mask probability.
            mask_scores = torch.cat(
                [scores[:, None, None] * probs, probs.new_full((1, orig_h, orig_w), -1)],
                dim=0,
            )
            winner = mask_scores.argmax(dim=0)  # (H, W) in [0, n]
            mask_orig = probs >= mask_threshold
            mask_new = winner[None] == torch.arange(num_keep, device=winner.device)[:, None, None]
            mask_final = mask_orig & mask_new
            area_orig = mask_orig.flatten(1).sum(1).double()
            area_new = mask_new.flatten(1).sum(1).double()
            area_final = mask_final.flatten(1).sum(1).double()
            area_ratio = area_new / area_orig.clamp(min=1.0)
            keep_area = (
                (area_orig > 0)
                & (area_new > 0)
                & (area_final > 0)
                & (area_ratio >= mask_overlap_threshold)
            )
            area_indices = keep_area.nonzero(as_tuple=False).flatten()
            mask_final = mask_final[area_indices]
            labels = labels[area_indices]
            scores = scores[area_indices]
            num_segments = area_indices.numel()
        else:
            scores = score.new_zeros(0)
            labels = label.new_zeros(0, dtype=torch.long)
            mask_final = torch.zeros(
                0, orig_h, orig_w, dtype=torch.bool, device=score.device
            )
            num_segments = 0

        # Stuff segments of the same class share the first segment's id.
        segment_ids = torch.arange(num_segments, device=labels.device)
        if num_segments:
            is_stuff = labels >= num_thing_classes
            stuff_to_segment = -labels.new_ones(num_labels)
            for position in range(num_segments):
                if bool(is_stuff[position]) and stuff_to_segment[labels[position]] < 0:
                    stuff_to_segment[labels[position]] = segment_ids[position]
            segment_ids = segment_ids.masked_scatter(
                is_stuff, stuff_to_segment[labels[is_stuff]]
            )
            # Reassign contiguous ids.
            present = torch.zeros(
                num_segments, dtype=torch.bool, device=segment_ids.device
            )
            present = present.scatter(0, segment_ids, torch.ones_like(segment_ids, dtype=torch.bool))
            contiguous = (present.cumsum(0) - 1).masked_fill(~present, -1)
            segment_ids = contiguous[segment_ids]

        # Per-pixel (class, segment) with -1 sentinels for unassigned pixels.
        def _per_pixel(values: torch.Tensor) -> torch.Tensor:
            per_pixel = mask_final * (values[:, None, None] + 1) - 1
            per_pixel = torch.cat(
                [per_pixel, per_pixel.new_full((1, orig_h, orig_w), -1)], dim=0
            )
            return per_pixel.max(dim=0).values

        label_per_pixel = _per_pixel(labels)
        label_per_pixel[label_per_pixel == -1] = num_labels  # ignore class
        segment_per_pixel = _per_pixel(segment_ids)
        results.append(
            {
                "masks": torch.stack([label_per_pixel, segment_per_pixel], dim=-1),
                "segment_ids": segment_ids,
                "scores": scores,
            }
        )
    return results
