# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PicoDet postprocessing: sigmoid scores, top-k, class-offset NMS, rescale.

The exported graph emits raw logits + normalized cxcywh boxes; the
coordinate contract (single rescale to original-frame pixels) matches the
shared DETR-family decode. Unlike the DETR families, the dense head keeps
duplicate detections, so a class-offset NMS finishes the decode.
"""

from __future__ import annotations

from typing import Any

from torchvision.ops import nms

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
)

__all__ = ["decode_pico_outputs"]


def decode_pico_outputs(
    raw_outputs: Any,
    metadata: list[dict[str, Any]],
    num_top_queries: int = 300,
    threshold: float = 0.0,
    nms_iou_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Decode raw head outputs to per-image boxes in original pixels.

    ``raw_outputs``: ``(logits, boxes)`` tuple or object with
    ``.logits``/``.boxes`` — logits (B, N, C) raw class scores, boxes
    (B, N, 4) normalized cxcywh. ``metadata``: per-image
    ``{"orig_w", "orig_h"}``.
    """
    if isinstance(raw_outputs, tuple):
        logits, boxes = raw_outputs
    else:
        logits, boxes = raw_outputs.logits, raw_outputs.boxes
    scores = logits.sigmoid()
    flat_scores = scores.flatten(1)
    num_top = min(num_top_queries, flat_scores.shape[1])
    scores_flat, indices = flat_scores.topk(num_top, dim=1)
    classes = indices % scores.shape[2]
    anchor_indices = indices // scores.shape[2]
    results = []
    for batch_index, meta in enumerate(metadata):
        keep = scores_flat[batch_index] > threshold
        labels = classes[batch_index][keep]
        confidences = scores_flat[batch_index][keep]
        xyxy = box_cxcywh_norm_to_xyxy(
            boxes[batch_index, anchor_indices[batch_index]][keep],
            (float(meta["orig_w"]), float(meta["orig_h"])),
        )
        if labels.numel() > 0:
            # Class-offset NMS: shift boxes per class so cross-class
            # overlaps never suppress each other.
            offset = float(max(meta["orig_w"], meta["orig_h"])) + 1.0
            instance = nms(
                xyxy + labels.to(xyxy.dtype).unsqueeze(-1) * offset,
                confidences,
                nms_iou_threshold,
            )
            labels = labels[instance]
            confidences = confidences[instance]
            xyxy = xyxy[instance]
        results.append({"labels": labels, "bboxes": xyxy, "scores": confidences})
    return results
