# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Shared detection output decoding (model outputs -> original-frame boxes).

Used by every DETR-family task model (RT-DETR, AV-DETR). The coordinate
contract lives in exactly one place: ``pred_boxes`` are normalized cxcywh
on the network input canvas, and normalized coordinates are
resolution-independent, so mapping back to the original frame multiplies
by (orig_w, orig_h) exactly once.
"""

from __future__ import annotations

from typing import Any

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
)

__all__ = ["decode_detection_outputs"]


def decode_detection_outputs(
    raw_outputs: Any,
    metadata: list[dict[str, Any]],
    num_top_queries: int = 300,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Top-k decode raw DETR outputs to per-image boxes in original pixels.

    ``raw_outputs``: object with ``.logits``/``.boxes`` (or such a tuple) —
    ``logits`` (B, Q, C) raw class scores, ``boxes`` (B, Q, 4) normalized
    cxcywh. ``metadata``: per-image ``{"orig_w", "orig_h"}``.
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
    query_indices = indices // scores.shape[2]
    results = []
    for batch_index, meta in enumerate(metadata):
        keep = scores_flat[batch_index] > threshold
        box = boxes[batch_index, query_indices[batch_index]]
        w, h = float(meta["orig_w"]), float(meta["orig_h"])
        xyxy = box_cxcywh_norm_to_xyxy(box, (w, h))
        results.append(
            {
                "labels": classes[batch_index][keep],
                "bboxes": xyxy[keep],
                "scores": scores_flat[batch_index][keep],
            }
        )
    return results
