# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Canonical key names for per-image inference result records.

Task models return plain dictionaries from ``postprocess``. The keys are part
of the public ``predict`` contract, so they live here instead of being
repeated as string literals in every consumer (visualization, serving, the
stream runtime, error analysis).
"""

from __future__ import annotations

__all__ = [
    "BBOXES",
    "DEPTH",
    "INPUT_H",
    "INPUT_W",
    "KEYPOINTS",
    "LABELS",
    "MASK",
    "MASKS",
    "POLYGONS",
    "SCORES",
    "SEGMENT_IDS",
]

#: Per-instance boxes, ``(N, 4)`` xyxy in original-frame pixels.
BBOXES = "bboxes"
#: Per-instance scores, ``(N,)``.
SCORES = "scores"
#: Per-instance class ids, ``(N,)``.
LABELS = "labels"
#: Per-instance masks ``(N, H, W)`` (instance segmentation) or a panoptic
#: label map ``(H, W, 2)`` with ``(class_id, segment_id)`` channels.
MASKS = "masks"
#: Pose keypoints, ``(N, K, 3)`` with ``[x, y, visibility]`` per point.
KEYPOINTS = "keypoints"
#: Oriented-box corners, ``(N, 4, 2)`` xy in original-frame pixels.
POLYGONS = "polygons"
#: Semantic-segmentation label map, ``(H, W)``.
MASK = "mask"
#: Depth map, ``(H, W)``.
DEPTH = "depth"
#: Panoptic segment ids, ``(S,)`` (one per region, not per pixel).
SEGMENT_IDS = "segment_ids"
#: Network input size recorded by segmentation postprocessors.
INPUT_W = "input_w"
INPUT_H = "input_h"
