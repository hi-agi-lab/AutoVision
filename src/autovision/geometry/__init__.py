# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pure geometry math shared across augment / metrics / tasks layers.

No torch.nn modules and no framework imports beyond torch itself, so every
layer may import downward from here (design doc 30: this retires the two
``ALLOWED_UPWARD`` debts for box math).
"""

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
    generalized_box_iou_loss,
)
from autovision.geometry.boxes import (
    box_area,
    box_iou,
    box_non_max_merge,
    box_non_max_suppression,
    clip_boxes,
    denormalize_boxes,
    move_boxes,
    pad_boxes,
    scale_boxes,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from autovision.geometry.masks import (
    approximate_polygon,
    filter_polygons_by_area,
    mask_to_polygons,
    mask_to_xyxy,
    polygon_area,
    polygon_to_mask,
    polygon_to_xyxy,
    xyxy_to_mask,
)

__all__ = [
    "approximate_polygon",
    "box_area",
    "box_cxcywh_norm_to_xyxy",
    "box_iou",
    "box_non_max_merge",
    "box_non_max_suppression",
    "box_xyxy_to_cxcywh_norm",
    "clip_boxes",
    "denormalize_boxes",
    "filter_polygons_by_area",
    "generalized_box_iou_loss",
    "mask_to_polygons",
    "mask_to_xyxy",
    "move_boxes",
    "pad_boxes",
    "polygon_area",
    "polygon_to_mask",
    "polygon_to_xyxy",
    "scale_boxes",
    "xcycwh_to_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_xcycarh",
    "xyxy_to_xywh",
    "xyxy_to_mask",
]
