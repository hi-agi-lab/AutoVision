# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotation-interchange converters (roadmap 90d, phase 3).

The converts feed the existing COCO loader: absolute xywh boxes, optional
polygon segmentation and a sorted category table. They are deliberately
file-based (no vendor server/API): the customer exports from CVAT / Label
Studio, converts, and trains with the same data config as any COCO set.
"""

from __future__ import annotations

from autovision.data.interchange.cvat import cvat_to_coco
from autovision.data.interchange.label_studio import label_studio_to_coco

__all__ = ["cvat_to_coco", "label_studio_to_coco"]
