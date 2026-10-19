# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR oriented object detection task model (inference/export facing)."""

from __future__ import annotations

from typing import Any

import torch

from autovision.tasks.detection.rtdetr.task_model import (
    RTDETRObjectDetection,
)
from autovision.tasks.oriented_detection.shared.obb_decoder import (
    OrientedRTDETRDecoder,
    decode_oriented_detection_outputs,
)
from autovision.tasks.task_model_io import BaseModelOutput

__all__ = ["OrientedObjectDetectionOutput", "RTDETROrientedObjectDetection"]


class OrientedObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes +
    angles (radians, [-π/4, π/4])."""

    def __init__(
        self, logits: torch.Tensor, boxes: torch.Tensor, angles: torch.Tensor
    ) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes
        self.angles = angles


class RTDETROrientedObjectDetection(RTDETRObjectDetection):
    """RT-DETR with the oriented (OBB) decoder: per-layer angle heads.

    The box pipeline is inherited verbatim; the decoder swap adds a
    ``pred_angles`` output and the postprocess decodes to rotated polygons
    with per-class rotated NMS.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "torchvision/resnet18",
        image_size: tuple[int, int] = (640, 640),
        hidden_dim: int = 256,
        num_queries: int = 100,
        dec_layers: int = 3,
        num_heads: int = 8,
        num_top_queries: int = 300,
        num_denoising: int = 100,
        nms_iou_threshold: float = 0.65,
        load_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            image_size=image_size,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            dec_layers=dec_layers,
            num_heads=num_heads,
            num_top_queries=num_top_queries,
            num_denoising=num_denoising,
            load_weights=load_weights,
            **kwargs,
        )
        self.nms_iou_threshold = nms_iou_threshold
        # Replace the decoder with the oriented variant, mirroring the exact
        # construction parameters of the axis-aligned decoder it replaces.
        old_decoder = self.decoder
        self.decoder = OrientedRTDETRDecoder(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * len(old_decoder.feat_strides),
            feat_strides=list(old_decoder.feat_strides),
            num_levels=old_decoder.num_levels,
            nhead=num_heads,
            num_layers=dec_layers,
            num_denoising=num_denoising,
        )

    def forward_backend(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, boxes, angles); the tuple form keeps the export
        graph free of custom dataclasses (dynamo limitation)."""
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"], out["pred_angles"]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw) -> OrientedObjectDetectionOutput:
        if isinstance(raw, OrientedObjectDetectionOutput):
            return raw
        logits, boxes, angles = raw
        return OrientedObjectDetectionOutput(logits=logits, boxes=boxes, angles=angles)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decode raw oriented outputs to per-image polygons in original
        pixels (per-class rotated NMS included)."""
        return decode_oriented_detection_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
            iou_threshold=self.nms_iou_threshold,
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        torch.testing.assert_close(
            torch_outputs[0].sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[1].sum(dim=1),
            torch.as_tensor(onnx_outputs[1]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[2].sum(dim=1),
            torch.as_tensor(onnx_outputs[2]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
