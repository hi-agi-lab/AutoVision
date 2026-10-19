# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PicoDet object detection task model (inference/export facing)."""

from __future__ import annotations

import pathlib
from typing import Any

import torch

from autovision.augment.transform_args import NormalizeArgs
from autovision.export import ONNXExportMixin, ONNXExportPrecisionPolicy
from autovision.tasks.detection.picodet.csp_pan import CSPPAN
from autovision.tasks.detection.picodet.esnet import (
    ESNet,
    resolve_backbone_scale,
)
from autovision.tasks.detection.picodet.pico_head import PicoHead
from autovision.tasks.detection.picodet.postprocessor import (
    decode_pico_outputs,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import BaseModelOutput

__all__ = ["ObjectDetectionOutput", "PicoDetObjectDetection"]


class ObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes."""

    def __init__(self, logits: torch.Tensor, boxes: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes


class PicoDetObjectDetection(ONNXExportMixin, TaskNetwork):
    """Three-stage PicoDet detector (ESNet -> CSP-PAN -> dense PicoHead).

    ``forward_backend`` returns ``(logits, boxes)`` — raw class scores and
    normalized cxcywh boxes — matching the shared detection output
    contract; NMS-based decoding stays in ``postprocess``.
    """

    onnx_export_precision_policy = ONNXExportPrecisionPolicy(
        fp32_onnx_op_types=("Softmax",)
    )
    #: Export-side postprocess contract marker (see export/postprocess.py).
    postprocess_spec_family = "picodet"

    def __init__(
        self,
        num_classes: int,
        backbone: str = "s",
        image_size: tuple[int, int] = (640, 640),
        neck_channels: int = 96,
        reg_max: int = 7,
        num_top_queries: int = 300,
        nms_iou_threshold: float = 0.6,
        load_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights", "kwargs", "self"})
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.num_top_queries = num_top_queries
        self.nms_iou_threshold = nms_iou_threshold
        self.normalize = NormalizeArgs()

        scale = resolve_backbone_scale(backbone)
        self.backbone = ESNet(scale=scale)
        self.neck = CSPPAN(self.backbone.out_channels, hidden_channels=neck_channels)
        self.head = PicoHead(
            num_classes=num_classes,
            in_channels=self.neck.out_channels,
            strides=(8, 16, 32),
            reg_max=reg_max,
        )

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(
        self, image: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as TF

        from autovision.data.file_helpers import open_image

        if isinstance(image, (str, pathlib.Path)):
            tensor = open_image(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(image).float() / 255.0
        _, height, width = tensor.shape
        resized = TF.resize(
            tensor,
            [self.image_size[0], self.image_size[1]],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def _forward_train(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.neck(self.backbone(x))
        return self.head(feats)

    def forward_backend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, boxes); the tuple form keeps the export graph
        free of custom dataclasses (dynamo limitation)."""
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw) -> ObjectDetectionOutput:
        if isinstance(raw, ObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return ObjectDetectionOutput(logits=logits, boxes=boxes)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decode raw outputs to per-image boxes in original-frame pixels
        (top-k + class-offset NMS + single rescale to the original frame)."""
        return decode_pico_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, *self.image_size), dtype=torch.float32, is_batched=True
                )
            }
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
