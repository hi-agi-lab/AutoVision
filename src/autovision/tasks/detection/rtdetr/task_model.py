# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR object detection task model (inference/export facing)."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from autovision.augment.transform_args import NormalizeArgs
from autovision.encoders import get_wrapped_model
from autovision.export import ONNXExportMixin, ONNXExportPrecisionPolicy
from autovision.tasks.detection.shared.decode import (
    decode_detection_outputs,
)
from autovision.tasks.detection.shared.hybrid_encoder import (
    HybridEncoder,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    RTDETRv2Decoder,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import BaseModelOutput

__all__ = ["ObjectDetectionOutput", "RTDETRObjectDetection"]

# Backbones supported for multi-level feature extraction.
_BACKBONE_LEVELS = {
    "resnet18": ["layer2", "layer3", "layer4"],
    "resnet34": ["layer2", "layer3", "layer4"],
    "resnet50": ["layer2", "layer3", "layer4"],
    "convnext_tiny": None,  # handled via features chunking below
}


class ObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes."""

    def __init__(self, logits: torch.Tensor, boxes: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes


class RTDETRObjectDetection(ONNXExportMixin, TaskNetwork):
    """Three-stage RT-DETR-family detector.

    ``forward_backend`` returns :class:`ObjectDetectionOutput` (class logits
    and normalized cxcywh boxes); decoding (top-k, rescaling to pixels) stays
    in ``postprocess`` so the exported graph stays minimal.
    """

    onnx_export_precision_policy = ONNXExportPrecisionPolicy(
        fp32_onnx_op_types=("Softmax", "MatMul")
    )
    #: Export-side postprocess contract marker (see export/postprocess.py).
    postprocess_spec_family = "detr"

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
        load_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights", "kwargs", "self"})
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.num_top_queries = num_top_queries
        self.normalize = NormalizeArgs()

        backbone_wrapper = get_wrapped_model(
            backbone, num_input_channels=3, load_weights=load_weights
        )
        raw_model = backbone_wrapper.raw_module()
        levels = self._feature_levels(raw_model, backbone)
        in_channels = self._level_channels(raw_model, levels)
        self.feature_extractor = create_feature_extractor(
            raw_model, {level: level for level in levels}
        )
        self.encoder = HybridEncoder(in_channels, hidden_dim=hidden_dim)
        # Full RT-DETRv2 decoder: grid anchor priors + encoder proposal
        # top-k selection + contrastive denoising + iterative refinement.
        self.decoder = RTDETRv2Decoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * len(levels),
            feat_strides=[8, 16, 32][: len(levels)],
            num_levels=len(levels),
            nhead=num_heads,
            num_layers=dec_layers,
            num_denoising=num_denoising,
        )

    # ── backbone plumbing ───────────────────────────────────────────────
    @staticmethod
    def _feature_levels(raw_model: nn.Module, backbone: str) -> list[str]:
        short_name = backbone.split("/")[-1]
        if short_name.startswith("resnet"):
            return ["layer2", "layer3", "layer4"]
        raise ValueError(
            f"Backbone '{backbone}' is not supported for the RT-DETR family; "
            f"supported: torchvision resnets (resnet18/34/50)."
        )

    @staticmethod
    def _level_channels(raw_model: nn.Module, levels: list[str]) -> list[int]:
        channels = []
        for level in levels:
            module = dict(raw_model.named_modules())[level]
            last_conv = None
            for sub in module.modules():
                if isinstance(sub, nn.Conv2d):
                    last_conv = sub
            channels.append(int(last_conv.out_channels))
        return channels

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

    def _forward_train(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict:
        features = list(self.feature_extractor(x).values())
        feats = self.encoder(features)
        return self.decoder(feats, targets=targets)

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
        """Decode raw outputs to per-image boxes in original-frame pixels.

        Delegates to the shared DETR-family decoder; the coordinate
        contract (normalized cxcywh -> original pixels, exactly one
        scaling) lives in
        :mod:`autovision.tasks.detection.shared.decode`.
        """
        return decode_detection_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
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
