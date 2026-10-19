# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE object detection task model (inference/export facing)."""

from __future__ import annotations

from typing import Any

from autovision.tasks.detection.rtdetr.task_model import (
    RTDETRObjectDetection,
)
from autovision.tasks.detection.shared.dfine_decoder import (
    DFINEv2Decoder,
)

__all__ = ["DFINEObjectDetection"]


class DFINEObjectDetection(RTDETRObjectDetection):
    """RT-DETR task model with the D-FINE distributional decoder.

    Everything except the decoder is inherited verbatim: backbone plumbing,
    three-stage preprocess/forward/postprocess contract, ONNX export policy,
    and the shared top-k decode. The base constructor builds an RTDETRv2
    decoder which is immediately replaced by :class:`DFINEv2Decoder` with
    the same shape parameters.
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
        reg_max: int = 32,
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
        num_levels = len(self.decoder.feat_strides)
        self.decoder = DFINEv2Decoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * num_levels,
            feat_strides=list(self.decoder.feat_strides),
            num_levels=num_levels,
            nhead=num_heads,
            num_layers=dec_layers,
            num_denoising=num_denoising,
            reg_max=reg_max,
        )
        # reg_max must survive into the export contract (the base records
        # only its own __init__ locals), or a non-default bin layout cannot
        # be reconstructed when loading the exported model.
        self._init_args["reg_max"] = reg_max
