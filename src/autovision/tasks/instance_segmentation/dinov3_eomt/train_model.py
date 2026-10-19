# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation training wrapper, DINOv3 variant."""

from __future__ import annotations

from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.tasks.instance_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtInstanceSegmentationTrain,
)
from autovision.tasks.instance_segmentation.dinov3_eomt.config import (
    Dinov3EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.instance_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtInstanceSegmentation,
)
from autovision.tasks.train_model import TrainModel

__all__ = ["Dinov3EomtInstanceSegmentationTrain"]


class Dinov3EomtInstanceSegmentationTrain(Dinov2EomtInstanceSegmentationTrain):
    """DINOv3 variant of the EOMT instance training wrapper. Only the
    task-model construction differs; instance losses, COCO mask AP metrics
    and decode behavior are inherited."""

    TASK_MODEL_CLS = Dinov3EomtInstanceSegmentation

    def __init__(self, model_args: Dinov3EomtInstanceSegmentationTrainArgs) -> None:
        task_model = self.TASK_MODEL_CLS(
            num_classes=model_args.num_classes,
            image_size=model_args.image_size,
            num_queries=model_args.num_queries,
            num_blocks=model_args.num_blocks,
            num_upscale_blocks=model_args.num_upscale_blocks,
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            patch_size=model_args.patch_size,
            mlp_ratio=model_args.mlp_ratio,
            use_gated_mlp=model_args.use_gated_mlp,
            drop_path_rate=model_args.drop_path_rate,
            layer_norm_eps=model_args.layer_norm_eps,
            load_weights=model_args.load_weights,
            weight_source=model_args.weight_source,
            score_threshold=model_args.score_threshold,
            mask_threshold=model_args.mask_threshold,
            num_top_queries=model_args.num_top_queries,
        )
        # The DINOv2 wrapper chain builds its own task model; initialize the
        # TrainModel base directly and replicate its bookkeeping.
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, SegmentationMetricCollector] = {}
