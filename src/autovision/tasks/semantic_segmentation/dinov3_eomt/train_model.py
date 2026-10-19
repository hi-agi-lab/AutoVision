# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation training wrapper, DINOv3 variant."""

from __future__ import annotations

from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.tasks.semantic_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtSemanticSegmentationTrain,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.config import (
    Dinov3EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov3_eomt.task_model import (
    Dinov3EomtSemanticSegmentation,
)
from autovision.tasks.train_model import TrainModel

__all__ = ["Dinov3EomtSemanticSegmentationTrain"]


class Dinov3EomtSemanticSegmentationTrain(Dinov2EomtSemanticSegmentationTrain):
    """DINOv3 variant of the EOMT semantic training wrapper. Only the
    task-model construction differs; losses, metrics, optimization and
    export behavior are inherited."""

    TASK_MODEL_CLS = Dinov3EomtSemanticSegmentation

    def __init__(self, model_args: Dinov3EomtSemanticSegmentationTrainArgs) -> None:
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
        )
        # The DINOv2 wrapper's __init__ builds its own task model; initialize
        # the TrainModel base directly and replicate its bookkeeping.
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, SegmentationMetricCollector] = {}
