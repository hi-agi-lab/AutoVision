# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT panoptic segmentation training wrapper."""

from __future__ import annotations

from typing import Any

import torch

from autovision.metrics.panoptic_segmentation import PanopticMetricCollector
from autovision.tasks.panoptic_segmentation.dinov2_eomt.config import (
    Dinov2EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.panoptic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtPanopticSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtSemanticSegmentationTrain,
)
from autovision.tasks.shared.eomt.postprocess import panoptic_segmentation_decode
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["Dinov2EomtPanopticSegmentationTrain"]


class Dinov2EomtPanopticSegmentationTrain(Dinov2EomtSemanticSegmentationTrain):
    """Per-segment binary masks feed the HF criterion directly; the PQ
    metric consumes the decoded (class, segment) maps against the dataset
    targets with void semantics (ignore/crowd/no-segment pixels excluded).
    """

    task = "panoptic_segmentation"
    watch_metric = "val_metric/pq"
    TASK_MODEL_CLS = Dinov2EomtPanopticSegmentation

    def __init__(self, model_args: Dinov2EomtPanopticSegmentationTrainArgs) -> None:
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
            use_swiglu_ffn=model_args.use_swiglu_ffn,
            drop_path_rate=model_args.drop_path_rate,
            load_weights=model_args.load_weights,
            weight_source=model_args.weight_source,
            num_thing_classes=model_args.num_thing_classes,
            threshold=model_args.threshold,
            mask_threshold=model_args.mask_threshold,
            mask_overlap_threshold=model_args.mask_overlap_threshold,
        )
        # The semantic wrapper's __init__ builds the task model from its own
        # fixed kwarg list; panoptic passes extra decode args, so initialize
        # the TrainModel base directly.
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        self.categories = {}
        self._things = list(range(model_args.num_thing_classes))
        self._stuffs = list(
            range(model_args.num_thing_classes, model_args.num_classes)
        )
        self._evaluators: dict[str, PanopticMetricCollector] = {}

    def _make_evaluator(self) -> PanopticMetricCollector:
        return PanopticMetricCollector(self._things, self._stuffs)

    # ── steps ───────────────────────────────────────────────────────────
    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, Any, Any]:
        x = batch["image"]
        binary_masks = batch["binary_masks"]
        mask_labels = [item["masks"].to(x.device) for item in binary_masks]
        class_labels = [item["labels"].to(x.device) for item in binary_masks]
        outputs = self.model.model(
            pixel_values=x, mask_labels=mask_labels, class_labels=class_labels
        )
        return outputs.loss, outputs, mask_labels

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, outputs, _ = self._forward_loss(batch)
        height, width = batch["masks"].shape[1:3]
        results = panoptic_segmentation_decode(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            [{"orig_h": height, "orig_w": width}] * len(batch["image"]),
            num_labels=self.model_args.num_classes,
            num_thing_classes=self.model_args.num_thing_classes,
            threshold=self.model_args.threshold,
            mask_threshold=self.model_args.mask_threshold,
            mask_overlap_threshold=self.model_args.mask_overlap_threshold,
        )
        evaluator = self._evaluators.setdefault("val", self._make_evaluator())
        predictions = torch.stack([item["masks"] for item in results])
        evaluator.update(predictions, batch["masks"])
        return TaskStepResult(
            loss=loss,
            log_dict={"val_loss/eomt": float(loss.detach())},
        )

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator = self._evaluators.setdefault(split, self._make_evaluator())
        evaluator.merge_state(state)
