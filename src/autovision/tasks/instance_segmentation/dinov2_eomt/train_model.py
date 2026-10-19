# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT instance segmentation training wrapper."""

from __future__ import annotations

from typing import Any

import torch

from autovision.metrics.instance_segmentation import (
    COCOMaskAPEvaluator,
    resize_masks,
)
from autovision.tasks.instance_segmentation.dinov2_eomt.config import (
    Dinov2EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.instance_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtInstanceSegmentation,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt.train_model import (
    Dinov2EomtSemanticSegmentationTrain,
)
from autovision.tasks.shared.eomt.postprocess import instance_segmentation_decode
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["Dinov2EomtInstanceSegmentationTrain"]


class Dinov2EomtInstanceSegmentationTrain(Dinov2EomtSemanticSegmentationTrain):
    """Instance targets are already per-instance masks, so the HF criterion
    consumes them directly (Hungarian matching + focal/dice/CE inside)."""

    task = "instance_segmentation"
    watch_metric = "val_metric/mask_map"
    TASK_MODEL_CLS = Dinov2EomtInstanceSegmentation

    def __init__(self, model_args: Dinov2EomtInstanceSegmentationTrainArgs) -> None:
        # Decode knobs are forwarded so val decoding (model_args) and
        # predict/postprocess (task model) share a single source of truth.
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
            score_threshold=model_args.score_threshold,
            mask_threshold=model_args.mask_threshold,
            num_top_queries=model_args.num_top_queries,
        )
        # The semantic wrapper's __init__ builds the task model from its own
        # fixed kwarg list; instance passes extra decode args, so initialize
        # the TrainModel base directly.
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, COCOMaskAPEvaluator] = {}

    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, Any, Any]:
        x = batch["image"]
        mask_labels = [mask.to(x.device) for mask in batch["masks"]]
        class_labels = [classes.to(x.device) for classes in batch["classes"]]
        outputs = self.model.model(
            pixel_values=x, mask_labels=mask_labels, class_labels=class_labels
        )
        return outputs.loss, outputs, mask_labels

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        loss, outputs, _ = self._forward_loss(batch)
        evaluator = self._evaluators.setdefault(
            "val",
            COCOMaskAPEvaluator(self.categories or {
                index: str(index) for index in range(self.model_args.num_classes)
            }),
        )
        heights = [int(size[0]) for size in batch.get("original_size", [])]
        widths = [int(size[1]) for size in batch.get("original_size", [])]
        if not heights:  # canvas-size fallback (transformed tensors)
            heights = [int(t.shape[-2]) for t in batch["masks"]]
            widths = [int(t.shape[-1]) for t in batch["masks"]]
        results = instance_segmentation_decode(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            [
                {"orig_h": h, "orig_w": w}
                for h, w in zip(heights, widths)
            ],
            score_threshold=self.model_args.score_threshold,
            mask_threshold=self.model_args.mask_threshold,
            num_top_queries=self.model_args.num_top_queries,
        )
        evaluator.update(results)
        evaluator.update_targets(
            [
                {
                    "masks": resize_masks(mask.to(x.device), h, w),
                    "labels": classes.to(x.device),
                    "size": (h, w),
                }
                for mask, classes, h, w in zip(
                    batch["masks"], batch["classes"], heights, widths
                )
            ]
        )
        return TaskStepResult(
            loss=loss,
            log_dict={"val_loss/eomt": float(loss.detach())},
        )

    def compute_metrics(self, split: str) -> dict[str, float]:
        evaluator = self._evaluators.get(split)
        if evaluator is None:
            return {}
        metrics = evaluator.compute()
        evaluator.reset()
        return {f"{split}_metric/{name}": value for name, value in metrics.items()}

    def reset_metrics(self, split: str) -> None:
        evaluator = self._evaluators.get(split)
        if evaluator is not None:
            evaluator.reset()

    def export_metric_state(self, split: str) -> dict | None:
        evaluator = self._evaluators.get(split)
        return evaluator.export_state() if evaluator is not None else None

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator = self._evaluators.setdefault(
            split,
            COCOMaskAPEvaluator(self.categories or {
                index: str(index) for index in range(self.model_args.num_classes)
            }),
        )
        evaluator.merge_state(state)
