# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation training wrapper."""

from __future__ import annotations

from typing import Any

import torch

from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.semantic_segmentation.dinov2_eomt.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision.tasks.shared.eomt.labels import semantic_masks_to_instances
from autovision.tasks.shared.eomt.postprocess import semantic_segmentation_decode
from autovision.tasks.shared.eomt.scheduler import build_eomt_optimizer
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["Dinov2EomtSemanticSegmentationTrain"]


class Dinov2EomtSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    watch_metric = "val_metric/miou"
    TASK_MODEL_CLS = Dinov2EomtSemanticSegmentation

    def __init__(self, model_args: Dinov2EomtSemanticSegmentationTrainArgs) -> None:
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
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, SegmentationMetricCollector] = {}

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    # ── steps ───────────────────────────────────────────────────────────
    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, Any]:
        x = batch["image"]
        mask_labels, class_labels = semantic_masks_to_instances(
            batch["mask"].to(x.device).long(), self.model_args.ignore_index
        )
        outputs = self.model.model(
            pixel_values=x, mask_labels=mask_labels, class_labels=class_labels
        )
        return outputs.loss, outputs, mask_labels

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, _, _ = self._forward_loss(batch)
        return TaskStepResult(
            loss=loss,
            log_dict={"train_loss/eomt": float(loss.detach())},
        )

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, outputs, _ = self._forward_loss(batch)
        # Semantic masks are class-index maps at the canvas resolution.
        height, width = batch["mask"].shape[-2:]
        metadata = [{"orig_w": width, "orig_h": height}] * len(batch["image"])
        results = semantic_segmentation_decode(
            outputs.masks_queries_logits, outputs.class_queries_logits, metadata
        )
        evaluator = self._evaluators.setdefault(
            "val",
            SegmentationMetricCollector(
                num_classes=self.model_args.num_classes,
                ignore_index=self.model_args.ignore_index,
            ),
        )
        predictions = torch.stack([item["mask"] for item in results])
        evaluator.update(predictions, batch["mask"])
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
            SegmentationMetricCollector(
                num_classes=self.model_args.num_classes,
                ignore_index=self.model_args.ignore_index,
            ),
        )
        evaluator.merge_state(state)

    # ── optimization ────────────────────────────────────────────────────
    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> tuple[torch.optim.Optimizer, Any]:
        args = self.model_args
        args.resolve_auto_fields(total_steps=total_steps)
        optimizer = build_eomt_optimizer(
            self.model.model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            layer_decay=args.layer_decay,
            betas=args.optimizer_betas,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=args.lr_warmup_steps, max_epochs=total_steps
        )
        return optimizer, scheduler

    # ── export ──────────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()
