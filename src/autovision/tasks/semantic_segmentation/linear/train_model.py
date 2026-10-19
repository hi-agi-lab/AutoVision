# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Linear semantic segmentation training wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.semantic_segmentation.linear.config import (
    LinearSemanticSegmentationTrainArgs,
)
from autovision.tasks.semantic_segmentation.linear.task_model import (
    LinearSemanticSegmentation,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["LinearSemanticSegmentationTrain"]


class LinearSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    watch_metric = "val_metric/miou"

    def __init__(self, model_args: LinearSemanticSegmentationTrainArgs) -> None:
        task_model = LinearSemanticSegmentation(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            load_weights=False,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        self._evaluators: dict[str, SegmentationMetricCollector] = {}

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    # ── steps ───────────────────────────────────────────────────────────
    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"]
        logits = self.model.forward_backend(x)
        masks = batch["mask"].to(x.device).long()
        loss = nn.functional.cross_entropy(
            logits, masks, ignore_index=self.model_args.ignore_index
        )
        return loss, logits.argmax(dim=1)

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, _ = self._forward_loss(batch)
        return TaskStepResult(
            loss=loss,
            log_dict={"train_loss/ce": float(loss.detach())},
        )

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, predictions = self._forward_loss(batch)
        evaluator = self._evaluators.setdefault(
            "val",
            SegmentationMetricCollector(
                num_classes=self.model_args.num_classes,
                ignore_index=self.model_args.ignore_index,
            ),
        )
        evaluator.update(predictions, batch["mask"])
        return TaskStepResult(
            loss=loss,
            log_dict={"val_loss/ce": float(loss.detach())},
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
        optimizer = torch.optim.AdamW(
            [{"name": "head", "params": list(self.model.head.parameters())}],
            lr=args.lr, betas=args.optimizer_betas, weight_decay=args.weight_decay,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=args.lr_warmup_steps, max_epochs=total_steps
        )
        return optimizer, scheduler

    # ── export ──────────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()
