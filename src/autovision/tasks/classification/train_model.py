# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification training wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from autovision.metrics.classification import ClassificationMetricCollector
from autovision.optim import get_weight_decay_parameters
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.classification.config import (
    ImageClassificationTrainArgs,
)
from autovision.tasks.classification.task_model import (
    ImageClassification,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["ImageClassificationTrain"]


class ImageClassificationTrain(TrainModel):
    task = "image_classification"
    watch_metric = "val_metric/accuracy_top1"

    def __init__(self, model_args: ImageClassificationTrainArgs) -> None:
        task_model = ImageClassification(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            load_weights=False,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.classes: dict[int, str] = {}
        self._evaluators: dict[str, ClassificationMetricCollector] = {}
        if model_args.freeze_backbone:
            for parameter in self.model.backbone.parameters():
                parameter.requires_grad_(False)

    def set_categories(self, categories: dict[int, str]) -> None:
        self.classes = dict(categories)

    # ── steps ───────────────────────────────────────────────────────────
    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"]
        logits = self.model.forward_backend(x)
        labels = batch["label"].to(x.device).long()
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, logits = self._forward_loss(batch)
        return TaskStepResult(
            loss=loss,
            log_dict={"train_loss/ce": float(loss.detach())},
        )

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, logits = self._forward_loss(batch)
        evaluator = self._evaluators.setdefault(
            "val", ClassificationMetricCollector()
        )
        evaluator.update(logits, batch["label"])
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
            split, ClassificationMetricCollector()
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
        lr = args.lr
        param_groups: list[dict[str, Any]] = [
            {"name": "head", "params": list(self.model.head.parameters()), "lr": lr},
        ]
        if not args.freeze_backbone:
            backbone_wd, backbone_no_wd = get_weight_decay_parameters(
                [self.model.backbone]
            )
            no_wd_ids = {id(p) for p in backbone_no_wd}
            param_groups.append(
                {
                    "name": "backbone",
                    "params": [p for p in backbone_wd if id(p) not in no_wd_ids],
                    "lr": lr * args.backbone_lr_factor,
                }
            )
            if backbone_no_wd:
                param_groups.append(
                    {
                        "name": "backbone_no_wd",
                        "params": backbone_no_wd,
                        "lr": lr * args.backbone_lr_factor,
                        "weight_decay": 0.0,
                    }
                )
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
        )
        if args.scheduler == "cosine":
            scheduler = CosineWarmupScheduler(
                optimizer, warmup_epochs=args.lr_warmup_steps,
                max_epochs=total_steps,
            )
        else:  # pragma: no cover - flat-cosine path exercised via detection
            from autovision.tasks.shared.flat_cosine import (
                FlatCosineLRScheduler,
            )

            scheduler = FlatCosineLRScheduler(
                optimizer, total_steps=total_steps,
                warmup_steps=args.lr_warmup_steps,
                flat_steps=0, no_aug_steps=0,
            )
        return optimizer, scheduler

    # ── export ──────────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()
