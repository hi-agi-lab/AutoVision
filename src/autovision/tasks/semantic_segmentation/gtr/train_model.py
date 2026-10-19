# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR semantic segmentation training wrapper (roadmap doc 31 §11 S1)."""

from __future__ import annotations

import torch

from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_weight_decay_parameters,
)
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.semantic_segmentation.gtr.config import GTRSemSegTrainArgs
from autovision.tasks.semantic_segmentation.gtr.task_model import (
    GTRSemanticSegmentation,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["GTRSemSegTrain"]


class GTRSemSegTrain(TrainModel):
    """Full fine-tune (backbone at ``lr * backbone_lr_factor``) with CE loss."""

    task = "semantic_segmentation"
    watch_metric = "val_metric/miou"

    def __init__(self, model_args: GTRSemSegTrainArgs) -> None:
        task_model = GTRSemanticSegmentation(
            num_classes=model_args.num_classes,
            variant=model_args.variant,
            image_size=model_args.image_size,
            dropout=model_args.dropout,
            checkpoint=model_args.checkpoint,
            load_checkpoint=model_args.load_checkpoint,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, SegmentationMetricCollector] = {}

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    # ── steps ───────────────────────────────────────────────────────────
    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"]
        logits = self.model.forward_backend(x)
        masks = batch["mask"].to(x.device).long()
        # sum/count reduction: an all-ignore batch yields 0/1 = 0 instead of
        # NaN, mirroring the official SemSegCriterion.
        loss_sum = torch.nn.functional.cross_entropy(
            logits, masks, ignore_index=self.model_args.ignore_index, reduction="sum"
        )
        valid = (masks != self.model_args.ignore_index).sum().clamp_(min=1)
        loss = loss_sum / valid
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
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        args = self.model_args
        args.resolve_auto_fields(total_steps=total_steps)
        param_groups: list[dict] = []
        for name, modules, lr in (
            ("backbone", [self.model.backbone], args.lr * args.backbone_lr_factor),
            (
                "detector",
                [self.model.encoder, self.model.head],
                args.lr,
            ),
        ):
            with_decay, no_decay = get_weight_decay_parameters(modules)
            if with_decay:
                param_groups.append(
                    {"name": name, "params": with_decay, "lr": lr}
                )
            if no_decay:
                param_groups.append(
                    {
                        "name": f"{name}_no_wd",
                        "params": no_decay,
                        "lr": lr,
                        "weight_decay": 0.0,
                    }
                )
        optimizer = build_adamw(
            param_groups,
            lr=args.lr,
            betas=args.optimizer_betas,
            weight_decay=args.weight_decay,
            fused=args.fused_optimizer,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=args.lr_warmup_steps, max_epochs=total_steps
        )
        return optimizer, scheduler

    # ── export ──────────────────────────────────────────────────────────
    def get_export_state_dict(self) -> dict:
        return self.model.state_dict()
