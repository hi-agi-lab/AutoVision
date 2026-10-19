# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PicoDet object detection training wrapper."""

from __future__ import annotations

import logging
from typing import Any

import torch

from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.optim import get_weight_decay_parameters
from autovision.tasks.detection.picodet.config import (
    PicoDetObjectDetectionTrainArgs,
)
from autovision.tasks.detection.picodet.losses import PicoCriterion
from autovision.tasks.detection.picodet.task_model import (
    PicoDetObjectDetection,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.shared.flat_cosine import (
    FlatCosineLRScheduler,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")

__all__ = ["PicoDetObjectDetectionTrain"]


class PicoDetObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: PicoDetObjectDetectionTrainArgs) -> None:
        task_model = PicoDetObjectDetection(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            neck_channels=model_args.neck_channels,
            reg_max=model_args.reg_max,
            num_top_queries=model_args.num_top_queries,
            nms_iou_threshold=model_args.nms_iou_threshold,
            load_weights=False,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.criterion = PicoCriterion(
            num_classes=model_args.num_classes,
            assigner=model_args.assigner,
            assigner_topk=model_args.assigner_topk,
            loss_cls_weight=model_args.loss_cls_weight,
            loss_giou_weight=model_args.loss_giou_weight,
            loss_dfl_weight=model_args.loss_dfl_weight,
        )
        self.ema_model: ModelEMA | None = (
            ModelEMA(task_model, decay=model_args.ema_momentum, warmup_steps=model_args.ema_warmup_steps)
            if model_args.use_ema_model
            else None
        )
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, COCOMAPEvaluator] = {}

    # ── steps ───────────────────────────────────────────────────────────
    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        preds = self.model._forward_train(x)
        loss_dict = self.criterion(preds, targets, image_size=self.model.image_size)
        log_dict = {key: float(value.detach()) for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        targets = [
            {"boxes": boxes.to(x.device), "labels": labels.to(x.device)}
            for boxes, labels in zip(batch["bboxes"], batch["classes"])
        ]
        preds = model._forward_train(x)
        loss_dict = self.criterion(preds, targets, image_size=model.image_size)
        decoded = model.postprocess(
            (preds["pred_logits"], preds["pred_boxes"]),
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=0.05,
        )
        evaluator = self._evaluators.setdefault(
            "val", COCOMAPEvaluator(self.categories)
        )
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {"boxes": boxes, "labels": labels}
                for boxes, labels in zip(batch["bboxes"], batch["classes"])
            ],
            original_sizes=batch["original_size"],
        )
        return TaskStepResult(loss=loss_dict["loss"], log_dict={
            key: float(value) for key, value in loss_dict.items()
        })

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

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
            split, COCOMAPEvaluator(self.categories)
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
        if (
            args.lr_warmup_steps == "auto"
            or args.scheduler_flat_steps == "auto"
            or args.scheduler_no_aug_steps == "auto"
        ):
            args.resolve_auto_fields(
                total_steps=total_steps,
                train_num_batches=train_num_batches or total_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        # SGD convention: linear lr scaling with the global batch size.
        lr = args.lr * (global_batch_size / args.default_batch_size)
        params_wd, params_no_wd = get_weight_decay_parameters([self.model])
        param_groups = [{"name": "picodet", "params": params_wd, "lr": lr}]
        if params_no_wd:
            param_groups.append({
                "name": "picodet_no_wd", "params": params_no_wd,
                "lr": lr, "weight_decay": 0.0,
            })
        optimizer = torch.optim.SGD(
            param_groups, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        scheduler = FlatCosineLRScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(args.lr_warmup_steps),
            flat_steps=int(args.scheduler_flat_steps),
            no_aug_steps=int(args.scheduler_no_aug_steps),
        )
        return optimizer, scheduler

    def clip_gradients(self, optimizer) -> float | None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return float(
            torch.nn.utils.clip_grad_norm_(params, self.model_args.gradient_clip_val)
        )

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    # ── export state ────────────────────────────────────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_model is not None:
            checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state)

    def get_export_state_dict(self) -> dict[str, Any]:
        if self.ema_model is not None:
            return {
                f"ema_model.{key}": value
                for key, value in self.ema_model.model.state_dict().items()
            }
        return self.state_dict()
