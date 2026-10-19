# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR metric-depth training wrapper (doc 31 §11 S4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autovision.metrics.depth_estimation import DepthMetricCollector
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_weight_decay_parameters,
)
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.depth_estimation.gtr.config import (
    GTRDepthEstimationTrainArgs,
)
from autovision.tasks.depth_estimation.gtr.task_model import GTRDepthEstimation
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["GTRDepthEstimationTrain"]


class GTRDepthEstimationTrain(TrainModel):
    task = "depth_estimation"
    watch_metric = "val_metric/delta1"

    def __init__(self, model_args: GTRDepthEstimationTrainArgs) -> None:
        task_model = GTRDepthEstimation(
            variant=model_args.variant,
            image_size=model_args.image_size,
            features=model_args.features,
            checkpoint=model_args.checkpoint,
            load_checkpoint=model_args.load_checkpoint,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, DepthMetricCollector] = {}

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    # ── steps ───────────────────────────────────────────────────────────
    def _silog(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Official GTRDepthCriterion (Depth-Anything-V2 SiLog)."""
        valid = (
            (gt >= self.model_args.min_depth)
            & (gt <= self.model_args.max_depth)
            & torch.isfinite(pred)
            & (pred > 0)
        )
        if not bool(valid.any()):
            return pred.sum() * 0.0
        diff = gt[valid].log() - pred[valid].clamp_min(1e-6).log()
        return torch.sqrt(
            diff.pow(2).mean() - self.model_args.silog_lambd * diff.mean().pow(2)
        )

    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"]
        depth = self.model.forward_backend(x)
        gt = batch["depth"].to(x.device).float()
        if tuple(depth.shape[-2:]) != tuple(gt.shape[-2:]):
            # The official criterion interpolates predictions to the
            # ground-truth resolution before the SiLog (align_corners=True).
            depth_for_loss = F.interpolate(
                depth.unsqueeze(1),
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        else:
            depth_for_loss = depth
        # Return the GT-aligned map: validation metrics compare against the
        # transformed GT (canvas resolution), so upsampling predictions back
        # to the original frame here would make the shapes disagree.
        return self._silog(depth_for_loss, gt), depth_for_loss

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, _ = self._forward_loss(batch)
        return TaskStepResult(
            loss=loss, log_dict={"train_loss/silog": float(loss.detach())}
        )

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        loss, depth = self._forward_loss(batch)
        evaluator = self._evaluators.setdefault(
            "val",
            DepthMetricCollector(
                min_depth=self.model_args.min_depth,
                max_depth=self.model_args.max_depth,
            ),
        )
        # Predictions and GT are both on the transformed canvas here.
        evaluator.update(depth, batch["depth"])
        return TaskStepResult(
            loss=loss, log_dict={"val_loss/silog": float(loss.detach())}
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
            DepthMetricCollector(
                min_depth=self.model_args.min_depth,
                max_depth=self.model_args.max_depth,
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
            ("detector", [self.model.encoder, self.model.decoder], args.lr),
        ):
            with_decay, no_decay = get_weight_decay_parameters(modules)
            if with_decay:
                param_groups.append({"name": name, "params": with_decay, "lr": lr})
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

    def get_export_state_dict(self) -> dict:
        return self.model.state_dict()
