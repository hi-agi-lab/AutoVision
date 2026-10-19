# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR oriented-detection training wrapper (roadmap doc 31 §11 S3b).

Criterion/matcher/CDN are the faithfully re-ported official OBB pieces
(``shared/gtr/obb_{criterion,denoising}.py``); the decoder runs the
oriented CDN internally in training mode and emits the upstream
criterion-ready dict. Validation uses the framework's oriented mAP
collector (the criterion's eval path expects the training-only aux
branch, so the val loss is not computed — the metric drives selection,
``watch_metric = val_metric/map``).
"""

from __future__ import annotations

import torch

from autovision.metrics.oriented_detection import OrientedDetectionMetricCollector
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_weight_decay_parameters,
)
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.tasks.detection.shared.gtr.obb_criterion import (
    OBBGTRCriterion,
    OBBHungarianMatcher,
)
from autovision.tasks.detection.shared.gtr.rbox_ops import rbox_rad_to_longedge
from autovision.tasks.oriented_detection.gtr.config import GTRROBBTrainArgs
from autovision.tasks.oriented_detection.gtr.task_model import (
    GTRROBBOrientedDetection,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["GTRROBBTrain"]


class GTRROBBTrain(TrainModel):
    task = "oriented_object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: GTRROBBTrainArgs) -> None:
        model_args.resolve_auto_fields()
        task_model = GTRROBBOrientedDetection(
            num_classes=model_args.num_classes,
            variant=model_args.variant,
            image_size=model_args.image_size,
            num_queries=model_args.num_queries,
            num_top_queries=model_args.num_queries,
            num_denoising=model_args.num_denoising,
            load_checkpoint=model_args.load_checkpoint,
            checkpoint=model_args.checkpoint,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        self.matcher = OBBHungarianMatcher(
            cost_class=model_args.cost_class,
            cost_chamfer=model_args.cost_chamfer,
            cost_kld=model_args.cost_kld,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = OBBGTRCriterion(
            self.matcher,
            weight_dict=model_args.loss_weight_dict,
            losses=model_args.losses,
            num_classes=model_args.num_classes,
            reg_max=32,
            group_detr=3,
        )
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, OrientedDetectionMetricCollector] = {}

    def set_categories(self, categories: dict[int, str]) -> None:
        self.categories = dict(categories)

    # ── targets ─────────────────────────────────────────────────────────
    def _build_targets(
        self, batch: dict, x: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Normalized corner batches -> GTR 5-dim long-edge rbox targets.

        The oriented transform keeps corners in the (square) model input
        frame, normalized per axis; GTR's box parameterization is
        (cx, cy, w, h, a) with w the LONGER side and ``a = theta / pi`` in
        [0, 1) — the same convention the official DOTA finetune trains on.
        """
        targets = []
        for polys, labels in zip(batch["bboxes"], batch["classes"]):
            xywhr = xyxyxyxy_to_whr_longedge(polys.to(x.device))
            targets.append(
                {"boxes": xywhr, "labels": labels.to(x.device).long()}
            )
        return targets

    # ── steps ───────────────────────────────────────────────────────────
    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = self._build_targets(batch, x)
        out = self.model._forward_train(x, targets=targets)
        loss_dict = self.criterion(out, targets)
        loss = sum(loss_dict.values())
        log_dict = {key: float(value.detach()) for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss, log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.model
        out = model._forward_train(x)
        decoded = model.postprocess(
            (out["pred_logits"], out["pred_boxes"]),
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=0.0,
        )
        evaluator = self._evaluators.setdefault(
            "val", OrientedDetectionMetricCollector(self.categories)
        )
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {"corners": corners, "labels": labels}
                for corners, labels in zip(batch["bboxes"], batch["classes"])
            ],
            original_sizes=batch["original_size"],
        )
        # Metrics-only validation (see module docstring): the training
        # criterion's aux branch does not exist at eval; the zero loss keeps
        # the engine's logging contract without inventing a number.
        zero = out["pred_logits"].sum() * 0.0
        return TaskStepResult(loss=zero, log_dict={"val_loss": 0.0})

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
            split, OrientedDetectionMetricCollector(self.categories)
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


def xyxyxyxy_to_whr_longedge(polys: torch.Tensor) -> torch.Tensor:
    """(N, 4, 2)/(N, 8) normalized corners -> (N, 5) normalized long-edge rbox.

    ``xyxyxyxy_to_xywhr`` canonicalizes into [-pi/4, pi/4] with w/h possibly
    swapped; the GTR convention needs w >= h and theta in [0, pi) —
    ``rbox_rad_to_longedge`` performs exactly that transform.
    """
    from autovision.geometry.obb import xyxyxyxy_to_xywhr

    xywhr = xyxyxyxy_to_xywhr(polys.to(torch.float32))
    long_edge = rbox_rad_to_longedge(xywhr)
    return torch.cat([long_edge[..., :4], (long_edge[..., 4:5] / 3.141592653589793)], dim=-1)
