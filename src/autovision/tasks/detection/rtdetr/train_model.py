# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR object detection training wrapper."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.optim import get_weight_decay_parameters
from autovision.tasks.detection.rtdetr.config import (
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.rtdetr.task_model import (
    RTDETRObjectDetection,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.shared.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.shared.flat_cosine import (
    FlatCosineLRScheduler,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

logger = logging.getLogger("autovision")

__all__ = ["RTDETRObjectDetectionTrain", "load_pretrained_backbone"]

_LOSS_WEIGHT_DICT = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def load_pretrained_backbone(task_model, backbone: str) -> int:
    """Warm-start the backbone from its package's pretrained weights.

    Implements the baseline §8.4 semantics for the common case: pretrained
    backbone weights are loaded strict-free (the detection head keeps its
    fresh initialization — class sets differ anyway). Currently supports the
    torchvision family; returns the number of tensors copied.
    """
    package, _, name = backbone.partition("/")
    if package != "torchvision":
        raise ValueError(
            f"backbone_pretrained=True does not support '{backbone}' yet; "
            f"supported: torchvision/<name>."
        )
    import torchvision.models as torchvision_models

    weights = torchvision_models.get_model_weights(name).DEFAULT
    reference = torchvision_models.get_model(name, weights=weights)
    state = {
        key: value
        for key, value in reference.state_dict().items()
        if not key.startswith("fc.")
    }
    incompatible = task_model.feature_extractor.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Backbone weights contain keys unknown to the feature extractor: "
            f"{incompatible.unexpected_keys[:5]}..."
        )
    copied = len(state) - len(incompatible.missing_keys)
    logger.info(
        "Loaded %d pretrained tensors into the '%s' backbone (%d skipped).",
        copied, backbone, len(incompatible.missing_keys),
    )
    return copied


class RTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: RTDETRObjectDetectionTrainArgs) -> None:
        task_model = RTDETRObjectDetection(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            hidden_dim=model_args.hidden_dim,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers,
            num_heads=model_args.num_heads,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            load_weights=False,
        )
        super().__init__(task_model=task_model)
        self.model_args = model_args
        if model_args.backbone_pretrained:
            load_pretrained_backbone(task_model, model_args.backbone)
        self.matcher = HungarianMatcher()
        self.criterion = RTDETRCriterionV2(self.matcher)
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
        out = self.model._forward_train(x, targets=targets)
        outputs_list = list(out.get("aux_outputs", [])) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        loss_dict = self.criterion(
            outputs_list, targets, world_size=1,
            dn_outputs=out.get("dn_aux_outputs", []),
            dn_meta=out.get("dn_meta"),
            enc_aux_outputs=out.get("enc_aux_outputs"),
            enc_meta=out.get("enc_meta"),
        )
        # Keep detached tensors: the engine's log window sums them on-device
        # and only syncs at log points (floats would be skipped entirely).
        log_dict = {key: value.detach() for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        outputs = model(x)
        decoded = model.postprocess(outputs, [
            {"orig_w": w, "orig_h": h} for w, h in batch["original_size"]
        ], threshold=0.05)
        evaluator = self._evaluators.setdefault(
            "val", COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets)
        )
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {"boxes": boxes, "labels": labels}
                for boxes, labels in zip(batch["bboxes"], batch["classes"])
            ],
            original_sizes=batch["original_size"],
        )
        if isinstance(outputs, dict):
            outputs_list = [outputs]
        elif isinstance(outputs, tuple):
            outputs_list = [{"pred_logits": outputs[0], "pred_boxes": outputs[1]}]
        else:  # ObjectDetectionOutput dataclass
            outputs_list = [
                {"pred_logits": outputs.logits, "pred_boxes": outputs.boxes}
            ]
        losses = self.criterion(
            outputs_list,
            [
                {"boxes": b.to(x.device), "labels": c.to(x.device)}
                for b, c in zip(batch["bboxes"], batch["classes"])
            ],
            world_size=1,
        )
        return TaskStepResult(loss=losses["loss"], log_dict={
            key: float(value) for key, value in losses.items()
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
            split,
            COCOMAPEvaluator(self.categories, max_dets=self.model_args.eval_max_dets),
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
        lr = args.lr * math.sqrt(global_batch_size / args.default_batch_size)
        backbone_lr = lr * args.backbone_lr_factor

        backbone_params_wd, backbone_params_no_wd = get_weight_decay_parameters(
            [self.model.feature_extractor]
        )
        detector_modules = [self.model.encoder, self.model.decoder]
        detector_params_wd, detector_params_no_wd = get_weight_decay_parameters(
            detector_modules
        )
        seen = {id(p) for p in backbone_params_no_wd} | {id(p) for p in detector_params_no_wd}
        detector_params_wd = [
            p for p in detector_params_wd if id(p) not in seen
        ]

        param_groups = [
            {"name": "backbone", "params": backbone_params_wd, "lr": backbone_lr},
            {"name": "detector", "params": detector_params_wd, "lr": lr},
        ]
        if backbone_params_no_wd:
            param_groups.append({
                "name": "backbone_no_wd", "params": backbone_params_no_wd,
                "lr": backbone_lr, "weight_decay": 0.0,
            })
        if detector_params_no_wd:
            param_groups.append({
                "name": "detector_no_wd", "params": detector_params_no_wd,
                "lr": lr, "weight_decay": 0.0,
            })
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=args.optimizer_betas, weight_decay=args.weight_decay
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
        max_norm = self.model_args.gradient_clip_val
        if max_norm <= 0:
            # Clipping disabled: report the total norm without mutating
            # grads (baseline contract — clip_grad_norm_ with a
            # nonpositive max_norm would ZERO every gradient).
            grads = [
                p.grad
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if not grads:
                return 0.0
            # One grouped kernel instead of one per tensor (hundreds on a
            # detector): _foreach_norm + the outer norm produce the same
            # total the per-tensor loop did.
            return float(
                torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            )
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

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
