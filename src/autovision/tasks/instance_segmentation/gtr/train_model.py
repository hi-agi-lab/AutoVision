# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR instance-segmentation training wrapper (roadmap doc 31 §11 S2).

Reuses the DETR training machinery wholesale (optimizer/EMA/criterion —
the mask branch rides the criterion's ``mask_losses`` flag) and replaces
the validation instrument with COCO mask AP (pycocotools segm), which
drives best-checkpoint selection (``val_metric/mask_map``).
"""

from __future__ import annotations

import torch

from autovision.metrics.instance_segmentation import (
    COCOMaskAPEvaluator,
    resize_masks,
)
from autovision.tasks.detection.detr.train_model import (
    DETRObjectDetectionTrain,
)
from autovision.tasks.instance_segmentation.gtr.config import GTRSegTrainArgs
from autovision.tasks.instance_segmentation.gtr.task_model import (
    GTRSegObjectDetection,
)
from autovision.tasks.train_model import TaskStepResult

__all__ = ["GTRSegTrain"]


class GTRSegTrain(DETRObjectDetectionTrain):
    task = "instance_segmentation"
    watch_metric = "val_metric/mask_map"

    # ── task-specific construction hooks ────────────────────────────────
    def _build_task_model(self, model_args: GTRSegTrainArgs) -> GTRSegObjectDetection:
        return GTRSegObjectDetection(
            num_classes=model_args.num_classes,
            variant=model_args.variant,
            image_size=model_args.image_size,
            num_queries=model_args.num_queries,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            mask_downsample_ratio=model_args.mask_downsample_ratio,
            checkpoint=model_args.checkpoint,
            load_checkpoint=model_args.load_checkpoint,
        )

    def _build_targets(
        self, batch: dict, x: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        targets = super()._build_targets(batch, x)
        instance_masks = batch.get("masks")
        if instance_masks is not None:
            for target, masks in zip(targets, instance_masks):
                target["masks"] = masks.to(x.device).float()
        return targets

    # ── validation: COCO mask AP (official protocol) ────────────────────
    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        # Eval-mode forward: the decoder runs the final layer only and
        # emits the mask logits of that layer.
        out = model._forward_train(x)
        outputs = {
            "pred_logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
            "pred_masks": out.get("pred_masks"),
        }
        decoded = model.postprocess(
            (outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]),
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=0.0,
        )
        evaluator = self._evaluators.setdefault(
            "val", COCOMaskAPEvaluator(self.categories)
        )
        evaluator.update(decoded)
        evaluator.update_targets(
            [
                {
                    "masks": resize_masks(masks, height, width),
                    "labels": labels,
                    "size": (int(height), int(width)),
                }
                for masks, labels, (width, height) in zip(
                    batch["masks"], batch["classes"], batch["original_size"]
                )
            ]
        )
        losses = self.criterion(
            [outputs], self._build_targets(batch, x), world_size=1
        )
        log_dict = {key: float(value) for key, value in losses.items()}
        return TaskStepResult(loss=losses["loss"], log_dict=log_dict)

    # ── metric plumbing (mask evaluator instead of the box one) ─────────
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
            split, COCOMaskAPEvaluator(self.categories)
        )
        evaluator.merge_state(state)
