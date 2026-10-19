# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR oriented object detection training wrapper."""

from __future__ import annotations

import torch

from autovision.geometry.obb import (
    xyxyxyxy_to_xywhr,
)
from autovision.metrics.oriented_detection import (
    OrientedDetectionMetricCollector,
)
from autovision.tasks.detection.rtdetr.train_model import (
    RTDETRObjectDetectionTrain,
    load_pretrained_backbone,
)
from autovision.tasks.oriented_detection.rtdetr.config import (
    RTDETROrientedObjectDetectionTrainArgs,
)
from autovision.tasks.oriented_detection.rtdetr.task_model import (
    RTDETROrientedObjectDetection,
)
from autovision.tasks.oriented_detection.shared.obb_criterion import (
    OrientedHungarianMatcher,
    OrientedRTDETRCriterion,
)
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["RTDETROrientedObjectDetectionTrain"]


class RTDETROrientedObjectDetectionTrain(RTDETRObjectDetectionTrain):
    """Oriented (YOLO-OBB) RT-DETR training.

    Differences from the axis-aligned parent: OBB criterion/matcher, angle
    supervision (targets are normalized-corner batches decoded to xywhr),
    and a rotated-IoU mAP evaluator. Optimizer/schedule/EMA logic is
    inherited unchanged.
    """

    task = "oriented_object_detection"
    watch_metric = "val_metric/map"

    def __init__(self, model_args: RTDETROrientedObjectDetectionTrainArgs) -> None:
        task_model = RTDETROrientedObjectDetection(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            hidden_dim=model_args.hidden_dim,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers,
            num_heads=model_args.num_heads,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            nms_iou_threshold=model_args.nms_iou_threshold,
            load_weights=False,
        )
        # The parent __init__ builds axis-aligned criterion/matcher/EMA;
        # initialize the TrainModel base directly instead.
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        if model_args.backbone_pretrained:
            load_pretrained_backbone(task_model, model_args.backbone)
        self.matcher = OrientedHungarianMatcher()
        self.criterion = OrientedRTDETRCriterion(self.matcher)
        self.ema_model: ModelEMA | None = (
            ModelEMA(
                task_model,
                decay=model_args.ema_momentum,
                warmup_steps=model_args.ema_warmup_steps,
            )
            if model_args.use_ema_model
            else None
        )
        self.categories: dict[int, str] = {}
        self._evaluators: dict[str, OrientedDetectionMetricCollector] = {}

    @staticmethod
    def _targets_from_batch(batch: dict, device: torch.device) -> list[dict]:
        """Normalized corner batches -> xywhr targets (boxes + angles)."""
        targets = []
        for polys, labels in zip(batch["bboxes"], batch["classes"]):
            xywhr = xyxyxyxy_to_xywhr(polys.to(device))  # (n, 5)
            targets.append(
                {
                    "boxes": xywhr[:, :4],
                    "angles": xywhr[:, 4:5],
                    "labels": labels.to(device),
                }
            )
        return targets

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        targets = self._targets_from_batch(batch, x.device)
        out = self.model._forward_train(x, targets=targets)
        outputs_list = list(out.get("aux_outputs", [])) + [
            {
                "pred_logits": out["pred_logits"],
                "pred_boxes": out["pred_boxes"],
                "pred_angles": out["pred_angles"],
            }
        ]
        loss_dict = self.criterion(
            outputs_list, targets, world_size=1,
            dn_outputs=out.get("dn_aux_outputs", []),
            dn_meta=out.get("dn_meta"),
        )
        log_dict = {key: float(value.detach()) for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)

    @torch.no_grad()
    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        x = batch["image"]
        model = self.ema_model.model if self.ema_model is not None else self.model
        outputs = model(x)  # (logits, boxes, angles)
        decoded = model.postprocess(
            outputs,
            [{"orig_w": w, "orig_h": h} for w, h in batch["original_size"]],
            threshold=self.model_args.score_threshold,
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
        outputs_list = [
            {
                "pred_logits": outputs[0],
                "pred_boxes": outputs[1],
                "pred_angles": outputs[2],
            }
        ]
        losses = self.criterion(
            outputs_list,
            self._targets_from_batch(batch, x.device),
            world_size=1,
        )
        return TaskStepResult(
            loss=losses["loss"],
            log_dict={key: float(value) for key, value in losses.items()},
        )

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is None:
            return
        evaluator = self._evaluators.setdefault(
            split, OrientedDetectionMetricCollector(self.categories)
        )
        evaluator.merge_state(state)
