# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D-FINE object detection training wrapper."""

from __future__ import annotations

from autovision.tasks.detection.dfine.config import (
    DFINEObjectDetectionTrainArgs,
)
from autovision.tasks.detection.dfine.task_model import (
    DFINEObjectDetection,
)
from autovision.tasks.detection.rtdetr.train_model import (
    RTDETRObjectDetectionTrain,
    load_pretrained_backbone,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)
from autovision.tasks.train_model import TaskStepResult, TrainModel

__all__ = ["DFINEObjectDetectionTrain"]


class DFINEObjectDetectionTrain(RTDETRObjectDetectionTrain):
    """RT-DETR training loop with the D-FINE criterion (FGL + DDF).

    Optimizer groups, LR schedule, EMA, gradient clipping, validation and
    metric plumbing are inherited from the RT-DETR wrapper; ``__init__``
    swaps the task model and criterion, and ``training_step`` forwards the
    distributional head outputs (corners + per-layer references) to the
    criterion. Validation uses the plain RT-DETR path (the exported decode
    contract is unchanged).
    """

    def __init__(self, model_args: DFINEObjectDetectionTrainArgs) -> None:
        task_model = DFINEObjectDetection(
            num_classes=model_args.num_classes,
            backbone=model_args.backbone,
            image_size=model_args.image_size,
            hidden_dim=model_args.hidden_dim,
            num_queries=model_args.num_queries,
            dec_layers=model_args.dec_layers,
            num_heads=model_args.num_heads,
            num_top_queries=model_args.num_top_queries,
            num_denoising=model_args.num_denoising,
            reg_max=model_args.reg_max,
            load_weights=False,
        )
        TrainModel.__init__(self, task_model=task_model)
        self.model_args = model_args
        if model_args.backbone_pretrained:
            load_pretrained_backbone(task_model, model_args.backbone)
        self.matcher = HungarianMatcher()
        self.criterion = DFINECriterionV2(
            self.matcher,
            reg_max=model_args.reg_max,
            fgl_loss_weight=model_args.fgl_loss_weight,
            ddf_loss_weight=model_args.ddf_loss_weight,
        )
        from autovision.tasks.shared.ema import ModelEMA

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
        self._evaluators: dict = {}

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
            outputs_list,
            targets,
            world_size=1,
            dn_outputs=out.get("dn_aux_outputs", []),
            dn_meta=out.get("dn_meta"),
            dn_pre_outputs=out.get("dn_pre_outputs"),
            corners_list=out.get("pred_corners_list"),
            ref_boxes_list=out.get("ref_boxes_list"),
            dn_corners_list=out.get("dn_pred_corners_list"),
            dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
            pre_outputs=out.get("pre_outputs"),
            enc_aux_outputs=out.get("enc_aux_outputs"),
            enc_meta=out.get("enc_meta"),
        )
        log_dict = {key: value.detach() for key, value in loss_dict.items()}
        return TaskStepResult(loss=loss_dict["loss"], log_dict=log_dict)
