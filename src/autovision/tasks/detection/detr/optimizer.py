# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR optimizer construction: param groups + warmup scheduler.

Extracted verbatim from ``TrainModel.get_optimizer`` (design doc 30
stage 4) so the grouping policy is testable without instantiating the
train wrapper.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from autovision.optim import build_adamw, get_weight_decay_parameters
from autovision.tasks.detection.detr.backbone_wrappers import (
    ViTDetectionBackbone,
)
from autovision.tasks.detection.shared.multistep_schedule import (
    MultiStepWarmupScheduler,
)
from autovision.tasks.detection.shared.open_vocab import ContrastiveClassHead
from autovision.tasks.shared.flat_cosine import FlatCosineLRScheduler

__all__ = ["build_detr_optimizer"]


def build_detr_optimizer(
    model: torch.nn.Module,
    args: Any,
    total_steps: int,
    global_batch_size: int,
    train_num_batches: int = 0,
    gradient_accumulation_steps: int = 1,
) -> tuple[torch.optim.Optimizer, Any]:
    args = args
    if (
        args.lr_warmup_steps == "auto"
        or args.scheduler_flat_steps == "auto"
        or args.scheduler_no_aug_steps == "auto"
        or args.lr_milestones == "auto"
        or args.backbone_lr_factor == "auto"
        or args.lr_reference_batch == "auto"
        or args.hidden_dim == "auto"
        or args.dec_layers == "auto"
        or args.num_heads == "auto"
    ):
        args.resolve_auto_fields(
            total_steps=total_steps,
            train_num_batches=train_num_batches or total_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    if args.token_drop is not None and args.token_drop.step_stop == "auto":
        # Same shared step schedule the transform-side augmentations use
        # (random_erase_step_stop precedent, doc 16 P1.5).
        from autovision.augment.detection.schedule import (
            resolve_detection_step_schedule,
        )

        steps_per_epoch = max(
            1, train_num_batches // max(1, gradient_accumulation_steps)
        )
        args.token_drop.step_stop = resolve_detection_step_schedule(
            total_steps, steps_per_epoch
        ).stop
    lr = args.lr * math.sqrt(global_batch_size / args.lr_reference_batch)
    backbone_lr = lr * args.backbone_lr_factor

    # Baseline LR grouping: no-weight-decay params across the whole
    # training model, then backbone/detector partition by membership.
    _, params_no_wd_list = get_weight_decay_parameters([model])
    params_no_wd = {id(p) for p in params_no_wd_list}

    backbone = model.backbone
    if isinstance(backbone, ViTDetectionBackbone):
        backbone_params = list(backbone.backbone_model.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}
        # Randomly initialized connector modules (STA prior +
        # projection convs/norms) train at detector LR.
        connector_params = [
            p for p in backbone.parameters() if id(p) not in backbone_param_ids
        ]
    else:
        backbone_params = list(backbone.parameters())
        connector_params = []

    param_groups: list[dict[str, Any]] = []
    backbone_wd = [p for p in backbone_params if id(p) not in params_no_wd]
    backbone_no_wd = [p for p in backbone_params if id(p) in params_no_wd]
    if backbone_wd:
        param_groups.append(
            {"name": "backbone", "params": backbone_wd, "lr": backbone_lr}
        )
    if backbone_no_wd:
        param_groups.append(
            {
                "name": "backbone_no_wd",
                "params": backbone_no_wd,
                "lr": backbone_lr,
                "weight_decay": 0.0,
            }
        )

    detector_params = (
        connector_params
        + list(model.encoder.parameters())
        + list(model.decoder.parameters())
    )
    # Open-vocab calibration scalars get their own explicitly named
    # no-weight-decay group at the base LR (design docs 26 W1 / 27
    # §3.9): the generic ndim<=1 rule would also catch them, but the
    # named group keeps the LR attribution legible in the optimizer
    # summary.
    ov_calibration_params: list[torch.nn.Parameter] = []
    if getattr(model.decoder, "_ov_text_embed_dim", None) is not None:
        for head in model.decoder.modules():
            if isinstance(head, ContrastiveClassHead):
                if head.class_bias is not None:
                    ov_calibration_params.append(head.class_bias)
                ov_calibration_params.append(head.logit_scale)
        if ov_calibration_params:
            ov_ids = {id(p) for p in ov_calibration_params}
            detector_params = [
                p for p in detector_params if id(p) not in ov_ids
            ]
    detector_wd = [p for p in detector_params if id(p) not in params_no_wd]
    detector_no_wd = [p for p in detector_params if id(p) in params_no_wd]
    if detector_wd:
        param_groups.append({"name": "detector", "params": detector_wd, "lr": lr})
    if detector_no_wd:
        param_groups.append(
            {
                "name": "detector_no_wd",
                "params": detector_no_wd,
                "lr": lr,
                "weight_decay": 0.0,
            }
        )
    if ov_calibration_params:
        param_groups.append(
            {
                "name": "ov_no_wd",
                "params": ov_calibration_params,
                "lr": lr,
                "weight_decay": 0.0,
            }
        )

    # The SSL projection head lives on the task model root (outside
    # encoder/decoder), so it needs its own explicit group at the base
    # LR — mirroring RF-DETR-SSL, which assigns the head the base LR.
    if getattr(model, "ssl_head", None) is not None:
        ssl_all = [p for p in model.ssl_head.parameters() if p.requires_grad]
        ssl_wd = [p for p in ssl_all if id(p) not in params_no_wd]
        ssl_no_wd = [p for p in ssl_all if id(p) in params_no_wd]
        if ssl_wd:
            param_groups.append({"name": "ssl", "params": ssl_wd, "lr": lr})
        if ssl_no_wd:
            param_groups.append(
                {
                    "name": "ssl_no_wd",
                    "params": ssl_no_wd,
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )

    optimizer = build_adamw(
        param_groups,
        lr=lr,
        betas=args.optimizer_betas,
        weight_decay=args.weight_decay,
        fused=args.fused_optimizer,
    )
    if args.lr_scheduler == "linear":
        # Baseline DINOv2 recipe: warmup then constant.
        scheduler = MultiStepWarmupScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(args.lr_warmup_steps),
            milestones=(),
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "multistep":
        scheduler = MultiStepWarmupScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(args.lr_warmup_steps),
            milestones=tuple(int(f * total_steps) for f in args.lr_milestones),
            gamma=args.lr_gamma,
        )
    else:
        scheduler = FlatCosineLRScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(args.lr_warmup_steps),
            flat_steps=int(args.scheduler_flat_steps),
            no_aug_steps=int(args.scheduler_no_aug_steps),
        )
    return optimizer, scheduler
