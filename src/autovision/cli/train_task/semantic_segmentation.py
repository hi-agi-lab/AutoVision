# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation finetuning (EOMT / linear head)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from autovision.cli import common_helpers
from autovision.cli.train import _is_rank_zero
from autovision.cli.train_task.task_helpers import (
    TaskTrainResult,
    _SaveCheckpointArgs,
)
from autovision.config.validate import pydantic_model_validate
from autovision.engine import TaskEngine, seed_everything
from autovision.runtime.accelerator import accelerator_supports_bf16
from autovision.runtime.logging import set_up_file_logging
from autovision.types import PathLike

logger = logging.getLogger("autovision")

def train_semantic_segmentation(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "torchvision/resnet18",
    *,
    num_classes: int,
    steps: int | str = "auto",
    batch_size: int | str = "auto",
    gradient_accumulation_steps: int = 1,
    num_workers: int | str = "auto",
    devices: int | str = "auto",
    precision: str = "auto",
    seed: int = 0,
    checkpoint: PathLike | None = None,
    resume_interrupted: bool = False,
    overwrite: bool = False,
    model_args: dict[str, Any] | None = None,
    save_checkpoint_args: dict[str, Any] | None = None,
    trainer_args: dict[str, Any] | None = None,
    debug_args: dict[str, Any] | None = None,
    loggers: dict[str, dict[str, Any] | None] | None = None,
) -> TaskTrainResult:
    """Semantic segmentation finetuning (linear head on frozen backbone).

    ``model`` accepts an EOMT model string (``"dinov2/eomt"`` or
    ``"dinov3/eomt"``) to train the encoder-only mask transformer end to
    end; any other string selects the linear head on that backbone.
    ``data`` accepts a dict (or YAML) ``{"train": {"images": path, "masks":
    path}, "val": {...}}``. Masks are PNG class-index images; label 255 is
    ignored by the loss and the mIoU.
    """
    from autovision.data.semantic_segmentation_dataset import (
        SemanticSegmentationDataset,
        list_segmentation_pairs,
    )
    from autovision.tasks.semantic_segmentation.linear import (
        LinearSemanticSegmentationTrain,
        LinearSemanticSegmentationTrainArgs,
    )
    from autovision.tasks.semantic_segmentation.linear.transforms import (
        build_segmentation_transform,
    )

    # The seed must be applied before the assembly builds random
    # modules (model init, transform parameters): the engine re-seeds
    # inside fit(), which is already too late for construction.
    seed_everything(seed)
    out_dir = common_helpers.get_out_dir(out, resume_interrupted, overwrite)
    set_up_file_logging(out_dir / "train.log")

    if isinstance(data, dict):
        payload = dict(data)
    elif Path(data).is_file():
        with open(data) as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        payload = {}
    if not isinstance(payload, dict) or "train" not in payload:
        raise ValueError(
            "The segmentation data spec must provide a 'train' entry with "
            "'images' and 'masks' directories."
        )

    def _pairs(split: str):
        images = Path(payload[split]["images"])
        masks = Path(payload[split]["masks"])
        return list_segmentation_pairs(images, masks)

    train_pairs, num_classes_detected = _pairs("train")
    if num_classes_detected > num_classes:
        raise ValueError(
            f"Masks contain label {num_classes_detected - 1} but num_classes "
            f"is {num_classes}."
        )
    val_pairs = _pairs("val")[0] if "val" in payload else []

    is_eomt = isinstance(model, str) and model.startswith(
        ("dinov2/eomt", "dinov3/eomt")
    )
    is_gtr = isinstance(model, str) and model.startswith("gtr/")
    if is_gtr:
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemSegTrain as train_cls,
        )
        from autovision.tasks.semantic_segmentation.gtr import (
            GTRSemSegTrainArgs as args_cls,
        )

        # ``gtr/<s|m|l|x>`` selects the official GTR semseg width;
        # ``model_args.checkpoint`` loads the official MIT weights
        # (registered asset name or path).
        resolved_model_args = pydantic_model_validate(
            args_cls,
            {
                "num_classes": num_classes,
                "variant": model.split("/", 1)[1],
                **(model_args or {}),
            },
        )
    elif is_eomt:
        if model.startswith("dinov3/eomt"):
            from autovision.tasks.semantic_segmentation.dinov3_eomt import (
                Dinov3EomtSemanticSegmentationTrain as train_cls,
            )
            from autovision.tasks.semantic_segmentation.dinov3_eomt import (
                Dinov3EomtSemanticSegmentationTrainArgs as args_cls,
            )
        else:
            from autovision.tasks.semantic_segmentation.dinov2_eomt import (
                Dinov2EomtSemanticSegmentationTrain as train_cls,
            )
            from autovision.tasks.semantic_segmentation.dinov2_eomt import (
                Dinov2EomtSemanticSegmentationTrainArgs as args_cls,
            )
        resolved_model_args = pydantic_model_validate(
            args_cls, {"num_classes": num_classes, **(model_args or {})}
        )
    else:
        args_cls = LinearSemanticSegmentationTrainArgs
        train_cls = LinearSemanticSegmentationTrain
        resolved_model_args = pydantic_model_validate(
            args_cls, {"num_classes": num_classes, "backbone": model, **(model_args or {})}
        )
    save_args = pydantic_model_validate(_SaveCheckpointArgs, dict(save_checkpoint_args or {}))
    batch_size_resolved = (
        args_cls.default_batch_size if batch_size == "auto" else int(batch_size)
    )
    steps_resolved = args_cls.default_steps if steps == "auto" else int(steps)
    num_devices = common_helpers.resolve_num_devices(devices)
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, len(train_pairs)), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = max(1, global_batch_size // num_devices)

    if is_eomt:
        from autovision.tasks.shared.eomt.transforms import (
            build_eomt_segmentation_transform,
        )

        normalize = resolved_model_args.normalize

        def _seg_transform(train: bool):
            return build_eomt_segmentation_transform(
                image_size=(resolved_model_args.image_size, resolved_model_args.image_size),
                normalize_mean=normalize.mean,
                normalize_std=normalize.std,
                train=train,
                horizontal_flip_prob=resolved_model_args.horizontal_flip_prob,
                scale_range=tuple(resolved_model_args.scale_jitter),
                ignore_index=resolved_model_args.ignore_index,
            )
    else:

        def _seg_transform(train: bool):
            return build_segmentation_transform(resolved_model_args, train=train)

    train_transform = _seg_transform(train=True)
    val_transform = _seg_transform(train=False)

    def _collate(batch):
        images = torch.stack([item["image"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        return {"image": images, "mask": masks}

    train_dataset = SemanticSegmentationDataset(
        train_pairs, transform=train_transform
    )
    val_dataset = (
        SemanticSegmentationDataset(val_pairs, transform=val_transform)
        if val_pairs
        else None
    )
    train_loader = common_helpers.build_task_dataloader(
        train_dataset, batch_size=per_device_batch,
        num_workers=num_workers_resolved, train=True,
    )
    train_loader.collate_fn = _collate
    val_loader = (
        common_helpers.build_task_dataloader(
            val_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=False,
        )
        if val_dataset is not None
        else None
    )
    if val_loader is not None:
        val_loader.collate_fn = _collate
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    train_model = train_cls(resolved_model_args)
    precision_resolved = precision
    if precision_resolved == "auto":
        precision_resolved = (
            "bf16-mixed"
            if accelerator_supports_bf16()
            else "32-true"
        )
    from autovision.debug.debug_args import DebugArgs
    from autovision.observability.events import track_training_started
    from autovision.observability.loggers import (
        finalize_loggers,
        get_loggers,
        log_hyperparams_to_loggers,
    )

    resolved_debug_args = pydantic_model_validate(DebugArgs, dict(debug_args or {}))
    logger_instances = get_loggers(loggers, out_dir)
    track_training_started(
        task_type="semantic_segmentation", model_name=model,
        batch_size=global_batch_size, devices=num_devices,
        steps=steps_resolved,
    )
    engine = TaskEngine()
    hyperparams = {
        "task": "semantic_segmentation",
        "steps": steps_resolved,
        "batch_size": global_batch_size,
        "num_classes": num_classes,
    }
    if _is_rank_zero():
        log_hyperparams_to_loggers(logger_instances, hyperparams)
    fit_result = engine.fit_task(
        train_model,
        train_loader,
        val_loader,
        out_dir=out_dir,
        steps=steps_resolved,
        gradient_accumulation_steps=gradient_accumulation_steps,
        step_hooks=[],
        val_every_num_steps=(trainer_args or {}).get("val_every_num_steps"),
        early_stopping_patience=(trainer_args or {}).get("early_stopping_patience"),
        gradient_checkpointing=common_helpers.resolve_gradient_checkpointing(trainer_args),
        parallel=common_helpers.resolve_parallel(trainer_args),
        early_stopping_min_delta=float(
            (trainer_args or {}).get("early_stopping_min_delta", 0.0)
        ),
        save_every_num_steps=save_args.save_every_num_steps,
        save_last=save_args.save_last,
        save_best=save_args.save_best,
        devices=devices,
        precision=precision_resolved,
        seed=seed,
        checkpoint=Path(checkpoint) if checkpoint else None,
        resume_interrupted=resume_interrupted,
        logger_fn=lambda metrics, step: [
            lg.log_metrics(metrics, step) for lg in logger_instances
        ] if _is_rank_zero() else None,
        log_every_n_steps=(trainer_args or {}).get(
            "log_every_n_steps", min(50, max(1, len(train_loader)))
        ),
        normalize_args=resolved_model_args.normalize.to_dict(),
        model_class_path=train_model.get_task_model().class_path,
        model_init_args={
            **train_model.get_task_model().init_args,
            "model_name": model,
        },
        debug_args=resolved_debug_args,
        visualize_examples=False,
    )
    finalize_loggers(logger_instances)
    if _is_rank_zero():
        logger.info("Segmentation training finished: %d steps.", fit_result.steps_run)
    return TaskTrainResult(
        out_dir=out_dir,
        steps_run=fit_result.steps_run,
        checkpoint_path=fit_result.checkpoint_path,
        best_checkpoint_path=fit_result.best_checkpoint_path,
        exported_model_path=fit_result.exported_model_path,
        best_exported_model_path=fit_result.best_exported_model_path,
        best_metrics=fit_result.best_metrics,
        stopped_early=fit_result.stopped_early,
        stop_step=fit_result.stop_step,
        classes={index: str(index) for index in range(num_classes)},
    )
