# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Instance segmentation finetuning."""

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

def train_instance_segmentation(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "dinov2/eomt",
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
    """Instance segmentation finetuning (EOMT, HF-backed).

    ``model`` accepts ``"dinov2/eomt"`` (default) or ``"dinov3/eomt"``.
    ``data`` accepts a dict (or YAML) ``{"train": {"annotations": coco.json,
    "images": dir}, "val": {...}}``. Annotations carry ``segmentation``
    polygons/RLE per instance; internal class ids follow the train
    categories in insertion order.
    """
    from autovision.data.coco_instance_segmentation_dataset import (
        COCOInstanceSegmentationDataset,
    )
    from autovision.tasks.shared.eomt.transforms import (
        build_eomt_instance_transform,
    )

    is_gtr = isinstance(model, str) and model.startswith("gtr/")
    if is_gtr:
        from autovision.tasks.instance_segmentation.gtr import (
            GTRSegTrain as train_cls,
        )
        from autovision.tasks.instance_segmentation.gtr import (
            GTRSegTrainArgs as args_cls,
        )

        # ``gtr/<s|m|l|x>`` selects the official GTR seg width;
        # ``model_args.checkpoint`` loads the official MIT weights.
        resolved_model_args = pydantic_model_validate(
            args_cls,
            {
                "num_classes": num_classes,
                "variant": model.split("/", 1)[1],
                **(model_args or {}),
            },
        )
    elif model.startswith("dinov3/eomt"):
        from autovision.tasks.instance_segmentation.dinov3_eomt import (
            Dinov3EomtInstanceSegmentationTrain as train_cls,
        )
        from autovision.tasks.instance_segmentation.dinov3_eomt import (
            Dinov3EomtInstanceSegmentationTrainArgs as args_cls,
        )
    else:
        from autovision.tasks.instance_segmentation.dinov2_eomt import (
            Dinov2EomtInstanceSegmentationTrain as train_cls,
        )
        from autovision.tasks.instance_segmentation.dinov2_eomt import (
            Dinov2EomtInstanceSegmentationTrainArgs as args_cls,
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
            "The instance segmentation data spec must provide a 'train' entry "
            "with 'annotations' and 'images'."
        )

    def _dataset(split: str):
        spec = payload[split]
        annotations = Path(spec["annotations"])
        images_dir = Path(spec["images"]) if spec.get("images") else None
        return COCOInstanceSegmentationDataset(
            annotations, images_dir=images_dir, transform=None
        )

    if not is_gtr:
        resolved_model_args = pydantic_model_validate(
            args_cls,
            {"num_classes": num_classes, **(model_args or {})},
        )
    save_args = pydantic_model_validate(_SaveCheckpointArgs, dict(save_checkpoint_args or {}))
    batch_size_resolved = (
        args_cls.default_batch_size
        if batch_size == "auto"
        else int(batch_size)
    )
    steps_resolved = (
        args_cls.default_steps
        if steps == "auto"
        else int(steps)
    )
    num_devices = common_helpers.resolve_num_devices(devices)
    probe = _dataset("train")
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, len(probe)), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = max(1, global_batch_size // num_devices)

    def _transformed(dataset: COCOInstanceSegmentationDataset, train: bool):
        image_size = resolved_model_args.image_size
        if isinstance(image_size, (tuple, list)):
            image_size = int(image_size[0])
        dataset.transform = build_eomt_instance_transform(
            image_size=image_size,
            normalize_mean=resolved_model_args.normalize.mean,
            normalize_std=resolved_model_args.normalize.std,
            train=train,
            horizontal_flip_prob=resolved_model_args.horizontal_flip_prob,
        )
        return dataset

    train_dataset = _transformed(probe, train=True)
    val_dataset = _transformed(_dataset("val"), train=False) if "val" in payload else None

    def _collate(items):
        return {
            "image": torch.stack([item["image"] for item in items]),
            "masks": [item["masks"] for item in items],
            "classes": [item["classes"] for item in items],
            "bboxes": [item["bboxes"] for item in items],
            "original_size": [item["original_size"] for item in items],
        }

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
    train_model.set_categories(
        {index: str(index) for index in range(num_classes)}
    )
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
        task_type="instance_segmentation", model_name=model,
        batch_size=global_batch_size, devices=num_devices,
        steps=steps_resolved,
    )
    engine = TaskEngine()
    hyperparams = {
        "task": "instance_segmentation",
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
        logger.info("Instance segmentation training finished: %d steps.", fit_result.steps_run)
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
