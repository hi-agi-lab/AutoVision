# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification finetuning (ImageFolder layout)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

def _resolve_classification_roots(
    data: PathLike | dict[str, Any],
) -> dict[str, Path]:
    """Normalize the classification ``data`` spec into train/val roots.

    Accepted: a dict (or YAML file) ``{"train": path, "val": path}``, or a
    root directory containing ``train/`` and ``val/`` subdirectories.
    """
    if isinstance(data, dict):
        payload = dict(data)
    elif isinstance(data, (str, Path)) and Path(data).is_file():
        with open(data) as file:
            payload = yaml.safe_load(file)
        if not isinstance(payload, dict):
            raise ValueError(
                f"The classification data config '{data}' must contain a mapping."
            )
    else:
        root = Path(data)
        payload = {
            split: root / split
            for split in ("train", "val")
            if (root / split).is_dir()
        }
    roots: dict[str, Path] = {}
    for split in ("train", "val"):
        if split in payload:
            roots[split] = Path(payload[split])
    if "train" not in roots:
        raise ValueError(
            "The classification data spec must provide a 'train' directory "
            "(ImageFolder layout: one subdirectory per class)."
        )
    return roots



def train_image_classification(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "torchvision/resnet18",
    *,
    num_classes: int | None = None,
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
    """Image classification finetuning over an ImageFolder-style dataset."""
    from autovision.data.image_classification_dataset import (
        ImageClassificationDataset,
        list_classification_files,
    )
    from autovision.tasks.classification import (
        ImageClassificationTrain,
        ImageClassificationTrainArgs,
    )
    from autovision.tasks.classification.transforms import (
        build_classification_transform,
    )

    # The seed must be applied before the assembly builds random
    # modules (model init, transform parameters): the engine re-seeds
    # inside fit(), which is already too late for construction.
    seed_everything(seed)
    out_dir = common_helpers.get_out_dir(out, resume_interrupted, overwrite)
    set_up_file_logging(out_dir / "train.log")

    roots = _resolve_classification_roots(data)
    train_files, train_labels, classes = list_classification_files(roots["train"])
    if num_classes is not None and num_classes != len(classes):
        raise ValueError(
            f"num_classes={num_classes} does not match the dataset "
            f"({len(classes)} classes)."
        )
    val_files = val_labels = None
    if "val" in roots:
        val_files, val_labels, val_classes = list_classification_files(roots["val"])
        if val_classes != classes:
            raise ValueError(
                "The train and val datasets declare different class sets."
            )

    resolved_model_args = pydantic_model_validate(
        ImageClassificationTrainArgs,
        {
            "num_classes": len(classes),
            "backbone": model,
            **(model_args or {}),
        },
    )
    save_args = pydantic_model_validate(_SaveCheckpointArgs, dict(save_checkpoint_args or {}))
    batch_size_resolved = (
        ImageClassificationTrainArgs.default_batch_size
        if batch_size == "auto"
        else int(batch_size)
    )
    steps_resolved = (
        ImageClassificationTrainArgs.default_steps if steps == "auto" else int(steps)
    )
    num_devices = common_helpers.resolve_num_devices(devices)
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, len(train_files)), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = max(1, global_batch_size // num_devices)

    train_transform = build_classification_transform(resolved_model_args, train=True)
    val_transform = build_classification_transform(resolved_model_args, train=False)
    train_dataset = ImageClassificationDataset(
        train_files, train_labels, classes, transform=train_transform
    )
    val_dataset = (
        ImageClassificationDataset(val_files, val_labels, classes, transform=val_transform)
        if val_files is not None
        else None
    )
    train_loader = common_helpers.build_task_dataloader(
        train_dataset, batch_size=per_device_batch,
        num_workers=num_workers_resolved, train=True,
    )
    val_loader = (
        common_helpers.build_task_dataloader(
            val_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=False,
        )
        if val_dataset is not None
        else None
    )
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    train_model = ImageClassificationTrain(resolved_model_args)
    train_model.set_categories(classes)

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
        task_type="image_classification", model_name=model,
        batch_size=global_batch_size, devices=num_devices,
        steps=steps_resolved,
    )
    engine = TaskEngine()
    hyperparams = {
        "task": "image_classification",
        "steps": steps_resolved,
        "batch_size": global_batch_size,
        "num_classes": len(classes),
        "classes": list(classes.values()),
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
        logger.info("Classification training finished: %d steps.", fit_result.steps_run)
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
        classes=classes,
    )
