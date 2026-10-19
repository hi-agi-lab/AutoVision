# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pose-estimation finetuning (GTR pose family, doc 31 §11 S5b)."""

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


def train_pose_estimation(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "gtr/s",
    *,
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
    """Person keypoint finetuning (GTR pose family).

    ``model`` accepts ``"gtr/<s|m|l|x>"``. ``data`` accepts a dict (or
    YAML) ``{"train": {"annotations": json, "images": dir}, "val":
    {...}}`` with COCO ``person_keypoints`` annotation files; ``val`` is
    optional. Validation reports the official COCO keypoint AP (OKS)
    against the raw val annotations; ``model_args.checkpoint`` loads the
    official MIT weights (registered asset name or path).

    Precision note: the official pose finetune config runs
    ``use_amp: False``; ``precision="auto"`` follows the family default
    (bf16 when the accelerator supports it) — pass ``"32-true"`` for the
    official recipe.
    """
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset
    from autovision.tasks.pose_estimation.gtr import (
        GTRPoseEstimationTrain as train_cls,
    )
    from autovision.tasks.pose_estimation.gtr import (
        GTRPoseEstimationTrainArgs as args_cls,
    )
    from autovision.tasks.pose_estimation.gtr.transforms import (
        build_pose_transform,
    )

    if isinstance(model, str) and not model.startswith("gtr/"):
        raise ValueError(
            f"Unsupported pose model '{model}' — expected 'gtr/<s|m|l|x>'."
        )
    variant = model.split("/", 1)[1]

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
            "The pose data spec must provide a 'train' entry with "
            "'annotations' and 'images' entries."
        )

    resolved_model_args = pydantic_model_validate(
        args_cls, {"variant": variant, **(model_args or {})}
    )
    save_args = pydantic_model_validate(
        _SaveCheckpointArgs, dict(save_checkpoint_args or {})
    )

    # One shared train transform: the engine feeds it step updates
    # (mosaic/mixup windows) through the step-hook interface.
    train_transform = build_pose_transform(resolved_model_args, train=True)

    def _dataset(split: str, train: bool) -> CocoKeypointDataset:
        entry = payload[split]
        return CocoKeypointDataset(
            entry["annotations"],
            entry["images"],
            transform=(
                train_transform
                if train
                else build_pose_transform(resolved_model_args, train=False)
            ),
            train=train,
            num_body_points=resolved_model_args.num_body_points,
        )

    train_dataset = _dataset("train", train=True)
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")
    val_dataset = _dataset("val", train=False) if "val" in payload else None

    def _collate(items):
        return {
            "image": torch.stack([item["image"] for item in items]),
            "keypoints": [item["keypoints"] for item in items],
            "bboxes": [item["bboxes"] for item in items],
            "classes": [item["classes"] for item in items],
            "area": [item["area"] for item in items],
            "original_size": [item["original_size"] for item in items],
            "index": [int(item["index"]) for item in items],
        }

    batch_size_resolved = (
        args_cls.default_batch_size if batch_size == "auto" else int(batch_size)
    )
    steps_resolved = args_cls.default_steps if steps == "auto" else int(steps)
    num_devices = common_helpers.resolve_num_devices(devices)
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, len(train_dataset)), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = max(1, global_batch_size // num_devices)

    train_loader = common_helpers.build_task_dataloader(
        train_dataset, batch_size=per_device_batch,
        num_workers=num_workers_resolved, train=True,
        # Step-windowed augmentation: see build_task_dataloader's note.
        persistent_workers=False,
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

    train_model = train_cls(resolved_model_args)
    categories = (
        val_dataset.categories if val_dataset is not None else train_dataset.categories
    )
    train_model.set_categories(categories)
    if "val" in payload:
        train_model.set_val_annotations(payload["val"]["annotations"])

    precision_resolved = precision
    if precision_resolved == "auto":
        precision_resolved = (
            "bf16-mixed" if accelerator_supports_bf16() else "32-true"
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
        task_type="pose_estimation", model_name=model,
        batch_size=global_batch_size, devices=num_devices,
        steps=steps_resolved,
    )
    engine = TaskEngine()
    hyperparams = {
        "task": "pose_estimation",
        "steps": steps_resolved,
        "batch_size": global_batch_size,
        "variant": variant,
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
        step_hooks=[train_transform],
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
        logger.info("Pose training finished: %d steps.", fit_result.steps_run)
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
        classes=categories,
    )
