# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Oriented (rotated-box) object detection finetuning (YOLO-OBB format)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from autovision.cli import common_helpers
from autovision.cli.train import _is_rank_zero
from autovision.cli.train_task.task_helpers import (
    TaskTrainResult,
    _dataset_size,
    _SaveCheckpointArgs,
)
from autovision.config.validate import pydantic_model_validate
from autovision.data.mmap_helpers import dataset_mmap_context
from autovision.engine import TaskEngine, seed_everything
from autovision.runtime.accelerator import accelerator_supports_bf16
from autovision.runtime.logging import set_up_file_logging
from autovision.types import PathLike

logger = logging.getLogger("autovision")

_DATA_FORMATS_ORIENTED = {"yolo-obb"}


def _load_oriented_data_config(data: PathLike | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        payload = dict(data)
    else:
        with open(data) as file:
            payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("'data' must be a dict or a YAML file with a mapping.")
    payload.setdefault("format", "yolo-obb")
    if payload["format"] not in _DATA_FORMATS_ORIENTED:
        supported = ", ".join(sorted(_DATA_FORMATS_ORIENTED))
        raise ValueError(
            f"Unknown oriented detection data format: '{payload['format']}'. "
            f"Supported: {supported}."
        )
    return payload


def train_oriented_object_detection(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "rtdetr-obb/resnet18",
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
    transform_args: dict[str, Any] | None = None,
    save_checkpoint_args: dict[str, Any] | None = None,
    trainer_args: dict[str, Any] | None = None,
    debug_args: dict[str, Any] | None = None,
    loggers: dict[str, dict[str, Any] | None] | None = None,
) -> TaskTrainResult:
    """Oriented (rotated-box) detection finetuning (YOLO-OBB format).

    ``data`` uses the Ultralytics YOLO-OBB layout with 8-component corner
    labels; ``model`` dispatches the oriented RT-DETR family
    (``"rtdetr-obb/<backbone>"``). Returns a :class:`TaskTrainResult`.
    """
    from autovision.augment.detection import (
        OrientedObjectDetectionCollateFunction,
        OrientedObjectDetectionTransform,
        OrientedObjectDetectionTransformArgs,
        build_oriented_val_transform_args,
    )
    from autovision.data.yolo_oriented_object_detection_dataset import (
        OrientedObjectDetectionDataset,
        YOLOOrientedObjectDetectionDataArgs,
    )
    from autovision.tasks.oriented_detection.rtdetr import (
        RTDETROrientedObjectDetectionTrain,
        RTDETROrientedObjectDetectionTrainArgs,
    )

    # The seed must be applied before the assembly builds random
    # modules (model init, transform parameters): the engine re-seeds
    # inside fit(), which is already too late for construction.
    seed_everything(seed)
    out_dir = common_helpers.get_out_dir(out, resume_interrupted, overwrite)
    set_up_file_logging(out_dir / "train.log")

    # 1. data
    data_config_cls = YOLOOrientedObjectDetectionDataArgs
    data_payload = _load_oriented_data_config(data)
    if isinstance(data, (str, Path)) and "data_config_file" in {
        f for f in data_config_cls.model_fields
    }:
        data_payload["data_config_file"] = str(Path(data).resolve())
    data_args = pydantic_model_validate(data_config_cls, data_payload)
    base_dir = Path(data).parent if isinstance(data, (str, Path)) else None
    data_args.resolve_data_paths(base_dir)
    classes = data_args.classes()
    num_classes = len(classes)

    # 2. model args (num_classes inferred from the data).
    is_gtr = isinstance(model, str) and model.startswith("gtr/")
    if is_gtr:
        from autovision.tasks.oriented_detection.gtr import (
            GTRROBBTrain as train_cls,
        )
        from autovision.tasks.oriented_detection.gtr import (
            GTRROBBTrainArgs as args_cls,
        )

        resolved_model_args = pydantic_model_validate(
            args_cls,
            {
                "num_classes": num_classes,
                "variant": model.split("/", 1)[1],
                **(model_args or {}),
            },
        )
    elif isinstance(model, str) and model.startswith("rtdetr-obb/"):
        train_cls = RTDETROrientedObjectDetectionTrain
        args_cls = RTDETROrientedObjectDetectionTrainArgs
        resolved_model_args = pydantic_model_validate(
            args_cls,
            {
                "num_classes": num_classes,
                "backbone": model.split("/", 1)[1],
                **(model_args or {}),
            },
        )
    else:
        raise ValueError(
            f"Unsupported oriented detection model '{model}'; supported: "
            f"'rtdetr-obb/<backbone>', 'gtr/<s|x>'."
        )
    default_image_size = resolved_model_args.image_size

    # 3. save args
    save_args = pydantic_model_validate(
        _SaveCheckpointArgs, dict(save_checkpoint_args or {})
    )

    # 4. batch size / steps
    batch_size_resolved = (
        RTDETROrientedObjectDetectionTrainArgs.default_batch_size
        if batch_size == "auto"
        else int(batch_size)
    )
    steps_resolved = (
        RTDETROrientedObjectDetectionTrainArgs.default_steps
        if steps == "auto"
        else int(steps)
    )

    # 5. datasets (mmap-managed)
    num_devices = common_helpers.resolve_num_devices(devices)
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, _dataset_size(data_args, "train")), num_devices
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = global_batch_size // num_devices

    with dataset_mmap_context(
        lambda: data_args.list_image_info("train"),
        data=str(data_args.train),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
        fingerprint_extra=repr(sorted(getattr(data_args, "ignore_classes", None) or [])),
    ) as train_rows, dataset_mmap_context(
        lambda: data_args.list_image_info("val"),
        data=str(data_args.val),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
        fingerprint_extra=repr(sorted(getattr(data_args, "ignore_classes", None) or [])),
    ) as val_rows:
        train_dataset = OrientedObjectDetectionDataset(data_args, train_rows)
        val_dataset = OrientedObjectDetectionDataset(data_args, val_rows)
        train_loader = common_helpers.build_task_dataloader(
            train_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=True,
            # Step-windowed augmentation: see build_task_dataloader's note.
            persistent_workers=False,
        )
        val_loader = common_helpers.build_task_dataloader(
            val_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=False,
        )
        if len(train_dataset) == 0:
            raise ValueError("The training dataset is empty.")

        # 6. transforms (straight validation — the oriented recipe has no
        # step-window augmentations to resolve).
        train_transform_args = pydantic_model_validate(
            OrientedObjectDetectionTransformArgs,
            {"image_size": default_image_size, **(transform_args or {})},
        )
        val_transform_args = build_oriented_val_transform_args(train_transform_args)
        train_transform = OrientedObjectDetectionTransform(train_transform_args)
        val_transform = OrientedObjectDetectionTransform(val_transform_args)
        train_dataset.set_transform(train_transform)
        val_dataset.set_transform(val_transform)
        train_collate = OrientedObjectDetectionCollateFunction(train_transform_args, train=True)
        val_collate = OrientedObjectDetectionCollateFunction(val_transform_args, train=False)
        train_loader.collate_fn = train_collate
        val_loader.collate_fn = val_collate

        # 7. model
        train_model = train_cls(resolved_model_args)
        train_model.set_categories(classes)

        # 8. fit
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

        resolved_debug_args = pydantic_model_validate(
            DebugArgs, dict(debug_args or {})
        )
        logger_instances = get_loggers(loggers, out_dir)
        track_training_started(
            task_type="oriented_object_detection", model_name=model,
            batch_size=global_batch_size, devices=num_devices,
            steps=steps_resolved,
        )
        engine = TaskEngine()
        hyperparams = {
            "task": "oriented_object_detection",
            "steps": steps_resolved,
            "batch_size": global_batch_size,
            "num_classes": num_classes,
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
            step_hooks=[train_transform, train_collate],
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
            normalize_args=train_transform_args.normalize.to_dict(),
            model_class_path=train_model.get_task_model().class_path,
            model_init_args={
                **train_model.get_task_model().init_args,
                "model_name": model,
            },
            debug_args=resolved_debug_args,
        )
        finalize_loggers(logger_instances)

    if _is_rank_zero():
        logger.info("Task training finished: %d steps.", fit_result.steps_run)
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
