# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Object detection finetuning (COCO / YOLO formats)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from autovision.augment.detection import (
    RTDETRDetectionCollateFunction,
    RTDETRDetectionTransform,
    RTDETRDetectionTransformArgs,
    TwoViewDetectionCollateFunction,
    build_val_transform_args,
)
from autovision.cli import common_helpers
from autovision.cli.train import _is_rank_zero
from autovision.cli.train_task.task_helpers import (
    TaskTrainResult,
    _dataset_size,
    _SaveCheckpointArgs,
)
from autovision.config.validate import pydantic_model_validate
from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.mmap_helpers import dataset_mmap_context
from autovision.data.object_detection_dataset import ObjectDetectionDataset
from autovision.data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from autovision.engine import TaskEngine, seed_everything
from autovision.runtime.accelerator import accelerator_supports_bf16
from autovision.runtime.logging import set_up_console_logging, set_up_file_logging
from autovision.tasks.detection.rtdetr import (
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.types import PathLike

logger = logging.getLogger("autovision")

_DATA_FORMATS = {"coco": COCOObjectDetectionDataArgs, "yolo": YOLOObjectDetectionDataArgs}


def _load_data_config(data: PathLike | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, str) and data == "coco8":
        from autovision.data.coco8 import ensure_coco8

        return ensure_coco8()
    if isinstance(data, dict):
        payload = dict(data)
    else:
        with open(data) as file:
            payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("'data' must be a dict or a YAML file with a mapping.")
    payload.setdefault("format", "yolo")
    if "format" in payload and payload["format"] not in _DATA_FORMATS:
        supported = ", ".join(sorted(_DATA_FORMATS))
        raise ValueError(
            f"Unknown detection data format: '{payload['format']}'. Supported: {supported}."
        )
    return payload


def train_object_detection(
    out: PathLike,
    data: PathLike | dict[str, Any],
    model: str = "rtdetr/resnet18",
    *,
    steps: int | str = "auto",
    batch_size: int | str = "auto",
    gradient_accumulation_steps: int | str = 1,
    nominal_batch_size: int = 64,
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
    recipe: str | Path | None = None,
    hard_examples: PathLike | None = None,
    hard_example_alpha: float = 1.0,
    hard_example_max_repeat: float = 5.0,
    long_tail_sampler: str | dict[str, Any] | None = None,
) -> TaskTrainResult:
    """Object detection finetuning (COCO/YOLO formats).

    ``recipe`` names a built-in detection recipe (``finetune``,
    ``coco-finetune``, ``low-data``) or a YAML file with
    ``model_args``/``transform_args``/``trainer_args`` overlays. Explicit
    keyword values win over recipe values.

    ``hard_examples`` points at an ``analyze-errors`` manifest
    (``hard_examples.json``); the training split is sampled with weights
    ``min(max_repeat, 1 + alpha * normalized_error)``. Single-device only:
    a DDP DistributedSampler would override these weights, so the
    combination is rejected rather than silently ignored.

    ``long_tail_sampler`` is ``"class_balanced"`` / ``"repeat_factor"`` or
    a config dict (``power`` / ``threshold`` / ``max_repeat`` / ``seed``).
    It is mutually exclusive with ``hard_examples`` (both replace the train
    sampler) and is also single-device only.

    See the design doc (§8) for the contract; returns a
    :class:`TaskTrainResult` with checkpoint/export paths and best metrics.
    """
    if recipe is not None:
        from autovision.cli.recipes import apply_recipe

        model_args, transform_args, trainer_args = apply_recipe(
            recipe,
            model_args=model_args,
            transform_args=transform_args,
            trainer_args=trainer_args,
        )
    common_helpers.validate_trainer_args(
        trainer_args,
        allowed={
            "val_every_num_steps",
            "log_every_n_steps",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "gradient_checkpointing",
            "parallel",
        },
    )
    trainer_args_resolved = dict(trainer_args or {})
    # The seed must be applied before the assembly builds random
    # modules (model init, transform parameters): the engine re-seeds
    # inside fit(), which is already too late for construction.
    seed_everything(seed)
    out_dir = common_helpers.get_out_dir(out, resume_interrupted, overwrite)
    set_up_file_logging(out_dir / "train.log")
    # Mirror the progress lines (step summary, validation tables, best-
    # checkpoint notices) to the console, reference-log style — the run dir
    # keeps the timestamped copy in train.log.
    set_up_console_logging()

    # 1. data
    data_config_cls = _DATA_FORMATS[_load_data_config(data)["format"]]
    data_payload = _load_data_config(data)
    if isinstance(data, (str, Path)) and "data_config_file" in {
        f for f in data_config_cls.model_fields
    }:
        data_payload["data_config_file"] = str(Path(data).resolve())
    data_args = pydantic_model_validate(data_config_cls, data_payload)
    base_dir = Path(data).parent if isinstance(data, (str, Path)) else None
    data_args.resolve_data_paths(base_dir)
    classes = data_args.classes()
    num_classes = len(classes)

    # 2. model args (num_classes inferred from the data). The model string
    # dispatches the detection family: "rtdetr/*" (default), "detr/*"
    # (SSL-backbone DETR), "dfine/*" (distributional refinement),
    # "rfdetr/*" (RF-DETR via HF transformers) or "picodet/*" (dense
    # anchor-point PicoDet).
    is_detr = isinstance(model, str) and model.startswith("detr/")
    is_dfine = isinstance(model, str) and model.startswith("dfine/")
    is_picodet = isinstance(model, str) and model.startswith("picodet/")
    is_rfdetr = isinstance(model, str) and model.startswith("rfdetr/")
    if is_detr:
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
            detr_default_batch_size,
)

        args_cls = DETRObjectDetectionTrainArgs
        train_cls: type = DETRObjectDetectionTrain
        family_model_args = {"backbone": model.split("/", 1)[1], **(model_args or {})}
    elif is_dfine:
        from autovision.tasks.detection.dfine import (
            DFINEObjectDetectionTrain,
            DFINEObjectDetectionTrainArgs,
        )

        args_cls = DFINEObjectDetectionTrainArgs
        train_cls = DFINEObjectDetectionTrain
        family_model_args = dict(model_args or {})
    elif is_picodet:
        from autovision.tasks.detection.picodet import (
            PicoDetObjectDetectionTrain,
            PicoDetObjectDetectionTrainArgs,
        )

        args_cls = PicoDetObjectDetectionTrainArgs
        train_cls = PicoDetObjectDetectionTrain
        family_model_args = {
            "backbone": model.split("/", 1)[1], **(model_args or {})
        }
    elif is_rfdetr:
        from autovision.tasks.detection.rfdetr import (
            RFDETRObjectDetectionTrain,
            RFDETRObjectDetectionTrainArgs,
        )

        args_cls = RFDETRObjectDetectionTrainArgs
        train_cls = RFDETRObjectDetectionTrain
        family_model_args = {
            "model": model.split("/", 1)[1], **(model_args or {})
        }
    else:
        args_cls = RTDETRObjectDetectionTrainArgs
        train_cls = RTDETRObjectDetectionTrain
        family_model_args = dict(model_args or {})
    resolved_model_args = pydantic_model_validate(
        args_cls, {"num_classes": num_classes, **family_model_args}
    )
    if is_detr or is_rfdetr:
        # Snap the image size to the backbone's patch grid (640 -> 644 for
        # patch-14 backbones; "auto" -> variant-native for rfdetr)
        # BEFORE the transform pipeline reads it.
        resolved_model_args.resolve_image_size()
    default_image_size = resolved_model_args.image_size

    # SSL two_crop pipeline (design doc 14 §4.5): every loader item becomes
    # an (view1, view2) pair — the batch doubles to 2B view-major rows.
    # The batch-matched baseline (ssl.mode="off") runs the same doubled
    # pipeline, so both runs see identical data throughput.
    ssl_args = getattr(resolved_model_args, "ssl", None)
    ssl_two_crop = ssl_args is not None and ssl_args.view == "two_crop"

    # 3. save args
    save_args = pydantic_model_validate(
        _SaveCheckpointArgs, dict(save_checkpoint_args or {})
    )

    # 4. batch size / steps
    # Baseline reference batch is variant-scoped for the DETR family
    # (32 across the family, 16 for dinov2); every other family keeps its
    # class-level default.
    if is_detr:
        default_bs = detr_default_batch_size(resolved_model_args.backbone)
    else:
        default_bs = args_cls.default_batch_size
    batch_size_resolved = (
        default_bs if batch_size == "auto" else int(batch_size)
    )
    default_steps = args_cls.default_steps
    steps_resolved = default_steps if steps == "auto" else int(steps)

    # 5. datasets (mmap-managed) — built without transforms first so the
    # loader length is known for the step-window resolution.
    num_devices = common_helpers.resolve_num_devices(devices)
    global_batch_size = common_helpers.get_global_batch_size(
        batch_size_resolved, max(1, _dataset_size(data_args, "train")), num_devices
    )
    grad_accum_resolved = common_helpers.resolve_gradient_accumulation(
        gradient_accumulation_steps, nominal_batch_size, global_batch_size
    )
    num_workers_resolved = common_helpers.get_num_workers(num_workers, num_devices)
    per_device_batch = global_batch_size // num_devices

    with dataset_mmap_context(
        lambda: data_args.list_image_info("train"),
        data=str(
            data_args.train.annotations
            if isinstance(data_args, COCOObjectDetectionDataArgs)
            else data_args.train
        ),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
        fingerprint_extra=repr(sorted(getattr(data_args, "ignore_classes", None) or [])),
    ) as train_rows, dataset_mmap_context(
        lambda: data_args.list_image_info("val"),
        data=str(
            data_args.val.annotations
            if isinstance(data_args, COCOObjectDetectionDataArgs)
            else data_args.val
        ),
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
        fingerprint_extra=repr(sorted(getattr(data_args, "ignore_classes", None) or [])),
    ) as val_rows:
        train_dataset = ObjectDetectionDataset(data_args, train_rows)
        if ssl_two_crop:
            from autovision.data.two_view import TwoViewObjectDetectionDataset

            train_dataset = TwoViewObjectDetectionDataset(train_dataset)
        val_dataset = ObjectDetectionDataset(data_args, val_rows)
        train_sampler = None
        if hard_examples is not None and long_tail_sampler is not None:
            raise ValueError(
                "hard_examples and long_tail_sampler both replace the train "
                "sampler; choose one weighting strategy."
            )
        if hard_examples is not None:
            if num_devices > 1:
                raise ValueError(
                    "hard_examples weighted sampling is single-device only "
                    "(a DDP DistributedSampler would override the weights); "
                    "run with devices=1 or drop hard_examples."
                )
            from autovision.data.hard_examples import (
                HardExampleManifest,
                build_weighted_sampler,
            )

            manifest = HardExampleManifest.load(hard_examples)
            train_paths = [str(row["image_path"]) for row in train_rows]
            train_sampler = build_weighted_sampler(
                manifest,
                train_paths,
                alpha=hard_example_alpha,
                max_repeat=hard_example_max_repeat,
                seed=seed,
            )
        elif long_tail_sampler is not None:
            if num_devices > 1:
                raise ValueError(
                    "long_tail_sampler weighted sampling is single-device only "
                    "(a DDP DistributedSampler would override the weights); "
                    "run with devices=1 or drop long_tail_sampler."
                )
            from autovision.data.samplers import build_long_tail_sampler

            if isinstance(long_tail_sampler, str):
                sampler_config: dict[str, Any] = {"strategy": long_tail_sampler}
            elif isinstance(long_tail_sampler, dict):
                sampler_config = dict(long_tail_sampler)
            else:
                raise ValueError(
                    "long_tail_sampler must be a strategy name or a config dict."
                )
            allowed = {
                "strategy", "power", "threshold", "max_repeat", "seed", "num_samples",
            }
            unknown = set(sampler_config) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown long_tail_sampler key(s) {sorted(unknown)}; "
                    f"accepted: {sorted(allowed)}."
                )
            sampler_config.setdefault("seed", seed)
            train_sampler = build_long_tail_sampler(
                train_rows, **sampler_config
            )
        train_loader = common_helpers.build_task_dataloader(
            train_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=True,
            # Step-windowed augmentation: see build_task_dataloader's note.
            persistent_workers=False,
            sampler=train_sampler,
        )
        val_loader = common_helpers.build_task_dataloader(
            val_dataset, batch_size=per_device_batch,
            num_workers=num_workers_resolved, train=False,
        )
        if len(train_dataset) == 0:
            raise ValueError("The training dataset is empty.")

        # 6. transforms (resolved after the loader length is known).
        # DETR gets the family recipe: patch-aligned scale jitter and
        # mosaic on (overrides still win).
        if is_detr:
            from autovision.tasks.detection.detr.transforms import (
                build_detr_train_transform_args,
            )

            train_transform_args = build_detr_train_transform_args(
                default_image_size, resolved_model_args.backbone, transform_args
            )
        else:
            train_transform_args = pydantic_model_validate(
                RTDETRDetectionTransformArgs,
                {"image_size": default_image_size, **(transform_args or {})},
            )
        train_transform_args.resolve_auto_fields(
            num_channels=3, default_image_size=default_image_size
        )
        if ssl_two_crop:
            # Contrastive views must stay pairable: independent RandomIoUCrop
            # / RandomZoomOut draws displace VisDrone's tiny boxes by far
            # more than the cross-view IoU threshold, wiping the matched
            # pool (measured paired_ratio 0 vs 0.99 with them disabled).
            # The view difference the InfoNCE objective needs comes from
            # the box-aware ERASURE (doc 16: "InfoNCE is forced to rely on
            # object evidence, not background pairing") plus photometric /
            # flip — flips are recovered by flip_invariant_pairing.
            train_transform_args.random_iou_crop_prob = None
            train_transform_args.random_zoom_out_prob = None
        train_transform_args.resolve_step_schedule(
            total_steps=steps_resolved,
            train_num_batches=len(train_loader),
            gradient_accumulation_steps=grad_accum_resolved,
        )
        train_transform_args.resolve_incompatible()
        val_transform_args = build_val_transform_args(train_transform_args)

        train_transform = RTDETRDetectionTransform(train_transform_args)
        val_transform = RTDETRDetectionTransform(val_transform_args)
        train_dataset.set_transform(train_transform)
        val_dataset.set_transform(val_transform)
        train_collate = RTDETRDetectionCollateFunction(train_transform_args, train=True)
        if ssl_two_crop:
            train_collate = TwoViewDetectionCollateFunction(train_collate)
        val_collate = RTDETRDetectionCollateFunction(val_transform_args, train=False)
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
            task_type="object_detection", model_name=model,
            batch_size=global_batch_size, devices=num_devices,
            steps=steps_resolved,
        )
        engine = TaskEngine()
        hyperparams = {
            "task": "object_detection",
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
            gradient_accumulation_steps=grad_accum_resolved,
            step_hooks=[train_transform, train_collate],
            val_every_num_steps=trainer_args_resolved.get("val_every_num_steps"),
            early_stopping_patience=trainer_args_resolved.get(
                "early_stopping_patience"
            ),
            early_stopping_min_delta=float(
                trainer_args_resolved.get("early_stopping_min_delta", 0.0)
            ),
            gradient_checkpointing=common_helpers.resolve_gradient_checkpointing(
                trainer_args_resolved
            ),
            parallel=common_helpers.resolve_parallel(trainer_args_resolved),
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
            log_every_n_steps=trainer_args_resolved.get(
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
