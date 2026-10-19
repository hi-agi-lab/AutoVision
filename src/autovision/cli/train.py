# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""The ``pretrain`` command: public API and the ordered assembly flow."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BeforeValidator, ConfigDict

from autovision._license import LICENSE_INFO
from autovision.cli import common_helpers, train_helpers
from autovision.cli.system_info import get_system_information
from autovision.config.base import PydanticConfig
from autovision.config.validate import no_auto, pydantic_model_validate
from autovision.data import ImageDataset
from autovision.data.file_helpers import list_image_filenames_from_dir
from autovision.data.mmap_helpers import dataset_mmap_context
from autovision.encoders import FeatureExtractor, get_wrapped_model
from autovision.engine import Engine, seed_everything
from autovision.pretext.teachers import get_method_cls
from autovision.runtime.accelerator import accelerator_supports_bf16
from autovision.runtime.checkpoint import CheckpointMetadata
from autovision.runtime.distributed import is_global_rank_zero
from autovision.runtime.logging import set_up_file_logging
from autovision.types import PathLike

__all__ = ["PretrainConfig", "PretrainResult", "pretrain", "train_from_config"]

logger = logging.getLogger("autovision")


def _is_rank_zero() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) <= 1 or is_global_rank_zero()


def _coerce_path(value: Any) -> Any:
    """Coerce strings to Paths before strict validation (CLI/YAML inputs)."""
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, (list, tuple)):
        return [Path(v) if isinstance(v, str) else v for v in value]
    return value


class PretrainConfig(PydanticConfig):
    """Validated configuration for :func:`pretrain`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    out: Annotated[Path, BeforeValidator(_coerce_path)]
    data: Annotated[Path | list[Path], BeforeValidator(_coerce_path)]
    mask_dir: Annotated[Path | None, BeforeValidator(_coerce_path)] = None
    model: str | Any
    method: str = "distillation"
    method_args: dict[str, Any] | None = None
    model_args: dict[str, Any] | None = None
    embed_dim: int | None = None
    epochs: int | str = "auto"
    batch_size: int = 128
    num_workers: int | str = "auto"
    devices: int | str = "auto"
    precision: Literal["auto", "32-true", "bf16-mixed", "16-mixed"] = "auto"
    seed: int = 0
    checkpoint: Annotated[Path | None, BeforeValidator(_coerce_path)] = None
    resume_interrupted: bool = False
    overwrite: bool = False
    optim: str = "auto"
    optim_args: dict[str, Any] | None = None
    transform_args: dict[str, Any] | None = None
    trainer_args: dict[str, Any] | None = None
    loggers: dict[str, Any] | None = None


@dataclass
class PretrainResult:
    """Result object returned by :func:`pretrain` for programmatic use."""

    out_dir: Path
    config: PretrainConfig
    epochs_run: int
    global_step: int
    checkpoint_path: Path
    exported_model_path: Path
    metrics: dict[str, float] | None = None


def pretrain(
    out: PathLike,
    data: PathLike | list[PathLike],
    model: str | Any,
    *,
    mask_dir: PathLike | None = None,
    method: str = "distillation",
    method_args: dict[str, Any] | None = None,
    model_args: dict[str, Any] | None = None,
    embed_dim: int | None = None,
    epochs: int | str = "auto",
    batch_size: int = 128,
    num_workers: int | str = "auto",
    devices: int | str = "auto",
    precision: str = "auto",
    seed: int = 0,
    checkpoint: PathLike | None = None,
    resume_interrupted: bool = False,
    overwrite: bool = False,
    optim: str = "auto",
    optim_args: dict[str, Any] | None = None,
    transform_args: dict[str, Any] | None = None,
    trainer_args: dict[str, Any] | None = None,
    loggers: dict[str, dict[str, Any] | None] | None = None,
) -> PretrainResult:
    """Self-supervised pretraining / distillation.

    ``mask_dir`` points at a directory of region-id masks (one ``.png`` per
    image, same filename stem); it enables region-contrastive methods such
    as DetCon. See the design doc (§4.2) for the full parameter contract.
    Returns a :class:`PretrainResult` with all artifact paths.
    """
    config = pydantic_model_validate(
        PretrainConfig,
        {
            "out": out,
            "data": data,
            "mask_dir": mask_dir,
            "model": model,
            "method": method,
            "method_args": method_args,
            "model_args": model_args,
            "embed_dim": embed_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "precision": precision,
            "seed": seed,
            "checkpoint": checkpoint,
            "resume_interrupted": resume_interrupted,
            "overwrite": overwrite,
            "optim": optim,
            "optim_args": optim_args,
            "transform_args": transform_args,
            "trainer_args": trainer_args,
            "loggers": loggers,
        },
    )
    return train_from_config(config)


def train_from_config(config: PretrainConfig) -> PretrainResult:
    """Ordered assembly flow (design doc §5.3)."""
    # The seed must be applied before the assembly builds random modules
    # (model init, transform parameters): the engine re-seeds inside fit(),
    # which is already too late for construction.
    seed_everything(config.seed)
    method_cls = get_method_cls(config.method)

    # 1. out dir + logging
    out_dir = common_helpers.get_out_dir(
        config.out, config.resume_interrupted, config.overwrite
    )
    set_up_file_logging(out_dir / "train.log")
    system = get_system_information()
    logger.info("AutoVision %s | torch %s | device %s", system.autovision_version, system.torch_version, system.gpus or "cpu")

    # 2. transform args (resolved first: num_channels feeds the model build)
    transform_args = train_helpers.get_transform_args(
        method_cls, config.transform_args
    )
    transform = train_helpers.get_transform(method_cls, transform_args)
    num_channels = no_auto(transform_args.num_channels)

    # 3. model
    wrapped_model = get_wrapped_model(
        config.model,
        num_input_channels=num_channels,
        model_args=config.model_args,
    )

    # 4. dataset (mmap-managed for the whole run)
    data_paths = common_helpers.resolve_data_paths(config.data)
    def _items():
        if len(data_paths) == 1 and data_paths[0].is_dir():
            return ({"filenames": str(p)} for p in list_image_filenames_from_dir(data_paths[0]))
        return ({"filenames": str(p)} for p in data_paths)

    with dataset_mmap_context(
        _items,
        data=str(data_paths[0]),
        out_dir=out_dir,
        resume_interrupted=config.resume_interrupted,
        overwrite=config.overwrite,
    ) as image_filenames:
        dataset = ImageDataset(
            image_dir=None,
            image_filenames=image_filenames,
            transform=transform,
            num_channels=num_channels,
            mask_dir=config.mask_dir,
        )
        dataset_size = train_helpers.get_dataset_size(dataset)
        if dataset_size <= 0:
            raise ValueError(
                f"The dataset at '{data_paths}' is empty; nothing to train on."
            )

        # 5. batch size (clamped first: epochs=auto must see the final value)
        num_devices = common_helpers.resolve_num_devices(config.devices)
        global_batch_size = common_helpers.get_global_batch_size(
            config.batch_size, dataset_size, num_devices
        )

        # 6. epochs + scaling info
        epochs = train_helpers.get_epochs(
            method_cls, config.epochs, dataset_size, global_batch_size
        )
        scaling = train_helpers.get_scaling_info(dataset_size, epochs)

        # 7. optimizer + method args (order matters: optim before method)
        optimizer_args = train_helpers.get_optimizer_args(
            method_cls, config.optim, config.optim_args, wrapped_model
        )
        method_args = train_helpers.get_method_args(
            method_cls, config.method_args, scaling, optimizer_args, wrapped_model
        )

        # 8. workers + dataloader
        num_workers = common_helpers.get_num_workers(
            config.num_workers, num_devices
        )
        dataloader = train_helpers.get_dataloader(
            dataset, batch_size_per_device=global_batch_size // num_devices,
            num_workers=num_workers,
        )

        # 9. method
        embedding_model = FeatureExtractor(
            wrapped_model=wrapped_model, embed_dim=config.embed_dim
        )
        method = train_helpers.get_method(
            method_cls=method_cls,
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
            num_input_channels=num_channels,
        )

        # 9.5 kNN probe (§7.5): attached when method_args.knn_eval resolves on.
        from autovision.pretext.knn_eval import attach_knn_eval_runner

        attach_knn_eval_runner(
            method=method,
            method_args=method_args,
            filenames=image_filenames,
            transform_args=transform_args,
        )

        # 10. loggers + telemetry
        from autovision.observability.events import track_training_started
        from autovision.observability.loggers import (
            finalize_loggers as _finalize_loggers,
        )
        from autovision.observability.loggers import (
            get_loggers as _get_loggers,
        )
        from autovision.observability.loggers import (
            log_hyperparams_to_loggers as _log_hyperparams,
        )

        logger_instances = _get_loggers(config.loggers, out_dir)
        track_training_started(
            task_type="ssl_pretraining", model_name=config.model,
            method=config.method, batch_size=global_batch_size,
            devices=num_devices, epochs=epochs,
        )

        # 11. fit (precision "auto" -> bf16-mixed on CUDA, fp32 elsewhere)
        precision = config.precision
        if precision == "auto":
            precision = (
                "bf16-mixed"
                if accelerator_supports_bf16()
                else "32-true"
            )
        if _is_rank_zero():
            try:
                from autovision.cli.plot import save_example_augmentations

                example_batch = next(iter(dataloader))
                save_example_augmentations(
                    example_batch, out_dir / "augmentation_examples.jpg"
                )
            except (StopIteration, RuntimeError, OSError):
                logger.debug("Augmentation example plot skipped.", exc_info=True)
        engine = Engine()
        if _is_rank_zero():
            _log_hyperparams(
                logger_instances,
                {"method": config.method, "model": str(config.model), "epochs": epochs,
                 "batch_size": global_batch_size, "dataset_size": dataset_size},
            )
        fit_result = engine.fit(
            method,
            dataloader,
            out_dir=out_dir,
            max_epochs=epochs,
            devices=config.devices,
            precision=precision,
            seed=config.seed,
            checkpoint=Path(config.checkpoint) if config.checkpoint else None,
            resume_interrupted=config.resume_interrupted,
            logger_fn=lambda metrics, step: [
                lg.log_metrics(metrics, step) for lg in logger_instances
            ] if _is_rank_zero() else None,
            log_every_n_steps=(config.trainer_args or {}).get(
                "log_every_n_steps",
                min(50, max(1, len(dataloader))),
            ),
            normalize_args=transform_args.normalize.to_dict(),
        )

    _finalize_loggers(logger_instances)

    # 12. export (rank zero)
    exported_path = None
    if _is_rank_zero():
        exported_dir = out_dir / "exported_models"
        exported_dir.mkdir(parents=True, exist_ok=True)
        exported_path = exported_dir / "exported_last.pt"
        torch.save(
            {
                "schema_version": 1,
                "kind": "embedding_model",
                "model_name": config.model if isinstance(config.model, str) else None,
                "embed_dim": embedding_model.embed_dim,
                "state_dict": embedding_model.state_dict(),
                "normalize_args": transform_args.normalize.to_dict(),
                "version": CheckpointMetadata().version,
                "license_info": LICENSE_INFO,
            },
            exported_path,
        )
        fit_result.exported_model_path = exported_path
        logger.info("Exported model to '%s'.", exported_path)

    return PretrainResult(
        out_dir=out_dir,
        config=config,
        epochs_run=fit_result.epochs_run,
        global_step=fit_result.global_step,
        checkpoint_path=fit_result.checkpoint_path,
        exported_model_path=exported_path,
        metrics=dict(fit_result.metrics),
    )
