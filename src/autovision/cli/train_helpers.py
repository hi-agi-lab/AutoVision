# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining command helpers: ordered config resolution and assembly.

Resolution order (design doc §5.3, mirrors the baseline):

    transform_args -> epochs -> strategy/precision -> batch_size/num_workers
    -> optimizer_args(wrapped_model) -> method_args(scaling, optimizer_args,
    wrapped_model)

Each resolver validates its input against the method-provided config class,
resolves ``"auto"`` in place, and asserts the result is fully resolved.
"""

from __future__ import annotations

from typing import Any, TypeVar

from torch.utils.data import DataLoader

from autovision.augment.method_transform import MethodTransform
from autovision.augment.transform_args import MethodTransformArgs
from autovision.config.validate import (
    assert_config_resolved,
    pydantic_model_validate,
)
from autovision.errors import UnresolvedAutoError
from autovision.pretext.recipe import PretextRecipe
from autovision.scaling import (
    ScalingInfo,
    epochs_from_steps,
)

__all__ = [
    "get_dataloader",
    "get_epochs",
    "get_method",
    "get_method_args",
    "get_optimizer_args",
    "get_transform",
    "get_transform_args",
]

T = TypeVar("T")


def get_transform_args(
    method_cls: type[PretextRecipe], transform_args: dict | MethodTransformArgs | None
) -> MethodTransformArgs:
    args_cls = method_cls.transform_cls().transform_args_cls()
    args = pydantic_model_validate(args_cls, dict(transform_args or {}))
    args.resolve_auto_fields()
    args.resolve_incompatible()
    assert_config_resolved(args)
    return args


def get_transform(
    method_cls: type[PretextRecipe], transform_args: MethodTransformArgs
) -> MethodTransform:
    assert_config_resolved(transform_args)
    return method_cls.transform_cls()(transform_args)


def get_dataset_size(dataset) -> int:
    try:
        return len(dataset)
    except TypeError as error:
        raise UnresolvedAutoError(
            "Dataset size is required to resolve 'epochs=auto'."
        ) from error


def get_epochs(
    method_cls: type[PretextRecipe],
    epochs: int | str,
    dataset_size: int,
    batch_size: int,
) -> int:
    """Resolve ``epochs``: user value > step-based recipe > default epochs."""
    if epochs != "auto":
        return int(epochs)
    args_cls = method_cls.method_args_cls()
    recipe_steps = getattr(args_cls, "auto_steps", None)
    if recipe_steps is not None:
        return epochs_from_steps(recipe_steps, batch_size, dataset_size)
    recipe_epochs = getattr(args_cls, "auto_epochs", None)
    if recipe_epochs is None:
        raise ValueError(
            f"{args_cls.__name__} defines neither 'auto_steps' nor "
            f"'auto_epochs'; set epochs explicitly."
        )
    return int(recipe_epochs)


def get_scaling_info(dataset_size: int, epochs: int) -> ScalingInfo:
    return ScalingInfo(dataset_size=dataset_size, epochs=epochs)


def get_optimizer_args(
    method_cls: type[PretextRecipe],
    optim: str,
    optim_args: dict | Any | None,
    wrapped_model: Any,
) -> Any:
    args_cls = method_cls.optimizer_args_cls(optim)
    args = pydantic_model_validate(args_cls, dict(optim_args or {}))
    args.resolve_auto_fields(wrapped_model=wrapped_model)
    assert_config_resolved(args)
    return args


def get_method_args(
    method_cls: type[PretextRecipe],
    method_args: dict | Any | None,
    scaling: ScalingInfo,
    optimizer_args: Any,
    wrapped_model: Any,
) -> Any:
    args_cls = method_cls.method_args_cls()
    args = pydantic_model_validate(args_cls, dict(method_args or {}))
    args.resolve_auto_fields(
        scaling=scaling,
        optimizer_args=optimizer_args,
        wrapped_model=wrapped_model,
    )
    assert_config_resolved(args)
    return args


def get_method(
    method_cls: type[PretextRecipe],
    method_args: Any,
    optimizer_args: Any,
    embedding_model: Any,
    global_batch_size: int,
    num_input_channels: int,
) -> PretextRecipe:
    return method_cls(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=embedding_model,
        global_batch_size=global_batch_size,
        num_input_channels=num_input_channels,
    )


def get_dataloader(
    dataset,
    batch_size_per_device: int,
    num_workers: int,
    collate_fn: Any = None,
) -> DataLoader:
    timeout = (
        _dataloader_timeout() if num_workers > 0 else 0.0
    )
    return DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        timeout=timeout,
    )


def _dataloader_timeout() -> float:
    from autovision.runtime.env import Env

    return Env.DATALOADER_TIMEOUT_SEC.value
