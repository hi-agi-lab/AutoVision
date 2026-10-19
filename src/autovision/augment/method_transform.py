# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""PretextRecipe transform protocol: maps one decoded image to a list of views."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from autovision.augment.transform_args import MethodTransformArgs
from autovision.augment.view_transform import ViewTransform

__all__ = ["MethodTransform", "MultiViewTransform"]


@runtime_checkable
class MethodTransform(Protocol):
    """Callable that turns ``{"image": tensor}`` into a list of view dicts."""

    transform_args: MethodTransformArgs

    def __call__(self, input: dict[str, Any]) -> list[dict[str, Any]]: ...

    @staticmethod
    def transform_args_cls() -> type[MethodTransformArgs]: ...


class MultiViewTransform:
    """Applies one ``ViewTransform`` per entry and returns one view dict each."""

    def __init__(self, transforms: list[ViewTransform], transform_args: MethodTransformArgs) -> None:
        self.transforms = transforms
        self.transform_args = transform_args

    def __call__(self, input: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"image": transform(input["image"])} for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[MethodTransformArgs]:
        raise NotImplementedError("MultiViewTransform is not bound to a method.")
