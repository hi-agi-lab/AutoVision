# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CatalogModelSource registry: name parsing, dispatch, and model instantiation.

Dispatch order matters: packages are checked in registration order and the
custom package (accepting raw ``EncoderProtocol`` instances) must always be
last — custom is the last resort.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from autovision.encoders.model_wrapper import (
    EncoderProtocol,
    missing_encoder_members,
)
from autovision.encoders.package import CatalogModelSource, ModelSource
from autovision.errors import UnknownModelError

__all__ = [
    "CUSTOM_SOURCE",
    "ModelSource",
    "CatalogModelSource",
    "custom_source",
    "get_source",
    "source_for_model",
    "get_wrapped_model",
    "list_sources",
    "catalog_names",
    "list_catalog_sources",
    "parse_model_name",
    "register_source",
]

_SOURCES: list[ModelSource] = []
_CUSTOM_SOURCES: list[ModelSource] = []


def register_source(package: ModelSource) -> None:
    """Register a model package (built-ins at import time, plugins at runtime)."""
    if any(p.name == package.name for p in _SOURCES):
        raise ValueError(f"A source named '{package.name}' is already registered.")
    _SOURCES.append(package)


def list_sources() -> list[ModelSource]:
    """All registered packages, custom always last."""
    return _SOURCES + _CUSTOM_SOURCES


def list_catalog_sources() -> list[CatalogModelSource]:
    """Packages with listable models (excludes custom)."""
    return [p for p in list_sources() if isinstance(p, CatalogModelSource)]


def get_source(package_name: str) -> ModelSource:
    for package in list_sources():
        if package.name == package_name:
            return package
    supported = ", ".join(p.name for p in list_sources())
    raise ValueError(
        f"Unknown package: '{package_name}'. Supported packages: {supported}"
    )


def catalog_names() -> list[str]:
    """Union of all package model names, sorted."""
    names: list[str] = []
    for package in list_catalog_sources():
        names.extend(package.catalog_names())
    return sorted(names)


def parse_model_name(model: str) -> tuple[str, str]:
    """Split ``"package/model_name"`` (first slash only)."""
    if "/" not in model:
        raise ValueError(
            f"The model name '{model}' must have the format '<package>/<model>'."
        )
    package_name, model_name = model.split("/", maxsplit=1)
    return package_name, model_name


def get_wrapped_model(
    model: str | EncoderProtocol | nn.Module,
    num_input_channels: int = 3,
    model_args: dict[str, Any] | None = None,
    load_weights: bool = True,
) -> EncoderProtocol:
    """Resolve any model specification to a ``EncoderProtocol``.

    - ``EncoderProtocol`` instance: returned as-is.
    - ``"package/model"`` string: built by the named package.
    - other ``nn.Module``: attributed to a package via ``recognizes``.
    """
    # Duck-typed check first: any object implementing the full protocol is
    # returned unchanged (structural subtyping, no base class required).
    if not isinstance(model, str) and not missing_encoder_members(model):
        return model  # type: ignore[return-value]
    if isinstance(model, str):
        package_name, model_name = parse_model_name(model)
        package = get_source(package_name)
        if not isinstance(package, CatalogModelSource):
            raise ValueError(f"Source '{package_name}' does not support named models.")
        raw_model = package.build(
            model_name,
            num_input_channels=num_input_channels,
            model_args=model_args,
            load_weights=load_weights,
        )
        return package.as_encoder(raw_model)
    if isinstance(model, nn.Module):
        package = source_for_model(
            model, include_custom=False, fallback_custom=False
        )
        if not isinstance(package, CatalogModelSource):
            raise UnknownModelError(
                f"Cannot attribute model of type '{type(model).__name__}' to any package."
            )
        return package.as_encoder(model)
    raise UnknownModelError(
        f"Cannot resolve model specification of type '{type(model).__name__}'."
    )


def source_for_model(
    model: Any,
    include_custom: bool = True,
    fallback_custom: bool = True,
) -> ModelSource | None:
    """Find the package a built model belongs to via ``recognizes``."""
    packages = list_sources() if include_custom else _SOURCES
    for package in packages:
        if package.recognizes(model):
            return package
    if fallback_custom:
        return custom_source()
    raise UnknownModelError(
        f"Unknown model of type '{type(model).__name__}'. If it is a custom "
        f"model, make it implement the EncoderProtocol protocol. Missing "
        f"attributes: {missing_encoder_members(model, exclude_module_attrs=True)}"
    )


# Custom is provided by its own module and must always be registered LAST
# (named packages are preferred; custom is the last resort).
from autovision.encoders.custom.custom_source import CustomSource  # noqa: E402

_CUSTOM_SOURCES.append(CustomSource())
CUSTOM_SOURCE = _CUSTOM_SOURCES[0]
custom_source = lambda: CUSTOM_SOURCE  # noqa: E731
