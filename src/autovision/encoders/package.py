# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model-source contract.

A *source* supplies backbone models: it recognizes the modules it built and
writes them out for checkpoints. Sources with a catalog additionally build
modules by name and adapt them to
:class:`~autovision.encoders.model_wrapper.EncoderProtocol`. Registration
lives in :mod:`autovision.encoders.package_helpers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["CatalogModelSource", "ModelSource"]


class ModelSource(ABC):
    """A provider of backbone modules (custom adapters included)."""

    name: str

    @classmethod
    @abstractmethod
    def recognizes(cls, model: Any) -> bool:
        """Whether ``model`` (a module or an adapter) was built by this source."""

    @classmethod
    @abstractmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        """Write a model built by this source to ``out``."""


class CatalogModelSource(ModelSource):
    """A source with a catalog of ``"<source>/<model>"`` names."""

    @classmethod
    @abstractmethod
    def catalog_names(cls) -> list[str]:
        """Every supported name, as ``"<source>/<model>"`` strings."""

    @classmethod
    @abstractmethod
    def build(
        cls,
        name: str,
        *,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        """Construct the raw module for ``name``."""

    @classmethod
    @abstractmethod
    def as_encoder(cls, model: Any) -> Any:
        """Adapt a module from :meth:`build` to the encoder protocol."""
