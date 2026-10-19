# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model layer: wrapper protocols, package registry, embedding model."""

from autovision.encoders import (
    dinov2,  # noqa: F401  (registers the package)
    dinov3,  # noqa: F401  (registers the package)
    rfdetr,  # noqa: F401  (registers the package)
    super_gradients,  # noqa: F401  (registers the package)
    timm,  # noqa: F401  (registers the package)
    torchvision,  # noqa: F401  (registers the package)
    ultralytics,  # noqa: F401  (registers the package)
)
from autovision.encoders.embedding_model import FeatureExtractor
from autovision.encoders.model_wrapper import (
    ArchTraits,
    ArchTraitsSource,
    EncoderProtocol,
    missing_encoder_members,
)
from autovision.encoders.package_helpers import (
    catalog_names,
    get_source,
    get_wrapped_model,
    parse_model_name,
    register_source,
    source_for_model,
)

__all__ = [
    "ArchTraits",
    "ArchTraitsSource",
    "FeatureExtractor",
    "EncoderProtocol",
    "get_source",
    "source_for_model",
    "get_wrapped_model",
    "catalog_names",
    "missing_encoder_members",
    "parse_model_name",
    "register_source",
]

def list_models() -> list[str]:
    """Public alias for :func:`catalog_names`."""
    return catalog_names()
