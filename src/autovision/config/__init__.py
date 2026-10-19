# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Configuration subsystem: base class, validation gateway, registries."""

from autovision.config.base import PydanticConfig
from autovision.config.registry import Registry
from autovision.config.validate import (
    assert_config_resolved,
    no_auto,
    prettify_validation_errors,
    pydantic_model_validate,
    suggest_key,
)

__all__ = [
    "PydanticConfig",
    "Registry",
    "assert_config_resolved",
    "no_auto",
    "prettify_validation_errors",
    "pydantic_model_validate",
    "suggest_key",
]
