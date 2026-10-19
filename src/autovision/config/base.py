# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Base classes and helpers for all Pydantic-based configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["PydanticConfig"]


def _has_auto(value: Any) -> bool:
    """Recursively check a value tree for the ``"auto"`` sentinel."""
    if isinstance(value, str):
        return value == "auto"
    if isinstance(value, dict):
        return any(_has_auto(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_auto(v) for v in value)
    if isinstance(value, PydanticConfig):
        return value.has_auto()
    return False


class PydanticConfig(BaseModel):
    """Base class for all configuration models.

    Policy (fixed, do not loosen per-class without an ADR):

    - ``extra="forbid"``: unknown keys are errors — the basis of friendly
      messages (``Unknown key: 'batch_szie'`` + did-you-mean).
    - ``frozen=False`` + ``validate_assignment=True``: configs are mutated
      in place by ``resolve_auto_fields()`` and every assignment is re-validated.
    - ``strict=True``: no silent coercions (CLI dialects may relax per field).
    - ``validate_default=True``: defaults are validated like user input.
    - ``protected_namespaces``: allows user fields named ``model`` and
      ``model_args`` despite pydantic's ``model_`` namespace.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        strict=True,
        validate_assignment=True,
        validate_default=True,
        protected_namespaces=tuple(
            name for name in dir(BaseModel) if name.startswith("model_")
        ),
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(**kwargs)

    def has_auto(self) -> bool:
        """True if any field value (recursively) is still ``"auto"``."""
        return any(
            _has_auto(getattr(self, name)) for name in type(self).model_fields
        )
