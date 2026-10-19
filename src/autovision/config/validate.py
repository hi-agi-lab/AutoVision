# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Validation gateway and error prettification for all configuration models.

Every config in the framework is built through :func:`pydantic_model_validate`
so error formatting is uniform. Error messages are single-line, key-path
addressed, and include a "did you mean" suggestion for unknown keys — a
deliberate enhancement over the baseline (which had no suggestions).
"""

from __future__ import annotations

import difflib
from typing import Any, Type, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from autovision.config.base import PydanticConfig, _has_auto
from autovision.errors import ConfigValidationError, UnresolvedAutoError

__all__ = [
    "assert_config_resolved",
    "no_auto",
    "prettify_validation_errors",
    "pydantic_model_validate",
    "suggest_key",
]

T = TypeVar("T")

_SUGGESTION_CUTOFF = 0.6


def suggest_key(unknown_key: str, known_keys: list[str]) -> str | None:
    """Return the closest known key for an unknown key, or ``None``."""
    matches = difflib.get_close_matches(
        unknown_key, known_keys, n=1, cutoff=_SUGGESTION_CUTOFF
    )
    return matches[0] if matches else None


def _loc_to_dot_sep(loc: tuple[Any, ...]) -> str:
    return ".".join(
        f"[{part}]" if isinstance(part, int) else str(part) for part in loc
    )


def _error_line(
    error: dict[str, Any], known_keys: list[str]
) -> str:
    loc = _loc_to_dot_sep(error["loc"])
    error_type: str = error["type"]
    input_value = error.get("input")
    msg: str = error.get("msg", "")
    if error_type == "missing":
        return f"Missing key: '{loc}'"
    if error_type == "extra_forbidden":
        line = f"Unknown key: '{loc}'"
        suggestion = suggest_key(str(error["loc"][-1]), known_keys)
        if suggestion is not None:
            line += f" Did you mean '{suggestion}'?"
        return line
    if error_type.endswith("_type") or error_type in ("literal_error", "is_instance_of"):
        return (
            f"Invalid type for key '{loc}': {msg} but got "
            f"{input_value!r} with type '{type(input_value).__name__}'"
        )
    return f"Error for key '{loc}': {msg} (error_type={error_type}, input={input_value!r})"


def prettify_validation_errors(
    error: ValidationError, model: Type[BaseModel] | None = None
) -> str:
    """Format a pydantic ValidationError as short, human-readable lines."""
    known_keys = list(getattr(model, "model_fields", {}).keys()) if model else []
    lines = [_error_line(err, known_keys) for err in error.errors()]
    return f"Found {len(lines)} errors in the config!\n" + "\n".join(
        f"  {line}" for line in lines
    )


def pydantic_model_validate(
    model: Type[PydanticConfig] | TypeAdapter[T],
    obj: dict[str, Any],
) -> PydanticConfig | T:
    """The single entry point for building any config from a plain dict."""
    try:
        if isinstance(model, TypeAdapter):
            return model.validate_python(obj)  # type: ignore[return-value]
        return model.model_validate(obj)
    except ValidationError as ex:
        raise ConfigValidationError(
            prettify_validation_errors(ex, model if not isinstance(model, TypeAdapter) else None)
        ) from None


def no_auto(value: Any) -> Any:
    """Return ``value`` or raise if it is still the ``"auto"`` sentinel.

    Reading an unresolved auto is a framework bug (the resolution chain must
    have covered it), never a user error — the message says so explicitly.
    """
    if _has_auto(value):
        raise UnresolvedAutoError(
            "Got unresolved 'auto' value. This is not a user error, "
            "please report this issue to the AutoVision team."
        )
    return value


def assert_config_resolved(config: Any) -> None:
    """Raise if the config (or plain dict/list tree) still contains ``"auto"``."""
    if _has_auto(config):
        raise UnresolvedAutoError(
            "Config contains unresolved 'auto' values. This is not a user error, "
            "please report this issue to the AutoVision team."
        )
