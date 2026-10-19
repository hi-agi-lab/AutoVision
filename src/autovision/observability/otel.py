# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Optional OpenTelemetry spans (design §15.2).

Four-phase instrumentation following the baseline's TrainingStepTimer
phases: dataload / step / checkpoint / validation. ``opentelemetry-api`` is
a soft dependency — without it (or with ``AUTOVISION_OTEL`` disabled) every
call here is a zero-cost no-op, so default training behavior is unchanged.

The SDK and exporter are **not** managed by autovision: embedding
applications register their own ``trace.set_tracer_provider(...)`` and then
set ``AUTOVISION_OTEL=1`` to turn span emission on. With the API installed
but no provider configured, spans land on the API's built-in no-op tracer.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

from autovision.runtime.env import Env

__all__ = ["end_span", "get_tracer", "is_otel_enabled", "otel_span", "start_span"]

logger = logging.getLogger("autovision")

TRACER_NAME = "autovision"

_otel_disabled_logged = False


def is_otel_enabled() -> bool:
    """``AUTOVISION_OTEL`` gate — declared in the Env registry like every
    other framework variable (fault-tolerant boolean parsing there)."""
    return Env.OTEL.value


def _get_opentelemetry_trace() -> Any | None:
    """Import ``opentelemetry.trace`` (API only); warn once if missing."""
    global _otel_disabled_logged
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        if not _otel_disabled_logged:
            logger.warning(
                "AUTOVISION_OTEL is set but opentelemetry is not installed; "
                "spans are disabled. Install with `pip install opentelemetry-api`."
            )
            _otel_disabled_logged = True
        return None
    return otel_trace


def get_tracer() -> Any | None:
    """The autovision tracer, or ``None`` when OTel is off/absent."""
    if not is_otel_enabled():
        return None
    otel_trace = _get_opentelemetry_trace()
    if otel_trace is None:
        return None
    return otel_trace.get_tracer(TRACER_NAME)


def otel_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Any:
    """Context-manager span; a no-op nullcontext when OTel is disabled."""
    tracer = get_tracer()
    if tracer is None:
        return nullcontext()
    return tracer.start_as_current_span(name, attributes=attributes)


def start_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Any | None:
    """Explicit-start span for long scopes (e.g. a whole ``fit`` run) where
    re-indenting the body is not worth the diff. Pair with :func:`end_span`;
    an abandoned span (exception path) simply never closes."""
    tracer = get_tracer()
    if tracer is None:
        return None
    return tracer.start_span(name, attributes=attributes)


def end_span(span: Any | None) -> None:
    """End a span from :func:`start_span`; ``None``-safe."""
    if span is not None:
        span.end()
