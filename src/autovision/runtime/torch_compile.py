# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Optional torch.compile integration.

Only ``forward``/``forward_backend`` should be compiled — compiling the
training step breaks metrics and batch-metadata handling (baseline
finding). Compilation is disabled by default; enable per run via
``torch_compile_args={"disable": False, ...}``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch

from autovision.config.base import PydanticConfig

__all__ = ["TorchCompileArgs", "try_compile", "disable_compile"]

logger = logging.getLogger("autovision")


class TorchCompileArgs(PydanticConfig):
    model_config = {"extra": "allow"}  # extra kwargs go to torch.compile

    disable: bool = True


def try_compile(
    fn: Callable, name: str, args: TorchCompileArgs | None = None
) -> Callable:
    """Compile ``fn`` when enabled; fall back to the original on any error."""
    if args is None or args.disable:
        return fn
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable; running eager.")
        return fn
    kwargs: dict[str, Any] = {
        key: value for key, value in args.model_dump().items() if key != "disable"
    }
    start = time.perf_counter()
    try:
        compiled = torch.compile(fn, **kwargs)
        logger.info("Compiled '%s' in %.1fs.", name, time.perf_counter() - start)
        return compiled
    except Exception as error:
        logger.warning("torch.compile failed for '%s' (%s); running eager.", name, error)
        return fn


def disable_compile(fn: Callable, recursive: bool = True) -> Callable:
    """Mark a callable to run eagerly inside compiled regions."""
    try:
        import torch._dynamo

        return torch._dynamo.disable(fn, recursive=recursive)
    except Exception:  # pragma: no cover - older torch fallback
        return fn
