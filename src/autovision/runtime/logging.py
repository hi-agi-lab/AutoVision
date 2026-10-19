# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Logging setup: colored console output + rotating file log in the run dir."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path

from autovision.runtime.distributed import is_global_rank_zero
from autovision.runtime.env import Env

__all__ = ["set_up_console_logging", "set_up_file_logging"]

_FORMAT_FILE = "[%(asctime)s][%(levelname)s] %(message)s"


def set_up_console_logging(names: Sequence[str] = ("autovision",)) -> None:
    """Attach a bare-format stream handler to the given loggers.

    Defaults to ``autovision`` only: mirroring the torch/py.warnings trio
    here would flip those loggers to DEBUG and flood the console with
    framework chatter. Additive and idempotent — existing handlers
    (including a previously attached RotatingFileHandler) are left alone,
    so call order vs :func:`set_up_file_logging` does not matter.
    """
    level = getattr(logging, Env.LOG_LEVEL.value.upper(), logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not any(
            isinstance(h, logging.StreamHandler)
            and type(h) is logging.StreamHandler
            for h in logger.handlers
        ):
            logger.addHandler(console)


def set_up_file_logging(log_file: Path) -> None:
    """Attach a rotating file handler (rank zero only).

    Also raises the ``autovision`` logger to DEBUG: left unconfigured it
    inherits the root logger's WARNING level, which would silently drop
    every INFO record before it reaches the handler (an empty train.log).

    Only one framework file handler is kept per process: a second call
    (HPO runs several trials in one process; tests run many fits) closes
    and detaches the previous handler before attaching the new one, so
    earlier runs' ``train.log`` files do not receive later runs' records.
    """
    if not is_global_rank_zero():
        return
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("autovision")
    for existing in list(logger.handlers):
        if getattr(existing, "_autovision_file_handler", False):
            logger.removeHandler(existing)
            existing.close()
    handler = RotatingFileHandler(
        log_file, maxBytes=1 << 30, backupCount=4, mode="a"
    )
    handler.setFormatter(logging.Formatter(_FORMAT_FILE))
    handler._autovision_file_handler = True  # type: ignore[attr-defined]
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
