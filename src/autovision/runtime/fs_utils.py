# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hygiene for temp files orphaned by crashed atomic writers.

Every durable artifact (checkpoint, ONNX graph, registry/run manifest) is
written to a uniquely named temp file and moved into place with
``os.replace``. A process killed between creating the temp file and the
rename leaves an orphan behind; the next writer to the same target sweeps
it, so residue from a crashed run cannot accumulate forever.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

__all__ = ["DEFAULT_MIN_AGE_S", "sweep_stale_temp_files"]

logger = logging.getLogger("autovision.runtime.fs")

#: An in-flight temp write older than this is considered abandoned. The
#: default is deliberately conservative: it can never race a healthy
#: (even very slow) writer, and deleting an abandoned temp file is safe —
#: the writer's later ``os.replace`` would fail loudly without touching
#: the previous artifact.
DEFAULT_MIN_AGE_S = 3600.0


def sweep_stale_temp_files(
    directory: Path | str,
    pattern: str,
    *,
    min_age_s: float = DEFAULT_MIN_AGE_S,
) -> int:
    """Delete orphaned temp files matching ``pattern`` (best effort).

    Only files whose mtime is older than ``min_age_s`` are removed, so a
    concurrent writer's temp file is never touched. Sweeping is hygiene,
    not correctness: it never raises and never blocks the caller. Pass a
    pattern as specific as possible — it is a glob, not an exact name.

    Returns the number of files removed.
    """
    directory = Path(directory)
    now = time.time()
    removed = 0
    try:
        candidates = list(directory.glob(pattern))
    except OSError:  # pragma: no cover - unreadable directory
        return 0
    for candidate in candidates:
        try:
            if now - candidate.stat().st_mtime < min_age_s:
                continue
            candidate.unlink()
            removed += 1
        except OSError:
            # Raced deletion, permissions, or a directory matching the
            # glob: leave it alone and keep sweeping the rest.
            continue
    if removed:
        logger.info(
            "Swept %d stale temp file(s) matching '%s' in '%s'.",
            removed, pattern, directory,
        )
    return removed
