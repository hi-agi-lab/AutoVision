# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Opt-in usage telemetry.

Compliance-first (design doc §14): telemetry is **disabled by default**
(the inverse of the baseline) and enables only via ``AUTOVISION_TELEMETRY=1``.
Payloads contain framework/model metadata only — never data paths or
image contents. Every send is best-effort: all transport errors are
swallowed so telemetry can never affect training.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import uuid
from collections import deque
from typing import Any

from autovision._version import __version__
from autovision.runtime.accelerator import npu_available
from autovision.runtime.distributed import is_global_rank_zero
from autovision.runtime.env import Env

__all__ = ["track_event", "track_inference_started", "track_training_started"]

logger = logging.getLogger("autovision")

_RATE_LIMIT_SECONDS = 30.0
_MAX_BUFFERED = 100
_FLUSH_TIMEOUT_SEC = 10.0

_events: deque = deque(maxlen=_MAX_BUFFERED)
_last_sent_at: dict[str, float] = {}
_lock = threading.Lock()


def _telemetry_enabled() -> bool:
    return Env.TELEMETRY.value and is_global_rank_zero()


def _anonymous_user_id() -> str:
    """Random per-installation UUID persisted in the cache dir."""
    try:
        cache_dir = Env.CACHE_DIR.value
        cache_dir.mkdir(parents=True, exist_ok=True)
        user_id_file = cache_dir / "userid.txt"
        if user_id_file.exists():
            existing = user_id_file.read_text().strip()
            if existing:
                return existing
        user_id = str(uuid.uuid4())
        tmp = user_id_file.with_suffix(".tmp")
        tmp.write_text(user_id)
        tmp.replace(user_id_file)
        return user_id
    except OSError:
        return str(uuid.uuid4())  # in-memory fallback


def _system_properties() -> dict[str, Any]:
    import platform

    import torch

    properties: dict[str, Any] = {
        "os": platform.system(),
        "version": __version__,
    }
    try:
        if torch.cuda.is_available():
            properties["gpu_name"] = torch.cuda.get_device_name(0)
        elif npu_available():
            properties["gpu_name"] = torch.npu.get_device_name(0)
    except Exception:  # pragma: no cover - best effort only
        pass
    return properties


def _sender(payload: bytes, endpoint: str) -> None:
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=_FLUSH_TIMEOUT_SEC) as response:  # nosec B310
        response.read()


def _flush_async() -> None:
    # Swap-and-clear under the lock: concurrent flushers must not duplicate
    # or wipe events enqueued by the other.
    with _lock:
        events = list(_events)
        _events.clear()
    if not events:
        return
    endpoint = Env.TELEMETRY_ENDPOINT.value

    def send() -> None:
        payload = json.dumps({"events": events}).encode("utf-8")
        try:
            _sender(payload, endpoint)
        except Exception:  # telemetry must never break training
            logger.debug("Telemetry send failed (ignored).", exc_info=True)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()


def track_event(name: str, properties: dict[str, Any] | None = None) -> None:
    """Record a usage event (no-op unless telemetry is enabled)."""
    if not _telemetry_enabled():
        return
    now = time.monotonic()
    with _lock:
        last = _last_sent_at.get(name)
        if last is not None and now - last < _RATE_LIMIT_SECONDS:
            return
        _last_sent_at[name] = now
        _events.append(
            {
                "event": name,
                "properties": {**_system_properties(), **(properties or {})},
                "distinct_id": _anonymous_user_id(),
                "timestamp": time.time(),
            }
        )
    _flush_async()


def track_training_started(
    *, task_type: str, model_name: str | None = None,
    method: str | None = None, batch_size: int | None = None,
    devices: int | None = None, steps: int | None = None,
    epochs: int | None = None,
) -> None:
    track_event(
        "training_started",
        {
            "task_type": task_type,
            "model_name": _model_descriptor(model_name),
            "method": method,
            "batch_size": batch_size,
            "devices": devices,
            "steps": steps,
            "epochs": epochs,
        },
    )


def track_inference_started(
    *, task_type: str, model_name: str | None = None, batch_size: int | None = None
) -> None:
    track_event(
        "inference_started",
        {
            "task_type": task_type,
            "model_name": _model_descriptor(model_name),
            "batch_size": batch_size,
        },
    )


def _model_descriptor(model_name: Any) -> str | None:
    if model_name is None:
        return None
    if isinstance(model_name, str):
        return model_name
    return type(model_name).__name__
