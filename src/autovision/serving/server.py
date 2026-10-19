# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""uvicorn launcher: TLS, multi-worker and single-process modes.

TLS and worker counts are transport concerns (not ASGI app concerns), so they
live here rather than in ``create_app``. Multi-worker mode requires a model
path because each worker process rebuilds the app from
``AUTOVISION_SERVE_CONFIG``; ready engine objects cannot be pickled into a
spawned worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.serving.app import create_app
from autovision.serving.worker import SERVE_CONFIG_ENV

__all__ = ["serve"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AutoVisionError(
        f"Serving option of type {type(value).__name__} is not JSON-serializable; "
        "multi-worker mode needs plain paths/scalars (use workers=1 for live objects)."
    )


def serve(
    model: Any = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    ssl_keyfile: str | Path | None = None,
    ssl_certfile: str | Path | None = None,
    ssl_ca_certs: str | Path | None = None,
    ssl_cert_reqs: int | None = None,
    timeout_graceful_shutdown: float | None = None,
    **app_kwargs: Any,
) -> None:
    """Run the reference serving app through uvicorn.

    ``workers > 1`` requires ``model`` to be a path (each worker rebuilds the
    app); ``variants=``/engine objects are single-process only. Provide both
    ``ssl_keyfile`` and ``ssl_certfile`` to enable TLS. ``timeout_graceful_shutdown``
    bounds the drain window on SIGTERM (uvicorn waits forever by default).
    """
    import uvicorn

    if timeout_graceful_shutdown is not None and timeout_graceful_shutdown <= 0:
        raise AutoVisionError(
            "timeout_graceful_shutdown must be > 0 or None, got "
            f"{timeout_graceful_shutdown}."
        )
    if workers < 1:
        raise AutoVisionError(f"workers must be >= 1, got {workers}.")
    if (ssl_keyfile is None) != (ssl_certfile is None):
        raise AutoVisionError(
            "TLS needs both ssl_keyfile= and ssl_certfile= (one without the "
            "other cannot start a server)."
        )
    if (ssl_ca_certs is not None or ssl_cert_reqs is not None) and ssl_keyfile is None:
        raise AutoVisionError(
            "ssl_ca_certs/ssl_cert_reqs require TLS (ssl_keyfile + ssl_certfile)."
        )
    tls_kwargs: dict[str, Any] = {}
    if ssl_keyfile is not None:
        tls_kwargs["ssl_keyfile"] = str(ssl_keyfile)
        tls_kwargs["ssl_certfile"] = str(ssl_certfile)
    if ssl_ca_certs is not None:
        tls_kwargs["ssl_ca_certs"] = str(ssl_ca_certs)
    if ssl_cert_reqs is not None:
        tls_kwargs["ssl_cert_reqs"] = int(ssl_cert_reqs)
    shutdown_kwargs: dict[str, Any] = {}
    if timeout_graceful_shutdown is not None:
        shutdown_kwargs["timeout_graceful_shutdown"] = timeout_graceful_shutdown

    if workers == 1:
        app = create_app(model, **app_kwargs)
        uvicorn.run(
            app, host=host, port=port, **tls_kwargs, **shutdown_kwargs
        )
        return

    if "variants" in app_kwargs:
        raise AutoVisionError(
            "variants= is not supported with workers>1; run one model per "
            "worker deployment or route variants at the gateway."
        )
    if model is None or not isinstance(model, (str, Path)):
        raise AutoVisionError(
            "workers>1 requires model=<path>; live engine/model objects cannot "
            "be rebuilt inside worker processes."
        )
    config = {"model": str(model), **app_kwargs}
    try:
        payload = json.dumps(_jsonable(config), sort_keys=True)
    except TypeError as error:
        raise AutoVisionError(f"Cannot serialize serving config: {error}") from error
    os.environ[SERVE_CONFIG_ENV] = payload
    uvicorn.run(
        "autovision.serving.worker:create_app_from_env",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        **tls_kwargs,
        **shutdown_kwargs,
    )
