# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference serving: FastAPI app, metrics, drift, launcher.

``create_app(...)`` builds the app (``/predict`` batch endpoint,
``/healthz``, ``/readyz``, ``/drift``, Prometheus ``/metrics``); the
``serving`` extra carries fastapi/uvicorn. ``serve(...)`` adds TLS,
multi-worker launch and the same app options. Boundaries (full production
gateway, global rate limits, alerting) are documented in the README and
module docstrings.
"""

from autovision.serving.app import create_app
from autovision.serving.drift import (
    DriftBaseline,
    DriftMonitor,
    build_drift_baseline,
    load_drift_baseline,
)
from autovision.serving.metrics import MetricsRegistry
from autovision.serving.ratelimit import TokenBucketRateLimiter
from autovision.serving.server import serve

__all__ = [
    "DriftBaseline",
    "DriftMonitor",
    "MetricsRegistry",
    "TokenBucketRateLimiter",
    "build_drift_baseline",
    "create_app",
    "load_drift_baseline",
    "serve",
]
