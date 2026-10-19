# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference serving (roadmap F2): FastAPI + batch + metrics.

A deliberately MINIMAL, testable reference — the prerequisite for
inference monitoring, not a production gateway:

- ``POST /predict`` — JSON body ``{"images": ["<base64>", ...]}`` (or a
  single ``{"image": ...}``); results come back in input order. Pre/post
  processing is the network's own (the same source as evaluation and the
  stream runtime), so served predictions match offline readings.
- ``GET /healthz`` — liveness + the loaded backend/model identity.
- ``GET /readyz`` — readiness: 200 only when every configured variant's
  inference engine is callable, 503 otherwise.
- ``GET /drift`` — prediction-distribution drift (PSI) against an optional
  baseline; ``enabled: false`` when no baseline was configured.
- ``GET /metrics`` — Prometheus exposition text (requests, errors, images,
  latency summary) for scraping.

Production guards (authentication is opt-in; size/concurrency limits default to
production-safe values):

- ``api_key=`` enables a constant-time bearer/API-key check; without it the
  reference stays unauthenticated (documented boundary).
- ``max_body_bytes`` bounds the decoded request body while streaming.
- ``max_image_pixels`` rejects decompression bombs before decoding.
- ``rate_limit_per_minute`` enables a per-client token bucket (per process).
- ``variants=`` / ``variant_weights=`` provide A/B or canary routing across
  ready engines; ``X-AB-Variant`` forces a variant for testing.
- Inference runs in a worker thread; a semaphore bounds concurrent engine
  calls (TensorRT/OpenVINO runners are not thread-safe, so the default
  serializes calls while keeping the event loop responsive).

Explicitly OUT of scope (documented boundary, README): TLS and multi-worker
orchestration live in ``autovision.serving.server.serve``; the monitoring
layers built on top of ``/metrics`` and ``/drift`` (alerting, full A/B
analysis) are deployment/product concerns.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from autovision.errors import AutoVisionError
from autovision.results import InferenceResult
from autovision.runtime.env import Env
from autovision.serving.drift import (
    DriftBaseline,
    DriftMonitor,
    load_drift_baseline,
)
from autovision.serving.metrics import MetricsRegistry
from autovision.serving.ratelimit import TokenBucketRateLimiter

__all__ = ["create_app"]

logger = logging.getLogger("autovision.serving")

_REQUESTS_HELP = "Served requests by endpoint and status"
_VARIANTS_HELP = "Served requests by variant and status"
_ERRORS_HELP = "Failed predictions"
_DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


class _PayloadTooLarge(AutoVisionError):
    """A request body or decoded image exceeded its configured size limit."""


def _decode_image(payload: str, *, max_pixels: int) -> Any:
    """base64 image payload → HWC uint8 numpy array (RGB)."""
    import numpy as np
    from PIL import Image

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as error:  # noqa: BLE001 - malformed payload is a 400
        raise AutoVisionError(f"Invalid base64 image payload: {error}") from error
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise _PayloadTooLarge(
                    f"Image is {width}x{height} ({width * height} pixels), "
                    f"which exceeds max_image_pixels={max_pixels}."
                )
            frame = image.convert("RGB")
    except AutoVisionError:
        raise
    except Image.DecompressionBombError as error:
        raise _PayloadTooLarge(
            "Image exceeds Pillow's decompression-bomb limit."
        ) from error
    except Exception as error:  # noqa: BLE001 - undecodable payload is a 400
        # PIL's own message embeds an object address — keep the body stable.
        raise AutoVisionError(
            "Unsupported image payload — expected base64-encoded PNG/JPEG bytes."
        ) from error
    return np.array(frame, copy=True)


async def _read_body_limited(request: Any, limit: int) -> bytes:
    """Read the request body, aborting as soon as ``limit`` is exceeded.

    ``Content-Length`` is checked first (cheap rejection); the streaming
    loop also enforces the cap for chunked/lying clients.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError as error:
            raise AutoVisionError(f"Invalid Content-Length header: {declared!r}.") from error
        if declared_bytes > limit:
            raise _PayloadTooLarge(
                f"Request body of {declared_bytes} bytes exceeds "
                f"max_body_bytes={limit}."
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _PayloadTooLarge(
                f"Request body exceeds max_body_bytes={limit}."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _authorized(request: Any, api_key: str | None) -> bool:
    """Constant-time API-key check (no-op when ``api_key`` is not set)."""
    if api_key is None:
        return True
    header = request.headers.get("authorization", "")
    candidate = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not candidate:
        candidate = request.headers.get("x-api-key", "")
    # compare_digest rejects non-ASCII str inputs with TypeError; compare the
    # UTF-8 bytes so a malformed credential is a 401, not an uncaught 500.
    return secrets.compare_digest(
        candidate.encode("utf-8"), api_key.encode("utf-8")
    )


def _client_key(request: Any, *, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip() or "unknown"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return str(host or "unknown")


def _engine_info(engine: Any) -> dict[str, Any]:
    network = getattr(engine, "network", None)
    artifact = getattr(engine, "artifact", None)
    # Probes are unauthenticated by design; expose only the filename, never
    # the deployment's absolute path.
    artifact_name = Path(artifact).name if artifact else None
    return {
        "backend": getattr(engine, "backend", None),
        "model": type(network).__name__ if network is not None else None,
        "artifact": artifact_name,
        "openvino_device": getattr(engine, "openvino_device", None),
    }


def _serialize_records(
    records: list[dict[str, Any]],
    names: dict[int, str],
    mask_format: str,
    max_json_elements: int,
) -> list[dict[str, Any]]:
    """RLE/JSON conversion off the event loop (heavy for instance masks)."""
    results: list[dict[str, Any]] = []
    for record in records:
        entry = InferenceResult.from_record(record).to_json(
            masks=mask_format, max_elements=max_json_elements
        )
        if names and isinstance(entry.get("labels"), list):
            entry["class_names"] = [
                names.get(int(label), str(label)) for label in entry["labels"]
            ]
        results.append(entry)
    return results


def create_app(
    model: Any = None,
    *,
    engine: Any = None,
    variants: dict[str, Any] | None = None,
    variant_weights: dict[str, float] | None = None,
    backend: str = "torch",
    artifact: "str | Path | None" = None,
    device: str | None = None,
    providers: "list[str] | None" = None,
    openvino_device: str = "CPU",
    threshold: float = 0.5,
    max_batch: int = 16,
    class_names: dict[int, str] | None = None,
    api_key: str | None = None,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS,
    max_concurrency: int = 1,
    queue_timeout_s: float | None = 30.0,
    inference_timeout_s: float | None = 120.0,
    mask_format: str = "rle",
    max_json_elements: int = 10_000,
    rate_limit_per_minute: int | None = None,
    rate_limit_burst: int | None = None,
    trust_proxy_headers: bool = False,
    docs_enabled: bool = False,
    protect_observability_endpoints: bool = True,
    drift_baseline: "str | Path | dict[str, Any] | DriftBaseline | None" = None,
    drift_window: int = 1000,
    drift_threshold: float = 0.2,
    drift_min_samples: int = 30,
) -> Any:
    """Build the FastAPI app around a task model, engine, or variant engines.

    ``model`` may be an exported ``.pt`` path or a ``TaskNetwork`` (it gets
    wrapped in an :class:`~autovision.inference.InferenceEngine`); pass a
    ready engine via ``engine=`` instead. ``variants`` maps variant names to
    ready engines for A/B/canary routing (mutually exclusive with
    ``model=``/``engine=``); ``variant_weights`` defaults to equal splits.

    ``api_key`` enables an opt-in bearer/``X-API-Key`` check. The other
    guards default to production-safe values; ``max_concurrency`` stays at
    1 because the TensorRT/OpenVINO runners are not thread-safe.
    ``queue_timeout_s`` bounds the wait for an inference slot (503 on
    overload; ``None`` waits indefinitely). ``inference_timeout_s`` bounds
    the engine call itself (504 on timeout; the engine is then marked
    unresponsive and later requests return 503 until restart, because a
    thread stuck in a non-thread-safe backend cannot be cancelled safely).
    ``docs_enabled`` exposes
    FastAPI's ``/docs``/``/openapi.json`` (off by default: they reveal the
    inference schema). ``protect_observability_endpoints`` puts ``/metrics``
    and ``/drift`` behind the same API key (probes stay public for Kubernetes).
    ``rate_limit_per_minute`` is per process (see the
    module docstring). ``drift_baseline`` accepts a baseline path/dict or a
    :class:`~autovision.serving.drift.DriftBaseline`.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as error:  # pragma: no cover - import guard path
        raise AutoVisionError(
            "The 'fastapi' package is required for the serving reference. "
            "Install it with: pip install 'autovision[serving]'."
        ) from error

    if max_batch < 1:
        raise AutoVisionError(f"max_batch must be >= 1, got {max_batch}.")
    if api_key is None:
        # Prefer the environment over a CLI argument: process listings and
        # shell history would otherwise expose the key.
        api_key = Env.SERVING_API_KEY.value
    if api_key is not None and (
        not isinstance(api_key, str) or not api_key
    ):
        raise AutoVisionError("api_key must be a non-empty string when provided.")
    if max_body_bytes < 1:
        raise AutoVisionError(f"max_body_bytes must be >= 1, got {max_body_bytes}.")
    if max_image_pixels < 1:
        raise AutoVisionError(
            f"max_image_pixels must be >= 1, got {max_image_pixels}."
        )
    if max_concurrency < 1:
        raise AutoVisionError(
            f"max_concurrency must be >= 1, got {max_concurrency}."
        )
    if queue_timeout_s is not None and queue_timeout_s <= 0:
        raise AutoVisionError(
            f"queue_timeout_s must be > 0 or None, got {queue_timeout_s}."
        )
    if inference_timeout_s is not None and inference_timeout_s <= 0:
        raise AutoVisionError(
            "inference_timeout_s must be > 0 or None, got "
            f"{inference_timeout_s}."
        )
    if mask_format not in {"rle", "dense", "none"}:
        raise AutoVisionError(
            "mask_format must be 'rle', 'dense', or 'none', "
            f"got {mask_format!r}."
        )
    if max_json_elements < 1:
        raise AutoVisionError(
            f"max_json_elements must be >= 1, got {max_json_elements}."
        )
    if rate_limit_burst is not None and rate_limit_per_minute is None:
        raise AutoVisionError(
            "rate_limit_burst requires rate_limit_per_minute."
        )

    import torch

    from autovision.inference import InferenceEngine

    if variants is not None:
        if engine is not None or model is not None:
            raise AutoVisionError(
                "pass either variants= or model=/engine=, not both."
            )
        if not isinstance(variants, dict) or not variants:
            raise AutoVisionError("variants must be a non-empty dict.")
        engines = dict(variants)
        for name in engines:
            if not isinstance(name, str) or not name:
                raise AutoVisionError("Variant names must be non-empty strings.")
    else:
        if engine is None:
            if model is None:
                raise AutoVisionError("create_app needs either engine= or model=.")
            engine = InferenceEngine(
                model, backend=backend, artifact=artifact, device=device,
                providers=providers, openvino_device=openvino_device,
            )
        engines = {"default": engine}

    default_variant = next(iter(engines))
    weights: dict[str, float]
    if variant_weights is None:
        weights = {name: 1.0 / len(engines) for name in engines}
    else:
        if set(variant_weights) != set(engines):
            raise AutoVisionError(
                "variant_weights keys must exactly match the variant names."
            )
        raw_weights = {name: float(value) for name, value in variant_weights.items()}
        if any(not math.isfinite(value) for value in raw_weights.values()):
            raise AutoVisionError("variant_weights must be finite.")
        if any(value < 0 for value in raw_weights.values()):
            raise AutoVisionError("variant_weights must be >= 0.")
        total = sum(raw_weights.values())
        if total <= 0:
            raise AutoVisionError("variant_weights must sum to a positive value.")
        weights = {name: value / total for name, value in raw_weights.items()}

    limiter: TokenBucketRateLimiter | None = None
    if rate_limit_per_minute is not None:
        try:
            limiter = TokenBucketRateLimiter(
                rate_limit_per_minute, rate_limit_burst
            )
        except ValueError as error:
            raise AutoVisionError(str(error)) from error

    drift_monitor: DriftMonitor | None = None
    if drift_baseline is not None:
        baseline = (
            load_drift_baseline(drift_baseline)
            if isinstance(drift_baseline, (str, Path))
            else drift_baseline
        )
        drift_monitor = DriftMonitor(
            baseline,
            window=drift_window,
            threshold=drift_threshold,
            min_samples=drift_min_samples,
        )

    names = dict(class_names or {})
    metrics = MetricsRegistry()
    metrics.set_gauge("autovision_model_loaded", 1.0, help_text="1 when a model is loaded")
    readiness = {
        name: callable(getattr(candidate, "run_batch", None))
        for name, candidate in engines.items()
    }
    metrics.set_gauge(
        "autovision_model_ready", 1.0 if all(readiness.values()) else 0.0,
        help_text="1 when every inference engine is callable",
    )
    for name, ready in readiness.items():
        metrics.set_gauge(
            "autovision_variant_ready", 1.0 if ready else 0.0,
            {"variant": name}, help_text="1 when this variant is callable",
        )
    inference_slots = asyncio.Semaphore(max_concurrency)
    # Set when an engine call exceeds inference_timeout_s. The worker thread
    # may still be inside a non-thread-safe backend, so the slot stays held
    # and the process must be restarted rather than reusing the engine.
    inference_poisoned = threading.Event()
    docs_kwargs = (
        {}
        if docs_enabled
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(
        title="AutoVision reference serving", version="0.3", **docs_kwargs
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Log in finally so unhandled 500s still carry the request id.
            logger.info(
                "request method=%s path=%s status=%d latency_ms=%.2f request_id=%s",
                request.method, request.url.path, status,
                (time.perf_counter() - started) * 1000.0, request_id,
            )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> Any:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.exception(
            "unhandled request error path=%s request_id=%s",
            request.url.path,
            request_id,
        )
        # /predict already counted its 500 before re-raising; avoid a double
        # count while still counting failures from other endpoints.
        if not getattr(request.state, "counted_500", False):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": request.url.path, "status": "500"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
        return JSONResponse(
            {"error": "Internal server error.", "request_id": request_id},
            status_code=500,
            headers={"X-Request-ID": request_id},
        )

    def _select_variant(request: Request) -> str | None:
        requested = request.headers.get("x-ab-variant")
        if requested is not None:
            return requested if requested in engines else None
        if len(engines) == 1:
            return default_variant
        request_id = getattr(request.state, "request_id", "")
        digest = hashlib.sha256(request_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
        cumulative = 0.0
        for name, weight in weights.items():
            cumulative += weight
            if bucket <= cumulative:
                return name
        return default_variant

    def _predict_one(selected_engine: Any, images_raw: list[str]) -> list[dict[str, Any]]:
        decoded = [
            _decode_image(payload, max_pixels=max_image_pixels)
            for payload in images_raw
        ]
        tensors = [torch.from_numpy(frame).permute(2, 0, 1) for frame in decoded]
        return selected_engine.run_batch(tensors, threshold=threshold)

    def _count_variant(variant: str, status: str) -> None:
        metrics.inc(
            "autovision_variant_requests_total",
            {"variant": variant, "status": status},
            help_text=_VARIANTS_HELP,
        )

    def _throttle(request: Request, endpoint: str) -> Any:
        """Apply the shared token bucket before auth; return a 429 or None."""
        if limiter is None:
            return None
        key = _client_key(request, trust_proxy_headers=trust_proxy_headers)
        allowed, retry_after = limiter.allow(key)
        if allowed:
            return None
        metrics.inc(
            "autovision_requests_total",
            {"endpoint": endpoint, "status": "429"},
            help_text=_REQUESTS_HELP,
        )
        return JSONResponse(
            {
                "error": "Rate limit exceeded.",
                "request_id": getattr(request.state, "request_id", ""),
                "retry_after_s": retry_after,
            },
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )

    def _update_drift_metrics(report: dict[str, Any]) -> None:
        status = report.get("status")
        if status is not None:
            metrics.set_gauge(
                "autovision_drift_detected", 1.0 if status == "drift" else 0.0,
                help_text="1 when the drift monitor reports drift",
            )
        if report.get("psi_label") is not None:
            metrics.set_gauge(
                "autovision_drift_psi", float(report["psi_label"]),
                {"kind": "label"}, help_text="PSI of the served label distribution",
            )
        if report.get("psi_score") is not None:
            metrics.set_gauge(
                "autovision_drift_psi", float(report["psi_score"]),
                {"kind": "score"}, help_text="PSI of the served score distribution",
            )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        info = _engine_info(engines[default_variant])
        return {
            "status": "ok",
            **info,
            "threshold": threshold,
            "variant": default_variant,
            "variants": sorted(engines),
            # Liveness stays 200 (restarting the stuck engine thread does
            # not help), but the flag lets orchestrators trigger a restart.
            "poisoned": inference_poisoned.is_set(),
        }

    @app.get("/readyz")
    def readyz() -> Any:
        if inference_poisoned.is_set():
            # The process must be restarted; report not-ready so an
            # orchestrator removes this instance from load balancing.
            metrics.set_gauge(
                "autovision_model_ready", 0.0,
                help_text="1 when every inference engine is callable",
            )
            for name in engines:
                metrics.set_gauge(
                    "autovision_variant_ready", 0.0,
                    {"variant": name},
                    help_text="1 when this variant is callable",
                )
            return JSONResponse(
                {
                    "ready": False,
                    "poisoned": True,
                    "error": (
                        "inference engine timed out and can no longer be "
                        "used; restart the process"
                    ),
                },
                status_code=503,
            )
        current = {
            name: callable(getattr(candidate, "run_batch", None))
            for name, candidate in engines.items()
        }
        ready = all(current.values())
        metrics.set_gauge(
            "autovision_model_ready", 1.0 if ready else 0.0,
            help_text="1 when every inference engine is callable",
        )
        for name, variant_ready in current.items():
            metrics.set_gauge(
                "autovision_variant_ready", 1.0 if variant_ready else 0.0,
                {"variant": name}, help_text="1 when this variant is callable",
            )
        if not ready:
            return JSONResponse(
                {
                    "ready": False,
                    "variants": current,
                    "error": "one or more inference engines are not callable",
                },
                status_code=503,
            )
        return {
            "ready": True,
            "variants": current,
            "default": default_variant,
            **_engine_info(engines[default_variant]),
        }

    @app.get("/drift")
    def drift(request: Request) -> Any:
        throttled = _throttle(request, "drift")
        if throttled is not None:
            return throttled
        if protect_observability_endpoints and not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "drift", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
        if drift_monitor is None:
            return {"enabled": False}
        report = drift_monitor.snapshot()
        _update_drift_metrics(report)
        return report

    @app.get("/metrics")
    def prometheus_metrics(request: Request) -> Any:
        throttled = _throttle(request, "metrics")
        if throttled is not None:
            return throttled
        if protect_observability_endpoints and not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "metrics", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
        if drift_monitor is not None:
            # A Prometheus scrape is the natural sampling point; computing PSI
            # here keeps the gauge fresh without doing O(window) work per request.
            _update_drift_metrics(drift_monitor.snapshot())
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict")
    async def predict(request: Request) -> Any:
        started = time.perf_counter()
        request_id = getattr(request.state, "request_id", "")
        if limiter is not None:
            # Rate-limit before auth so unauthenticated brute force is
            # throttled too.
            key = _client_key(request, trust_proxy_headers=trust_proxy_headers)
            allowed, retry_after = limiter.allow(key)
            if not allowed:
                metrics.inc(
                    "autovision_requests_total",
                    {"endpoint": "predict", "status": "429"},
                    help_text=_REQUESTS_HELP,
                )
                return JSONResponse(
                    {
                        "error": "Rate limit exceeded.",
                        "request_id": request_id,
                        "retry_after_s": retry_after,
                    },
                    status_code=429,
                    headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
                )
        if not _authorized(request, api_key):
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "401"},
                help_text=_REQUESTS_HELP,
            )
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": "Unauthorized.", "request_id": request_id},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        variant = _select_variant(request)
        if variant is None:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            return JSONResponse(
                {
                    "error": "Unknown X-AB-Variant; expected one of "
                    f"{sorted(engines)}.",
                    "request_id": request_id,
                },
                status_code=400,
            )
        request.state.variant = variant
        if inference_poisoned.is_set():
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "503"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "503")
            return JSONResponse(
                {
                    "error": (
                        "Inference engine is unresponsive after a previous "
                        "timeout; restart the service."
                    ),
                    "request_id": request_id,
                },
                status_code=503,
                headers={"Retry-After": "60"},
            )
        try:
            body = await _read_body_limited(request, max_body_bytes)
        except _PayloadTooLarge as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "413")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=413
            )
        except AutoVisionError as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "400")
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=400
            )
        try:
            payload = json.loads(body) if body else None
        except Exception as error:  # noqa: BLE001
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "400")
            return JSONResponse(
                {"error": f"Invalid JSON body: {error}", "request_id": request_id},
                status_code=400,
            )
        if isinstance(payload, dict) and "image" in payload and "images" not in payload:
            payload = {"images": [payload["image"]]}
        images_raw = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images_raw, list) or not images_raw:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "400")
            return JSONResponse(
                {
                    "error": "Body must be {\"images\": [<base64>, ...]} (or {\"image\": ...}).",
                    "request_id": request_id,
                },
                status_code=400,
            )
        if len(images_raw) > max_batch:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "413")
            return JSONResponse(
                {
                    "error": f"Batch of {len(images_raw)} exceeds max_batch={max_batch}.",
                    "request_id": request_id,
                },
                status_code=413,
            )
        slot_acquired = False
        try:
            if queue_timeout_s is None:
                await inference_slots.acquire()
            else:
                try:
                    await asyncio.wait_for(
                        inference_slots.acquire(), timeout=queue_timeout_s
                    )
                except asyncio.TimeoutError:
                    metrics.inc(
                        "autovision_requests_total",
                        {"endpoint": "predict", "status": "503"},
                        help_text=_REQUESTS_HELP,
                    )
                    _count_variant(variant, "503")
                    metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
                    return JSONResponse(
                        {
                            "error": (
                                "Inference queue is saturated; retry later "
                                f"(queue_timeout_s={queue_timeout_s})."
                            ),
                            "request_id": request_id,
                        },
                        status_code=503,
                        headers={"Retry-After": "1"},
                    )
            slot_acquired = True
            if inference_timeout_s is None:
                records = await run_in_threadpool(
                    _predict_one, engines[variant], images_raw
                )
            else:
                try:
                    records = await asyncio.wait_for(
                        run_in_threadpool(
                            _predict_one, engines[variant], images_raw
                        ),
                        timeout=inference_timeout_s,
                    )
                except asyncio.TimeoutError:
                    # The worker may still be inside the engine. Keep the
                    # slot held and poison the process: reusing a stuck
                    # TensorRT/OpenVINO runner would be unsafe.
                    inference_poisoned.set()
                    metrics.inc(
                        "autovision_requests_total",
                        {"endpoint": "predict", "status": "504"},
                        help_text=_REQUESTS_HELP,
                    )
                    _count_variant(variant, "504")
                    metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
                    metrics.inc(
                        "autovision_inference_timeouts_total",
                        help_text="Engine calls that exceeded inference_timeout_s",
                    )
                    logger.error(
                        "Inference timed out after %.3fs (request_id=%s); "
                        "marking the engine unresponsive.",
                        inference_timeout_s,
                        request_id,
                    )
                    return JSONResponse(
                        {
                            "error": (
                                "Inference timed out after "
                                f"{inference_timeout_s}s; the engine is now "
                                "unresponsive and the service must be restarted."
                            ),
                            "request_id": request_id,
                        },
                        status_code=504,
                        headers={"Retry-After": "60"},
                    )
            if len(records) != len(images_raw):
                raise RuntimeError(
                    "Engine returned "
                    f"{len(records)} records for {len(images_raw)} images."
                )
            results = await run_in_threadpool(
                _serialize_records,
                records,
                names,
                mask_format,
                max_json_elements,
            )
        except _PayloadTooLarge as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "413"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "413")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=413
            )
        except AutoVisionError as error:
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "400"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "400")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            return JSONResponse(
                {"error": str(error), "request_id": request_id}, status_code=400
            )
        except Exception:
            # Anything else becomes FastAPI's 500 — count it before re-raising
            # so a failing deployment stays visible on /metrics. The global
            # handler must not count the same failure twice.
            request.state.counted_500 = True
            metrics.inc(
                "autovision_requests_total",
                {"endpoint": "predict", "status": "500"},
                help_text=_REQUESTS_HELP,
            )
            _count_variant(variant, "500")
            metrics.inc("autovision_errors_total", help_text=_ERRORS_HELP)
            metrics.observe(
                "autovision_predict_latency_seconds",
                time.perf_counter() - started,
                help_text="Per-request predict latency (seconds)",
            )
            raise
        finally:
            if slot_acquired and not inference_poisoned.is_set():
                inference_slots.release()
        elapsed = time.perf_counter() - started
        metrics.inc(
            "autovision_requests_total",
            {"endpoint": "predict", "status": "200"},
            help_text=_REQUESTS_HELP,
        )
        _count_variant(variant, "200")
        metrics.inc(
            "autovision_images_total", value=float(len(records)),
            help_text="Images served",
        )
        metrics.observe(
            "autovision_predict_latency_seconds", elapsed,
            help_text="Per-request predict latency (seconds)",
        )
        metrics.observe(
            "autovision_variant_predict_latency_seconds", elapsed,
            {"variant": variant},
            help_text="Per-variant predict latency (seconds)",
        )
        if drift_monitor is not None:
            drift_monitor.record_many(records)
        return {
            "results": results,
            "count": len(results),
            "latency_s": elapsed,
            "request_id": request_id,
            "variant": variant,
        }

    return app
