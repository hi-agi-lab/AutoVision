# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Token-bucket rate limiting: unit behavior and /predict integration."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from tests.test_tracking import tiny_model  # noqa: F401 - fixture reuse

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import TokenBucketRateLimiter, create_app  # noqa: E402


def _b64_image(seed: int = 0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestTokenBucket:
    def test_burst_then_refill(self):
        clock = _Clock()
        limiter = TokenBucketRateLimiter(60, burst=1, clock=clock)
        assert limiter.allow("a") == (True, 0.0)
        allowed, retry_after = limiter.allow("a")
        assert allowed is False
        assert retry_after == pytest.approx(1.0)
        clock.now += 1.0
        assert limiter.allow("a")[0] is True

    def test_keys_are_independent(self):
        clock = _Clock()
        limiter = TokenBucketRateLimiter(60, burst=1, clock=clock)
        assert limiter.allow("a")[0] is True
        assert limiter.allow("b")[0] is True
        assert limiter.allow("a")[0] is False

    def test_default_burst_is_ten_seconds(self):
        limiter = TokenBucketRateLimiter(60, clock=_Clock())
        assert limiter.burst == 10.0

    def test_invalid_limits_rejected(self):
        with pytest.raises(ValueError, match="per_minute"):
            TokenBucketRateLimiter(0)
        with pytest.raises(ValueError, match="burst"):
            TokenBucketRateLimiter(60, burst=0)


class TestRateLimitIntegration:
    def test_second_request_is_429_and_other_client_is_isolated(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model, threshold=0.0,
            rate_limit_per_minute=60, rate_limit_burst=1,
            trust_proxy_headers=True,
        )
        client = TestClient(app)
        payload = {"image": _b64_image(1)}
        first = client.post(
            "/predict", json=payload, headers={"X-Forwarded-For": "10.0.0.1"}
        )
        assert first.status_code == 200
        second = client.post(
            "/predict", json=payload, headers={"X-Forwarded-For": "10.0.0.1"}
        )
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) >= 1
        other = client.post(
            "/predict", json=payload, headers={"X-Forwarded-For": "10.0.0.2"}
        )
        assert other.status_code == 200
        assert 'status="429"' in client.get("/metrics").text

    def test_burst_without_rate_is_rejected(self, tiny_model):  # noqa: F811
        with pytest.raises(AutoVisionError, match="rate_limit_burst"):
            create_app(tiny_model, rate_limit_burst=5)
