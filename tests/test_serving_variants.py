# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""A/B and canary routing across ready engines."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import create_app  # noqa: E402


def _b64_image(seed: int = 0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode()


class _Engine:
    backend = "torch"
    artifact = None
    network = None

    def __init__(self, score: float) -> None:
        self.score = score

    def run_batch(self, images, *, threshold=0.5):
        return [
            {
                "labels": np.array([0]),
                "scores": np.array([self.score]),
                "bboxes": np.zeros((1, 4)),
            }
            for _ in images
        ]


def _two_variants(**kwargs):
    return create_app(
        variants={"stable": _Engine(0.1), "canary": _Engine(0.9)},
        threshold=0.0,
        **kwargs,
    )


class TestVariantRouting:
    def test_weighted_selection_and_forced_header(self):
        client = TestClient(
            _two_variants(variant_weights={"stable": 0.0, "canary": 1.0})
        )
        payload = {"image": _b64_image(1)}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["variant"] == "canary"
        assert response.json()["results"][0]["scores"] == [0.9]

        forced = client.post(
            "/predict", json=payload, headers={"X-AB-Variant": "stable"}
        )
        assert forced.status_code == 200
        assert forced.json()["variant"] == "stable"
        assert forced.json()["results"][0]["scores"] == [0.1]

        unknown = client.post(
            "/predict", json=payload, headers={"X-AB-Variant": "ghost"}
        )
        assert unknown.status_code == 400
        assert "expected one of" in unknown.json()["error"]

    def test_equal_weights_exercise_both_variants(self):
        client = TestClient(_two_variants())
        payload = {"image": _b64_image(2)}
        seen = set()
        for index in range(100):
            response = client.post(
                "/predict", json=payload, headers={"X-Request-ID": f"req-{index}"}
            )
            assert response.status_code == 200
            seen.add(response.json()["variant"])
        assert seen == {"stable", "canary"}

    def test_same_request_id_is_sticky(self):
        client = TestClient(_two_variants())
        payload = {"image": _b64_image(3)}
        first = client.post(
            "/predict", json=payload, headers={"X-Request-ID": "sticky"}
        ).json()["variant"]
        second = client.post(
            "/predict", json=payload, headers={"X-Request-ID": "sticky"}
        ).json()["variant"]
        assert first == second

    def test_health_and_readyz_report_variants(self):
        client = TestClient(_two_variants())
        health = client.get("/healthz").json()
        assert health["variant"] == "stable"
        assert health["variants"] == ["canary", "stable"]
        ready = client.get("/readyz").json()
        assert ready["ready"] is True
        assert ready["variants"] == {"stable": True, "canary": True}

    def test_readyz_503_when_any_variant_is_not_callable(self):
        app = create_app(
            variants={"stable": _Engine(0.1), "broken": object()},
            threshold=0.0,
        )
        response = TestClient(app).get("/readyz")
        assert response.status_code == 503
        assert response.json()["variants"] == {"stable": True, "broken": False}

    def test_variant_requests_are_counted(self):
        client = TestClient(
            _two_variants(variant_weights={"stable": 0.0, "canary": 1.0})
        )
        client.post("/predict", json={"image": _b64_image(4)})
        assert (
            'autovision_variant_requests_total{status="200",variant="canary"}'
            in client.get("/metrics").text
        )
        assert (
            'autovision_variant_predict_latency_seconds_count{variant="canary"}'
            in client.get("/metrics").text
        )


class TestVariantValidation:
    def test_variants_are_exclusive_with_engine_and_model(self):
        with pytest.raises(AutoVisionError, match="not both"):
            create_app(variants={"a": _Engine(0.1)}, engine=_Engine(0.2))
        with pytest.raises(AutoVisionError, match="not both"):
            create_app("model.pt", variants={"a": _Engine(0.1)})

    def test_empty_variants_rejected(self):
        with pytest.raises(AutoVisionError, match="non-empty"):
            create_app(variants={})

    def test_weight_validation(self):
        with pytest.raises(AutoVisionError, match="exactly match"):
            _two_variants(variant_weights={"stable": 1.0})
        with pytest.raises(AutoVisionError, match=">= 0"):
            _two_variants(variant_weights={"stable": -1.0, "canary": 1.0})
        with pytest.raises(AutoVisionError, match="finite"):
            _two_variants(
                variant_weights={"stable": float("nan"), "canary": 1.0}
            )
        with pytest.raises(AutoVisionError, match="positive"):
            _two_variants(variant_weights={"stable": 0.0, "canary": 0.0})
