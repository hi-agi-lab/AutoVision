# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference serving (roadmap F2): /predict batching, /healthz, /metrics."""

import asyncio
import base64
import io
import threading
import time

import httpx
import numpy as np
import pytest
from tests.test_tracking import tiny_model  # noqa: F401 - local fixture reuse

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import create_app  # noqa: E402


def _b64_image(size: int = 64, seed: int = 0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def client(tiny_model):  # noqa: F811 - pytest fixture parameter
    app = create_app(tiny_model, threshold=0.0, max_batch=4, class_names={0: "cat", 1: "dog"})
    return TestClient(app)


class TestServing:
    def test_healthz_reports_the_backend(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["backend"] == "torch"
        assert response.json()["status"] == "ok"

    def test_probes_do_not_leak_artifact_paths(self):
        class _PathEngine:
            backend = "onnx"
            artifact = "/srv/private/models/model.onnx"
            network = None

            def run_batch(self, images, *, threshold=0.5):
                return []

        client = TestClient(create_app(engine=_PathEngine(), threshold=0.0))
        payload = client.get("/healthz").json()
        assert payload["artifact"] == "model.onnx"
        assert "/srv/private" not in str(payload)

    def test_predict_single_and_batch_order(self, client):
        first = _b64_image(seed=1)
        second = _b64_image(seed=2)
        single = client.post("/predict", json={"image": first})
        assert single.status_code == 200, single.text
        payload = single.json()
        assert payload["count"] == 1
        assert set(payload["results"][0]) >= {"bboxes", "scores", "labels", "class_names"}
        assert len(payload["results"][0]["scores"]) == 10  # num_top_queries

        batch = client.post("/predict", json={"images": [first, second, first]})
        assert batch.status_code == 200, batch.text
        results = batch.json()["results"]
        assert batch.json()["count"] == 3
        # Input order is preserved: A, B, A → A, B, A (not sorted/deduped).
        # Batched forwards may differ from single ones in the last float bits,
        # hence allclose (bit-equality is the export gate's job, not a
        # different-batch-size run's).
        assert np.allclose(results[0]["scores"], payload["results"][0]["scores"], atol=1e-6)
        assert np.allclose(results[0]["scores"], results[2]["scores"], atol=1e-12)
        assert not np.allclose(results[0]["scores"], results[1]["scores"], atol=1e-3)

    def test_metrics_endpoint_renders_prometheus_text(self, client):
        client.post("/predict", json={"image": _b64_image(seed=4)})
        text = client.get("/metrics").text
        assert "# TYPE autovision_requests_total counter" in text
        assert 'autovision_requests_total{endpoint="predict",status="200"}' in text
        assert "# HELP autovision_requests_total" in text  # never order-dependent
        assert "autovision_images_total" in text
        assert "# TYPE autovision_predict_latency_seconds summary" in text
        assert "autovision_predict_latency_seconds_count" in text
        assert "autovision_predict_latency_seconds_sum" in text
        # The peak rides in its own gauge family (not inside the summary).
        assert "# TYPE autovision_predict_latency_seconds_max gauge" in text
        assert "autovision_predict_latency_seconds_max " in text
        assert "autovision_model_loaded 1" in text

    def test_bad_payloads_are_400_and_counted(self, client):
        missing = client.post("/predict", json={"nope": 1})
        assert missing.status_code == 400
        assert "images" in missing.json()["error"]
        bad_json = client.post(
            "/predict", content=b"not json", headers={"content-type": "application/json"}
        )
        assert bad_json.status_code == 400
        bad_b64 = client.post("/predict", json={"images": ["!!not-base64!!"]})
        assert bad_b64.status_code == 400
        text = client.get("/metrics").text
        assert 'status="400"' in text

    def test_batch_limit_is_413(self, client):
        response = client.post(
            "/predict", json={"images": [_b64_image(seed=5)] * 5}
        )
        assert response.status_code == 413
        assert "max_batch" in response.json()["error"]

    def test_batch_matches_single_calls(self, client):
        """Serving a batch equals serving the images one by one."""
        images = [_b64_image(seed=6), _b64_image(seed=7)]
        batch = client.post("/predict", json={"images": images}).json()["results"]
        singles = [
            client.post("/predict", json={"image": payload}).json()["results"][0]
            for payload in images
        ]
        for batched, single in zip(batch, singles):
            assert np.allclose(batched["scores"], single["scores"], atol=1e-6)
            assert np.allclose(batched["bboxes"], single["bboxes"], atol=1e-4)


class TestFailureVisibility:
    def test_engine_errors_are_500_and_counted(self):
        """A crashing forward stays visible on /metrics (no silent 5xx)."""

        class _BrokenEngine:
            backend = "torch"
            artifact = None
            network = None

            def run_batch(self, images, *, threshold=0.5):
                raise RuntimeError("engine exploded")

        app = create_app(engine=_BrokenEngine(), threshold=0.0, max_batch=4)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/predict", json={"image": _b64_image(seed=8)})
        assert response.status_code == 500
        assert response.headers["X-Request-ID"]
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        text = client.get("/metrics").text
        assert 'autovision_requests_total{endpoint="predict",status="500"}' in text
        assert "autovision_errors_total 1" in text

    def test_inference_timeout_returns_504_then_poisons_the_engine(self):
        class _HungEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.release = threading.Event()

            def run_batch(self, images, *, threshold=0.5):
                self.release.wait(timeout=5.0)
                return [{} for _ in images]

        engine = _HungEngine()
        app = create_app(
            engine=engine,
            threshold=0.0,
            inference_timeout_s=0.05,
            queue_timeout_s=0.05,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            first = client.post("/predict", json={"image": _b64_image(seed=21)})
            assert first.status_code == 504
            assert "timed out" in first.json()["error"]
            assert first.headers["X-Request-ID"]
            second = client.post("/predict", json={"image": _b64_image(seed=22)})
            assert second.status_code == 503
            assert "restart" in second.json()["error"]
            text = client.get("/metrics").text
            assert 'status="504"' in text
            assert "autovision_inference_timeouts_total 1" in text
        finally:
            engine.release.set()

    def test_poisoned_engine_reports_unready(self):
        """After a timeout, /readyz must 503 so orchestrators restart."""

        class _HungEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.release = threading.Event()

            def run_batch(self, images, *, threshold=0.5):
                self.release.wait(timeout=5.0)
                return [{} for _ in images]

        engine = _HungEngine()
        app = create_app(
            engine=engine,
            threshold=0.0,
            inference_timeout_s=0.05,
            queue_timeout_s=0.05,
        )
        client = TestClient(app, raise_server_exceptions=False)
        try:
            first = client.post("/predict", json={"image": _b64_image(seed=31)})
            assert first.status_code == 504
            ready = client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["poisoned"] is True
            assert "restart" in ready.json()["error"]
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["poisoned"] is True
            text = client.get("/metrics").text
            assert "autovision_model_ready 0" in text
            assert 'autovision_variant_ready{variant="default"} 0' in text
        finally:
            engine.release.set()


class TestAppValidation:
    def test_missing_model_fails_loud(self):
        with pytest.raises(AutoVisionError, match="engine= or model="):
            create_app()

    def test_bad_max_batch_fails_loud(self, tiny_model):  # noqa: F811 - fixture param
        with pytest.raises(AutoVisionError, match="max_batch"):
            create_app(tiny_model, max_batch=0)

    def test_bad_guards_fail_loud(self, tiny_model):  # noqa: F811 - fixture param
        with pytest.raises(AutoVisionError, match="api_key"):
            create_app(tiny_model, api_key="")
        with pytest.raises(AutoVisionError, match="api_key"):
            create_app(tiny_model, api_key=123)  # type: ignore[arg-type]
        with pytest.raises(AutoVisionError, match="max_body_bytes"):
            create_app(tiny_model, max_body_bytes=0)
        with pytest.raises(AutoVisionError, match="max_image_pixels"):
            create_app(tiny_model, max_image_pixels=0)
        with pytest.raises(AutoVisionError, match="max_concurrency"):
            create_app(tiny_model, max_concurrency=0)


class TestProductionGuards:
    def test_readyz_reports_ready(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert "autovision_model_ready 1" in client.get("/metrics").text

    def test_readyz_is_503_without_a_callable_engine(self):
        class _NotReady:
            backend = "torch"
            artifact = None
            network = None

        client = TestClient(create_app(engine=_NotReady()))
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_api_key_is_opt_in_and_constant_time_checked(self, tiny_model):  # noqa: F811
        app = create_app(tiny_model, threshold=0.0, api_key="s3cret")
        client = TestClient(app)
        payload = {"image": _b64_image(seed=11)}

        assert client.post("/predict", json=payload).status_code == 401
        assert (
            client.post(
                "/predict", json=payload, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/predict", json=payload, headers={"Authorization": "Bearer s3cret"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/predict", json=payload, headers={"X-API-Key": "s3cret"}
            ).status_code
            == 200
        )
        assert 'status="401"' in client.get(
            "/metrics", headers={"X-API-Key": "s3cret"}
        ).text

    def test_api_key_can_come_from_env(self, tiny_model, monkeypatch):  # noqa: F811
        monkeypatch.setenv("AUTOVISION_SERVING_API_KEY", "env-secret")
        client = TestClient(create_app(tiny_model, threshold=0.0))
        payload = {"image": _b64_image(seed=17)}
        assert client.post("/predict", json=payload).status_code == 401
        assert (
            client.post(
                "/predict", json=payload, headers={"X-API-Key": "env-secret"}
            ).status_code
            == 200
        )

    def test_body_limit_is_413(self, tiny_model):  # noqa: F811
        client = TestClient(
            create_app(tiny_model, threshold=0.0, max_body_bytes=64)
        )
        response = client.post("/predict", json={"image": _b64_image(seed=12)})
        assert response.status_code == 413
        assert "max_body_bytes" in response.json()["error"]
        assert 'status="413"' in client.get("/metrics").text

    def test_image_pixel_limit_is_413(self, tiny_model):  # noqa: F811
        client = TestClient(
            create_app(tiny_model, threshold=0.0, max_image_pixels=100)
        )
        response = client.post("/predict", json={"image": _b64_image(seed=13)})
        assert response.status_code == 413
        assert "max_image_pixels" in response.json()["error"]

    def test_request_id_is_echoed(self, client):
        response = client.post(
            "/predict",
            json={"image": _b64_image(seed=14)},
            headers={"X-Request-ID": "req-abc"},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "req-abc"
        assert response.json()["request_id"] == "req-abc"

    def test_inference_runs_off_the_event_loop_and_is_serialized(self):
        class _SlowEngine:
            backend = "torch"
            artifact = None
            network = None

            def __init__(self):
                self.active = 0
                self.max_active = 0

            def run_batch(self, images, *, threshold=0.5):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                time.sleep(0.5)
                self.active -= 1
                return [{"score": 1.0} for _ in images]

        engine = _SlowEngine()
        app = create_app(engine=engine, threshold=0.0)

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as async_client:
                first = asyncio.create_task(
                    async_client.post("/predict", json={"image": _b64_image(seed=15)})
                )
                second = asyncio.create_task(
                    async_client.post("/predict", json={"image": _b64_image(seed=16)})
                )
                await asyncio.sleep(0.05)
                started = time.perf_counter()
                health = await async_client.get("/healthz")
                health_elapsed = time.perf_counter() - started
                responses = await asyncio.gather(first, second)
                return health, health_elapsed, responses

        health, health_elapsed, responses = asyncio.run(scenario())
        # The blocking forward must not stall the event loop: /healthz answers
        # while both predictions are in flight.
        assert health.status_code == 200
        assert health_elapsed < 0.35, (
            f"/healthz took {health_elapsed:.3f}s while /predict was blocked"
        )
        assert [response.status_code for response in responses] == [200, 200]
        # Default max_concurrency=1 serializes engine calls (TRT/OVN runners
        # are not thread-safe).
        assert engine.max_active == 1


class TestServingHardening:
    def test_docs_endpoints_are_disabled_by_default(self, client):
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_docs_can_be_enabled_explicitly(self, tiny_model):  # noqa: F811
        client = TestClient(create_app(tiny_model, threshold=0.0, docs_enabled=True))
        assert client.get("/openapi.json").status_code == 200

    def test_non_ascii_credentials_are_rejected_not_raised(self):
        # httpx refuses non-ASCII header values, but a raw HTTP client can
        # send them (latin-1 bytes). compare_digest would raise TypeError on
        # str inputs; the app must treat them as a normal 401 credential.
        from autovision.serving.app import _authorized

        class _Request:
            headers = {"authorization": "Bearer caf\u00e9"}

        assert _authorized(_Request(), "s3cret") is False
        assert _authorized(_Request(), "other-caf\u00e9") is False


class TestQueueBackpressure:
    def test_saturated_queue_returns_503(self, tiny_model, monkeypatch):  # noqa: F811
        import autovision.serving.app as app_module

        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            async def slow_threadpool(*args, **kwargs):
                started.set()
                await release.wait()
                return [{}]

            monkeypatch.setattr(app_module, "run_in_threadpool", slow_threadpool)
            app = create_app(
                tiny_model,
                threshold=0.0,
                max_concurrency=1,
                queue_timeout_s=0.05,
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/predict", json={"image": _b64_image(seed=31)})
                )
                await asyncio.wait_for(started.wait(), timeout=2.0)
                second = await client.post(
                    "/predict", json={"image": _b64_image(seed=32)}
                )
                release.set()
                first_response = await first
            return first_response, second

        first_response, second = asyncio.run(scenario())
        assert first_response.status_code == 200
        assert second.status_code == 503
        assert second.headers["Retry-After"] == "1"
        assert "saturated" in second.json()["error"]


class TestObservabilityAuth:
    def test_metrics_and_drift_follow_the_api_key(self, tiny_model):  # noqa: F811
        app = create_app(tiny_model, threshold=0.0, api_key="s3cret")
        client = TestClient(app)
        assert client.get("/metrics").status_code == 401
        assert client.get("/drift").status_code == 401
        headers = {"X-API-Key": "s3cret"}
        assert client.get("/metrics", headers=headers).status_code == 200
        assert client.get("/drift", headers=headers).status_code == 200
        # Kubernetes probes remain unauthenticated.
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200

    def test_observability_auth_can_be_disabled_explicitly(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model,
            threshold=0.0,
            api_key="s3cret",
            protect_observability_endpoints=False,
        )
        assert TestClient(app).get("/metrics").status_code == 200

    def test_observability_failures_are_rate_limited(self, tiny_model):  # noqa: F811
        app = create_app(
            tiny_model,
            threshold=0.0,
            api_key="s3cret",
            rate_limit_per_minute=1,
            rate_limit_burst=1,
        )
        client = TestClient(app)
        assert client.get("/metrics").status_code == 401
        throttled = client.get("/metrics")
        assert throttled.status_code == 429
        assert throttled.headers["Retry-After"]
