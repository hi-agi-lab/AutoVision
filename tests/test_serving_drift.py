# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Prediction-drift monitor: baseline build, PSI status, and /drift wiring."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from tests.test_tracking import tiny_model  # noqa: F401 - fixture reuse

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from autovision.errors import AutoVisionError  # noqa: E402
from autovision.serving import (  # noqa: E402
    DriftBaseline,
    DriftMonitor,
    build_drift_baseline,
    create_app,
    load_drift_baseline,
)


def _records(label: int = 0, score: float = 0.1, count: int = 5):
    return [
        {"labels": np.array([label]), "scores": np.array([score])}
        for _ in range(count)
    ]


def _b64_image(seed: int = 0) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode()


class _FakeEngine:
    backend = "torch"
    artifact = None
    network = None

    def __init__(self, label: int = 0, score: float = 0.1) -> None:
        self.label = label
        self.score = score

    def run_batch(self, images, *, threshold=0.5):
        return [
            {
                "labels": np.array([self.label]),
                "scores": np.array([self.score]),
                "bboxes": np.zeros((1, 4)),
            }
            for _ in images
        ]


class TestBaseline:
    def test_build_and_roundtrip(self, tmp_path):
        baseline = build_drift_baseline(_records(0, 0.1, 10))
        assert baseline.sample_count == 10
        assert baseline.label_counts == {0: 10}
        assert sum(baseline.score_histogram) == 10
        path = baseline.save(tmp_path / "baseline.json")
        loaded = load_drift_baseline(path)
        assert loaded == baseline

    def test_invalid_baselines_fail_loud(self):
        with pytest.raises(AutoVisionError, match="schema_version"):
            DriftBaseline.from_dict({"schema_version": 99})
        with pytest.raises(AutoVisionError, match="score_histogram"):
            DriftBaseline.from_dict(
                {
                    "schema_version": 1, "bins": 2, "sample_count": 1,
                    "label_counts": {"0": 1}, "score_histogram": [1],
                }
            )
        with pytest.raises(AutoVisionError, match="zero predictions"):
            build_drift_baseline([])


class TestMonitor:
    def test_same_distribution_is_ok(self):
        monitor = DriftMonitor(
            build_drift_baseline(_records(0, 0.1, 20)),
            window=50, threshold=0.2, min_samples=10,
        )
        monitor.record_many(_records(0, 0.1, 20))
        report = monitor.snapshot()
        assert report["status"] == "ok"
        assert report["psi_label"] == pytest.approx(0.0, abs=1e-9)
        assert report["psi_score"] == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_reports_drift(self):
        monitor = DriftMonitor(
            build_drift_baseline(_records(0, 0.1, 20)),
            window=50, threshold=0.2, min_samples=10,
        )
        monitor.record_many(_records(1, 0.9, 20))
        report = monitor.snapshot()
        assert report["status"] == "drift"
        assert report["psi_label"] > 0.2
        assert report["psi_score"] > 0.2

    def test_insufficient_data_is_explicit(self):
        monitor = DriftMonitor(
            build_drift_baseline(_records(0, 0.1, 20)),
            min_samples=10,
        )
        monitor.record_many(_records(0, 0.1, 2))
        report = monitor.snapshot()
        assert report["status"] == "insufficient_data"
        assert report["psi_label"] is None

    def test_window_bounds_the_sample(self):
        monitor = DriftMonitor(
            build_drift_baseline(_records(0, 0.1, 20)),
            window=3, min_samples=1,
        )
        monitor.record_many(_records(0, 0.1, 10))
        assert monitor.snapshot()["sample_count"] == 3


class TestDriftEndpoint:
    def test_disabled_without_baseline(self, tiny_model):  # noqa: F811
        client = TestClient(create_app(tiny_model, threshold=0.0))
        assert client.get("/drift").json() == {"enabled": False}

    def test_endpoint_tracks_predictions_and_drift(self, tmp_path):
        baseline = build_drift_baseline(_records(0, 0.1, 10))
        path = baseline.save(tmp_path / "baseline.json")
        engine = _FakeEngine(label=0, score=0.1)
        client = TestClient(
            create_app(
                engine=engine,
                drift_baseline=path,
                drift_min_samples=1,
                threshold=0.0,
            )
        )
        before = client.get("/drift").json()
        assert before["enabled"] is True
        assert before["status"] == "insufficient_data"

        response = client.post("/predict", json={"image": _b64_image(1)})
        assert response.status_code == 200
        after = client.get("/drift").json()
        assert after["sample_count"] == 1
        assert after["status"] == "ok"

        engine.label = 1
        engine.score = 0.9
        assert client.post("/predict", json={"image": _b64_image(2)}).status_code == 200
        # A metrics scrape refreshes the PSI gauges without a /drift call.
        metrics = client.get("/metrics").text
        assert 'autovision_drift_psi{kind="label"}' in metrics
        assert 'autovision_drift_psi{kind="score"}' in metrics
        assert "autovision_drift_detected 1" in metrics
        drifted = client.get("/drift").json()
        assert drifted["status"] == "drift"

    def test_invalid_baseline_fails_at_creation(self):
        with pytest.raises(AutoVisionError, match="schema_version"):
            create_app(
                engine=_FakeEngine(),
                drift_baseline={"schema_version": 99},
            )
