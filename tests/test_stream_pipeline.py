# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Threaded stream pipeline: stage accounting, backpressure and failures."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference import run_stream
from autovision.inference.video import redact_source
from autovision.tracking import LineZone, PolygonZone

cv2 = pytest.importorskip("cv2", reason="opencv not installed")


def _clip(tmp_path, frames: int = 5) -> str:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), (index * 30) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


class _Engine:
    backend = "torch"

    def __init__(self, *, short: bool = False, fail: bool = False) -> None:
        self.short = short
        self.fail = fail

    def run_batch(self, images, threshold=0.5):
        if self.fail:
            raise RuntimeError("inference exploded")
        count = len(images) - 1 if self.short else len(images)
        return [
            {
                "bboxes": torch.tensor([[2, 2, 20, 18]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in range(count)
        ]


class _MovingEngine:
    """A consistently tracked box moving down the frame."""

    backend = "torch"

    def __init__(self) -> None:
        self.index = 0

    def run_batch(self, images, threshold=0.5):
        records = []
        for _ in images:
            y = 2.0 + self.index * 3.0
            self.index += 1
            records.append(
                {
                    "bboxes": torch.tensor([[4, y, 20, y + 10]], dtype=torch.float32),
                    "scores": torch.tensor([0.95]),
                    "labels": torch.tensor([0]),
                }
            )
        return records


class _NanEngine(_Engine):
    def run_batch(self, images, threshold=0.5):
        records = super().run_batch(images, threshold=threshold)
        for record in records:
            record["scores"] = torch.tensor([float("nan")])
        return records


class _ClassificationEngine:
    backend = "torch"

    def run_batch(self, images, threshold=0.5):
        return [
            {"scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
            for _ in images
        ]


class TestPipeline:
    def test_stage_timings_and_fps(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(),
            batch_size=2, render=True, prefetch=1, writer_buffer=1,
        )
        assert result.frames == 5
        assert result.fps > 0
        assert result.decode_s > 0
        assert result.infer_s > 0
        assert result.render_s > 0
        assert result.write_s > 0
        assert result.video_path is not None

    def test_render_false_skips_writer(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(), render=False
        )
        assert result.video_path is None
        assert result.write_s == 0.0
        assert result.render_s == 0.0
        assert result.frames == 5

    def test_jsonl_entries_stay_ordered(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_Engine(), batch_size=3
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert [record["frame"] for record in records] == list(range(5))
        assert all("predictions" in record for record in records)

    def test_short_engine_result_raises(self, tmp_path):
        with pytest.raises(AutoVisionError, match="records for"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(short=True)
            )

    def test_engine_failure_propagates(self, tmp_path):
        with pytest.raises(RuntimeError, match="inference exploded"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(fail=True)
            )

    def test_writer_failure_propagates_without_hanging(self, tmp_path, monkeypatch):
        import autovision.inference.stream as stream_module

        class _BadWriter:
            def __init__(self, *args, **kwargs):
                self.path = Path(args[0])

            def write(self, frame):
                raise RuntimeError("encoder exploded")

            def close(self):
                pass

        monkeypatch.setattr(stream_module, "AnnotatedVideoWriter", _BadWriter)
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="encoder exploded"):
            run_stream(
                _clip(tmp_path, frames=60),
                tmp_path / "out",
                engine=_Engine(),
                batch_size=1,
                render=True,
            )
        # The producer must notice the writer failure instead of processing
        # all 60 frames first.
        assert time.perf_counter() - started < 3.0

    def test_tracking_requires_a_detection_result(self, tmp_path):
        with pytest.raises(AutoVisionError, match="requires a detection model"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_ClassificationEngine(),
                track=True,
            )


def test_redact_source_hides_url_credentials():
    assert (
        redact_source("rtsp://user:pass@cam.local:554/live")
        == "rtsp://***@cam.local:554/live"
    )
    assert redact_source("/data/clip.mp4") == "/data/clip.mp4"

    def test_prefetch_and_writer_buffer_validation(self, tmp_path):
        with pytest.raises(AutoVisionError, match="prefetch"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(), prefetch=0
            )
        with pytest.raises(AutoVisionError, match="writer_buffer"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(),
                writer_buffer=0,
            )

    def test_zones_require_tracking(self, tmp_path):
        with pytest.raises(AutoVisionError, match="track=True"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_Engine(),
                line_zones=[LineZone((0, 10), (32, 10))],
            )

    def test_smoothing_requires_tracking(self, tmp_path):
        with pytest.raises(AutoVisionError, match="smooth=True"):
            run_stream(
                _clip(tmp_path), tmp_path / "out", engine=_Engine(), smooth=True
            )

    def test_duplicate_zone_names_are_rejected(self, tmp_path):
        with pytest.raises(AutoVisionError, match="unique"):
            run_stream(
                _clip(tmp_path),
                tmp_path / "out",
                engine=_Engine(),
                track=True,
                line_zones=[LineZone((0, 10), (32, 10), name="z")],
                polygon_zones=[
                    PolygonZone(
                        [(0, 0), (32, 0), (32, 24), (0, 24)], name="z"
                    )
                ],
            )

    def test_line_and_polygon_counts_reach_jsonl_and_result(self, tmp_path):
        line = LineZone((0, 22), (32, 22), name="gate")
        polygon = PolygonZone(
            [(-100, -100), (100, -100), (100, 100), (-100, 100)],
            name="frame",
        )
        result = run_stream(
            _clip(tmp_path, frames=8),
            tmp_path / "out",
            engine=_MovingEngine(),
            track=True,
            line_zones=[line],
            polygon_zones=[polygon],
            render=True,
            batch_size=2,
        )
        assert result.zones["lines"]["gate"]["in"] >= 1
        assert result.zones["polygons"]["frame"]["total"] >= 1
        assert result.video_path is not None and result.video_path.exists()
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert all("zones" in record for record in records)
        assert "gate" in records[-1]["zones"]["lines"]

    def test_smoothing_does_not_change_jsonl_predictions(self, tmp_path):
        result = run_stream(
            _clip(tmp_path),
            tmp_path / "out",
            engine=_MovingEngine(),
            track=True,
            smooth=True,
            render=True,
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert len(records[0]["predictions"]["bboxes"][0]) == 4
        assert result.video_path is not None

    def test_non_finite_scores_become_null(self, tmp_path):
        result = run_stream(
            _clip(tmp_path), tmp_path / "out", engine=_NanEngine(), render=False
        )
        record = json.loads(result.predictions_path.read_text().splitlines()[0])
        assert record["predictions"]["scores"] == [None]

    def test_classification_records_render(self, tmp_path):
        result = run_stream(
            _clip(tmp_path),
            tmp_path / "out",
            engine=_ClassificationEngine(),
            class_names={0: "cat"},
            render=True,
        )
        assert result.frames == 5
        assert result.video_path is not None
