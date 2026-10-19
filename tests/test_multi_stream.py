# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-source runtime: shared batching, per-source artifacts, reconnect."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference.multi_stream import (
    MultiStreamRunner,
    StreamSource,
)
from autovision.inference.video import VideoInfo


def _frames(count: int, value: int = 0) -> list[np.ndarray]:
    return [
        np.full((48, 64, 3), (value + index) % 255, dtype=np.uint8)
        for index in range(count)
    ]


class FakeReader:
    def __init__(self, frames=None, *, fail: bool = False, fps: float = 10.0):
        self.info = VideoInfo(
            width=64, height=48, fps=fps, frame_count=len(frames or [])
        )
        self._frames = frames or []
        self._fail = fail
        self.closed = False

    def frames(self):
        if self._fail:
            raise RuntimeError("reader failed")
        for index, frame in enumerate(self._frames):
            yield index, frame

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self):
        self.batches: list[int] = []

    def run_batch(self, images, *, threshold=0.5):
        self.batches.append(len(images))
        return [
            {
                "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


class FailingEngine:
    def run_batch(self, images, *, threshold=0.5):
        raise RuntimeError("inference exploded")


def test_two_sources_write_per_source_jsonl(tmp_path):
    readers = {
        "file1.mp4": FakeReader(_frames(4, 10)),
        "file2.mp4": FakeReader(_frames(4, 100)),
    }
    runner = MultiStreamRunner(
        [StreamSource("cam1", "file1.mp4"), StreamSource("cam2", "file2.mp4")],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=2,
        queue_size=4,
        render=False,
        reader_factory=lambda url: readers[url],
    )
    result = runner.run()
    assert result.sources["cam1"].frames_written == 4
    assert result.sources["cam2"].frames_written == 4
    for name in ("cam1", "cam2"):
        path = tmp_path / "out" / name / "predictions.jsonl"
        assert len(path.read_text().splitlines()) == 4
        record = json.loads(path.read_text().splitlines()[0])
        assert record["source"] == name
        assert record["predictions"]["labels"] == [0]
        assert (tmp_path / "out" / "summary.json").is_file()


def test_max_frames_per_source(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        max_frames_per_source=2,
        reader_factory=lambda url: FakeReader(_frames(5)),
    )
    result = runner.run()
    assert result.sources["cam"].frames_read == 2
    assert result.sources["cam"].frames_written == 2


def test_live_source_reconnects(tmp_path):
    calls: list[int] = []

    def factory(url: str):
        calls.append(1)
        if len(calls) == 1:
            return FakeReader(fail=True)
        return FakeReader(_frames(3, 50))

    runner = MultiStreamRunner(
        [StreamSource("live", "rtsp://fake", reconnect=True, max_reconnects=1)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        reader_factory=factory,
    )
    result = runner.run()
    stats = result.sources["live"]
    assert stats.reconnects == 1
    assert stats.frames_written == 3
    assert stats.errors


def test_inference_failures_are_counted_per_frame(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FailingEngine(),
        batch_size=2,
        render=False,
        reader_factory=lambda url: FakeReader(_frames(3)),
    )
    result = runner.run()
    stats = result.sources["cam"]
    assert stats.frames_written == 0
    assert stats.frames_failed == 3
    assert stats.errors
    assert result.to_dict()["sources"]["cam"]["frames_failed"] == 3
    assert "failed=3" in result.summary()


def test_stream_stats_redact_credentials(tmp_path):
    runner = MultiStreamRunner(
        [
            StreamSource(
                "cam", "rtsp://user:secret@cam.local:554/live", reconnect=False
            )
        ],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=False,
        reader_factory=lambda url: FakeReader(_frames(1)),
    )
    result = runner.run()
    url = result.to_dict()["sources"]["cam"]["url"]
    assert url == "rtsp://***@cam.local:554/live"
    assert "secret" not in url


def test_file_source_does_not_reconnect(tmp_path):
    runner = MultiStreamRunner(
        [StreamSource("file", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        render=False,
        reader_factory=lambda url: FakeReader(fail=True),
    )
    result = runner.run()
    stats = result.sources["file"]
    assert stats.reconnects == 0
    assert stats.frames_written == 0
    assert stats.errors


def test_render_writes_annotated_video(tmp_path):
    pytest.importorskip("cv2", reason="opencv not installed")
    runner = MultiStreamRunner(
        [StreamSource("cam", "file.mp4", reconnect=False)],
        tmp_path / "out",
        engine=FakeEngine(),
        batch_size=1,
        render=True,
        reader_factory=lambda url: FakeReader(_frames(3)),
    )
    result = runner.run()
    video = result.sources["cam"].video
    assert video is not None and Path(video).is_file()
    assert Path(video).stat().st_size > 0


def test_validation_and_queue_drop_policy(tmp_path):
    with pytest.raises(AutoVisionError, match="at least one source"):
        MultiStreamRunner([], tmp_path, engine=FakeEngine())
    with pytest.raises(AutoVisionError, match="unique"):
        MultiStreamRunner(
            [StreamSource("x", "a"), StreamSource("x", "b")], tmp_path,
            engine=FakeEngine(),
        )
    with pytest.raises(AutoVisionError, match="drop_policy"):
        MultiStreamRunner(
            [StreamSource("x", "a")], tmp_path, engine=FakeEngine(),
            drop_policy="bogus",
        )
    with pytest.raises(AutoVisionError, match="run_batch"):
        MultiStreamRunner([StreamSource("x", "a")], tmp_path, engine=object())
