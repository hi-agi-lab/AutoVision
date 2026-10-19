# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Video helpers: frame generator, process_video, sinks and FPS monitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autovision.errors import AutoVisionError
from autovision.inference import (
    FPSMonitor,
    ImageSink,
    VideoSink,
    get_video_frames_generator,
    process_video,
)
from autovision.inference import video as video_module


def _write_video(path: Path, frames: int = 3, size: int = 32) -> None:
    """Write a tiny MJPG AVI; skip when the local OpenCV lacks the codec."""
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (size, size)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment")
    try:
        for index in range(frames):
            writer.write(np.full((size, size, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()


def _read_frames(path: Path) -> int:
    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    return count


class TestFrameGenerator:
    def test_generator_yields_rgb_frames(self, tmp_path):
        """The generator hides the stream object and yields RGB frames."""
        source = tmp_path / "input.avi"
        _write_video(source, frames=3)
        frames = list(get_video_frames_generator(source))
        assert len(frames) == 3
        assert frames[0].shape == (32, 32, 3)

    def test_generator_honours_max_frames(self, tmp_path):
        """max_frames stops decoding early."""
        source = tmp_path / "input.avi"
        _write_video(source, frames=4)
        assert len(list(get_video_frames_generator(source, max_frames=2))) == 2


class TestProcessVideo:
    def test_callback_runs_in_order_and_output_is_written(self, tmp_path):
        """Every decoded frame reaches the callback and the encoded output."""
        source = tmp_path / "input.avi"
        output = tmp_path / "output.mp4"
        _write_video(source, frames=3)
        seen: list[int] = []

        def callback(frame, index):
            seen.append(index)
            return frame

        count = process_video(source, callback, output_path=output)
        assert count == 3
        assert seen == [0, 1, 2]
        assert output.is_file()
        assert _read_frames(output) == 3

    def test_none_callback_result_keeps_the_frame(self, tmp_path):
        """Returning None is a pass-through, not a dropped frame."""
        source = tmp_path / "input.avi"
        _write_video(source, frames=2)
        assert process_video(source, lambda frame, index: None) == 2


class TestSinks:
    def test_video_sink_rejects_shape_mismatch(self, tmp_path):
        """A wrong frame size fails instead of silently writing nothing."""
        output = tmp_path / "out.mp4"
        with pytest.raises(AutoVisionError, match="Frame size mismatch"):
            with VideoSink(output, fps=10.0, frame_size=(32, 32)) as sink:
                sink.write_frame(np.zeros((16, 16, 3), dtype=np.uint8))

    def test_video_sink_rejects_writes_after_close(self, tmp_path):
        """Writing after close is an explicit error."""
        output = tmp_path / "out.mp4"
        sink = VideoSink(output, fps=10.0, frame_size=(32, 32))
        sink.close()
        with pytest.raises(AutoVisionError, match="closed"):
            sink.write_frame(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_image_sink_samples_frames(self, tmp_path):
        """every_n=2 writes indices 0 and 2 only."""
        sink = ImageSink(tmp_path / "frames", every_n=2, image_format="png")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        paths = [sink.write_frame(frame, index=index) for index in range(4)]
        assert [path.name for path in paths if path is not None] == [
            "frame_000000.png",
            "frame_000002.png",
        ]
        assert len(list((tmp_path / "frames").glob("*.png"))) == 2

    def test_image_sink_validates_options(self, tmp_path):
        """Bad sampling intervals and formats are rejected."""
        with pytest.raises(AutoVisionError, match="every_n"):
            ImageSink(tmp_path, every_n=0)
        with pytest.raises(AutoVisionError, match="image_format"):
            ImageSink(tmp_path, image_format="webp")


class TestFPSMonitor:
    def test_fps_uses_the_tick_window(self, monkeypatch):
        """FPS is (samples - 1) / elapsed over the retained timestamps."""
        ticks = iter([0.0, 0.5, 1.0, 1.5])

        class FakeTime:
            @staticmethod
            def perf_counter() -> float:
                return next(ticks)

        monkeypatch.setattr(video_module, "time", FakeTime)
        monitor = FPSMonitor(window=3)
        assert monitor.tick() == 0.0
        assert monitor.tick() == 2.0
        assert monitor.tick() == 2.0
        monitor.reset()
        assert monitor.fps == 0.0

    def test_window_must_be_at_least_two(self):
        """A one-sample window cannot estimate a rate."""
        with pytest.raises(AutoVisionError, match="window"):
            FPSMonitor(window=1)
