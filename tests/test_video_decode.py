# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Truncated/corrupt local videos must not be reported as a clean EOF."""

import pytest

import autovision.inference.video as video_module
from autovision.errors import AutoVisionError


class _FakeCapture:
    def set(self, *args, **kwargs):
        return True

    def open(self, source):
        return True

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            _FakeCV2.CAP_PROP_FRAME_WIDTH: 10,
            _FakeCV2.CAP_PROP_FRAME_HEIGHT: 10,
            _FakeCV2.CAP_PROP_FPS: 25.0,
            _FakeCV2.CAP_PROP_FRAME_COUNT: 5,
        }.get(prop, 0)

    def read(self):
        return False, None

    def release(self):
        return None


class _FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    VideoCapture = _FakeCapture


def test_short_read_before_frame_count_raises(tmp_path, monkeypatch):
    path = tmp_path / "truncated.mp4"
    path.write_bytes(b"pretend-video")
    monkeypatch.setattr(video_module, "_require_cv2", lambda: _FakeCV2)
    with pytest.raises(AutoVisionError, match="decode failed"):
        list(video_module.VideoFrameStream(path))
