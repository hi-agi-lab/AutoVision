# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Video / stream I/O for the reference inference runtime.

OpenCV's FFmpeg backend reads local files and network streams (``rtsp://``,
``http://``) through one interface; frames are handed to the runtime as
RGB ``uint8`` ``(H, W, 3)`` arrays — the same channel order the training
data layer's image readers produce, so pre/post-processing stays identical
between file, camera, and stream sources.

OpenCV is the ``video`` extra (``pip install -e '.[video]'``); importing
this module without it succeeds, use fails with guidance.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from autovision.errors import AutoVisionError
from autovision.types import PathLike

__all__ = [
    "AnnotatedVideoWriter",
    "FPSMonitor",
    "ImageSink",
    "VideoFrameStream",
    "VideoInfo",
    "VideoSink",
    "get_video_frames_generator",
    "process_video",
    "redact_source",
]


def redact_source(source: PathLike) -> str:
    """Return a log-safe source string with URL credentials removed."""
    text = str(source)
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.scheme or parts.hostname is None:
        return text
    netloc = parts.hostname
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    if parts.username is not None or parts.password is not None:
        netloc = f"***@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise AutoVisionError(
            "Video/stream inference needs OpenCV: pip install -e '.[video]'"
        ) from error
    return cv2


@dataclass(frozen=True)
class VideoInfo:
    """Source properties; ``fps``/``frame_count`` are 0 when unknown (live streams)."""

    width: int
    height: int
    fps: float
    frame_count: int


class VideoFrameStream:
    """Iterate a video file or network stream as ``(index, RGB frame)``.

    ``source`` is anything OpenCV opens: a path, ``rtsp://``,
    ``http://``, or a camera index. Iteration ends at the source's last
    frame, at ``max_frames``, or on read failure (live streams: a dropped
    read ends the run — reconnect handling is out of scope). The
    ``*_timeout_ms`` knobs are best-effort FFmpeg-backend properties for
    network sources; local files ignore them.
    """

    def __init__(
        self,
        source: PathLike,
        *,
        max_frames: int | None = None,
        open_timeout_ms: int | None = 10_000,
        read_timeout_ms: int | None = 5_000,
    ) -> None:
        cv2 = _require_cv2()
        self.source = str(source)
        self.max_frames = max_frames
        self._capture = cv2.VideoCapture()
        # Timeouts must be set on the empty capture, before open().
        for prop, value in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout_ms),
            ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_ms),
        ):
            if value is not None and hasattr(cv2, prop):
                self._capture.set(getattr(cv2, prop), value)
        self._capture.open(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            raise AutoVisionError(
                f"Could not open video source '{redact_source(self.source)}' — "
                "check the path or stream URL (RTSP/HTTP sources need OpenCV's "
                "FFmpeg backend)."
            )
        self.info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        # Only local files have a trustworthy frame count. A short read before
        # that count means truncation/corruption, not a normal EOF.
        self._strict_frame_count = (
            self.info.frame_count > 0 and Path(self.source).is_file()
        )

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        cv2 = _require_cv2()
        index = 0
        try:
            while self.max_frames is None or index < self.max_frames:
                ok, frame = self._capture.read()
                if not ok:
                    if self._strict_frame_count and index < self.info.frame_count:
                        raise AutoVisionError(
                            "Video decode failed at frame "
                            f"{index}/{self.info.frame_count} for "
                            f"'{redact_source(self.source)}'; the file may be "
                            "truncated or corrupt."
                        )
                    break
                yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            self._capture.release()


class AnnotatedVideoWriter:
    """Write RGB frames to an MP4 file (OpenCV ``mp4v`` codec)."""

    def __init__(
        self, path: PathLike, *, fps: float, frame_size: tuple[int, int]
    ) -> None:
        cv2 = _require_cv2()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps and fps > 0 else 25.0,
            frame_size,
        )
        if not self._writer.isOpened():
            raise AutoVisionError(
                f"Could not open an MP4 writer for '{self.path}' (codec mp4v)."
            )

    def write(self, frame: Any) -> None:
        cv2 = _require_cv2()
        if not self._writer.isOpened():
            raise AutoVisionError(
                f"Video writer for '{self.path}' is closed; frame not written."
            )
        try:
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except cv2.error as error:
            raise AutoVisionError(
                f"Video writer for '{self.path}' failed: {error}"
            ) from error

    def close(self) -> None:
        self._writer.release()


def get_video_frames_generator(
    source: PathLike,
    *,
    max_frames: int | None = None,
    open_timeout_ms: int | None = 10_000,
    read_timeout_ms: int | None = 5_000,
) -> Iterator[Any]:
    """Yield RGB frames from a file, camera index or network stream."""
    for _, frame in VideoFrameStream(
        source,
        max_frames=max_frames,
        open_timeout_ms=open_timeout_ms,
        read_timeout_ms=read_timeout_ms,
    ):
        yield frame


class VideoSink:
    """Context-managed RGB video writer (MP4, ``mp4v`` codec).

    The writer opens eagerly so a bad path or codec fails before the first
    inference call instead of silently dropping frames at the end.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        fps: float = 25.0,
        frame_size: tuple[int, int],
    ) -> None:
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise AutoVisionError(
                f"frame_size must be positive, got {frame_size!r}."
            )
        self._writer = AnnotatedVideoWriter(
            path, fps=fps, frame_size=self.frame_size
        )
        self._closed = False

    def write_frame(self, frame: Any) -> None:
        """Write one RGB frame; a shape mismatch is an error, not a skip."""
        if self._closed:
            raise AutoVisionError("Cannot write to a closed VideoSink.")
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise AutoVisionError(
                "Frame size mismatch: VideoSink expects "
                f"{self.frame_size}, got {(width, height)}."
            )
        self._writer.write(frame)

    def close(self) -> None:
        """Release the underlying writer; safe to call more than once."""
        if not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ImageSink:
    """Save selected RGB frames as JPEG/PNG files in a directory.

    ``index`` is the source frame number; when omitted the sink keeps its own
    counter. Only frames where ``index % every_n == 0`` are written.
    """

    def __init__(
        self,
        directory: PathLike,
        *,
        every_n: int = 1,
        image_format: str = "jpg",
        quality: int = 95,
        prefix: str = "frame",
    ) -> None:
        if every_n < 1:
            raise AutoVisionError(f"every_n must be >= 1, got {every_n!r}.")
        normalized = image_format.lower()
        if normalized not in {"jpg", "jpeg", "png"}:
            raise AutoVisionError(
                "image_format must be 'jpg', 'jpeg' or 'png', "
                f"got {image_format!r}."
            )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every_n = int(every_n)
        self.extension = "jpg" if normalized in {"jpg", "jpeg"} else "png"
        self.quality = min(max(int(quality), 1), 100)
        self.prefix = prefix
        self._index = 0

    def write_frame(self, frame: Any, *, index: int | None = None) -> Path | None:
        """Write ``frame`` when it falls on the sampling interval."""
        from PIL import Image

        current = self._index if index is None else int(index)
        if index is None:
            self._index += 1
        else:
            self._index = max(self._index, current + 1)
        if current % self.every_n != 0:
            return None
        path = self.directory / f"{self.prefix}_{current:06d}.{self.extension}"
        image = Image.fromarray(frame)
        if self.extension == "jpg":
            image.save(path, quality=self.quality)
        else:
            image.save(path)
        return path

    def __enter__(self) -> "ImageSink":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class FPSMonitor:
    """Moving-window frames-per-second estimate from ``tick()`` calls."""

    def __init__(self, window: int = 60) -> None:
        if window < 2:
            raise AutoVisionError(f"window must be >= 2, got {window!r}.")
        self.window = int(window)
        self._timestamps: deque[float] = deque(maxlen=self.window)

    def tick(self) -> float:
        """Record one frame and return the current FPS estimate."""
        self._timestamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        """FPS over the retained window (0.0 until two ticks exist)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear the window."""
        self._timestamps.clear()


def process_video(
    source: PathLike,
    callback: Callable[[Any, int], Any],
    *,
    output_path: PathLike | None = None,
    max_frames: int | None = None,
    fps: float | None = None,
    frame_size: tuple[int, int] | None = None,
) -> int:
    """Decode ``source``, call ``callback(frame, index)`` and optionally encode.

    ``callback`` returns the frame to write; returning ``None`` keeps the
    decoded frame. The callback runs in decode order on the calling thread,
    so stateful annotators (trace/heatmap/zones) stay ordered. Returns the
    number of decoded frames. Encoding uses the source FPS and resolution
    unless ``fps`` / ``frame_size`` override them.
    """
    stream = VideoFrameStream(source, max_frames=max_frames)
    info = stream.info
    sink: VideoSink | None = None
    if output_path is not None and info.width > 0 and info.height > 0:
        sink = VideoSink(
            output_path,
            fps=fps or info.fps or 25.0,
            frame_size=frame_size or (info.width, info.height),
        )
    frames = 0
    try:
        for index, frame in stream:
            processed = callback(frame, index)
            if processed is None:
                processed = frame
            if output_path is not None and sink is None:
                height, width = processed.shape[:2]
                sink = VideoSink(
                    output_path,
                    fps=fps or info.fps or 25.0,
                    frame_size=frame_size or (width, height),
                )
            if sink is not None:
                sink.write_frame(processed)
            frames += 1
    finally:
        if sink is not None:
            sink.close()
    return frames
