# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-source video/stream inference (roadmap 90d, phase 2).

One reader thread per source feeds a bounded per-source queue; a single
batcher thread builds mixed-source batches and is the only caller of the
inference engine (TensorRT/OpenVINO runners are not thread-safe). Each
source gets its own ``predictions.jsonl`` (+ ``annotated.mp4`` when
rendering); a dropping policy bounds memory when a source outruns the GPU.

Reconnect is for live sources only (``rtsp://`` / ``http(s)://``); a file
source ends at EOF. Backoff is exponential (0.5s, 1s, 2s, ... capped at
10s) with a per-source reconnect budget.

``reader_factory`` is injectable so tests can exercise scheduling,
reconnect and backpressure without cameras or OpenCV; the default reader
reuses :class:`~autovision.inference.video.VideoFrameStream`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlparse

import numpy as np

from autovision.errors import AutoVisionError
from autovision.inference.engine import InferenceEngine
from autovision.inference.video import (
    AnnotatedVideoWriter,
    VideoFrameStream,
    VideoInfo,
    redact_source,
)
from autovision.results import InferenceResult
from autovision.types import PathLike
from autovision.visualize.detection import draw_prediction

__all__ = [
    "MultiStreamResult",
    "MultiStreamRunner",
    "StreamSource",
    "StreamStats",
    "run_multi_stream",
]

_LIVE_SCHEMES = {"rtsp", "rtmp", "http", "https"}

logger = logging.getLogger("autovision.inference.multi_stream")


@dataclass(frozen=True)
class StreamSource:
    """One input; ``reconnect=None`` derives it from the URL scheme."""

    name: str
    url: str
    reconnect: bool | None = None
    max_reconnects: int = 10

    @property
    def is_live(self) -> bool:
        if self.reconnect is not None:
            return bool(self.reconnect)
        return urlparse(self.url).scheme.lower() in _LIVE_SCHEMES


@dataclass
class StreamStats:
    name: str
    url: str
    fps: float = 25.0
    frames_read: int = 0
    frames_written: int = 0
    frames_dropped: int = 0
    frames_failed: int = 0
    reconnects: int = 0
    errors: list[str] = field(default_factory=list)
    jsonl: Path | None = None
    video: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": redact_source(self.url),
            "fps": self.fps,
            "frames_read": self.frames_read,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "frames_failed": self.frames_failed,
            "reconnects": self.reconnects,
            "errors": list(self.errors[-5:]),
            "jsonl": str(self.jsonl) if self.jsonl else None,
            "video": str(self.video) if self.video else None,
        }


@dataclass
class MultiStreamResult:
    out_dir: Path
    sources: dict[str, StreamStats]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "elapsed_s": self.elapsed_s,
            "sources": {
                name: stats.to_dict() for name, stats in sorted(self.sources.items())
            },
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.out_dir / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target

    def summary(self) -> str:
        parts = [
            f"{stats.name}: frames={stats.frames_written} "
            f"failed={stats.frames_failed} dropped={stats.frames_dropped} "
            f"reconnects={stats.reconnects}"
            for stats in sorted(self.sources.values(), key=lambda item: item.name)
        ]
        return (
            f"sources={len(self.sources)} elapsed={self.elapsed_s:.2f}s | "
            + " | ".join(parts)
        )


@dataclass(frozen=True)
class _FramePacket:
    source_index: int
    frame_index: int
    frame: np.ndarray
    timestamp: float


class _ReaderProtocol:
    info: VideoInfo

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _OpenCVReader(_ReaderProtocol):
    def __init__(self, url: str) -> None:
        self._stream = VideoFrameStream(url)
        self.info = self._stream.info

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        return iter(self._stream)

    def close(self) -> None:
        # VideoFrameStream releases in its iterator's finally block.
        return None


class MultiStreamRunner:
    """Run one engine over N sources with per-source artifacts and metrics."""

    def __init__(
        self,
        sources: Sequence[str | StreamSource],
        out_dir: PathLike,
        *,
        engine: Any,
        batch_size: int = 4,
        queue_size: int = 8,
        drop_policy: str = "drop_oldest",
        threshold: float = 0.5,
        render: bool = True,
        class_names: dict[int, str] | None = None,
        mask_format: str = "rle",
        max_json_elements: int = 10_000,
        max_frames_per_source: int | None = None,
        reader_factory: Callable[[str], _ReaderProtocol] | None = None,
    ) -> None:
        if not sources:
            raise AutoVisionError("MultiStreamRunner needs at least one source.")
        if batch_size < 1:
            raise AutoVisionError(f"batch_size must be >= 1, got {batch_size}.")
        if queue_size < 1:
            raise AutoVisionError(f"queue_size must be >= 1, got {queue_size}.")
        if mask_format not in {"rle", "dense", "none"}:
            raise AutoVisionError(
                "mask_format must be 'rle', 'dense', or 'none', "
                f"got {mask_format!r}."
            )
        if max_json_elements < 1:
            raise AutoVisionError(
                f"max_json_elements must be >= 1, got {max_json_elements}."
            )
        if drop_policy not in {"drop_oldest", "block"}:
            raise AutoVisionError(
                f"drop_policy must be 'drop_oldest' or 'block', got {drop_policy!r}."
            )
        if not callable(getattr(engine, "run_batch", None)):
            raise AutoVisionError("engine must expose run_batch(images, threshold=...).")
        if max_frames_per_source is not None and max_frames_per_source < 1:
            raise AutoVisionError("max_frames_per_source must be >= 1.")
        resolved: list[StreamSource] = []
        seen: set[str] = set()
        for index, source in enumerate(sources):
            entry = source if isinstance(source, StreamSource) else StreamSource(
                name=Path(str(source)).stem or f"source{index}", url=str(source)
            )
            if not entry.name or entry.name in seen:
                raise AutoVisionError(
                    f"Stream names must be unique and non-empty; got "
                    f"{entry.name!r} at index {index}."
                )
            if entry.max_reconnects < 0:
                raise AutoVisionError(
                    f"Stream '{entry.name}' has negative max_reconnects."
                )
            seen.add(entry.name)
            resolved.append(entry)
        self.sources = tuple(resolved)
        self.out_dir = Path(out_dir)
        self.engine = engine
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.drop_policy = drop_policy
        self.threshold = threshold
        self.render = render
        self.class_names = dict(class_names or {})
        self.mask_format = mask_format
        self.max_json_elements = max_json_elements
        self.max_frames_per_source = max_frames_per_source
        self.reader_factory = reader_factory or _OpenCVReader

    def run(self, *, max_seconds: float | None = None) -> MultiStreamResult:
        if max_seconds is not None and max_seconds <= 0:
            raise AutoVisionError("max_seconds must be > 0.")
        started = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        queues: list[queue.Queue[Any]] = [
            queue.Queue(maxsize=self.queue_size) for _ in self.sources
        ]
        done = [False for _ in self.sources]
        stop = threading.Event()
        lock = threading.Lock()
        stats = {
            source.name: StreamStats(
                name=source.name, url=redact_source(source.url)
            )
            for source in self.sources
        }
        stats_list = [stats[source.name] for source in self.sources]

        def _inc(stat: StreamStats, field_name: str, amount: int = 1) -> None:
            with lock:
                setattr(stat, field_name, getattr(stat, field_name) + amount)

        def _record_error(stat: StreamStats, message: str) -> None:
            with lock:
                stat.errors.append(message)
                # A persistent failure must not grow the list without bound.
                del stat.errors[:-100]

        def _put(packet: _FramePacket, stat: StreamStats, source_index: int) -> None:
            target = queues[source_index]
            if self.drop_policy == "block":
                try:
                    target.put(packet, timeout=0.2)
                    return
                except queue.Full:
                    _inc(stat, "frames_dropped")
                    return
            try:
                target.put_nowait(packet)
                return
            except queue.Full:
                pass
            try:
                target.get_nowait()
                _inc(stat, "frames_dropped")
            except queue.Empty:
                pass
            try:
                target.put_nowait(packet)
            except queue.Full:
                _inc(stat, "frames_dropped")

        def _reader(source_index: int, source: StreamSource) -> None:
            stat = stats_list[source_index]
            attempts = 0
            try:
                while not stop.is_set():
                    reader: _ReaderProtocol | None = None
                    try:
                        reader = self.reader_factory(source.url)
                        with lock:
                            stat.fps = float(reader.info.fps or 25.0)
                        for _frame_index, frame in reader.frames():
                            if stop.is_set():
                                break
                            if self.max_frames_per_source is not None:
                                with lock:
                                    if stat.frames_read >= self.max_frames_per_source:
                                        break
                            _inc(stat, "frames_read")
                            _put(
                                _FramePacket(
                                    source_index=source_index,
                                    frame_index=int(stat.frames_read) - 1,
                                    frame=frame,
                                    timestamp=time.time(),
                                ),
                                stat,
                                source_index,
                            )
                        if self.max_frames_per_source is not None and stat.frames_read >= self.max_frames_per_source:
                            break
                        if not source.is_live:
                            break
                        raise AutoVisionError("live stream ended")
                    except Exception as error:  # noqa: BLE001 - source fault
                        _record_error(stat, f"{type(error).__name__}: {error}")
                        if stop.is_set() or not source.is_live:
                            break
                        if stat.reconnects >= source.max_reconnects:
                            break
                        _inc(stat, "reconnects")
                        attempts += 1
                        delay = min(10.0, 0.5 * (2 ** min(attempts - 1, 4)))
                        stop.wait(delay)
                    finally:
                        if reader is not None:
                            try:
                                reader.close()
                            except Exception:  # noqa: BLE001 - best effort
                                pass
            finally:
                done[source_index] = True

        threads = [
            threading.Thread(
                target=_reader, args=(index, source), name=f"av-{source.name}", daemon=True
            )
            for index, source in enumerate(self.sources)
        ]
        for thread in threads:
            thread.start()

        jsonl_handles: dict[int, Any] = {}
        videos: dict[int, AnnotatedVideoWriter] = {}
        active = set(range(len(self.sources)))
        try:
            for index, source in enumerate(self.sources):
                source_dir = self.out_dir / source.name
                source_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = source_dir / "predictions.jsonl"
                stats_list[index].jsonl = jsonl_path
                jsonl_handles[index] = open(
                    jsonl_path, "w", encoding="utf-8"
                )
            while not stop.is_set():
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    stop.set()
                    break
                batch: list[_FramePacket] = []
                for index in list(active):
                    try:
                        batch.append(queues[index].get_nowait())
                    except queue.Empty:
                        continue
                if not batch:
                    if all(done[index] for index in active):
                        break
                    time.sleep(0.01)
                    continue
                frames = [packet.frame for packet in batch]
                try:
                    records = self.engine.run_batch(frames, threshold=self.threshold)
                except Exception as error:  # noqa: BLE001 - keep other sources alive
                    logger.warning(
                        "Inference batch failed (%d frames): %s", len(batch), error
                    )
                    for packet in batch:
                        stat = stats_list[packet.source_index]
                        _inc(stat, "frames_failed")
                        _record_error(stat, f"inference failed: {error}")
                    continue
                if len(records) != len(batch):
                    logger.warning(
                        "Engine returned %d records for %d frames",
                        len(records),
                        len(batch),
                    )
                    for packet in batch:
                        stat = stats_list[packet.source_index]
                        _inc(stat, "frames_failed")
                        _record_error(
                            stat,
                            "engine returned "
                            f"{len(records)} records for {len(batch)} frames",
                        )
                    continue
                for packet, record in zip(batch, records):
                    index = packet.source_index
                    stat = stats_list[index]
                    payload = {
                        "source": stat.name,
                        "frame_index": packet.frame_index,
                        "timestamp": packet.timestamp,
                        "predictions": InferenceResult.from_record(record).to_json(
                            masks=self.mask_format,
                            max_elements=self.max_json_elements,
                        ),
                    }
                    jsonl_handles[index].write(json.dumps(payload, allow_nan=False) + "\n")
                    jsonl_handles[index].flush()
                    _inc(stat, "frames_written")
                    if self.render:
                        writer = videos.get(index)
                        height, width = packet.frame.shape[:2]
                        if writer is None or writer.frame_size != (width, height):
                            if writer is not None:
                                writer.close()
                            source_dir = self.out_dir / stat.name
                            writer = AnnotatedVideoWriter(
                                source_dir / "annotated.mp4",
                                fps=stat.fps,
                                frame_size=(width, height),
                            )
                            videos[index] = writer
                            stat.video = writer.path
                        from PIL import Image

                        try:
                            rendered = draw_prediction(
                                Image.fromarray(packet.frame),
                                record,
                                class_names=self.class_names,
                            )
                        except Exception as error:  # noqa: BLE001
                            _record_error(stat, f"render failed: {error}")
                            rendered = packet.frame
                        writer.write(np.asarray(rendered))
                for index in list(active):
                    if done[index] and queues[index].empty():
                        active.discard(index)
                if not active:
                    break
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=5.0)
            for handle in jsonl_handles.values():
                handle.close()
            for writer in videos.values():
                writer.close()
        elapsed = time.perf_counter() - started
        result = MultiStreamResult(
            out_dir=self.out_dir,
            sources=stats,
            elapsed_s=elapsed,
        )
        result.save()
        return result


def _sources(values: Sequence[str | StreamSource] | str) -> list[StreamSource]:
    parts: list[str | StreamSource]
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
    else:
        parts = list(values)
    result: list[StreamSource] = []
    for index, value in enumerate(parts):
        if isinstance(value, StreamSource):
            result.append(value)
        else:
            result.append(
                StreamSource(
                    name=Path(str(value)).stem or f"source{index}",
                    url=str(value),
                )
            )
    return result


def run_multi_stream(
    sources: Sequence[str | StreamSource] | str,
    out_dir: PathLike,
    *,
    model: PathLike | None = None,
    backend: str = "torch",
    artifact: PathLike | None = None,
    device: str | None = None,
    providers: Sequence[str] | None = None,
    openvino_device: str = "CPU",
    **kwargs: Any,
) -> MultiStreamResult:
    """Convenience wrapper mirroring :func:`run_stream` for N sources.

    ``sources`` is a comma-separated string or a sequence; each entry gets
    a name from its file stem (provide :class:`StreamSource` for explicit
    names). ``kwargs`` are the runner scheduling/render options.
    """
    if model is None:
        raise AutoVisionError("run_multi_stream needs model= (or build a runner).")
    engine = InferenceEngine(
        model,
        backend=backend,
        artifact=artifact,
        device=device,
        providers=providers,
        openvino_device=openvino_device,
    )
    runner = MultiStreamRunner(_sources(sources), out_dir, engine=engine, **kwargs)
    return runner.run()
