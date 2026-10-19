# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Stream inference run: video/RTSP -> predictions JSONL (+ annotated MP4).

The loop is deliberately thin: :class:`~autovision.inference.engine.InferenceEngine`
runs the network's own pre/post-processing per batch, records land in
``predictions.jsonl`` (one line per frame, boxes in original-frame pixels),
and rendered frames go to ``annotated.mp4`` when ``render=True`` and the
task emits detection boxes.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from autovision.errors import AutoVisionError
from autovision.inference.engine import InferenceEngine
from autovision.inference.video import AnnotatedVideoWriter, VideoFrameStream
from autovision.results import InferenceResult
from autovision.tracking.analytics import LineZone, PolygonZone, TrackSmoother
from autovision.tracking.byte_track import Track
from autovision.types import PathLike
from autovision.visualize.zones import draw_zones

__all__ = ["StreamRunResult", "run_stream"]

logger = logging.getLogger("autovision")


@dataclass(frozen=True)
class StreamRunResult:
    """Artifacts + counters of one :func:`run_stream` call."""

    out_dir: Path
    predictions_path: Path
    video_path: Path | None
    frames: int
    elapsed_s: float
    backend: str
    #: Number of distinct track ids over the run (0 when tracking is off).
    tracks: int = 0
    #: Wall-clock seconds spent decoding, inferring, rendering and encoding.
    decode_s: float = 0.0
    infer_s: float = 0.0
    render_s: float = 0.0
    write_s: float = 0.0
    #: Cumulative line-zone and current polygon-zone counts.
    zones: dict[str, Any] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """End-to-end throughput (decoded frames per wall-clock second)."""
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def summary(self) -> str:
        """One-line run summary for the CLI."""
        video = f", video={self.video_path}" if self.video_path else ""
        tracked = f", tracks={self.tracks}" if self.tracks else ""
        return (
            f"frames={self.frames} backend={self.backend} "
            f"fps={self.fps:.1f} elapsed={self.elapsed_s:.2f}s "
            f"jsonl={self.predictions_path}{video}{tracked}"
        )


def _render_frame(
    frame: Any,
    record: dict[str, Any],
    class_names: dict[int, str] | None,
    tracks: list[Any] | None = None,
    track_class_names: dict[int, str] | None = None,
) -> Any:
    """Overlay the result's visual layers on one RGB frame; pass-through otherwise.

    Detection boxes, oriented polygons, pose keypoints, masks and depth maps
    are all rendered through :func:`draw_prediction`; an unknown record keeps
    the original frame.
    """
    if not any(
        key in record
        for key in (
            "bboxes",
            "polygons",
            "keypoints",
            "mask",
            "masks",
            "depth",
            "labels",
        )
    ):
        return frame
    from PIL import ImageDraw

    from autovision.visualize.detection import draw_prediction

    tensor = torch.from_numpy(frame).permute(2, 0, 1)
    rendered = draw_prediction(tensor, record, class_names=class_names)
    if tracks:
        draw = ImageDraw.Draw(rendered)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            label_id = track.class_id
            name = (track_class_names or {}).get(label_id, str(label_id))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=2)
            draw.text(
                (x1 + 2, max(0.0, y2 - 12)),
                f"#{track.track_id} {name} {track.score:.2f}",
                fill=(255, 200, 0),
            )
    return np.asarray(rendered)


def _finite_or_none(value: float) -> float | None:
    """JSON-safe float: non-finite values become ``null``."""
    number = float(value)
    return number if math.isfinite(number) else None


def _as_numpy(value: Any) -> np.ndarray:
    """Tensor/array/list -> NumPy array (no copy for NumPy inputs)."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _track_record(
    record: dict[str, Any],
    tracker: Any,
    class_names: dict[int, str] | None,
) -> tuple[list[Any] | None, dict[int, str] | None]:
    """Feed one detection record to ByteTrack; return tracks and label map."""
    if tracker is None or "bboxes" not in record:
        return None, class_names
    boxes = _as_numpy(record["bboxes"]).reshape(-1, 4)
    scores = record.get("scores")
    labels = record.get("labels")
    score_array = (
        np.ones(boxes.shape[0], dtype=np.float32)
        if scores is None
        else _as_numpy(scores).reshape(-1)
    )
    label_array = None if labels is None else _as_numpy(labels).reshape(-1)
    tracks = tracker.update(boxes, score_array, label_array)
    if label_array is None:
        # ByteTrack defaults to class 0 when no labels are supplied; expose
        # "unknown" (-1) instead of fabricating a real class id.
        tracks = [
            Track(
                track_id=track.track_id,
                bbox=track.bbox,
                score=track.score,
                class_id=-1,
                state=track.state,
                age=track.age,
            )
            for track in tracks
        ]
    # No class ids in this record — label the tracks with the bare id instead
    # of a fabricated class-0 name.
    return tracks, (class_names if label_array is not None else None)


def _safe_render(
    frame: np.ndarray,
    record: dict[str, Any],
    class_names: dict[int, str] | None,
    tracks: list[Any] | None,
    track_names: dict[int, str] | None,
) -> np.ndarray:
    """Render one frame, falling back to the source frame on a bad record."""
    try:
        return _render_frame(frame, record, class_names, tracks, track_names)
    except Exception as error:  # noqa: BLE001 - rendering must not kill the run
        logger.warning("Frame rendering failed (%s); writing the source frame.", error)
        return frame


def run_stream(
    source: PathLike,
    out_dir: PathLike,
    *,
    engine: InferenceEngine | None = None,
    model: PathLike | None = None,
    backend: str = "torch",
    artifact: PathLike | None = None,
    device: str | None = None,
    providers: Sequence[str] | None = None,
    openvino_device: str = "CPU",
    batch_size: int = 1,
    threshold: float = 0.5,
    render: bool = True,
    max_frames: int | None = None,
    class_names: dict[int, str] | None = None,
    mask_format: str = "rle",
    max_json_elements: int = 10_000,
    prefetch: int = 8,
    writer_buffer: int = 8,
    line_zones: Sequence[LineZone] | None = None,
    polygon_zones: Sequence[PolygonZone] | None = None,
    smooth: bool = False,
    track: bool = False,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
) -> StreamRunResult:
    """Run inference over a video file / RTSP stream and persist artifacts.

    Pass either a ready ``engine`` or ``model`` (+ backend selection, where
    ``onnx``/``tensorrt``/``openvino`` also need ``artifact``). Writes
    ``out_dir/predictions.jsonl`` always, and ``out_dir/annotated.mp4``
    when ``render`` and the task's predictions carry boxes.

    ``track=True`` (roadmap A3) runs a :class:`~autovision.tracking.
    BYTETracker` over the detections: each frame's JSONL entry gains
    ``tracks`` (``track_id`` / ``bbox`` / ``score`` / ``class_id`` /
    ``age``) and the rendered overlays carry the id — ids are stable within
    one run, in stream order (batch aggregation does not reorder frames).
    The tracker receives the source's real frame rate, so ``track_buffer``
    keeps its documented meaning (frames at that rate).

    The loop is a three-stage pipeline: a reader thread decodes into a
    bounded queue (``prefetch``), the caller thread batches/infers and
    updates tracker state in frame order, and a writer thread encodes the
    rendered frames (``writer_buffer``). ``StreamRunResult`` reports the
    per-stage seconds and end-to-end FPS.

    ``line_zones`` / ``polygon_zones`` (``track=True`` required) add
    cumulative crossing counts and current in-zone counts to every JSONL
    entry and to the rendered video. ``smooth=True`` averages each track's
    box over recent frames for rendering; the JSONL predictions stay raw.
    ``mask_format`` is ``"rle"`` (default), ``"dense"`` or ``"none"``.
    """
    if batch_size < 1:
        raise AutoVisionError(f"batch_size must be >= 1, got {batch_size}.")
    if mask_format not in {"rle", "dense", "none"}:
        raise AutoVisionError(
            "mask_format must be 'rle', 'dense', or 'none', "
            f"got {mask_format!r}."
        )
    if max_json_elements < 1:
        raise AutoVisionError(
            f"max_json_elements must be >= 1, got {max_json_elements}."
        )
    if prefetch < 1:
        raise AutoVisionError(f"prefetch must be >= 1, got {prefetch}.")
    if writer_buffer < 1:
        raise AutoVisionError(
            f"writer_buffer must be >= 1, got {writer_buffer}."
        )
    if engine is None:
        if model is None:
            raise AutoVisionError("run_stream needs either engine= or model=.")
        engine = InferenceEngine(
            model,
            backend=backend,
            artifact=artifact,
            device=device,
            providers=providers,
            openvino_device=openvino_device,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    started = time.perf_counter()
    stream = VideoFrameStream(source, max_frames=max_frames)

    tracker = None
    if track:
        from autovision.tracking import BYTETracker

        # ``track_buffer`` counts frames AT the source frame rate.
        fps = stream.info.fps if stream.info.fps and stream.info.fps > 0 else 30.0
        tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            frame_rate=max(1, int(round(fps))),
        )

    line_zone_list = list(line_zones or ())
    polygon_zone_list = list(polygon_zones or ())
    if (line_zone_list or polygon_zone_list) and not track:
        raise AutoVisionError(
            "line_zones/polygon_zones require track=True (zones count tracks)."
        )
    if smooth and not track:
        raise AutoVisionError("smooth=True requires track=True (track smoothing).")
    zone_names = [zone.name for zone in line_zone_list] + [
        zone.name for zone in polygon_zone_list
    ]
    if len(zone_names) != len(set(zone_names)):
        raise AutoVisionError(
            "Zone names must be unique across line_zones and polygon_zones."
        )
    smoother = TrackSmoother() if smooth else None

    stats = {"decode_s": 0.0, "infer_s": 0.0, "render_s": 0.0, "write_s": 0.0}
    read_queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
        maxsize=prefetch
    )
    write_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=writer_buffer)
    stop_event = threading.Event()
    writer_stop = threading.Event()
    writer_done = threading.Event()
    reader_error: list[BaseException] = []
    writer_error: list[BaseException] = []

    def reader_worker() -> None:
        iterator = None
        try:
            iterator = iter(stream)
            while not stop_event.is_set():
                read_started = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                stats["decode_s"] += time.perf_counter() - read_started
                while not stop_event.is_set():
                    try:
                        read_queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:  # noqa: BLE001 - re-raised by the caller
            reader_error.append(error)
        finally:
            if iterator is not None:
                try:
                    # Release the VideoCapture even when the main thread stops
                    # early.
                    iterator.close()
                except BaseException as error:  # noqa: BLE001
                    reader_error.append(error)
            # Always make the main loop's blocking get() wake up; the caller
            # drains this queue in its own finally block.
            sentinel_deadline = time.monotonic() + 5.0
            while time.monotonic() < sentinel_deadline:
                try:
                    read_queue.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            else:
                logger.warning("Could not enqueue the reader sentinel.")

    def writer_worker() -> None:
        writer_obj: AnnotatedVideoWriter | None = None
        try:
            while True:
                try:
                    frame = write_queue.get(timeout=0.1)
                except queue.Empty:
                    if writer_stop.is_set():
                        break
                    continue
                if writer_obj is None:
                    height, width = frame.shape[:2]
                    writer_obj = AnnotatedVideoWriter(
                        out_dir / "annotated.mp4",
                        fps=stream.info.fps,
                        frame_size=(width, height),
                    )
                write_started = time.perf_counter()
                writer_obj.write(frame)
                stats["write_s"] += time.perf_counter() - write_started
        except BaseException as error:  # noqa: BLE001 - re-raised by the caller
            writer_error.append(error)
            # Keep draining until the producer stops, so it cannot block on a
            # full queue after the writer has failed.
            while not writer_stop.is_set():
                try:
                    write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
        finally:
            if writer_obj is not None:
                writer_obj.close()
            writer_done.set()

    frames_done = 0
    track_ids: set[int] = set()
    buffer: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        nonlocal frames_done
        if not buffer:
            return
        if writer_error:
            raise writer_error[0]
        tensors = [torch.from_numpy(frame).permute(2, 0, 1) for _, frame in buffer]
        infer_started = time.perf_counter()
        records = engine.run_batch(tensors, threshold=threshold)
        stats["infer_s"] += time.perf_counter() - infer_started
        if len(records) != len(buffer):
            raise AutoVisionError(
                f"engine returned {len(records)} records for {len(buffer)} frames."
            )
        if tracker is not None:
            for record in records:
                if "bboxes" not in record:
                    raise AutoVisionError(
                        "track=True requires a detection model that returns "
                        "'bboxes'; this result has no boxes. Disable tracking "
                        "or use a detection/segmentation model."
                    )
        for (index, frame), record in zip(buffer, records):
            entry: dict[str, Any] = {"frame": index}
            if stream.info.fps and stream.info.fps > 0:
                entry["time_s"] = round(index / stream.info.fps, 4)
            entry["predictions"] = InferenceResult.from_record(record).to_json(
                masks=mask_format, max_elements=max_json_elements
            )
            tracks, track_names = _track_record(record, tracker, class_names)
            render_tracks = tracks
            if tracks is not None:
                track_ids.update(track.track_id for track in tracks)
                if smoother is not None:
                    render_tracks = smoother.update(tracks)
                if line_zone_list or polygon_zone_list:
                    entry["zones"] = {
                        "lines": {
                            zone.name: zone.update(tracks).as_dict()
                            for zone in line_zone_list
                        },
                        "polygons": {
                            zone.name: zone.update(tracks).as_dict()
                            for zone in polygon_zone_list
                        },
                    }
                entry["tracks"] = [
                    {
                        "track_id": track.track_id,
                        "bbox": [_finite_or_none(value) for value in track.bbox],
                        "score": _finite_or_none(track.score),
                        "class_id": track.class_id,
                        "age": track.age,
                    }
                    for track in tracks
                ]
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
            if render:
                render_started = time.perf_counter()
                rendered = _safe_render(
                    frame, record, class_names, render_tracks, track_names
                )
                if line_zone_list or polygon_zone_list:
                    try:
                        from PIL import Image

                        canvas = (
                            Image.fromarray(rendered)
                            if isinstance(rendered, np.ndarray)
                            else rendered
                        )
                        rendered = np.asarray(
                            draw_zones(canvas, line_zone_list, polygon_zone_list)
                        )
                    except Exception as error:  # noqa: BLE001 - keep the frame
                        logger.warning(
                            "Zone rendering failed (%s); writing the frame without "
                            "zone overlays.",
                            error,
                        )
                stats["render_s"] += time.perf_counter() - render_started
                write_queue.put(rendered)
                if writer_error:
                    raise writer_error[0]
            frames_done += 1
        buffer.clear()

    reader_thread = threading.Thread(
        target=reader_worker, name="autovision-reader", daemon=True
    )
    writer_thread = (
        threading.Thread(
            target=writer_worker, name="autovision-writer", daemon=True
        )
        if render
        else None
    )

    writer_timed_out = False
    with predictions_path.open("w", encoding="utf-8") as handle:
        reader_thread.start()
        if writer_thread is not None:
            writer_thread.start()
        try:
            while True:
                if writer_error:
                    raise writer_error[0]
                try:
                    item = read_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                buffer.append(item)
                if len(buffer) >= batch_size:
                    flush()
            flush()
            if writer_error:
                raise writer_error[0]
            if reader_error:
                raise reader_error[0]
        finally:
            stop_event.set()
            # A reader parked on a full queue must be drained before join;
            # bound the drain so a wedged decoder cannot hang shutdown.
            drain_deadline = time.monotonic() + 10.0
            while (
                reader_thread.is_alive()
                and time.monotonic() < drain_deadline
            ):
                try:
                    read_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
            reader_thread.join(timeout=1.0)
            if reader_thread.is_alive():
                logger.warning("Reader thread did not stop within 10s.")
            if writer_thread is not None:
                # The writer exits after draining once stop is set; no
                # sentinel is needed, so a full queue cannot block shutdown.
                writer_stop.set()
                if not writer_done.wait(timeout=30.0):
                    writer_timed_out = True
                writer_thread.join(timeout=1.0)
                if writer_thread.is_alive():
                    logger.warning("Writer thread did not stop within 30s.")

    if writer_error:
        raise writer_error[0]
    if writer_timed_out:
        raise AutoVisionError(
            "Video writer did not finish within 30s; the output may be "
            "incomplete. Increase writer_buffer or lower the render load."
        )

    elapsed = time.perf_counter() - started
    video_candidate = out_dir / "annotated.mp4"
    video_path: Path | None = None
    if render and frames_done and video_candidate.exists():
        video_path = video_candidate
    elif render:
        video_candidate.unlink(missing_ok=True)
    zone_summary: dict[str, Any] = {}
    if line_zone_list or polygon_zone_list:
        zone_summary = {
            "lines": {
                zone.name: zone.counts.as_dict() for zone in line_zone_list
            },
            "polygons": {
                zone.name: zone.counts.as_dict() for zone in polygon_zone_list
            },
        }
    logger.info(
        "Stream inference finished: %d frames in %.2fs (%.1f fps, %s).",
        frames_done,
        elapsed,
        frames_done / elapsed if elapsed > 0 else 0.0,
        predictions_path,
    )
    return StreamRunResult(
        out_dir=out_dir,
        predictions_path=predictions_path,
        video_path=video_path,
        frames=frames_done,
        elapsed_s=elapsed,
        backend=engine.backend,
        tracks=len(track_ids),
        decode_s=stats["decode_s"],
        infer_s=stats["infer_s"],
        render_s=stats["render_s"],
        write_s=stats["write_s"],
        zones=zone_summary,
    )
