# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference runtime: torch / ONNX / TensorRT / OpenVINO.

The engine keeps the task network's own pre/post-processing and swaps only
the forward pass (the export parity gate's exact contract), so file, video,
and stream inference produce the same coordinates as ``val``/``predict``.
``run_stream`` drives a video file or RTSP source through the engine and
persists per-frame predictions (JSONL) plus an annotated MP4.
``run_multi_stream`` runs N sources with one shared batcher.

    from autovision.inference import InferenceEngine, run_stream, run_multi_stream

    run_stream(model="runs/d1/exported_models/exported_best.pt",
               source="rtsp://camera/live", out_dir="preds/",
               backend="onnx", artifact="model.onnx", batch_size=4)

    run_multi_stream(["rtsp://cam1/live", "rtsp://cam2/live"], "preds/",
                     model="best.pt", batch_size=4, queue_size=8)

Video/stream I/O needs the ``video`` extra (OpenCV); the ``onnx`` and
``tensorrt`` backends additionally need ``export`` / the ``tensorrt``
package respectively.
"""

from __future__ import annotations

from autovision.inference.engine import InferenceEngine
from autovision.inference.multi_stream import (
    MultiStreamResult,
    MultiStreamRunner,
    StreamSource,
    StreamStats,
    run_multi_stream,
)
from autovision.inference.stream import StreamRunResult, run_stream
from autovision.inference.video import (
    AnnotatedVideoWriter,
    FPSMonitor,
    ImageSink,
    VideoFrameStream,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)
from autovision.results import InferenceResult

__all__ = [
    "AnnotatedVideoWriter",
    "FPSMonitor",
    "ImageSink",
    "InferenceEngine",
    "InferenceResult",
    "MultiStreamResult",
    "MultiStreamRunner",
    "StreamRunResult",
    "StreamSource",
    "StreamStats",
    "VideoFrameStream",
    "VideoInfo",
    "VideoSink",
    "get_video_frames_generator",
    "process_video",
    "run_multi_stream",
    "run_stream",
]
