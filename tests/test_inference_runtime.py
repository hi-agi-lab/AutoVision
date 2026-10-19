# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference inference runtime (design doc 08 §12): engine backends, video I/O,
stream end-to-end.

The load-bearing check is the torch-vs-ONNX parity: both backends share the
network's own pre/post-processing and only swap the forward, so records must
agree to the export gate's band (observed ~1e-5 px boxes / ~1e-8 scores on
the fp32 tiny model). RTSP is exercised through the same reader as files;
without an RTSP server in CI the test locks the unreachable-URL error path.
"""

from __future__ import annotations

import importlib.util
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.inference import (
    AnnotatedVideoWriter,
    InferenceEngine,
    VideoFrameStream,
    run_stream,
)
from autovision.tasks import RTDETRObjectDetection

_cv2_installed = importlib.util.find_spec("cv2") is not None
_ort_installed = importlib.util.find_spec("onnxruntime") is not None
_onnx_installed = importlib.util.find_spec("onnx") is not None

requires_video = pytest.mark.skipif(
    not _cv2_installed, reason="opencv not installed (pip install -e '.[video]')"
)
requires_onnx = pytest.mark.skipif(
    not (_ort_installed and _onnx_installed),
    reason="onnx/onnxruntime not installed (pip install -e '.[export]')",
)

VIDEO_FRAMES = 8
VIDEO_SIZE = (96, 64)  # (width, height)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """A deterministic synthetic MP4 (moved boxes over noise)."""
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    width, height = VIDEO_SIZE
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for index in range(VIDEO_FRAMES):
        frame = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (40 + index, 40), (255, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def frames(clip) -> list[np.ndarray]:
    return [frame for _, frame in VideoFrameStream(clip)]


@pytest.fixture()
def live_stream_url():
    """In-process MJPEG-over-HTTP live source (same URL path as RTSP).

    Serves ``VIDEO_FRAMES`` frames over ``multipart/x-mixed-replace``, then
    closes — the reader sees a genuine network stream with live framing
    (no probe-able frame count), which is the RTSP code path minus the
    transport.
    """
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")
    rng = np.random.default_rng(0)
    payloads = []
    for index in range(VIDEO_FRAMES):
        frame = (rng.random((VIDEO_SIZE[1], VIDEO_SIZE[0], 3)) * 255).astype(np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (40 + index, 40), (255, 255, 0), -1)
        payloads.append(cv2.imencode(".jpg", frame)[1].tobytes())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                for payload in payloads:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                        % len(payload)
                    )
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/stream.mjpg"
    finally:
        server.shutdown()
        server.server_close()


@requires_video
def test_frame_stream_contract(clip):
    stream = VideoFrameStream(clip)
    assert stream.info.width == VIDEO_SIZE[0]
    assert stream.info.height == VIDEO_SIZE[1]
    assert stream.info.frame_count == VIDEO_FRAMES
    assert stream.info.fps == pytest.approx(10.0)
    frames = list(stream)
    assert [index for index, _ in frames] == list(range(VIDEO_FRAMES))
    frame = frames[0][1]
    assert frame.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3)
    assert frame.dtype == np.uint8


@requires_video
def test_frame_stream_max_frames(clip):
    assert len(list(VideoFrameStream(clip, max_frames=3))) == 3


@requires_video
def test_frame_stream_unreachable_rtsp_raises():
    with pytest.raises(AutoVisionError, match="Could not open video source"):
        VideoFrameStream("rtsp://127.0.0.1:1/unreachable", open_timeout_ms=1000)


@requires_video
def test_live_stream_reader(live_stream_url):
    """A live network source reads frame-by-frame (max_frames stops early)."""
    stream = VideoFrameStream(live_stream_url, max_frames=5)
    frames = list(stream)
    assert [index for index, _ in frames] == [0, 1, 2, 3, 4]
    assert frames[0][1].shape[:2] == (VIDEO_SIZE[1], VIDEO_SIZE[0])


@requires_video
def test_live_stream_end_to_end(tiny_model, live_stream_url, tmp_path):
    """Live source -> engine -> JSONL + annotated MP4 (the RTSP code path)."""
    result = run_stream(
        live_stream_url, tmp_path / "live", model=tiny_model,
        batch_size=2, threshold=0.0, max_frames=5,
    )
    assert result.frames == 5
    records = [
        json.loads(line)
        for line in result.predictions_path.read_text().splitlines()
    ]
    assert [record["frame"] for record in records] == list(range(5))
    assert result.video_path is not None and result.video_path.exists()
    assert len(list(VideoFrameStream(result.video_path))) == 5


@requires_video
def test_annotated_writer_roundtrip(clip, tmp_path):
    out = tmp_path / "written.mp4"
    writer = AnnotatedVideoWriter(out, fps=10.0, frame_size=VIDEO_SIZE)
    for _, frame in VideoFrameStream(clip):
        writer.write(frame)
    writer.close()
    assert len(list(VideoFrameStream(out))) == VIDEO_FRAMES


@requires_video
def test_engine_torch_matches_network_predict(tiny_model, frames):
    """The torch engine mirrors ``TaskNetwork.predict`` (same contract)."""
    tensor = torch.from_numpy(frames[0]).permute(2, 0, 1)
    engine = InferenceEngine(tiny_model, backend="torch")
    engine_record = engine.run(tensor, threshold=0.0)
    network_record = tiny_model.predict(tensor, threshold=0.0)
    assert torch.equal(engine_record["labels"], network_record["labels"])
    torch.testing.assert_close(engine_record["bboxes"], network_record["bboxes"])
    torch.testing.assert_close(engine_record["scores"], network_record["scores"])


@requires_video
def test_engine_rejects_unknown_backend(tiny_model):
    with pytest.raises(AutoVisionError, match="Unknown inference backend"):
        InferenceEngine(tiny_model, backend="tflite")


@requires_video
def test_engine_requires_artifact_for_compiled_backends(tiny_model):
    with pytest.raises(AutoVisionError, match="compiled artifact"):
        InferenceEngine(tiny_model, backend="onnx")
    with pytest.raises(AutoVisionError, match="compiled artifact"):
        InferenceEngine(tiny_model, backend="openvino")


@requires_video
def test_engine_openvino_requires_xml_artifact(tiny_model, tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_text("not an IR")
    with pytest.raises(AutoVisionError, match="\\.xml IR artifact"):
        InferenceEngine(tiny_model, backend="openvino", artifact=artifact)


@requires_video
def test_engine_openvino_forward_wiring(monkeypatch, tiny_model, tmp_path):
    import autovision.export.openvino_helpers as ov_helpers

    seen: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, ir_path, device="CPU"):
            seen["ir_path"] = str(ir_path)
            seen["device"] = device

        def run(self, batch):
            return (
                torch.zeros(batch.shape[0], 10, 3),
                torch.full((batch.shape[0], 10, 4), 0.5),
            )

    monkeypatch.setattr(ov_helpers, "OpenVINOEngineRunner", FakeRunner)
    artifact = tmp_path / "model.xml"
    artifact.write_text("<xml />")
    engine = InferenceEngine(
        tiny_model, backend="openvino", artifact=artifact,
        openvino_device="GPU",
    )
    raw = engine._forward(torch.zeros(1, 3, 64, 64))
    assert raw[0].shape == (1, 10, 3)
    assert seen["device"] == "GPU"
    assert seen["ir_path"].endswith("model.xml")
    assert engine.openvino_device == "GPU"


@requires_video
def test_stream_end_to_end_jsonl_and_video(tiny_model, clip, tmp_path):
    out_dir = tmp_path / "preds"
    result = run_stream(
        clip, out_dir, model=tiny_model, batch_size=3, threshold=0.0,
        class_names={0: "left", 1: "right"},
    )
    assert result.frames == VIDEO_FRAMES
    assert result.backend == "torch"
    records = [
        json.loads(line)
        for line in result.predictions_path.read_text().splitlines()
    ]
    assert [record["frame"] for record in records] == list(range(VIDEO_FRAMES))
    assert records[0]["time_s"] == 0.0
    predictions = records[0]["predictions"]
    assert set(predictions) == {"bboxes", "scores", "labels"}
    assert len(predictions["scores"]) == 10  # num_top_queries
    # Boxes are decoded in original-frame pixel scale (the network's own
    # postprocess; small negative/overshoot values are unclipped raw decode).
    boxes = np.asarray(predictions["bboxes"])
    assert np.isfinite(boxes).all()
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
    assert np.abs(boxes).max() > 1.0  # not normalized-space coordinates
    # Annotated video reopens with every frame.
    assert result.video_path is not None and result.video_path.exists()
    assert len(list(VideoFrameStream(result.video_path))) == VIDEO_FRAMES


@requires_video
def test_stream_batch_size_invariance(tiny_model, clip, tmp_path):
    """Batching must not change records beyond float32 rounding.

    (CPU kernels reduce in a batch-size-dependent order; measured deltas are
    ~2e-8 on scores and 0.0 on boxes for the tiny model.)
    """
    single = run_stream(
        clip, tmp_path / "b1", model=tiny_model, batch_size=1, threshold=0.0
    )
    batched = run_stream(
        clip, tmp_path / "b4", model=tiny_model, batch_size=4, threshold=0.0
    )
    single_records = [
        json.loads(line) for line in single.predictions_path.read_text().splitlines()
    ]
    batched_records = [
        json.loads(line) for line in batched.predictions_path.read_text().splitlines()
    ]
    assert len(single_records) == len(batched_records) == VIDEO_FRAMES
    for one, four in zip(single_records, batched_records):
        assert one["frame"] == four["frame"]
        assert one["predictions"]["labels"] == four["predictions"]["labels"]
        np.testing.assert_allclose(
            one["predictions"]["bboxes"], four["predictions"]["bboxes"], atol=1e-4
        )
        np.testing.assert_allclose(
            one["predictions"]["scores"], four["predictions"]["scores"], atol=1e-6
        )


@requires_onnx
def test_engine_onnx_matches_torch(tiny_model, frames, tmp_path):
    """ONNX path vs torch path: the same pre/post, so records must agree."""
    onnx_path = tiny_model.export_onnx(tmp_path / "model.onnx", precision="fp32")
    torch_engine = InferenceEngine(tiny_model, backend="torch")
    onnx_engine = InferenceEngine(tiny_model, backend="onnx", artifact=onnx_path)
    tensors = [
        torch.from_numpy(frame).permute(2, 0, 1) for frame in frames[:4]
    ]
    torch_records = torch_engine.run_batch(tensors, threshold=0.0)
    onnx_records = onnx_engine.run_batch(tensors, threshold=0.0)
    for torch_record, onnx_record in zip(torch_records, onnx_records):
        assert torch.equal(torch_record["labels"], onnx_record["labels"])
        torch.testing.assert_close(
            torch_record["bboxes"], onnx_record["bboxes"], atol=5e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            torch_record["scores"], onnx_record["scores"], atol=5e-3, rtol=1e-3
        )


@requires_onnx
@requires_video
def test_stream_onnx_backend_end_to_end(tiny_model, clip, tmp_path):
    onnx_path = tiny_model.export_onnx(tmp_path / "model.onnx", precision="fp32")
    result = run_stream(
        clip, tmp_path / "preds", model=tiny_model, backend="onnx",
        artifact=onnx_path, batch_size=2, threshold=0.0, render=False,
    )
    assert result.frames == VIDEO_FRAMES
    assert result.backend == "onnx"
    assert result.video_path is None  # render=False
    assert len(result.predictions_path.read_text().splitlines()) == VIDEO_FRAMES
