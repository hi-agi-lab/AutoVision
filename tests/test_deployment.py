# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the deployment layer: ONNX export, SAHI, benchmark, load_model."""

import pytest
import torch

from autovision.export import benchmark_inference
from autovision.tasks import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
)
from autovision.tasks.shared.sahi import (
    combine_object_detection_tiles,
    tile_image,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_helpers import load_task_model

onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed (pip install -e '.[export]')")
pytest.importorskip("onnxslim", reason="onnxslim not installed (pip install -e '.[export]')")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )


class TestOnnxExport:
    def test_fp32_export_with_parity_gate(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model.onnx",
            precision="fp32",
            dynamic_batch_size=True,
            format_args={"opset_version": 18},
        )
        assert out.exists()
        assert out.stat().st_size > 100_000

    def test_fp16_export(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model_fp16.onnx",
            precision="fp16",
            format_args={"opset_version": 18},
        )
        assert out.exists()
        # FP16 graph actually contains half tensors.
        import onnx

        model = onnx.load(str(out))
        dtypes = {t.data_type for i in model.graph.initializer for t in [i]}
        assert onnx.TensorProto.FLOAT16 in dtypes

    def test_dynamic_batch(self, tiny_model, tmp_path):
        out = tiny_model.export_onnx(
            tmp_path / "model_dyn.onnx",
            format_args={"opset_version": 18},
        )
        import onnxruntime as ort

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        for batch in (1, 3):
            outputs = session.run(
                None, {"images": torch.randn(batch, 3, 64, 64).numpy()}
            )
            assert outputs[0].shape[0] == batch
            assert outputs[1].shape[0] == batch

    def test_bad_precision_raises(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="precision"):
            tiny_model.export_onnx(tmp_path / "x.onnx", precision="int8")

    def test_fp16_requires_simplify(self, tiny_model, tmp_path):
        with pytest.raises(ValueError, match="simplify"):
            tiny_model.export_onnx(
                tmp_path / "x.onnx", precision="fp16", simplify=False
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_resident_model_exports(self, tiny_model, tmp_path):
        """The example input follows the model's device (trained-on-GPU flow)."""
        import copy

        model = copy.deepcopy(tiny_model).cuda()
        out = model.export_onnx(
            tmp_path / "cuda_model.onnx",
            precision="fp32",
            format_args={"opset_version": 18},
        )
        assert out.exists() and out.stat().st_size > 100_000


class TestSAHI:
    def test_tile_image_overlap(self):
        image = torch.randn(3, 200, 300)
        tiles, coordinates = tile_image(image, overlap=0.2, tile_size=128)
        assert tiles.shape == (len(coordinates), 3, 128, 128)
        assert coordinates.min() >= 0
        # Overlap guarantees more than the no-overlap count.
        tiles_no_overlap, coords_no_overlap = tile_image(image, 0.0, 128)
        assert tiles.shape[0] >= tiles_no_overlap.shape[0]

    def test_tile_image_small_image(self):
        # Small images are upscaled (default) or padded to the tile size.
        tiles, coordinates = tile_image(torch.randn(3, 32, 32), 0.2, 64)
        assert tiles.shape == (1, 3, 64, 64)
        assert coordinates.tolist() == [[0, 0]]
        tiles_pad, _ = tile_image(torch.randn(3, 32, 32), 0.2, 64, padding_mode="pad")
        assert tiles_pad.shape == (1, 3, 64, 64)

    def test_tile_image_padding_mode(self):
        tiles, _ = tile_image(torch.randn(3, 100, 100), 0.0, 128, padding_mode="pad")
        assert tiles.shape[-2:] == (128, 128)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            tile_image(torch.randn(3, 100, 100), 1.0, 64)

    def test_combine_suppresses_duplicates(self):
        global_output = {
            "bboxes": torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        tile_outputs = [
            {  # same object seen inside a tile (near-duplicate)
                "bboxes": torch.tensor([[20.0, 20.0, 70.0, 70.0],
                                        [200.0, 200.0, 240.0, 240.0]]),
                "scores": torch.tensor([0.8, 0.7]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        coordinates = torch.tensor([[80, 80]])
        merged = combine_object_detection_tiles(
            global_output, tile_outputs, coordinates, original_size=(300, 300)
        )
        # The duplicate tile box (IoU > 0.1 with the global box) is dropped;
        # the second (different object) survives.
        assert merged["bboxes"].shape[0] == 2

    def test_combine_no_tiles(self):
        global_output = {
            "bboxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.5]),
            "labels": torch.tensor([0]),
        }
        merged = combine_object_detection_tiles(
            global_output, [], torch.zeros(0, 2), (100, 100)
        )
        assert merged["bboxes"].shape[0] == 1


class _SahiStubModel:
    """Stub wiring a TaskNetwork's three-stage contract to canned outputs,
    so the SAHI coordinate pipeline is verified without a real detector."""

    image_size = (64, 64)
    training = False

    def __init__(self, outputs):
        self.outputs = outputs
        self.seen_metadata = None
        self.seen_threshold = None

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def preprocess_image(self, image):
        from torchvision.transforms.v2 import functional as TF

        tensor = image.float() if isinstance(image, torch.Tensor) else image
        _, h, w = tensor.shape
        # Real models resize to the fixed input size here.
        resized = TF.resize(tensor, list(self.image_size))
        return resized, {"orig_w": w, "orig_h": h}

    def preprocess_batch(self, batch):
        return batch

    def forward_backend(self, batch):
        return None

    def postprocess(self, raw, metadata, threshold=0.0):
        self.seen_metadata = metadata
        self.seen_threshold = threshold
        return self.outputs

    def _module_device(self):
        return torch.device("cpu")

    # TaskNetwork unbound methods bound to the stub instance.
    predict_sahi = TaskNetwork.predict_sahi
    _load_image_unchanged = TaskNetwork._load_image_unchanged


def _det_output(boxes, scores, labels):
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32),
        "scores": torch.as_tensor(scores, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


class TestPredictSahi:
    def test_end_to_end_contract_and_clipping(self, tiny_model):
        torch.manual_seed(0)
        tiny_model.eval()
        out = tiny_model.predict_sahi(
            torch.rand(3, 220, 300), tile_size=128, overlap=0.2, threshold=0.0
        )
        assert set(out) == {"bboxes", "scores", "labels"}
        assert out["bboxes"].shape[1] == 4
        assert out["bboxes"].shape[0] == out["scores"].shape[0]
        # Boxes are clipped to the original canvas.
        assert (out["bboxes"][:, 0::2] >= 0).all()
        assert (out["bboxes"][:, 0::2] <= 300 + 1e-4).all()
        assert (out["bboxes"][:, 1::2] <= 220 + 1e-4).all()

    def test_default_tile_size_is_model_input_size(self, tiny_model, monkeypatch):
        seen = {}
        from autovision.tasks.shared import tiling_utils

        real_tile = tiling_utils.tile_image

        def spy_tile(image, overlap, tile_size, padding_mode="resize"):
            seen["tile_size"] = tile_size
            return real_tile(image, overlap, tile_size, padding_mode)

        monkeypatch.setattr(tiling_utils, "tile_image", spy_tile)
        tiny_model.predict_sahi(torch.rand(3, 70, 70), threshold=0.0)
        assert seen["tile_size"] == min(tiny_model.image_size)

    def test_tile_metas_effective_canvas(self):
        # resize mode: (3, 32, 90) with tile 64 slices into two tiles
        # (x=0 and x=26); edge tiles are interpolated, so the effective
        # canvas is the *pre-interpolation* edge, not the full tile.
        outputs = [_det_output([[1.0, 1.0, 5.0, 5.0]], [0.9], [0])] * 3
        stub = _SahiStubModel(outputs)
        stub.predict_sahi(torch.rand(3, 32, 90), tile_size=64, threshold=0.0)
        assert stub.seen_metadata[0] == {"orig_w": 90, "orig_h": 32}  # global
        assert stub.seen_metadata[1] == {"orig_w": 64, "orig_h": 32}
        assert stub.seen_metadata[2] == {"orig_w": 64, "orig_h": 32}
        # pad mode: the tile frame is the full padded tile.
        stub_pad = _SahiStubModel(outputs)
        stub_pad.predict_sahi(
            torch.rand(3, 32, 90), tile_size=64, threshold=0.0, padding_mode="pad"
        )
        assert stub_pad.seen_metadata[1] == {"orig_w": 64, "orig_h": 64}

    def test_tile_offset_and_global_passthrough(self, monkeypatch):
        from autovision.tasks.shared import tiling_utils

        seen = {}
        real_combine = tiling_utils.combine_object_detection_tiles

        def spy_combine(global_output, tile_outputs, coordinates, size, **kwargs):
            seen["global"] = global_output
            seen["tiles"] = tile_outputs
            seen["coordinates"] = coordinates
            seen["size"] = size
            seen["nms_iou_threshold"] = kwargs["nms_iou_threshold"]
            return real_combine(
                global_output, tile_outputs, coordinates, size, **kwargs
            )

        monkeypatch.setattr(
            tiling_utils, "combine_object_detection_tiles", spy_combine
        )
        global_out = _det_output([[10.0, 10.0, 50.0, 50.0]], [0.9], [0])
        tile_out = _det_output(
            [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 30.0, 30.0]], [0.8, 0.7], [0, 1]
        )
        # (3, 128, 128) with tile 64 and no overlap -> 4 tiles; tile outputs
        # must line up one-to-one with the tile coordinates.
        stub = _SahiStubModel([global_out, tile_out, tile_out, tile_out, tile_out])
        merged = stub.predict_sahi(
            torch.rand(3, 128, 128), tile_size=64, overlap=0.0, threshold=0.0
        )
        assert seen["size"] == (128, 128)
        assert seen["global"] is global_out
        assert seen["nms_iou_threshold"] == 0.3
        assert seen["coordinates"].tolist() == [[0, 0], [64, 0], [0, 64], [64, 64]]
        assert len(seen["tiles"]) == 4
        # Tile-0 output (no offset needed at (0,0)): the different-label box
        # survives merging alongside the global box.
        assert merged["bboxes"].shape[0] >= 2
        assert merged["scores"][0] == 0.9  # global first (descending)

    def test_threshold_forwarded_to_postprocess(self):
        stub = _SahiStubModel(
            [_det_output([[1.0, 1.0, 5.0, 5.0]], [0.9], [0])]
        )
        stub.predict_sahi(torch.rand(3, 32, 32), tile_size=64, threshold=0.55)
        assert stub.seen_threshold == 0.55

    def test_non_detection_model_raises(self):
        stub = _SahiStubModel([{"logits": torch.zeros(2, 3)}])
        with pytest.raises(TypeError, match="bboxes"):
            stub.predict_sahi(torch.rand(3, 32, 32), tile_size=64)


class TestBenchmark:
    def test_benchmark_latency(self, tiny_model):
        inputs = {"images": torch.randn(1, 3, 64, 64)}
        result = benchmark_inference(
            tiny_model, inputs, model_name="tiny", warmup_steps=2, steps=5
        )
        assert result.num_steps == 5
        assert result.latency_ms_p50 <= result.latency_ms_p90
        assert result.latency_ms_mean > 0
        assert result.images_per_sec > 0
        assert "tiny" in result.to_markdown()


class TestLoadTaskModel:
    def _export(self, tmp_path):
        torch.manual_seed(0)
        model = RTDETRObjectDetection(
            num_classes=2, backbone="torchvision/resnet18",
            image_size=(64, 64), hidden_dim=32, num_queries=10,
            dec_layers=1, num_heads=4, num_top_queries=10,
        )
        train_model = RTDETRObjectDetectionTrain(
            RTDETRObjectDetectionTrainArgs(
                num_classes=2, image_size=(64, 64), hidden_dim=32,
                num_queries=10, dec_layers=1, num_heads=4,
                backbone="torchvision/resnet18", num_top_queries=10,
            )
        )
        train_model.model = model
        train_model.on_train_batch_end()
        out = tmp_path / "exported_last.pt"
        torch.save(
            {
                "schema_version": 1,
                "kind": "task_model",
                "model_class_path": model.class_path,
                "model_init_args": {**model.init_args, "model_name": "rtdetr/test"},
                "train_model": train_model.get_export_state_dict(),
            },
            out,
        )
        return out

    def test_load_and_predict(self, tmp_path):
        out = self._export(tmp_path)
        model = load_task_model(out, device="cpu")  # dec_layers matches export
        assert not model.training
        prediction = model.predict(
            torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8), threshold=0.0
        )
        assert set(prediction) == {"labels", "bboxes", "scores"}

    def test_missing_file_raises(self, tmp_path):
        from autovision.errors import CheckpointError

        with pytest.raises(CheckpointError, match="does not exist"):
            load_task_model(tmp_path / "nope.pt")

    def test_foreign_file_raises(self, tmp_path):
        p = tmp_path / "foreign.pt"
        torch.save({"something": 1}, p)
        from autovision.errors import CheckpointError

        with pytest.raises(CheckpointError, match="not an exported"):
            load_task_model(p)
