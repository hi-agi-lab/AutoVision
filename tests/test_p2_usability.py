# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""P2 usability: coco8 smoke dataset, model handle (train/val/predict/
export), YOLO<->COCO conversion, download registry, CLI commands.

The end-to-end test below doubles as the <2-minute smoke gate (roadmap
design doc 29, J4): coco8 train -> val -> predict -> export on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovision import detection, ensure_coco8, evaluate_object_detection, load_model
from autovision.data.convert import coco_to_yolo, yolo_to_coco
from autovision.data.downloads import DATASET_REGISTRY, download_dataset
from autovision.errors import AutoVisionError

TINY_MODEL_ARGS = {
    "backbone": "torchvision/resnet18",
    "image_size": (64, 64),
    "hidden_dim": 32,
    "num_queries": 10,
    "dec_layers": 2,
    "num_heads": 4,
    "num_top_queries": 10,
}
TINY_TRANSFORM_ARGS = {
    "scale_jitter": None,
    "photometric_distort_prob": 0.5,
    "random_zoom_out_prob": None,
    "random_iou_crop_prob": None,
}


@pytest.fixture(scope="module")
def coco8(tmp_path_factory):
    return ensure_coco8(tmp_path_factory.mktemp("coco8"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory, coco8):
    """One 3-step CPU training run shared by the E2E/CLI tests (<2min)."""
    return detection("rtdetr/resnet18").train(
        out=tmp_path_factory.mktemp("run") / "run",
        data=coco8,
        steps=3,
        batch_size=2,
        num_workers=0,
        model_args=TINY_MODEL_ARGS,
        transform_args=TINY_TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 2},
    )


class TestCoco8:
    def test_creates_valid_dataset(self, coco8):
        root = Path(coco8["train"]["annotations"]).parent
        train = json.loads(Path(coco8["train"]["annotations"]).read_text())
        val = json.loads(Path(coco8["val"]["annotations"]).read_text())
        assert len(train["images"]) == 8
        assert len(train["annotations"]) == 8
        assert len(val["images"]) == 2
        assert {c["name"] for c in train["categories"]} == {"cat", "dog"}
        for image in train["images"]:
            assert (root / "images" / image["file_name"]).exists()

    def test_cached_and_deterministic(self, tmp_path):
        first = ensure_coco8(tmp_path / "d1")
        second = ensure_coco8(tmp_path / "d1")
        assert first == second
        assert (tmp_path / "d1" / "train.json").read_bytes() == (
            tmp_path / "d1" / "train.json"
        ).read_bytes()
        forced = ensure_coco8(tmp_path / "d2", force=True)
        assert forced["format"] == "coco"


class TestModelHandleE2E:
    def test_train_val_predict_export(self, trained, coco8, tmp_path):
        assert trained.train_result is not None
        assert trained.train_result.steps_run == 3

        metrics = trained.val(data=coco8)
        assert "map" in metrics and "map_50" in metrics

        image = str(Path(coco8["train"]["images"]) / "img_000.png")
        prediction = trained.predict(image, threshold=0.0)
        assert set(prediction) == {"labels", "bboxes", "scores"}

        exported = trained.export(tmp_path / "model.pt")
        assert exported.exists()
        reloaded = load_model(exported)
        assert set(reloaded.predict(image, threshold=0.0)) == {
            "labels",
            "bboxes",
            "scores",
        }

        path_metrics = evaluate_object_detection(exported, coco8, max_images=1)
        assert "map" in path_metrics

    def test_train_requires_registry_key(self):
        from autovision.cli.model_handle import DetectionModel

        with pytest.raises(AutoVisionError, match="registry model key"):
            DetectionModel(model=None).train(out="unused", data="coco8")

    def test_predict_before_train_raises(self):
        with pytest.raises(AutoVisionError, match="No trained/exported model"):
            detection("rtdetr/resnet18").predict("whatever.png")


class TestPredictInferenceContract:
    """Regression (2026-10-16 GTR val): predict/predict_batch must run
    under ``torch.no_grad`` (retained autograd graphs OOM'd a 24 GiB GPU
    during 5000-image COCO eval) and must move path-loaded inputs onto the
    network's device (``.to("cuda")`` previously crashed on CPU inputs)."""

    def test_predict_does_not_build_graphs(self, trained, coco8):
        import torch

        network = trained._require_network()
        seen: dict[str, object] = {}
        original = network.forward_backend

        def probe(x):
            seen["grad_enabled"] = torch.is_grad_enabled()
            seen["device"] = x.device
            return original(x)

        network.forward_backend = probe
        image = str(Path(coco8["train"]["images"]) / "img_000.png")
        network.predict(image, threshold=0.0)
        network.predict_batch([image], threshold=0.0)
        assert seen["grad_enabled"] is False
        assert seen["device"] == network._module_device()

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="CUDA required"
    )
    def test_predict_serves_path_inputs_on_cuda(self, trained, coco8):
        network = trained._require_network().to("cuda")
        try:
            image = str(Path(coco8["train"]["images"]) / "img_000.png")
            prediction = network.predict(image, threshold=0.0)
            assert prediction["bboxes"].device.type == "cuda"
        finally:
            network.to("cpu")

    def test_unknown_trainer_args_raise(self, coco8, tmp_path):
        """trainer_args is an allowlist: unknown keys used to be dropped
        silently (ema_warmup_steps belongs in model_args) — now loud."""
        from autovision import train_object_detection

        with pytest.raises(ValueError, match="Unknown trainer_args"):
            train_object_detection(
                out=tmp_path / "run",
                data=coco8,
                model="rtdetr/resnet18",
                steps=1,
                batch_size=2,
                num_workers=0,
                model_args=TINY_MODEL_ARGS,
                transform_args=TINY_TRANSFORM_ARGS,
                trainer_args={"ema_warmup_steps": 10},
            )


class TestCoco8Token:
    def test_train_data_loader_resolves_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(tmp_path / "cache"))
        from autovision.cli.train_task.object_detection import _load_data_config

        payload = _load_data_config("coco8")
        assert payload["format"] == "coco"
        assert (tmp_path / "cache" / "datasets" / "coco8" / "train.json").exists()


class TestConvert:
    def test_yolo_coco_roundtrip(self, tmp_path):
        from tests.conftest import create_image

        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        first = create_image(images_dir / "a.png", size=(32, 32), seed=0)
        second = create_image(images_dir / "b.png", size=(32, 32), seed=1)
        (labels_dir / f"{first.stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        (labels_dir / f"{second.stem}.txt").write_text("1 0.25 0.25 0.25 0.25\n")

        out_json = yolo_to_coco(
            images_dir,
            names=["cat", "dog"],
            out_json=tmp_path / "train.json",
            labels_dir=labels_dir,
        )
        payload = json.loads(out_json.read_text())
        assert [c["name"] for c in payload["categories"]] == ["cat", "dog"]
        assert len(payload["annotations"]) == 2
        annotation = next(a for a in payload["annotations"] if a["image_id"] == 1)
        assert annotation["bbox"] == pytest.approx([8.0, 8.0, 16.0, 16.0])

        out_labels = coco_to_yolo(
            out_json, tmp_path / "labels2", names_out=tmp_path / "classes.txt"
        )
        line = (out_labels / f"{first.stem}.txt").read_text().strip().splitlines()[0]
        class_index, cx, cy, w, h = line.split()
        assert class_index == "0"
        assert max(abs(float(v) - e) for v, e in zip((cx, cy, w, h), (0.5, 0.5, 0.5, 0.5))) < 1e-6
        assert (tmp_path / "classes.txt").read_text().splitlines() == ["cat", "dog"]

    def test_malformed_label_line_raises(self, tmp_path):
        from PIL import Image

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        Image.new("RGB", (8, 8)).save(images_dir / "a.png")
        (labels_dir / "a.txt").write_text("0 0.5 0.5\n")
        with pytest.raises(ValueError, match="cls cx cy w h"):
            yolo_to_coco(
                images_dir,
                names=["cat"],
                out_json=tmp_path / "x.json",
                labels_dir=labels_dir,
            )


class TestDownloadRegistry:
    def test_entries_are_wellformed(self):
        for name, spec in DATASET_REGISTRY.items():
            assert spec.description and spec.homepage
            if spec.manual_instructions:
                assert not spec.files
            else:
                assert spec.files
                for entry in spec.files:
                    assert entry.url.startswith("http")

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(AutoVisionError, match="Unknown dataset"):
            download_dataset("nope", tmp_path)

    def test_manual_dataset_refuses(self, tmp_path):
        with pytest.raises(AutoVisionError, match="manual acquisition"):
            download_dataset("visdrone2017-det", tmp_path)


class TestCliCommands:
    def test_val_command(self, trained, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(tmp_path / "cache"))
        from autovision.cli._app import val_command

        exported = trained.export(tmp_path / "m.pt")
        val_command(
            [f"model={exported}", "data=coco8", f"out={tmp_path / 'val.json'}"]
        )
        assert "map" in capsys.readouterr().out
        assert "map" in json.loads((tmp_path / "val.json").read_text())

    def test_predict_command(self, trained, coco8, tmp_path):
        from autovision.cli._app import predict_command

        exported = trained.export(tmp_path / "m.pt")
        predict_command(
            [
                f"model={exported}",
                f"images={coco8['train']['images']}",
                f"out={tmp_path / 'preds'}",
                "threshold=0.5",
            ]
        )
        records = json.loads((tmp_path / "preds" / "predictions.json").read_text())
        assert records and "bboxes" in records[0]
        assert (tmp_path / "preds" / "visuals").is_dir()

    def test_plot_curves_optional_dependency(self, trained, tmp_path):
        from autovision.cli._app import plot_curves_command

        run_dir = trained.train_result.out_dir
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            with pytest.raises(AutoVisionError, match="matplotlib"):
                plot_curves_command([f"run={run_dir}"])
            return
        plot_curves_command([f"run={run_dir}"])
        assert (run_dir / "training_curves.png").exists()
