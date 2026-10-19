"""Tests for the EOMT instance segmentation family (HF-backed)."""

import json

import numpy as np
import pytest
import torch

from autovision import load_task_model as load_model
from autovision import train_instance_segmentation
from autovision.data.coco_instance_segmentation_dataset import (
    COCOInstanceSegmentationDataset,
)
from autovision.metrics.instance_segmentation import COCOMaskAPEvaluator
from autovision.tasks.instance_segmentation.dinov2_eomt import (
    Dinov2EomtInstanceSegmentationTrain,
    Dinov2EomtInstanceSegmentationTrainArgs,
)
from autovision.tasks.shared.eomt.postprocess import instance_segmentation_decode

MODEL_KWARGS = {
    "image_size": 64,
    "num_queries": 6,
    "num_blocks": 2,
    "num_upscale_blocks": 1,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}


def make_train_model(**overrides) -> Dinov2EomtInstanceSegmentationTrain:
    torch.manual_seed(0)
    args = Dinov2EomtInstanceSegmentationTrainArgs(
        num_classes=2, lr_warmup_steps=2, **{**MODEL_KWARGS, **overrides}
    )
    return Dinov2EomtInstanceSegmentationTrain(args)


def make_batch(size: int = 64, images: int = 2):
    mask = torch.zeros(size, size)
    mask[: size // 2, : size // 2] = 1.0  # one instance in the top-left
    return {
        "image": torch.randn(images, 3, size, size),
        "masks": [mask.unsqueeze(0) for _ in range(images)],
        "classes": [torch.tensor([1]) for _ in range(images)],
        "bboxes": [torch.tensor([[0.25, 0.25, 0.5, 0.5]]) for _ in range(images)],
        "original_size": [(size, size)] * images,
    }


def write_coco(tmp_path, num_images: int = 3, size: int = 64):
    """Two square instances per image (classes 0 and 1) as polygons."""
    annotations = {
        "images": [
            {"id": i, "file_name": f"img_{i}.jpg", "width": size, "height": size}
            for i in range(num_images)
        ],
        "categories": [
            {"id": 1, "name": "square_a"},
            {"id": 2, "name": "square_b"},
        ],
        "annotations": [],
    }
    for i in range(num_images):
        annotations["annotations"].extend(
            [
                {
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [4, 4, size // 4, size // 4],
                    "segmentation": [[4, 4, 4 + size // 4, 4, 4 + size // 4, 4 + size // 4, 4, 4 + size // 4]],
                    "area": (size // 4) ** 2,
                    "iscrowd": 0,
                },
                {
                    "image_id": i,
                    "category_id": 2,
                    "bbox": [size // 2, size // 2, size // 4, size // 4],
                    "segmentation": [
                        [
                            size // 2, size // 2,
                            size // 2 + size // 4, size // 2,
                            size // 2 + size // 4, size // 2 + size // 4,
                            size // 2, size // 2 + size // 4,
                        ]
                    ],
                    "area": (size // 4) ** 2,
                    "iscrowd": 0,
                },
            ]
        )
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations))
    return path


class TestInstanceDecode:
    def test_decode_instances(self):
        masks = torch.full((1, 2, 4, 4), -10.0)
        masks[0, 0, :2, :2] = 10.0  # query 0 mask: top-left block
        classes = torch.full((1, 2, 3), -10.0)
        classes[0, 0, 0] = 10.0  # query 0 -> class 0, strong score
        classes[0, 1, 0] = 2.0  # query 1 -> class 0, weaker score
        results = instance_segmentation_decode(
            masks, classes, [{"orig_w": 8, "orig_h": 8}],
            score_threshold=0.1, mask_threshold=0.5,
        )
        result = results[0]
        assert result["masks"].shape == (2, 8, 8)
        assert result["masks"].dtype == torch.bool
        # Sorted by descending score; the strong query first.
        assert result["scores"][0] > result["scores"][1]
        assert result["masks"][0, :4, :4].all()
        assert not result["masks"][0, 4:, :].any()

    def test_decode_empty(self):
        classes = torch.full((1, 2, 3), -10.0)
        masks = torch.full((1, 2, 4, 4), -10.0)
        results = instance_segmentation_decode(
            masks, classes, [{"orig_w": 8, "orig_h": 8}], score_threshold=0.5
        )
        assert results[0]["masks"].shape == (0, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_decode_keeps_device(self):
        # Regression: the empty branch created CPU masks alongside GPU
        # scores/labels, mixing devices across batch results.
        masks = torch.full((1, 2, 4, 4), -10.0, device="cuda")
        classes = torch.full((1, 2, 3), -10.0, device="cuda")
        results = instance_segmentation_decode(
            masks, classes, [{"orig_w": 8, "orig_h": 8}], score_threshold=0.5
        )
        assert results[0]["masks"].device.type == "cuda"
        assert results[0]["labels"].device.type == "cuda"
        assert results[0]["scores"].device.type == "cuda"


class TestMaskAPEvaluator:
    def test_perfect_prediction(self):
        evaluator = COCOMaskAPEvaluator({0: "a", 1: "b"})
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)
        mask[0, 0, 0] = True
        evaluator.update_targets(
            [{"masks": mask, "labels": torch.tensor([0]), "size": (2, 2)}]
        )
        evaluator.update(
            [{"masks": mask, "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]
        )
        assert evaluator.compute()["mask_map"] == pytest.approx(1.0)

    def test_empty_returns_zeros(self):
        evaluator = COCOMaskAPEvaluator({0: "a"})
        assert evaluator.compute() == {"mask_map": 0.0, "mask_map_50": 0.0}

    def test_state_roundtrip(self):
        evaluator = COCOMaskAPEvaluator({0: "a"})
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        evaluator.update_targets(
            [{"masks": mask, "labels": torch.tensor([0]), "size": (2, 2)}]
        )
        evaluator.update(
            [{"masks": mask, "labels": torch.tensor([0]), "scores": torch.tensor([0.5])}]
        )
        other = COCOMaskAPEvaluator({0: "a"})
        other.merge_state(evaluator.export_state())
        assert other.compute()["mask_map"] == pytest.approx(1.0)


class TestCocoDataset:
    def test_loads_polygons_and_masks(self, tmp_path):
        annotations = write_coco(tmp_path, num_images=2)
        from PIL import Image

        for i in range(2):
            Image.fromarray(
                np.full((64, 64, 3), 128, dtype=np.uint8)
            ).save(tmp_path / f"img_{i}.jpg")
        dataset = COCOInstanceSegmentationDataset(annotations, images_dir=tmp_path)
        assert len(dataset) == 2
        item = dataset[0]
        assert item["masks"].shape == (2, 64, 64)
        assert item["classes"].tolist() == [0, 1]
        assert item["masks"][0, 4:20, 4:20].all()  # square instance
        assert item["bboxes"].shape == (2, 4)


class TestInstanceTrainModel:
    def test_training_step_backprop(self):
        train = make_train_model()
        result = train.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert train.model.model.class_predictor.weight.grad is not None

    def test_validation_updates_mask_ap(self):
        train = make_train_model()
        train.model.eval()
        result = train.validation_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        metrics = train.compute_metrics("val")
        assert set(metrics) == {"val_metric/mask_map", "val_metric/mask_map_50"}
        assert 0.0 <= metrics["val_metric/mask_map"] <= 1.0

    def test_metric_state_roundtrip(self):
        train = make_train_model()
        train.model.eval()
        train.validation_step(make_batch(), step=0)
        other = make_train_model()
        other.merge_metric_state("val", train.export_metric_state("val"))
        assert 0.0 <= other.compute_metrics("val")["val_metric/mask_map"] <= 1.0

    def test_class_var_defaults(self):
        assert Dinov2EomtInstanceSegmentationTrainArgs.default_batch_size == 16
        assert Dinov2EomtInstanceSegmentationTrainArgs.default_steps == 360_000

    def test_decode_args_forwarded_to_task_model(self):
        # Single source of truth: val decoding (model_args) and
        # predict/postprocess (task model attrs) must agree.
        args_kwargs = {
            "score_threshold": 0.42,
            "mask_threshold": 0.61,
            "num_top_queries": 7,
        }
        train = make_train_model(**args_kwargs)
        assert train.model.score_threshold == 0.42
        assert train.model.mask_threshold == 0.61
        assert train.model.num_top_queries == 7
        assert train.model_args.score_threshold == 0.42
        assert train.model_args.mask_threshold == 0.61
        assert train.model_args.num_top_queries == 7


class TestInstanceEndToEnd:
    def test_train_and_predict(self, tmp_path):
        from PIL import Image

        annotations = write_coco(tmp_path)
        for i in range(3):
            Image.fromarray(
                np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(tmp_path / f"img_{i}.jpg")
        result = train_instance_segmentation(
            out=tmp_path / "run",
            data={"train": {"annotations": annotations, "images": tmp_path}},
            model="dinov2/eomt",
            num_classes=2,
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()

        loaded = load_model(result.exported_model_path, device="cpu")
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["masks"].shape[1:] == (64, 64)
        assert prediction["masks"].dtype == torch.bool


class TestInstanceResolutionContract:
    def test_resize_masks_stays_binary(self):
        from autovision.metrics.instance_segmentation import resize_masks

        masks = torch.zeros(2, 4, 4, dtype=torch.bool)
        masks[:, :2, :2] = True
        resized = resize_masks(masks, 8, 6)
        assert resized.shape == (2, 8, 6)
        assert set(resized.unique().tolist()) <= {0.0, 1.0}

    def test_validation_resizes_gt_masks_to_original_size(self):
        # Regression: GT masks stay on the network canvas while predictions
        # are decoded to the original frame; COCO RLE then disagrees with the
        # declared image size.
        train = make_train_model()
        train.model.eval()
        batch = make_batch()
        batch["original_size"] = [(96, 48)] * 2
        train.validation_step(batch, step=0)
        state = train.export_metric_state("val")
        assert state["targets"][0]["masks"].shape[-2:] == (96, 48)
        metrics = train.compute_metrics("val")
        assert set(metrics) == {"val_metric/mask_map", "val_metric/mask_map_50"}
