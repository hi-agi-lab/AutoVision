# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the classification and segmentation task paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from autovision import (
    load_task_model,
    train_image_classification,
    train_semantic_segmentation,
)
from autovision.data.image_classification_dataset import (
    ImageClassificationDataset,
    list_classification_files,
)
from autovision.data.semantic_segmentation_dataset import (
    SemanticSegmentationDataset,
    list_segmentation_pairs,
)
from autovision.metrics.classification import ClassificationMetricCollector
from autovision.metrics.segmentation import SegmentationMetricCollector
from autovision.tasks.classification.config import ImageClassificationTrainArgs
from autovision.tasks.semantic_segmentation.linear.config import (
    LinearSemanticSegmentationTrainArgs,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _write_classification_tree(root: Path, per_class: int = 4, size: int = 24) -> None:
    from tests.conftest import create_image

    for class_index, class_name in enumerate(("alpha", "beta")):
        for item in range(per_class):
            create_image(
                root / class_name / f"img_{item}.jpg",
                size=(size, size),
                seed=class_index * 10 + item,
            )


def _write_segmentation_tree(
    images: Path, masks: Path, num_pairs: int = 3, size: int = 32
) -> None:
    rng = np.random.default_rng(0)
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in range(num_pairs):
        array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(images / f"seg_{index}.jpg")
        mask = rng.integers(0, 3, size=(size, size), dtype=np.uint8)
        Image.fromarray(mask).save(masks / f"seg_{index}.png")


class TestAutoDefaultResolution:
    def test_classification_auto_defaults_use_task_config(self):
        """``auto`` must read the task's declared class-level defaults."""
        assert ImageClassificationTrainArgs.default_batch_size == 128
        assert ImageClassificationTrainArgs.default_steps == 10_000
        assert "default_batch_size" not in ImageClassificationTrainArgs.model_fields
        assert "default_steps" not in ImageClassificationTrainArgs.model_fields

    def test_linear_segmentation_auto_defaults_use_task_config(self):
        """Linear segmentation overrides must not fall back to the base 16/10k."""
        assert LinearSemanticSegmentationTrainArgs.default_batch_size == 32
        assert LinearSemanticSegmentationTrainArgs.default_steps == 20_000
        assert (
            "default_batch_size"
            not in LinearSemanticSegmentationTrainArgs.model_fields
        )
        assert "default_steps" not in LinearSemanticSegmentationTrainArgs.model_fields


class TestClassificationData:
    def test_list_classification_files(self, tmp_path: Path):
        root = tmp_path / "train"
        _write_classification_tree(root)
        files, labels, classes = list_classification_files(root)
        assert len(files) == 8
        assert set(classes.values()) == {"alpha", "beta"}
        assert set(labels) == {"alpha", "beta"}

    def test_single_class_rejected(self, tmp_path: Path):
        root = tmp_path / "train"
        _write_classification_tree(root)
        import shutil

        shutil.rmtree(root / "beta")
        with pytest.raises(ValueError, match="two class"):
            list_classification_files(root)

    def test_dataset_returns_image_label(self, tmp_path: Path):
        root = tmp_path / "train"
        _write_classification_tree(root)
        files, labels, classes = list_classification_files(root)
        dataset = ImageClassificationDataset(files, labels, classes)
        item = dataset[0]
        assert item["image"].dtype == torch.uint8
        assert isinstance(item["label"], int)


class TestClassificationMetric:
    def test_accuracy_exact(self):
        collector = ClassificationMetricCollector()
        # 4 samples: logits favor the correct class in 3 of them.
        logits = torch.tensor(
            [
                [3.0, 0.0, 0.0],  # correct -> 0
                [0.0, 3.0, 0.0],  # correct -> 1
                [0.0, 3.0, 0.0],  # correct -> 1
                [3.0, 0.0, 0.0],  # wrong (label 2)
            ]
        )
        labels = torch.tensor([0, 1, 1, 2])
        collector.update(logits, labels)
        result = collector.compute()
        assert result["accuracy_top1"] == pytest.approx(0.75)
        assert result["accuracy_top5"] == pytest.approx(1.0)

    def test_reset_and_state_roundtrip(self):
        first = ClassificationMetricCollector()
        # Deterministic: every label is 0 and the logits favor class 0.
        logits = torch.tensor([[3.0, 0.0, 0.0]] * 4)
        first.update(logits, torch.zeros(4, dtype=torch.long))
        second = ClassificationMetricCollector()
        second.merge_state(first.export_state())
        first.reset()
        assert second.compute()["accuracy_top1"] == pytest.approx(1.0)


class TestClassificationCommand:
    def test_end_to_end(self, tmp_path: Path):
        train_root = tmp_path / "train"
        val_root = tmp_path / "val"
        _write_classification_tree(train_root)
        _write_classification_tree(val_root)
        result = train_image_classification(
            out=tmp_path / "run",
            data={"train": train_root, "val": val_root},
            model="torchvision/resnet18",
            steps=4,
            batch_size=4,
            num_workers=0,
            model_args={"image_size": (32, 32), "lr_warmup_steps": 2},
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
        assert "val_metric/accuracy_top1" in result.best_metrics
        assert 0.0 <= result.best_metrics["val_metric/accuracy_top1"] <= 1.0

    def test_exported_model_predicts(self, tmp_path: Path):
        train_root = tmp_path / "train"
        _write_classification_tree(train_root)
        result = train_image_classification(
            out=tmp_path / "run",
            data={"train": train_root},
            model="torchvision/resnet18",
            steps=2,
            batch_size=4,
            num_workers=0,
            model_args={"image_size": (32, 32), "lr_warmup_steps": 2},
        )
        model = load_task_model(result.exported_model_path, device="cpu")
        from tests.conftest import create_image

        image = create_image(tmp_path / "probe.jpg", size=(24, 24), seed=42)
        prediction = model.predict(image)
        assert set(prediction["labels"].tolist()) <= {0, 1}

    def test_num_classes_mismatch_rejected(self, tmp_path: Path):
        train_root = tmp_path / "train"
        _write_classification_tree(train_root)
        with pytest.raises(ValueError, match="num_classes"):
            train_image_classification(
                out=tmp_path / "run",
                data={"train": train_root},
                num_classes=5,
                steps=2,
                batch_size=2,
                num_workers=0,
                model_args={"image_size": (32, 32)},
            )


class TestSegmentationData:
    def test_list_pairs(self, tmp_path: Path):
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        pairs, num_classes_detected = list_segmentation_pairs(images, masks)
        assert len(pairs) == 3
        assert num_classes_detected == 3  # labels 0..2

    def test_missing_mask_raises(self, tmp_path: Path):
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        for mask_path in masks.glob("*.png"):
            mask_path.unlink()
        with pytest.raises(ValueError, match="pairs"):
            list_segmentation_pairs(images, masks)

    def test_dataset_shapes(self, tmp_path: Path):
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        pairs, _ = list_segmentation_pairs(images, masks)
        dataset = SemanticSegmentationDataset(pairs)
        item = dataset[0]
        assert item["image"].dtype == torch.uint8
        assert item["mask"].dtype == torch.long


class TestSegmentationMetric:
    def test_miou_exact(self):
        collector = SegmentationMetricCollector(num_classes=2)
        # One missed pixel: predictions all 0, targets have one 1.
        predictions = torch.tensor([[[0, 0], [0, 0]]])
        targets = torch.tensor([[[0, 0], [0, 1]]])
        collector.update(predictions, targets)
        result = collector.compute()
        # Class 0: intersection 3, union 4 (the false positive counts) -> 0.75.
        # Class 1: intersection 0, union 1 -> 0.
        assert result["miou"] == pytest.approx(0.375)
        assert result["pixel_accuracy"] == pytest.approx(0.75)

    def test_perfect_prediction(self):
        collector = SegmentationMetricCollector(num_classes=2)
        predictions = torch.tensor([[[0, 1], [1, 0]]])
        targets = torch.tensor([[[0, 1], [1, 0]]])
        collector.update(predictions, targets)
        assert collector.compute()["miou"] == pytest.approx(1.0)

    def test_ignore_index_excluded(self):
        collector = SegmentationMetricCollector(num_classes=2)
        predictions = torch.tensor([[[0, 0]]])
        targets = torch.tensor([[[0, 255]]])
        collector.update(predictions, targets)
        assert collector.compute()["pixel_accuracy"] == pytest.approx(1.0)

    def test_state_roundtrip(self):
        first = SegmentationMetricCollector(num_classes=2)
        first.update(torch.ones(1, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, dtype=torch.long))
        second = SegmentationMetricCollector(num_classes=2)
        second.merge_state(first.export_state())
        miou = second.compute()["miou"]
        assert 0.0 <= miou <= 1.0


class TestSegmentationCommand:
    def test_end_to_end(self, tmp_path: Path):
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        result = train_semantic_segmentation(
            out=tmp_path / "run",
            data={"train": {"images": images, "masks": masks}},
            model="torchvision/resnet18",
            num_classes=3,
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={"image_size": (32, 32), "lr_warmup_steps": 2},
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()

    def test_label_out_of_range_rejected(self, tmp_path: Path):
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        _write_segmentation_tree(images, masks)
        with pytest.raises(ValueError, match="num_classes"):
            train_semantic_segmentation(
                out=tmp_path / "run",
                data={"train": {"images": images, "masks": masks}},
                num_classes=2,  # masks contain labels up to 2
                steps=2,
                batch_size=2,
                num_workers=0,
                model_args={"image_size": (32, 32)},
            )


class TestReviewFixes:
    def test_ignore_label_not_counted_as_class(self, tmp_path: Path):
        import numpy as np
        from PIL import Image as PILImage

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        rng = np.random.default_rng(0)
        PILImage.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
            images / "a.jpg"
        )
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[0, 0] = 255  # ignore convention must not count as a class
        PILImage.fromarray(mask).save(masks / "a.png")
        pairs, num_classes_detected = list_segmentation_pairs(images, masks)
        assert len(pairs) == 1
        assert num_classes_detected == 1

    def test_parity_requires_torch_backend(self, tmp_path: Path):
        from autovision.export import benchmark
        from autovision.export.benchmark import ONNXBackendArgs

        with pytest.raises(ValueError, match="requires the 'torch' backend"):
            benchmark(
                "rtdetr/resnet18",
                tmp_path / "bench",
                num_classes=2,
                image_size=(64, 64),
                backends={"onnx": ONNXBackendArgs()},
                verify_parity=True,
            )


def test_load_task_model_restores_normalize_args(tmp_path: Path):
    from autovision.tasks.classification.task_model import ImageClassification

    torch.manual_seed(0)
    net = ImageClassification(
        num_classes=2, image_size=(32, 32), load_weights=False
    )
    path = tmp_path / "exported.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "task_model",
            "model_class_path": net.class_path,
            "model_init_args": net.init_args,
            "train_model": net.state_dict(),
            "normalize_args": {
                "mean": [0.1, 0.2, 0.3],
                "std": [0.4, 0.5, 0.6],
            },
            "version": "test",
        },
        path,
    )
    loaded = load_task_model(path, device="cpu")
    assert tuple(loaded.normalize.mean) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(loaded.normalize.std) == pytest.approx((0.4, 0.5, 0.6))
