# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the oriented RT-DETR (OBB) family: ProbIoU, decoder, criterion,
matcher, decode, metric, task model, and end-to-end training."""

from __future__ import annotations

import math

import pytest
import torch
from tests.conftest import create_image

from autovision.geometry.obb import (
    probiou_matrix,
    xywhr_to_xyxyxyxy,
)
from autovision.metrics.oriented_detection import (
    OrientedDetectionMetricCollector,
)
from autovision.tasks.oriented_detection.rtdetr.task_model import (
    RTDETROrientedObjectDetection,
)
from autovision.tasks.oriented_detection.shared.obb_criterion import (
    OrientedHungarianMatcher,
    OrientedRTDETRCriterion,
    loss_boxes_oriented,
)
from autovision.tasks.oriented_detection.shared.obb_decoder import (
    OrientedRTDETRDecoder,
    decode_oriented_detection_outputs,
)

# Normalized square corners in a 64x64 frame: (16,16)-(32,32) px.
SQ_NORM = [0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]


def tiny_decoder() -> OrientedRTDETRDecoder:
    return OrientedRTDETRDecoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=10,
        num_layers=2,
        nhead=4,
        num_denoising=10,
        feat_channels=[32, 32, 32],
    )


def tiny_feats(batch_size: int = 2):
    return [
        torch.randn(batch_size, 32, 8, 8),
        torch.randn(batch_size, 32, 4, 4),
        torch.randn(batch_size, 32, 2, 2),
    ]


def tiny_targets(batch_size: int = 2):
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]])
    return [
        {
            "boxes": boxes[:1],
            "angles": torch.tensor([[0.1]]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": boxes[1:],
            "angles": torch.tensor([[0.2]]),
            "labels": torch.tensor([0]),
        },
    ][:batch_size]


class TestProbIoU:
    def test_identical_boxes_score_one(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.1]])
        assert torch.allclose(probiou_matrix(boxes, boxes), torch.ones(1, 1))

    def test_monotone_decay_with_displacement(self):
        base = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.0]])
        shifted = torch.tensor([[0.55, 0.5, 0.2, 0.2, 0.0], [0.7, 0.5, 0.2, 0.2, 0.0]])
        scores = probiou_matrix(base, shifted)[0]
        assert scores[0] > scores[1]
        assert scores[1] < 1.0

    def test_symmetry(self):
        a = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.1], [0.2, 0.7, 0.1, 0.1, -0.3]])
        b = torch.tensor([[0.45, 0.5, 0.2, 0.3, 0.2], [0.6, 0.6, 0.3, 0.2, 0.0]])
        assert torch.allclose(probiou_matrix(a, b), probiou_matrix(b, a).T, atol=1e-6)

    def test_gradient_flows_to_angle_and_size(self):
        boxes = torch.tensor(
            [[0.5, 0.5, 0.2, 0.3, 0.1]], requires_grad=True
        )
        target = torch.tensor([[0.55, 0.5, 0.25, 0.3, 0.2]])
        probiou_matrix(boxes, target).sum().backward()
        assert boxes.grad is not None
        assert bool(torch.isfinite(boxes.grad).all())
        assert float(boxes.grad.abs().sum()) > 0


class TestOrientedDecoder:
    def test_training_outputs_carry_angles_everywhere(self):
        decoder = tiny_decoder()
        decoder.train()
        out = decoder(tiny_feats(), tiny_targets())
        assert out["pred_angles"].shape == (2, 10, 1)
        assert out["aux_outputs"][0]["pred_angles"].shape == (2, 10, 1)
        assert out["dn_aux_outputs"][0]["pred_angles"].shape[0] == 2
        assert out["pred_angles"].dtype == torch.float32

    def test_zero_init_angles(self):
        decoder = tiny_decoder()
        decoder.train()
        out = decoder(tiny_feats(), tiny_targets())
        assert float(out["pred_angles"].abs().max()) == 0.0

    def test_angle_range_bounded(self):
        decoder = tiny_decoder()
        decoder.eval()
        with torch.no_grad():
            out = decoder(tiny_feats())
        bound = math.pi / 4 + 1e-6
        assert float(out["pred_angles"].abs().max()) <= bound

    def test_eval_mode_returns_pair_free_tuples(self):
        decoder = tiny_decoder()
        decoder.eval()
        with torch.no_grad():
            out = decoder(tiny_feats())
        assert set(out) == {"pred_logits", "pred_boxes", "pred_angles"}
        assert "aux_outputs" not in out

    def test_angle_head_receives_gradients(self):
        decoder = tiny_decoder()
        decoder.train()
        out = decoder(tiny_feats(), tiny_targets())
        criterion = OrientedRTDETRCriterion(OrientedHungarianMatcher())
        outputs_list = out["aux_outputs"] + [
            {
                "pred_logits": out["pred_logits"],
                "pred_boxes": out["pred_boxes"],
                "pred_angles": out["pred_angles"],
            }
        ]
        losses = criterion(
            outputs_list, tiny_targets(),
            dn_outputs=out["dn_aux_outputs"], dn_meta=out["dn_meta"],
        )
        losses["loss"].backward()
        grad = decoder.dec_angle_head[0].layers[-1].weight.grad
        assert grad is not None and bool((grad != 0).any())


class TestOrientedCriterionAndMatcher:
    def test_zero_box_loss_when_pred_matches_target(self):
        outputs = {
            "pred_logits": torch.zeros(1, 4, 2),
            "pred_boxes": torch.full((1, 4, 4), 0.05),
            "pred_angles": torch.zeros(1, 4, 1),
        }
        outputs["pred_logits"][0, 2, 1] = 5.0
        outputs["pred_boxes"][0, 2] = torch.tensor([0.5, 0.5, 0.2, 0.3])
        outputs["pred_angles"][0, 2, 0] = 0.1
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
                "angles": torch.tensor([[0.1]]),
                "labels": torch.tensor([1]),
            }
        ]
        indices = OrientedHungarianMatcher()(outputs, targets)
        assert len(indices[0][0]) == 1 and int(indices[0][0][0]) == 2
        losses = loss_boxes_oriented(outputs, targets, indices, torch.tensor(1.0))
        assert float(losses["loss_bbox"].abs()) < 1e-6
        assert float(losses["loss_giou"].abs()) < 1e-5

    def test_matcher_prefers_oriented_overlap(self):
        # Two queries with identical axis-aligned boxes but different angles:
        # the matcher must pick the one matching the GT angle.
        box = torch.tensor([0.5, 0.5, 0.4, 0.1])
        outputs = {
            "pred_logits": torch.zeros(1, 2, 1),
            "pred_boxes": box.expand(1, 2, 4).clone(),
            "pred_angles": torch.tensor([[[0.4], [-0.4]]]),
        }
        targets = [
            {
                "boxes": box.view(1, 4),
                "angles": torch.tensor([[0.4]]),
                "labels": torch.tensor([0]),
            }
        ]
        indices = OrientedHungarianMatcher()(outputs, targets)
        assert int(indices[0][0][0]) == 0

    def test_criterion_full_forward_with_dn(self):
        decoder = tiny_decoder()
        decoder.train()
        out = decoder(tiny_feats(), tiny_targets())
        criterion = OrientedRTDETRCriterion(OrientedHungarianMatcher())
        outputs_list = out["aux_outputs"] + [
            {
                "pred_logits": out["pred_logits"],
                "pred_boxes": out["pred_boxes"],
                "pred_angles": out["pred_angles"],
            }
        ]
        losses = criterion(
            outputs_list, tiny_targets(),
            dn_outputs=out["dn_aux_outputs"], dn_meta=out["dn_meta"],
        )
        assert any(key.endswith("_aux_0") for key in losses)
        assert any(key.endswith("_dn_0") for key in losses)
        assert torch.isfinite(losses["loss"]).all()


class TestOrientedDecode:
    def _raw(self, angles):
        logits = torch.full((1, 4, 2), -10.0)
        logits[0, 0, 0] = 10.0
        logits[0, 1, 1] = 10.0
        boxes = torch.zeros(1, 4, 4)
        boxes[0, 0] = torch.tensor([0.25, 0.25, 0.2, 0.2])
        boxes[0, 1] = torch.tensor([0.25, 0.25, 0.2, 0.2])
        return logits, boxes, angles.view(1, 4, 1)

    def test_decode_polygon_geometry(self):
        raw = self._raw(torch.tensor([0.0, 0.3, 0.0, 0.0]))
        results = decode_oriented_detection_outputs(
            raw, [{"orig_w": 64, "orig_h": 64}],
            num_top_queries=4, threshold=0.5,
        )
        result = results[0]
        assert set(result) == {"labels", "polygons", "scores"}
        assert result["labels"].tolist() == [0, 1]
        assert result["polygons"].shape == (2, 4, 2)
        expected = xywhr_to_xyxyxyxy(
            torch.tensor([[16.0, 16.0, 12.8, 12.8, 0.0]])
        )[0]
        poly0 = result["polygons"][0]
        # Same 4-vertex set regardless of winding/start ordering.
        assert torch.allclose(
            poly0[poly0[:, 0].argsort()], expected[expected[:, 0].argsort()],
            atol=1e-4,
        )

    def test_nms_suppresses_same_class_keeps_other_class(self):
        angles = torch.tensor([0.0, 0.0, 0.0, 0.0])
        same_class = self._raw(angles)  # both detections are class 0/1... fix below
        # Force both top queries into class 0 (heavy overlap -> NMS).
        logits, boxes, angles_t = same_class
        logits[0, 1, 1] = -10.0
        logits[0, 1, 0] = 10.0
        kept = decode_oriented_detection_outputs(
            (logits, boxes, angles_t), [{"orig_w": 64, "orig_h": 64}],
            num_top_queries=4, threshold=0.5, iou_threshold=0.1,
        )[0]
        assert kept["labels"].numel() == 1

    def test_nms_keeps_other_class(self):
        angles = torch.tensor([0.0, 0.0, 0.0, 0.0])
        logits, boxes, angles_t = self._raw(angles)  # classes 0 and 1
        kept = decode_oriented_detection_outputs(
            (logits, boxes, angles_t), [{"orig_w": 64, "orig_h": 64}],
            num_top_queries=4, threshold=0.5, iou_threshold=0.1,
        )[0]
        assert sorted(kept["labels"].tolist()) == [0, 1]


class TestOrientedMetric:
    def _gt_corners(self) -> torch.Tensor:
        return torch.tensor([SQ_NORM], dtype=torch.float32)

    def _pred_exact(self) -> dict:
        polys = torch.tensor([[[16.0, 16.0], [32.0, 16.0], [32.0, 32.0], [16.0, 32.0]]])
        return {
            "labels": torch.tensor([0]),
            "polygons": polys,
            "scores": torch.tensor([0.9]),
        }

    def test_perfect_predictions_map_one(self):
        evaluator = OrientedDetectionMetricCollector({0: "plane", 1: "ship"})
        evaluator.update_targets(
            [{"corners": self._gt_corners(), "labels": torch.tensor([0])}],
            [(64, 64)],
        )
        evaluator.update([self._pred_exact()])
        metrics = evaluator.compute()
        assert metrics["map"] == pytest.approx(1.0)
        assert metrics["map_50"] == pytest.approx(1.0)

    def test_missed_object_scores_zero(self):
        evaluator = OrientedDetectionMetricCollector({0: "plane", 1: "ship"})
        evaluator.update_targets(
            [{"corners": self._gt_corners(), "labels": torch.tensor([0])}],
            [(64, 64)],
        )
        pred = self._pred_exact()
        pred["polygons"] = pred["polygons"] * 0 + torch.tensor(
            [[[50.0, 50.0], [60.0, 50.0], [60.0, 60.0], [50.0, 60.0]]]
        )
        evaluator.update([pred])
        assert evaluator.compute()["map"] == pytest.approx(0.0)

    def test_merge_state_matches_single_evaluator(self):
        targets = [
            {"corners": self._gt_corners(), "labels": torch.tensor([0])},
            {"corners": self._gt_corners(), "labels": torch.tensor([0])},
        ]
        preds = [self._pred_exact(), self._pred_exact()]
        single = OrientedDetectionMetricCollector({0: "plane"})
        single.update_targets(targets, [(64, 64), (64, 64)])
        single.update(preds)
        merged = OrientedDetectionMetricCollector({0: "plane"})
        first, second = OrientedDetectionMetricCollector({0: "plane"}), None
        first.update_targets(targets[:1], [(64, 64)])
        first.update(preds[:1])
        second = OrientedDetectionMetricCollector({0: "plane"})
        second.update_targets(targets[1:], [(64, 64)])
        second.update(preds[1:])
        merged.merge_state(first.export_state())
        merged.merge_state(second.export_state())
        assert merged.compute() == pytest.approx(single.compute())


class TestOrientedTaskModel:
    def test_decoder_replaced_with_oriented_variant(self):
        model = RTDETROrientedObjectDetection(
            num_classes=2,
            backbone="torchvision/resnet18",
            image_size=(64, 64),
            hidden_dim=32,
            num_queries=10,
            dec_layers=2,
            num_heads=4,
            num_top_queries=10,
            num_denoising=10,
        )
        assert isinstance(model.decoder, OrientedRTDETRDecoder)
        assert len(model.decoder.dec_angle_head) == 2
        assert model.decoder.dec_angle_head[0].layers[-1].bias.abs().sum() == 0

    def test_forward_returns_three_tensors(self):
        model = RTDETROrientedObjectDetection(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=10, dec_layers=2, num_heads=4,
            num_top_queries=10, num_denoising=10,
        )
        model.eval()
        with torch.no_grad():
            logits, boxes, angles = model(torch.rand(1, 3, 64, 64))
        assert logits.shape == (1, 10, 2)
        assert boxes.shape == (1, 10, 4)
        assert angles.shape == (1, 10, 1)

    def test_postprocess_returns_polygons(self):
        model = RTDETROrientedObjectDetection(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=10, dec_layers=2, num_heads=4,
            num_top_queries=10, num_denoising=10,
        )
        model.eval()
        with torch.no_grad():
            raw = model(torch.rand(1, 3, 64, 64))
        results = model.postprocess(
            raw, [{"orig_w": 64, "orig_h": 64}], threshold=0.0
        )
        assert set(results[0]) == {"labels", "polygons", "scores"}


# ── end to end ──────────────────────────────────────────────────────────


def make_yolo_obb_dataset(root):
    root = root / "obbroot"
    names = {"0": "plane", "1": "ship"}
    sq_line = "0 " + " ".join(f"{v:g}" for v in SQ_NORM)
    num = {"train": 4, "val": 2}
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index in range(num[split]):
            create_image(images_dir / f"img_{index:03d}.jpg", size=(64, 64), seed=index)
            (labels_dir / f"img_{index:03d}.txt").write_text(sq_line)
    return {
        "format": "yolo-obb",
        "path": str(root),
        "train": str(root / "images" / "train"),
        "val": str(root / "images" / "val"),
        "names": names,
    }


def e2e_model_args():
    return {
        "backbone": "torchvision/resnet18",
        "image_size": (64, 64),
        "hidden_dim": 32,
        "num_queries": 10,
        "dec_layers": 2,
        "num_heads": 4,
        "num_top_queries": 10,
    }


def e2e_transform_args():
    return {"photometric_distort_prob": None, "horizontal_flip_prob": 0.0}


@pytest.fixture(scope="module")
def obb_data(tmp_path_factory):
    return make_yolo_obb_dataset(tmp_path_factory.mktemp("obb"))


class TestTrainOrientedObjectDetection:
    def test_full_run_produces_all_artifacts(self, tmp_path, obb_data):
        from autovision import train_oriented_object_detection

        result = train_oriented_object_detection(
            out=tmp_path / "obb1",
            data=obb_data,
            steps=3,
            batch_size=4,
            num_workers=0,
            model_args=e2e_model_args(),
            transform_args=e2e_transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 3
        assert result.checkpoint_path and result.checkpoint_path.exists()
        assert result.exported_model_path and result.exported_model_path.exists()
        assert (tmp_path / "obb1" / "metrics.jsonl").exists()
        assert result.classes == {0: "plane", 1: "ship"}
        assert result.best_checkpoint_path and result.best_checkpoint_path.exists()
        assert "val_metric/map" in result.best_metrics

    def test_exported_model_loads_and_predicts(self, tmp_path, obb_data):
        from autovision import load_task_model as load_model
        from autovision import train_oriented_object_detection

        result = train_oriented_object_detection(
            out=tmp_path / "obb2",
            data=obb_data,
            steps=2,
            batch_size=4,
            num_workers=0,
            model_args=e2e_model_args(),
            transform_args=e2e_transform_args(),
        )
        model = load_model(result.exported_model_path, device="cpu")
        assert not model.training
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        prediction = model.predict(image, threshold=0.0)
        assert set(prediction) == {"labels", "polygons", "scores"}
        assert prediction["polygons"].shape[1:] == (4, 2)

    def test_deterministic_double_run(self, tmp_path, obb_data):
        import random as _random

        from autovision import train_oriented_object_detection

        runs = []
        # Pin cuDNN to deterministic conv algorithms to remove one source of
        # float noise, restoring the global flags afterwards so other tests
        # keep their default backend behavior.  The remaining drift comes
        # from ops whose CUDA backwards are inherently nondeterministic
        # (grid_sample in the deformable attention, index_put_ in the
        # criterion use atomic accumulation), so structural metrics must
        # match exactly while float losses get a tight tolerance.
        cudnn_flags = (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            for run in ("obb3a", "obb3b"):
                # Model weights are drawn at construction time inside the
                # command, so seed the process RNGs before each run.
                torch.manual_seed(0)
                _random.seed(0)
                result = train_oriented_object_detection(
                    out=tmp_path / run,
                    data=obb_data,
                    steps=2,
                    batch_size=4,
                    num_workers=0,
                    model_args=e2e_model_args(),
                    transform_args=e2e_transform_args(),
                )
                runs.append(result.best_metrics)
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        assert set(runs[0]) == set(runs[1])
        for key, value in runs[0].items():
            other = runs[1][key]
            if isinstance(value, float):
                assert abs(value - other) <= 1e-4 * max(1.0, abs(value)), (key, value, other)
            else:
                assert value == other, (key, value, other)

    def test_unknown_model_raises(self, tmp_path, obb_data):
        from autovision import train_oriented_object_detection

        with pytest.raises(ValueError, match="Unsupported oriented detection model"):
            train_oriented_object_detection(
                out=tmp_path / "obb4",
                data=obb_data,
                model="rtdetr/resnet18",
                steps=2,
                batch_size=4,
                num_workers=0,
                model_args=e2e_model_args(),
            )

    def test_unknown_format_raises(self, tmp_path, obb_data):
        from autovision import train_oriented_object_detection

        with pytest.raises(ValueError, match="Unknown oriented detection data format"):
            train_oriented_object_detection(
                out=tmp_path / "obb5",
                data={"format": "coco"},
                steps=2,
                batch_size=4,
                num_workers=0,
                model_args=e2e_model_args(),
            )

    def test_checkpoint_continuation(self, tmp_path, obb_data):
        from autovision import train_oriented_object_detection

        common = dict(
            data=obb_data, batch_size=4, num_workers=0,
            model_args=e2e_model_args(), transform_args=e2e_transform_args(),
        )
        first = train_oriented_object_detection(out=tmp_path / "obb6a", steps=2, **common)
        result = train_oriented_object_detection(
            out=tmp_path / "obb6a", steps=4, resume_interrupted=True, **common
        )
        assert result.steps_run == 2  # continued, not restarted
        assert first.checkpoint_path.exists()
