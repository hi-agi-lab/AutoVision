# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection error analysis: matching, slices, gallery report (roadmap C2)."""

import json
import re
from pathlib import Path

import pytest
import torch
from tests.conftest import create_image

from autovision.cli.analyze_errors import analyze_errors
from autovision.metrics.error_analysis import (
    analyze_detection_errors,
    match_detections,
    summarize_matches,
)
from autovision.metrics.map_evaluator import COCOMAPEvaluator

CATEGORIES = {1: "cat", 2: "dog"}


def _pred(boxes, scores, labels):
    return {
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32).reshape(-1),
        "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
    }


def _target(boxes, labels, iscrowds=None):
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
    }
    if iscrowds is not None:
        target["iscrowds"] = torch.tensor(iscrowds, dtype=torch.long).reshape(-1)
    return target


def _kinds(matches):
    return [(m.kind, m.class_name) for m in matches]


class TestMatching:
    def test_perfect_match_is_tp(self):
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.9], [0])],
            [_target([[0, 0, 10, 10]], [0])],
            class_names={0: "cat"},
        )
        assert _kinds(matches) == [("tp", "cat")]
        assert matches[0].iou == pytest.approx(1.0)

    def test_low_iou_is_fp_plus_fn(self):
        # IoU 0.333 (inter 50 / union 150): same class, above the 0.1 floor.
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.9], [0])],
            [_target([[5, 0, 15, 10]], [0])],
            class_names={0: "cat"},
        )
        assert sorted(_kinds(matches)) == [("fn", "cat"), ("fp", "cat")]
        fp = next(m for m in matches if m.kind == "fp")
        assert fp.fp_reason == "localization"
        assert fp.iou == pytest.approx(1 / 3, abs=1e-5)

    def test_second_prediction_on_taken_gt_is_duplicate(self):
        matches = match_detections(
            [_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])],
            [_target([[0, 0, 10, 10]], [0])],
            class_names={0: "cat"},
        )
        tps = [m for m in matches if m.kind == "tp"]
        fps = [m for m in matches if m.kind == "fp"]
        assert len(tps) == 1 and len(fps) == 1
        assert fps[0].fp_reason == "duplicate"

    def test_box_on_other_class_gt_is_classification(self):
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.9], [0])],
            [_target([[0, 0, 10, 10]], [1])],
            class_names={0: "cat", 1: "dog"},
        )
        fp = next(m for m in matches if m.kind == "fp")
        assert fp.fp_reason == "classification"
        assert any(m.kind == "fn" and m.class_name == "dog" for m in matches)

    def test_far_box_is_background(self):
        matches = match_detections(
            [_pred([[100, 100, 110, 110]], [0.9], [0])],
            [_target([[0, 0, 10, 10]], [0])],
            class_names={0: "cat"},
        )
        fp = next(m for m in matches if m.kind == "fp")
        assert fp.fp_reason == "background"

    def test_crowd_gt_neither_matched_nor_missed(self):
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.9], [0])],
            [_target([[0, 0, 10, 10]], [0], iscrowds=[1])],
            class_names={0: "cat"},
        )
        # pycocotools dtIg semantics: a detection matching a crowd (ignore)
        # GT at IoU >= threshold is neither a TP nor an FP, and the crowd
        # box itself is never a FN.
        assert _kinds(matches) == [("ignored", "cat")]
        assert matches[0].fp_reason is None
        summary = summarize_matches(matches, {0: "cat"})
        assert summary["totals"]["fp"] == 0
        assert summary["totals"]["ignored"] == 1
        assert summary["totals"]["precision"] == 0.0  # 0 / (0 + 0)

    def test_stronger_regular_match_beats_a_weaker_crowd_hit(self):
        """pycocotools order: the BEST-IoU GT decides — a stronger
        other-class overlap keeps ``classification`` instead of ``ignored``."""
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.9], [0])],
            # Regular dog box: IoU 0.47; crowd cat box: IoU 0.33.
            [_target([[2, 2, 12, 12], [5, 0, 15, 10]], [1, 0], iscrowds=[0, 1])],
            class_names={0: "cat", 1: "dog"},
            iou_threshold=0.3,
        )
        fp = next(m for m in matches if m.kind in ("fp", "ignored"))
        assert fp.kind == "fp"
        assert fp.fp_reason == "classification"

    def test_score_threshold_drops_low_confidence(self):
        matches = match_detections(
            [_pred([[0, 0, 10, 10]], [0.2], [0])],
            [_target([[0, 0, 10, 10]], [0])],
            class_names={0: "cat"},
            score_threshold=0.5,
        )
        assert _kinds(matches) == [("fn", "cat")]


class TestSummarize:
    def test_counts_ratios_and_area_buckets(self):
        matches = match_detections(
            [
                _pred([[0, 0, 10, 10], [50, 50, 150, 150]], [0.9, 0.7], [0, 1]),
                _pred([[0, 0, 20, 20]], [0.9], [0]),
            ],
            [
                _target([[0, 0, 10, 10], [50, 50, 150, 150]], [0, 1]),  # 100 / 10000 px²
                _target([[5, 5, 25, 25]], [1]),  # 400 px², missed by class
            ],
        )
        summary = summarize_matches(matches, {0: "cat", 1: "dog"})
        assert summary["totals"]["tp"] == 2
        assert summary["totals"]["fn"] == 1  # the dog GT on image 1
        assert summary["totals"]["fp"] == 1  # image-2 prediction IOU 0.39 (localization)
        assert summary["totals"]["precision"] == pytest.approx(2 / 3)
        assert summary["totals"]["recall"] == pytest.approx(2 / 3)
        assert summary["totals"]["fp"] == 1
        assert summary["per_area"]["small"]["tp"] == 1  # 100 px²
        assert summary["per_area"]["small"]["fp"] == 1  # 400 px² < 32² = 1024
        assert summary["per_area"]["small"]["fn"] == 1
        assert summary["per_area"]["medium"]["tp"] == 0
        assert summary["per_area"]["large"]["tp"] == 1  # 10000 px² ≥ 96²

    def test_area_boundary_values_follow_coco(self):
        # 32² px² belongs to medium (small is [0, 32²)); 96² to large.
        matches = match_detections(
            [_pred([[0, 0, 32, 32], [0, 0, 96, 96]], [0.9, 0.8], [0, 1])],
            [_target([[0, 0, 32, 32], [0, 0, 96, 96]], [0, 1])],
        )
        summary = summarize_matches(matches)
        assert summary["per_area"]["small"]["tp"] == 0
        assert summary["per_area"]["medium"]["tp"] == 1
        assert summary["per_area"]["large"]["tp"] == 1


class TestReport:
    def test_class_names_follow_category_id_order(self, tmp_path):
        """Regression: internal ids rank ``sorted(categories.items())``, so
        names that do not sort in id order must not be swapped (COCO-shaped
        class lists: ids 1..N with alphabetical or arbitrary names)."""
        images = [create_image(tmp_path / "im" / "img_0.jpg", size=(64, 64), seed=0)]
        predictions = [_pred([], [], [])]
        targets = [_target([[0, 0, 10, 10]], [0])]  # internal 0 -> category 1
        report = analyze_detection_errors(
            predictions, targets, {1: "zebra", 2: "ant"}, images, tmp_path / "out"
        )
        assert list(report.summary["per_class"]) == ["zebra"]
        assert report.summary["per_class"]["zebra"]["fn"] == 1
        folders = {path.parent.name for path in report.gallery}
        assert "fn_zebra" in folders


    def test_report_and_gallery_links_resolve(self, tmp_path):
        images = [
            create_image(tmp_path / "im" / f"img_{i}.jpg", size=(64, 64), seed=i)
            for i in range(2)
        ]
        predictions = [
            _pred([[10, 10, 50, 50]], [0.9], [0]),          # TP cat
            _pred([[0, 0, 30, 30], [40, 40, 60, 60]], [0.8, 0.3], [1, 1]),  # dog FP+FP
        ]
        targets = [
            _target([[10, 10, 50, 50]], [0]),
            _target([[0, 0, 10, 10]], [0]),                  # small missed cat
        ]
        report = analyze_detection_errors(
            predictions, targets, {0: "cat", 1: "dog"}, images,
            tmp_path / "out", iou_threshold=0.5, top_k=2,
        )
        assert report.report_json.exists() and report.report_md.exists()
        assert report.gallery, "gallery must contain rendered overlays"
        for path in report.gallery:
            assert path.exists() and path.suffix == ".jpg"
        # Every md image link resolves on disk (the gate: 画廊可打开).
        text = report.report_md.read_text()
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        assert links, "report.md must embed the gallery"
        for link in links:
            assert (report.out_dir / link).exists(), link
        # fp_all / fn_all + per-class folders were produced.
        folders = {path.parent.name for path in report.gallery}
        assert {"fp_all", "fn_all", "fn_cat"} <= folders
        payload = json.loads(report.report_json.read_text())
        assert payload["summary"]["totals"]["tp"] == 1
        assert payload["summary"]["per_class"]["dog"]["fp"] == 2
        assert payload["summary"]["per_class"]["cat"]["fn"] == 1

    def test_crowd_hit_is_counted_as_ignored_not_an_error(self, tmp_path):
        """An ignore-region hit is reported separately and never presented
        as a FP/FN; when it shares an image with a real error it is rendered
        (grey IGNORE overlay) alongside it."""
        images = [create_image(tmp_path / "im" / "img_0.jpg", size=(64, 64), seed=0)]
        predictions = [
            _pred(
                [[0, 0, 20, 20], [10, 10, 40, 40], [55, 55, 63, 63]],
                [0.9, 0.85, 0.8],
                [0, 0, 0],
            )
        ]
        targets = [
            _target([[0, 0, 20, 20], [10, 10, 40, 40]], [0, 0], iscrowds=[1, 0])
        ]
        report = analyze_detection_errors(
            predictions, targets, {0: "cat"}, images, tmp_path / "out"
        )
        assert report.summary["totals"]["ignored"] == 1  # the crowd hit
        assert report.summary["totals"]["tp"] == 1  # the box on regular GT
        assert report.summary["totals"]["fp"] == 1  # the far-away box
        assert report.summary["totals"]["fn"] == 0
        assert report.gallery  # the FP gallery case renders (IGNORE overlay path)
        text = report.report_md.read_text()
        assert "ignore 区命中（crowd）1" in text

    def test_length_mismatch_fails_loud(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            analyze_detection_errors(
                [_pred([[0, 0, 5, 5]], [0.9], [0])], [], {0: "cat"}, [tmp_path / "x.jpg"],
                tmp_path / "out",
            )

    def test_offline_cli_mode_from_predictions_json(self, tmp_path):
        images = [
            create_image(tmp_path / "im" / f"img_{i}.jpg", size=(64, 64), seed=i)
            for i in range(2)
        ]
        annotations = {
            "images": [
                {"id": index + 1, "file_name": Path(path).name, "width": 64, "height": 64}
                for index, path in enumerate(images)
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        }
        annotations_path = tmp_path / "ann.json"
        annotations_path.write_text(json.dumps(annotations))
        records = [
            {"image": str(images[0]), "bboxes": [[10, 10, 50, 50]],
             "scores": [0.9], "labels": [0]},  # cat TP (internal 0)
            {"image": str(images[1]), "bboxes": [],
             "scores": [], "labels": []},      # dog missed
        ]
        predictions_path = tmp_path / "preds.json"
        predictions_path.write_text(json.dumps(records))
        report = analyze_errors(
            predictions=predictions_path, annotations=annotations_path,
            out=tmp_path / "offline_report",
        )
        assert report.summary["totals"]["tp"] == 1
        assert report.summary["totals"]["fn"] == 1
        assert report.summary["per_class"]["dog"]["fn"] == 1
        assert (report.out_dir / "gallery" / "fn_all").is_dir()

    def test_offline_cli_mode_needs_both_inputs(self, tmp_path):
        with pytest.raises(Exception):
            analyze_errors(predictions=tmp_path / "p.json", out=tmp_path / "r")


class TestEvaluatorAreaSlices:
    def test_area_keys_present_and_consistent(self):
        evaluator = COCOMAPEvaluator(CATEGORIES)
        boxes = torch.tensor([[0.15, 0.15, 0.2, 0.2]])  # 40x40 px on 200² frame
        evaluator.update_targets(
            [{"boxes": boxes, "labels": torch.tensor([0])}], original_sizes=[(200, 200)]
        )
        evaluator.update(
            [_pred([[10, 10, 50, 50]], [0.9], [0])]
        )
        metrics = evaluator.compute()
        for key in ("map_small", "map_medium", "map_large"):
            assert key in metrics, key
        # 40² = 1600 px² sits in the medium bucket; the perfect detection
        # scores there and the other buckets have no GT (pycocotools: -1).
        assert metrics["map_medium"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["map_small"] == -1.0
        assert metrics["map_large"] == -1.0

    def test_empty_evaluator_has_area_keys(self):
        metrics = COCOMAPEvaluator(CATEGORIES).compute()
        assert metrics["map_small"] == 0.0
        assert metrics["map_medium"] == 0.0
        assert metrics["map_large"] == 0.0
