# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""COCO keypoint data plane + official AP metric + pose CLI (doc 31 §11 S5b).

Covers: the official ``CocoPoseDetection`` split/annotation filters and
``PoseNormalize`` target conversion, the COCO hflip left/right keypoint
swap, the pycocotools keypoint evaluator (perfect / degraded / ignored-GT
semantics; parity with pycocotools' own ``stats``), and the
``train-pose-estimation`` closure with the AP metric driving validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tests.conftest import create_image

K = 17


# ── synthetic COCO person_keypoints fixture ─────────────────────────────
def _keypoints(offset: float = 0.0, visibility: int = 2) -> list[float]:
    flat: list[float] = []
    for index in range(K):
        flat += [10.0 + index + offset, 20.0 + index + offset, visibility]
    return flat


def make_keypoint_dataset(
    root: Path,
    *,
    num_images: int = 3,
    image_size: tuple[int, int] = (100, 80),  # (height, width), numpy convention
    seed: int = 0,
) -> dict:
    """COCO ``person_keypoints`` layout:

    - image 1: one fully-labelled person (+ one crowd person, one
      ``num_keypoints == 0`` person that both the dataset and pycocotools
      must filter / ignore);
    - image 2: person with ``num_keypoints == 0`` only (train-split filter
      drops the image; val keeps it);
    - image 3: no annotations at all (val keeps it as a pure negative).
    """
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_names = []
    for index in range(num_images):
        path = create_image(
            images_dir / f"img{index + 1}.jpg", size=image_size, seed=seed + index
        )
        file_names.append(path.name)
    images = [
        {
            "id": index + 1,
            "file_name": file_names[index],
            "width": image_size[1],
            "height": image_size[0],
        }
        for index in range(num_images)
    ]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [5.0, 10.0, 60.0, 50.0],
            "area": 3000.0,
            "iscrowd": 0,
            "num_keypoints": K,
            "keypoints": _keypoints(),
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [70.0, 10.0, 25.0, 60.0],
            "area": 1500.0,
            "iscrowd": 1,
            "num_keypoints": K,
            "keypoints": _keypoints(offset=3.0),
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "bbox": [40.0, 5.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
            "num_keypoints": 0,
            "keypoints": [0.0] * (3 * K),
        },
        {
            "id": 4,
            "image_id": 2,
            "category_id": 1,
            "bbox": [10.0, 10.0, 30.0, 30.0],
            "area": 900.0,
            "iscrowd": 0,
            "num_keypoints": 0,
            "keypoints": [0.0] * (3 * K),
        },
    ]
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    annotations_path = root / "person_keypoints.json"
    annotations_path.write_text(json.dumps(coco))
    return {
        "annotations": annotations_path,
        "images": images_dir,
        "coco": coco,
        "image_size": image_size,
    }


def _build_dataset(spec: dict, train: bool, image_size=(64, 64)):
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset
    from autovision.tasks.pose_estimation.gtr.config import (
        GTRPoseEstimationTrainArgs,
    )
    from autovision.tasks.pose_estimation.gtr.transforms import build_pose_transform

    args = GTRPoseEstimationTrainArgs(variant="s", image_size=image_size)
    return CocoKeypointDataset(
        spec["annotations"],
        spec["images"],
        transform=build_pose_transform(args, train=train),
        train=train,
    )


class TestCocoKeypointDataset:
    def test_split_filters_match_official(self, tmp_path: Path):
        spec = make_keypoint_dataset(tmp_path)
        train_dataset = _build_dataset(spec, train=True)
        val_dataset = _build_dataset(spec, train=False)
        # Train keeps only images with at least one annotated keypoint
        # (image 1); val keeps everything, including the empty image 3.
        assert len(train_dataset) == 1
        assert len(val_dataset) == 3
        assert train_dataset.image_ids == [1]
        assert val_dataset.image_ids == [1, 2, 3]
        assert val_dataset.categories == {1: "person"}

    def test_annotation_filters_drop_crowd_and_empty(self, tmp_path: Path):
        spec = make_keypoint_dataset(tmp_path)
        dataset = _build_dataset(spec, train=False)
        item = dataset[0]
        # Only the fully-labelled regular person survives (crowd + zero-kp
        # persons are filtered like ConvertCocoKeypointsToTarget).
        assert item["keypoints"].shape == (1, 51)
        assert item["classes"].tolist() == [1]

    def test_normalized_target_contract(self, tmp_path: Path):
        spec = make_keypoint_dataset(tmp_path)
        height, width = spec["image_size"]
        dataset = _build_dataset(spec, train=False)
        item = dataset[0]
        keypoints = item["keypoints"][0]
        # xy block first, visibility block last (official PoseNormalize).
        assert keypoints.shape[0] == 3 * K
        # First keypoint sits at (10, 20) in a 100x80 frame.
        assert keypoints[0].item() == pytest.approx(10.0 / width, abs=1e-6)
        assert keypoints[1].item() == pytest.approx(20.0 / height, abs=1e-6)
        # Labelled-visible (v == 2) maps to 1 in the trailing block.
        assert torch.all(keypoints[2 * K :] == 1.0)
        # Box: xyxy (5, 10, 65, 60) -> normalized cxcywh.
        box = item["bboxes"][0]
        assert box[0].item() == pytest.approx(35.0 / width, abs=1e-6)
        assert box[1].item() == pytest.approx(35.0 / height, abs=1e-6)
        assert box[2].item() == pytest.approx(60.0 / width, abs=1e-6)
        assert box[3].item() == pytest.approx(50.0 / height, abs=1e-6)
        assert item["area"][0].item() == pytest.approx(
            box[2].item() * box[3].item(), abs=1e-9
        )
        assert item["original_size"] == (width, height)
        assert item["index"] == 0 and item["image_id"] == 1

    def test_invisible_keypoints_zeroed_and_visibility_mapped(
        self, tmp_path: Path
    ):
        spec = make_keypoint_dataset(tmp_path)
        coco = spec["coco"]
        coco["annotations"][0]["keypoints"][0:3] = [0.0, 0.0, 0]  # invisible
        coco["annotations"][0]["keypoints"][3:6] = [12.0, 22.0, 1]  # labelled
        spec["annotations"].write_text(json.dumps(coco))
        dataset = _build_dataset(spec, train=False)
        keypoints = dataset[0]["keypoints"][0]
        assert keypoints[0].item() == 0.0 and keypoints[1].item() == 0.0
        assert keypoints[2 * K].item() == 0.0  # visibility 0 preserved
        assert keypoints[2].item() == pytest.approx(12.0 / 80.0, abs=1e-6)
        assert keypoints[2 * K + 1].item() == 1.0  # v == 1 labelled

    def test_hflip_mirrors_and_swaps_left_right(self, tmp_path: Path):
        spec = make_keypoint_dataset(tmp_path)
        # Force the flip: prob 1.0, and a distinctive asymmetry on the
        # (1, 2) pair (right/left eye — COCO flip pairs are 0-based
        # keypoint indices): index 1 at x=70, index 2 at x=10.
        coco = spec["coco"]
        coco["annotations"][0]["keypoints"][3:6] = [70.0, 20.0, 2]
        coco["annotations"][0]["keypoints"][6:9] = [10.0, 20.0, 2]
        spec["annotations"].write_text(json.dumps(coco))
        dataset = _build_dataset(spec, train=True)
        stack = dataset.transform
        original = stack.horizontal_flip_prob
        stack.horizontal_flip_prob = 1.0
        try:
            import random

            random.seed(0)
            keypoints = dataset[0]["keypoints"][0]
        finally:
            stack.horizontal_flip_prob = original
        height, width = spec["image_size"]
        # Mirrored keypoints: index 1 -> 80 - 70 - 1 = 9, index 2 -> 69;
        # after the L/R swap slot 1 carries the mirrored index-2 value and
        # slot 2 the mirrored index-1 value (x of keypoint k at flat 2k).
        assert keypoints[2].item() == pytest.approx(69.0 / width, abs=1e-6)
        assert keypoints[4].item() == pytest.approx(9.0 / width, abs=1e-6)
        # The pair was exchanged, so the two slots differ.
        assert keypoints[2].item() != keypoints[4].item()

    def test_degenerate_person_filtered(self, tmp_path: Path):
        spec = make_keypoint_dataset(tmp_path)
        coco = spec["coco"]
        coco["annotations"][0]["bbox"] = [5.0, 10.0, 0.0, 50.0]
        spec["annotations"].write_text(json.dumps(coco))
        dataset = _build_dataset(spec, train=False)
        assert dataset[0]["keypoints"].shape == (0, 51)


class TestCocoKeypointEvaluator:
    def _perfect_predictions(self, spec: dict):
        coco = spec["coco"]
        person = coco["annotations"][0]
        keypoints = torch.tensor(person["keypoints"], dtype=torch.float32).reshape(
            1, 3 * K
        )
        return {
            "keypoints": keypoints,
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }

    def _empty(self):
        return {
            "keypoints": torch.zeros(0, 3 * K),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    def test_perfect_predictions(self, tmp_path: Path):
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        spec = make_keypoint_dataset(tmp_path)
        evaluator = COCOKeypointAPEvaluator(spec["annotations"])
        evaluator.update(
            [self._perfect_predictions(spec), self._empty(), self._empty()],
            image_ids=[0, 1, 2],
        )
        metrics = evaluator.compute()
        assert metrics["kp_map"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["kp_map_50"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["kp_mar"] == pytest.approx(1.0, abs=1e-6)

    def test_ignored_ground_truth_does_not_penalize(self, tmp_path: Path):
        """Crowd / zero-keypoint persons are ignore regions: a perfect
        detection of the single regular person still scores AP 1.0, and a
        wrong-class detection (label 0 like non-person model classes) never
        matches the person category."""
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        spec = make_keypoint_dataset(tmp_path)
        evaluator = COCOKeypointAPEvaluator(spec["annotations"])
        perfect = self._perfect_predictions(spec)
        noise = {
            "keypoints": torch.zeros(1, 3 * K) + 2.0,
            "scores": torch.tensor([0.99]),
            "labels": torch.tensor([0]),
        }
        evaluator.update(
            [perfect, self._empty(), self._empty()], image_ids=[0, 1, 2]
        )
        perfect_metrics = evaluator.compute()
        evaluator.reset()
        evaluator.update(
            [{**perfect, "scores": torch.tensor([0.4])}],
            image_ids=[0],
        )
        _ = evaluator.compute()
        evaluator.reset()
        evaluator.update([noise], image_ids=[0])
        assert evaluator.compute()["kp_map"] == pytest.approx(0.0, abs=1e-6)
        assert perfect_metrics["kp_map"] == pytest.approx(1.0, abs=1e-6)

    def test_degraded_and_missing_predictions(self, tmp_path: Path):
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        spec = make_keypoint_dataset(tmp_path)
        shifted = self._perfect_predictions(spec)
        shifted["keypoints"] = shifted["keypoints"] + 200.0
        evaluator = COCOKeypointAPEvaluator(spec["annotations"])
        evaluator.update([shifted, self._empty(), self._empty()], image_ids=[0, 1, 2])
        degraded = evaluator.compute()["kp_map"]
        assert degraded == pytest.approx(0.0, abs=1e-6)
        evaluator.reset()
        evaluator.update([self._empty(), self._empty(), self._empty()], image_ids=[0, 1, 2])
        assert evaluator.compute()["kp_map"] == 0.0

    def test_matches_pycocotools_stats(self, tmp_path: Path):
        """Our from-array computation equals pycocotools' own summarize()."""
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        from autovision.metrics.pose_estimation import (
            COCOKeypointAPEvaluator,
            predictions_to_coco_keypoints,
        )

        spec = make_keypoint_dataset(tmp_path)
        perfect = self._perfect_predictions(spec)
        noisy = {
            "keypoints": perfect["keypoints"] + 3.0,
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        }
        predictions = [perfect, noisy, self._empty()]
        image_ids = [0, 1, 2]
        evaluator = COCOKeypointAPEvaluator(spec["annotations"])
        evaluator.update(predictions, image_ids=image_ids)
        metrics = evaluator.compute()

        gt_dict = evaluator._ground_truth()
        with __import__("tempfile").TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.json"
            gt_file.write_text(json.dumps(gt_dict))
            coco_gt = COCO(str(gt_file))
        dt = predictions_to_coco_keypoints(predictions, image_ids)
        coco_dt = coco_gt.loadRes(dt)
        reference = COCOeval(coco_gt, coco_dt, "keypoints")
        reference.evaluate()
        reference.accumulate()
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            reference.summarize()  # populates .stats
        stats = [float(v) for v in reference.stats]
        expected = [stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]]
        actual = [
            metrics["kp_map"],
            metrics["kp_map_50"],
            metrics["kp_map_75"],
            metrics["kp_map_m"],
            metrics["kp_map_l"],
            metrics["kp_mar"],
        ]
        for ours, theirs in zip(actual, expected, strict=True):
            if theirs < 0:
                continue
            assert ours == pytest.approx(theirs, abs=1e-9), (ours, theirs)
        # The noisy duplicate is a pure FP at both OKS thresholds.
        assert metrics["kp_map"] < metrics["kp_map_50"] or metrics["kp_map"] == 1.0
        assert np.isfinite(metrics["kp_map"])

    def test_export_merge_state_roundtrip(self, tmp_path: Path):
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        spec = make_keypoint_dataset(tmp_path)
        left = COCOKeypointAPEvaluator(spec["annotations"])
        right = COCOKeypointAPEvaluator(spec["annotations"])
        left.update([self._perfect_predictions(spec)], image_ids=[0])
        right.merge_state(left.export_state())
        assert right.compute()["kp_map"] == pytest.approx(1.0, abs=1e-6)

    def test_image_id_subset_restricts_ground_truth(self, tmp_path: Path):
        """A val-subset run (e.g. ``--limit``) must not count the GT of
        excluded images as missed detections."""
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        spec = make_keypoint_dataset(tmp_path)
        # Second person on image 2 that we will NOT predict; a full-GT run
        # must count it as a miss, a subset run confined to image 1 not.
        coco = spec["coco"]
        coco["annotations"].append(
            {
                "id": 6,
                "image_id": 2,
                "category_id": 1,
                "bbox": [12.0, 10.0, 40.0, 50.0],
                "area": 2000.0,
                "iscrowd": 0,
                "num_keypoints": K,
                "keypoints": _keypoints(offset=2.0),
            }
        )
        spec["annotations"].write_text(json.dumps(coco))

        full = COCOKeypointAPEvaluator(spec["annotations"])
        full.update(
            [self._perfect_predictions(spec), self._empty(), self._empty()],
            image_ids=[0, 1, 2],
        )
        assert full.compute()["kp_map"] < 0.9  # the unpredicted person hurts

        subset = COCOKeypointAPEvaluator(spec["annotations"], image_ids=[0])
        subset.update([self._perfect_predictions(spec)], image_ids=[0])
        assert subset.compute()["kp_map"] == pytest.approx(1.0, abs=1e-6)


# ── CLI closure ─────────────────────────────────────────────────────────
_fla = True
try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # pragma: no cover - import environment dependent
    _fla = False

requires_gla = pytest.mark.skipif(
    not (_fla and torch.cuda.is_available()),
    reason="GTR GLA forward needs flash-linear-attention + CUDA",
)


class TestTrainPoseEstimationCli:
    @requires_gla
    def test_closure_with_keypoint_ap_metric(self, tmp_path: Path):
        from autovision import train_pose_estimation

        spec = make_keypoint_dataset(tmp_path / "data", image_size=(64, 64))
        # The train split needs at least two eligible images for a
        # drop_last DataLoader batch; give image 2 a regular person.
        coco = spec["coco"]
        coco["annotations"].append(
            {
                "id": 5,
                "image_id": 2,
                "category_id": 1,
                "bbox": [12.0, 10.0, 40.0, 50.0],
                "area": 2000.0,
                "iscrowd": 0,
                "num_keypoints": K,
                "keypoints": _keypoints(offset=2.0),
            }
        )
        spec["annotations"].write_text(json.dumps(coco))
        result = train_pose_estimation(
            out=tmp_path / "out",
            data={
                "train": {
                    "annotations": str(spec["annotations"]),
                    "images": str(spec["images"]),
                },
                "val": {
                    "annotations": str(spec["annotations"]),
                    "images": str(spec["images"]),
                },
            },
            model="gtr/s",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "load_checkpoint": False,
                "lr_warmup_steps": 2,
            },
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 4
        assert Path(result.best_exported_model_path).exists()
        assert "val_metric/kp_map" in result.best_metrics
        assert "val_metric/oks" in result.best_metrics
        assert result.classes == {1: "person"}

    def test_rejects_unknown_family(self, tmp_path: Path):
        from autovision import train_pose_estimation

        spec = make_keypoint_dataset(tmp_path / "data", image_size=(64, 64))
        with pytest.raises(ValueError, match="gtr/<s\\|m\\|l\\|x>"):
            train_pose_estimation(
                out=tmp_path / "out",
                data={
                    "train": {
                        "annotations": str(spec["annotations"]),
                        "images": str(spec["images"]),
                    }
                },
                model="eomt/s",
                steps=1,
                batch_size=1,
                num_workers=0,
                model_args={"load_checkpoint": False},
            )
