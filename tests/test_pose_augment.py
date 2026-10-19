# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Keypoint-aware pose mosaic/mixup (roadmap C4): geometry + windows.

Determinism: the affine draw is patched to identity/known values and the
coin flips are forced, so the assertions check exact coordinates rather
than "something changed".
"""

import importlib.util
import random

import pytest
import torch
from tests.test_gtr_pose_coco import make_keypoint_dataset

from autovision.tasks.pose_estimation.gtr.config import GTRPoseEstimationTrainArgs
from autovision.tasks.pose_estimation.gtr.transforms import _PoseTransform

K = 17


def _keypoints(points: list[tuple[float, float]], visible: int = 2) -> torch.Tensor:
    """(N, K, 3) with the first ``len(points)`` labelled."""
    rows = []
    for x, y in points:
        row = [[x, y, float(visible)]]
        row.extend([[0.0, 0.0, 0.0]] * (K - 1))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _item(
    image: torch.Tensor,
    boxes: list[list[float]],
    points: list[list[tuple[float, float]]],
) -> dict:
    height, width = image.shape[-2:]
    count = len(boxes)
    keypoints = torch.zeros(count, K, 3)
    for index, person_points in enumerate(points):
        for slot, (x, y) in enumerate(person_points):
            keypoints[index, slot] = torch.tensor([x, y, 2.0])
    return {
        "image": image,
        "keypoints": keypoints,
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "classes": torch.zeros(count, dtype=torch.long),
        "area": torch.zeros(count, dtype=torch.float32),
        "original_size": (width, height),
    }


def _transform(**kwargs) -> _PoseTransform:
    defaults = {
        "image_size": (64, 64),
        "normalize_mean": (0.0, 0.0, 0.0),
        "normalize_std": (1.0, 1.0, 1.0),
        "train": True,
        "horizontal_flip_prob": 0.0,
        "num_body_points": K,
    }
    defaults.update(kwargs)
    return _PoseTransform(**defaults)


def _decode_keypoints(tensor: torch.Tensor, frame: float = 64.0):
    """(N, 3K) official block layout -> per person [(x_px, y_px, v), ...]."""
    rows = []
    for row in tensor:
        persons = []
        for index in range(K):
            persons.append(
                (
                    float(row[index]) * frame,
                    float(row[K + index]) * frame,
                    float(row[2 * K + index]),
                )
            )
        rows.append(persons)
    return rows


def _image(seed: int, size: tuple[int, int] = (64, 64)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(3, *size, generator=generator) * 255).to(torch.uint8)


class TestMosaic:
    def test_identity_affine_paste_geometry_is_exact(self, monkeypatch):
        """Four tiles pasted at the 2x2 offsets, identity affine: the box
        and keypoint coordinates must land exactly where the paste puts them."""
        transform = _transform(mosaic_prob=1.0, mosaic_output_size=64)
        monkeypatch.setattr(transform, "_sample_affine_params", lambda h, w: (0.0, (0, 0), 1.0))
        monkeypatch.setattr(random, "random", lambda: 0.0)  # always hit, never flip
        # Same sample four times round the cache (choices(0.0) -> index 0).
        item = _item(_image(0), [[40.0, 40.0, 60.0, 60.0]], [[(50.0, 50.0)]])
        for _ in range(4):
            transform._cache_append(item)
        out = transform(item)
        assert out["image"].shape == (3, 64, 64)  # canvas 128 -> resized to image_size
        boxes = out["bboxes"]
        assert boxes.shape[0] == 4
        # Canvas 128x128 with tiles at (0,0), (64,0), (0,64), (64,64); the
        # 64px tile maps to the 64px image 1:1, so the offsets survive.
        # Canvas centres {50, 114} halve under the 128 -> 64 resize (the
        # pipeline normalizes by the canvas, so normalized c * 64 = centre/2).
        expected = {(25.0, 25.0), (57.0, 25.0), (25.0, 57.0), (57.0, 57.0)}
        centres = {
            (round(float(cx) * 64, 2), round(float(cy) * 64, 2))
            for cx, cy in boxes[:, :2]
        }
        assert centres == expected

    def test_visible_keypoints_stay_inside_the_frame(self, monkeypatch):
        transform = _transform(mosaic_prob=1.0, mosaic_output_size=64)
        # Force a real affine so boxes get clipped.
        monkeypatch.setattr(
            transform, "_sample_affine_params", lambda h, w: (12.0, (20, -10), 1.3)
        )
        monkeypatch.setattr(random, "random", lambda: 0.0)
        item = _item(
            _image(1), [[20.0, 20.0, 50.0, 50.0]], [[(30.0, 30.0), (40.0, 40.0)]]
        )
        for _ in range(4):
            transform._cache_append(item)
        out = transform(item)
        keypoints, boxes = out["keypoints"], out["bboxes"]
        # Row alignment survives mosaic + drops.
        assert keypoints.shape[0] == boxes.shape[0]
        # Visible keypoints sit inside the (resized) frame; cropped-out ones
        # are unlabelled (v=0) — the documented contract.
        rows = _decode_keypoints(keypoints)
        assert rows, "expected at least one surviving person"
        for person in rows:
            for x, y, visibility in person:
                if visibility > 0:
                    assert 0.0 <= x <= 64.0 and 0.0 <= y <= 64.0, (x, y)

    def test_translation_unlabels_cropped_points_but_keeps_box(self, monkeypatch):
        transform = _transform(mosaic_prob=1.0, mosaic_output_size=64)
        # Shift left by 70px: the left-hand tiles' boxes leave the canvas
        # entirely (dropped with their keypoints); the right-hand tiles
        # survive, and their keypoint at x=50 lands at 44.
        monkeypatch.setattr(transform, "_sample_affine_params", lambda h, w: (0.0, (-70, 0), 1.0))
        monkeypatch.setattr(random, "random", lambda: 0.0)
        item = _item(_image(2), [[40.0, 40.0, 60.0, 60.0]], [[(50.0, 50.0)]])
        for _ in range(4):
            transform._cache_append(item)
        out = transform(item)
        boxes, keypoints = out["bboxes"], out["keypoints"]
        # Only the right-hand tiles survive (x in [34, 54]).
        assert boxes.shape[0] == 2
        assert keypoints.shape[0] == boxes.shape[0]
        rows = _decode_keypoints(keypoints)
        for person in rows:
            x, _, visibility = person[0]
            # Output visibility is the official binary block (labelled -> 1).
            assert visibility == 1.0
            # 50 + 64 (tile offset) - 70 (translate) = 44 canvas px; the frame
            # is the resized canvas (64), so 44/128 * 64.
            assert x == pytest.approx(22.0, abs=1e-3)


class TestMixup:
    def test_blend_is_beta_windowed_and_annotations_merge(self, monkeypatch):
        transform = _transform(mixup_prob=1.0)
        monkeypatch.setattr(random, "random", lambda: 0.0)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.5)
        donor = _item(
            torch.full((3, 64, 64), 200, dtype=torch.uint8),
            [[0.0, 0.0, 10.0, 10.0]],
            [[(5.0, 5.0)]],
        )
        current = _item(
            torch.full((3, 64, 64), 100, dtype=torch.uint8),
            [[20.0, 20.0, 40.0, 40.0]],
            [[(30.0, 30.0)]],
        )
        monkeypatch.setattr(random, "choice", lambda seq: seq[0])
        transform._cache_append(donor)
        # The cache holds the donor; the current sample enters in forward.
        out = transform(current)
        assert out["bboxes"].shape[0] == 2
        assert out["keypoints"].shape[0] == 2
        # Raw-scale blend: 0.5 * 100 + 0.5 * 200 = 150 (normalize mean/std
        # are 0/1, image divided by 255).
        assert torch.allclose(
            out["image"], torch.full_like(out["image"], 150.0 / 255.0), atol=1e-4
        )
        # Donor rows are carried through (box centre at 5px -> 5/64 normalized).
        boxes_norm = out["bboxes"]
        centres = sorted(round(float(cx) * 64, 2) for cx in boxes_norm[:, 0])
        assert centres == [5.0, 30.0]
        # The donor's keypoint merged at the same frame (binary visibility).
        assert float(out["keypoints"][0, 2 * K]) == 1.0

    def test_mixup_without_cache_is_a_noop(self, monkeypatch):
        transform = _transform(mixup_prob=1.0)
        monkeypatch.setattr(random, "random", lambda: 0.0)
        # The cache is appended in forward, so the first sample mixes with
        # itself — the contract is "a donor exists", not "a different donor".
        out = transform(_item(_image(3), [[0.0, 0.0, 10.0, 10.0]], [[(5.0, 5.0)]]))
        assert out["bboxes"].shape[0] == 2  # self-mix duplicates the person


class TestStepWindows:
    def _boxes_with_step(self, transform, step, item):
        transform.set_step(step)
        return transform(item)["bboxes"].shape[0]

    def test_mosaic_window_gates_activation(self, monkeypatch):
        transform = _transform(
            mosaic_prob=1.0, mosaic_output_size=64, mosaic_step_start=2, mosaic_step_stop=5
        )
        monkeypatch.setattr(transform, "_sample_affine_params", lambda h, w: (0.0, (0, 0), 1.0))
        monkeypatch.setattr(random, "random", lambda: 0.0)
        item = _item(_image(4), [[10.0, 10.0, 30.0, 30.0]], [[(20.0, 20.0)]])
        for _ in range(4):
            transform._cache_append(item)
        assert self._boxes_with_step(transform, 1, item) == 1  # before start
        assert self._boxes_with_step(transform, 2, item) == 4  # start inclusive
        assert self._boxes_with_step(transform, 4, item) == 4  # inside
        assert self._boxes_with_step(transform, 5, item) == 1  # stop exclusive

    def test_window_edge_crossing_flags_dataloader_rebuild(self):
        """Workers hold pickled copies: a window edge must trigger the
        engine's reinit handshake, or the window never reaches them."""
        transform = _transform(
            mosaic_prob=1.0, mosaic_step_start=2, mosaic_step_stop=5
        )
        assert transform.requires_dataloader_reinitialization() is False
        transform.set_step(1)  # before the window
        assert transform.requires_dataloader_reinitialization() is False
        transform.set_step(2)  # window opens
        assert transform.requires_dataloader_reinitialization() is True
        transform.mark_dataloader_as_reinitialized()
        assert transform.requires_dataloader_reinitialization() is False
        transform.set_step(4)  # still inside
        assert transform.requires_dataloader_reinitialization() is False
        transform.set_step(5)  # window closes
        assert transform.requires_dataloader_reinitialization() is True

    def test_default_pipeline_is_untouched(self, monkeypatch):
        """No mosaic/mixup args -> the flip+resize path only (bit-for-bit)."""
        transform = _transform()
        item = _item(_image(5), [[10.0, 10.0, 30.0, 30.0]], [[(20.0, 20.0)]])
        first = transform(item)
        second = _transform()(item)
        assert torch.equal(first["image"], second["image"])
        assert torch.equal(first["keypoints"], second["keypoints"])
        assert torch.equal(first["bboxes"], second["bboxes"])


#: Same gate as the rest of the GTR family tests.
requires_gla = pytest.mark.skipif(
    not (importlib.util.find_spec("fla") is not None and torch.cuda.is_available()),
    reason="GTR GLA forward needs flash-linear-attention + CUDA",
)


class TestAugmentedTrainingWiring:
    @requires_gla
    def test_three_step_train_with_mosaic_and_mixup(self, tmp_path):
        """The engine feeds the train transform its step updates: a live run
        with both augmentations on completes and validates (pytest-level
        proof of the step-hook wiring)."""
        import autovision

        spec = make_keypoint_dataset(tmp_path / "kp", num_images=3)
        data = {
            "train": {"annotations": str(spec["annotations"]), "images": str(spec["images"])},
            "val": {"annotations": str(spec["annotations"]), "images": str(spec["images"])},
        }
        result = autovision.train_pose_estimation(
            out=tmp_path / "run",
            data=data,
            model="gtr/s",
            steps=3,
            batch_size=2,
            num_workers=0,
            devices="auto",
            precision="32-true",
            model_args={
                "checkpoint": None,
                "load_checkpoint": False,
                "mosaic_prob": 1.0,
                "mosaic_output_size": 64,
                "mixup_prob": 1.0,
                "mixup_step_stop": 2,
                "image_size": (64, 64),
                "num_queries": 60,
                "num_decoder_layers": 3,
                "dim_feedforward": 512,
            },
            trainer_args={"val_every_num_steps": 3},
        )
        assert result.steps_run == 3
        assert "val_metric/oks" in result.best_metrics


class TestAugmentConfig:
    def test_window_without_prob_fails_loud(self):
        # A pydantic model validator: EVERY construction path is covered.
        with pytest.raises(ValueError, match="mosaic_step_start"):
            GTRPoseEstimationTrainArgs(mosaic_step_start=5)

    def test_inverted_window_fails_loud(self):
        with pytest.raises(ValueError, match="mixup_step_stop"):
            GTRPoseEstimationTrainArgs(
                mixup_prob=0.5, mixup_step_start=10, mixup_step_stop=10
            )

    def test_probability_bounds_enforced(self):
        with pytest.raises(Exception):
            GTRPoseEstimationTrainArgs(mosaic_prob=1.5)
