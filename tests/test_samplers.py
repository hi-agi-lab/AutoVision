# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Long-tail samplers: weights, determinism, training hook."""

from __future__ import annotations

import pytest

from autovision import ensure_coco8, train_object_detection
from autovision.data.samplers import (
    build_long_tail_sampler,
    class_counts,
    image_weights,
)
from autovision.errors import AutoVisionError


def _rows(rare: int = 1, common: int = 9):
    rows = [{"class_labels": "[0]"} for _ in range(rare)]
    rows.extend({"class_labels": "[1]"} for _ in range(common))
    rows.append({"class_labels": "[]"})
    return rows


def test_class_counts_and_class_balanced_weights():
    rows = _rows()
    assert class_counts(rows) == {0: 1, 1: 9}
    weights = image_weights(rows, strategy="class_balanced", max_repeat=5.0)
    assert len(weights) == len(rows)
    assert weights[0] == pytest.approx(5.0)  # rare class gets the cap
    assert weights[1] == pytest.approx(1.0)  # common class baseline
    assert weights[-1] == pytest.approx(1.0)  # unlabeled image baseline


def test_repeat_factor_respects_threshold_and_cap():
    rows = _rows(rare=1, common=99)
    weights = image_weights(
        rows, strategy="repeat_factor", threshold=0.5, max_repeat=4.0
    )
    assert weights[0] <= 4.0
    assert weights[0] > weights[1]
    assert weights[-1] == pytest.approx(1.0)


def test_sampler_is_deterministic():
    rows = _rows()
    first = list(build_long_tail_sampler(rows, seed=3))
    second = list(build_long_tail_sampler(rows, seed=3))
    assert first == second
    assert set(first) <= set(range(len(rows)))


def test_invalid_parameters():
    with pytest.raises(AutoVisionError, match="Unknown long-tail strategy"):
        image_weights(_rows(), strategy="nope")
    with pytest.raises(AutoVisionError, match="power"):
        image_weights(_rows(), power=-1)
    with pytest.raises(AutoVisionError, match="threshold"):
        image_weights(_rows(), strategy="repeat_factor", threshold=0)
    with pytest.raises(AutoVisionError, match="at least one image"):
        image_weights([])


def test_training_entry_accepts_long_tail_sampler(tmp_path):
    result = train_object_detection(
        out=tmp_path / "run",
        data=ensure_coco8(),
        model="rtdetr/resnet18",
        steps=1,
        batch_size=1,
        num_workers=0,
        devices=1,
        precision="32-true",
        long_tail_sampler="class_balanced",
    )
    assert result.exported_model_path.exists()


def test_training_entry_rejects_both_samplers(tmp_path):
    with pytest.raises(ValueError, match="both replace the train sampler"):
        train_object_detection(
            out=tmp_path / "run",
            data=ensure_coco8(),
            model="rtdetr/resnet18",
            steps=1,
            batch_size=1,
            num_workers=0,
            devices=1,
            precision="32-true",
            hard_examples=tmp_path / "missing.json",
            long_tail_sampler="class_balanced",
        )


def test_unknown_sampler_config_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown long_tail_sampler key"):
        train_object_detection(
            out=tmp_path / "run",
            data=ensure_coco8(),
            model="rtdetr/resnet18",
            steps=1,
            batch_size=1,
            num_workers=0,
            devices=1,
            precision="32-true",
            long_tail_sampler={"strategy": "class_balanced", "bogus": 1},
        )
