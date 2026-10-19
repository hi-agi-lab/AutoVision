# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end detection parity: final boxes, not tensor sums."""

from __future__ import annotations

import json

import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.export.e2e_parity import (
    compare_detection_records,
    verify_end_to_end_parity,
)


def record(boxes, scores, labels):
    return {
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def test_identical_records_pass():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    report = compare_detection_records(ref, ref)
    assert report.passed
    assert report.matched == 1
    assert report.max_box_diff_px == 0.0


def test_subpixel_shift_passes_and_large_shift_fails():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    close = [record([[0.5, 0, 10, 10]], [0.9], [0])]
    assert compare_detection_records(ref, close).passed

    far = [record([[2, 0, 10, 10]], [0.9], [0])]
    report = compare_detection_records(ref, far)
    assert not report.passed
    assert report.max_box_diff_px == pytest.approx(2.0)


def test_missing_and_extra_detections_fail():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    empty = [record([], [], [])]
    missing = compare_detection_records(ref, empty)
    assert not missing.passed and missing.missing == 1

    extra = compare_detection_records(empty, ref)
    assert not extra.passed and extra.extra == 1


def test_label_swap_fails():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    cand = [record([[0, 0, 10, 10]], [0.9], [1])]
    report = compare_detection_records(ref, cand)
    assert not report.passed
    assert report.missing == 1 and report.extra == 1


def test_score_tolerance_is_enforced():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    cand = [record([[0, 0, 10, 10]], [0.8], [0])]
    report = compare_detection_records(ref, cand, score_tolerance=1e-3)
    assert not report.passed
    assert report.max_score_diff == pytest.approx(0.1)


def test_record_count_mismatch_is_rejected():
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    with pytest.raises(AutoVisionError, match="Record count mismatch"):
        compare_detection_records(ref, [])


def test_record_without_boxes_is_rejected():
    with pytest.raises(AutoVisionError, match="needs detection records"):
        compare_detection_records([{"scores": [0.9]}], [{"scores": [0.9]}])


class FakeEngine:
    def __init__(self, records):
        self._records = records
        self.calls = []

    def run_batch(self, images, *, threshold=0.5):
        self.calls.append((list(images), threshold))
        return self._records


def test_verify_end_to_end_parity_and_report(tmp_path):
    ref = [record([[0, 0, 10, 10]], [0.9], [0])]
    same = FakeEngine(ref)
    report = verify_end_to_end_parity(
        FakeEngine(ref), same, ["a.jpg"], threshold=0.3,
        report_path=tmp_path / "parity.json",
    )
    assert report.passed
    assert same.calls == [(["a.jpg"], 0.3)]
    payload = json.loads((tmp_path / "parity.json").read_text())
    assert payload["passed"] is True

    shifted = FakeEngine([record([[5, 0, 10, 10]], [0.9], [0])])
    assert not verify_end_to_end_parity(FakeEngine(ref), shifted, ["a.jpg"]).passed


def test_verify_rejects_a_non_engine():
    with pytest.raises(AutoVisionError, match="run_batch"):
        verify_end_to_end_parity(object(), FakeEngine([]), ["a.jpg"])
