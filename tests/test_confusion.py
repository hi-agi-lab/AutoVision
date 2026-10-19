# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Confusion matrix built from error-analysis matches."""

from __future__ import annotations

import torch

from autovision.metrics.confusion import ConfusionMatrix
from autovision.metrics.error_analysis import match_detections


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


def _matches():
    predictions = [
        _pred([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 0]),
        _pred([[100, 100, 110, 110]], [0.7], [1]),
    ]
    targets = [
        _target([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], [0, 1, 1]),
        _target([], []),
    ]
    return match_detections(predictions, targets, class_names={0: "cat", 1: "dog"})


class TestConfusionMatrix:
    def test_places_tp_classification_fp_and_fn(self):
        matrix = ConfusionMatrix.from_matches(
            _matches(), class_names={0: "cat", 1: "dog"}, class_ids=[0, 1]
        )
        # rows: cat, dog, FP; columns: cat, dog, FN.
        assert matrix.class_names == ["cat", "dog"]
        assert matrix.matrix.tolist() == [
            [1, 0, 0],
            [1, 0, 2],
            [0, 1, 0],
        ]

    def test_markdown_has_fn_column_and_fp_row(self):
        matrix = ConfusionMatrix.from_matches(_matches(), class_names={0: "cat", 1: "dog"})
        text = matrix.to_markdown()
        assert "True \\ Predicted" in text
        assert "| FP |" in text
        assert "| FN |" in text
        assert "cat" in text and "dog" in text

    def test_empty_matches_render_placeholder(self):
        matrix = ConfusionMatrix.from_matches([], class_names={})
        assert matrix.matrix.shape == (1, 1)
        assert matrix.to_markdown() == "_（无类别）_"

    def test_to_dict_round_trips_shape(self):
        matrix = ConfusionMatrix.from_matches(_matches(), class_names={0: "cat", 1: "dog"})
        payload = matrix.to_dict()
        assert payload["class_ids"] == [0, 1]
        assert payload["fp_row"] == 2
        assert payload["fn_col"] == 2
        assert len(payload["matrix"]) == 3
