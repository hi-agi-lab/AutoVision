# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""SAHI merge strategies: joint tile filtering, NMM and IOS."""

from __future__ import annotations

import pytest
import torch

from autovision.tasks.shared.tiling_utils import combine_object_detection_tiles
from autovision.tasks.task_model import TaskNetwork


def _det(boxes, scores, labels):
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.as_tensor(scores, dtype=torch.float32).reshape(-1),
        "labels": torch.as_tensor(labels, dtype=torch.int64).reshape(-1),
    }


class _Stub:
    image_size = (64, 64)
    training = False

    def __init__(self):
        self.calls: list[int] = []

    def preprocess_image(self, image):
        tensor = image if torch.is_tensor(image) else torch.as_tensor(image)
        if tensor.shape[-2:] != (64, 64):
            tensor = torch.nn.functional.interpolate(
                tensor[None].float(), size=(64, 64), mode="bilinear",
                align_corners=False,
            )[0]
        return tensor.float(), {"orig_w": image.shape[-1], "orig_h": image.shape[-2]}

    def preprocess_batch(self, batch):
        return batch

    def forward_backend(self, batch):
        return batch

    def postprocess(self, raw, metadata, threshold=0.0):
        self.calls.append(len(metadata))
        return [
            _det([[1.0, 1.0, 5.0, 5.0]], [0.9], [0]) for _ in metadata
        ]

    def _module_device(self):
        return torch.device("cpu")

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    predict_sahi = TaskNetwork.predict_sahi
    _load_image_unchanged = TaskNetwork._load_image_unchanged


class TestCombineStrategies:
    def test_nmm_unions_fragments(self):
        global_out = _det([[0.0, 0.0, 10.0, 10.0]], [0.9], [0])
        tile_out = _det([[0.0, 0.0, 6.0, 10.0]], [0.7], [0])
        merged = combine_object_detection_tiles(
            global_out, [tile_out], torch.tensor([[4, 0]]), (20, 20),
            nms_iou_threshold=0.3, merge="nmm",
        )
        assert merged["bboxes"].shape[0] == 1
        assert merged["bboxes"].tolist() == [[0.0, 0.0, 10.0, 10.0]]
        assert 0.7 < float(merged["scores"][0]) < 0.9

    def test_ios_suppresses_nested_box(self):
        # Small box entirely inside the large one: IoU 0.0625 < 0.3 but
        # IOS 1.0 > 0.3, so IOS keeps only the higher-scoring box.
        global_out = _det([[0.0, 0.0, 40.0, 40.0]], [0.9], [0])
        tile_out = _det([[0.0, 0.0, 10.0, 10.0]], [0.8], [0])
        iou = combine_object_detection_tiles(
            global_out, [tile_out], torch.tensor([[0, 0]]), (40, 40),
            nms_iou_threshold=0.3, overlap_metric="iou",
        )
        ios = combine_object_detection_tiles(
            global_out, [tile_out], torch.tensor([[0, 0]]), (40, 40),
            nms_iou_threshold=0.3, overlap_metric="ios",
        )
        assert iou["bboxes"].shape[0] == 2
        assert ios["bboxes"].shape[0] == 1

    def test_cross_tile_duplicates_are_removed(self):
        tile_a = _det([[0.0, 0.0, 10.0, 10.0]], [0.9], [0])
        tile_b = _det([[0.0, 0.0, 10.0, 10.0]], [0.8], [0])
        merged = combine_object_detection_tiles(
            _det([], [], []),
            [tile_a, tile_b],
            torch.tensor([[0, 0], [0, 0]]),
            (20, 20),
        )
        assert merged["bboxes"].shape[0] == 1

    def test_invalid_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="merge must be"):
            combine_object_detection_tiles(
                _det([], [], []), [], torch.zeros(0, 2), (10, 10), merge="soft"
            )
        with pytest.raises(ValueError, match="overlap_metric must be"):
            combine_object_detection_tiles(
                _det([], [], []), [], torch.zeros(0, 2), (10, 10),
                overlap_metric="giou",
            )


class TestChunkedSahi:
    def test_batch_size_splits_views(self):
        stub = _Stub()
        stub.predict_sahi(torch.rand(3, 40, 100), tile_size=64, batch_size=2)
        # global + 2 tiles = 3 views -> chunks [2, 1].
        assert stub.calls == [2, 1]

    def test_none_keeps_single_batch(self):
        stub = _Stub()
        stub.predict_sahi(torch.rand(3, 40, 100), tile_size=64)
        assert stub.calls == [3]

    def test_invalid_batch_size(self):
        stub = _Stub()
        with pytest.raises(ValueError, match="batch_size"):
            stub.predict_sahi(torch.rand(3, 32, 32), batch_size=0)
