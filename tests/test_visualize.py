# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Task-aware rendering: every task family renders without raising."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from autovision.inference import InferenceResult
from autovision.visualize import draw_prediction
from autovision.visualize.annotators import (
    ColorPalette,
    KeypointAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    PanopticAnnotator,
    PolygonAnnotator,
    SemanticMaskAnnotator,
    annotate_result,
)


def _base() -> Image.Image:
    return Image.new("RGB", (64, 64), (0, 0, 0))


def _changed(original: Image.Image, rendered: Image.Image) -> bool:
    return not np.array_equal(np.asarray(original), np.asarray(rendered))


class TestAnnotatorLayers:
    def test_box_annotator_draws_and_keeps_input(self):
        image = _base()
        result = InferenceResult(
            bboxes=torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        rendered = annotate_result(image, result)
        assert _changed(image, rendered)
        assert np.array_equal(np.asarray(image), np.asarray(_base()))

    def test_label_annotator_draws_background(self):
        image = _base()
        result = InferenceResult(
            bboxes=torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
        )
        rendered = LabelAnnotator().annotate(image, result, class_names={1: "cat"})
        assert _changed(_base(), rendered)

    def test_mask_annotator_blends_inside_box(self):
        image = _base()
        masks = torch.zeros(1, 64, 64, dtype=torch.bool)
        masks[0, 10:50, 10:50] = True
        result = InferenceResult(
            bboxes=torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
            masks=masks,
        )
        rendered = MaskAnnotator().annotate(image, result)
        assert _changed(_base(), rendered)
        assert tuple(np.asarray(rendered)[30, 30]) != (0, 0, 0)

    def test_polygon_annotator_draws_oriented_box(self):
        image = _base()
        result = InferenceResult(
            polygons=torch.tensor(
                [[[20, 10], [50, 20], [40, 50], [10, 40]]], dtype=torch.float32
            ),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        assert _changed(_base(), PolygonAnnotator().annotate(image, result))

    def test_keypoint_annotator_draws_coco_skeleton(self):
        image = _base()
        points = torch.zeros(1, 17, 3, dtype=torch.float32)
        for index in range(17):
            points[0, index] = torch.tensor([10 + index * 2, 10 + index * 2, 1.0])
        result = InferenceResult(
            keypoints=points,
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        assert _changed(_base(), KeypointAnnotator().annotate(image, result))

    def test_semantic_mask_annotator_colorizes_label_map(self):
        image = _base()
        labels = torch.zeros(64, 64, dtype=torch.long)
        labels[:, 32:] = 3
        result = InferenceResult(mask=labels)
        rendered = SemanticMaskAnnotator().annotate(image, result)
        assert _changed(_base(), rendered)
        assert tuple(np.asarray(rendered)[10, 50]) != (0, 0, 0)

    def test_panoptic_annotator_colorizes_and_outlines(self):
        image = _base()
        masks = torch.zeros(64, 64, 2, dtype=torch.long)
        masks[:, :32, 0] = 2
        masks[:, :32, 1] = 0
        masks[:, 32:, 0] = 4
        masks[:, 32:, 1] = 1
        result = InferenceResult(
            masks=masks,
            segment_ids=torch.tensor([0, 1]),
            scores=torch.tensor([0.9, 0.8]),
        )
        rendered = PanopticAnnotator().annotate(image, result)
        assert _changed(_base(), rendered)
        assert tuple(np.asarray(rendered)[10, 32]) == (255, 255, 255)


class TestDrawPredictionDispatch:
    def test_detection(self):
        result = {
            "bboxes": torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        assert _changed(_base(), draw_prediction(_base(), result))

    def test_pose(self):
        points = torch.zeros(1, 17, 3)
        points[0, :, 0] = torch.arange(17) * 3 + 4
        points[0, :, 1] = torch.arange(17) * 3 + 4
        points[0, :, 2] = 1.0
        result = {"keypoints": points, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
        assert _changed(_base(), draw_prediction(_base(), result))

    def test_oriented_boxes(self):
        result = {
            "polygons": torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        assert _changed(_base(), draw_prediction(_base(), result))

    def test_classification(self):
        result = {"scores": torch.tensor([0.9]), "labels": torch.tensor([3])}
        assert _changed(_base(), draw_prediction(_base(), result))

    def test_instance_segmentation(self):
        masks = torch.zeros(1, 64, 64, dtype=torch.bool)
        masks[0, 20:40, 20:40] = True
        result = {
            "bboxes": torch.tensor([[20, 20, 40, 40]], dtype=torch.float32),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "masks": masks,
        }
        assert _changed(_base(), draw_prediction(_base(), result))

    def test_semantic_segmentation(self):
        labels = torch.zeros(64, 64, dtype=torch.long)
        labels[20:40, 20:40] = 2
        assert _changed(_base(), draw_prediction(_base(), {"mask": labels}))

    def test_masks_only_instance_result_renders_mask_and_label(self):
        masks = torch.zeros(1, 64, 64, dtype=torch.bool)
        masks[0, 20:40, 20:40] = True
        result = {
            "masks": masks,
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        rendered = draw_prediction(_base(), result, class_names={0: "cat"})
        assert _changed(_base(), rendered)

    def test_panoptic_ignore_pixels_stay_background(self):
        masks = torch.zeros(64, 64, 2, dtype=torch.long)
        masks[:, :, 0] = 5  # ignore class id
        masks[:, :, 1] = -1  # unassigned segment
        result = InferenceResult(
            masks=masks,
            segment_ids=torch.tensor([0]),
            scores=torch.tensor([0.9]),
        )
        rendered = draw_prediction(_base(), result, min_score=0.5)
        assert not _changed(_base(), rendered)

    def test_float_probability_masks_use_half_threshold(self):
        masks = torch.full((1, 64, 64), 0.01)
        result = InferenceResult(
            bboxes=torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
            masks=masks,
        )
        rendered = draw_prediction(_base(), result)
        assert tuple(np.asarray(rendered)[35, 35]) == (0, 0, 0)

    def test_grayscale_inputs_render(self):
        gray = torch.zeros(1, 64, 64)
        assert isinstance(draw_prediction(gray, {"custom": 1}), Image.Image)
        array = np.zeros((64, 64), dtype=np.uint8)
        assert isinstance(draw_prediction(array, {"custom": 1}), Image.Image)

    def test_depth(self):
        depth = torch.linspace(0, 1, 64 * 64).reshape(64, 64)
        assert _changed(_base(), draw_prediction(_base(), {"depth": depth}))

    def test_unknown_record_is_a_noop_copy(self):
        result = draw_prediction(_base(), {"custom": 1})
        assert not _changed(_base(), result)

    def test_empty_keypoints_are_a_noop(self):
        result = InferenceResult(
            keypoints=torch.zeros(0, 17, 3),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )
        rendered = draw_prediction(_base(), result)
        assert not _changed(_base(), rendered)

    def test_min_score_filters_everything(self):
        result = InferenceResult(
            bboxes=torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
            scores=torch.tensor([0.2]),
            labels=torch.tensor([0]),
        )
        rendered = draw_prediction(_base(), result, min_score=0.5)
        assert not _changed(_base(), rendered)

    def test_tensor_and_numpy_inputs_are_not_mutated(self):
        tensor = torch.zeros(3, 64, 64)
        draw_prediction(tensor, {"bboxes": torch.tensor([[5, 5, 40, 40]]), "labels": torch.tensor([0])})
        assert torch.count_nonzero(tensor) == 0
        array = np.zeros((64, 64, 3), dtype=np.uint8)
        draw_prediction(array, {"bboxes": torch.tensor([[5, 5, 40, 40]]), "labels": torch.tensor([0])})
        assert array.max() == 0

    def test_path_input(self, tmp_path):
        path = tmp_path / "img.png"
        _base().save(path)
        rendered = draw_prediction(
            path,
            {"bboxes": torch.tensor([[5, 5, 40, 40]]), "labels": torch.tensor([0])},
        )
        assert isinstance(rendered, Image.Image)

    def test_class_names_are_used(self):
        result = InferenceResult(
            bboxes=torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        assert _changed(_base(), draw_prediction(_base(), result, class_names={0: "cat"}))


class TestPalette:
    def test_index_wraps(self):
        palette = ColorPalette(colors=[(1, 2, 3), (4, 5, 6)])
        assert palette.by_index(0) == (1, 2, 3)
        assert palette.by_index(3) == (4, 5, 6)

    def test_empty_palette_raises(self):
        with pytest.raises(ValueError, match="at least one color"):
            ColorPalette(colors=[]).by_index(0)
