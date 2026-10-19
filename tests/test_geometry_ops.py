# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""NumPy box/mask/polygon utilities: conversions, IoU, NMS and simplification."""

from __future__ import annotations

import numpy as np
import pytest

from autovision.errors import AutoVisionError
from autovision.geometry import (
    approximate_polygon,
    box_area,
    box_iou,
    box_non_max_merge,
    box_non_max_suppression,
    clip_boxes,
    denormalize_boxes,
    filter_polygons_by_area,
    mask_to_polygons,
    mask_to_xyxy,
    move_boxes,
    pad_boxes,
    polygon_area,
    polygon_to_mask,
    polygon_to_xyxy,
    scale_boxes,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_mask,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)


class TestBoxConversions:
    @pytest.mark.parametrize(
        ("xywh", "expected"),
        [
            pytest.param([[10, 20, 30, 40]], [[10, 20, 40, 60]], id="single"),
            pytest.param(
                [[0, 0, 5, 5], [5, 10, 2, 4]],
                [[0, 0, 5, 5], [5, 10, 7, 14]],
                id="batch",
            ),
            pytest.param([], [], id="empty"),
        ],
    )
    def test_xywh_to_xyxy(self, xywh, expected):
        """(x, y, w, h) converts to (x1, y1, x2, y2)."""
        converted = xywh_to_xyxy(xywh)
        wanted = np.asarray(expected, dtype=float).reshape(-1, 4)
        assert np.array_equal(converted, wanted)

    def test_xyxy_round_trips_through_xywh(self):
        """xyxy -> xywh -> xyxy is the identity."""
        boxes = np.asarray([[10, 20, 40, 60], [3, 4, 15, 33]], dtype=float)
        assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)

    def test_xcycwh_to_xyxy(self):
        """Center boxes expand symmetrically around the center."""
        converted = xcycwh_to_xyxy([[50.0, 50.0, 20.0, 30.0]])
        assert np.allclose(converted, [[40.0, 35.0, 60.0, 65.0]])

    def test_xyxy_to_xcycarh(self):
        """Aspect ratio is width/height and zero height becomes zero."""
        converted = xyxy_to_xcycarh([[10, 20, 40, 60], [0, 0, 5, 0]])
        assert np.allclose(converted[0], [25.0, 40.0, 0.75, 40.0])
        assert converted[1, 2] == 0.0

    def test_clip_boxes(self):
        """Boxes clip to the frame on all four sides."""
        clipped = clip_boxes([[-10, -20, 30, 40], [15, 25, 350, 450]], (320, 240))
        assert np.array_equal(clipped, [[0, 0, 30, 40], [15, 25, 320, 240]])

    def test_denormalize_boxes(self):
        """Normalized coordinates map to pixels, including a custom factor."""
        pixels = denormalize_boxes([[0.1, 0.2, 0.5, 0.6]], (1280, 720))
        assert np.allclose(pixels, [[128.0, 144.0, 640.0, 432.0]])
        scaled = denormalize_boxes([[256.0, 128.0, 768.0, 640.0]], (1280, 720), 1024.0)
        assert np.allclose(scaled, [[320.0, 90.0, 960.0, 450.0]])

    def test_pad_scale_and_move(self):
        """Padding, scaling and translation keep the expected geometry."""
        box = np.asarray([[10.0, 10.0, 20.0, 20.0]])
        assert np.array_equal(pad_boxes(box, 5, 10), [[5, 0, 25, 30]])
        assert np.allclose(scale_boxes(box, 2.0), [[5.0, 5.0, 25.0, 25.0]])
        assert np.array_equal(move_boxes(box, (3, -2)), [[13, 8, 23, 18]])

    def test_invalid_shapes_raise(self):
        """A non-(N, 4) input is rejected early."""
        with pytest.raises(AutoVisionError, match="shape"):
            xywh_to_xyxy([1, 2, 3])


class TestIoUAndSuppression:
    def test_box_iou_matrix(self):
        """IoU is symmetric, diagonal 1.0 and disjoint boxes are 0."""
        boxes = np.asarray([[0, 0, 10, 10], [5, 0, 15, 10], [30, 30, 40, 40]])
        matrix = box_iou(boxes, boxes)
        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix, matrix.T)
        assert matrix[0, 1] == pytest.approx(1.0 / 3.0)
        assert matrix[0, 2] == 0.0

    def test_box_area_and_empty_inputs(self):
        """Area is non-negative and empty inputs keep a valid shape."""
        assert np.array_equal(box_area([[0, 0, 4, 5]]), [20.0])
        assert box_iou([], [[0, 0, 1, 1]]).shape == (0, 1)
        assert box_non_max_suppression([]).shape == (0,)

    def test_nms_suppresses_same_class_only(self):
        """Overlapping boxes of another class survive NMS."""
        predictions = np.asarray(
            [
                [0, 0, 10, 10, 0.9, 0],
                [1, 1, 11, 11, 0.8, 0],
                [0, 0, 10, 10, 0.7, 1],
            ]
        )
        keep = box_non_max_suppression(predictions, 0.5)
        assert keep.tolist() == [True, False, True]

    def test_nms_keeps_higher_score(self):
        """The lower-scoring duplicate is the one dropped."""
        predictions = np.asarray(
            [[0, 0, 10, 10, 0.4], [0, 0, 10, 10, 0.9]]
        )
        assert box_non_max_suppression(predictions).tolist() == [False, True]

    def test_nmm_groups_same_class_duplicates(self):
        """NMM returns index groups with the top-scoring member first."""
        predictions = np.asarray(
            [
                [0, 0, 10, 10, 0.9, 0],
                [1, 1, 11, 11, 0.8, 0],
                [0, 0, 10, 10, 0.7, 1],
            ]
        )
        assert box_non_max_merge(predictions, 0.5) == [[0, 1], [2]]

    def test_invalid_iou_threshold_raises(self):
        """IoU thresholds outside [0, 1] are rejected."""
        with pytest.raises(AutoVisionError, match="iou_threshold"):
            box_non_max_suppression([[0, 0, 1, 1, 0.5]], 1.5)


class TestMaskPolygon:
    def test_mask_to_xyxy_half_open(self):
        """Bounding boxes use exclusive maxima for NumPy slicing."""
        masks = np.zeros((2, 6, 6), dtype=bool)
        masks[0, 1:4, 2:5] = True
        assert mask_to_xyxy(masks).tolist() == [[2, 1, 5, 4], [0, 0, 0, 0]]

    def test_xyxy_to_mask_round_trip(self):
        """Rasterizing a mask's box covers the original pixels."""
        masks = np.zeros((1, 8, 8), dtype=bool)
        masks[0, 2:6, 1:5] = True
        boxes = mask_to_xyxy(masks)
        assert np.array_equal(xyxy_to_mask(boxes, (8, 8)), masks)

    def test_polygon_to_mask_and_envelope(self):
        """A square polygon fills 16 pixels and reports its envelope."""
        polygon = np.asarray([[0, 0], [3, 0], [3, 3], [0, 3]])
        mask = polygon_to_mask(polygon, (5, 5))
        assert int(mask.sum()) == 16
        assert np.array_equal(polygon_to_xyxy(polygon), [0, 0, 3, 3])
        assert polygon_area(polygon) == 9.0

    def test_filter_polygons_by_area(self):
        """Only polygons inside the requested area band are kept."""
        small = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
        large = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]])
        kept = filter_polygons_by_area([small, large], min_area=2.0)
        assert len(kept) == 1
        assert kept[0].shape == large.shape

    def test_approximate_polygon_reduces_points(self):
        """RDP simplification reaches the requested point budget."""
        polygon = np.asarray(
            [[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]]
        )
        simplified = approximate_polygon(polygon, percentage=0.5)
        assert 3 <= simplified.shape[0] <= 4
        assert simplified.shape[1] == 2

    def test_approximate_polygon_keeps_zero_percentage(self):
        """A zero percentage returns the original points untouched."""
        polygon = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert np.array_equal(approximate_polygon(polygon, 0.0), polygon)

    def test_mask_to_polygons(self):
        """OpenCV contour extraction returns the square boundary."""
        pytest.importorskip("cv2")
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        polygons = mask_to_polygons(mask)
        assert len(polygons) == 1
        assert polygons[0].shape[1] == 2
        assert len(polygons[0]) == 4
