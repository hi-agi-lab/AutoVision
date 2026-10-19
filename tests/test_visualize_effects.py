# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Privacy and attention overlays: trace, heatmap, blur, pixelate and outlines."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autovision.errors import AutoVisionError
from autovision.inference import InferenceResult
from autovision.visualize import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    BoxCornerAnnotator,
    EdgeAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    PixelateAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
)
from autovision.visualize.annotators import KeypointAnnotator
from autovision.visualize.skeletons import SKELETONS


def _black(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (0, 0, 0))


def _pattern(size: int = 64) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size]
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[x >= size // 2] = 255
    array[..., 1] = (x * 3) % 256
    array[..., 2] = (y * 5) % 256
    return Image.fromarray(array, mode="RGB")


def _box_result() -> InferenceResult:
    return InferenceResult(
        bboxes=np.asarray([[10, 10, 50, 50]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
    )


def _mask_result(size: int = 64) -> InferenceResult:
    masks = np.zeros((1, size, size), dtype=bool)
    masks[0, 20:40, 20:40] = True
    return InferenceResult(
        bboxes=np.asarray([[20, 20, 40, 40]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
        masks=masks,
    )


class TestBoxEffects:
    def test_box_corner_only_draws_edges(self):
        """Corner brackets color the corners but not the box center."""
        image = _black()
        rendered = BoxCornerAnnotator(color=(255, 0, 0), thickness=2).annotate(
            image, _box_result()
        )
        array = np.asarray(rendered)
        assert tuple(array[10, 10]) == (255, 0, 0)
        assert tuple(array[30, 30]) == (0, 0, 0)

    def test_round_box_draws_outline(self):
        """Rounded boxes change the outline without filling the interior."""
        image = _black()
        rendered = RoundBoxAnnotator(color=(0, 255, 0), thickness=2).annotate(
            image, _box_result()
        )
        array = np.asarray(rendered)
        assert tuple(array[30, 10]) == (0, 255, 0)
        assert tuple(array[30, 30]) == (0, 0, 0)

    def test_blur_changes_inside_and_preserves_outside(self):
        """Blur only touches the detection box."""
        image = _pattern()
        original = np.asarray(image).copy()
        rendered = BlurAnnotator(radius=4, use_masks=False).annotate(
            image, _box_result()
        )
        array = np.asarray(rendered)
        assert not np.array_equal(array[30, 30], original[30, 30])
        assert np.array_equal(array[2, 2], original[2, 2])

    def test_blur_uses_masks_when_available(self):
        """Mask-mode blur leaves pixels outside the mask untouched."""
        image = _pattern()
        original = np.asarray(image).copy()
        rendered = BlurAnnotator(radius=4).annotate(image, _mask_result())
        array = np.asarray(rendered)
        assert not np.array_equal(array[30, 30], original[30, 30])
        assert np.array_equal(array[30, 10], original[30, 10])

    def test_pixelate_changes_inside_and_preserves_outside(self):
        """Pixelation only touches the detection box."""
        image = _pattern()
        original = np.asarray(image).copy()
        rendered = PixelateAnnotator(pixel_size=8, use_masks=False).annotate(
            image, _box_result()
        )
        array = np.asarray(rendered)
        assert not np.array_equal(array[33, 33], original[33, 33])
        assert np.array_equal(array[2, 2], original[2, 2])

    def test_invalid_effect_parameters_raise(self):
        """Non-positive radii and pixel sizes are rejected."""
        with pytest.raises(AutoVisionError, match="radius"):
            BlurAnnotator(radius=0)
        with pytest.raises(AutoVisionError, match="pixel_size"):
            PixelateAnnotator(pixel_size=0)


class TestMaskEffects:
    def test_halo_rings_the_mask(self):
        """Halo colors pixels just outside the mask, not inside it."""
        rendered = HaloAnnotator(halo_width=4, opacity=1.0).annotate(
            _black(), _mask_result()
        )
        array = np.asarray(rendered)
        assert tuple(array[18, 30]) != (0, 0, 0)
        assert tuple(array[30, 30]) == (0, 0, 0)

    def test_edge_draws_boundary_only(self):
        """Mask edges are colored while the interior stays untouched."""
        rendered = EdgeAnnotator(thickness=1).annotate(_black(), _mask_result())
        array = np.asarray(rendered)
        assert tuple(array[20, 30]) != (0, 0, 0)
        assert tuple(array[30, 30]) == (0, 0, 0)

    def test_background_overlay_dims_outside(self):
        """Background dimming preserves masked pixels and darkens the rest."""
        image = Image.new("RGB", (64, 64), (255, 255, 255))
        rendered = BackgroundOverlayAnnotator(opacity=0.6).annotate(
            image, _mask_result()
        )
        array = np.asarray(rendered)
        assert tuple(array[30, 30]) == (255, 255, 255)
        assert int(array[5, 5, 0]) < 255

    def test_mask_effects_without_masks_are_noops(self):
        """Mask-only effects return the source image when no mask is present."""
        image = _black()
        result = _box_result()
        for annotator in (
            HaloAnnotator(),
            EdgeAnnotator(),
            BackgroundOverlayAnnotator(),
        ):
            assert annotator.annotate(image, result) is image

    def test_mask_effects_ignore_mismatched_shapes(self):
        """A mask from another resolution cannot corrupt the frame."""
        image = _black(size=32)
        masks = np.ones((1, 64, 64), dtype=bool)
        result = InferenceResult(
            bboxes=np.asarray([[0, 0, 31, 31]], dtype=np.float32),
            labels=np.asarray([0]),
            masks=masks,
        )
        for annotator in (
            BlurAnnotator(radius=2, use_masks=True),
            HaloAnnotator(),
            EdgeAnnotator(),
            BackgroundOverlayAnnotator(),
        ):
            rendered = annotator.annotate(image.copy(), result)
            assert np.array_equal(np.asarray(rendered), np.asarray(image))


class TestTemporalEffects:
    def test_trace_draws_history_for_tracker_ids(self):
        """A second frame connects the previous anchor to the current one."""
        annotator = TraceAnnotator(thickness=2)
        first = InferenceResult(
            bboxes=np.asarray([[10, 10, 20, 20]], dtype=np.float32),
            labels=np.asarray([0]),
            extra={"tracker_id": np.asarray([7])},
        )
        second = InferenceResult(
            bboxes=np.asarray([[40, 40, 50, 50]], dtype=np.float32),
            labels=np.asarray([0]),
            extra={"tracker_id": np.asarray([7])},
        )
        assert not np.any(np.asarray(annotator.annotate(_black(), first)))
        rendered = annotator.annotate(_black(), second)
        assert tuple(np.asarray(rendered)[30, 30]) != (0, 0, 0)
        annotator.reset()
        assert not np.any(np.asarray(annotator.annotate(_black(), second)))

    def test_trace_requires_tracker_ids(self):
        """A trace without identity is an explicit error."""
        with pytest.raises(AutoVisionError, match="tracker ids"):
            TraceAnnotator().annotate(_black(), _box_result())

    def test_trace_accepts_explicit_ids(self):
        """The tracker_ids keyword overrides result.extra."""
        annotator = TraceAnnotator()
        first = _box_result()
        second = InferenceResult(
            bboxes=np.asarray([[20, 20, 60, 60]], dtype=np.float32),
            labels=np.asarray([0]),
        )
        annotator.annotate(_black(), first, tracker_ids=[3])
        rendered = annotator.annotate(_black(), second, tracker_ids=[3])
        assert np.asarray(rendered).any()

    def test_heatmap_accumulates_and_resets(self):
        """Repeated frames raise the accumulated heat; reset clears it."""
        annotator = HeatMapAnnotator(radius=10, blur_radius=3)
        result = _box_result()
        first = annotator.annotate(_black(), result)
        assert annotator.heat is not None
        assert float(annotator.heat.max()) == 1.0
        annotator.annotate(_black(), result)
        assert float(annotator.heat.max()) == 2.0
        assert np.asarray(first).any()
        annotator.reset()
        assert annotator.heat is None


class TestSkeletonRegistry:
    def test_coco_registry_is_zero_based(self):
        """The COCO registry matches the 19-edge official skeleton."""
        assert len(SKELETONS["coco"]) == 19
        assert SKELETONS["coco"][0] == (15, 13)

    def test_custom_skeleton_draws_edges(self):
        """An explicit zero-based edge list renders for custom pose models."""
        image = _black()
        result = InferenceResult(
            keypoints=np.asarray([[[10, 10, 1], [40, 40, 1]]], dtype=np.float32),
            labels=np.asarray([0]),
        )
        rendered = KeypointAnnotator(skeleton=[(0, 1)]).annotate(image, result)
        assert np.asarray(rendered).any()

    def test_unknown_skeleton_raises(self):
        """Unknown registry names fail fast instead of silently drawing nothing."""
        with pytest.raises(AutoVisionError, match="Unknown skeleton"):
            KeypointAnnotator(skeleton="not-a-skeleton")
