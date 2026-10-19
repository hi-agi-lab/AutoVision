# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""InferenceResult: record normalization, selection, and mask serialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autovision.inference import InferenceResult


def _detection(n: int = 2) -> InferenceResult:
    return InferenceResult.from_record(
        {
            "bboxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)[:n],
            "scores": torch.tensor([0.9, 0.8], dtype=torch.float32)[:n],
            "labels": torch.tensor([0, 1], dtype=torch.long)[:n],
        },
        metadata={"source": "clip.mp4", "frame": 3},
    )


class TestConstruction:
    def test_detection_record_round_trips_and_infers_task(self):
        result = _detection()
        assert len(result) == 2
        assert result.inferred_task == "object_detection"
        record = result.to_record()
        assert set(record) == {"bboxes", "scores", "labels", "metadata"}
        assert record["metadata"]["source"] == "clip.mp4"

    def test_boxes_alias_is_normalized(self):
        result = InferenceResult.from_record(
            {"boxes": torch.zeros(1, 4), "scores": torch.ones(1)}
        )
        assert result.bboxes is not None
        assert result.to_record()["bboxes"].shape == (1, 4)

    def test_unknown_fields_are_preserved_in_extra(self):
        result = InferenceResult.from_record(
            {
                "bboxes": torch.zeros(2, 4),
                "scores": torch.ones(2),
                "labels": torch.zeros(2, dtype=torch.long),
                "custom": {"nested": 1},
            }
        )
        assert result.extra["custom"] == {"nested": 1}
        assert result.to_record()["custom"] == {"nested": 1}

    def test_extra_cannot_shadow_typed_fields(self):
        with pytest.raises(ValueError, match="must not duplicate"):
            InferenceResult(bboxes=torch.zeros(1, 4), extra={"bboxes": []})

    def test_pose_and_classification_cardinality(self):
        pose = InferenceResult(
            keypoints=torch.zeros(3, 17, 3),
            scores=torch.ones(3),
            labels=torch.zeros(3, dtype=torch.long),
        )
        assert len(pose) == 3
        assert pose.inferred_task == "pose_estimation"
        classification = InferenceResult(
            scores=torch.tensor([0.7]), labels=torch.tensor([4])
        )
        assert len(classification) == 1
        assert classification.inferred_task == "image_classification"

    def test_obb_polygons_accept_flat_form(self):
        result = InferenceResult(
            polygons=torch.tensor([[0, 0, 10, 0, 10, 10, 0, 10]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        assert result.polygons is not None
        assert tuple(result.polygons.shape) == (1, 4, 2)
        assert result.inferred_task == "oriented_object_detection"

    def test_panoptic_cardinality_uses_segment_ids(self):
        result = InferenceResult(
            masks=torch.zeros(8, 8, 2, dtype=torch.long),
            segment_ids=torch.tensor([0, 1, 2]),
            scores=torch.tensor([0.9, 0.8, 0.7]),
        )
        assert len(result) == 3
        assert result.inferred_task == "panoptic_segmentation"
        with pytest.raises(ValueError, match="not defined"):
            result.select(0)

    def test_depth_and_semantic_masks_are_image_level(self):
        depth = InferenceResult(depth=torch.zeros(4, 5))
        assert len(depth) == 1
        assert depth.inferred_task == "depth_estimation"
        semantic = InferenceResult(mask=torch.zeros(4, 5, dtype=torch.long))
        assert len(semantic) == 1
        assert semantic.inferred_task == "semantic_segmentation"

    def test_empty_detection(self):
        empty = InferenceResult.from_record({"bboxes": [], "scores": [], "labels": []})
        assert empty.is_empty
        assert empty.inferred_task == "object_detection"


class TestSelection:
    def test_select_returns_fresh_metadata_and_rows(self):
        result = _detection()
        selected = result.select([1])
        assert selected.bboxes.tolist() == [[20.0, 20.0, 30.0, 30.0]]
        selected.metadata["frame"] = 99
        assert result.metadata["frame"] == 3

    def test_filter_scores_and_boolean_lists(self):
        result = _detection()
        filtered = result.filter_scores(0.85)
        assert filtered.labels.tolist() == [0]
        filtered = result.filter([True, False])
        assert filtered.scores.tolist() == pytest.approx([0.9])

    def test_select_does_not_slice_image_level_masks(self):
        result = InferenceResult(
            bboxes=torch.zeros(2, 4),
            scores=torch.ones(2),
            labels=torch.zeros(2, dtype=torch.long),
            mask=torch.zeros(4, 4, dtype=torch.long),
        )
        selected = result.select(0)
        assert selected.mask is result.mask
        assert selected.bboxes.shape == (1, 4)


class TestSerialization:
    def test_instance_masks_default_to_rle(self):
        from pycocotools import mask as mask_utils

        masks = torch.zeros(2, 6, 8, dtype=torch.bool)
        masks[0, 1:3, 2:5] = True
        masks[1, 4:6, 0:2] = True
        result = InferenceResult(
            bboxes=torch.tensor([[0, 0, 8, 6], [0, 0, 8, 6]], dtype=torch.float32),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 1]),
            masks=masks,
        )
        payload = result.to_json()
        assert len(payload["masks"]) == 2
        decoded = mask_utils.decode(
            {
                "size": payload["masks"][0]["size"],
                "counts": payload["masks"][0]["counts"].encode("ascii"),
            }
        )
        assert np.array_equal(decoded.astype(bool), masks[0].numpy())

    def test_masks_none_omits_the_payload(self):
        masks = torch.ones(1, 2, 2, dtype=torch.bool)
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            masks=masks,
        )
        assert "masks" not in result.to_json(masks="none")

    def test_dense_mode_is_capped_for_large_arrays(self):
        result = InferenceResult(depth=torch.arange(100, dtype=torch.float32).reshape(10, 10))
        payload = result.to_json(max_elements=10)
        assert payload["depth"]["summarized"] is True
        assert payload["depth"]["shape"] == [10, 10]

    def test_semantic_bool_mask_encodes_as_rle(self):
        result = InferenceResult(mask=torch.zeros(4, 5, dtype=torch.bool))
        payload = result.to_json()
        assert set(payload["mask"]) == {"size", "counts"}

    def test_semantic_label_map_uses_lossless_rle(self):
        labels = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
        result = InferenceResult(mask=labels)
        payload = result.to_json()["mask"]
        assert payload["size"] == [2, 3]
        assert payload["counts"] == [2, 2, 2]
        assert payload["values"] == [0, 1, 2]

    def test_panoptic_map_encodes_both_channels(self):
        masks = torch.zeros(4, 5, 2, dtype=torch.long)
        masks[:, :, 0] = 1
        masks[:, :, 1] = 4
        result = InferenceResult(
            masks=masks,
            segment_ids=torch.tensor([0, 1, 2, 3]),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        )
        payload = result.to_json()["masks"]
        assert payload["size"] == [4, 5]
        assert payload["class_ids"]["values"] == [1]
        assert payload["segment_ids"]["values"] == [4]

    def test_metadata_and_task_are_serialized_when_present(self):
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            metadata={"source": "a.mp4"},
            task="object_detection",
        )
        payload = result.to_json()
        assert payload["metadata"] == {"source": "a.mp4"}
        assert payload["task"] == "object_detection"

    def test_to_numpy_converts_typed_fields(self):
        result = _detection().to_numpy()
        assert isinstance(result.bboxes, np.ndarray)
        assert result.to_json()["bboxes"] == [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]

    def test_invalid_masks_mode_raises(self):
        with pytest.raises(ValueError, match="masks must be"):
            _detection().to_json(masks="png")

    def test_integer_and_float_masks_are_rle_encoded(self):
        from pycocotools import mask as mask_utils

        masks = torch.zeros(2, 4, 4, dtype=torch.uint8)
        masks[0, 1:3, 1:3] = 255
        masks[1, 1:3, 1:3] = 1
        floats = torch.zeros(1, 4, 4)
        floats[0, 0:2, 0:2] = 0.9
        for value in (masks, floats):
            result = InferenceResult(
                bboxes=torch.zeros(value.shape[0], 4),
                scores=torch.ones(value.shape[0]),
                labels=torch.zeros(value.shape[0], dtype=torch.long),
                masks=value,
            )
            payload = result.to_json()
            decoded = mask_utils.decode(
                {
                    "size": payload["masks"][0]["size"],
                    "counts": payload["masks"][0]["counts"].encode("ascii"),
                }
            )
            assert decoded.sum() == 4

    def test_per_instance_fields_are_never_summarized(self):
        result = InferenceResult(
            bboxes=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=torch.float32),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 1]),
            depth=torch.arange(100, dtype=torch.float32).reshape(10, 10),
        )
        payload = result.to_json(max_elements=4)
        assert payload["bboxes"] == [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
        assert payload["labels"] == [0, 1]
        assert payload["depth"]["summarized"] is True

    def test_non_finite_summary_is_json_safe(self):
        depth = torch.tensor([[0.0, float("nan")], [float("inf"), 2.0]])
        payload = InferenceResult(depth=depth).to_json(max_elements=1)
        assert payload["depth"]["summarized"] is True
        assert payload["depth"]["min"] == 0.0
        assert payload["depth"]["max"] == 2.0

    def test_filter_scores_moves_index_to_mask_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required to exercise mixed-device selection")
        result = InferenceResult(
            bboxes=torch.zeros(2, 4),
            scores=torch.tensor([0.9, 0.1], device="cuda"),
            labels=torch.zeros(2, dtype=torch.long),
            masks=torch.zeros(2, 4, 4, dtype=torch.bool),
        )
        filtered = result.filter_scores(0.5)
        assert filtered.masks is not None
        assert filtered.masks.shape[0] == 1

    def test_panoptic_filter_scores_is_a_noop(self):
        result = InferenceResult(
            masks=torch.zeros(4, 4, 2, dtype=torch.long),
            segment_ids=torch.tensor([0, 1]),
            scores=torch.tensor([0.9, 0.8]),
        )
        assert result.filter_scores(0.95) is result

    def test_masks_only_result_is_instance_segmentation(self):
        result = InferenceResult(
            labels=torch.tensor([0]),
            scores=torch.tensor([0.9]),
            masks=torch.ones(1, 4, 4, dtype=torch.bool),
        )
        assert result.inferred_task == "instance_segmentation"

    def test_numpy_bool_list_filters_rows(self):
        result = _detection()
        filtered = result.filter([np.True_, np.False_])
        assert filtered.labels.tolist() == [0]


class TestValidation:
    def test_bbox_shape_is_validated(self):
        with pytest.raises(ValueError, match="bboxes must have shape"):
            InferenceResult(bboxes=torch.zeros(2, 3))

    def test_row_counts_must_match(self):
        with pytest.raises(ValueError, match="labels has 1 rows"):
            InferenceResult(
                bboxes=torch.zeros(2, 4),
                labels=torch.zeros(1, dtype=torch.long),
            )

    def test_instance_masks_must_match_label_rows(self):
        with pytest.raises(ValueError, match="masks has 2 rows"):
            InferenceResult(
                labels=torch.zeros(1, dtype=torch.long),
                masks=torch.zeros(2, 4, 4, dtype=torch.bool),
            )

    def test_panoptic_segment_scores_must_match(self):
        with pytest.raises(ValueError, match="segment_ids has 2 entries"):
            InferenceResult(
                masks=torch.zeros(4, 4, 2, dtype=torch.long),
                segment_ids=torch.tensor([0, 1]),
                scores=torch.ones(3),
            )

    def test_classification_scores_must_match_labels(self):
        with pytest.raises(ValueError, match="labels has 2 rows"):
            InferenceResult(
                labels=torch.tensor([0, 1]),
                scores=torch.ones(1),
            )
