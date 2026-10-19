# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pseudo-label generation + COCO merging (roadmap C1)."""

import json
from pathlib import Path

import pytest
import torch
from tests.conftest import create_image

from autovision.data.convert import merge_coco
from autovision.tasks.detection.shared.pseudo_label import (
    PseudoLabelResult,
    generate_pseudo_labels,
    list_images,
)

CATEGORIES = {2: "cat", 5: "dog"}


class _StubModel:
    """Duck-typed detection model recording its call shapes."""

    def __init__(self, per_image: dict[str, dict]):
        self.per_image = per_image
        self.batch_calls: list[tuple[list[str], float]] = []
        self.sahi_calls: list[tuple[str, dict]] = []

    def predict_batch(self, images, threshold=0.0):
        self.batch_calls.append(([Path(p).name for p in images], threshold))
        return [self.per_image[Path(p).name] for p in images]

    def predict_sahi(self, path, **kwargs):
        self.sahi_calls.append((Path(path).name, kwargs))
        return self.per_image[Path(path).name]


def _prediction(boxes, scores, labels):
    return {
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@pytest.fixture()
def pool(tmp_path):
    paths = [
        create_image(tmp_path / "pool" / f"img_{index}.jpg", size=(64, 64), seed=index)
        for index in range(3)
    ]
    return tmp_path / "pool", paths


def _stub(pool_paths):
    return _StubModel(
        {
            "img_0.jpg": _prediction([[10, 20, 30, 50]], [0.9], [0]),
            "img_1.jpg": _prediction([[0, 0, 32, 32], [4, 4, 96, 96]], [0.8, 0.2], [1, 0]),
            "img_2.jpg": _prediction([], [], []),
        }
    )


class TestGeneratePseudoLabels:
    def test_coco_shape_and_category_mapping(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        result = generate_pseudo_labels(
            model, root, tmp_path / "pseudo.json", threshold=0.5, categories=CATEGORIES
        )
        assert isinstance(result, PseudoLabelResult)
        payload = json.loads(result.out_json.read_text())
        assert [c for c in payload["categories"]] == [
            {"id": 2, "name": "cat"},
            {"id": 5, "name": "dog"},
        ]
        assert result.images == 3
        assert result.annotations == 2  # 0.9 cat + 0.8 dog (0.2 dropped)
        assert result.per_class == {"cat": 1, "dog": 1}
        assert result.score_min == pytest.approx(0.8)
        assert result.score_max == pytest.approx(0.9)
        # Internal id 0 -> first sorted category key (2); id 1 -> 5.
        assert [a["category_id"] for a in payload["annotations"]] == [2, 5]
        assert [a["score"] for a in payload["annotations"]] == pytest.approx([0.9, 0.8])
        # bbox xywh + area from the absolute xyxy prediction.
        assert payload["annotations"][0]["bbox"] == [10.0, 20.0, 20.0, 30.0]
        assert payload["annotations"][0]["area"] == pytest.approx(600.0)
        assert payload["info"]["category_mapping"].startswith("sorted(categories)")
        # file_name relative to the images root; size read from the file.
        assert payload["images"][0]["file_name"] == "img_0.jpg"
        assert (payload["images"][0]["width"], payload["images"][0]["height"]) == (64, 64)

    def test_sequence_and_positional_categories(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        result = generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5,
            categories=["cat", "dog"],
        )
        payload = json.loads(result.out_json.read_text())
        assert [c["id"] for c in payload["categories"]] == [1, 2]
        assert [a["category_id"] for a in payload["annotations"]] == [1, 2]

    def test_positional_fallback_without_categories(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        result = generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5
        )
        payload = json.loads(result.out_json.read_text())
        assert payload["info"]["category_mapping"] == "label + 1"
        assert [a["category_id"] for a in payload["annotations"]] == [1, 2]
        assert [c["name"] for c in payload["categories"]] == ["class_0", "class_1"]

    def test_class_thresholds_override(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        result = generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5,
            class_thresholds={"dog": 0.9}, categories=CATEGORIES,
        )
        # The 0.8 dog label falls below its 0.9 class threshold; cat stays.
        assert result.per_class == {"cat": 1}

    def test_int_threshold_keys_are_category_ids(self, pool, tmp_path):
        """Docstring contract: int keys are COCO category ids (what the
        emitted json carries), not internal label indices."""
        root, paths = pool
        model = _stub(paths)
        # Document "cat" (category id 5 -> internal 1) at 0.9: the 0.8 dog
        # label drops, the 0.9 cat label stays.
        result = generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5,
            class_thresholds={5: 0.9}, categories=CATEGORIES,
        )
        assert result.per_class == {"cat": 1}

    def test_unknown_class_threshold_key_fails_loud(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        with pytest.raises(ValueError, match="matches no category"):
            generate_pseudo_labels(
                model, root, tmp_path / "p.json", categories=CATEGORIES,
                class_thresholds={"bird": 0.5},
            )

    def test_batching_and_max_images(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        result = generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5,
            categories=CATEGORIES, batch_size=2, max_images=3,
        )
        assert model.batch_calls == [(["img_0.jpg", "img_1.jpg"], 0.0), (["img_2.jpg"], 0.0)]
        assert result.images == 3

    def test_sahi_uses_lowest_active_threshold(self, pool, tmp_path):
        root, paths = pool
        model = _stub(paths)
        generate_pseudo_labels(
            model, root, tmp_path / "p.json", threshold=0.5,
            class_thresholds={"cat": 0.2}, categories=CATEGORIES,
            sahi=True, tile_size=32, overlap=0.25,
        )
        assert len(model.sahi_calls) == 3
        for _, kwargs in model.sahi_calls:
            # 0.2 (cat override) is the floor the internal SAHI filter may use.
            assert kwargs["threshold"] == pytest.approx(0.2)
            assert kwargs["tile_size"] == 32
            assert kwargs["overlap"] == pytest.approx(0.25)

    def test_nested_pool_is_listed_recursively(self, tmp_path):
        nested = tmp_path / "pool" / "batch_a"
        nested.mkdir(parents=True)
        create_image(nested / "img_0.jpg", size=(32, 32), seed=0)
        create_image(tmp_path / "pool" / "img_top.jpg", size=(32, 32), seed=1)
        paths = list_images(tmp_path / "pool")
        assert {path.name for path in paths} == {"img_0.jpg", "img_top.jpg"}

    def test_empty_pool_fails_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_images(tmp_path / "missing")
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            list_images(empty)


class TestMergeCoco:
    def test_merge_renumbers_ids_and_unions_categories(self, tmp_path):
        first = tmp_path / "a.json"
        first.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
                    ],
                    "categories": [{"id": 1, "name": "cat"}],
                }
            )
        )
        second = tmp_path / "b.json"
        second.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 7, "file_name": "b1.jpg", "width": 64, "height": 64},
                        {"id": 8, "file_name": "b2.jpg", "width": 64, "height": 64},
                    ],
                    "annotations": [
                        {"id": 5, "image_id": 7, "category_id": 4,
                         "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0},
                        {"id": 6, "image_id": 8, "category_id": 9,
                         "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
                    ],
                    # "cat" repeats with a conflicting id (4); "bird" is new (9).
                    "categories": [{"id": 4, "name": "cat"}, {"id": 9, "name": "bird"}],
                }
            )
        )
        merged_path = merge_coco([first, second], tmp_path / "merged.json")
        merged = json.loads(merged_path.read_text())
        assert [image["id"] for image in merged["images"]] == [1, 2, 3]
        assert [image["file_name"] for image in merged["images"]] == [
            "a.jpg", "b1.jpg", "b2.jpg"
        ]
        assert [annotation["id"] for annotation in merged["annotations"]] == [1, 2, 3]
        assert [annotation["image_id"] for annotation in merged["annotations"]] == [1, 2, 3]
        # "cat" keeps the first file's id (1); the conflicting 4 is dropped;
        # "bird" keeps 9 (free).
        assert merged["categories"] == [
            {"id": 1, "name": "cat"},
            {"id": 9, "name": "bird"},
        ]
        assert [a["category_id"] for a in merged["annotations"]] == [1, 1, 9]

    def test_merge_needs_inputs(self, tmp_path):
        with pytest.raises(ValueError):
            merge_coco([], tmp_path / "out.json")
