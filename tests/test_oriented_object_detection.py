# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the YOLO-OBB trio: geometry (obb.py), dataset, transform/NMS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from tests.conftest import create_image
from torchvision.ops import box_iou

from autovision.augment.detection.args import (
    OrientedObjectDetectionTransformArgs,
)
from autovision.augment.detection.oriented_transform import (
    OrientedObjectDetectionCollateFunction,
    OrientedObjectDetectionTransform,
    build_oriented_val_transform_args,
)
from autovision.data.yolo_oriented_object_detection_dataset import (
    OrientedObjectDetectionDataset,
    YOLOOrientedObjectDetectionDataArgs,
)
from autovision.geometry.obb import (
    obb_nms,
    rotated_iou_matrix,
    xywhr_to_xyxyxyxy,
    xyxyxyxy_to_xywhr,
)

# A square with corners (8,8),(24,8),(24,24),(8,24) in a 64x64 image.
SQ_NORM = [0.125, 0.125, 0.375, 0.125, 0.375, 0.375, 0.125, 0.375]
SQ_PX = torch.tensor([[8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0]])


def make_yolo_obb_dataset(tmp_path: Path, skip_if_missing: bool = False):
    """Ultralytics layout: <root>/images/<split> + <root>/labels/<split>."""
    root = tmp_path / "obbroot"
    names = {0: "plane", 1: "ship"}
    num = {"train": 3, "val": 2}
    sq_line = "0 " + " ".join(f"{v:g}" for v in SQ_NORM)
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index in range(num[split]):
            create_image(images_dir / f"img_{index:03d}.jpg", size=(64, 64), seed=index)
            if index == 1:
                continue  # no label file at all: exercises the missing path
            lines = [
                sq_line,
                "0 0.1 0.1 0.2 0.2",  # wrong field count -> dropped
                "7 0.1 0.1 0.2 0.2 0.3 0.2 0.3 0.1",  # unknown class -> dropped
            ]
            (labels_dir / f"img_{index:03d}.txt").write_text("\n".join(lines))
    return YOLOOrientedObjectDetectionDataArgs(
        path=root,
        train=root / "images" / "train",
        val=root / "images" / "val",
        names=names,
        skip_if_label_file_missing=skip_if_missing,
    )


class TestOBBGeometry:
    def test_corner_roundtrip_exact_for_axis_aligned(self):
        polys = torch.tensor([[8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0]])
        xywhr = xyxyxyxy_to_xywhr(polys)
        assert torch.allclose(xywhr[0, :4], torch.tensor([16.0, 16.0, 16.0, 16.0]))
        assert xywhr[0, 4].abs() < 1e-6
        recovered = xywhr_to_xyxyxyxy(xywhr).reshape(1, 8)
        assert torch.allclose(recovered, polys, atol=1e-5)

    def test_xywhr_roundtrip_region_preserved_for_rotations(self):
        angles = torch.tensor(
            [0.0, 0.3, math.pi / 4, 1.2, -0.7, -1.4, 2.0, -2.0, 2.5, math.pi]
        )
        boxes = torch.stack(
            [
                torch.full_like(angles, 5.0),
                torch.full_like(angles, 7.0),
                torch.full_like(angles, 4.0),
                torch.full_like(angles, 2.0),
                angles,
            ],
            dim=-1,
        )
        polys = xywhr_to_xyxyxyxy(boxes)
        recovered = xyxyxyxy_to_xywhr(polys)
        # Canonical form: the width edge lands in [-pi/4, pi/4].
        assert recovered[:, 4].abs().max() <= math.pi / 4 + 1e-6
        iou = rotated_iou_matrix(boxes, recovered).diagonal()
        assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)

    def test_canonical_angle_normalization_swaps_wh(self):
        # width axis vertical (theta=pi/2) == width axis flipped at theta=-pi/2
        box = torch.tensor([[0.0, 0.0, 1.0, 3.0, math.pi / 2]])
        canonical = xyxyxyxy_to_xywhr(xywhr_to_xyxyxyxy(box))
        assert canonical[0, 2] == pytest.approx(3.0)
        assert canonical[0, 3] == pytest.approx(1.0)
        assert -math.pi / 2 <= canonical[0, 4].item() < math.pi / 2
        iou = rotated_iou_matrix(box, canonical)[0, 0]
        assert iou == pytest.approx(1.0, abs=1e-5)

    def test_rotated_iou_matches_torchvision_for_axis_aligned(self):
        # xywhr (cx, cy, w, h, 0) -> xyxy corners.
        boxes = torch.tensor(
            [[10.0, 10.0, 8.0, 4.0, 0.0], [13.0, 12.0, 8.0, 4.0, 0.0]]
        )
        polys = xywhr_to_xyxyxyxy(boxes).reshape(2, 8)
        xyxy = torch.stack(
            [polys[:, 0::2].min(-1).values, polys[:, 1::2].min(-1).values,
             polys[:, 0::2].max(-1).values, polys[:, 1::2].max(-1).values],
            dim=-1,
        )
        ours = rotated_iou_matrix(boxes, boxes)
        ref = box_iou(xyxy, xyxy)
        assert torch.allclose(ours, ref, atol=1e-6)

    def test_rotated_iou_45deg_centered_overlap(self):
        # Two unit-area-4 squares (side 2), one rotated 45 deg about the same
        # center: intersection is an octagon, IoU == sqrt(2)/2.
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        b = torch.tensor([[0.0, 0.0, 2.0, 2.0, math.pi / 4]])
        iou = rotated_iou_matrix(a, b)[0, 0]
        assert iou == pytest.approx(math.sqrt(2) / 2, abs=1e-4)

    def test_rotated_iou_disjoint_is_zero(self):
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.3]])
        b = torch.tensor([[20.0, 20.0, 2.0, 2.0, -0.9]])
        assert rotated_iou_matrix(a, b)[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_obb_nms_suppresses_only_high_iou(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0, 0.0],   # top score
                [10.0, 10.0, 2.0, 2.0, 0.0],  # disjoint -> kept
                [0.0, 0.0, 2.0, 2.0, 0.0],   # duplicate -> suppressed
                [1.0, 0.0, 2.0, 2.0, 0.0],   # IoU 1/3 with box 0 -> kept
            ]
        )
        scores = torch.tensor([0.9, 0.7, 0.8, 0.85])
        keep = obb_nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [0, 3, 1]  # score-descending order

    def test_obb_nms_empty_input(self):
        keep = obb_nms(torch.zeros(0, 5), torch.zeros(0), 0.5)
        assert keep.numel() == 0


class TestYOLOOBBDataset:
    def test_list_image_info_parses_nine_field_lines_only(self, tmp_path):
        args = make_yolo_obb_dataset(tmp_path)
        rows = list(args.list_image_info("train"))
        # img_001 has no label file -> empty boxes, still listed.
        assert len(rows) == 3
        boxes = json.loads(rows[0]["bboxes"])
        assert boxes == [SQ_NORM]
        assert json.loads(rows[0]["class_labels"]) == [0]
        assert json.loads(rows[1]["bboxes"]) == []

    def test_skip_if_label_file_missing(self, tmp_path):
        args = make_yolo_obb_dataset(tmp_path, skip_if_missing=True)
        assert len(list(args.list_image_info("train"))) == 2

    def test_dataset_item_absolute_pixel_corners(self, tmp_path):
        args = make_yolo_obb_dataset(tmp_path)
        rows = list(args.list_image_info("train"))
        dataset = OrientedObjectDetectionDataset(args, rows)
        item = dataset[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["bboxes"].shape == (1, 8)
        assert torch.allclose(item["bboxes"], SQ_PX)
        assert item["classes"].tolist() == [0]
        assert item["original_size"] == (64, 64)

    def test_dataset_item_with_empty_boxes(self, tmp_path):
        args = make_yolo_obb_dataset(tmp_path)
        rows = list(args.list_image_info("train"))
        dataset = OrientedObjectDetectionDataset(args, rows)
        item = dataset[1]  # empty label file
        assert item["bboxes"].shape == (0, 8)
        assert item["classes"].numel() == 0


def _val_args(image_size: tuple[int, int] = (64, 64)) -> OrientedObjectDetectionTransformArgs:
    return OrientedObjectDetectionTransformArgs(
        image_size=image_size,
        photometric_distort_prob=None,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
    )


def _item(polys: torch.Tensor, size: tuple[int, int] = (64, 64)):
    return {
        "image_path": "img.jpg",
        "image": torch.randint(0, 256, (3, size[1], size[0]), dtype=torch.uint8),
        "bboxes": polys.clone(),
        "classes": torch.zeros(polys.shape[0], dtype=torch.int64),
        "original_size": size,
    }


class TestOrientedTransform:
    def test_resize_keeps_normalized_corners_invariant(self):
        transform = OrientedObjectDetectionTransform(_val_args(image_size=(32, 32)))
        out = transform(_item(SQ_PX))
        assert out["image"].shape == (3, 32, 32)
        assert out["image"].dtype == torch.float32
        assert torch.isfinite(out["image"]).all()
        assert torch.allclose(out["bboxes"], torch.tensor([SQ_NORM]), atol=1e-6)
        assert out["bboxes"].min() >= 0.0 and out["bboxes"].max() <= 1.0

    def test_horizontal_flip_mirrors_x(self, monkeypatch):
        monkeypatch.setattr(
            "autovision.augment.detection.oriented_transform.random.random",
            lambda: 0.0,
        )
        args = _val_args()
        args.horizontal_flip_prob = 1.0
        out = OrientedObjectDetectionTransform(args)(_item(SQ_PX))
        expected = torch.tensor([SQ_NORM])
        expected[:, 0::2] = 1.0 - expected[:, 0::2]
        assert torch.allclose(out["bboxes"], expected, atol=1e-6)

    def test_vertical_flip_mirrors_y(self, monkeypatch):
        monkeypatch.setattr(
            "autovision.augment.detection.oriented_transform.random.random",
            lambda: 0.0,
        )
        args = _val_args()
        args.vertical_flip_prob = 1.0
        out = OrientedObjectDetectionTransform(args)(_item(SQ_PX))
        expected = torch.tensor([SQ_NORM])
        expected[:, 1::2] = 1.0 - expected[:, 1::2]
        assert torch.allclose(out["bboxes"], expected, atol=1e-6)

    def test_degenerate_polygon_dropped(self):
        polys = torch.cat(
            [SQ_PX, torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])]  # zero area
        )
        transform = OrientedObjectDetectionTransform(_val_args())
        out = transform(_item(polys))
        assert out["bboxes"].shape == (1, 8)
        assert out["classes"].tolist() == [0]

    def test_val_args_disable_augs_and_relax_filter(self):
        train_args = OrientedObjectDetectionTransformArgs()
        val_args = build_oriented_val_transform_args(train_args)
        assert val_args.photometric_distort_prob is None
        assert val_args.horizontal_flip_prob == 0.0
        assert val_args.vertical_flip_prob == 0.0
        assert val_args.min_bbox_size_px == 1.0
        assert train_args.min_bbox_size_px == 4.0  # train args untouched


class TestOrientedCollate:
    def test_batch_shapes_and_tiny_polygon_filter(self):
        args = _val_args()
        args.min_bbox_size_px = 4.0
        collate = OrientedObjectDetectionCollateFunction(args, train=True)
        tiny = torch.tensor([[0.0, 0.0, 1 / 32, 0.0, 1 / 32, 1 / 32, 0.0, 1 / 32]])
        batch = [
            {
                "image": torch.randn(3, 32, 32),
                "bboxes": torch.tensor([SQ_NORM]),
                "classes": torch.tensor([1]),
                "image_path": "a.jpg",
                "original_size": (64, 64),
            },
            {
                "image": torch.randn(3, 32, 32),
                "bboxes": torch.cat([torch.tensor([SQ_NORM]), tiny]),  # tiny: 1 px^2 < 4
                "classes": torch.tensor([0, 1]),
                "image_path": "b.jpg",
                "original_size": (64, 64),
            },
        ]
        out = collate(batch)
        assert out["image"].shape == (2, 3, 32, 32)
        assert out["input_size"] == (32, 32)
        assert out["bboxes"][0].shape == (1, 8)
        assert out["bboxes"][1].shape == (1, 8)  # tiny polygon filtered
        assert out["classes"][1].tolist() == [0]
        assert out["image_path"] == ["a.jpg", "b.jpg"]
        assert out["original_size"] == [(64, 64), (64, 64)]

    def test_collate_contract_methods_are_noop(self):
        collate = OrientedObjectDetectionCollateFunction(_val_args(), train=False)
        collate.set_step(10)
        assert not collate.requires_dataloader_reinitialization()
        collate.mark_dataloader_as_reinitialized()


class TestEndToEnd:
    def test_dataset_transform_collate_pipeline(self, tmp_path):
        args = make_yolo_obb_dataset(tmp_path)
        rows = list(args.list_image_info("train"))
        transform = OrientedObjectDetectionTransform(_val_args(image_size=(32, 32)))
        dataset = OrientedObjectDetectionDataset(args, rows, transform=transform)
        collate = OrientedObjectDetectionCollateFunction(_val_args(image_size=(32, 32)))
        batch = collate([dataset[0], dataset[1], dataset[2]])
        assert batch["image"].shape == (3, 3, 32, 32)
        assert all(b.shape[-1] == 8 for b in batch["bboxes"])
        assert batch["bboxes"][0].min() >= 0.0 and batch["bboxes"][0].max() <= 1.0
        assert len(batch["image_path"]) == 3
