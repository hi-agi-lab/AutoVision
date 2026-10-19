# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the task data layer and detection transforms."""

import json
from pathlib import Path

import pytest
import torch
from tests.conftest import create_image
from torchvision import tv_tensors

from autovision.augment.detection import (
    RTDETRDetectionCollateFunction,
    RTDETRDetectionTransform,
    RTDETRDetectionTransformArgs,
    build_val_transform_args,
    resolve_detection_step_schedule,
)
from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
    yolo_images_to_labels_path,
)
from autovision.errors import AutoVisionError


def make_coco(tmp_path, num_images: int = 4, num_classes: int = 2):
    images_dir = tmp_path / "images"
    rows = []
    categories = [
        {"id": index + 10, "name": f"class_{index}"}  # non-contiguous ids
        for index in range(num_classes)
    ]
    for index in range(num_images):
        path = create_image(images_dir / f"img_{index}.jpg", size=(64, 64), seed=index)
        rows.append(
            {
                "id": index + 1,
                "file_name": path.name,
                "width": 64,
                "height": 64,
            }
        )
    annotations = []
    box_id = 1
    for index, row in enumerate(rows):
        annotations.append(
            {
                "id": box_id,
                "image_id": row["id"],
                "category_id": 10 + (index % num_classes),
                "bbox": [10.0, 12.0, 20.0, 24.0],  # absolute xywh
                "area": 480.0,
                "iscrowd": 0,
            }
        )
        box_id += 1
    payload = {"images": rows, "annotations": annotations, "categories": categories}
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(payload))
    val_json = tmp_path / "val.json"
    val_payload = {"images": rows[:1], "annotations": annotations[:1], "categories": categories}
    val_json.write_text(json.dumps(val_payload))
    return train_json, val_json, images_dir, categories


class TestCOCOData:
    def test_args_and_rows(self, tmp_path):
        train_json, val_json, images_dir, categories = make_coco(tmp_path)
        args = COCOObjectDetectionDataArgs(
            train={"annotations": train_json, "images": images_dir},
            val={"annotations": val_json, "images": images_dir},
        )
        assert args.classes() == {0: "class_0", 1: "class_1"}
        rows = list(args.list_image_info("train"))
        assert len(rows) == 4
        row = rows[0]
        boxes = json.loads(row["bboxes"])
        labels = json.loads(row["class_labels"])
        # [10, 12, 20, 24] xywh pixels in a 64x64 image -> normalized cxcywh
        assert boxes[0] == pytest.approx([(10 + 10) / 64, (12 + 12) / 64, 20 / 64, 24 / 64])
        assert labels[0] == 0  # contiguous internal ids

    def test_ignore_classes(self, tmp_path):
        train_json, val_json, images_dir, _ = make_coco(tmp_path)
        args = COCOObjectDetectionDataArgs(
            train={"annotations": train_json, "images": images_dir},
            val={"annotations": val_json, "images": images_dir},
            ignore_classes={11},
        )
        rows = list(args.list_image_info("train"))
        assert len(rows) == 4
        all_labels = [json.loads(row["class_labels"]) for row in rows]
        assert all(labels in ([0], []) for labels in all_labels)
        assert sum(len(labels) for labels in all_labels) == 2  # half ignored

    def test_file_name_path_traversal_is_rejected(self, tmp_path):
        train_json, val_json, images_dir, _ = make_coco(tmp_path)
        payload = json.loads(train_json.read_text())
        payload["images"][0]["file_name"] = "../escape.jpg"
        train_json.write_text(json.dumps(payload))
        args = COCOObjectDetectionDataArgs(
            train={"annotations": train_json, "images": images_dir},
            val={"annotations": val_json, "images": images_dir},
        )
        with pytest.raises(AutoVisionError, match="escapes"):
            list(args.list_image_info("train"))

    def test_mmap_hash_changes_with_mtime(self, tmp_path):
        train_json, val_json, images_dir, _ = make_coco(tmp_path)
        args = COCOObjectDetectionDataArgs(
            train={"annotations": train_json, "images": images_dir},
            val={"annotations": val_json, "images": images_dir},
        )
        hash1 = args.train_data_mmap_hash()
        import os
        os.utime(train_json, (hash1 and 0, 999999999))
        assert args.train_data_mmap_hash() != hash1


class TestYOLOData:
    def _make_yolo(self, tmp_path, num_images: int = 4):
        images_dir = tmp_path / "train" / "images"
        labels_dir = yolo_images_to_labels_path(images_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)
        names = {3: "dog", 7: "cat"}
        for index in range(num_images):
            create_image(images_dir / f"img_{index}.jpg", size=(64, 64), seed=index)
            (labels_dir / f"img_{index}.txt").write_text(
                "3 0.5 0.5 0.2 0.3\n7 0.25 0.25 0.1 0.1\n"
            )
        args = YOLOObjectDetectionDataArgs(
            path=tmp_path, train=tmp_path / "train" / "images",
            val=tmp_path / "train" / "images", names=names,
        )
        return args, names

    def test_images_to_labels_path(self, tmp_path):
        assert yolo_images_to_labels_path(
            tmp_path / "train" / "images"
        ) == tmp_path / "train" / "labels"

    def test_rows(self, tmp_path):
        args, _ = self._make_yolo(tmp_path)
        rows = list(args.list_image_info("train"))
        assert len(rows) == 4
        boxes = json.loads(rows[0]["bboxes"])
        labels = json.loads(rows[0]["class_labels"])
        assert labels == [0, 1]  # internal contiguous mapping of {3, 7}
        assert boxes[0] == pytest.approx([0.5, 0.5, 0.2, 0.3])

    def test_missing_labels_skipped_when_configured(self, tmp_path):
        args, _ = self._make_yolo(tmp_path)
        args.skip_if_label_file_missing = True
        (args.train.parent / "labels" / "img_0.txt").unlink()
        rows = list(args.list_image_info("train"))
        assert len(rows) == 3

    def test_images_part_validated(self, tmp_path):
        with pytest.raises(ValueError, match="images"):
            YOLOObjectDetectionDataArgs(
                path=tmp_path, train=tmp_path / "x", val=tmp_path / "y",
                names={0: "a"},
            )

    def test_relative_splits_resolve_against_path(self, tmp_path):
        # Dict payloads pass no base_dir: relative splits must still anchor
        # to the (absolute) `path` root — the Ultralytics layout contract.
        args, _ = self._make_yolo(tmp_path)
        relative = YOLOObjectDetectionDataArgs(
            path=tmp_path, train="train/images", val="train/images",
            names={3: "dog", 7: "cat"},
        )
        relative.resolve_data_paths(base_dir=None)  # dict payload: no base_dir
        assert relative.train == tmp_path / "train" / "images"
        assert relative.val == tmp_path / "train" / "images"
        rows = list(relative.list_image_info("train"))
        assert len(rows) == 4

    def test_relative_path_anchors_with_base_dir(self, tmp_path):
        # Config-file payloads: base_dir anchors `path` first, then splits.
        relative = YOLOObjectDetectionDataArgs(
            path=Path("dataset"), train="train/images", val="val/images",
            names={3: "dog", 7: "cat"},
        )
        relative.resolve_data_paths(base_dir=tmp_path)
        assert relative.path == tmp_path / "dataset"
        assert relative.train == tmp_path / "dataset" / "train" / "images"


class TestStepSchedule:
    def test_short_run(self):
        schedule = resolve_detection_step_schedule(total_steps=120, steps_per_epoch=10)
        # 12 epochs: start = 12/3 = 4, flat = 4+6 = 10, stop = 12 - min(12, 12*12/72=2) = 10
        assert schedule.start == 40
        assert schedule.flat == 100
        assert schedule.stop == 100

    def test_long_run(self):
        schedule = resolve_detection_step_schedule(total_steps=7200, steps_per_epoch=100)
        # 72 epochs: start = 4, flat = 4 + 36 = 40, stop = 72 - 12 = 60
        assert schedule.start == 400
        assert schedule.flat == 4000
        assert schedule.stop == 6000

    def test_windows(self):
        schedule = resolve_detection_step_schedule(total_steps=7200, steps_per_epoch=100)
        assert schedule.window("mixup") == (400, 4000)
        assert schedule.window("photometric") == (400, 6000)

    def test_copyblend_step_window_auto_resolves(self):
        # Baseline CopyBlend window is [step_start, step_stop) with the
        # start shared with the photometric family. Both MUST resolve:
        # the collate treats an unresolved stop as None (whole-run ON)
        # and an unresolved start as OFF.
        args = RTDETRDetectionTransformArgs(
            image_size=(32, 32),
            copyblend_prob=0.5,
            scale_jitter=None,
        )
        args.resolve_step_schedule(
            total_steps=7200, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.copyblend_step_start == 400
        assert args.copyblend_step_stop == 6000
        assert args.random_erase_step_stop == 6000

    def test_copyblend_collate_gate_is_windowed(self):
        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(32, 32),
            copyblend_prob=0.5,
            mixup_prob=None,
            mosaic_prob=None,
            scale_jitter=None,
        )
        args.resolve_step_schedule(
            total_steps=7200, train_num_batches=100, gradient_accumulation_steps=1
        )
        collate = RTDETRDetectionCollateFunction(args, train=True)
        collate.set_step(399)
        assert collate._copyblend_active is False
        collate.set_step(400)
        assert collate._copyblend_active is True
        collate.set_step(6000)
        assert collate._copyblend_active is False

    def test_copyblend_unresolved_start_fails_closed(self):
        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(32, 32),
            copyblend_prob=0.5,
            mixup_prob=None,
            mosaic_prob=None,
            scale_jitter=None,
        )
        # No resolve_step_schedule call: start stays "auto".
        collate = RTDETRDetectionCollateFunction(args, train=True)
        collate.set_step(10_000)
        assert collate._copyblend_active is False

    def test_validation(self):
        with pytest.raises(ValueError):
            resolve_detection_step_schedule(100, 0)


class TestDetectionTransform:
    def _args(self) -> RTDETRDetectionTransformArgs:
        args = RTDETRDetectionTransformArgs(
            image_size=(32, 32),
            scale_jitter=None,
            photometric_distort_prob=0.5,
            random_zoom_out_prob=0.5,
            random_iou_crop_prob=0.8,
        )
        args.resolve_auto_fields(num_channels=3)
        args.resolve_step_schedule(total_steps=120, train_num_batches=12, gradient_accumulation_steps=1)
        args.resolve_incompatible()
        return args

    def _item(self):
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[8.0, 8.0, 56.0, 56.0]]), format="XYXY", canvas_size=(64, 64)
        )
        return {
            "image": image, "bboxes": boxes, "classes": torch.tensor([0]),
            "image_path": "img.jpg", "original_size": (64, 64),
        }

    def test_output_contract(self):
        transform = RTDETRDetectionTransform(self._args())
        transform.set_step(0)
        item = transform(self._item())
        assert item["image"].dtype == torch.float32
        assert item["image"].shape == (3, 32, 32)
        assert item["bboxes"].shape[1] == 4
        assert (item["bboxes"] >= 0).all() and (item["bboxes"] <= 1).all()
        assert item["classes"].dtype == torch.int64

    def test_step_windows_gate_augmentations(self):
        args = self._args()
        transform = RTDETRDetectionTransform(args)
        transform.set_step(int(args.photometric_step_start))  # active window
        # After the stop step, photometric/zoomout/ioucrop are inactive.
        late_step = int(args.photometric_step_stop) + 1
        transform.set_step(late_step)
        # The active tuple changes -> dataloader reinitialization pending.
        assert transform.requires_dataloader_reinitialization()
        transform.mark_dataloader_as_reinitialized()
        assert not transform.requires_dataloader_reinitialization()
        item = transform(self._item())
        assert item["image"].shape == (3, 32, 32)

    def test_val_args_disable_augmentations(self):
        args = self._args()
        val_args = build_val_transform_args(args)
        assert val_args.photometric_distort_prob is None
        assert val_args.random_iou_crop_prob is None
        assert val_args.scale_jitter is None
        val_transform = RTDETRDetectionTransform(val_args)
        val_transform.set_step(0)
        item = val_transform(self._item())
        assert item["image"].shape == (3, 32, 32)

    def test_compose_caching(self):
        transform = RTDETRDetectionTransform(self._args())
        transform.set_step(0)
        compose_a = transform.compose.compose_for(mosaic_sample=False)
        transform.set_step(0)
        compose_b = transform.compose.compose_for(mosaic_sample=False)
        assert compose_a is compose_b  # cached per active tuple


class TestDetectionCollate:
    """Collate consumes per-sample TRANSFORM OUTPUTS (normalized cxcywh)."""

    def _args(self):
        args = RTDETRDetectionTransformArgs(
            image_size=(32, 32), scale_jitter=None,
            photometric_distort_prob=None, random_zoom_out_prob=None,
            random_iou_crop_prob=None, mixup_prob=None,
        )
        args.resolve_auto_fields(num_channels=3)
        return args

    def _items(self, num: int = 2):
        # As emitted by RTDETRDetectionTransform: float image resized to
        # args.image_size (32x32), boxes normalized cxcywh.
        items = []
        for index in range(num):
            image = torch.rand(3, 32, 32)
            boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
            items.append(
                {"image": image, "bboxes": boxes, "classes": torch.tensor([0]),
                 "image_path": f"img_{index}.jpg", "original_size": (64, 48)}
            )
        return items

    def test_batch_output(self):
        collate = RTDETRDetectionCollateFunction(self._args(), train=True)
        collate.set_step(0)
        batch = collate(self._items(3))
        assert batch["image"].shape == (3, 3, 32, 32)
        assert len(batch["bboxes"]) == 3
        assert batch["bboxes"][0].shape[1] == 4
        assert len(batch["image_path"]) == 3
        assert batch["original_size"] == [(64, 48)] * 3  # TRUE original sizes

    def test_scale_jitter_keeps_normalized_boxes(self):
        args = self._args()
        from autovision.augment.detection import ScaleJitterArgs
        args.scale_jitter = ScaleJitterArgs(sizes=[40, 48])
        collate = RTDETRDetectionCollateFunction(args, train=True)
        collate.set_step(0)
        batch = collate(self._items(2))
        # Whole batch shares one grid; boxes stay in [0,1].
        assert batch["image"].shape[2] == batch["image"].shape[3]
        for boxes in batch["bboxes"]:
            assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_mixup_concatenates_boxes(self):
        args = self._args()
        args.mixup_prob = 1.0
        args.mixup_step_start = 0
        args.mixup_step_stop = None
        collate = RTDETRDetectionCollateFunction(args, train=True)
        collate.set_step(0)  # window starts at 0 -> active
        batch = collate(self._items(2))
        assert batch["bboxes"][0].shape[0] == 2  # own + rolled neighbor's boxes

    def test_small_boxes_filtered(self):
        collate = RTDETRDetectionCollateFunction(self._args(), train=True)
        collate.set_step(0)
        items = self._items(1)
        items[0]["bboxes"] = torch.tensor([[0.5, 0.5, 2.0 / 64, 2.0 / 48]])
        batch = collate(items)
        assert batch["bboxes"][0].shape[0] == 0  # 2x2 px < 4px minimum
