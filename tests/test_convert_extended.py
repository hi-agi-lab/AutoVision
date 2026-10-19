# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Conversion extensions (roadmap C3): VOC, YOLO-pose, YOLO-seg, merge CLI."""

import json
from pathlib import Path

import pytest
from tests.conftest import create_image
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.data.convert import (
    merge_coco,
    voc_to_coco,
    yolo_pose_to_coco,
    yolo_seg_to_coco,
)


def _voc_tree(tmp_path: Path) -> Path:
    root = tmp_path / "VOC2007"
    (root / "Annotations").mkdir(parents=True)
    images = root / "JPEGImages"
    create_image(images / "000001.jpg", size=(64, 48), seed=0)
    create_image(images / "000002.jpg", size=(64, 48), seed=1)
    (root / "ImageSets" / "Main").mkdir(parents=True)
    (root / "ImageSets" / "Main" / "train.txt").write_text("000001\n")

    def _xml(stem: str, width: int, height: int, objects: str) -> None:
        (root / "Annotations" / f"{stem}.xml").write_text(
            f"""<annotation>
  <folder>VOC2007</folder>
  <filename>{stem}.jpg</filename>
  <size><width>{width}</width><height>{height}</height><depth>3</depth></size>
  {objects}
</annotation>"""
        )

    _xml(
        "000001", 64, 48,
        """<object><name>cat</name><pose>Left</pose><truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object>
    <object><name>dog</name><truncated>1</truncated><difficult>1</difficult>
      <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>""",
    )
    _xml(
        "000002", 64, 48,
        """<object><name>cat</name><difficult>0</difficult>
      <bndbox><xmin>3</xmin><ymin>3</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>""",
    )
    return root


class TestVocToCoco:
    def test_voc_tree_to_coco(self, tmp_path):
        root = _voc_tree(tmp_path)
        out = voc_to_coco(root, tmp_path / "voc.json", imagesets="train")
        payload = json.loads(out.read_text())
        # ImageSets/Main/train.txt restricts to 000001.
        assert [image["file_name"] for image in payload["images"]] == ["000001.jpg"]
        assert [(c["id"], c["name"]) for c in payload["categories"]] == [
            (1, "cat"), (2, "dog")
        ]
        first, second = payload["annotations"]
        # 1-based inclusive VOC box -> 0-based xywh: (0, 1) .. (9, 19).
        assert first["bbox"] == [0.0, 1.0, 10.0, 19.0]
        assert first["area"] == pytest.approx(190.0)
        assert first["iscrowd"] == 0
        # difficult=1 becomes a COCO ignore region (iscrowd=1) + flags kept.
        assert second["iscrowd"] == 1 and second["difficult"] == 1
        assert second["truncated"] == 1

    def test_full_tree_without_split(self, tmp_path):
        root = _voc_tree(tmp_path)
        out = voc_to_coco(root, tmp_path / "all.json")
        payload = json.loads(out.read_text())
        assert [image["file_name"] for image in payload["images"]] == [
            "000001.jpg", "000002.jpg"
        ]

    def test_class_names_override_fixes_ids(self, tmp_path):
        root = _voc_tree(tmp_path)
        out = voc_to_coco(
            root, tmp_path / "fixed.json", class_names=["dog", "cat"]
        )
        payload = json.loads(out.read_text())
        assert [(c["id"], c["name"]) for c in payload["categories"]] == [
            (1, "dog"), (2, "cat")
        ]
        cat_rows = [a for a in payload["annotations"] if a["category_id"] == 2]
        assert len(cat_rows) == 2

    def test_missing_image_fails_loud(self, tmp_path):
        root = _voc_tree(tmp_path)
        (root / "JPEGImages" / "000001.jpg").unlink()
        with pytest.raises(FileNotFoundError, match="000001"):
            voc_to_coco(root, tmp_path / "x.json", imagesets="train")


class TestYoloPoseToCoco:
    def _pose_tree(self, tmp_path: Path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        labels.mkdir()
        create_image(images / "p0.jpg", size=(64, 64), seed=0)
        create_image(images / "p1.jpg", size=(64, 64), seed=1)
        # 17 keypoints: first 3 visible (v=2), rest unlabeled (0,0,0).
        visible = "0.30 0.30 2 0.35 0.35 2 0.40 0.40 2 " + " ".join(
            ["0.0 0.0 0"] * 14
        )
        (labels / "p0.txt").write_text(f"0 0.5 0.5 0.75 0.75 {visible}\n")
        (labels / "p1.txt").write_text(
            "0 0.5 0.5 0.5 0.5 " + " ".join(["0.0 0.0 0"] * 17) + "\n"
        )
        return images, labels

    def test_pose_labels_load_through_the_keypoint_dataset(self, tmp_path):
        images, labels = self._pose_tree(tmp_path)
        out = yolo_pose_to_coco(
            images, ["person"], tmp_path / "pose.json", num_keypoints=17,
            labels_dir=labels,
        )
        payload = json.loads(out.read_text())
        first, second = payload["annotations"]
        assert first["num_keypoints"] == 3
        assert second["num_keypoints"] == 0
        # cx=cy=32, w=h=48 -> xyxy (8, 8, 56, 56).
        assert first["bbox"] == pytest.approx([8.0, 8.0, 48.0, 48.0])
        assert first["keypoints"][:3] == pytest.approx([19.2, 19.2, 2.0])
        assert first["keypoints"][3:6] == pytest.approx([22.4, 22.4, 2.0])
        assert first["keypoints"][9:] == pytest.approx([0.0] * (3 * 14))

        # Round-trip through the pose data plane (the strongest check).
        from autovision.data.coco_keypoint_dataset import CocoKeypointDataset

        dataset = CocoKeypointDataset(out, images, num_body_points=17)
        item = dataset[0]
        assert item["keypoints"].shape == (1, 17, 3)
        assert item["bboxes"].shape == (1, 4)
        assert item["area"][0] == pytest.approx(48.0 * 48.0)
        # p1's all-invisible annotation is filtered by the dataset.
        assert len(dataset) == 2 and dataset[1]["keypoints"].shape == (0, 17, 3)

    def test_wrong_keypoint_count_fails_loud(self, tmp_path):
        images, labels = self._pose_tree(tmp_path)
        with pytest.raises(ValueError, match="expected"):
            yolo_pose_to_coco(
                images, ["person"], tmp_path / "x.json", num_keypoints=13,
                labels_dir=labels,
            )


class TestVocNamespaces:
    def test_default_namespaced_voc_tree_parses(self, tmp_path):
        """Tool-exported VOC trees wrap elements in a namespace; local-name
        matching must still read them."""
        root = tmp_path / "VOC2007"
        (root / "Annotations").mkdir(parents=True)
        create_image(root / "JPEGImages" / "000001.jpg", size=(32, 32), seed=0)
        (root / "Annotations" / "000001.xml").write_text(
            """<annotation xmlns="http://example.org/voc">
  <size><width>32</width><height>32</height></size>
  <object><name>cat</name>
    <bndbox><xmin>2</xmin><ymin>2</ymin><xmax>12</xmax><ymax>12</ymax></bndbox>
  </object>
</annotation>"""
        )
        payload = json.loads(voc_to_coco(root, tmp_path / "ns.json").read_text())
        assert [c["name"] for c in payload["categories"]] == ["cat"]
        assert payload["annotations"][0]["bbox"] == [1.0, 1.0, 11.0, 11.0]


class TestYoloClassIndexBounds:
    def test_negative_index_fails_loud(self, tmp_path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        labels.mkdir()
        create_image(images / "a.png", size=(32, 32), seed=0)
        (labels / "a.txt").write_text("-1 0.5 0.5 0.2 0.2\n")
        with pytest.raises(ValueError, match="class index -1"):
            from autovision.data.convert import yolo_to_coco

            yolo_to_coco(images, names=["cat", "dog"], out_json=tmp_path / "x.json",
                         labels_dir=labels)
        with pytest.raises(ValueError, match="class index -1"):
            yolo_seg_to_coco(
                images, names=["cat", "dog"], out_json=tmp_path / "y.json",
                labels_dir=labels,
            )


class TestMergeWarnsOnDropped:
    def test_undeclared_annotations_are_reported(self, tmp_path, caplog):
        payload = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 8, "height": 8}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
                {"id": 2, "image_id": 99, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 77,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        path = tmp_path / "odd.json"
        path.write_text(json.dumps(payload))
        with caplog.at_level("WARNING"):
            merged = json.loads(merge_coco([path], tmp_path / "m.json").read_text())
        assert len(merged["annotations"]) == 1
        assert any("dropped" in record.message for record in caplog.records)


class TestYoloSegToCoco:
    def test_polygon_geometry(self, tmp_path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        labels.mkdir()
        create_image(images / "s0.png", size=(100, 100), seed=0)
        # Right triangle with 10px legs (normalized by 100).
        (labels / "s0.txt").write_text("0 0.1 0.1 0.2 0.1 0.1 0.2\n")
        out = yolo_seg_to_coco(images, ["cat"], tmp_path / "seg.json", labels_dir=labels)
        payload = json.loads(out.read_text())
        annotation = payload["annotations"][0]
        assert annotation["segmentation"] == [
            pytest.approx([10.0, 10.0, 20.0, 10.0, 10.0, 20.0])
        ]
        assert annotation["bbox"] == pytest.approx([10.0, 10.0, 10.0, 10.0])
        assert annotation["area"] == pytest.approx(50.0)  # shoelace

    def test_odd_coordinate_count_fails_loud(self, tmp_path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        labels.mkdir()
        create_image(images / "s0.png", size=(100, 100), seed=0)
        (labels / "s0.txt").write_text("0 0.1 0.1 0.2 0.1 0.1\n")
        with pytest.raises(ValueError, match="segmentation rows need"):
            yolo_seg_to_coco(images, ["cat"], tmp_path / "seg.json", labels_dir=labels)


class TestConvertCli:
    def test_voc_and_merge_via_cli(self, tmp_path):
        root = _voc_tree(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["convert-data", "from=voc", f"voc-root={root}",
             f"out={tmp_path / 'cli_voc.json'}", "imagesets=train"],
        )
        assert result.exit_code == 0, result.output
        merged = runner.invoke(
            app,
            ["convert-data", "from=merge",
             f"parts={tmp_path / 'cli_voc.json'},{tmp_path / 'cli_voc.json'}",
             f"out={tmp_path / 'cli_merged.json'}"],
        )
        assert merged.exit_code == 0, merged.output
        payload = json.loads((tmp_path / "cli_merged.json").read_text())
        assert len(payload["images"]) == 2  # same file merged twice

    def test_documented_yolo_and_coco_examples_work(self, tmp_path):
        """Regression: the docstring examples (out= / annotations= /
        out-labels=) previously died on parameter-name mismatches."""
        from tests.conftest import create_image as _create

        images = tmp_path / "images"
        labels = tmp_path / "labels"
        labels.mkdir()
        _create(images / "a.png", size=(32, 32), seed=0)
        (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        names = tmp_path / "names.yaml"
        names.write_text("names: [cat]\n")
        runner = CliRunner()
        forward = runner.invoke(
            app,
            ["convert-data", "from=yolo", f"images={images}", f"names={names}",
             f"out={tmp_path / 'train.json'}"],
        )
        assert forward.exit_code == 0, forward.output
        backward = runner.invoke(
            app,
            ["convert-data", "from=coco", f"annotations={tmp_path / 'train.json'}",
             f"out-labels={tmp_path / 'labels2'}"],
        )
        assert backward.exit_code == 0, backward.output
        assert (tmp_path / "labels2" / "a.txt").read_text().strip() == (
            "0 0.500000 0.500000 0.500000 0.500000"
        )

    def test_unknown_source_fails_loud(self, tmp_path):
        result = CliRunner().invoke(app, ["convert-data", "from=nope", "out=x.json"])
        assert result.exit_code != 0
        assert "from= must be one of" in result.output
