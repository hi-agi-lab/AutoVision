# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""CVAT / Label Studio annotation interchange (phase 3 tail)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from autovision.data.interchange import cvat_to_coco, label_studio_to_coco
from autovision.errors import AutoVisionError


def _image(path: Path, size=(100, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(path)
    return path


def _cvat_xml(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>cat</name></label>
        <label><name>dog</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="a.png" width="100" height="80">
    <box label="cat" xtl="10" ytl="20" xbr="40" ybr="60" occluded="0"/>
    <polygon label="dog" points="50,10;80,10;80,40;50,40"/>
  </image>
  <image id="1" name="b.png" width="100" height="80"/>
</annotations>
"""
    )
    return path


def test_cvat_to_coco_and_loader_compatibility(tmp_path):
    images = tmp_path / "images"
    _image(images / "a.png", (100, 80))
    _image(images / "b.png", (100, 80))
    xml = _cvat_xml(tmp_path / "annotations.xml")
    out = cvat_to_coco(xml, tmp_path / "coco.json", images_dir=images)
    payload = json.loads(out.read_text())
    assert [c["name"] for c in payload["categories"]] == ["cat", "dog"]
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 2
    box, polygon = payload["annotations"]
    assert box["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert box["area"] == pytest.approx(1200.0)
    assert box["category_id"] == 1
    assert polygon["bbox"] == [50.0, 10.0, 30.0, 30.0]
    assert polygon["area"] == pytest.approx(900.0)
    assert len(polygon["segmentation"][0]) == 8

    data = COCOObjectDetectionDataArgs(
        format="coco",
        train={"annotations": out, "images": images},
        val={"annotations": out, "images": images},
    )
    rows = list(data.list_image_info("train"))
    assert len(rows) == 2
    assert json.loads(rows[0]["class_labels"]) == [0, 1]
    assert json.loads(rows[1]["class_labels"]) == []


def _ls_json(path: Path) -> Path:
    tasks = [
        {
            "id": 1,
            "data": {"image": "/data/local-files/?d=ls1.png"},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                                "rotation": 0,
                                "rectanglelabels": ["cat"],
                            },
                        },
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {
                                "points": [[10, 10], [50, 10], [50, 50]],
                                "polygonlabels": ["dog"],
                            },
                        },
                        {
                            "type": "keypointlabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {"x": 50, "y": 50, "keypointlabels": ["cat"]},
                        },
                    ]
                }
            ],
        },
        {"id": 2, "data": {"image": "ls2.png"}, "annotations": []},
    ]
    path.write_text(json.dumps(tasks))
    return path


def test_label_studio_to_coco(tmp_path):
    images = tmp_path / "images"
    _image(images / "ls1.png", (100, 200))
    _image(images / "ls2.png", (100, 200))
    source = _ls_json(tmp_path / "ls.json")
    out = label_studio_to_coco(source, tmp_path / "coco.json", images_dir=images)
    payload = json.loads(out.read_text())
    assert [c["name"] for c in payload["categories"]] == ["cat", "dog"]
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 3
    rectangle, polygon, keypoint = payload["annotations"]
    assert rectangle["bbox"] == [10.0, 40.0, 30.0, 80.0]
    assert polygon["bbox"] == [10.0, 20.0, 40.0, 80.0]
    assert polygon["area"] == pytest.approx(1600.0)
    assert keypoint["bbox"] == [48.0, 98.0, 4.0, 4.0]
    assert keypoint["keypoint"] == [50.0, 100.0]


def test_malformed_inputs_are_rejected(tmp_path):
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<annotations>")
    with pytest.raises(AutoVisionError, match="Cannot parse CVAT XML"):
        cvat_to_coco(bad_xml, tmp_path / "out.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{")
    with pytest.raises(AutoVisionError, match="Cannot read Label Studio JSON"):
        label_studio_to_coco(bad_json, tmp_path / "out.json")


def test_convert_data_cli_wires_both_sources(tmp_path):
    images = tmp_path / "images"
    _image(images / "a.png", (100, 80))
    _image(images / "b.png", (100, 80))
    _image(images / "ls1.png", (100, 200))
    _image(images / "ls2.png", (100, 200))
    xml = _cvat_xml(tmp_path / "annotations.xml")
    ls = _ls_json(tmp_path / "ls.json")
    runner = CliRunner()

    cvat_out = tmp_path / "cvat.json"
    result = runner.invoke(
        app,
        [
            "convert-data", "from=cvat", f"cvat-xml={xml}",
            f"images={images}", f"out={cvat_out}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert cvat_out.is_file()

    ls_out = tmp_path / "ls.json.coco"
    result = runner.invoke(
        app,
        [
            "convert-data", "from=label-studio", f"ls-json={ls}",
            f"images={images}", f"out={ls_out}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ls_out.is_file()
