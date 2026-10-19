# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Annotation XML is parsed with defusedxml (entity/DTD hardening)."""

from __future__ import annotations

from pathlib import Path

import pytest
from defusedxml.common import DefusedXmlException

from autovision.data.convert import voc_to_coco
from autovision.data.interchange.cvat import cvat_to_coco
from autovision.errors import AutoVisionError

_MALICIOUS = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<annotation>
  <filename>a.jpg</filename>
  <size><width>4</width><height>4</height><depth>3</depth></size>
  <object><name>cat</name><bndbox>
    <xmin>0</xmin><ymin>0</ymin><xmax>2</xmax><ymax>2</ymax>
  </bndbox></object>
  <note>&lol2;</note>
</annotation>
"""


def test_voc_rejects_entity_expansion(tmp_path: Path):
    annotations = tmp_path / "VOC2007" / "Annotations"
    annotations.mkdir(parents=True)
    (annotations / "a.xml").write_text(_MALICIOUS)
    with pytest.raises(DefusedXmlException):
        voc_to_coco(tmp_path / "VOC2007", tmp_path / "out.json")


def test_cvat_rejects_entity_expansion(tmp_path: Path):
    xml = tmp_path / "annotations.xml"
    xml.write_text(_MALICIOUS)
    with pytest.raises(DefusedXmlException):
        cvat_to_coco(xml, tmp_path / "out.json")


def test_cvat_image_path_cannot_escape_images_dir(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    xml = tmp_path / "annotations.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta><task><labels><label><name>cat</name></label></labels></task></meta>
  <image id="0" name="../outside.png">
    <box label="cat" xtl="0" ytl="0" xbr="2" ybr="2" occluded="0"/>
  </image>
</annotations>
"""
    )
    with pytest.raises(AutoVisionError, match="escapes images_dir"):
        cvat_to_coco(xml, tmp_path / "out.json", images_dir=images)
