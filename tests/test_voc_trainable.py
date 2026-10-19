# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""B3 gate: a VOC tree trains directly (convert -> train -> val), CPU-only.

The synthetic corpus below mimics the real VOC2007 layout (20 classes with
their canonical names, per-object ``difficult`` flags, ImageSets splits);
the official download host throttles to ~14 kB/s in some environments, so
the automated gate exercises the same code path on a generated tree and the
real-corpus run is recorded separately.
"""

import json

from tests.conftest import create_image
from typer.testing import CliRunner

import autovision
from autovision.cli._app import app

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

MODEL_ARGS = {"image_size": (64, 64), "num_queries": 10, "num_top_queries": 10}


def _write_voc_tree(root, *, num_images: int = 6) -> None:
    (root / "Annotations").mkdir(parents=True)
    (root / "ImageSets" / "Main").mkdir(parents=True)
    stems = []
    for index in range(num_images):
        stem = f"{index + 1:06d}"
        stems.append(stem)
        create_image(root / "JPEGImages" / f"{stem}.jpg", size=(64, 64), seed=index)
        name = VOC_CLASSES[index % len(VOC_CLASSES)]
        (root / "Annotations" / f"{stem}.xml").write_text(
            f"""<annotation>
  <folder>VOC2007</folder>
  <filename>{stem}.jpg</filename>
  <size><width>64</width><height>64</height><depth>3</depth></size>
  <object>
    <name>{name}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>{8 + index}</xmin><ymin>10</ymin><xmax>{30 + index}</xmax><ymax>40</ymax></bndbox>
  </object>
  <object>
    <name>{VOC_CLASSES[(index + 7) % len(VOC_CLASSES)]}</name>
    <truncated>0</truncated><difficult>{int(index % 3 == 0)}</difficult>
    <bndbox><xmin>40</xmin><ymin>40</ymin><xmax>58</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>"""
        )
        if num_images > 3:
            (root / "ImageSets" / "Main" / "train.txt").write_text(
                "\n".join(stems[: num_images - 2]) + "\n"
            )
            (root / "ImageSets" / "Main" / "val.txt").write_text(
                "\n".join(stems[num_images - 2:]) + "\n"
            )


class TestVocTrainsDirectly:
    def test_convert_then_train_then_val(self, tmp_path):
        """The full B3 path via the CLI + the public API (CPU, tiny)."""
        root = tmp_path / "VOC2007"
        _write_voc_tree(root)
        train_json = tmp_path / "voc_train.json"
        val_json = tmp_path / "voc_val.json"

        runner = CliRunner()
        for split, out in (("train", train_json), ("val", val_json)):
            result = runner.invoke(
                app,
                ["convert-data", "from=voc", f"voc-root={root}", f"out={out}",
                 f"imagesets={split}"],
            )
            assert result.exit_code == 0, result.output

        # The parser preserves the canonical VOC class names (sorted ids).
        payload = json.loads(train_json.read_text())
        names = [category["name"] for category in payload["categories"]]
        assert names == sorted(names)
        assert set(names) <= set(VOC_CLASSES)
        # `difficult` objects ride as COCO ignore regions.
        assert any(annotation["iscrowd"] == 1 for annotation in payload["annotations"])

        data = {
            "format": "coco",
            "train": {"annotations": str(train_json), "images": str(root / "JPEGImages")},
            "val": {"annotations": str(val_json), "images": str(root / "JPEGImages")},
        }
        trained = autovision.train_object_detection(
            out=tmp_path / "run", data=data, model="rtdetr/resnet18",
            steps=2, batch_size=2, devices="cpu", num_workers=0,
            model_args=MODEL_ARGS,
            trainer_args={"val_every_num_steps": 2},
        )
        assert trained.steps_run == 2
        assert trained.classes  # VOC class names landed in the training run
        assert set(trained.classes.values()) <= set(VOC_CLASSES)

        model = autovision.load_model(trained.best_exported_model_path, device="cpu")
        metrics = model.val(data=data, split="val")
        assert "map" in metrics and isinstance(metrics["map"], float)

    def test_voc_classes_are_independent_of_registry(self, tmp_path):
        """A custom VOC class list (class_names=) is honored end to end."""
        from autovision.data.convert import voc_to_coco

        root = tmp_path / "VOC2007"
        _write_voc_tree(root)
        out = voc_to_coco(root, tmp_path / "pets.json", class_names=["cat", "dog"])
        payload = json.loads(out.read_text())
        assert [(c["id"], c["name"]) for c in payload["categories"]] == [(1, "cat"), (2, "dog")]
        assert {a["category_id"] for a in payload["annotations"]} <= {1, 2}
