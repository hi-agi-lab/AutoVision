# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset snapshots (roadmap 数据版本化): content addressing + drift report."""

import json
import shutil
from pathlib import Path

import pytest
from tests.conftest import create_image
from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.data.versioning import (
    dataset_snapshot,
    load_snapshot,
    verify_dataset_snapshot,
)
from autovision.errors import AutoVisionError


def _tree(root: Path) -> dict:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        create_image(images / f"img_{index}.jpg", size=(32, 32), seed=index)
    annotations = root / "train.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img_0.jpg", "width": 32, "height": 32}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1,
                     "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}
                ],
                "categories": [{"id": 1, "name": "cat"}],
            }
        )
    )
    return {
        "format": "coco",
        "train": {"annotations": str(annotations), "images": str(images)},
    }


class TestSnapshot:
    def test_snapshot_records_counts_and_digests(self, tmp_path):
        spec = _tree(tmp_path / "data")
        payload = dataset_snapshot(spec, tmp_path / "snap.json")
        assert payload["kind"] == "autovision-dataset-snapshot"
        assert payload["deep"] is False
        split = payload["splits"]["train"]
        assert split["files"] == 4  # 3 images + the annotation file
        assert split["bytes"] > 0
        assert len(split["digest"]) == 64
        assert "content_digest" not in split
        assert len(split["manifest"]) == 4
        assert (tmp_path / "snap.json").exists()

    def test_deep_snapshot_hashes_contents(self, tmp_path):
        spec = _tree(tmp_path / "data")
        payload = dataset_snapshot(spec, tmp_path / "deep.json", deep=True)
        assert payload["deep"] is True
        assert len(payload["splits"]["train"]["content_digest"]) == 64

    def test_partial_missing_split_fails_strictly(self, tmp_path):
        root = tmp_path / "data"
        spec = _tree(root)
        spec["val"] = {"images": str(root / "missing_val")}
        with pytest.raises(AutoVisionError, match="missing path"):
            dataset_snapshot(spec, tmp_path / "snap.json")
        payload = dataset_snapshot(spec, tmp_path / "snap.json", strict=False)
        assert "val" not in payload["splits"]
        assert "train" in payload["splits"]


class TestVerify:
    def test_unchanged_tree_verifies_clean(self, tmp_path):
        spec = _tree(tmp_path / "data")
        dataset_snapshot(spec, tmp_path / "snap.json")
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is True
        assert report["differences"] == []
        assert report["checked"] == ["train"]

    def test_added_removed_resized_files_are_reported(self, tmp_path):
        root = tmp_path / "data"
        spec = _tree(root)
        dataset_snapshot(spec, tmp_path / "snap.json")

        create_image(root / "images" / "new.jpg", size=(32, 32), seed=9)
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False
        manifest_diff = next(
            difference for difference in report["differences"] if "added" in difference
        )
        assert any("new.jpg" in name for name in manifest_diff["added"])

        (root / "images" / "new.jpg").unlink()
        (root / "images" / "img_0.jpg").write_bytes(b"tiny")  # same count, new size
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False
        fields = {difference["field"] for difference in report["differences"]}
        assert {"bytes", "digest", "manifest"} <= fields
        detail = next(
            difference for difference in report["differences"]
            if difference["field"] == "manifest"
        )
        assert any("img_0.jpg" in name for name in detail["resized"])

    def test_same_size_content_edit_needs_a_deep_snapshot(self, tmp_path):
        root = tmp_path / "data"
        spec = _tree(root)
        target = root / "images" / "img_0.jpg"
        original = target.read_bytes()
        dataset_snapshot(spec, tmp_path / "deep.json", deep=True)
        dataset_snapshot(spec, tmp_path / "shallow.json")

        target.write_bytes(bytes(reversed(original)))  # same length, new bytes
        assert target.stat().st_size == len(original)

        assert verify_dataset_snapshot(tmp_path / "shallow.json")["ok"] is True
        deep_report = verify_dataset_snapshot(tmp_path / "deep.json")
        assert deep_report["ok"] is False
        assert deep_report["differences"][0]["field"] == "content_digest"

    def test_moved_tree_verifies_with_its_own_spec(self, tmp_path):
        """Digests are root-relative: an identical copy verifies clean."""
        spec = _tree(tmp_path / "origin")
        dataset_snapshot(spec, tmp_path / "snap.json", deep=True)

        copy_root = tmp_path / "copy"
        shutil.copytree(tmp_path / "origin", copy_root)
        copy_spec = {
            "format": "coco",
            "train": {
                "annotations": str(copy_root / "train.json"),
                "images": str(copy_root / "images"),
            },
        }
        report = verify_dataset_snapshot(tmp_path / "snap.json", spec_override=copy_spec)
        assert report["ok"] is True, report["differences"]

    def test_relative_spec_paths_resolve_against_the_spec_directory(self, tmp_path, monkeypatch):
        """Regression: a spec file's relative paths must not follow the CWD.

        Training resolves spec-relative paths against the spec file's
        directory; the snapshot has to pin the SAME tree (a decoy directory
        in the CWD must stay invisible).
        """
        root = tmp_path / "data"
        _tree(root)
        spec_path = tmp_path / "data.yaml"
        import yaml

        spec_path.write_text(
            yaml.safe_dump(
                {
                    "format": "coco",
                    "train": {
                        "annotations": "data/train.json",
                        "images": "data/images",
                    },
                }
            )
        )
        decoy = tmp_path / "elsewhere"
        (decoy / "images").mkdir(parents=True)
        create_image(decoy / "images" / "img_0.jpg", size=(32, 32), seed=11)
        (decoy / "train.json").write_text("{}")

        monkeypatch.chdir(decoy)  # the decoy tree must NOT be captured
        payload = dataset_snapshot(spec_path, tmp_path / "snap.json", deep=True)
        assert payload["spec_base_dir"] == str(tmp_path)
        assert payload["splits"]["train"]["files"] == 4  # decoy has 2 files only

        monkeypatch.chdir(tmp_path / "..")  # verify from an unrelated CWD
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is True, report["differences"]

    def test_overlapping_roots_with_distinct_labels_stay_content_exact(self, tmp_path):
        """Nested roots (distinct labels) keep per-file content hashing."""
        root = tmp_path / "data"
        _tree(root)
        overlapping = {
            "format": "coco",
            "train": {"outer": str(root), "inner": str(root / "images")},
        }
        dataset_snapshot(overlapping, tmp_path / "snap.json", deep=True)
        target = root / "images" / "img_0.jpg"
        original = target.read_bytes()
        target.write_bytes(bytes(reversed(original)))  # same size, new bytes
        report = verify_dataset_snapshot(tmp_path / "snap.json")
        assert report["ok"] is False
        assert any(difference["field"] == "content_digest" for difference in report["differences"])

    def test_deep_verification_needs_a_deep_snapshot(self, tmp_path):
        spec = _tree(tmp_path / "data")
        dataset_snapshot(spec, tmp_path / "shallow.json")
        with pytest.raises(AutoVisionError, match="deep"):
            verify_dataset_snapshot(tmp_path / "shallow.json", deep=True)

    def test_duplicate_root_labels_fail_loud(self, tmp_path):
        root = tmp_path / "data"
        _tree(root)
        ambiguous = {
            "format": "coco",
            "train": {
                "part_a": {"images": str(root / "images")},
                "part_b": {"images": str(root / "images")},
            },
        }
        with pytest.raises(AutoVisionError, match="duplicate root label"):
            dataset_snapshot(ambiguous, tmp_path / "snap.json")

    def test_snapshot_never_overwrites_silently(self, tmp_path):
        spec = _tree(tmp_path / "data")
        dataset_snapshot(spec, tmp_path / "snap.json")
        with pytest.raises(AutoVisionError, match="overwrite=true"):
            dataset_snapshot(spec, tmp_path / "snap.json")
        dataset_snapshot(spec, tmp_path / "snap.json", overwrite=True)  # deliberate

    def test_boolean_flags_coerce_strictly(self, tmp_path):
        spec = _tree(tmp_path / "data")
        shallow = dataset_snapshot(spec, tmp_path / "a.json", deep="False")
        assert shallow["deep"] is False
        deep = dataset_snapshot(spec, tmp_path / "b.json", deep="yes")
        assert deep["deep"] is True
        with pytest.raises(AutoVisionError, match="boolean"):
            dataset_snapshot(spec, tmp_path / "c.json", deep="banana")

    def test_errors_are_loud(self, tmp_path):
        with pytest.raises(AutoVisionError, match="Snapshot not found"):
            verify_dataset_snapshot(tmp_path / "missing.json")
        bad = tmp_path / "bad.json"
        bad.write_text('{"kind": "something-else"}')
        with pytest.raises(AutoVisionError, match="kind"):
            load_snapshot(bad)
        with pytest.raises(AutoVisionError, match="no existing paths"):
            dataset_snapshot({"format": "coco", "train": {"images": "/nope/missing"}},
                             tmp_path / "x.json")
        with pytest.raises(AutoVisionError, match="Unsupported data spec type"):
            dataset_snapshot(123, tmp_path / "y.json")
        with pytest.raises(AutoVisionError, match="mapping or a path"):
            dataset_snapshot("/definitely/not/a/spec.yaml", tmp_path / "y.json")


class TestSnapshotCli:
    def test_snapshot_and_verify_roundtrip(self, tmp_path):
        spec = _tree(tmp_path / "data")
        spec_path = tmp_path / "data.yaml"
        import yaml

        spec_path.write_text(yaml.safe_dump(spec))
        runner = CliRunner()
        written = runner.invoke(
            app,
            ["data-snapshot", f"data={spec_path}", f"out={tmp_path / 'snap.json'}",
             "deep=true"],
        )
        assert written.exit_code == 0, written.output
        assert "deep=True" in written.output

        verified = runner.invoke(app, ["data-snapshot", f"snapshot={tmp_path / 'snap.json'}"])
        assert verified.exit_code == 0, verified.output
        assert json.loads(verified.output)["ok"] is True

        create_image(tmp_path / "data" / "images" / "drift.jpg", size=(32, 32), seed=5)
        drifted = runner.invoke(app, ["data-snapshot", f"snapshot={tmp_path / 'snap.json'}"])
        assert drifted.exit_code != 0
        assert json.loads(drifted.output)["ok"] is False

    def test_missing_arguments_fail_loud(self):
        runner = CliRunner()
        assert runner.invoke(app, ["data-snapshot"]).exit_code != 0
        assert runner.invoke(app, ["data-snapshot", "data=/tmp/x"]).exit_code != 0
