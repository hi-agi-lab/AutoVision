# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Every task example must stay importable and print its data contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "tasks"

TASKS = [
    pytest.param("detection_train", id="detection-train"),
    pytest.param("oriented_detection_train", id="oriented-detection-train"),
    pytest.param("instance_segmentation_train", id="instance-segmentation-train"),
    pytest.param("panoptic_segmentation_train", id="panoptic-segmentation-train"),
    pytest.param("semantic_segmentation_train", id="semantic-segmentation-train"),
    pytest.param("pose_estimation_train", id="pose-estimation-train"),
    pytest.param("depth_estimation_train", id="depth-estimation-train"),
    pytest.param("classification_train", id="classification-train"),
]

INFERENCE = [
    pytest.param("detection", id="detection"),
    pytest.param("oriented_detection", id="oriented-detection"),
    pytest.param("instance_segmentation", id="instance-segmentation"),
    pytest.param("panoptic_segmentation", id="panoptic-segmentation"),
    pytest.param("semantic_segmentation", id="semantic-segmentation"),
    pytest.param("pose_estimation", id="pose-estimation"),
    pytest.param("depth_estimation", id="depth-estimation"),
    pytest.param("classification", id="classification"),
    pytest.param("inference_demo", id="inference-demo-alias"),
]

YOLO_EXAMPLES = [
    pytest.param("detection", id="yolo-detection"),
    pytest.param("segmentation", id="yolo-segmentation"),
    pytest.param("pose", id="yolo-pose"),
    pytest.param("obb", id="yolo-obb"),
    pytest.param("classification", id="yolo-classification"),
]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / f"{script}.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _run_yolo(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / "yolo" / f"{script}.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("task", INFERENCE)
def test_inference_help(task: str) -> None:
    """Every inference script exposes image/weights/threshold flags."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert "--weights" in result.stdout
    assert "--threshold" in result.stdout


@pytest.mark.parametrize("script", YOLO_EXAMPLES)
def test_yolo_example_help(script: str) -> None:
    """YOLO examples parse --help without importing the AGPL package."""
    result = _run_yolo(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--image" in result.stdout
    assert "--threshold" in result.stdout


def test_detection_inference_is_not_training() -> None:
    """detection.py is inference-only; training lives in detection_train.py."""
    result = _run("detection", "--help")
    assert result.returncode == 0, result.stderr
    assert "--steps" not in result.stdout


def test_detection_offers_supervision_effects() -> None:
    """detection.py exposes the rendering effects migrated from supervision."""
    result = _run("detection", "--help")
    assert result.returncode == 0, result.stderr
    assert "--effect" in result.stdout
    for effect in ("corner", "round", "blur", "pixelate"):
        assert effect in result.stdout


def test_bundled_demo_asset_is_readable() -> None:
    """The bundled CC0 demo image exists with the expected dimensions."""
    from PIL import Image

    expected = {
        "airport_aerial.jpg": (1280, 606),
    }
    for filename, size in expected.items():
        path = EXAMPLES / "assets" / filename
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == size


@pytest.mark.parametrize("task", TASKS)
def test_example_help(task: str) -> None:
    """--help parses without importing a task-specific optional stack."""
    result = _run(task, "--help")
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("task", TASKS)
def test_example_dry_run_writes_data_contract(task: str, tmp_path: Path) -> None:
    """--dry-run generates demo data and prints the config without training."""
    result = _run(task, "--dry-run", "--out", str(tmp_path / "run"))
    assert result.returncode == 0, result.stderr
    assert "demo data config" in result.stdout
    if task != "detection_train":  # coco8 comes from the shared cache, not --out
        demo_root = tmp_path / "_demo_data"
        assert demo_root.is_dir() and any(demo_root.iterdir())
