# Task examples

One small, runnable example per task family. They live outside `src/` on
purpose: example code is **not** part of the installed wheel, and training
examples import the CLI layer, which the package layering contract forbids
inside `autovision.tasks`.

Every example:

- writes a tiny synthetic dataset when `--data` is omitted (except detection,
  which uses the bundled `coco8` smoke set);
- trains a small model for 1-3 steps so it runs on CPU (pose/depth need the
  GTR CUDA stack);
- prints the checkpoint, exported model, validation metrics and class map;
- can render one prediction with `--predict-image` and copy the exported
  `.pt` with `--export`;
- supports `--dry-run`, which only writes/prints the demo data config.

## Install

```bash
pip install -e ".[transformers,timm,export]"   # EOMT / detection examples
# pose + depth only:
pip install -e ".[gtr]"                        # CUDA + Triton required
```

## Quick start

```bash
# Inference-only: released/official weights, no training
python examples/tasks/detection.py
python examples/tasks/oriented_detection.py
python examples/tasks/instance_segmentation.py
python examples/tasks/semantic_segmentation.py
python examples/tasks/pose_estimation.py
python examples/tasks/depth_estimation.py
# Panoptic/classification need an exported checkpoint:
python examples/tasks/panoptic_segmentation.py \
    --weights runs/examples/panoptic_segmentation_train/exported_models/exported_last.pt

# Training examples (tiny synthetic data / coco8)
python examples/tasks/detection_train.py --steps 2
python examples/tasks/classification_train.py --steps 2
python examples/tasks/instance_segmentation_train.py --dry-run

# After training, render one image and copy the exported model
python examples/tasks/semantic_segmentation_train.py --steps 2 \
    --predict-image path/to/image.png --export runs/semantic.pt
```

`inference_demo.py` is a compatibility alias for `detection.py`. All inference
examples default to the CC0 `assets/airport_aerial.jpg` (see `assets/README.md`
for provenance); pass `--image` for task-specific imagery. The OBB example uses
the same overhead image because its checkpoint is DOTA-trained.

Outputs default to `runs/examples/<task>/` (inference) or
`runs/examples/<task>_train/` (training); generated demo data goes to
`runs/examples/_demo_data/<task>/` (outside `--out`, so `--overwrite` does not
delete it).

## YOLO baselines (optional, AGPL)

The `yolo/` subdirectory adds five Ultralytics YOLO inference examples
(detection / segmentation / pose / OBB / classification) using the same
images. They require the **AGPL** extra:

```bash
pip install -e ".[ultralytics]"
python examples/tasks/yolo/detection.py
python examples/tasks/yolo/obb.py        # airport_aerial.jpg
```

See [`yolo/README.md`](yolo/README.md) for the full table and flags. Detection,
segmentation and pose weights are auto-downloaded from the ModelScope
`Ultralytics/YOLO11` mirror; OBB/classification fall back to Ultralytics or a
local `--model` path. These examples are separate from AutoVision's
Apache-2.0 core.

## Inference flags (all `<task>.py`)

| Flag | Meaning |
|---|---|
| `--image PATH` | input image (default: bundled `assets/airport_aerial.jpg`) |
| `--weights PATH` | exported `.pt` from `<task>_train.py` |
| `--class-names PATH` | optional JSON `{class_id: name}` for exported checkpoints |
| `--threshold F` | score threshold (default: 0.3) |
| `--device auto\|cpu\|cuda` | inference device |
| `--out DIR` | output directory (default: `runs/examples/<task>`) |

Detection additionally accepts `--model KEY` (default `rfdetr/rf-detr-nano`)
and the GTR tasks accept `--checkpoint NAME` / `--variant s|m|l|x`
(official MIT assets). Panoptic/classification have no released task
checkpoint and require `--weights`.

## Common flags (training examples)

| Flag | Meaning |
|---|---|
| `--data PATH` | real data YAML/JSON config (or directory, where supported) |
| `--out DIR` | run directory (default `runs/examples/<task>_train`) |
| `--steps N` | training steps (default 2) |
| `--batch-size N` | batch size (default 2) |
| `--seed N` | training seed |
| `--num-classes N` | class count for classification / instance / semantic |
| `--overwrite` | replace an existing `--out` |
| `--dry-run` | write/print the demo data config; no training |
| `--predict-image PATH` | run the exported model and save `prediction.png` |
| `--export PATH` | copy the exported `.pt` (ONNX/TRT via `autovision-cli export`) |
| `--threshold F` | prediction threshold (default 0.3) |

## Task matrix

| Task | Inference | Training | Default inference weights |
|---|---|---|---|
| Detection | `detection.py` | `detection_train.py` | `rfdetr/rf-detr-nano` |
| Oriented detection | `oriented_detection.py` | `oriented_detection_train.py` | `gtrobb_s_dota` (GTR, DOTA) |
| Instance segmentation | `instance_segmentation.py` | `instance_segmentation_train.py` | `gtrseg_s_coco` (GTR, COCO) |
| Panoptic segmentation | `panoptic_segmentation.py` | `panoptic_segmentation_train.py` | requires `--weights` |
| Semantic segmentation | `semantic_segmentation.py` | `semantic_segmentation_train.py` | `gtrsemseg_s_cityscapes` (GTR) |
| Pose estimation | `pose_estimation.py` | `pose_estimation_train.py` | `gtrpose_s_coco` (GTR, COCO) |
| Depth estimation | `depth_estimation.py` | `depth_estimation_train.py` | `gtrdepth_s` (GTR) |
| Classification | `classification.py` | `classification_train.py` | requires `--weights` |

Training defaults: `detection_train.py` uses coco8; the other `_train.py`
scripts generate tiny synthetic data (see `--dry-run`). GTR inference/training
needs `pip install -e ".[gtr]"` and CUDA; panoptic/classification inference
uses a checkpoint exported by the matching training script.

## Real data contracts

```python
# detection: COCO / YOLO data.yaml
data = "data.yaml"

# oriented detection: YOLO-OBB
data = {"format": "yolo-obb", "path": "obb_root", "train": "images/train",
        "val": "images/val", "names": {0: "plane", 1: "ship"}}

# instance segmentation: COCO instances (segmentation polygons/RLE)
data = {"train": {"annotations": "instances_train.json", "images": "train2017"},
        "val":   {"annotations": "instances_val.json",   "images": "val2017"}}

# panoptic: masks PNG (segment ids, 0 = unlabeled) + segments JSON
data = {"train": {"images": "train/images", "masks": "train/masks",
                  "segments": "train/panoptic.json"},
        "val":   {"images": "val/images", "masks": "val/masks",
                  "segments": "val/panoptic.json"}}

# semantic: paired PNGs; 255 is ignored by loss and mIoU
data = {"train": {"images": "images/train", "masks": "masks/train"},
        "val":   {"images": "images/val", "masks": "masks/val"}}

# pose: COCO person_keypoints
data = {"train": {"annotations": "person_keypoints_train2017.json",
                  "images": "train2017"},
        "val":   {"annotations": "person_keypoints_val2017.json",
                  "images": "val2017"}}

# depth: .npy (float metres) or 16-bit PNG
data = {"train": {"images": "images/train", "depths": "depths/train"},
        "val":   {"images": "images/val", "depths": "depths/val"}}

# classification: ImageFolder layout
data = {"train": "data/train", "val": "data/val"}
```

## Notes

- Demo metrics after 1-3 steps are meaningless by design; they only prove the
  pipeline runs. Use real data and the recipes in `docs/benchmarks.md` for
  numbers.
- Pose/depth require the GTR CUDA stack. Without it, use `--dry-run` to inspect
  the data contract, or run the other six examples on CPU.
- `--export` only copies `.pt`; use `autovision-cli export` for ONNX,
  OpenVINO and TensorRT parity gates.
- `assets/airport_aerial.jpg` is CC0 (see `assets/README.md`); it ships in the
  sdist so the examples run from a source distribution.
