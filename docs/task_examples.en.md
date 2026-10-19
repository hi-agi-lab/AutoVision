# Task examples

The repository ships two runnable scripts per task family under
`examples/tasks/`:

- `<task>.py`: inference only — load weights → predict → annotate, no training;
- `<task>_train.py`: training example, which generates a small synthetic dataset
  by default (detection uses `coco8`).

The examples live outside `src/autovision/tasks/`: they are not shipped in the
wheel, and training examples import the CLI orchestration layer, which would
break the repository's one-way layering contract.

Inference examples:

- default image: `examples/tasks/assets/airport_aerial.jpg` (CC0, San Diego
  airport aerial view; source and sha256 in `assets/README.md`);
- support `--weights` (an exported `.pt`), `--image` and `--threshold`;
- write `prediction.jpg` + `detections.json`.

Training examples:

- run 1–3 steps with a tiny model configuration; CPU is enough for most tasks
  (pose/depth require the GTR CUDA stack);
- print the checkpoint, exported model, validation metrics and class map;
- support `--predict-image`, `--export` and `--dry-run`.

## Install and quick start

```bash
pip install -e ".[transformers,timm,export]"

# Inference with official weights, no training
python examples/tasks/detection.py
python examples/tasks/oriented_detection.py
python examples/tasks/instance_segmentation.py
python examples/tasks/semantic_segmentation.py
python examples/tasks/pose_estimation.py
python examples/tasks/depth_estimation.py

# Panoptic/classification have no official task weights; pass an exported .pt
python examples/tasks/panoptic_segmentation.py \
    --weights runs/examples/panoptic_segmentation_train/exported_models/exported_last.pt

# Training examples
python examples/tasks/detection_train.py --steps 2
python examples/tasks/classification_train.py --steps 2
python examples/tasks/instance_segmentation_train.py --dry-run
```

`detection.py` loads the official `rfdetr/rf-detr-nano` weights directly; GTR
inference scripts load `gtrobb_s_dota` / `gtrseg_s_coco` /
`gtrsemseg_s_cityscapes` / `gtrpose_s_coco` / `gtrdepth_s` by default. All
inference scripts default to the CC0 `assets/airport_aerial.jpg` (its overhead
domain matches the DOTA-trained OBB checkpoint); pass `--image` for street-level
or task-specific imagery. GTR requires:

```bash
pip install -e ".[gtr]"   # CUDA + Triton
```

`detection.py --effect` demonstrates the enhanced annotators migrated from
supervision: `corner` / `round` replace plain boxes with corner brackets or
rounded boxes, while `blur` / `pixelate` apply privacy filtering before drawing
boxes and labels. The default `none` keeps the original rendering:

```bash
python examples/tasks/detection.py --effect corner
python examples/tasks/detection.py --effect blur --threshold 0.5
```

## YOLO baseline examples (optional, AGPL)

`examples/tasks/yolo/` adds five Ultralytics YOLO inference examples:
detection, segmentation, pose, OBB and classification, reusing the same images.
`ultralytics` is **AGPL-3.0** and is not part of the Apache-2.0 core:

```bash
pip install -e ".[ultralytics]"
python examples/tasks/yolo/detection.py
python examples/tasks/yolo/segmentation.py
python examples/tasks/yolo/pose.py
python examples/tasks/yolo/obb.py            # airport aerial image
python examples/tasks/yolo/classification.py
```

Weights are cached under `~/.cache/autovision/yolo`, never in the repository.
Detection / segmentation / pose weights are downloaded from the ModelScope
[`Ultralytics/YOLO11`](https://www.modelscope.cn/models/Ultralytics/YOLO11/files)
mirror; OBB / classification are not in that repository, so they fall back to
the Ultralytics default source or a local `--model /path/to/*.pt`. See
`examples/tasks/yolo/README.md` for details.

## Tasks and default weights

| Task | Inference | Training | Default inference weights |
|---|---|---|---|
| Detection | `detection.py` | `detection_train.py` | `rfdetr/rf-detr-nano` |
| Oriented detection | `oriented_detection.py` | `oriented_detection_train.py` | `gtrobb_s_dota` (GTR, DOTA) |
| Instance segmentation | `instance_segmentation.py` | `instance_segmentation_train.py` | `gtrseg_s_coco` (GTR, COCO) |
| Panoptic segmentation | `panoptic_segmentation.py` | `panoptic_segmentation_train.py` | requires `--weights` |
| Semantic segmentation | `semantic_segmentation.py` | `semantic_segmentation_train.py` | `gtrsemseg_s_cityscapes` |
| Pose | `pose_estimation.py` | `pose_estimation_train.py` | `gtrpose_s_coco` (GTR, COCO) |
| Depth | `depth_estimation.py` | `depth_estimation_train.py` | `gtrdepth_s` (GTR) |
| Classification | `classification.py` | `classification_train.py` | requires `--weights` |

The training data contracts are unchanged: detection COCO/YOLO, oriented
detection YOLO-OBB, instance COCO instances, panoptic masks + segments, semantic
paired images/masks, pose COCO person_keypoints, depth images + depths, and
classification ImageFolder.

Real-data configuration examples:

```python
# Instance segmentation
data = {"train": {"annotations": "instances_train.json", "images": "train2017"},
        "val":   {"annotations": "instances_val.json",   "images": "val2017"}}

# Semantic segmentation
data = {"train": {"images": "images/train", "masks": "masks/train"},
        "val":   {"images": "images/val", "masks": "masks/val"}}

# Pose
data = {"train": {"annotations": "person_keypoints_train2017.json",
                  "images": "train2017"},
        "val":   {"annotations": "person_keypoints_val2017.json",
                  "images": "val2017"}}

# Depth
data = {"train": {"images": "images/train", "depths": "depths/train"},
        "val":   {"images": "images/val", "depths": "depths/val"}}
```

See `examples/tasks/README.md` for the full data schemas, common flags and
export notes.

## Notes

- Training examples run only 1–3 steps by default; their metrics are meaningless
  and only prove the pipeline runs. Use the recipes in [benchmarks](benchmarks.md)
  for real numbers.
- `--export` only copies `.pt`; use `autovision-cli export` for ONNX / OpenVINO /
  TensorRT (numerical parity gates are enabled by default).
- Outputs default to `runs/examples/<task>/` (inference) or
  `runs/examples/<task>_train/` (training); demo data goes to
  `runs/examples/_demo_data/<task>/` and is not removed by `--overwrite`.
