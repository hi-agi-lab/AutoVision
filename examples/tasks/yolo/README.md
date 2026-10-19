# YOLO task examples

Five inference-only examples for the Ultralytics YOLO family. They use the
bundled demo images and reuse AutoVision's `InferenceResult` + task-aware
annotators for the output image.

> **License boundary**: `ultralytics` is **AGPL-3.0** and is an optional extra,
> separate from AutoVision's Apache-2.0 core. The YOLO11 checkpoints are also
> distributed by Ultralytics; review their terms before redistribution.

## Install

```bash
pip install -e ".[ultralytics]"
```

Without the extra, `--help` still works but inference exits with an install
message. Detection / segmentation / pose weights are downloaded from the
ModelScope mirror [`Ultralytics/YOLO11`](https://www.modelscope.cn/models/Ultralytics/YOLO11/files)
into `$AUTOVISION_CACHE_DIR/yolo` (default `~/.cache/autovision/yolo`), not
into the repository. OBB and classification are not in that ModelScope
repository, so they fall back to the Ultralytics source (or pass a local
`--model` path).

## Scripts

| Script | Default weights | Default image | Output |
|---|---|---|---|
| `detection.py` | `yolo11n.pt` | `../assets/airport_aerial.jpg` | boxes + labels |
| `segmentation.py` | `yolo11n-seg.pt` | `../assets/airport_aerial.jpg` | boxes + masks |
| `pose.py` | `yolo11n-pose.pt` | `../assets/airport_aerial.jpg` | boxes + COCO keypoints |
| `obb.py` | `yolo11n-obb.pt` | `../assets/airport_aerial.jpg` | rotated boxes |
| `classification.py` | `yolo11n-cls.pt` | `../assets/airport_aerial.jpg` | top-1 label |

## Run

```bash
python examples/tasks/yolo/detection.py
python examples/tasks/yolo/segmentation.py
python examples/tasks/yolo/pose.py
python examples/tasks/yolo/obb.py
python examples/tasks/yolo/classification.py
```

Common flags:

```bash
--image PATH          # override the bundled image
--model NAME|PATH     # yolo11n / yolo11s / yolo11m / ... or a local .pt
--threshold F         # confidence threshold (default 0.25)
--device auto|cpu|cuda
--out DIR             # default runs/examples/yolo_<task>
```

Each run writes `prediction.jpg` and `detections.json` (masks as COCO RLE via
AutoVision's `InferenceResult.to_json`). The examples do not train; for
AutoVision training use the `*_train.py` scripts in the parent directory.

## Notes

- `yolo11n-obb.pt` is trained on DOTA aerial imagery; the bundled default is
  an overhead CC0 image, suitable for all five examples.
- First run downloads the checkpoint (about 5–6 MB) from ModelScope for
  detection/segmentation/pose; OBB/classification use the Ultralytics default
  source. If your network is restricted, pass `--model /path/to/yolo11n.pt`.
- Offline smoke check without downloading weights: pass an Ultralytics YAML
  (for example `--model yolo11n-seg.yaml`). It builds a random model; use
  `--threshold 0` if you only want to verify the conversion/render path.
  Random OBB weights can emit enormous rotated polygons (scores ~1e-4); that
  is expected and not a rendering bug — use `yolo11n-obb.pt` for real output.
- These examples are intentionally outside the AutoVision wheel: they depend
  on an AGPL package and are not part of the Apache-2.0 core.
