# Task examples

Each task family has a small runnable example under
[`examples/tasks/`](../../../examples/tasks/README.md). They are intentionally
outside this package: examples are not shipped in the wheel, and they import
the CLI/training orchestration layer (which the package layering contract
forbids inside `autovision.tasks`).

Inference-only demo (bundled street image, released RF-DETR weights):

```bash
python examples/tasks/detection.py
```

| Task | Inference | Training |
|---|---|---|
| Detection | `examples/tasks/detection.py` | `examples/tasks/detection_train.py` |
| Oriented detection | `examples/tasks/oriented_detection.py` | `examples/tasks/oriented_detection_train.py` |
| Instance segmentation | `examples/tasks/instance_segmentation.py` | `examples/tasks/instance_segmentation_train.py` |
| Panoptic segmentation | `examples/tasks/panoptic_segmentation.py` | `examples/tasks/panoptic_segmentation_train.py` |
| Semantic segmentation | `examples/tasks/semantic_segmentation.py` | `examples/tasks/semantic_segmentation_train.py` |
| Pose estimation | `examples/tasks/pose_estimation.py` | `examples/tasks/pose_estimation_train.py` |
| Depth estimation | `examples/tasks/depth_estimation.py` | `examples/tasks/depth_estimation_train.py` |
| Classification | `examples/tasks/classification.py` | `examples/tasks/classification_train.py` |

Inference scripts load released/official weights (RF-DETR detection, GTR
OBB/seg/semseg/pose/depth) or an exported `.pt` via `--weights`; panoptic and
classification require `--weights`. Training scripts support `--dry-run` to
print the data contract without training. See
[`examples/tasks/README.md`](../../../examples/tasks/README.md) for install
extras, real-data schemas and prediction/export commands.

Optional YOLO baselines (AGPL `ultralytics` extra) live in
`examples/tasks/yolo/`: detection, segmentation, pose, OBB and classification
inference on the bundled images.

Task implementation map:

- `detection/` — RT-DETR, DETR/GTR, D-FINE, PicoDet, RF-DETR
- `oriented_detection/` — RT-DETR-OBB, GTR-OBB
- `instance_segmentation/` — EOMT (DINOv2/DINOv3), GTR
- `panoptic_segmentation/` — EOMT (DINOv2/DINOv3)
- `semantic_segmentation/` — EOMT, GTR, linear head
- `pose_estimation/` — GTR pose
- `depth_estimation/` — GTR depth
- `classification/` — torchvision/timm backbones
