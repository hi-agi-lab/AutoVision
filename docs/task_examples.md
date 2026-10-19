# 任务示例

仓库为 8 个任务族各提供两个脚本，位于 `examples/tasks/`：

- `<task>.py`：纯推理，加载现有权重 → 预测 → 标注，不训练；
- `<task>_train.py`：训练示例，默认生成小型合成数据（检测用 `coco8`）。

示例不在 `src/autovision/tasks/` 内：它们不随 wheel 发布，且训练示例会导入
CLI 编排层，放在任务包内会破坏仓库的单向分层约束。

推理示例：

- 默认图片 `examples/tasks/assets/airport_aerial.jpg`（CC0，San Diego 机场
  俯视图；来源与 sha256 见 `assets/README.md`）；
- 支持 `--weights`（训练导出的 `.pt`）、`--image`、`--threshold`；
- 输出 `prediction.jpg` + `detections.json`。

训练示例：

- 用很小的模型配置跑 1–3 步，CPU 即可运行（姿态/深度需要 GTR CUDA 栈）；
- 打印 checkpoint、导出模型、验证指标与类别表；
- 支持 `--predict-image`、`--export`、`--dry-run`。

## 安装与快速开始

```bash
pip install -e ".[transformers,timm,export]"

# 推理：官方权重，不训练
python examples/tasks/detection.py
python examples/tasks/oriented_detection.py
python examples/tasks/instance_segmentation.py
python examples/tasks/semantic_segmentation.py
python examples/tasks/pose_estimation.py
python examples/tasks/depth_estimation.py

# 全景/分类没有官方任务权重，需用训练导出的 .pt
python examples/tasks/panoptic_segmentation.py \
    --weights runs/examples/panoptic_segmentation_train/exported_models/exported_last.pt

# 训练示例
python examples/tasks/detection_train.py --steps 2
python examples/tasks/classification_train.py --steps 2
python examples/tasks/instance_segmentation_train.py --dry-run
```

`detection.py` 直接加载 `rfdetr/rf-detr-nano` 官方权重；GTR 推理脚本默认加载
`gtrobb_s_dota` / `gtrseg_s_coco` / `gtrsemseg_s_cityscapes` /
`gtrpose_s_coco` / `gtrdepth_s`。所有推理脚本默认使用 CC0 的
`assets/airport_aerial.jpg`（与 DOTA 旋转框权重的俯视域匹配）；需要街景或其他
任务图片时用 `--image` 指定。GTR 需要：

```bash
pip install -e ".[gtr]"   # CUDA + Triton
```

`detection.py --effect` 展示迁移自 supervision 的增强标注器：
`corner` / `round` 用角标框或圆角框替代普通框，`blur` / `pixelate` 在画框前
对检测区域做隐私模糊/马赛克；默认 `none` 保持原渲染：

```bash
python examples/tasks/detection.py --effect corner
python examples/tasks/detection.py --effect blur --threshold 0.5
```

## YOLO 基线示例（可选，AGPL）

`examples/tasks/yolo/` 另有 5 个 Ultralytics YOLO 推理示例：检测、分割、
姿态、旋转框、分类，复用同一批图片。`ultralytics` 是 **AGPL-3.0**，不属于
Apache-2.0 核心：

```bash
pip install -e ".[ultralytics]"
python examples/tasks/yolo/detection.py
python examples/tasks/yolo/segmentation.py
python examples/tasks/yolo/pose.py
python examples/tasks/yolo/obb.py            # 默认机场航拍图
python examples/tasks/yolo/classification.py
```

权重缓存到 `~/.cache/autovision/yolo`，不会写进仓库。检测 / 分割 / 姿态权重
从 ModelScope 的
[`Ultralytics/YOLO11`](https://www.modelscope.cn/models/Ultralytics/YOLO11/files)
下载；OBB / 分类不在该仓库，回退到 Ultralytics 默认源或使用本地
`--model /path/to/*.pt`。详见仓库内 `examples/tasks/yolo/README.md`。

## 任务与默认权重

| 任务 | 推理脚本 | 训练脚本 | 默认推理权重 |
|---|---|---|---|
| 检测 | `detection.py` | `detection_train.py` | `rfdetr/rf-detr-nano` |
| 旋转框 | `oriented_detection.py` | `oriented_detection_train.py` | `gtrobb_s_dota`（GTR, DOTA） |
| 实例分割 | `instance_segmentation.py` | `instance_segmentation_train.py` | `gtrseg_s_coco`（GTR, COCO） |
| 全景分割 | `panoptic_segmentation.py` | `panoptic_segmentation_train.py` | 需 `--weights` |
| 语义分割 | `semantic_segmentation.py` | `semantic_segmentation_train.py` | `gtrsemseg_s_cityscapes` |
| 姿态 | `pose_estimation.py` | `pose_estimation_train.py` | `gtrpose_s_coco`（GTR, COCO） |
| 深度 | `depth_estimation.py` | `depth_estimation_train.py` | `gtrdepth_s`（GTR） |
| 分类 | `classification.py` | `classification_train.py` | 需 `--weights` |

训练脚本的数据契约不变（检测 COCO/YOLO、旋转框 YOLO-OBB、实例 COCO
instances、全景 masks + segments、语义成对 images/masks、姿态 COCO
person_keypoints、深度 images + depths、分类 ImageFolder）。

真实数据配置示例：

```python
# 实例分割
data = {"train": {"annotations": "instances_train.json", "images": "train2017"},
        "val":   {"annotations": "instances_val.json",   "images": "val2017"}}

# 语义分割
data = {"train": {"images": "images/train", "masks": "masks/train"},
        "val":   {"images": "images/val", "masks": "masks/val"}}

# 姿态
data = {"train": {"annotations": "person_keypoints_train2017.json",
                  "images": "train2017"},
        "val":   {"annotations": "person_keypoints_val2017.json",
                  "images": "val2017"}}

# 深度
data = {"train": {"images": "images/train", "depths": "depths/train"},
        "val":   {"images": "images/val", "depths": "depths/val"}}
```

完整的数据 schema、公共参数和导出说明见仓库内
`examples/tasks/README.md`。

## 说明

- 训练示例默认只跑 1–3 步，指标没有参考意义，只用于确认管线能跑通；真实精度请用
  [基准](benchmarks.md)中的配方。
- `--export` 只复制 `.pt`；ONNX / OpenVINO / TensorRT 请用
  `autovision-cli export`（默认带数值对拍门禁）。
- 输出默认在 `runs/examples/<task>/`（推理）或
  `runs/examples/<task>_train/`（训练），演示数据在
  `runs/examples/_demo_data/<task>/`，`--overwrite` 不会删除演示数据。
