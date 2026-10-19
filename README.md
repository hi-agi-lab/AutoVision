# AutoVision

**简体中文** | [English](README.en.md)

企业级视觉模型训练框架：自监督预训练 / 蒸馏 → 微调 → 评测 → 导出与部署。
Python API 与 `autovision-cli`（30 条命令）共享同一套 Pydantic 严格校验，覆盖检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类。

当前版本 **0.3.0（Beta）**。适合需要自托管训练、评测、导出和私有部署的团队；不提供托管 API，也不替代生产网关。核心代码 Apache-2.0，受限许可权重默认不下载。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

- **许可边界清晰**：核心代码 Apache-2.0；DINOv2 / DINOv3 等受限权重默认不下载，AGPL 组件仅通过 extras 隔离；[模型库](docs/model_zoo.md)逐资产给出许可与 sha256。
- **可验证一致性**：导出默认数值对拍门禁；已接入的官方权重加载 / 前向逐位对拍；金标准 loss 轨迹快照。
- **协议化基准**：[基准表](docs/benchmarks.md)只引官方发表值；自家数字必附复现脚本；只增不改并附勘误节。

文档：[任务示例](docs/task_examples.md) · [模型库](docs/model_zoo.md) · [权重下载](docs/download_weights.md) · [基准](docs/benchmarks.md) ·
[架构总览](docs/framework_design.md) · [导出与部署](docs/export.md) ·
[插件协议](docs/plugins.md) · [私有注册表](docs/registry.md) ·
[合规报告](docs/compliance.md) · [硬件伙伴包](docs/hardware_packs.md) ·
[用户自持权重](docs/user_held_weights.md) · [LTS 政策](docs/lts_policy.md) ·
[支持政策](docs/support_policy.md) · [已知限制](docs/known_limits.md) ·
[贡献](CONTRIBUTING.md)

**目录**：[安装](#安装) · [快速开始](#快速开始) · [任务入口](#任务入口) ·
[能力与子系统](#能力与子系统) · [模型与指标](#模型与指标) ·
[导出与部署](#导出与部署) · [推理运行时与服务](#推理运行时与服务) ·
[进阶功能](#进阶功能) · [致谢](#致谢acknowledgements) · [开发](#开发) ·
[已知限制](#已知限制) · [许可与溯源](#许可与溯源)

## 安装

Python ≥ 3.10。源码安装是当前可用路径；PyPI 发布后可用
`pip install autovision-framework`。发行名计划使用 `autovision-framework`
（import 名仍是 `autovision`）。

从源码安装（在仓库根目录执行）：

```bash
# 常用：HF 骨干 + timm + ONNX 导出
pip install -e ".[transformers,timm,export]"

# 预训练 RF-DETR 直接推理
pip install -e ".[rfdetr,export]"

# 开发：pytest / ruff（CI 同款）
pip install -e ".[dev]"
```

```bash
# PyPI 首发后可用（当前包尚未上传）
pip install "autovision-framework[transformers,timm,export]"
```

> PyPI 上另有一个无关的 `autovision` 0.0.1 包；`pip install autovision` 不会安装本项目。

<details>
<summary>extras 一览（模型包 / 运行时 / 记录器 / 开发）</summary>

| extras | 能力 |
|---|---|
| `transformers` | `dinov2/*`、`dinov3/*` 骨干（HuggingFace 权重） |
| `timm` | timm 注册表全量模型（任意输入通道数） |
| `rfdetr` | `rfdetr/rf-detr-{base,nano,small,medium,large}`（需较新 transformers） |
| `super-gradients` | CustomizableDetector 检测家族（分割 wrapper 未接入） |
| `ultralytics`（AGPL） | YOLOv5/6/8/11/12/26、RT-DETR（隔离加载） |
| `gtr` | GTR GLA 骨干（`detr/gtrvit-*`）；Triton 前向需 CUDA |
| `download` | GTR 上游权重按需下载（`huggingface_hub`） |
| `export` | ONNX 导出管线（onnx / onnxruntime / onnxslim） |
| `openvino` | OpenVINO IR 导出 + 运行时后端（CPU 优先，可切 GPU/AUTO） |
| `video` | 参考推理运行时的视频 / 流 I/O（OpenCV，文件 + RTSP/HTTP） |
| `serving` | 参考 serving（FastAPI + uvicorn；`/predict` 批处理 + `/metrics`） |
| `lsap` | 检测匹配器批量线性分配求解（CUDA；自动回退 scipy） |
| `tensorboard` / `mlflow` / `wandb` | 训练记录器，缺失时惰性降级 |
| `bitsandbytes` | 8-bit AdamW |
| `dev` | pytest / ruff（CI 同款） |
| `docs` | MkDocs Material（`mkdocs build --strict`） |

</details>

注：`super-gradients` 与 `export` / `transformers` / `rfdetr` / `video` 互斥
（上游钉死 onnxruntime / protobuf / numpy 版本；同时安装会由 pip/uv 报错）。

可选系统依赖：`ffmpeg`（`extract-video-frames`）、`matplotlib`（`plot-curves`）、
CUDA / Triton（GTR 前向）、TensorRT（TensorRT 引擎导出）。

预训练权重从 HuggingFace 下载；受限网络设镜像 `HF_ENDPOINT=https://hf-mirror.com`
（框架侧开关：`AUTOVISION_MIRROR_URL` / `AUTOVISION_OFFLINE`）。`-notpretrained`
变体离线本地构建。**权重许可**：DINOv2 权重 CC-BY-NC 4.0、DINOv3 权重 DINOv3
License，默认均不下载——需分别设 `AUTOVISION_DINOV2_WEIGHTS=1` /
`AUTOVISION_DINOV3_WEIGHTS=1` 自证已接受对应条款；不受限的干净路径是
`-notpretrained` + 本框架自监督预训练（政策见 `NOTICE`）。完整下载命令、
缓存位置与镜像配置见[权重下载](docs/download_weights.md)。

## 快速开始

以下示例依赖 `rfdetr` + `export` extra：

```bash
pip install -e ".[rfdetr,export]"
```

不训练，直接加载发布权重并预测：

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # 默认 parity 门禁 + postprocess 规格
```

可用模型、许可与导出边界：

```bash
autovision-cli list-models --task detection --pretrained
autovision-cli model-info rfdetr/rf-detr-small
```

目标检测，训练 → 验证 → 预测 → 导出：

```python
import autovision

data = autovision.ensure_coco8()   # 本地生成的 8 张冒烟数据集；可换 COCO/YOLO 配置
model = autovision.detection("rtdetr/resnet18").train(
    out="runs/det1", data=data, steps=2000, batch_size="auto",
)
model.val(data=data)          # {'map': ..., 'map_50': ...}
model.predict("image.jpg")    # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")    # 按 .onnx / .engine / .xml / .pt 后缀分发
```

等价命令行（同一套 Pydantic 校验）：

```bash
autovision-cli train-object-detection out=runs/det1 data=data.yaml steps=2000
autovision-cli val     model=runs/det1/exported_models/exported_best.pt data=coco8
autovision-cli predict model=runs/det1/exported_models/exported_best.pt images=dir/ out=preds/ threshold=0.5
```

## 任务入口

所有任务共用同一套 API / CLI 形式；主要差异在 `data` 结构与必填参数。

| 任务 | Python API / CLI | 默认 model | `data` 结构 | 额外要求 |
|---|---|---|---|---|
| 检测 | `train_object_detection` / `train-object-detection` | `rtdetr/resnet18` | COCO / YOLO；VOC 先 `convert-data`；`coco8` 可冒烟 | — |
| 旋转框 | `train_oriented_object_detection` / `train-oriented-object-detection` | `rtdetr-obb/resnet18` | YOLO-OBB（8 点角点） | — |
| 实例分割 | `train_instance_segmentation` / `train-instance-segmentation` | `dinov2/eomt` | COCO instances JSON + images | `transformers` extra；必填 `num_classes` |
| 全景分割 | `train_panoptic_segmentation` / `train-panoptic-segmentation` | `dinov2/eomt` | images + masks PNG + segments JSON | `transformers` extra |
| 语义分割 | `train_semantic_segmentation` / `train-semantic-segmentation` | `torchvision/resnet18` | images + masks PNG | 必填 `num_classes` |
| 姿态 | `train_pose_estimation` / `train-pose-estimation` | `gtr/s` | COCO `person_keypoints` JSON + images | `gtr` extra；官方权重需 `download` |
| 深度 | `train_depth_estimation` / `train-depth-estimation` | `gtr/s` | images + depths（`.npy` 米或 16-bit PNG） | `gtr` extra；官方权重需 `download` |
| 分类 | `train_image_classification` / `train-image-classification` | `torchvision/resnet18` | ImageFolder `train/`、`val/` | — |

每个任务都有“纯推理 + 训练”两个示例（推理加载现有权重，训练默认生成合成数据；
`--dry-run` 只打印数据契约）：

```bash
python examples/tasks/detection.py                  # 推理，不训练
python examples/tasks/instance_segmentation_train.py --dry-run
python examples/tasks/yolo/detection.py             # 可选 AGPL YOLO 基线
```

完整列表见 [任务示例](docs/task_examples.md) / `examples/tasks/README.md`。

`dinov2/eomt` 与 GTR 模型分别依赖 DINOv2 权重与 GTR 官方权重；离线或受限网络
环境请使用 `-notpretrained` 变体、本地 checkpoint，或先按许可说明设置权重门控。
GTR 官方权重可用 `scripts/download_gtr_assets.py` 下载。

<details>
<summary>最小示例：旋转框 / 实例分割 / 姿态 / 分类 / 语义分割</summary>

```python
import autovision

# 旋转框：data.yaml 使用 YOLO-OBB 布局
autovision.train_oriented_object_detection(
    out="runs/obb",
    data="obb.yaml",
    model="rtdetr-obb/resnet18",
)

# 实例分割：COCO instances JSON
autovision.train_instance_segmentation(
    out="runs/ins",
    data={
        "train": {"annotations": "instances_train.json", "images": "train2017"},
        "val": {"annotations": "instances_val.json", "images": "val2017"},
    },
    num_classes=80,
    model="dinov2/eomt",
)

# 姿态：COCO person_keypoints；官方权重直载
autovision.train_pose_estimation(
    out="runs/pose",
    data={
        "train": {
            "annotations": "person_keypoints_train2017.json",
            "images": "train2017",
        },
        "val": {
            "annotations": "person_keypoints_val2017.json",
            "images": "val2017",
        },
    },
    model="gtr/s",
    model_args={"checkpoint": "gtrpose_s_coco"},
)

# 分类：ImageFolder（每个子目录一个类别）
autovision.train_image_classification(
    out="runs/cls",
    data={"train": "data/train/", "val": "data/val/"},
    model="torchvision/resnet18",
)

# 语义分割：成对 images / masks，255 为忽略标签
autovision.train_semantic_segmentation(
    out="runs/seg",
    data={"train": {"images": "images/", "masks": "masks/"}},
    num_classes=3,
    model="torchvision/resnet18",
)
```

</details>

## 能力与子系统

- **一套校验、两个入口**：Python API 与 `autovision-cli`（30 条命令）共享 Pydantic
  严格校验；拼错或越界直接报错（did-you-mean）。
- **模型面**：7 个注册源（torchvision、timm、super-gradients、ultralytics（AGPL
  隔离）、dinov2、dinov3、rfdetr）+ 树内 `rtdetr/*`、`detr/*`、`dfine/*`、
  `picodet/*`、`rfdetr/*`、`gtr/*` 装配，覆盖检测、分割、姿态、深度、分类。
- **训练与数据**：自监督预训练（DINO / DINOv2 / DINOv3、SimCLR、DetCon 等）、
  蒸馏、DDP / 混合精度 / 原子 checkpoint + resume；每次训练在输出目录写
  `run_manifest.json`（命令 / git commit / 环境白名单 / 依赖版本 / GPU）；
  mmap 文件清单与 VOC / YOLO 转换。
- **部署与闭环**：ONNX / TensorRT / OpenVINO + 数值对拍；伪标注、误差分析、HPO、
  内容寻址快照，均带 CLI。

<details>
<summary>子系统总览（配置 / 数据 / 方法 / 运行时 / 任务 / 部署）</summary>

| 子系统 | 能力 |
|---|---|
| 配置 / 注册 | Pydantic 严格校验 + did-you-mean 报错 + Registry；CLI 与 API 同一套校验 |
| 模型来源 | 7 个注册源：`torchvision`、`timm`、`super_gradients`、`ultralytics`（AGPL 隔离）、`dinov2`、`dinov3`、`rfdetr`；`custom` 兜底；`arch_traits()` → `weight_decay="auto"` 按架构解析；树内 `rtdetr/*` 无第三方包 |
| 数据 / 变换 | 图像、ImageFolder 分类、成对分割、COCO 关键点、深度图；mmap 文件清单（引用计数 / 原子发布 / 碰撞守卫 / 数据指纹）；ViewTransform 管线 + MethodTransform 多视图协议；VOC / YOLO-pose / YOLO-seg 转换与 COCO 合并 |
| 预训练方法 | DistillationV3（全规格）、SimCLR（NT-Xent）、DINO（EMA 教师 + 中心化损失 + 多裁剪 + 动量/温度/权重衰减三调度）、DINOv2 / DINOv3、DenseCL / DetCon B/S、训练中 kNN 探针 |
| 运行时 | step 循环、预训练与任务训练双 DDP、混合精度、原子 checkpoint + resume、`on_before_optimizer_step` 钩子、自描述 checkpoint 格式 |
| 任务层 | 检测：`rtdetr/*`、`detr/*`、`dfine/*`、`picodet/*`、`rfdetr/*`；旋转框：`rtdetr-obb/*`、`gtr/*`；分割：实例 / 全景 / 语义；姿态、深度、分类；任务指标支持跨 rank 状态汇聚（DDP 验证）；误差分析（FP 归因 + 面积/类别切片 + top-FP/FN 画廊）与伪标注/自训练闭环 |
| 部署 | ONNX（FP16 黑名单 + 数值对拍门禁）、TensorRT（batch-only profile、attention FP32）、benchmark（torch / ONNX / TensorRT + parity 门禁，报告携带参数量/FLOPs）、模型成本（`measure_model_cost`）、`export` CLI（导出件后缀分发）、参考推理运行时（视频/RTSP + batch，torch/ONNX/TRT 同源前后处理） |

</details>

## 模型与指标

41 个已注册检测键（39 个规格变体 + 2 个产品档位别名）、26 个 GTR 官方权重、
7 个用户自持检测器（其中 3 个 STA-era 当前不可加载）。许可、sha256、官方指标与
加载代码见[模型库](docs/model_zoo.md)。

<details>
<summary>检测谱系 <code>detr/*</code>（41 个注册键）</summary>

| 骨干族 | 变体 | 数量 |
|---|---|---|
| DINOv2 ViT（patch 14） | `vits14` / `vitb14` / `vitl14` / `vitg14`，各带 `-noreg` 与 `-notpretrained`；`vittest-notpretrained` 为离线测试变体 | 13 |
| DINOv3 ViT（patch 16） | `vits16` / `vitb16` / `vitl16` / `vitt16` / `vitt16plus`；`-eupe` 架构别名，各档带 `-notpretrained` | 13 |
| DINOv3 ConvNeXt | `tiny` / `small` / `base` / `large`；`-eupe` 架构别名 | 7 |
| GTR GLA 骨干 | `gtrvit-s` / `gtrvit-m` / `gtrvit-l` / `gtrvit-x`（需 `gtr` extra） | 4 |
| timm HGNetV2 | `hgnetv2-b0` / `hgnetv2-b1` | 2 |
| **规格小计** | | **39** |
| 产品档位别名 | `detr/n` → `hgnetv2-b0`，`detr/s` → `hgnetv2-b1` | 2 |
| **注册键总计** | | **41** |

DINOv2 与 DINOv3 ViT 均使用 no-STA adapter（语义 tap 直投 `hidden_dim`）；DINOv3
ViT 保留，但含 `backbone.sta.*` 的旧检查点会被加载器拒绝。DINOv3 ConvNeXt 末三段
stage tap，整骨干走 backbone LR。

`detr/n` / `detr/s` 为 spec-frozen（规格冻结，预训练权重 / AP 待发布）；
`detr/m/l/x` 为 planned，尚未注册。编码器 / 解码器配方逐变体携带，不共享单一全局默认。

</details>

<details>
<summary>GTR 多任务家族（26 个官方权重，MIT）</summary>

Intellindust GTR（MIT）整族 26 个官方权重已在 `autovision.registry.gtr_assets`
注册（sha256 全钉住、上游 revision 可钉住，`resolve_gtr_asset` 解析、
`verify_gtr_asset` 离线核验）。二进制不随仓库分发；用
`python scripts/download_gtr_assets.py --all` 从上游 HF `Phoenix8125/GTR`
按需下载到 `AUTOVISION_GTR_ROOT`（默认 `~/Research/Model/gtr`；需
`pip install -e ".[download]"`；上游 gated 时先 `hf auth login`）。
整族共享一套 GLA 骨干（`ViTAdapterSpatialSwiGLU`）+ `GTREncoder`，各任务只换
头 / 解码器。

| 任务 | model 键 | 变体 | 官方指标 | 权重资产 |
|---|---|---|---|---|
| 检测 COCO | `detr/gtrvit-<s\|m\|l\|x>` | s/m/l/x | AP 53.6 / 57.3 / 58.9 / 59.4 | `gtr_{s,m,l,x}_coco` |
| 检测 Obj365 预训练 | —（作检测初始化权重） | s/m/l/x | — | `gtr_{s,m,l,x}_obj365` |
| 实例分割 COCO | `gtr/<s\|m\|l\|x>` | s/m/l/x | mask AP 45.0 / 47.7 / 49.5 / 49.8 | `gtrseg_{s,m,l,x}_coco` |
| 语义分割 Cityscapes | `gtr/<s\|m\|l\|x>` | s/m/l/x | mIoU 81.5 / 83.0 / 83.2 / 83.3 | `gtrsemseg_{s,m,l,x}_cityscapes` |
| 深度 NYU | `gtr/<s\|m\|l\|x>` | s/m/l/x | absRel 0.074 / 0.069 / 0.069 / 0.067 | `gtrdepth_{s,m,l,x}` |
| 旋转框 DOTA | `gtr/<s\|x>` | s/x | AP50 80.0 / 81.3 | `gtrobb_{s,x}_dota` |
| 姿态 COCO | `gtr/<s\|m\|l\|x>` | s/m/l/x | kp AP 70.1 / 74.1 / 74.7 / 75.5 | `gtrpose_{s,m,l,x}_coco` |

`gtr/*` 任务族用 `model_args={"checkpoint": "<注册资产名或路径>"}` 直载官方权重
后微调，不传即随机初始化。上述任务组的权重加载与前向数值对拍已打通（检测解码器
谱系金标准见 `tests/test_gtr_integration.py`；姿态全链读数与 delta 见
[docs/benchmarks.md](docs/benchmarks.md)）。GTR 前向依赖 Triton（CUDA，`gtr` extra），
且不可导出 ONNX / TensorRT。

</details>

### 用户自持权重库（仓库外，不可再分发）

7 个第三方训练的 COCO 检测器（DINOv3 骨干 + 第三方检测头）经合规重打包——张量
逐位不变，容器与元数据换为框架格式并携带一等许可块，逐位忠实性对拍（logits /
boxes 逐位相等）——存于**仓库之外**（默认 `~/Research/Model/AutoVision`，
`AUTOVISION_USER_HELD_ROOT` 覆盖）。资产表与 sha256 见[模型库](docs/model_zoo.md)；
两条加载路线与离线核验见[用户自持权重指南](docs/user_held_weights.md)。

其中 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 为 STA-era
（含 `backbone.sta.*`），STA 删除后加载器会明确拒绝；4 个 ConvNeXt 资产仍可加载。

**许可边界**（`DINOv3-License+NOASSERTION`）：骨干继承 DINOv3 License 义务，检测头
无许可声明（默认保留全部权利）。**禁再分发、禁商用**；不随仓库或导出产物分发，
不注册可下载资产清单（`ASSET_MANIFEST`）。使用即视为已自行接受相应条款。

## 导出与部署

需要 `export` extra；导出 OpenVINO `.xml` 另需 `openvino` extra，TensorRT 引擎需
自行安装 `tensorrt`。

```python
result = autovision.train_object_detection(out="runs/det1", data="coco8", steps=2000)
model = autovision.load_task_model(result.best_exported_model_path)   # 导出件 → 任务模型
model.export_onnx("model.onnx", precision="fp16")                     # FP16 + 数值对拍门禁
model.export_tensorrt("model.engine", precision="fp16")               # 需自行安装 tensorrt

autovision.benchmark("rtdetr/resnet18", out="bench/", num_classes=2,
                     backends={"torch": {}, "onnx": {"provider": "cpu"}},
                     verify_parity=True)

autovision.measure_model_cost(model)   # 参数量 / FLOPs（含 MACs；benchmark 报告内同样携带）
```

等价命令行（导出件后缀分发 `.onnx` / `.engine` / `.xml`）：

```bash
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.onnx precision=fp16
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.onnx dynamic_spatial=true
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.xml        # OpenVINO IR（对拍门禁）
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.engine precision=int8 calibration_data=calib_images/
```

检测任务的 ONNX / OpenVINO / TensorRT 导出件旁会写 `<artifact>.postprocess.json`：
记录 top-k、分数阈值、原图映射与预处理（resize / mean / std）；DETR 家族走 top-k，
PicoDet 额外带 class-offset NMS。设备侧可用
`autovision.export.decode_detections_numpy` 复现最终框；
`autovision.export.verify_end_to_end_parity` 比较最终框 / 分数 / 类别，而不是
张量求和，适合作为跨后端发布门禁。

int8 全量 AP 报告（同 split / 同前后处理，默认 AP 下降 ≤ 1.0）：

```bash
python scripts/measure_int8_accuracy.py --model best.pt --data coco.yaml \
    --calibration-images calib/ --out bench/int8 --max-images 5000
```

设备侧最小 C++ 参考实现见 `examples/cpp/`（ONNX Runtime + OpenCV；host-only
后处理测试只需 g++）。

动态 H/W 支持面、int8 首层 fp32 配方、TensorRT 11 适配与 parity 跳过条件见
[导出与部署细节](docs/export.md)。

## 推理运行时与服务

视频 / RTSP 参考推理（`autovision.inference`，需 `.[video]`；后端换 `onnx` /
`tensorrt` / `openvino` 时附 `artifact=` 编译产物，前后处理与评测同源）：

```bash
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ batch_size=4
autovision-cli predict model=exported_best.pt source=rtsp://camera/live out=preds/ backend=onnx artifact=model.onnx
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ backend=openvino artifact=model.xml openvino_device=CPU
```

输出 `out_dir/predictions.jsonl`（逐帧框，原始像素）+ `out_dir/annotated.mp4`（渲染画面）。

多路视频 / RTSP（共享批处理、每路独立产物、断流重连、队列背压）：

```bash
autovision-cli predict model=exported_best.pt sources=rtsp://cam1/live,rtsp://cam2/live \
    out=preds/ batch_size=4 queue_size=8 drop_policy=drop_oldest
```

每路输出 `out/<source>/predictions.jsonl` 与 `annotated.mp4`，汇总
`out/summary.json`（帧数 / 丢弃 / 重连 / 错误）。`drop_policy` 可选
`drop_oldest`（实时优先）或 `block`（离线批处理）；`max_frames_per_source` /
`max_seconds` 可限制运行。v1 不做跨路跟踪 / ReID。

<details>
<summary>ByteTrack 跟踪（class-aware，官方门限语义）</summary>

卡尔曼运动模型 + 三阶段 IoU 关联，class-aware，官方门限：新轨迹需
`score ≥ track_thresh + 0.1` 且连续两帧命中才发布（首帧例外）、低分框 `>0.1`
只维持轨迹不新建。`age` 是轨迹创建以来的帧数；tracker 接收源视频的真实帧率，
`track_buffer` 按该帧率折算丢失窗口。

```bash
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ track=true
```

JSONL 逐帧增加 `tracks`（`track_id` / `bbox` / `score` / `class_id` / `age`），
渲染画面标注 `#id 类别 分数`，汇总带唯一 id 数；API 侧 `run_stream(..., track=True)`。

区域统计（`track=True`）：`LineZone(start, end)` 统计定向穿越 in/out（按类别），
`PolygonZone(points)` 统计当前区域内目标数；结果写入 JSONL 的 `zones` 并叠加到
渲染画面。`smooth=True` 对每条轨迹的框做滑动平均（仅影响渲染，JSONL 保留原始
预测）。单路 `run_stream` 采用 reader / infer / writer 三段式管线，返回
`decode_s` / `infer_s` / `render_s` / `write_s` 与端到端 FPS。

```python
from autovision.tracking import LineZone, PolygonZone
from autovision.inference import run_stream

run_stream("clip.mp4", "preds/", model="best.pt", track=True,
           line_zones=[LineZone((0, 360), (1280, 360), name="gate")],
           polygon_zones=[PolygonZone([(0, 0), (640, 0), (640, 720), (0, 720)], name="left")],
           smooth=True)
```

</details>

<details>
<summary>参考 serving（FastAPI：/predict、/healthz、/readyz、/drift、/metrics）</summary>

需要 `.[serving]`，属参考实现而非完整生产网关：

```bash
autovision-cli serve model=exported_best.pt port=8000
# 可选加固：API key（推荐用环境变量，避免进入 shell history）+ 上限 + 并发闸门
export AUTOVISION_SERVING_API_KEY="$SERVE_KEY"
autovision-cli serve model=exported_best.pt port=8000 \
    max_body_bytes=8388608 max_image_pixels=40000000 max_concurrency=1
# TLS + 2 个 uvicorn worker（多 worker 要求 model= 是路径）
autovision-cli serve model=exported_best.pt port=8443 workers=2 \
    ssl_keyfile=key.pem ssl_certfile=cert.pem
# 进程内限流（每 worker 一份桶；全局配额放网关）
autovision-cli serve model=exported_best.pt rate_limit_per_minute=600 \
    rate_limit_burst=20
```

- `POST /predict`：`{"images": ["<base64>", ...]}`（或单张 `{"image": ...}`），
  批处理按输入顺序返回，前后处理与评测同源；推理在线程池执行，不阻塞事件循环；
- `GET /healthz`：liveness + 后端 / 模型 / threshold / variant；
- `GET /readyz`：readiness（所有 variant 引擎可调用时 200，否则 503）；
- `GET /drift`：预测分布 PSI（标签分布 + 分数直方图）；未配置基线时
  `{"enabled": false}`，样本不足时 `status: insufficient_data`，超过阈值时
  `status: drift`；
- `GET /metrics`：Prometheus 文本（请求数 / 错误数 / 图片数 / 延迟 summary /
  variant 请求数 / drift PSI）。
- 防护：`api_key=` 开启 Bearer / `X-API-Key` 常量时间校验；`max_body_bytes`
  在流式读取时限制请求体；`max_image_pixels` 在解码前拒绝解压炸弹；
  `max_concurrency` 默认 1（TensorRT/OpenVINO runner 非线程安全）；
  `inference_timeout_s` 默认 120s，单次引擎调用超时返回 504 并将该实例标记为
  不可再用（后续请求 503，需重启），避免卡死的后端被继续复用。
  `trust_proxy_headers=true` 只在可信反向代理后面使用（否则客户端可伪造
  `X-Forwarded-For` 绕过按 IP 限流）。
- 每个响应带 `X-Request-ID`（可用同名请求头透传）与 `variant`（A/B/canary）。

**A/B / canary（Python API）**：

```python
from autovision.serving import build_drift_baseline, create_app

app = create_app(
    variants={"stable": stable_engine, "canary": canary_engine},
    variant_weights={"stable": 0.95, "canary": 0.05},   # 5% canary
    drift_baseline="drift_baseline.json",               # 可选
)
# 基线由参考预测构建：
# build_drift_baseline(reference_records).save("drift_baseline.json")
```

变体按请求 ID 稳定哈希分流；客户端可传 `X-Request-ID` 做粘性分流，
`X-AB-Variant: stable|canary` 可强制指定（测试/回滚；生产网关可按需剥离该头）。
每个 variant 的请求数与延迟在 `/metrics` 单独计数。

</details>

**边界**：已提供 TLS、多 worker、进程内限流、分布漂移与 A/B/canary 分流；
限流桶与漂移窗口在多 worker 下各自独立（跨 worker 聚合需指标后端/网关）；
`/drift` 是服务级聚合（A/B 混合流量会混合分布，按 variant 分析需拆分实例或
在指标后端按 variant 聚合）；全局配额、告警/值班、A/B 统计显著性与金丝雀
自动回滚仍属生产网关/平台职责。

## 进阶功能

<details>
<summary>自监督预训练与蒸馏</summary>

```python
result = autovision.pretrain(
    out="runs/exp1",
    data="path/to/images",            # 目录或文件列表
    model="torchvision/resnet18",     # 学生骨干
    epochs=2,
    batch_size=4,
    method_args={"knn_eval": {"enabled": True, "k": 5}},   # 训练中 kNN 探针
)
print(result.exported_model_path, result.metrics)
```

</details>

<details>
<summary>大图切片推理（SAHI）</summary>

`autovision.tasks.shared.sahi` 提供 `tile_image` + `combine_object_detection_tiles`；
检测模型另有 `predict_sahi(image, tile_size, overlap, threshold, ...)`（默认 tile
取模型输入尺寸，全图 resize 视图作全局瓦片一并推理）。

- `batch_size=N` 限制单次前向的视图数（全局 + tile），避免高分辨率图一次堆叠全部
  tile 的显存峰值；`None` 保留原单批语义。
- `merge="nmm"` 对重叠碎片做非极大值合并（并集框 + 面积加权分数），适合被切断 /
  相邻目标；默认 `"nms"`。
- `overlap_metric="ios"` 用 Intersection-over-Smaller 判定嵌套框，默认 `"iou"`。
- tile 之间在合并前统一做 per-label 过滤，重叠区重复框不会漏到全局阶段。

</details>

<details>
<summary>数据版本化（内容寻址快照）</summary>

mmap 清单的缓存键是「路径 + mtime」；做**内容寻址快照**用 `data-snapshot`
（每分片记录根路径、文件数/字节数与 `(相对路径, 大小)` 清单的 sha256；`deep=true`
再对文件内容哈希）：

```bash
autovision-cli data-snapshot data=coco8 out=coco8.snap.json deep=true   # 写入
autovision-cli data-snapshot snapshot=coco8.snap.json                   # 校验（漂移非零退出）
autovision-cli data-snapshot snapshot=coco8.snap.json spec=copied.yaml  # 迁移/副本校验
```

漂移报告给出逐项差异（added / removed / resized / content_digest）与涉及文件；
摘要是「根相对」的，目录整体搬迁后可用副本 spec 校验通过。`spec` 内的相对路径按
spec 文件所在目录解析（与训练入口同一规则；快照记录该基准目录，换 CWD 校验不会
漂移）；`out` 已存在时报错，需显式 `overwrite=true`；`snapshot=... deep=true` 只对
deep 快照有效——浅快照会明确报错，避免「内容校验」空转。

</details>

<details>
<summary>早停与训练回调（task training）</summary>

早停通过 `trainer_args` 暴露，和验证节奏一起配置：

```bash
autovision-cli train-object-detection ... \
    trainer_args='{"val_every_num_steps": 500, "early_stopping_patience": 5, "early_stopping_min_delta": 0.001}'
```

`early_stopping_patience` 是「连续多少次验证没有超过 `best + min_delta`」；
触发时引擎保存 latest checkpoint（即使不是计划保存步），`TaskTrainResult`
返回 `stopped_early` / `stop_step`。必须同时配置 `val_every_num_steps`
和验证集，否则配置阶段直接报错。

Python API 还支持训练回调（rank 0 执行，异常只告警、不中断训练）：

```python
class MyCallback:
    def on_train_start(self, steps, out_dir): ...
    def on_step_end(self, step, log_dict): ...
    def on_validation_end(self, step, metrics): ...
    def on_early_stop(self, step, best_metric, patience): ...
    def on_train_end(self, result): ...

engine.fit_task(..., callbacks=[MyCallback()])
```

</details>

<details>
<summary>超参搜索（随机 / 网格 / ASHA）</summary>

随机 / 网格搜索与 ASHA（逐 rung 续训，非重训），复用与 CLI 同一套 Pydantic
覆盖面（`model_args.*` / `transform_args.*` / `trainer_args.*` 与顶层参数），
越界或拼错的键由配置校验直接报错：

```yaml
# search.yaml（点号键 → 分布；裸列表 = choice）
model_args.lr: {loguniform: [1.0e-5, 2.0e-3]}
model_args.weight_decay: {choice: [0.0, 1.0e-4, 5.0e-4]}
model_args.ema_warmup_steps: {choice: [100, 500, 2000]}
batch_size: [2, 4]
```

```bash
autovision-cli hpo space=search.yaml out=runs/hpo task=object_detection \
    data=coco8 model=rtdetr/resnet18 trials=6 budget=300 \
    objective=val_metric/map_50 strategy=random seed=0
autovision-cli hpo space=search.yaml out=runs/hpo-asha task=object_detection \
    data=coco8 model=rtdetr/resnet18 strategy=asha trials=9 budget=200 asha_max_budget=1800
```

产出 `out/report.md`（空间摘要 + 按目标排序的试验表 + 最优配置）、
`out/trials.jsonl`（逐试验机器可读记录）、`out/best.json`；已完成的试验目录
命中 `metrics.json` 即复用（中断后可续跑）。API：`autovision.hpo.run_search` +
`task_trial`（任意 `autovision.train_*` 入口）。Ray Tune 用户可用同一 trial 函数
接 `ASHAScheduler`（本实现零依赖）。

</details>

<details>
<summary>数据闭环（VOC 转换 / 数据体检 / 伪标注 / 误差分析）</summary>

VOC 数据可直接训练（下载 → 转换 → 训练三步；官方源较慢时自备数据集同流程）：

```bash
autovision-cli download dataset=voc2007-trainval out=datasets/       # 解包出 VOCdevkit/VOC2007
autovision-cli convert-data from=voc voc-root=datasets/VOCdevkit/VOC2007 out=voc_train.json imagesets=train
autovision-cli convert-data from=voc voc-root=datasets/VOCdevkit/VOC2007 out=voc_val.json   imagesets=val
autovision-cli train-object-detection out=runs/voc model=rtdetr/resnet18 steps=2000 \
    data='{"format": "coco", "train": {"annotations": "voc_train.json", "images": "datasets/VOCdevkit/VOC2007/JPEGImages"}, "val": {"annotations": "voc_val.json", "images": "datasets/VOCdevkit/VOC2007/JPEGImages"}}'
```

`difficult` 目标转为 COCO ignore 区（不罚不欠召回）；关键点 / 分割的 YOLO 布局
用 `from=yolo-pose | yolo-seg`。CVAT / Label Studio 可直接转 COCO：

```bash
autovision-cli convert-data from=cvat cvat-xml=annotations.xml images=images/ out=cvat.json
autovision-cli convert-data from=label-studio ls-json=export.json images=images/ out=ls.json
```

CVAT 支持 bbox / polygon；Label Studio 支持 rectangle / polygon / keypoint
（keypoint 以 `keypoint_radius` 小框输出，原始坐标保留在 `keypoint` 字段）。

数据体检（坏图 / 重复图 / split 泄漏 / 框越界 / 类别分布）：

```bash
autovision-cli data-check data=coco.yaml out=qa/ near_duplicate=true
```

输出 `qa/report.md` + `report.json` + `manifest.json`；`fail_on=error`（默认）
发现问题时非零退出。`near_duplicate` 为 O(n²)，大数据集可关闭或只做精确重复。

难例回灌（**在训练 split 上挖掘**，避免把验证集样本采样进训练）：

```bash
autovision-cli analyze-errors model=best.pt data=coco_train.yaml split=train out=report/
autovision-cli train-object-detection out=runs/r2 data=coco.yaml model=rfdetr/rf-detr-small \
    hard_examples=report/hard_examples.json
```

`analyze-errors` 输出 `report/hard_examples.json`；训练入口按
`min(max_repeat, 1 + alpha × 归一化误差)` 加权采样。v1 只支持单机
（DDP 会覆盖 sampler，组合被显式拒绝）。

伪标注 / 自训练与长尾采样：

```bash
# 单轮伪标注；scripts/self_train_detect.py 可串联多轮（--rounds N）
autovision-cli pseudo-label model=teacher.pt images=unlabeled/ out=pseudo.json data=coco.yaml threshold=0.5
python scripts/self_train_detect.py --teacher teacher.pt --images unlabeled/ \
    --val-data coco_val.yaml --out runs/self_train --steps 2000 --rounds 3 \
    --min-improvement 0.2

# 长尾类别采样（class_balanced | repeat_factor）；与 hard_examples 互斥
autovision-cli train-object-detection out=runs/longtail data=coco.yaml \
    model=rfdetr/rf-detr-small long_tail_sampler=class_balanced
```

误差分析（TP/FP/FN 匹配、FP 归因 duplicate/classification/localization/background、
混淆矩阵、按类别与 COCO 面积分档切片、top-FP/FN 可视化画廊）：

```bash
autovision-cli analyze-errors model=best.pt data=coco.yaml out=report/ iou=0.5 top_k=5 threshold=0.3
```

输出 `report/report.md`（含内嵌画廊）+ `report/report.json`（逐条匹配记录）。

</details>

<details>
<summary>全部 CLI 命令（30）</summary>

| 组 | 命令 |
|---|---|
| 训练 | `pretrain`、`train-object-detection`、`train-oriented-object-detection`、`train-instance-segmentation`、`train-panoptic-segmentation`、`train-semantic-segmentation`、`train-pose-estimation`、`train-depth-estimation`、`train-image-classification` |
| 评测 / 推理 | `val`、`predict`、`benchmark`、`embed`、`pseudo-label`、`analyze-errors` |
| 导出 / 服务 | `export`、`serve` |
| 数据 | `download`、`convert-data`、`data-check`、`data-snapshot` |
| 模型 / 治理 | `list-models`、`model-info`、`list-methods`、`plugins`、`compliance-report`、`registry` |
| 调参 / 工具 | `hpo`、`extract-video-frames`、`plot-curves`（需 matplotlib） |

</details>

## 致谢（Acknowledgements）

AutoVision 的架构与实现受益于以下开源工作。代码 lineage 与修改说明见
[溯源台账](docs/provenance/LEDGER.md)，第三方许可文本见
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)，权重许可见
[模型库](docs/model_zoo.md)。

- **代码 / 算法 lineage**：[DEIM](https://github.com/Intellindust-AI-Lab/DEIM)
  （Apache-2.0）、[D-FINE](https://github.com/Peterande/D-FINE)（Apache-2.0）、
  [RT-DETR](https://github.com/lyuwenyu/RT-DETR)（Apache-2.0）、
  [Detectron2](https://github.com/facebookresearch/detectron2)（Apache-2.0）、
  [GTR](https://github.com/Intellindust-AI-Lab/GTR)（MIT）、
  [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment)
  （Apache-2.0）。
- **模型与权重**：[Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
  （Apache-2.0）、[Intellindust GTR](https://github.com/Intellindust-AI-Lab/GTR)
  （MIT）、[Meta DINOv2](https://github.com/facebookresearch/dinov2)（代码
  Apache-2.0；权重 CC-BY-NC 4.0）、
  [Meta DINOv3](https://github.com/facebookresearch/dinov3)（权重 DINOv3
  License）、[PaddleDetection PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection)
  （Apache-2.0）。
- **跟踪 / 切片**：[ByteTrack](https://github.com/ifzhang/ByteTrack)（ECCV
  2022，MIT）、[SAHI](https://github.com/obss/sahi)（MIT）提供了算法参考；
  本仓库为 AutoVision 实现。
- **可视化 / 视频工具**：[Roboflow supervision](https://github.com/roboflow/supervision)
  （MIT）提供了增强标注器、骨架注册表、框/mask/polygon 工具与视频 sink 的
  API/算法参考；本仓库为 AutoVision 实现，不引入 supervision 运行时依赖。
- **推理 / 训练工具**：PyTorch、torchvision、HuggingFace Transformers、
  ONNX Runtime、NVIDIA TensorRT、OpenVINO、FastAPI、MLflow、
  Weights & Biases、TensorBoard。
- **数据集**：COCO、PASCAL VOC、Cityscapes、DOTA、NYU Depth V2、Objects365、
  ImageNet；各自条款适用，不随本仓库分发。

> YOLO / Ultralytics 名称仅用于兼容性描述，不表示官方背书或关联；
> `ultralytics` 组件以 AGPL-3.0 extra 隔离，不进入核心依赖、不随仓库分发。
> README 致谢是社区信用；法律归属以 `NOTICE` / `THIRD_PARTY_LICENSES.md`
> 为准。

## 开发

```bash
pip install -e ".[dev,export,serving,video,transformers]"   # CI 同款测试依赖
pytest tests                     # 2500+ 个测试（含金标准 loss 轨迹快照）
ruff check src tests scripts     # lint（规则集见 pyproject.toml）
bandit -r src/autovision -ll -ii -q   # SAST（medium/high）
python scripts/smoke_enterprise.py --json /tmp/smoke.json   # 企业冒烟
python scripts/check_provenance.py && python scripts/check_layering.py   # CI 门禁
python scripts/check_docs_links.py && mkdocs build --strict              # 文档门禁（需 .[docs]）
```

- 金标准快照在 `tests/golden/`；**有意的**配方变更后删除快照重新录制（对应测试跑两遍）。
- 真实 `from_pretrained` 的测试默认跳过，`AUTOVISION_HF_TESTS=1` 开启（timm 权重下载测试同门控）。
- 多进程 DDP 测试：`AUTOVISION_DDP_TESTS=1 pytest tests/test_torchrun_distributed.py`。
- TensorRT int8 引擎构建测试（慢，需 tensorrt + CUDA）：`AUTOVISION_TRT_TESTS=1 pytest tests/test_trt_int8.py`。
- 私有用户自持权重集成测试（需本地 store）：`AUTOVISION_USER_HELD_TESTS=1 pytest tests/test_user_held_detr_assets.py`。
- 贡献入口：[CONTRIBUTING.md](CONTRIBUTING.md)（环境 / 门禁 / 许可与溯源红线）；
  发布与版本政策 [docs/release_process.md](docs/release_process.md)；变更记录 [CHANGELOG.md](CHANGELOG.md)；
  安全漏洞披露见 [SECURITY.md](SECURITY.md)（请勿公开 issue）。
- 文档：模型库（许可 / sha256 / 官方指标 / 加载代码）[docs/model_zoo.md](docs/model_zoo.md)
  （由 `scripts/gen_model_zoo.py` 从注册表机器生成）；as-built 架构总览
  [docs/framework_design.md](docs/framework_design.md)；基准表与口径协议
  [docs/benchmarks.md](docs/benchmarks.md)；环境变量全表 `src/autovision/runtime/env.py`
  （`AUTOVISION_*`，遥测默认关闭）。文档站点 = `docs/` 下 `mkdocs.yml` 导航收录的页面
  （内部分析稿经 `exclude_docs` 显式排除，不进站点）。

## 已知限制

- 定位：训练 + 参考推理框架；serving 支持可选 API key、请求体/像素上限与
  `/readyz`，但仍不提供 TLS、多进程编排、限流；漂移检测 / A-B / 金丝雀不在范围内。
- `16-mixed` 在 CUDA/NPU 上走 `GradScaler`（实例状态随 checkpoint 恢复）；
  确定性训练需显式 `AUTOVISION_DETERMINISTIC=1`，部分算子无确定性实现会报错。
- 内置 COCO/VOC 数据集条目未钉官方 sha256；如需强制完整性，设置
  `AUTOVISION_REQUIRE_DATASET_HASHES=1` 并使用带摘要的镜像条目。
- GTR 模型（含 `detr/gtrvit-*`）不可导出 ONNX / TensorRT（GLA 自定义 CUDA 核无
  ONNX 降级，导出入口以明确错误拒绝）。
- 动态 H/W 不支持 HF 骨干（`detr/dinov2-*`、`detr/dinov3-*`）；ONNX FP16 parity
  在无 CUDA execution provider 时跳过并告警。
- 蒸馏教师离线时传 `teacher_pretrained=False`；受限许可权重默认不下载。
- albumentations 适配层与 super-gradients 分割 wrapper 未接入。
- `detr/n` / `detr/s` 为 spec-frozen（规格冻结，预训练权重 / AP 待发布）；
  `detr/m/l/x` 为 planned——状态见[模型库](docs/model_zoo.md)。

完整边界、TensorRT 11 / int8 注记与 GTR 部署工程现状见[已知限制](docs/known_limits.md)。

## 许可与溯源

核心代码 Apache-2.0；许可证全文见仓库根目录 `LICENSE`
（也可查阅 <https://www.apache.org/licenses/LICENSE-2.0>）。第三方组件与权重
溯源见仓库根 `NOTICE`、[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 与
[溯源台账](docs/provenance/LEDGER.md)。

> **0.3.0 许可状态（2026-10-17）**：DEIMv2 派生的 `SpatialPriorModule`
> 已从代码库删除，DINOv3 ViT 变体保留但使用 no-STA adapter，因此本发行物
> 不再包含 DEIMv2 代码，可按 Apache-2.0 发布。旧的 STA-era ViT 检查点
> （`backbone.sta.*`）会被加载器明确拒绝；需重新训练 / 导出，或使用 ConvNeXt
> 检查点。处置记录见溯源台账。

GTR 权重为 MIT（整族 26 个，sha256 钉住）；`ultralytics` 为 AGPL-3.0，仅作
extras 隔离加载，不进核心依赖、不在模块级 import。用户自持权重库的 7 个第三方
COCO 检测器为 `DINOv3-License+NOASSERTION`，禁再分发 / 商用；其中 3 个 STA-era
ViT 检查点已不被加载器支持，4 个 ConvNeXt 检查点仍可使用。边界见
[用户自持权重指南](docs/user_held_weights.md)。
