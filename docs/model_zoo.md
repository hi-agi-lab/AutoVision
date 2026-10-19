# 模型库

> 本页由 `scripts/gen_model_zoo.py` 从权重注册表机器生成（许可 / sha256 / 官方指标 / 接入键），**勿手改**；注册表变更后重新生成，漂移由
> `scripts/gen_model_zoo.py --check` 与 `tests/test_model_zoo.py` 拦截。
>
> 仓库**不携带权重二进制**：注册表钉身份（sha256），不钉分发。表中数字为**官方发表值**（出处见各节）；我方复现读数与协议见[基准](benchmarks.md)，许可红线见仓库根 `NOTICE`。

## 1. 资产概览

| 类别 | 数量 | 许可 | 分发形态 | 注册表 |
|---|---|---|---|---|
| GTR 官方家族 | 26 | MIT | 来源 HF `Phoenix8125/GTR`，按需下载到 `AUTOVISION_GTR_ROOT`（仓库不携带二进制） | `src/autovision/registry/gtr_assets.py` |
| HF 权重清单 | 31 | Apache-2.0 / MIT / CC-BY-NC-4.0 / DINOv3 License | HF 下载 + sha256 校验（受限许可门控） | `src/autovision/encoders/weight_hashes.py` |
| 可下载资产（warm-start） | 3 | Apache-2.0 | HF / GitHub 镜像下载 + sha256 校验 | `src/autovision/registry/apache_assets.py` |
| 用户自持库 | 7 | DINOv3-License+NOASSERTION | 仓库外存储，**不可再分发** | `src/autovision/registry/user_held_assets.py` |

当前环境可用的模型键（按已安装 extras 过滤）：`autovision-cli list-models`。

## 2. GTR 多任务家族（MIT，旗舰）

整族共享一套 GLA 骨干（`ViTAdapterSpatialSwiGLU`）+ `GTREncoder`，各任务只换头 / 解码器；权重为官方 **EMA 口径**（官方所有发表数字均用 EMA）。

权重二进制不随仓库分发：用 `scripts/download_gtr_assets.py` 从上游 HF `Phoenix8125/GTR` 按需下载到 `AUTOVISION_GTR_ROOT`（默认 `~/Research/Model/gtr`），逐文件核验 sha256；上游 revision 可钉住（下载器会在首次解析后打印实际 commit）。上游若为 gated，先 `hf auth login`；区域镜像用 `HF_ENDPOINT`。

| 任务 | 变体 | 资产 | 官方指标 | 官方延迟 (ms) | sha256 |
|---|---|---|---|---|---|
| 检测（COCO 2017，AP） | l | `gtr_l_coco` | AP 58.9 | 1.908 | `d7a3e07b2159` |
| 检测（COCO 2017，AP） | m | `gtr_m_coco` | AP 57.3 | 1.462 | `a1a90111bb75` |
| 检测（COCO 2017，AP） | s | `gtr_s_coco` | AP 53.6 | 1.225 | `69b8bed4d2bd` |
| 检测（COCO 2017，AP） | x | `gtr_x_coco` | AP 59.4 | 2.115 | `547cc2c2d29b` |
| 检测初始化（Objects365 预训练） | l | `gtr_l_obj365` | —（预训练权重，无基准行） | — | `364c76be3c21` |
| 检测初始化（Objects365 预训练） | m | `gtr_m_obj365` | —（预训练权重，无基准行） | — | `5338fcf02f18` |
| 检测初始化（Objects365 预训练） | s | `gtr_s_obj365` | —（预训练权重，无基准行） | — | `49be6e816149` |
| 检测初始化（Objects365 预训练） | x | `gtr_x_obj365` | —（预训练权重，无基准行） | — | `a3bdf1b3771e` |
| 实例分割（COCO，mask AP） | l | `gtrseg_l_coco` | mask AP 49.5 | — | `33cabf1f8119` |
| 实例分割（COCO，mask AP） | m | `gtrseg_m_coco` | mask AP 47.7 | — | `60963e2f22d1` |
| 实例分割（COCO，mask AP） | s | `gtrseg_s_coco` | mask AP 45.0 | — | `14113f7b7cf3` |
| 实例分割（COCO，mask AP） | x | `gtrseg_x_coco` | mask AP 49.8 | — | `63f7b3e32e5d` |
| 语义分割（Cityscapes，mIoU） | l | `gtrsemseg_l_cityscapes` | mIoU 83.2 | — | `42dec87b707f` |
| 语义分割（Cityscapes，mIoU） | m | `gtrsemseg_m_cityscapes` | mIoU 83.0 | — | `f5206bca92cb` |
| 语义分割（Cityscapes，mIoU） | s | `gtrsemseg_s_cityscapes` | mIoU 81.5 | — | `365ed33e3ba8` |
| 语义分割（Cityscapes，mIoU） | x | `gtrsemseg_x_cityscapes` | mIoU 83.3 | — | `ee4ade38e2e6` |
| 单目深度（NYU，absRel） | l | `gtrdepth_l` | absRel 0.069 | — | `15cee2d0fd37` |
| 单目深度（NYU，absRel） | m | `gtrdepth_m` | absRel 0.069 | — | `39b3bcd05768` |
| 单目深度（NYU，absRel） | s | `gtrdepth_s` | absRel 0.074 | — | `e798c6347bc2` |
| 单目深度（NYU，absRel） | x | `gtrdepth_x` | absRel 0.067 | — | `fef32854352d` |
| 旋转框（DOTA，AP50） | s | `gtrobb_s_dota` | AP50 80.0 | — | `15a57e2ecab0` |
| 旋转框（DOTA，AP50） | x | `gtrobb_x_dota` | AP50 81.3 | — | `deb785e2d648` |
| 姿态（COCO 关键点，kp AP） | l | `gtrpose_l_coco` | kp AP 74.7 | — | `aa1396b9d458` |
| 姿态（COCO 关键点，kp AP） | m | `gtrpose_m_coco` | kp AP 74.1 | — | `e93c957dd130` |
| 姿态（COCO 关键点，kp AP） | s | `gtrpose_s_coco` | kp AP 70.1 | — | `ead7a3958a11` |
| 姿态（COCO 关键点，kp AP） | x | `gtrpose_x_coco` | kp AP 75.5 | — | `dbfb964f3bc6` |

- 官方延迟 = GTR 论文协议（RTX 4090 / FP16 / batch 1 / torch.compile + CUDA Graphs / 640×640），仅检测档发表；其余任务官方未发表延迟（我方 torch 口径读数见[基准](benchmarks.md) §2）。引用日期 2026-10-15。
- sha256 为完整摘要前 12 位；完整值以注册表为准。

### 2.1 `gtr/*` 任务族（五任务直载）

`model_args` 的 `checkpoint` 接受注册资产名或路径（`resolve_gtr_asset` 解析 + sha256 核验 + 加载覆盖门：encoder/decoder 覆盖率不足即报错）：

```python
import autovision

# 姿态；其余四个任务族同为 `model="gtr/{s,m,l,x}"` + `model_args={"checkpoint": "<资产名>"}`
autovision.train_pose_estimation(out="runs/pose-s", data="pose.yaml", model="gtr/s",
    model_args={"checkpoint": "gtrpose_s_coco"})
autovision.train_semantic_segmentation(..., model="gtr/s", model_args={"checkpoint": "gtrsemseg_s_cityscapes"})   # 语义分割（Cityscapes，mIoU）
autovision.train_depth_estimation(..., model="gtr/s", model_args={"checkpoint": "gtrdepth_s"})   # 单目深度（NYU，absRel）
autovision.train_instance_segmentation(..., model="gtr/s", model_args={"checkpoint": "gtrseg_s_coco"})   # 实例分割（COCO，mask AP）
autovision.train_oriented_object_detection(..., model="gtr/s", model_args={"checkpoint": "gtrobb_s_dota"})   # 旋转框（DOTA，AP50）
```

离线核验（不加载模型，仅对文件做 sha256 核验）：

```python
from autovision.registry.gtr_assets import resolve_gtr_asset
resolve_gtr_asset("gtrpose_s_coco")   # 文件不存在 / sha256 不符即抛 GTRAssetError
```

### 2.2 检测线（`detr/gtrvit-s`）

检测的官方权重经一步**装配**（`build_dfine_official_warm_start`，EMA 口径 + 解码器覆盖率门）后以 `model_args.warm_start` 注入；`detr/gtrvit-s` 的 GLA 骨干另由 `extract_gtr_backbone` 提取为 `backbone_weights`。以下写法在 v0.2.0 已实机验证（装配报告：`decoder_coverage=1.0`、`skipped_shape=0`，短程训练闭环通过）：

```python
import autovision
from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.detection.detr.dfine_official_warm_start import (
    build_dfine_official_warm_start,
)

asset = resolve_gtr_asset("gtr_s_coco")            # MIT；sha256 核验
# `detr/gtrvit-s`（GLA 骨干线，需 `gtr` extra + CUDA）：装配 encoder+decoder，
# 骨干状态单独提取
import torch
from autovision.tasks.detection.detr.gtr_teacher import (
    extract_gtr_backbone,
    load_gtr_teacher,
)

build_dfine_official_warm_start("gtrvit-s", out="ws_gtrvit_s.pt", checkpoint=asset.path,
    backbone_pretrained=False, license_id="MIT", origin=asset.source,
    source_format="gtr")
torch.save(extract_gtr_backbone(load_gtr_teacher(asset.path)), "gtr_s_backbone.pth")
autovision.train_object_detection(out="runs/gtrvit-s", data="coco.yaml",
    model="detr/gtrvit-s", model_args={"warm_start": "ws_gtrvit_s.pt",
    "backbone_weights": "gtr_s_backbone.pth"})
```

GTR 变体补充（2026-10-17）：

- `gtrvit-{s,m,l,x}`：完整 GTR ViT 检测目标；M/L/X 已注册，装配后 encoder/decoder coverage 1.0（L/X 的 ViT 384 宽，encoder/decoder 256 宽，由 `backbone_embed_dim` 区分）。推理应走这一线。
- Objects365 检测头是 **366** 类（365 数据集类 + no-object）；装配时用 `GTRAsset.model_num_classes`，不要传 365（会丢掉 class-dependent tensors）。
- `gtr/*` 实例分割任务族现支持 `s/m/l/x`（此前只注册了 `s`）。

## 3. 检测档位谱系（`detr/n` … `detr/x`）

产品档位的冻结状态与骨干线方向记录在 `tasks/detection/detr/variants.py` 的 `SPECTRUM` 表（机器可读）：`released` 为已发布档（权重 + 模型卡 + 全量 AP + 延迟齐备），`spec-frozen` 为规格冻结档（组件与配方已承诺，预训练权重 / AP 待发布），`planned` 为方向已定、实现待落地。

| 档位 | 变体键 | 路线 | 状态 | 基准锚点 |
|---|---|---|---|---|
| `detr/n` | `hgnetv2-b0` | 蒸馏 | **spec-frozen**（规格冻结，权重/AP 待发布） | YOLOv8n / YOLO11n |
| `detr/s` | `hgnetv2-b1` | 装配微调 | **spec-frozen**（规格冻结，权重/AP 待发布） | YOLOv8s / YOLO11s |
| `detr/m` | — | 装配微调 | planned | YOLOv8m / RT-DETR-R18 |
| `detr/l` | — | 装配微调 | planned | RT-DETR-R34 / YOLOv8l |
| `detr/x` | — | 装配微调 | planned | RT-DETR-R50 / YOLOv8x |

`released` 是唯一带预训练权重的状态，由 `autovision.registry.spectrum_release` 的发布门禁校验；当前 `n/s` 为 `spec-frozen`，需用 `scripts/train_spectrum.py` 自行训练、`scripts/eval_spectrum_coco.py` 产出全量 AP。

```python
model = autovision.detection("detr/s")   # 规格冻结档；n/s 权重/AP 待发布，m/l/x 为 planned
```

## 4. HF 权重清单（revision + sha256 钉住）

预训练骨干、RF-DETR 检测权重与开放词表文本塔，按许可分组。CC-BY-NC / DINOv3 License 默认**不下载**：设 `AUTOVISION_DINOV2_WEIGHTS=1` / `AUTOVISION_DINOV3_WEIGHTS=1` 表示已自行接受对应条款。sha256 列为该资产按文件名排序的首个权重文件（双格式资产另有 `pytorch_model.bin`、多分片资产另有其余分片，完整清单见注册表）。`revision` 为 8 位短哈希，完整值见注册表。

| 资产（HF repo） | 许可 | 接入键 / 角色 | 官方指标 | revision | sha256 |
|---|---|---|---|---|---|
| **Apache-2.0（可自由再分发）** | | | | | |
| `Roboflow/rf-detr-base` | apache-2.0 | `rfdetr/rf-detr-base`（同上） | —（上游弃用） | `7b95b089` | `02ad73fd09f0` |
| `Roboflow/rf-detr-large` | apache-2.0 | `rfdetr/rf-detr-large`（同上） | AP 56.5 / 75.1 · 33.9M · 704² | `f62f7dd5` | `1ec604598aea` |
| `Roboflow/rf-detr-medium` | apache-2.0 | `rfdetr/rf-detr-medium`（同上） | AP 54.7 / 73.6 · 33.7M · 576² | `1b5b6724` | `e52098adc469` |
| `Roboflow/rf-detr-nano` | apache-2.0 | `rfdetr/rf-detr-nano`（检测变体（HF transformers 路径；`-notpretrained` 可离线构建）） | AP 48.4 / 67.6 · 30.5M · 384² | `9f201577` | `497b1dbcc8f9` |
| `Roboflow/rf-detr-small` | apache-2.0 | `rfdetr/rf-detr-small`（同上） | AP 53.0 / 72.1 · 32.1M · 512² | `3bdc4650` | `2bfab2fa5f5f` |
| `google/siglip-base-patch16-224` | apache-2.0 | `open_vocab.text_encoder="siglip-base"`（开放词表文本塔（SigLIP）） | — | `7fd15f06` | `2c63cb7d1f2e` |
| `harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers` | apache-2.0 | `（未接线）`（外部转换权重；DEIMv2 上游非商用，许可需按卡复核） | — | `b23bfb92` | `c7ea18e94f96` |
| `harshaljanjani/DEIMv2_HGNetv2_PICO_COCO_Transformers` | apache-2.0 | `（未接线）`（外部转换权重（PICO 档）；许可需按卡复核） | — | `2101e0c1` | `635377483bdd` |
| **MIT（可自由再分发）** | | | | | |
| `openai/clip-vit-base-patch16` | mit | `open_vocab.text_encoder="clip-b16"`（开放词表文本塔（CLIP）） | — | `57c21647` | `ec89c7b09c74` |
| `openai/clip-vit-base-patch32` | mit | `open_vocab.text_encoder="clip-b32"`（开放词表文本塔（CLIP）） | — | `3d74acf9` | `a63082132ba4` |
| **CC-BY-NC-4.0（禁商用；默认不下载，需显式自证）** | | | | | |
| `facebook/dinov2-base` | cc-by-nc-4.0 | `dinov2/vitb14-noreg`（骨干（不带 register）） | — | `f9e44c81` | `d73036b56966` |
| `facebook/dinov2-giant` | cc-by-nc-4.0 | `dinov2/vitg14-noreg`（骨干（不带 register）） | — | `611a9d42` | `917d3c470db9` |
| `facebook/dinov2-large` | cc-by-nc-4.0 | `dinov2/vitl14-noreg`（骨干（不带 register）） | — | `47b73eef` | `399fba97a95f` |
| `facebook/dinov2-small` | cc-by-nc-4.0 | `dinov2/vits14-noreg`（骨干（不带 register）） | — | `ed25f3a3` | `ae1e99fcefd5` |
| `facebook/dinov2-with-registers-base` | cc-by-nc-4.0 | `dinov2/vitb14`（骨干（默认带 register）） | — | `a1d738cc` | `7a6f7b3b9fa4` |
| `facebook/dinov2-with-registers-giant` | cc-by-nc-4.0 | `dinov2/vitg14`（骨干（默认带 register）） | — | `8d0d49f7` | `c03832d44691` |
| `facebook/dinov2-with-registers-large` | cc-by-nc-4.0 | `dinov2/vitl14`（骨干（默认带 register）） | — | `e4c89a4e` | `edccedab2c4e` |
| `facebook/dinov2-with-registers-small` | cc-by-nc-4.0 | `dinov2/vits14`（骨干（默认带 register）） | — | `0d9846e5` | `78e7f952c5af` |
| **DINOv3 License（默认不下载，需显式自证；条款以仓库卡片为准）** | | | | | |
| `facebook/dinov3-convnext-base-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/convnext-base`（骨干（ConvNeXt）） | — | `e619358a` | `ec90bd798b5f` |
| `facebook/dinov3-convnext-large-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/convnext-large`（骨干（ConvNeXt）） | — | `e959efa7` | `8bab82eaa133` |
| `facebook/dinov3-convnext-small-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/convnext-small`（骨干（ConvNeXt）） | — | `42102001` | `8182492b7334` |
| `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/convnext-tiny`（骨干（ConvNeXt）） | — | `10d30274` | `bd30a9459d61` |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vit7b16`（骨干（7B，多文件分片）） | — | `b8036775` | `7132627f2545` |
| `facebook/dinov3-vit7b16-pretrain-sat493m` | dinov3-custom (see repository card) | `dinov3/vit7b16-sat493m`（骨干（卫星预训练，7B）） | — | `915e1b53` | `e0c59d3e2114` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vitb16`（骨干） | — | `5931719e` | `9a21ac3df0c6` |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vith16plus`（骨干（H 档 + 蒸馏版）） | — | `c807c9ee` | `3e1d4d18b9bf` |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vitl16`（骨干（L 档）） | — | `ea8dc286` | `dcb2e45127cc` |
| `facebook/dinov3-vitl16-pretrain-sat493m` | dinov3-custom (see repository card) | `dinov3/vitl16-sat493m`（骨干（卫星预训练 SAT-493M）） | — | `f692fa42` | `4e6356d992c1` |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vits16`（骨干（LVD-1689M 预训练）） | — | `114c1379` | `4610ad75edef` |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | dinov3-custom (see repository card) | `dinov3/vits16plus`（骨干（+ 蒸馏版）） | — | `c93d816f` | `208146e499da` |
| `harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers` | dinov3-custom (see repository card) | `（未接线）`（外部转换权重（DINOv3 骨干）；DINOv3/DEIMv2 条款需按卡复核） | — | `598fb1fc` | `f127a21daf86` |

RF-DETR 行官方指标为 COCO val2017 AP / AP50、Params（M）与输入尺寸，来源为 Roboflow 官方口径；T4 / TensorRT 延迟与许可说明见[基准](benchmarks.md) §3.1。Base 上游已弃用，当前官方表不再发布指标。

一行加载（骨干走模型键；开放词表文本塔经 `model_args` 接线）：

```python
autovision.pretrain(out="runs/ssl1", data="images/", model="dinov2/vits14")   # 骨干；CC-BY-NC / DINOv3 需先按上表自证
model = autovision.detection("rfdetr/rf-detr-base")                          # 检测变体
autovision.train_object_detection(..., num_classes=2, model_args={"open_vocab": {
    "text_encoder": "siglip-base", "class_names": ("class-a", "class-b")}})
```

## 5. 可下载资产（warm-start / 蒸馏素材）

经 `resolve_downloadable`（`autovision.registry.downloadable`）解析：下载到本地缓存并核验 sha256（缓存命中同样复核），主源失败时回退镜像（hf-mirror / GitHub Release）。

| 资产 | 许可 | sha256 | 用途 |
|---|---|---|---|
| `d-fine/dfine_l_coco` | Apache-2.0 | `f1d76b20c066` | l 档装配素材 |
| `d-fine/dfine_m_coco` | Apache-2.0 | `b44a7586bf49` | m 档装配素材 |
| `d-fine/dfine_s_coco` | Apache-2.0 | `48a6c8cc43eb` | `detr/s` 装配暖启（Apache-2.0 上游权重） |

## 6. 用户自持库（仓库外，不可再分发）

7 个第三方训练的 COCO 检测器（DINOv3 骨干 + 第三方检测头）经合规重打包，逐位忠实性对拍后存于**仓库之外**（默认 `~/Research/Model/AutoVision`，`AUTOVISION_USER_HELD_ROOT` 覆盖）；不进可下载清单、不随仓库或导出产物分发。**禁再分发、禁商用**，使用即视为已自行接受相应条款。其中 3 个 ViT 资产为 STA-era（含 `backbone.sta.*`），DEIMv2 派生 spatial prior 删除后不再受支持；4 个 ConvNeXt 资产不受影响。

| 资产 | 加载 model 键 | 构建覆盖（加载必需） | weights sha256 |
|---|---|---|---|
| `dinov3-vitt16` | `detr/dinov3-vitt16-notpretrained` | 不可用（STA 已删除；需重训 no-STA 检查点） | `e7153c95379c` |
| `dinov3-vitt16plus` | `detr/dinov3-vitt16plus-notpretrained` | 不可用（STA 已删除；需重训 no-STA 检查点） | `96603bff98fb` |
| `dinov3-vits16` | `detr/dinov3-vits16-notpretrained` | 不可用（STA 已删除；需重训 no-STA 检查点） | `9c38f900d1a5` |
| `dinov3-convnext-tiny` | `detr/dinov3-convnext-tiny` | `dec_layers=6` + `decoder_hidden_dim=256` | `2925ef6afacf` |
| `dinov3-convnext-small` | `detr/dinov3-convnext-small` | `dec_layers=6` + `decoder_hidden_dim=256` | `72e37da6bdfd` |
| `dinov3-convnext-base` | `detr/dinov3-convnext-base` | `dec_layers=6` + `decoder_hidden_dim=256` | `503eaea85c85` |
| `dinov3-convnext-large` | `detr/dinov3-convnext-large` | `dec_layers=6` + `decoder_hidden_dim=256` | `7e617f3fbf69` |

```python
from autovision.registry.user_held_assets import user_held_model_args

autovision.train_object_detection(out="runs/d1", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args=user_held_model_args("dinov3-convnext-base"))
```

## 7. 门控与环境变量

| 变量 | 作用 |
|---|---|
| `AUTOVISION_DINOV2_WEIGHTS=1` | 显式接受 DINOv2 权重条款（CC-BY-NC）后允许下载 |
| `AUTOVISION_DINOV3_WEIGHTS=1` | 显式接受 DINOv3 License 后允许下载 |
| `AUTOVISION_GTR_ROOT` | GTR 检查点目录（`scripts/download_gtr_assets.py` 写入；默认 `~/Research/Model/gtr`） |
| `AUTOVISION_USER_HELD_ROOT` | 用户自持库目录（默认 `~/Research/Model/AutoVision`） |
| `AUTOVISION_OFFLINE=1` | 禁止联网（只用本地资产） |
| `AUTOVISION_MIRROR_URL` / `HF_ENDPOINT` | 下载镜像端点 |

框架级环境变量总表见 `src/autovision/runtime/env.py`（`AUTOVISION_*`，遥测默认关闭）；上表的权重门控与资产根变量定义在各自的注册表模块（`encoders/dinov2|dinov3/`、`registry/gtr_assets.py`、`registry/user_held_assets.py`）。
