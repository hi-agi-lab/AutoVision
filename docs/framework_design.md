# AutoVision 架构与设计总览（as-built）

| 文档属性 | 内容 |
|---|---|
| 版本 | v4.0（as-built 总览） |
| 状态 | v0.3.0 发布基线 |
| 日期 | 2026-10-19 |
| 读者 | 框架研发、算法团队、平台 / MLOps、安全合规 |
| 关联 | [模型库](model_zoo.md)、[基准](benchmarks.md)、[发布流程](release_process.md)、[贡献指南](contributing.md) |

本文档描述**建成后**（as-built）的系统结构与工程契约，以代码为唯一事实来源
（模块路径与门禁脚本为准）。用户面的用法看 [首页](index.md)；早期 v2/v3
设计蓝图分册（按子系统拆分的历史文档）不随公开树分发，其中的运行时行为以
本总览与代码 docstring 为准。

## 1. 分层契约（机器强制）

`autovision` 包按单向依赖分层，由 `scripts/check_layering.py` 以无依赖的
AST 静态扫描在 CI 强制（数字越小越高层，高层可依赖低层）：

| 层（rank） | 包 |
|---|---|
| 0 | `cli` |
| 1 | `engine`（训练引擎）、`hpo`（超参搜索驱动）、`inference`（参考推理运行时）、`serving`（参考推理 serving） |
| 2 | `pretext`（自监督 / 蒸馏方法）、`tasks`（任务层） |
| 3 | `augment`、`features` |
| 4 | `encoders`（模型层） |
| 5 | `data`、`optim`、`metrics`、`config`、`registry`、`scaling`、`observability`、`export`、`debug`、`visualize`、`tracking` |
| 6 | `errors`、`geometry` 等基础单模块（叶层） |

**两条硬禁**（不因 rank 豁免）：

- `tasks` 不得 import `pretext`（蒸馏教师注入走协议，不走直接依赖）；
- `encoders` 不得 import 其上的任何框架层（模型层必须自包含，`tasks`/`pretext`/`engine`/`cli` 全禁）。

**`ALLOWED_UPWARD` 债务表**：个别经批准的反向 import 登记在门禁脚本内
（附理由），是待收敛的技术债清单而非白名单机制——新增条目需评审。

## 2. 模块地图

| 包 / 模块 | 职责 | 关键入口 |
|---|---|---|
| `cli` | 30 条命令；参数校验与 Python API 完全同源 | `cli/_app.py` |
| `engine` | step 循环、DDP / AMP、原子 checkpoint + resume、回调编排（`Engine` / `TaskEngine` 双引擎） | `engine/engine.py` |
| `hpo` | 超参搜索：空间规格（分布 / 网格）、随机 / 网格 / ASHA 驱动、报告与断点复用 | `hpo/space.py`、`hpo/search.py`、`hpo/asha.py` |
| `inference` | 视频 / 流推理运行时：三段式 reader / infer / writer 管线、batch 聚合、后端切换（torch / ONNX / TRT / OpenVINO）、同源前后处理；单路跟踪 / 区域计数 / 平滑 / 多路流共享批处理；统一结果容器 `InferenceResult`（mask 默认 RLE）；逐帧 `process_video` / `VideoSink` / `ImageSink` / `FPSMonitor` | `autovision/results.py`、`inference/stream.py`、`inference/multi_stream.py`、`inference/video.py` |
| `tracking` | 多目标跟踪：ByteTrack 关联（卡尔曼 + 三阶段 IoU 与官方门限，class-aware）；LineZone / PolygonZone 计数与 TrackSmoother | `tracking/byte_track.py`、`tracking/analytics.py` |
| `serving` | 参考推理 serving：FastAPI `/predict`（批处理，mask 默认 RLE）+ `/healthz` + `/readyz` + `/drift` + Prometheus `/metrics`；TLS/多 worker/限流/A-B 分流 | `serving/app.py`、`serving/server.py`、`serving/drift.py`、`serving/ratelimit.py` |
| `pretext` | 自监督预训练与蒸馏方法（method 契约 + 注册表） | `pretext/`（DistillationV3、DINO(v2/v3)、SimCLR、DenseCL、DetCon） |
| `tasks` | 8 个任务面；TaskModel / TrainModel 双层契约、损失、训练配方；伪标注 `tasks/detection/shared/pseudo_label.py` | `tasks/<task>/`，检测谱系 `tasks/detection/detr/variants.py` |
| `augment` | 检测增广全套（mosaic / mixup / copyblend / box-aware eraser）+ 逐 step 窗口调度；多视图协议；mosaic 几何共享模块（关键点感知 pose mosaic 复用） | `augment/detection/`、`augment/mosaic_geometry.py` |
| `features` | 功能面（embed） | `features/embed.py` |
| `encoders` | 包装协议、包注册表（7 个注册源）、权重清单（revision + sha256 + 许可） | `encoders/weight_hashes.py` |
| `data` | 11 类数据集、mmap 文件清单（缓存指纹 / 引用计数 / 原子发布）、数据下载与转换（YOLO / COCO / VOC / YOLO-pose / YOLO-seg、CVAT / Label Studio、COCO 合并）、内容寻址数据集快照与漂移校验、数据体检、难例/长尾采样 | `data/`、`data/versioning.py`、`data/quality.py`、`data/interchange/` |
| `optim` | 优化器 / 调度器 / 参数组（含 8-bit AdamW） | `optim/optimizer_args.py` |
| `metrics` | 任务指标（COCO AP + `map_small/medium/large` 面积分档）+ 跨 rank 状态汇聚（DDP 验证）；误差分析（匹配 / 归因 / 画廊 / hard_examples） | `metrics/map_evaluator.py`、`metrics/error_analysis.py` |
| `config` | Pydantic 严格校验配置（API / CLI / YAML 三方言同源） | `config/` |
| `registry` | 资产注册表（HF 权重 / GTR / 自持）+ 模型注册表（dev / staging / prod + current/rollback/audit + 部署门禁 + MLflow + model card）+ 任务级模型目录 | `registry/model_registry.py`、`registry/deployment_gates.py`、`registry/catalog.py` |
| `plugins` | entry-point 插件协议（model_source/dataset/metric/loss/task/postprocess）、API 版本校验、发现错误收集 | `plugins.py` |
| `compliance` | 企业合规包：依赖许可清单、模型卡/sha256、数据快照、JSON/Markdown 报告 | `compliance/` |
| `scaling` | 规模策略（batch `auto` 解析等） | `scaling.py` |
| `observability` | JSONL / TensorBoard / MLflow / W&B 记录器、OTel（opt-in，遥测默认关闭） | `observability/` |
| `export` | ONNX / TensorRT / OpenVINO 导出管线（对拍门禁）、int8 QDQ、版本化后处理规格 + 端到端框级 parity、benchmark、参数量 / FLOPs 统计 | `export/` |
| `debug` | NaN 捕获、underflow / overflow 检查 | `debug/nan_capture.py` |
| `visualize` | 任务感知渲染：检测框 / mask / 旋转框 / 姿态骨架 / 语义与全景分割 / 深度；统一 `InferenceResult` 分派与区域叠加；隐私/轨迹/热力/角标/光晕等增强标注器；COCO/GHUM 与自定义骨架注册表 | `visualize/annotators.py`、`visualize/effects.py`、`visualize/skeletons.py`、`visualize/detection.py`、`visualize/zones.py` |
| `geometry`、`errors` | 几何工具与异常类型（叶层）：可微 torch 框运算 + 推理侧 NumPy 框/mask/polygon 工具（IoU/NMS/NMM、坐标转换、RDP 简化） | `geometry/`、`errors.py` |

### 2.1 supervision 能力映射

[supervision](https://github.com/roboflow/supervision) 0.30（MIT）是推理侧
可视化与视频工具的重要参考。AutoVision 只迁移与本仓库契约一致的部分，不引入
第二个结果容器；下表是当前映射（“新增”指本轮迁移）：

| supervision 能力 | AutoVision 对应 | 状态 |
|---|---|---|
| `Detections` / 各类 `from_*` 转换器 | `InferenceResult` + `data/convert.py`、`data/interchange/` | 已有（容器不同） |
| Box / Label / Mask / Polygon 标注器 | `visualize/annotators.py` | 已有 |
| `KeyPoints`、`Skeleton` | `visualize/skeletons.py` + `KeypointAnnotator(skeleton=...)` | 新增注册表/自定义骨架 |
| `TraceAnnotator` / `HeatMapAnnotator` | `visualize/effects.py` | 新增 |
| `BlurAnnotator` / `PixelateAnnotator` | `visualize/effects.py`（mask 感知） | 新增 |
| `BoxCornerAnnotator` / `RoundBoxAnnotator` | `visualize/effects.py` | 新增 |
| `HaloAnnotator` / `EdgeAnnotator` / `BackgroundOverlayAnnotator` | `visualize/effects.py` | 新增 |
| `box_iou` / NMS / NMM / 框坐标转换 | `geometry/boxes.py` | 新增 |
| `mask_to_xyxy` / `polygon_to_mask` / `mask_to_polygons` / RDP | `geometry/masks.py` | 新增 |
| `process_video` / `VideoSink` / `ImageSink` / `FPSMonitor` | `inference/video.py` | 新增（同步管线，复用 RGB 约定） |
| `LineZone` / `PolygonZone` / `ByteTrack` | `tracking/` + `visualize/zones.py` | 已有 |
| `InferenceSlicer` | `tasks/shared/sahi.py`、`tiling_utils.py` | 已有（模型 API 内） |
| mAP / ConfusionMatrix | `metrics/` | 已有 |
| 数据集 load/save | `data/convert.py`、`data/interchange/`、各任务 Dataset | 已有（接口按训练侧设计） |

**刻意不迁移**：`sv.Detections` 作为第二套结果对象（会与 `InferenceResult`
重复，违反单一容器契约）；`IconAnnotator` / `ComparisonAnnotator` 等依赖外部
图标资产或双图拼接的展示型功能（需要额外资产登记，暂不进入核心路径）。
`mask_to_polygons` 惰性依赖 OpenCV（`video` extra），其余新增工具只依赖
NumPy/Pillow。

## 3. 任务面

| 任务 | model 键（示例） | 指标 |
|---|---|---|
| 检测 | `rtdetr/*`、`detr/*`（39 变体）、`dfine/*`、`picodet/*`、`rfdetr/*` | COCO AP |
| 旋转框 | `rtdetr-obb/*`、`gtr/*` | AP50（DOTA 口径） |
| 实例分割 | `gtr/*`、mask 系列 | mask AP |
| 全景分割 | panoptic 系列 | PQ |
| 语义分割 | `gtr/*` 等 | mIoU |
| 姿态 | `gtr/*`（COCO 关键点） | keypoint AP（OKS） |
| 深度 | `gtr/*` | absRel 等深度指标 |
| 分类 | `torchvision/*`、`timm/*` 等 | top-1 / top-5 |

检测档位谱系（`detr/n` … `detr/x`）的产品冻结状态与方向决策机器可读地记录在
`tasks/detection/detr/variants.py` 的 `SPECTRUM` 表（`planned` / `spec-frozen` / `released`）。

## 4. 权重与许可模型

仓库**不携带权重二进制**；一切权重经三个注册表钉身份（sha256），许可边界
逐条登记（完整表格见[模型库](model_zoo.md)）：

- **HF 权重清单**（`encoders/weight_hashes.py`，31 个资产）：revision + 每文件
  sha256 + 许可；CC-BY-NC / 自定义 License 默认不下载，需显式自证
  （`AUTOVISION_DINOV2_WEIGHTS` / `AUTOVISION_DINOV3_WEIGHTS`）。
- **GTR 家族**（`registry/gtr_assets.py`，26 个官方权重，MIT）：整族 sha256
  钉住，`resolve_gtr_asset` 解析、`verify_gtr_asset` 离线核验。
- **用户自持库**（`registry/user_held_assets.py`，7 个检测器）：仓库外存储、
  不可再分发、不注册进可下载清单（`redistributable=False`）。

AGPL 组件（`ultralytics`）与超许可依赖策略一致：仅 extras 隔离加载，不进核心
依赖、不在模块级 import。全仓禁词与许可扫描见 §5 门禁。

## 5. 质量门禁与测试策略

CI（GitHub Actions / GitLab CI 同款）逐项强制：

| 门禁 | 命令 | 保证 |
|---|---|---|
| 测试 | `pytest tests -q` | 功能与回归（含金标准快照） |
| lint | `ruff check src tests scripts` | pycodestyle / pyflakes / import 排序 |
| 溯源 | `python scripts/check_provenance.py` | 禁词零命中 + allowlist 卫生 |
| 分层 | `python scripts/check_layering.py` | §1 的依赖契约 |
| 断链 | `python scripts/check_docs_links.py` | 公开树相对链接全部可解析 |
| 文档构建 | `mkdocs build --strict` | 站点零警告（导航 / 站内链接） |
| 模型库一致性 | `scripts/gen_model_zoo.py --check` | 模型库页面与注册表零漂移 |

**测试策略**：单测覆盖行为；数值类改动一律对拍（与官方实现 bit-equal 优先 /
独立 oracle / 数值容差），不满足于「跑通」；金标准快照（`tests/golden/`）在
有意配方变更后重录并在 PR 说明；需网络 / GPU 的测试用门控环境变量
（`AUTOVISION_HF_TESTS` / `AUTOVISION_DDP_TESTS`），默认路径 CPU-only。

**数值一致性承诺**（导出与加载面）：

- ONNX fp32 / fp16 导出默认执行 torch ↔ onnxruntime 数值对拍门禁，超容差即
  拒绝产出；TensorRT 后端纳入 benchmark parity 对比。
- 官方权重直载做加载对拍（张量逐位 / 前向逐位，按资产能力）；重打包容器附
  逐位忠实性对拍。
- GLA 骨干（`detr/gtrvit-*` 与 `gtr/*`）当前不可导出：导出入口以明确报错
  拒绝（自定义 CUDA 核无 ONNX 降级），见 `GTRViTBackbone.export_blocker`。
- **动态 H/W**（`dynamic_spatial=True`）仅在 dynamo 导出路径可用（fail-closed：
  legacy 图会把尺寸烘成常量），并内置"第二尺寸 parity 探针"；HF 骨干
  （`detr/dinov2-*`、`detr/dinov3-*`）在上游 transformers 的输入尺寸分支上
  不可符号化，以可读错误拒绝——支持面 = torchvision/CNN 骨干与 HGNetV2 谱系。
- 静态导出在严格 trace 失败时回退 legacy TorchScript（记录告警，`verify`
  门禁照常执行）。

## 6. 事实来源与维护

- 用户面用法：README（仓库根）与[首页](index.md)；模型与许可：[模型库](model_zoo.md)；
  数字：[基准](benchmarks.md)（协议不可变）。
- 行为细节：代码 docstring 与注册表定义；架构决策记录在提交历史与代码注释
  （如 `SPECTRUM`、`export_blocker`、`ALLOWED_UPWARD`）。
- 本文件与站点导航的一致性由 `tests/test_infrastructure.py` 与
  `mkdocs build --strict` 强制。
