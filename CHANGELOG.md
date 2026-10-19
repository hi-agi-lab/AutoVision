# 变更日志

本文件记录 AutoVision 的对外变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本语义遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)；发布流程与版本政策见
`docs/release_process.md`（站点页「发布流程」）。

`0.1.x` 为内部开发线（未对外发布，无变更记录）；本文件自 `0.2.0` 起对外。

## [Unreleased]

（下一版本开发中。）

## [0.3.2] - 2026-10-19

### 文档

- **PyPI 页面移除 0.3.0 licensing status 说明段**（该信息保留在仓库
  `CHANGELOG.md` 与 `docs/provenance/LEDGER.md`，不再出现在发行页正文）。

## [0.3.1] - 2026-10-19

### 文档

- **PyPI 页面改为英文**：`README_PYPI.md`（发行包长描述）整体改写为英文；
  包内中文 README（`README.md` / `README.en.md`）不变，仓库仍为双语入口。

## [0.3.0] - 2026-10-19

### 许可与兼容性

- **删除 DEIMv2 派生的 STA（2026-10-17）**：`SpatialPriorModule` 从代码库
  移除，`DETRVariantSpec` 不再携带 `use_sta` / `conv_inplane`；DINOv3 ViT
  变体保留并改用 no-STA adapter，DINOv3 ConvNeXt 与其他家族不变。发行物
  不再包含 DEIMv2 代码，许可阻塞解除。
- **旧 STA-era ViT 检查点不再兼容**：含 `backbone.sta.*` 的基线 `.pt` 与
  用户自持的 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 会被
  加载器显式拒绝；需使用重新训练/导出的 no-STA 检查点，或改用 ConvNeXt
  资产。用户自持的 4 个 ConvNeXt 资产不受影响。

### 新增

- **许可与致谢文档**：README 新增 `致谢（Acknowledgements）`；
  新增 `THIRD_PARTY_LICENSES.md`（GTR MIT 全文）与
  `docs/provenance/LEDGER.md`；`NOTICE` 补齐 RF-DETR / PicoDet /
  ByteTrack / SAHI 归属，并记录 DEIMv2 派生 STA 的删除与兼容性变化。
- **发布依赖**：`export` extra 下限提升为 `numpy>=1.25`（onnx / onnxscript
  在导入时使用 `numpy.exceptions`，numpy 1.23/1.24 下会直接失败）；核心训练
  安装仍保持 `numpy>=1.23`，与 super-gradients 的互斥由既有 extras 冲突声明
  处理。

- **黄金路径 / 部署契约（roadmap phase 1–2 代码基础）**：
  - **任务级模型目录**：`autovision.registry.catalog` 汇总 RF-DETR N/S/M/L、
    GTR s/m/l/x、自研 DETR 谱系与 RT-DETR 的许可 / 权重 / AP / 可导出状态；
    `list-models --task detection --pretrained`、`model-info <key>` 可查。
  - **预训练检测句柄**：`autovision.detection(<key>, pretrained=True)` /
    `DetectionModel.from_pretrained` 惰性加载发布权重（RF-DETR 全链；GTR 在
    GLA/CUDA 可用时走官方 EMA 权重装配，否则给出可读安装/硬件指引）；
    `predict / val / export` 不再要求先训练。
  - **训练配方**：`recipe=finetune | coco-finetune | low-data` 或 YAML 路径，
    只覆盖 `model_args` / `transform_args` / `trainer_args`，显式参数优先。
  - **版本化后处理规格**：ONNX / OpenVINO / TensorRT 导出件旁写
    `<artifact>.postprocess.json`（schema_version=1；top-k、阈值、原图映射、
    resize/mean/std），附带 NumPy 参考解码器；分割 / 姿态 / 深度 / 分类导出
    不写 sidecar。
  - **端到端框级 parity**：`autovision.export.verify_end_to_end_parity`
    比较最终框 / 分数 / 类别（宽松匹配 + 像素容差 + missing/extra 计数），
    可直接输出 JSON 报告；不再只比张量求和。
  - **OpenVINO 运行时**：`InferenceEngine(backend="openvino")`、`run_stream`、
    serving 与 CLI `predict` 均支持 `.xml` IR（`openvino_device=CPU|GPU|AUTO`）。
  - 修复 legacy ONNX 导出回退时多传一个位置参数导致的
    `TypeError`（dynamo 不可用时的真正回退路径）。
  - 新增测试：模型目录、预训练句柄、配方（含 1 步真实训练）、后处理规格、
    框级 parity、OpenVINO 运行时接线。

- **数据体检与难例闭环（phase 3 代码基础）**：
  - **数据体检**：`autovision.data.quality` + CLI `data-check`：坏图、精确/近重复
    图、跨 split 泄漏、框越界/非法尺寸、原始类别 id、类别分布与 val 缺类；
    产出 `report.json` / `report.md` / `manifest.json`，`fail_on` 控制退出码。
  - **难例清单**：`analyze-errors` 额外写 `hard_examples.json`（逐图 TP/FP/FN、
    FP 归因、漏检面积）；`split=` 记录来源，采样器只消费 `train`，避免验证集
    泄漏进训练。
  - **难例采样**：`autovision.data.hard_examples` 的 manifest + 确定性
    `WeightedRandomSampler`；`train_object_detection(..., hard_examples=...)`
    按 `min(max_repeat, 1 + alpha × 归一化误差)` 加权。v1 单机；DDP 组合被
    显式拒绝（DistributedSampler 会覆盖权重）。
  - **多轮自训练**：`scripts/self_train_detect.py --rounds N --min-improvement`；
    逐轮产物（`rounds/r{i}/`）、学生提升后升级为下一轮教师、不升即停；
    顶层 `summary.json` / `summary.md` 记录 stop_reason 与最优模型。
    `--rounds 1` 保持原有单轮布局。
  - **长尾采样**：`autovision.data.samplers` 的 `class_balanced` /
    `repeat_factor`；`train_object_detection(..., long_tail_sampler=...)`
    接入训练，与 `hard_examples` 互斥，v1 单机。
  - **标注工具互通**：`autovision.data.interchange` 的 CVAT XML（bbox /
    polygon）与 Label Studio JSON（rectangle / polygon / keypoint-as-box）→
    COCO；`convert-data from=cvat | label-studio`，含 CLI 与 loader 往返测试。
  - 新增测试：数据体检（核心 + CLI 夹具）、难例 manifest/采样/误差分析输出/
    1 步真实训练接入、多轮自训练（真实 2 轮）、长尾采样权重/确定性/训练接入、
    CVAT / Label Studio 转换。

- **生态与企业能力（phase 4 代码基础）**：
  - **插件协议**：`autovision.plugins` 的 `autovision.plugins` entry-point
    发现、`PluginInfo` / `PLUGIN_API_VERSION`、kind 校验、失败收集 /
    strict 模式；`model_source` 自动接入编码器目录；CLI
    `plugins action=list|check`；`examples/plugin-demo` 可安装模板。
  - **企业合规报告**：`autovision.compliance` 汇总框架/依赖许可/模型卡+
    sha256/数据快照；CLI `compliance-report` 输出 `compliance.json` +
    `compliance.md`，支持 `strict` / `no_timestamp`。
  - **私有注册表**：`current` 指针、`rollback`、append-only `audit.jsonl`、
    模型卡/后处理边车随产物复制；部署门禁
    `Sha256Gate` / `ModelCardGate` / `LicenseGate` / `BenchmarkGate` /
    `ParityGate` / `DataSnapshotGate`；CLI
    `registry action=register|promote|rollback|current|list|audit`。
  - **LTS / SLA 文档**：`docs/lts_policy.md`、`docs/support_policy.md`
    （6 个月 LTS 节奏、12/6 个月支持窗、P1–P3 响应目标）。
  - **硬件伙伴包模板**：`deploy/packs/_template/` 的 `convert.py`（校验
    ONNX/spec、执行厂商命令）与 `parity.py`（目标端 JSONL 对拍）；一伙伴
    一包，不预研厂商矩阵。

- **谱系发布治理（P0）**：
  - `SPECTRUM` 档位状态改为三态 `planned / spec-frozen / released`；
    `detr/n`、`detr/s` 明确为 spec-frozen（规格冻结，预训练权重 / AP 待发布）。
  - 新增 `autovision.registry.spectrum_release` 发布门禁：`released` 必须有
    已发布资产、模型卡（许可 + sha256）、训练 seed、全量 COCO AP、延迟记录
    与存在的复现脚本。
  - 新增 `scripts/train_spectrum.py`、`scripts/eval_spectrum_coco.py` 骨架
    （支持 `--dry-run` 与 coco8 smoke 路径）。
  - 新增 RF-DETR 五档模型卡（官方 COCO 指标 + pinned sha256，
    `autovision.registry.model_cards`；Base 标记上游弃用）；基准表补
    RF-DETR 官方 T4 口径（N/S/M/L）。
- **数据闭环（P1）**：
  - 伪标注 / 自训练：`autovision.tasks.detection.shared.pseudo_label` +
    `scripts/self_train_detect.py`（教师 `.pt` → 伪标注 → 学生训练 → 标注 val 评测
    → 报告），CLI `pseudo-label`（逐类阈值、SAHI 可选、批量推理）。
  - 误差分析：`autovision.metrics.error_analysis` + CLI `analyze-errors`（在线 /
    离线两模式）：TP/FP/FN 匹配、FP 归因（duplicate / classification /
    localization / background）、按类别与 COCO 面积分档切片、top-FP/FN 画廊
    （`report.md` / `report.json`）。
  - 转换扩展：PASCAL VOC 解析器、YOLO-pose → COCO 关键点、YOLO-seg 多边形 →
    COCO 分割、COCO 合并（`convert-data from=voc | yolo-pose | yolo-seg | merge`）。
  - pose 增广：关键点感知 mosaic + mixup（默认关闭、步窗口可调），mosaic 几何与
    检测族共享同一实现。
- 检测 mAP 指标新增 `map_small` / `map_medium` / `map_large`（COCO 面积分档，
  此前固定 `"all"`；headline 数字不变）。

- **部署广度（P1）**：
  - ONNX 导出支持**动态 H/W**（`export_onnx(dynamic_spatial=True)`，含第二尺寸
    torch↔ONNXRuntime parity 门禁；`input_dynamic_shapes` 规格位被导出器消费），
    并要求 dynamo 导出器（legacy 路径会把尺寸烘成常量——fail-closed）。
  - **OpenVINO** 一等后端：`export_openvino`（ONNX→IR + 编译后对拍门禁 +
    metadata 边车）、`export out=model.xml`、benchmark 后端
    `openvino`（设备可切 CPU/GPU/AUTO，纳入 parity 门禁）。
  - **TensorRT int8**（QDQ 路线）：`export_tensorrt(precision="int8",
    calibration_data=...)`（QDQ ONNX 先过 int8↔fp32 门禁，再以 strongly-typed
    网络构建引擎），benchmark 的 `tensorrt` 后端支持 `precision="int8"` +
    `quantization={...}`（首层 fp32 配方用 `nodes_to_exclude`）。
  - **多路流运行时**：`MultiStreamRunner` / `run_multi_stream`（`predict
    sources=...`）：每路 reader 线程 + 单一共享 batcher、每路
    `predictions.jsonl` / `annotated.mp4`、断流指数退避重连、队列
    `drop_oldest|block` 背压、`summary.json` 每路帧数/丢弃/重连/错误。
    v1 不做跨路跟踪。
  - **PicoDet NMS 后处理规格**：`postprocess.json` 新增 `family="picodet"` +
    class-offset NMS；NumPy 参考解码与 torch 后处理逐框对拍（含重复框抑制）。
  - **int8 vs fp32 AP 报告**：`scripts/measure_int8_accuracy.py` +
    `autovision.export.int8_report`：同一 split/前后处理下导出 fp32/int8、
    跑全量 COCO AP、输出 `report.json`/`report.md`（delta、门禁、校准集、
    硬件）；`--max-ap-drop` 控制退出码。
  - **C++ ONNX Runtime 示例**：`examples/cpp/`（header-only decode + host
    test + ORT/OpenCV main + CMake）；host 测试由 g++ 真实编译运行，
    完整 runtime 构建依赖 ORT/OpenCV 开发库。
- int8 QDQ 链与 TensorRT 的互操作修复：int32 常量 `DequantizeLinear` 折叠回
  fp32（TRT 11 解析器要求）；激活量化默认**对称**（`activation_symmetric`，
  非对称 zero-point 被 TRT 拒绝）。
- `export` extra 显式声明 `onnxscript`（dynamo 导出器的依赖，缺失时静默回退
  legacy）；新增 `openvino` extra。
- **int8 首层 fp32 配方自动建议**：`suggest_first_layer_nodes`（导出自
  `autovision.export`）枚举 fp32 图的前导 Conv 节点；TensorRT int8 构建失败
  的提示直接给出可粘贴的 `nodes_to_exclude` 配方，复现脚本
  `scripts/measure_new_backends.py --quantization-exclude auto` 一键套用；
  OpenVINO IR 支持 `precision="fp16"`（编译后对拍门禁照常）。

- **超参搜索（B2）**：`autovision.hpo`（`SearchSpace` 分布规格 + `run_search`
  随机 / 网格 / ASHA 驱动 + `task_trial` 复用训练入口与 Pydantic 覆盖面）与
  CLI `hpo`；产出 `report.md` + `trials.jsonl` + `best.json`，已完成试验目录
  自动复用（断点续跑），ASHA 逐 rung 以框架原生 resume 续训而非重训。
- **VOC 可直接训练（B3）**：`download → convert-data from=voc → train` 三步配方
  入 README；转换器保持 VOC 规范类名（排序 id）、`difficult → iscrowd` ignore
  区；端到端训练测试（CPU）在案。

- **推理侧能力（P2 战略项的一部分）**：
  - **ByteTrack 跟踪**（`autovision.tracking`，A3）：卡尔曼运动模型 + 三阶段
    IoU 关联（高分池融合检测分数 / 低分池 / 未确认轨迹），class-aware；官方
    门限齐备（新轨迹需 `track_thresh + 0.1` 且连续命中才发布，低分下限 0.1，
    首帧激活例外）；`predict source=... track=true` → JSONL 逐帧 `tracks`
    （`age` = 创建以来帧数）+ 渲染 id 标注，tracker 按源视频真实帧率折算
    丢失窗口。
  - **参考 serving**（`autovision.serving`，F2；`serving` extra）：FastAPI
    `/predict`（批处理按输入顺序、与评测同源前后处理）+ `/healthz` +
    Prometheus `/metrics`（请求按端点与状态、图片数、延迟 summary + 峰值；
    5xx 同样计数）；显式边界（无鉴权/TLS/编排，漂移检测等按产品触发）。
  - **数据版本化**（`autovision.data.versioning` + `data-snapshot` CLI）：内容
    寻址快照（根相对清单 sha256，`deep=true` 含内容哈希）+ 漂移报告
    （added/removed/resized/content_digest）+ 搬迁副本校验；spec 内相对路径按
    spec 文件目录解析，写入默认不覆盖（`overwrite=true`），deep 校验要求
    deep 快照（浅快照显式报错）。

### 变更

- **PyPI 打包**：发行名改为 `autovision-framework`（import 名仍 `autovision`，
  CLI 仍 `autovision-cli`）；新增 PyPI 专用 README、PEP 639 许可元数据、
  sdist 的 GTR / MSDA CUDA-C++ 源清单，以及 tag 触发的 trusted-publishing
  CI job。`super-gradients` 与 `export` / `transformers` / `rfdetr` / `video`
  的 extras 互斥已显式声明。
- **STA-era 用户自持资产**：`resolve_user_held_asset` 默认在 sha256 校验前
  拒绝 3 个含 `backbone.sta.*` 的 ViT 资产；新增
  `USER_HELD_LOADABLE_ASSETS` / `USER_HELD_UNSUPPORTED_ASSETS` 区分可加载与
  仅审计记录，错误消息列出 4 个可加载 ConvNeXt 替代项。审计场景需显式传
  `allow_unsupported=True`；`verify_user_held_asset` 仍可离线核验 identity。

### 修复

- **分类 / 线性语义分割的 `auto` 默认值取错**：`default_batch_size` /
  `default_steps` 此前声明为 Pydantic 字段，但 CLI 解析 `auto` 时读取类属性，
  实际回退到父类的 16 / 10000（分类应为 128，线性语义分割应为 32 / 20000）。
  现改为与其它任务一致的 `ClassVar`，并加回归测试。
- **HF 骨干（DINOv2/DINOv3 系）的 ONNX 导出在 transformers ≥5 环境下全线失败**：
  首个 eager 前向会先安装 output-capturing hooks（在 `threading.Lock` 下），
  而 dynamo 无法 trace `with <lock>`，严格导出直接失败——现于 trace 前做一次
  warm-up 前向（数值透明）。同批修复：严格 trace 失败且空间轴为**静态**时回退
  legacy TorchScript 导出（此前不回退、直接抛 dynamo 内部错误——文档承诺的
  回退路径此前只覆盖第二步之后）；动态轴失败改为**可读的 AutoVisionError**
  （含"模型在输入尺寸上有 Python 级分支/守卫"的处置指引）。token→patch 网格的
  折叠改由**输入尺寸**推导（`H // patch`），动态 H/W 下不再对符号形状调
  `math.isqrt`：HF/ViT 线的 eager 报错与既有行为一致；timm 包装器在 patch size
  可发现时按输入推导网格（`dynamic_img_size` 类模型由"报错"变为可用，token 数
  与网格不匹配仍报错）。
- int8 首层 fp32 配方此前需手工枚举 fp32 图节点名（见上「自动建议」）。

- CLI 训练入口的显式设备串（`devices="cpu"` / `"cuda:1"`）此前在 9 个任务入口以
  `int(devices)` 崩溃——现按单设备解析（与 `engine.resolve_device` 语义一致）。
- `convert-data` 的文档示例此前无法执行（`out=` / `images=` / `annotations=` 与
  函数参数名不匹配、连字符键直接报错）——已修并加回归测试。
- 新版 CLI 键名统一支持 `--kebab-case`（自动映射到下划线参数名）。
- **步窗口增强在 `num_workers>0` 下静默不生效**（先存缺陷，检测 / 旋转框 / 姿态三族同存）：
  含步窗口调度的训练 loader 现以非持久 worker 构建，窗口边界跨越时把当前 step
  重新下发到 worker。

### 视觉增强与视频工具（迁移自 supervision 0.30）

#### Added

- **增强标注器**（`autovision.visualize.effects`）：`TraceAnnotator`（按
  `tracker_id` 累积轨迹）、`HeatMapAnnotator`（时序热力累积）、
  `BlurAnnotator` / `PixelateAnnotator`（隐私模糊/马赛克）、
  `BoxCornerAnnotator` / `RoundBoxAnnotator`（角标/圆角框）、
  `HaloAnnotator` / `EdgeAnnotator`（mask 光晕/轮廓）、
  `BackgroundOverlayAnnotator`（mask 外压暗）。全部为 Pillow + NumPy
  实现，mask 存在时优先按像素处理。
- **骨架注册表**（`autovision.visualize.skeletons`）：COCO-17 与 GHUM-33
  边表；`KeypointAnnotator(skeleton=...)` 支持注册名或自定义零基边序列，
  默认行为保持 COCO-17 不变。
- **检测几何工具**（`autovision.geometry.boxes` / `autovision.geometry.masks`）：
  `clip_boxes` / `pad_boxes` / `denormalize_boxes` / `scale_boxes` /
  `move_boxes`、`xywh` / `xcycwh` / `xcycarh` 转换、`box_iou`、
  `box_non_max_suppression`、`box_non_max_merge`、`mask_to_xyxy` /
  `xyxy_to_mask`、`polygon_to_mask`、`mask_to_polygons`、
  `approximate_polygon`（NumPy RDP）、`filter_polygons_by_area`。
  `mask_to_polygons` 惰性依赖 OpenCV（`video` extra），其余仅需 NumPy/Pillow。
- **视频工具**（`autovision.inference.video`）：`get_video_frames_generator`、
  `process_video`（逐帧回调 + 可选编码）、`VideoSink`、`ImageSink`
  （按间隔落盘）、`FPSMonitor`。沿用 `InferenceResult` 的 RGB 通道约定，
  与 `run_stream` 的单路/多路管线并存。
- **测试**：`tests/test_geometry_ops.py`、`tests/test_visualize_effects.py`、
  `tests/test_video_utils.py` 覆盖转换、NMS/NMM、mask/polygon、隐私效果、
  轨迹/热力、骨架注册与视频 sink。
- **示例**：`examples/tasks/detection.py` 新增
  `--effect {none,corner,round,blur,pixelate}`，用上述标注器演示检测结果的
  角标/圆角框与隐私模糊/马赛克；默认 `none` 行为不变。
- **溯源**：`THIRD_PARTY_LICENSES.md`、`NOTICE`、`docs/provenance/LEDGER.md`
  登记 Roboflow supervision 0.30 的 MIT 参考关系；实现为 AutoVision 重写。

### 任务示例（examples/tasks）

- 为检测、旋转框、实例/全景/语义分割、姿态、深度、分类各新增一个可运行示例；
  默认生成 2–4 张图的合成数据并训练 1–3 步，支持 `--dry-run`、
  `--predict-image`、`--export`。
- `src/autovision/tasks/README.md` 提供任务到示例的索引；MkDocs 新增
  “任务示例”页面。
- **任务示例统一拆分**：每个任务提供 `<task>.py`（纯推理：RF-DETR/GTR 官方
  权重或 `--weights` 导出的 `.pt` → 预测 → 标注）与 `<task>_train.py`
  （训练/验证/导出）；`panoptic` / `classification` 无官方任务权重，推理需
  `--weights`。
- 新增 `examples/tasks/detection.py`（纯推理：加载 `rfdetr/rf-detr-nano`
  官方权重 → 预测 → 标注）与 `detection_train.py`（原 coco8 训练示例）；
  `inference_demo.py` 保留为兼容别名。
- 移除来源/许可未知的本地街景图；所有推理示例改用 CC0
  `examples/tasks/assets/airport_aerial.jpg`（San Diego 机场俯视图，
  Wikimedia Commons / Mertbiol），并随 sdist 发布。
- 新增 `examples/tasks/yolo/`：Ultralytics YOLO 检测 / 分割 / 姿态 / 旋转框 /
  分类 5 个推理示例（AGPL `ultralytics` extra，权重缓存到
  `~/.cache/autovision/yolo`，复用 AutoVision 的 `InferenceResult` 与标注器）。
  检测/分割/姿态权重默认从 ModelScope `Ultralytics/YOLO11` 下载。
- 新增 `tests/test_task_examples.py`：逐个校验 `--help` 与 `--dry-run`
  数据契约。

### 推理侧：统一结果、任务感知渲染与分析（借鉴 supervision 设计）

#### Added

- **`InferenceResult`**（`autovision.results`）：推理边界统一容器，
  覆盖检测 / 实例分割 / 旋转框 / 姿态 / 分类 / 语义 / 全景 / 深度；
  支持 `select` / `filter` / `to_numpy` / `to_record`，以及
  `to_json(masks="rle"|"dense"|"none", max_elements=...)`。
- **键常量**（`autovision.keys`）：`bboxes` / `scores` / `labels` /
  `masks` / `keypoints` / `polygons` / `mask` / `depth` / `segment_ids`。
- **任务感知渲染**（`autovision.visualize.annotators`）：Box / Label / Mask /
  Polygon(OBB) / Keypoint(COCO 骨架) / 分类 / 语义 / 全景 / 深度；
  `draw_prediction` 现在按结果字段分派，pose / OBB / 分类不再因缺少 `bboxes`
  抛 `TypeError`。
- **区域统计与平滑**（`autovision.tracking.analytics`）：`LineZone`（按类别
  in/out 穿越）、`PolygonZone`（当前区域内计数）、`TrackSmoother`（按轨迹
  滑动平均）；`run_stream(line_zones=..., polygon_zones=..., smooth=True)`
  将 `zones` 写入 JSONL 并叠加到渲染画面。
- **混淆矩阵**（`autovision.metrics.confusion`）：`analyze-errors` 的
  `report.md` / `report.json` 新增 `(类别 + FP) x (类别 + FN)` 矩阵，
  classification FP 落在 `[真值类, 预测类]`。
- **SAHI 合并策略**：`predict_sahi(..., batch_size=N, merge="nms"|"nmm",
  overlap_metric="iou"|"ios")`；tile 间先统一 per-label 过滤，再与全局结果
  合并；NMM 用并集框 + 面积加权分数。

#### Changed

- **单路视频管线**：`run_stream` 改为 reader / infer / writer 三段式
  （`prefetch` / `writer_buffer` 有界队列）；新增 `decode_s` / `infer_s` /
  `render_s` / `write_s` / `fps`；`StreamRunResult.zones` 汇总区域计数。
- **mask 序列化**：serving `/predict`、`predict_to_files`、stream JSONL 默认
  输出 COCO RLE；新增 `mask_format` / `max_json_elements`（后者约束图像级
  mask/depth 与扩展字段，实例行不截断）。
  实例分割 dense mask 不再被直接 `tolist()` 成数十 MB 的响应。
- **依赖**：核心依赖新增 `defusedxml>=0.7.1`；VOC / CVAT 标注 XML 改用
  `defusedxml` 解析，移除 `# nosec B314` 豁免。

#### Fixed

- `draw_prediction` 对 pose / OBB / 分类 / 深度结果的 `TypeError`；mask-only
  实例分割（EOMT）现在渲染 mask 与类别标签。
- `run_stream(line_zones=..., render=True)` 的 zone 叠加崩溃（PIL/NumPy 类型
  不匹配）；writer 失败或超时不再静默返回截断视频。
- `run_stream` 单路解码、推理、编码串行导致的吞吐损失；停止时保证 reader
  sentinel 与 VideoCapture 释放。
- 大图 SAHI 一次性堆叠全部 tile 的显存峰值风险；默认 NMS 走
  `torchvision.ops.nms`，NMM/IOS 为显式选项。
- 非有限浮点（NaN/Inf）在 serving / JSONL / predictions.json 中输出为
  `null`，严格 JSON 解析器不再失败。
- **RF-DETR 预训练类别映射**：官方权重是 91 类 COCO category-id 头
  （0=N/A，1..90 含缺口），此前按 80 类连续索引截断，导致预测标签与类名
  整体错位；现保持 91 类并为每个 id 返回正确类名（GTR/DETR 仍为 0..79
  连续头）。
- CVAT XML 的 `file_name` 路径穿越（缺失 width/height 时）；mask 的
  uint8/float 类型统一按非零 / `>0.5` 归一化。
- 全景分割的 ignore 像素不再被涂色；`filter_scores` 对全景结果不再抛错。

### 修复（企业级审计轮次）

- **DDP 正确性**：D-FINE / RT-DETR 系 criterion 不再对训练 loss 调用
  `reduce_dict`（它会 detach）；2-rank backward 恢复正常，并新增多 rank
  回归测试。日志 reduction 与反传图解耦。
- **训练引擎**：无验证集时 `steps_run` 正确计数；`gradient_accumulation_steps<1`
  在入口报错；每个 rank 调用 `set_device`；实验 logger 故障不再中断训练；
  `save_last=False` 不再写 last checkpoint；确定性模式下不再强制 TF32。
- **指标尺寸**：depth 验证统一在 GT canvas 上计算；instance segmentation 将
  GT mask 重采样到原图尺寸后再做 COCO mask AP，非方形/原图≠网络输入不再崩溃
  或误评分。
- **HPO**：trial 复用要求 `params + budget` 指纹 sidecar 匹配；空间或预算变化
  会重跑，ASHA rung 同样生效。
- **数据完整性**：下载改为临时文件 + 校验 + `os.replace`；缓存压缩包在解压前
  重新校验 size/sha256；zip/tar 解压拒绝路径穿越。
- **合规**：`normalize_license` 改为锚定别名 + AND/OR/WITH 表达式归一化，
  修复 BSD-2、Apache 全名、专有许可证被误判为 MIT 等问题；未知原子 fail-closed。
- **日志**：文件 handler 在多次 run 之间替换而非累积（HPO 不再互相污染）；
  非 rank0 不再构造 MLflow/W&B/TensorBoard run；`finalize_loggers` 隔离单后端故障。
- **serving**：`/docs`/`/openapi.json` 默认关闭；新增有界排队
  `queue_timeout_s`（饱和返回 503 + `Retry-After`）与
  `timeout_graceful_shutdown`；非 ASCII 凭据返回 401 而非 500。
- **导出**：TensorRT runner 检查 `set_input_shape` / `set_tensor_address` /
  `execute_async_v3` 返回值；`export_tensorrt(verify=True)` 默认执行
  eager-vs-engine 数值门禁，失败删除 engine。
- **模型加载**：`load_task_model` 恢复导出时保存的 `normalize_args`；
  `TaskNetwork` 不再把 `__class__` 写进 `init_args`。旧 teacher checkpoint
  的完整 unpickler 回退改为显式 opt-in
  （`AUTOVISION_TRUST_TEACHER_CHECKPOINT=1`）。
- **CI**：测试作业安装 `serving`/`video`/`transformers` 等测试依赖，
  修复 collection 阶段的 ModuleNotFoundError；新增 CI 依赖契约测试。

### 第二轮企业级加固

- **统一冒烟入口**：新增 `scripts/smoke_enterprise.py`（quick/full 两档）与
  `docs/smoke_test.md`，覆盖训练→ONNX/OpenVINO/TensorRT→serving→registry→
  compliance→数据快照→K8s→DDP/FSDP，并作为 CI 门禁；full 在本机 RTX 4090D
  全绿（TensorRT 使用最终框级 parity，int8 保持张量级门禁）。

- **训练规模**：新增 `trainer_args.gradient_checkpointing`（HF 原生 / 自定义
  ViT-GTR blocks）与 `parallel="fsdp"` 任务训练：full-state checkpoint/optimizer
  gather、所有 rank resume、rank0 导出、验证期 `summon_full_params`、EMA 按
  `AUTOVISION_FSDP_EMA_INTERVAL` 更新、rank0 写失败 barrier 与进程组超时。
  `docs/parallelism.md` 记录 DDP/FSDP 及 DeepSpeed/ZeRO/弹性训练的明确边界。
- **EoMT**：补 `forward`（ONNX 导出可用）；加载 hub 权重强制 DINOv2/DINOv3
  许可 opt-in，未钉 sha256 时默认拒绝（`AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1`
  显式放行）；`adapt_eomt_head` 已接入构建流程。
- **注册表**：`promote(gates=None)` 在文件系统和 MLflow 后端都默认执行标准
  门禁；MLflow 注册/晋级保存并取回 sidecar、`get` 重校验 sha256、
  `list_entries` 跳过非 AutoVision 模型版本。
- **数据**：mmap 指纹覆盖标签文件路径/大小/mtime 与 `ignore_classes`；
  `overwrite=True` 强制重建；user-held 权重解析强制 sha256 校验（摘要缓存）。
- **serving**：`/metrics`、`/drift` 在设置 API key 后要求鉴权
  （`protect_observability_endpoints`，探针保持公开）；新增
  `deploy/k8s/` Deployment/Service/HPA/PDB/PVC 参考清单。
- **合规/质量**：`compliance-report fail_on=error|warning|none`、
  CycloneDX SBOM、报告 sha256 清单、数据快照重算与漂移检测；CI 新增
  coverage 下限、bandit（medium/high）与核心模块 mypy 门禁。
- **TensorRT**：新增 `scripts/verify_tensorrt_engine.py` 与 self-hosted
  `trt-nightly.yml`，在目标 GPU 上执行真实 engine build/parity。

### 安全与生产加固

- **serving**：新增可选 `api_key`（Bearer / `X-API-Key`，常量时间比较）、
  `max_body_bytes` 流式请求体上限、`max_image_pixels` 解压炸弹防护、
  `/readyz` 就绪探针、`X-Request-ID` 透传；推理改在线程池执行并以
  `max_concurrency`（默认 1）限制并发，事件循环不再被 forward 阻塞。
- **serving 生产化**：新增 TLS 与多 worker 启动（`serve(workers=...)`，
  多 worker 走 import-string factory）、进程内 token-bucket 限流
  （`rate_limit_per_minute` / `burst` / 可信代理头）、预测分布漂移
  （`/drift`，标签分布 + 分数直方图 PSI，基线可由
  `build_drift_baseline` 生成）以及 A/B/canary 分流
  （`variants=` / `variant_weights=`，按 request ID 稳定哈希，支持
  `X-AB-Variant` 强制）。每个 variant 的请求数与 PSI 进 `/metrics`。
- **训练早停/回调**：task training 新增 `early_stopping_patience` /
  `early_stopping_min_delta`（CLI `trainer_args` 与 Python API），触发时保存
  latest checkpoint 并在 `TaskTrainResult` 返回 `stopped_early` / `stop_step`；
  `fit_task(callbacks=...)` 提供 `on_train_start` / `on_step_end` /
  `on_validation_end` / `on_early_stop` / `on_train_end` 钩子（rank 0，
  callback 异常只告警不中断训练）。
- **fp16 训练**：`16-mixed` 在 CUDA/NPU 上接入 `GradScaler`，scaler 状态随
  checkpoint 保存/恢复；梯度检查与裁剪在 unscale 之后进行。旧 checkpoint
  无 scaler 键时保持兼容。
- **确定性**：新增 `AUTOVISION_DETERMINISTIC=1`，开启 torch 确定性算子和
  cuDNN deterministic（默认关闭，部分算子会报错）。
- **下载完整性**：`AUTOVISION_OFFLINE=1` 现在同时拒绝数据集与可下载模型联网；
  新增 `AUTOVISION_REQUIRE_DATASET_HASHES=1` 拒绝无 sha256 的数据集条目；
  COCO 数据集 URL 改为 HTTPS；数据集/模型下载传入超时。
- **模型注册表**：key 校验拒绝 `.`、`..`、路径分隔符、NUL 和空白键；
  读写路径使用跨进程文件锁（读不会看到半拷贝条目，并发注册有唯一胜者）；
  MLflow 后端接受 `gates=` 并显式 fail-closed（不会静默跳过部署门禁）。
- **运行清单**：训练输出目录新增 `run_manifest.json`（命令、git commit、
  环境白名单、依赖版本、GPU）；resume 写独立 `run_manifest.resume-*.json`。

### 工程与发布

- 新增 `SECURITY.md`（私有漏洞披露流程、支持范围、响应目标）与
  `.github/dependabot.yml`（pip / GitHub Actions 每周更新）。
- 新增 `py.typed`（PEP 561），包数据随 wheel 发布。
- GitHub CI 测试矩阵覆盖 Python 3.10 / 3.11。
- **双语文档站点**：MkDocs Material + `mkdocs-static-i18n`，中文为默认站点、
  英文构建在 `/en/`；新增 `README.en.md` 与首页 / 任务示例 / 已知限制 /
  用户自持权重 4 个英文页，未翻译页自动回退中文。
- **发布元数据**：`pyproject.toml` 补齐 `[project.urls]`，MkDocs 配置写入
  `site_url` / `repo_url`；新增 `scripts/publish_clean_cut.sh` 生成并校验
  单提交发布快照。
- 新增回归测试：serving 鉴权/体积限制/并发/就绪、TLS/多 worker 启动参数、
  限流、漂移 PSI、A/B/canary 分流、fp16 scaler、确定性开关、早停/回调、
  下载门禁、注册表并发与路径校验、运行清单、打包/安全政策。
- 测试稳定性：panoptic 端到端确定性用例固定 CPU，避免 GPU bf16 归约顺序
  造成的假失败。

### 发布前可靠性加固（2026-10-19）

- **serving / 流**：新增 `inference_timeout_s`（默认 120s；超时返回 504 并标记引擎
  不可再用，后续请求 503，避免复用卡死的非线程安全后端）；未处理异常统一返回带
  `request_id` 的结构化 500；探针不再返回绝对路径，RTSP 凭据在错误/统计中脱敏；
  `/metrics`、`/drift` 与 `/predict` 共用限流；writer 失败即时中止；截断视频
  解码失败不再被当成 EOF；多路流新增 `frames_failed` 计数；`track=True` 用于
  非检测结果时显式报错；区域/平滑边界与零面积框不再产生误计数或 Kalman 崩溃。
- **训练 / 数据**：有限 loss + NaN/Inf 梯度现在会中止训练；checkpoint 使用唯一
  临时文件 + fsync，并保存/恢复 Python/NumPy/Torch RNG 状态；pretrain 与
  best-export/sidecar 写失败统一走 rank-0 barrier；空训练/验证 loader 显式报错；
  mmap refcount 损坏可恢复；数据快照对缺失路径默认严格失败；COCO `file_name`
  做目录逃逸校验；JPEG EXIF 方向归一化，截断图片不再静默进入训练；DDP 下自定义
  采样器显式拒绝而不是被静默替换。
- **导出 / 注册表**：ONNX 与 int8 导出改为临时文件 + 原子替换，失败保留上一版；
  `ParityGate` 默认要求真实 parity 报告（可绑定 sha256），不再接受自报标志；
  注册表读取时重新校验模型与 sidecar sha256、拒绝 manifest 路径逃逸；manifest
  原子写入；审计日志拒绝畸形行、使用 hash chain + fsync；插件拒绝混合迭代返回、
  严格发现失败时回滚本次已注册插件并加锁。
- **故障恢复与验收补强（同日第二轮）**：checkpoint / ONNX / int8 / 注册表
  manifest / run manifest 写入前清扫同目标超过 1 小时的崩溃残留临时文件
  （`runtime/fs_utils.sweep_stale_temp_files`，年龄保护绝不触碰在写文件）；
  checkpoint 磁盘满（ENOSPC）统一包装为 `CheckpointError` 且旧文件完好；
  加载跨版本 checkpoint 记录 warning；mmap 缓存损坏/截断改为带重建指引的
  明确报错（运行中被外部截断属 OS 层 SIGBUS，已写入 known_limits）；serving
  引擎毒化后 `/readyz` 返回 503（`poisoned=true`）、`/healthz` 携带
  `poisoned` 字段，编排器可据此摘流重启；新增 kill -9 中途击杀训练 + 断点
  续训的故障注入测试（`tests/test_crash_recovery.py`）。

## [0.2.0] - 2026-10-16

首个对外发布版本（clean-provenance 发布线）。

### 新增

- **任务面 8 个**：检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类；
  Python API 与 `autovision-cli`（20 条命令）共享同一套 Pydantic 严格校验。
- **检测谱系 `detr/*`**（39 个注册变体）：DINOv2 / DINOv3 ViT 与 ConvNeXt、
  HGNetV2 装配、GTR 解码器装配；档位谱系 `n` / `s` 冻结为公开档位。
- **GTR 多任务家族**（MIT，26 个官方权重，sha256 全钉住）：检测 / 实例分割 /
  语义分割 / 深度 / 旋转框 / 姿态（s/m/l/x，旋转框 s/x）；官方 EMA 权重直载，
  各任务加载与前向数值对拍（检测解码器谱系 bit-equal）。
- **姿态估计任务面**：COCO 关键点数据面、OKS 指标与官方损失（matcher / 损失 /
  CDN 与官方实现逐位一致）。
- **自监督预训练与蒸馏**：DistillationV3、SimCLR、DINO、DINOv2、DINOv3、
  DenseCL、DetCon B/S、训练中 kNN 探针。
- **运行时**：DDP（预训练 + 任务训练双引擎）、混合精度、原子 checkpoint +
  resume、自描述 checkpoint 格式、NaN 捕获、JSONL / TensorBoard / MLflow / W&B
  记录器、OTel（opt-in，默认关闭）。
- **导出与部署**：ONNX（fp32 / fp16，默认数值对拍门禁）、TensorRT（batch-only
  profile）、int8 QDQ 量化链、`export` CLI、benchmark 三后端（torch / ONNX /
  TensorRT）parity 门禁。
- **参考推理运行时** `autovision.inference`：视频文件 / RTSP / HTTP 流 + batch
  聚合、torch / ONNX / TensorRT 后端、与评测同源的前后处理、
  `predict --source` 输出逐帧 JSONL + 渲染视频。
- **数据层**：11 类数据集、检测增广全套（mosaic / mixup / copyblend /
  box-aware eraser，逐 step 窗口调度）、mmap 文件清单（内容指纹 / 原子发布 /
  碰撞守卫）。
- **权重供应链**：31 个 HuggingFace 资产（revision + sha256 钉住 + 许可门控）、
  GTR 26 个官方权重（MIT）、用户自持资产库（仓库外，不可再分发，见模型库页面）。
- **文档**：公开文档站点与模型库页面（每模型许可 / sha256 / 官方指标 / 加载
  代码）、协议化基准表 `docs/benchmarks.md`。


### 变更

- 版本号对齐 `0.2` 线：`0.1.0` → `0.2.0`；打包版本改为单一源
  （`src/autovision/_version.py`，pyproject 动态读取）。
- 文档树重构为站点导航结构（as-built 架构总览 + 模型库 + 基准 + 贡献 + 发布
  流程）；历史设计分册不再随公开树分发。

### 安全与合规

- 溯源与分层门禁（`scripts/check_provenance.py` / `scripts/check_layering.py`）
  在 CI 强制。
- 受限许可权重默认不下载（DINOv2 CC-BY-NC 4.0 / DINOv3 License 需显式自证
  接受条款后启用）。
- AGPL 组件（`ultralytics`）仅允许 extras 隔离加载：不进核心依赖、不在模块级
  import。
