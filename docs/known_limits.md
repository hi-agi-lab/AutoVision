# 已知限制与路线图

> 本页汇总对外可见的边界与工程注记；导出相关细节见[导出与部署](export.md)。

## 定位

训练 + 参考推理框架。训练、评测、导出是生产路径；跟踪与 serving 是参考实现。
serving 支持可选 API key（`AUTOVISION_SERVING_API_KEY` 或 `api_key=`）、请求体/像素
上限、线程池推理、有界排队（`queue_timeout_s`，饱和返回 503）、单次引擎调用
超时（`inference_timeout_s`，默认 120s；超时返回 504 并将该实例标记为不可再用，
此后 `/readyz` 返回 503（`poisoned=true`）、`/healthz` 携带 `poisoned` 字段，
编排器应摘除流量并重启进程）、`/readyz`、TLS、多 worker、进程内 token-bucket
限流、预测分布漂移
（PSI，`/drift`）与 A/B/canary 分流；`/docs` 与 `/openapi.json` 默认关闭。
全局配额、告警、跨进程一致限流、A/B 统计显著性与金丝雀自动回滚仍属生产网关职责。

## 运行与完整性

- **确定性**：`AUTOVISION_DETERMINISTIC=1` 开启 torch 确定性算子与 cuDNN
  deterministic（部分算子无确定性实现会报错，且通常更慢）；默认关闭。
- **fp16**：CUDA/NPU 上 `precision="16-mixed"` 使用 `GradScaler`，其状态随
  checkpoint 保存/恢复；bf16/fp32 不需要。AMP 溢出由 scaler 跳过该步并回退
  scale，不会被非有限梯度检查中止。
- **断点续训**：checkpoint 保存/恢复 Python/NumPy/Torch RNG 状态，但不恢复
  mid-epoch sampler 位置，也不校验配置/版本差异；续训时优化器状态会覆盖新的
  `lr` 等参数。需要严格轨迹复现时请从 epoch 边界续训。加载跨版本 checkpoint
  会记录 warning（不阻断）。checkpoint/导出/manifest 写入均先清扫同目标
  超过 1 小时的崩溃残留临时文件，再开始本次写入。
- **mmap 缓存完整性**：缓存文件打开时校验，损坏/截断会报带重建指引的错误
  （删除文件或 `overwrite=True` 重建）。运行中文件被外部截断（与他人共享
  `AUTOVISION_DATA_CACHE_DIR` 时的外部写入者）会导致 SIGBUS 进程崩溃——这是
  OS 层约束；请勿让不受信进程写缓存目录。
- **下载完整性**：权重下载校验钉住的 sha256；数据集下载写入临时文件并原子
  替换，缓存压缩包在解压前重新校验 size/sha256（有钉住值时）。内置 COCO/VOC
  数据集条目没有官方摘要，默认只告警。设置
  `AUTOVISION_REQUIRE_DATASET_HASHES=1` 可拒绝未钉摘要的数据集下载；
  `AUTOVISION_OFFLINE=1` 同时拒绝数据集与权重下载。
- **旧 teacher checkpoint**：当前导出的 task model 可被 pickle-free 加载器读取；
  旧导出若嵌入 `__class__` 等对象，默认拒绝。只有可信本地产物才可设置
  `AUTOVISION_TRUST_TEACHER_CHECKPOINT=1` 启用完整 unpickler（RCE 风险自担）。
- **mmap 缓存键**：包含标签文件（COCO JSON / YOLO `.txt` 等）的路径、大小与
  mtime，以及 `ignore_classes` 指纹；编辑标签或改变忽略类别会重建缓存，
  `overwrite=True` 会强制重建。user-held 权重解析现在强制校验 sha256（按
  路径/大小/mtime 缓存摘要）。
- **EoMT 权重**：`load_weights=True` 强制 DINOv2/DINOv3 许可 opt-in；
  hub checkpoint 未在 `WEIGHT_MANIFEST` 中钉 sha256 时默认拒绝，只有显式
  `AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1` 才加载（可信镜像）。class head 会按
  目标类别数/query 数重建；EoMT 现可走 ONNX 导出（legacy TorchScript 路径，
  动态 batch 关闭）。
- **可视化/视频工具**：`visualize/effects.py` 与 `geometry/boxes.py` /
  `geometry/masks.py` 的框、mask、polygon 工具只依赖 NumPy/Pillow；
  `mask_to_polygons` 惰性依赖 OpenCV（`video` extra）。`process_video` /
  `VideoSink` / `ImageSink` 是同步、RGB 约定的轻量管线，不替代
  `run_stream` 的 reader/infer/writer 三段式、后端切换与 JSONL 持久化。
- **运行清单**：每次训练在输出目录写 `run_manifest.json`（命令、git commit、
  环境白名单、依赖版本、GPU）；resume 写独立的 `run_manifest.resume-*.json`，
  不覆盖原始记录。
- **早停/回调**：task training 支持 `early_stopping_patience` /
  `early_stopping_min_delta`（CLI `trainer_args`）与 Python API 的 callback 钩子；
  pretrain 仍按 epochs 跑满。

## 企业能力边界

- **训练并行**：DDP（torchrun）、FSDP 任务训练
  （`trainer_args={"parallel": "fsdp"}`，full-state checkpoint/resume/导出，
  2-rank 单卡测试）与梯度检查点已实现；FSDP 的 EMA 更新需按
  `AUTOVISION_FSDP_EMA_INTERVAL` 做 full-param gather，整模型未做 auto-wrap。
  DeepSpeed / ZeRO / 弹性训练未接入；多机 NCCL 与故障注入未在公开 CI 验证，
  边界见[并行训练](parallelism.md)。
- **注册表**：文件系统后端与 MLflow 后端都执行标准部署门禁
  （`gates=None` 默认 fail-closed）；rollback/audit 仍以文件系统后端为准。
- **CI**：公开 CI 为 Ubuntu + Python 3.10/3.11 的 CPU 任务，测试作业安装
  `serving`/`video`/`transformers` 测试依赖；新增核心子集 coverage 下限、bandit
  （medium/high）、核心模块 mypy 与 `scripts/smoke_enterprise.py` 冒烟门禁。TensorRT/OpenVINO/多机 DDP 等重环境
  用例按门控变量跳过；真实 TensorRT engine/parity 已在 RTX 4090D + TensorRT 11.4 上验证
  int8 QDQ 路径（含 `export_tensorrt(verify=True)` 门禁）；公开 CI/夜间
  self-hosted workflow（`.github/workflows/trt-nightly.yml`）继续覆盖。
- **serving**：TLS、多 worker、进程内限流、漂移与 A/B/canary 分流已提供；
  `/metrics` 与 `/drift` 在设置 API key 后同样要求鉴权（K8s 探针保持公开）；
  限流桶与漂移窗口在多 worker 下各自独立（跨 worker 聚合需指标后端/网关），
  `/drift` 是服务级聚合（A/B 混合流量会混合分布）；无模型热加载（用滚动重启），
  `deploy/k8s/` 提供 Deployment/Service/HPA/PDB/PVC 参考清单；全局配额、告警、
  A/B 统计显著性与自动回滚仍需平台承接（见上文定位）。

## 许可状态与 STA 迁移

- **DEIMv2 组件已删除（2026-10-17）**：`SpatialPriorModule` 已从代码库移除，
  DINOv3 ViT 变体保留但改用 no-STA adapter；发行物不再包含 DEIMv2 代码，
  可按 Apache-2.0 发布。处置记录见[溯源台账](provenance/LEDGER.md)。
- **旧 STA-era ViT 检查点不受支持**：任何含 `backbone.sta.*` 的基线 `.pt`
  或用户自持 ViT 资产（`dinov3-vitt16` / `dinov3-vitt16plus` /
  `dinov3-vits16`）会被加载器明确拒绝，解析入口在读取/哈希容器前即失败，
  需重新训练/导出 no-STA 检查点。DINOv3 ConvNeXt 及用户自持的 4 个
  ConvNeXt 资产不受影响；`USER_HELD_LOADABLE_ASSETS` 只含这 4 个，
  3 个 ViT 留在 `USER_HELD_UNSUPPORTED_ASSETS` 仅供审计
  （`allow_unsupported=True`）。
- **受限权重仍然存在**：DINOv2 权重 CC-BY-NC 4.0、DINOv3 权重 DINOv3
  License、用户自持检测器禁再分发/商用；这些是权重条款，与已删除的 STA
  代码无关。DEIMv2 外部权重仍为非商用。

## 适配缺口

- albumentations 适配层未接入（变换基于 torchvision v2）。
- super-gradients 分割 wrapper 未接入（CustomizableDetector 检测家族可用）。
- 难例加权采样（`hard_examples=`）与长尾采样（`long_tail_sampler=`）仅支持
  单机；DDP 的 DistributedSampler 会覆盖采样权重，组合被显式拒绝。
- Label Studio 的 `keypointlabels` 以 `keypoint_radius` 小框转换为检测框；姿态
  训练应使用原生关键点格式（COCO keypoints / YOLO-pose）。
- Label Studio 的旋转矩形（`rotation != 0`）按轴对齐框导入，旋转角保留在
  标注字段；旋转框训练仍使用 YOLO-OBB / DOTA 格式。
- 多路流运行时 v1 不做跨路跟踪 / ReID；tracking 仅单路 `run_stream`。
  每路渲染 writer 在该路分辨率变化时重建。`LineZone` / `PolygonZone` /
  `TrackSmoother` 与 reader/infer/writer 三段式管线也只接入单路
  `run_stream`；`run_multi_stream` 保持每源独立批处理。
- serving 的 `max_image_pixels` 按单图限制，不限制一次请求的总像素与 RLE
  响应字节数；高并发大图 / 稠密 mask 场景需在网关层再加总预算。

## 检测谱系状态

`n` / `s` 为 spec-frozen（规格冻结：组件与配方已承诺，预训练权重与 COCO AP
待发布）；`m` / `l` / `x` 骨干线方向已定（timm ConvNeXt，tiny→m / small→l /
base→x，选型标准为可导出 + 许可干净），实现待落地。`detr/gtrvit-s` 为 planned。只有 `released` 档带预训练权重：发布门禁要求权重 +
模型卡（许可 / sha256 / seed）+ 全量 AP + 可复现延迟。机器可读状态与变体清单见
[模型库](model_zoo.md)。

## 导出与部署

- GTR 模型（含 `detr/gtrvit-*`）不可导出 ONNX/TensorRT：导出入口以明确报错拒绝
  （GLA 自定义 CUDA 核无 ONNX 降级，见 `GTRViTBackbone.export_blocker`）；四个 GTR
  任务模型的导出 verify 钩子已就位，随部署工程落地生效。
- ONNX FP16 parity 门禁在无 CUDA execution provider 时跳过并告警；ONNX 导出的示例
  输入跟随模型设备（GPU 上训练后可直接导出）。
- 动态 H/W 支持面、int8 首层 fp32 配方、TensorRT 11 适配见[导出与部署](export.md)。
- int8 全量 AP 报告（`scripts/measure_int8_accuracy.py`）需要兼容的
  ONNX Runtime / TensorRT 环境；`export` extra 已要求 `numpy>=1.25`
  （onnx/onnxscript 的导入前提），重装该 extra 后即可复测。当前开发 venv
  仍是旧 numpy，所以本地 ONNX 集成测试按环境门控跳过。
- `examples/cpp/` 是参考运行时，不是 ABI/SDK；完整构建需要 ONNX Runtime 与
  OpenCV 开发库，host-only 后处理测试只需 g++。
- GTR 部署工程：Triton GLA 核已迁移（2× 提速），但 `torch.compile` 对 GLA 段负收益、
  CUDA Graphs 与自研核原地写接口冲突（需 op 改造）、TensorRT 部署需官方
  `gla_plugin.cu`。
- 官方训练增广（mosaic / mixup / zoom-out / color-jitter）未移植；姿态官方配方为
  `precision="32-true"`（`use_amp: False`）。

## 离线与权重

- 蒸馏教师默认请求预训练权重（`teacher_pretrained=True`）；离线时传
  `teacher_pretrained=False`。
- 预训练权重清单按 commit + LFS sha256 全量钉住
  （`src/autovision/encoders/weight_hashes.py`）。
- 受限许可权重（DINOv2 CC-BY-NC 4.0 / DINOv3 License）默认不下载，需显式自证；
  用户自持库禁再分发 / 商用，见[用户自持权重](user_held_weights.md)。
