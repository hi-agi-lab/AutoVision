# AutoVision

企业级视觉模型训练框架：**自监督预训练 / 蒸馏 → 微调 → 评测 → 导出**一条链走完。
Python API 与 `autovision-cli`（30 条命令）共享同一套 Pydantic 严格校验，覆盖
检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类的完整任务面。

框架的三条差异化主线：

- **许可干净**：Apache-2.0；全仓溯源门禁与权重许可门控（受限许可权重默认
  不下载，AGPL 组件仅 extras 隔离），模型库页面对每个权重给出许可与 sha256。
- **可验证一致性**：ONNX / TensorRT / OpenVINO 导出默认执行数值对拍门禁，
  并输出可移植的后处理规格；官方权重加载与前向对拍（含 bit-equal）；
  金标准 loss 轨迹快照。
- **协议化基准**：[基准表](benchmarks.md)只引官方发表值、自家数字必附复现
  脚本、只增不改 + 勘误节。

## 快速开始

不训练，直接加载发布权重并预测：

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # parity 门禁 + postprocess 规格
```

可用模型与导出边界：`autovision-cli list-models --task detection --pretrained`、
`autovision-cli model-info rfdetr/rf-detr-small`。

目标检测，训练 → 验证 → 预测 → 导出：

```python
import autovision

data = autovision.ensure_coco8()   # 合成冒烟数据集;换 COCO/YOLO 配置、YAML 路径或字面量 "coco8"
model = autovision.detection("rtdetr/resnet18").train(
    out="runs/det1", data=data, steps=2000, batch_size="auto",
)
model.val(data=data)          # {'map': ..., 'map_50': ...}
model.predict("image.jpg")    # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")    # 按 .onnx / .engine / .pt 后缀分发
```

等价命令行（同一套 Pydantic 校验）：

```bash
autovision-cli train-object-detection out=runs/det1 data=data.yaml steps=2000
autovision-cli val     model=runs/det1/exported_models/exported_best.pt data=coco8
autovision-cli export  model=runs/det1/exported_models/exported_best.pt out=model.onnx precision=fp16
autovision-cli predict model=runs/det1/exported_models/exported_best.pt source=clip.mp4 out=preds/ batch_size=4
```

## 安装

Python ≥ 3.10。源码安装是当前可用路径；PyPI 发布后可用
`pip install autovision-framework`（模型包按 extras 惰性加载，缺失时
对应清单为空而不是报错）：

```bash
pip install -e ".[transformers,timm,export]"    # 常用: HF 骨干 + timm + ONNX 导出
pip install -e ".[rfdetr,export]"               # 预训练 RF-DETR 直接推理
pip install -e ".[video]"                       # 参考推理运行时(视频/RTSP, OpenCV)
pip install -e ".[dev,export,serving,video,transformers]"  # 开发/全量测试
```

PyPI 发行名计划使用 `autovision-framework`，发布后可用。

GTR 全族（MIT）需要 `gtr` extra；完整 extras 表见仓库 README。预训练权重从
HuggingFace 下载，受限网络可设镜像；**受限许可权重默认不下载**（DINOv2
CC-BY-NC 4.0 / DINOv3 License 需显式自证接受条款），许可边界见仓库 `NOTICE`。
各类权重的下载命令、缓存与镜像配置见[权重下载](download_weights.md)。

## 能力速览

| 子系统 | 能力 |
|---|---|
| 配置 / 注册 | Pydantic 严格校验 + did-you-mean 报错 + 注册表；CLI 与 API 同一套校验 |
| 模型层 | 7 个注册源（torchvision / timm / super_gradients / ultralytics（AGPL 隔离）/ dinov2 / dinov3 / rfdetr）+ 树内 `rtdetr/*`、`detr/*`、`dfine/*`、`picodet/*`、`rfdetr/*`、`gtr/*` |
| 数据 / 变换 | 11 类数据集；mmap 文件清单（内容指纹 / 原子发布）；检测增广全套（mosaic / mixup / copyblend / box-aware eraser）；VOC / YOLO-pose / YOLO-seg 转换 + COCO 合并 |
| 预训练方法 | DistillationV3、SimCLR、DINO、DINOv2、DINOv3、DenseCL、DetCon B/S、训练中 kNN 探针 |
| 运行时 | DDP、混合精度、原子 checkpoint + resume、NaN 捕获、JSONL / TB / MLflow / W&B、OTel（opt-in） |
| 任务层 | 8 个任务面；任务指标支持跨 rank 汇聚（DDP 验证）；误差分析（FP 归因 + 混淆矩阵 + top-FP/FN 画廊）+ 伪标注 / 自训练闭环；超参搜索（随机 / 网格 / ASHA + 报告）；任务感知可视化（框 / mask / 旋转框 / 姿态 / 语义与全景分割 / 深度） |
| 部署 | ONNX（FP16 黑名单 + 数值对拍门禁）、TensorRT、int8 QDQ、benchmark 三后端 parity、参考推理运行时（视频 / RTSP + 三段式管线 + batch，同源前后处理；可选 ByteTrack 跟踪、LineZone / PolygonZone 计数、轨迹平滑；mask 默认 RLE）、参考 serving（FastAPI + TLS/多 worker + 限流 + 漂移 + A/B + `/metrics`） |

各模型的许可 / sha256 / 官方指标 / 一行加载代码见[模型库](model_zoo.md)。

## 文档导航

| 页面 | 内容 |
|---|---|
| [模型库](model_zoo.md) | 每模型：许可 / sha256 / 官方指标 / 加载代码（机器生成） |
| [基准](benchmarks.md) | 精度与延迟基准（不可变协议 + 勘误节） |
| [导出与部署](export.md) | 动态 H/W / int8 / TensorRT 11 / parity / GTR 部署注记 |
| [插件协议](plugins.md) | entry-point 插件、API 版本、model_source 接入与模板 |
| [私有注册表](registry.md) | current / rollback / audit / 部署门禁 |
| [合规报告](compliance.md) | 依赖许可 / 模型卡 / 数据快照证据包 |
| [硬件伙伴包](hardware_packs.md) | 一伙伴一包、转换与目标端 parity 契约 |
| [LTS 政策](lts_policy.md) | LTS 节奏、支持窗口、安全 backport |
| [支持政策](support_policy.md) | 社区/企业支持、P1–P3 SLA、支持矩阵 |
| [用户自持权重](user_held_weights.md) | 仓库外 DINOv3 检测器的加载路线与许可边界 |
| [已知限制](known_limits.md) | 定位、适配缺口、谱系状态、导出边界 |
| [架构总览](framework_design.md) | as-built 分层契约 / 模块地图 / 质量门禁 |
| [贡献指南](contributing.md) | 开发环境 / 提交前门禁 / 许可与溯源红线 |
| [发布流程](release_process.md) | 版本政策 / 发布清单 / 权重发布政策 |
| [PyPI 发布](pypi_publish.md) | 发行名 / TestPyPI / PyPI / trusted publishing / 排错 |
| [模型发布](model_publish.md) | 许可审计 / 模型卡 / HF 上传 / 注册表 / 发布门禁 |
| [变更日志](changelog.md) | Keep a Changelog 格式的对外变更记录 |

## 许可

Apache-2.0（全文：<https://www.apache.org/licenses/LICENSE-2.0>；仓库根
`LICENSE`）。第三方组件与权重溯源见仓库根 `NOTICE`；交互式命令全表：

```bash
autovision-cli --help
autovision-cli list-models    # 当前环境可用的模型清单(按已安装 extras 过滤)
```
