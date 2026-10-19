# AutoVision

企业级视觉模型训练框架：自监督预训练 / 蒸馏 → 微调 → 评测 → 导出一条链走完。
覆盖检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类；Python API 与
`autovision-cli` 共享同一套 Pydantic 严格校验。

> **发行名说明**：PyPI 包名是 `autovision-framework`，import 包名仍是
> `autovision`，命令行仍是 `autovision-cli`。本项目与 PyPI 上无关的
> `autovision` 0.0.1 包没有关系；请确认安装的是 `autovision-framework`。

**链接**：[源码仓库](https://github.com/hi-agi-lab/AutoVision) ·
[文档站](https://hi-agi-lab.github.io/AutoVision/) ·
[变更日志](https://github.com/hi-agi-lab/AutoVision/blob/main/CHANGELOG.md)

## 安装

Python ≥ 3.10：

```bash
pip install autovision-framework

# 常用组合：HF 骨干 + timm + ONNX 导出
pip install "autovision-framework[transformers,timm,export]"

# 预训练 RF-DETR 直接推理
pip install "autovision-framework[rfdetr,export]"
```

模型包按 extras 惰性加载，缺失时对应清单为空而不是报错。完整 extras 表见
[仓库 README](https://github.com/hi-agi-lab/AutoVision#安装)。

## 快速开始

不训练，直接加载发布权重并预测：

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # parity 门禁 + postprocess 规格
```

目标检测，训练 → 验证 → 预测 → 导出：

```python
import autovision

data = autovision.ensure_coco8()   # 合成冒烟数据集；可换 COCO/YOLO 配置
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
autovision-cli predict model=runs/det1/exported_models/exported_best.pt images=dir/ out=preds/ threshold=0.5
```

## 常用 extras

| extra | 能力 |
|---|---|
| `transformers` | DINOv2 / DINOv3 骨干（HuggingFace 权重） |
| `timm` | timm 全量骨干（任意输入通道数） |
| `export` | ONNX 导出 + 数值对拍门禁 |
| `openvino` | OpenVINO IR 导出与运行时 |
| `video` | 视频 / RTSP 参考推理运行时 |
| `serving` | FastAPI 参考 serving（`/predict`、`/metrics`） |
| `gtr` | GTR GLA 骨干（Triton，需 CUDA） |
| `download` | GTR 上游权重按需下载（huggingface_hub） |
| `dev` | pytest / ruff（CI 同款） |

## 许可

Apache-2.0（AutoVision 自有代码；全文：
<https://www.apache.org/licenses/LICENSE-2.0>）。第三方组件与权重溯源见发行包内的
`NOTICE` 与 `THIRD_PARTY_LICENSES.md`。受限许可权重（DINOv2 CC-BY-NC 4.0 / DINOv3
License）默认不下载；用户自持权重不可再分发。

> **0.3.0 许可状态**：DEIMv2 派生的 `SpatialPriorModule` 已删除；DINOv3
> ViT 变体保留并使用 no-STA adapter，发行物不再包含 DEIMv2 代码。旧的
> `backbone.sta.*` ViT 检查点不受支持，需重新训练/导出。详见
> `docs/provenance/LEDGER.md`。

完整文档（模型库、基准、导出细节、已知限制）见
[文档站](https://hi-agi-lab.github.io/AutoVision/) 与源码仓库的 `docs/` 目录。
