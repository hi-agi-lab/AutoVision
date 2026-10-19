# 导出与部署

导出入口按目标后缀分发（`.onnx` / `.engine` / `.xml` / `.pt`），默认执行数值对拍门禁；
benchmark 覆盖 torch / ONNX / TensorRT 后端，报告携带参数量与 FLOPs。快速示例见
仓库根 README 的「导出与部署」。

## 动态 H/W（`dynamic_spatial=true`）

ONNX 图的 H/W 轴动态，导出时内置「第二尺寸 parity 探针」（torch ↔ ONNX Runtime
在未见尺寸对拍），并要求 dynamo 导出器（fail-closed；严格 trace 失败时静态导出
会回退 legacy TorchScript）。

| 骨干线 | 动态 H/W | 说明 |
|---|---|---|
| `rtdetr/resnet18` 等 torchvision CNN | 支持 | 纯 CNN，可符号化 |
| HGNetV2 谱系（`detr/n`、`detr/s`、`detr/hgnetv2-*`） | 支持 | 同上 |
| HF 骨干（`detr/dinov2-*`、`detr/dinov3-*`） | 不支持 | 上游 transformers 在输入尺寸上有 Python 级分支 / 守卫，严格导出无法符号化；会以可读错误指引（导出静态 H/W 或改用可符号化骨干） |
| `detr/gtrvit-s` | 不支持 | 带 export_blocker（GLA 无 ONNX 降级） |

## int8 首层 fp32 配方

观测到（间歇性）部分主干（如 torchvision stem 的 Conv+Relu+MaxPool 融合）没有
可用的 TensorRT int8 tactic，把首层留在 fp32 是受支持的兜底配方；构建失败时错误
信息会给出可粘贴的 `nodes_to_exclude`，也可直接：

```python
from autovision.export import Int8QuantizationConfig, suggest_first_layer_nodes
nodes = suggest_first_layer_nodes("model.int8.fp32.onnx")   # 前两个 Conv 节点名
config = Int8QuantizationConfig(nodes_to_exclude=tuple(nodes))
```

## 后处理规格与端到端 parity

每个 ONNX / OpenVINO / TensorRT 导出成功后会写 `<artifact>.postprocess.json`
（`schema_version=1`）：记录家族、输入尺寸、top-k、分数阈值、box 格式、
预处理（resize 类型 / mean / std）与类别名。设备侧可以只依赖这一份 JSON +
NumPy 参考实现复现最终框：

```python
from autovision.export import (
    decode_detections_numpy,
    load_postprocess_spec,
    verify_end_to_end_parity,
)

spec = load_postprocess_spec("model.onnx.postprocess.json")
records = decode_detections_numpy(raw_outputs, metadata, spec)

# torch 引擎 vs ONNX / OpenVINO / TRT 引擎：比较最终框，不比较张量求和。
report = verify_end_to_end_parity(
    torch_engine, onnx_engine, images, threshold=0.5,
    box_tolerance_px=1.0, report_path="parity.json",
)
print(report.passed, report.max_box_diff_px, report.missing, report.extra)
```

- 匹配门限是宽松的（默认 IoU 0.5），判定标准是 `box_tolerance_px`；
  这样 0.5px 的合法数值差不会被误判为漏检，而 2px 的偏移会被抓出。
- 目前为 DETR 家族（RT-DETR / DETR / RF-DETR，top-k）与 PicoDet
  （top-k + class-offset NMS）生成规格；分割 / 姿态 / 深度 / 分类导出不写
  sidecar。
- int8 精度验收：`scripts/measure_int8_accuracy.py` 导出 fp32/int8 并跑同一
  split 的全量 AP，产出 `report.json` + `report.md`（AP delta、门禁判定、
  硬件/校准集记录）。真实读数需要兼容的 ONNX Runtime / TensorRT 环境；
  校准集必须与评测 split 不相交。

## OpenVINO 运行时

导出 IR 后可以直接在推理运行时 / serving 里使用（`.xml` 与同名 `.bin`
必须放在一起）：

```bash
autovision-cli export model=exported_best.pt out=model.xml
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ \
    backend=openvino artifact=model.xml openvino_device=CPU
```

```python
from autovision.inference import InferenceEngine

engine = InferenceEngine(
    "exported_best.pt", backend="openvino",
    artifact="model.xml", openvino_device="GPU",
)
record = engine.run("image.jpg", threshold=0.4)
```

`openvino_device` 接受 `CPU` / `GPU` / `AUTO`；serving 的
`create_app(..., backend="openvino", artifact="model.xml", openvino_device="CPU")`
走同一条路径。

## int8 精度报告

```bash
python scripts/measure_int8_accuracy.py --model best.pt --data coco.yaml \
    --calibration-images calib/ --out bench/int8 --max-images 5000
```

- 同一 split、同一前后处理、同一 pycocotools 协议下比较 fp32 与 int8；
  `--max-ap-drop 1.0` 为默认门禁（绝对 AP 点），失败时退出码 1（`--no-fail`
  可关闭）。
- 报告包含 `map` / `map_50` / `map_75` / 面积分档 / precision / recall 的
  delta，以及后端、校准集、图像数与硬件；可直接贴进模型卡。
- TensorRT 路径加 `--backend tensorrt`（需要 CUDA + tensorrt）。

## C++ 参考运行时

`examples/cpp/` 提供最小 ONNX Runtime + OpenCV 参考实现：读取导出的
`(logits, boxes)`，用与 `postprocess.json` 相同的 DETR top-k / PicoDet NMS
解码，输出原图像素框。host-only 的 `postprocess_test` 只需 g++：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexamples/cpp/src \
    examples/cpp/src/postprocess_test.cpp -o /tmp/postprocess_test
/tmp/postprocess_test
```

测试覆盖 DETR top-k、PicoDet class-aware / class-agnostic NMS、
`max_detections`、batch 内不同原图尺寸与严格分数阈值；完整 runtime
构建说明见 `examples/cpp/README.md`。这是参考示例，不是 ABI/SDK 承诺。

## TensorRT 11 适配

TensorRT 11 移除了 EXPLICIT_BATCH 与 FP16 / 精度约束 builder flags——fp16 引擎
改为经 fp16-ONNX + strongly-typed 网络路线构建（与旧版 builder flag 路线语义等价，
FP32 黑名单继续保护溢出敏感算子）。

## parity 门禁与示例输入

- ONNX FP16 parity 门禁在无 CUDA execution provider 时跳过并告警。
- ONNX 导出的示例输入跟随模型设备（GPU 上训练后可直接导出）。
- OpenVINO IR 同样走对拍门禁；`precision=fp16` 可出半精度 IR。

## GTR 导出与部署现状

- GTR 模型（含 `detr/gtrvit-*`）当前不可导出 ONNX/TensorRT：导出入口以明确报错
  拒绝（GLA 自定义 CUDA 核无 ONNX 降级，见 `GTRViTBackbone.export_blocker`）；
  四个 GTR 任务模型的导出 verify 钩子已就位，随部署工程落地生效。
- Triton GLA 核已迁移（2× 提速）；`torch.compile` 对 GLA 段负收益；CUDA Graphs
  与自研核原地写接口冲突（需 op 改造）；TensorRT 部署需官方 `gla_plugin.cu`。
- 官方训练增广（mosaic / mixup / zoom-out / color-jitter）未移植；姿态官方配方为
  `precision="32-true"`（`use_amp: False`）。

模型与基准读数分别见[模型库](model_zoo.md)与[基准](benchmarks.md)。
