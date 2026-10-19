# DETR 谱系基准表（roadmap design doc 29 §4）

> **协议（D5/D6/D7，不可变）**：AP 口径 = COCO 2017 val2017 + pycocotools
> 标准；对外数字一律标注 schedule 档位与种子，关键档位 ≥2 seeds 报均值。
> 延迟 = TensorRT，batch=1 与 batch=4 两档，含 H2D 拷贝与预处理，warmup 50
> 取 p50；表头登记硬件型号，不同硬件不同表。对手数字**只引用官方仓库/论文
> 发表值**并给出处与引用日期，不复现对手；自家数字必须附复现脚本。
> **本表只增不改**：修正进文末勘误节（日期 + 原因）。

## 1. 自家精度（COCO val2017 AP）

**§1.0 待训练档位（spec-frozen；无权重、无读数——不对外叙事）**：`detr/n`（hgnetv2-b0，12ep
蒸馏路线，RF-DETR-n 教师）与 `detr/s`（hgnetv2-b1 + dfine_s 装配后短程微调）
尚未产出训练权重与 AP 读数；训练入口见 `scripts/train_spectrum.py`，全量评测见
`scripts/eval_spectrum_coco.py`；短程趋势证据见 §1.1。

| 模型 | 档位来源 | schedule | seeds | AP | AP50 | 权重 | 模型卡 | 复现脚本 | 日期 |
|---|---|---|---|---|---|---|---|---|---|
| gtrpose/s | 官方 EMA 权重直载（`gtrpose_s_coco`） | 无训练（推理复现） | — | **70.05** | 89.64 | 注册表（sha256 核验，MIT） | — | `scripts/eval_gtr_pose_ap.py` | 2026-10-16 |
| gtrpose/m | 官方 EMA 权重直载（`gtrpose_m_coco`） | 无训练（推理复现） | — | **74.04** | 91.47 | 注册表（sha256 核验，MIT） | — | `scripts/eval_gtr_pose_ap.py` | 2026-10-16 |
| gtrpose/l | 官方 EMA 权重直载（`gtrpose_l_coco`） | 无训练（推理复现） | — | **74.70** | 91.90 | 注册表（sha256 核验，MIT） | — | `scripts/eval_gtr_pose_ap.py` | 2026-10-16 |
| gtrpose/x | 官方 EMA 权重直载（`gtrpose_x_coco`） | 无训练（推理复现） | — | **75.48** | 92.39 | 注册表（sha256 核验，MIT） | — | `scripts/eval_gtr_pose_ap.py` | 2026-10-16 |

> **pose 行口径**：COCO val2017 **keypoint AP（OKS）**，官方 EMA 权重 × 我方全链（数据面 → 变换 → 网络 → 后处理 → pycocotools 评估器），全量 5000 图；与 README 70.1 / 74.1 / 74.7 / 75.5 的 delta 为 **−0.05 / −0.06 / −0.00 / −0.02**（AP50 列同链读出）。val500 子集读数 70.61 / 73.38 / 73.79 / 74.74。残余来源：预处理语义差（tensor antialias bilinear vs 官方 PIL bilinear，det 侧同源差 0.03–0.10 AP）。本表为**权重级复现**；官方 train 增广管线（mosaic/mixup/zoom-out/color-jitter）未移植，训练复现为独立切片。

### 1.1 短程趋势冒烟（非发布口径——流程/数值健康证据，数字不进对外叙事）

| 模型 | 配置 | 步数（epoch） | AP | AP50 | 备注 | 日期 |
|---|---|---|---|---|---|---|
| detr/s | dfine_s 装配暖启 + timm 预训练骨干 + EMA warmup 100 | 2000（0.17ep） | 0.1302 | 0.2026 | val 单调升 9.6→13.0；RTX 4090D bf16 | 2026-10-15 |
| detr/gtrvit-s | **官方 GTR-S 骨干直载** + 随机 GTREncoder/decoder | 2000（0.17ep） | 0.1902 | 0.3109 | 较 detr/s 同期 **+6.0 AP**；GLA 骨干质量兑现 | 2026-10-15 |
| detr/gtrvit-s | 全随机（对照） | 500（0.04ep） | ~0.0008 | ~0.0018 | 从零训 ViT 系无信号，符合预期 | 2026-10-15 |
| detr/gtrvit-s | **官方全权重直载**（骨干 + encoder + decoder，EMA 口径）后短程微调 | 3000（0.8ep） | **0.5735**※ | 0.7422 | val 为 **500 图子集**口径；轨迹：anchor 0.5797（未训）→ 0.5688(500) → 0.5713(750) → 0.5700(1000) → 0.5675(1500) → 0.5706(2250) → **0.5735(3000)**；loss 22.7→21.5 单调降；0.4 s/step bs16/640 bf16；smoke 脚本已归档（公开树不携带；加载/训练闭环由 `tests/test_g2_gtr.py` 锁定），读数保留原文 | 2026-10-16 |

## 2. 自家延迟（torch eager 口径，含预处理；TRT 列待 P3）

硬件：NVIDIA GeForce RTX 4090D 24GB / torch 2.x cu / Python 3.10；p50/50 iters/warmup 20；复现 `scripts/measure_spectrum_benchmarks.py`。日期 2026-10-15。

> ⚠️ **口径警示**：本表为 **torch eager 含 CPU 预处理** 口径，与 §3 的官方协议（torch.compile + CUDA Graphs + 自研 CUDA GLA 核 + forward-only FP16）**不可直接对比**。同口径粗测：gtrvit-s forward-only FP16 bs1 在我们集成路径为 ~33ms vs GTR 官方全模型 1.225ms（~27×）——差距来源：官方用自研 chunkwise CUDA GLA 核（比 fla Triton 快 4×@1.6K tokens）+ torch.compile + CUDA Graphs + convert_to_deploy 融合。**GTR 的速度优势必须配套这三件部署工程才成立**（分册 31 G2 剩余矩阵）；TRT 部署还需其自定义 gla_plugin.cu（官方车端 DRIVE AGX Thor 2.28-8.77ms 已验证该路线）。**勘误（2026-10-15 晚）**：早前版本称 torch.compile「需 py≥3.11、本 venv 3.10 不可用」系误读——实测 `.venv`（py3.10.12）下 torch.compile 可正常编译（dynamo frames ok=1）；那条「not available in Python 3.10」警告来自 **fla 库自身的降级策略**（fla 在 <3.11 停用其内部编译装饰器，官方参考环境为 py3.11）。torch.compile 对本仓库完全可用。
>
> **部署工程迁移实测（2026-10-15，GTR-S 骨干 640px FP16 bs1，无预处理/无 decoder）**：fla Triton eager **16.25ms** → deploy（convert_to_deploy 递归 28 钩子）+ **自研 CUDA GLA 核 7.97ms（2.04×）**——核已从 GTR-main csrc 编译迁移（nvcc 13.0/torch cu130 匹配，`-std=c++20` 补丁，官方 test_gla ALL PASS，`.so` 经包 `__init__` sys.path 注入零改动接线）。另两项实测结论：① **torch.compile 默认模式对 GLA 骨干负收益**（9.62ms：自研核经 custom op + `torch.compiler` 图断裂，编译无法覆盖 GLA 段、且引入 guard/重编译开销；对 GLA 无 custom-op 断裂的路径 compile 仍可用——py3.10 不是障碍，见上勘误）；② **CUDA Graphs 当前被拒**（custom op `mutates_args={"out"}` 原地写与 graphs 冲突，需 op 接口改造）；③ **整网 half eager 与我们 decoder 的 bbox FP32 精修块（fp16 溢出保护，有意设计）冲突**——混合精度导出（TRT）才是正确部署形态，不做整网 half。

| 模型 | 精度 | batch=1 p50 (ms) | batch=4 p50 (ms) | 备注 |
|---|---|---|---|---|
| detr/n | fp32 | 19.3 | 20.6 | eager 含 CPU 预处理，TRT 口径另测 |
| detr/n | fp16 | 22.6 | 24.8 | autocast 开销为正收益前口径 |
| detr/s | fp32 | 20.0 | 20.6 | |
| detr/s | fp16 | 24.1 | 24.8 | |
| detr/gtrvit-s | fp32 | 29.2 | 34.2 | GLA Triton 核；fp16 慢于 fp32（核内 fp32 计算+转换） |

> **补记（2026-10-16，口径标定）**：按现有脚本重测，§2 口径 = **未 deploy 的 eager
> `forward_backend`**（detr/n fp32 b1 19.5ms vs 表内 19.3ms，b4 21.5 vs 20.6；
> detr/s 21.5/20.7 vs 20.0/20.6——n/s 一致）；**detr/gtrvit-s 偏高
> 2.3–5.0ms**（34.2/32.4 vs 29.2/34.2，原脚本未入库、残余差不可考）。若把 CPU 预处理（resize+normalize）计入，b1 为 27–35ms 量级
> （§2.1 全链列同源）；「含 CPU 预处理」表述自此以脚本口径为准。

### 2.1 模型成本与 TensorRT 延迟（2026-10-16 首测）

硬件与 §2 同（RTX 4090D 24GB；torch 2.14.1+cu130；TensorRT 11.4）。复现
`scripts/measure_spectrum_benchmarks.py`（`--backends torch tensorrt`；引擎由
`--engine-dir` 缓存复用）。

**成本口径**：Params = 部署前（权重/训练态）参数量；GFLOPs = 部署后推理图执行的
FLOPs（torch flop counter，乘加计 2 flops）。该口径与 §3 对手表交叉验证成立：
`detr/gtrvit-s` 读出 **33.9 GFLOPs / 12.12M**，官方 GTR-S 为 33.8 / 12.1。
detr/n、detr/s 的 deploy 融合会额外驻留一份融合卷积（训练分支不删——上游同源
实现），部署态参数量更高（n 24.8M / s 25.2M）；表内取权重口径，`gtrvit-s` 两态
一致（12.07M，GTR 部署为净删除）。成本与延迟均为架构级读数（随机初始化，权重
不影响形状）。

**延迟口径**：TensorRT **fp16** 引擎（batch profile 1..4；TRT 11 上经 fp16-ONNX
+ strongly-typed 网络构建），warmup 50 / 200 iters / p50，两列读数：

- `engine` = 引擎单次前向（含 Python 侧 shape 绑定与流同步开销，不含预处理）；
- `全链` = 预处理（resize+normalize，图像解码不计）→ H2D 拷贝 → 引擎——即 §1
  协议头部登记的「含 H2D 拷贝与预处理」口径（不含后处理）。

| 模型 | Params (M) | GFLOPs | TRT-fp16 b1 engine (ms) | TRT-fp16 b1 全链 (ms) | TRT-fp16 b4 engine (ms) | TRT-fp16 b4 全链 (ms) | 复现脚本 | 日期 |
|---|---|---|---|---|---|---|---|---|
| detr/n | 17.79 | 59.1 | 2.70 | 13.06 | 3.46 | 19.68 | `scripts/measure_spectrum_benchmarks.py` | 2026-10-16 |
| detr/s | 18.14 | 61.8 | 2.83 | 12.71 | 3.54 | 20.22 | 同上 | 2026-10-16 |
| detr/gtrvit-s | 12.12 | 33.9 | — | — | — | — | 同上（导出被显式拒绝） | 2026-10-16 |
| rtdetr/resnet18 | 26.46 | 87.1 | 1.92 | 11.34 | 3.10 | 23.61 | 同上 | 2026-10-16 |

> **读数解读**：① `detr/gtrvit-s` 的 TRT 列不可测——GLA 自定义 CUDA 核无 ONNX
> 降级，导出入口以明确报错拒绝（部署工程 G2；TRT 需官方 `gla_plugin.cu` 路线）；
> 成本列不受影响。② 与 §3 不可直接对比：对手为 torch.compile + CUDA Graphs +
> FP16 的 forward-only / CUDA-event 口径，本表为 Python 驱动的引擎读数。③ **全链
> 列揭示：CPU 预处理（1280×720 → 640² resize+normalize ≈ 10ms）在 b1 链路占
> ~80%**——预处理下沉（GPU resize 或引擎内嵌前处理）与引擎优化同等重要。

### 2.2 新后端首测（2026-10-16，P1 部署广度）

硬件：NVIDIA GeForce RTX 4090D 24GB（TensorRT 11.4）/ CPU（OpenVINO、ONNX
CPU EP）；模型：`rtdetr/resnet18`（我方 2 步冒烟权重，**仅测速**）；复现脚本：
`scripts/measure_new_backends.py`（每行命令见下表备注；parity 门禁默认开启）。

| 后端 / 精度 | 尺寸 | batch | mean (ms) | p50 (ms) | p90 (ms) | parity | 复现 |
|---|---|---|---|---|---|---|---|
| torch fp32（eager） | 640×640 | 1 | — | **16.95** | 17.38 | 基准 | `--image-size 640 --backends torch tensorrt-int8 --calibration-images <dir> --quantization-exclude <stem convs>` |
| tensorrt **int8**（QDQ） | 640×640 | 1 | — | **1.60** | 1.61 | ✅（int8 容差 atol 2.0/rtol 0.5） | 同上（52MB 引擎；首层 fp32 配方） |
| torch fp32（eager） | 64×64 | 1 | 15.84 | 15.71 | 16.65 | 基准 | `--backends torch onnx openvino`（默认） |
| onnx fp32（CPU EP） | 64×64 | 1 | 10.43 | 10.12 | 13.11 | ✅ | 同上 |
| **openvino** fp32（CPU） | 64×64 | 1 | **10.18** | **10.19** | 10.32 | ✅ | 同上（98.2 imgs/s） |

复跑波动（2026-10-16 二次执行，同机同权重）：torch 640 p50 16.9–17.1ms、
int8 640 p50 1.58–1.60ms、64px 三后端 ±3%。口径：`benchmark` 计时含各后端自身
的 H2D/拷贝（ONNX/OpenVINO 在 CPU 上无拷贝）；
parity 与导出侧同一 `verify_onnx_export_outputs` 契约（比较输入为**带种子的随机
张量**——全零输入在 DETR 顶-k 选择处存在并列，元素级对比会抖动；见 §4 勘误）。
动态 H/W（D2）验证读数（`export_onnx(dynamic_spatial=True)`，`rtdetr/resnet18`
64px 权重）：ONNX Runtime 在 48/64/96/160 四个尺寸全部运行且输出形状一致，导出
时内置"第二尺寸 parity 探针"通过；legacy（TorchScript）导出器的动态尺寸图为
"假动态"（尺寸被烘成常量，其他尺寸触发 ORT broadcast 错误），已被门禁拒绝并
改为 fail-closed（含 `input_dynamic_shapes` 规格标记路径）。int8 ONNX↔fp32 的
QDQ 对拍读数由 `scripts/measure_int8_accuracy.py` 复现。

### 2.3 四后端同尺寸复测（2026-10-17，P1 部署广度第二轮）

硬件/软件同 §2.2；模型：`rtdetr/resnet18`（我方 **2 步冒烟权重**，640px，
`/tmp` 训练目录，仅测速）；四后端**同尺寸同批** 640×640 b1，parity 门禁全开。
复现（仓库脚本，`auto` 从模型自身 fp32 导出自动取首层 Conv 配方）：

```bash
python scripts/measure_new_backends.py \
    --model <run>/exported_models/exported_best.pt --image-size 640 \
    --calibration-images <dir> --out bench_640 \
    --backends torch onnx openvino tensorrt-int8 --quantization-exclude auto
```

| 后端 / 精度 | 尺寸 | p50 (ms) | p90 (ms) | mean (ms) | img/s | parity |
|---|---|---|---|---|---|---|
| torch fp32（CUDA, eager） | 640×640 | **14.92 / 16.73** | 15.51 / 19.19 | 14.97 | 58.7–66.8 | 基准 |
| onnx fp32（CPU EP） | 640×640 | **188.31 / 187.00** | 190.63 / 189.16 | 190.43 | 5.3 | ✅ |
| openvino fp32（CPU） | 640×640 | **109.06 / 125.52** | 877.32 / 845.51 | 227.03 | 4.4 | ✅ |
| tensorrt **int8**（QDQ） | 640×640 | **1.535** | 1.686 | 1.56 | **639.7** | ✅（int8 容差） |

复跑（括号第二值）与波动说明：torch ±12%、onnx ±0.7% 稳定；**openvino
640px 存在稀疏长尾**——两轮 20 步读数里 p90（即次大值）分别为 877/846ms，
即**每轮 ≥2/20 次 >845ms**；随机权重模型上的 60 次直测为 2–3 次
350–739ms、p50 ~102ms（同机、同尺寸）。长尾在 640px 上稳定复现、64px
无此现象（见 §2.2）——疑似 CPU 插件的内存/线程抖动，记录为观察项（后端
目前不暴露 `ov_config`，PERFORMANCE_HINT/AFFINITY 复测列为待办）。
int8 引擎与 §2.2 同量级（1.53–1.60ms），**首层 fp32 配方由
`suggest_first_layer_nodes` 自动给出**（本次取到 `node_Conv_1952/1954`；
导出图节点名随运行而异，`--quantization-exclude auto` 才是可复现路径），
构建与 int8↔fp32 对拍一次通过。另记：不带排除的构建在复测中**也曾成功**
（历史失败为间歇性）——排除首层是受支持的兜底配方而非硬性前提。

parity 口径提示（本轮实测发现并写入 `benchmark()` docstring）：parity 比较
假设**数值上已收敛的检查点**。`model="rtdetr/resnet18"` 规格串是随机权重，
未训练模型的框头在**同一权重的不同前向之间**就能差 ~0.8–1.0（归一化）——
GPU↔CPU 如此，CPU↔`deepcopy` 亦然——parity 门禁报出的差异来自模型自身
条件数而非后端保真度；测速可用规格串，**验 parity 必须带（哪怕 2 步的）检查点**。

动态 H/W（D2）家族覆盖（`export_onnx(dynamic_spatial=True)`，含第二尺寸
parity 探针，本轮实测）：`rtdetr/resnet18` ✅、`detr/hgnetv2-b0`（256px）✅、
`detr/s` ✅（timm CNN 线通过；`hgnetv2-b0` 另在 192/320
两尺寸做 torch↔ORT 对拍通过）；HF 骨干线（`detr/dinov2-*`、`detr/dinov3-vit*`、
`detr/dinov3-convnext-*`）不支持——上游 transformers 在**输入尺寸上有 Python 级
分支/守卫**（ViT 嵌入的 patch 数分支、ConvNeXt 的 stage-grid 守卫），严格导出
无法把它们符号化，现以可读错误指引（导出静态 H/W 或改用可符号化骨干）拒绝；`detr/gtrvit-s`
本身带 export_blocker（GLA 无 ONNX 降级），不适用。HF 骨干的**静态**导出此前
在 transformers ≥5.x 全线失败（首个 eager 前向在 `threading.Lock` 下安装
output-capturing hooks，dynamo 无法 trace `with <lock>`），已修复（trace 前
warm-up 一次前向）。

## 3. 对手官方数字（只引官方发表值）

**统一协议来源**：GTR 论文（arXiv:2609.26590，Intellindust-AI-Lab）将下列全部基线在**同一协议**下重测延迟——RTX 4090 / FP16 / batch=1 / torch.compile + CUDA Graphs / median CUDA-event latency / 640×640；基线 AP 引自各自原文。引用日期 2026-10-15。⚠️ 该协议与我们 §2 的 torch eager 口径不可直接对比（见 §2 表注）。

| 模型 | Params (M) | GFLOPs | 延迟 (ms) | COCO AP |
|---|---|---|---|---|
| **GTR-S**（O365 预训练） | 12.1 | 33.8 | **1.225** | **53.6** |
| GTR-M | 22.7 | 62.6 | 1.462 | 57.3 |
| GTR-L | 37.2 | 106 | 1.908 | 58.9 |
| GTR-X | 46.5 | 130.2 | 2.115 | 59.4 |
| D-FINE-S | 10 | 25 | 1.194 | 50.7 |
| D-FINE-M | 19 | 57 | 1.740 | 55.1 |
| D-FINE-L | 31 | 91 | 2.462 | 57.1 |
| D-FINE-X | 62 | 202 | 3.385 | 59.3 |
| DEIM-S | 10 | — | 1.180 | 49.0 |
| DEIM-M | 19 | — | 1.744 | 52.7 |
| DEIM-L | 31 | — | 2.460 | 54.7 |
| DEIM-X | 62 | — | 3.384 | 56.5 |
| DEIMv2-S | 10 | — | 1.687 | 50.9 |
| DEIMv2-M | 18 | — | 2.398 | 53.0 |
| DEIMv2-L | 32 | — | 2.776 | 56.0 |
| DEIMv2-X | 50 | — | 3.276 | 57.8 |
| RT-DETRv2-S | — | — | 1.291 | 48.1 |
| RT-DETRv2-M | — | — | 1.741 | 49.9 |
| RT-DETRv2-L | — | — | 2.332 | 53.4 |
| RT-DETRv2-X | — | — | 3.259 | 54.3 |
| RF-DETR-S | 32 | 60 | 1.299 | 52.9 |
| RF-DETR-M | 34 | 79 | 1.431 | 54.7 |
| RF-DETR-L | 34 | 126 | 1.968 | 56.5 |
| RF-DETR-X | 126 | 300 | 3.243 | 58.6 |
| YOLO26-S | 10 | 21 | 0.962 | 47.8 |
| YOLO26-M | 20 | 68 | 1.473 | 52.5 |
| YOLO26-L | 25 | 86 | 1.967 | 54.3 |
| YOLO26-X | 55 | 194 | 3.087 | 56.9 |
| YOLO11-S | — | — | 1.490 | 46.6 |
| YOLO11-M | — | — | 2.061 | 51.2 |
| YOLO11-L | — | — | 2.617 | 53.4 |
| YOLO11-X | — | — | 3.747 | 54.7 |

**同延迟档读数**（4090 FP16）：GTR-S 1.225ms/AP 53.6 为 ~1.2ms 档冠军（D-FINE-S 1.194/50.7、RF-DETR-S 1.299/52.9、DEIM-S 1.180/49.0）；M/L/X 档对 D-FINE 同档分别 **快 16%/22%/37%** 且 AP +2.2/+1.8/+0.1；对 YOLO26：更快一档但 GTR 同延迟 AP 高 5+。

| 模型 | AP（val2017） | 延迟（官方口径） | 出处 | 引用日期 |
|---|---|---|---|---|
| _（首批已并入上表；后续逐行补官方原始出处链接）_ | — | — | — | — |

### 3.1 RF-DETR 官方口径（T4 / TensorRT 10.4 / CUDA 12.4 / FP16 / bs1）

> 与 §3 的统一协议（RTX 4090 / torch.compile + CUDA Graphs）**不可直接对比**。
> 来源：Roboflow RF-DETR 官方文档（<https://rfdetr.roboflow.com>）与 ICLR 2026
> 论文（arXiv:2511.09554），引用日期 2026-10-17。官方表未列 GFLOPs；我方全链
> AP / 延迟复测待 Phase 1 补。

| 模型 | 输入 | COCO AP | COCO AP50 | Params (M) | 官方延迟 (ms) | 许可 |
|---|---|---|---|---|---|---|
| RF-DETR Nano | 384² | 48.4 | 67.6 | 30.5 | 2.3 | Apache-2.0 |
| RF-DETR Small | 512² | 53.0 | 72.1 | 32.1 | 3.5 | Apache-2.0 |
| RF-DETR Medium | 576² | 54.7 | 73.6 | 33.7 | 4.4 | Apache-2.0 |
| RF-DETR Large | 704² | 56.5 | 75.1 | 33.9 | 6.8 | Apache-2.0 |
| RF-DETR Base（上游弃用） | — | — | — | — | — | Apache-2.0 |

注：

- 当前官方阵容为 N / S / M / L；Base 已被 Medium 取代，仓库保留 `rf-detr-base`
  资产用于兼容，选型时请用 N/S/M/L，不要使用 Base。
- XL（AP 58.6 / 126.4M / 11.5ms）与 2XL 为 PML 1.0，不属于 Apache 家族；§3 表中的
  `RF-DETR-X` 对应 XL，勿与 Apache 五档混淆。
- §3 行是 GTR 论文统一协议下的延迟读数 + 各原文 AP；本节是 Roboflow 官方 T4 口径。
  Small 档 AP 在两处为 52.9 / 53.0（版本差），选型时以本节官方卡为准并注明出处。

## 4. 勘误

| 日期 | 条目 | 修正 | 原因 |
|---|---|---|---|
| 2026-10-16 | §2 表头复现脚本 | `scripts/measure_spectrum_latency_torch.py` → `scripts/measure_spectrum_benchmarks.py`（数字未变）<!-- docpath-ok --> | 脚本重组更名（协议不变） |
| 2026-10-16 | §1.1 `detr/gtrvit-s` 冒烟行的脚本引用 | smoke 脚本归档（公开树不携带；读数与轨迹保留原文） | 公开树脚本清理 |
| 2026-10-16 | §1 `detr/n`、`detr/s` 两行 | 移入新增 §1.0「待训练档位」块（结果表只保留已产出读数的行） | A2 透明基准补全（结果表禁空列） |
| 2026-10-16 | §2 口径补记 | 见 §2「补记」：口径 = 未 deploy 的 eager forward；gtrvit-s 重测偏高 2.3–5.0ms（数字未改） | 口径标定实验 |
| 2026-10-16 | 导出/benchmark parity 比较输入 | 由全零样例改为带种子的随机张量（数字未变；门禁复现性修复） | 全零输入在 DETR 顶-k 处并列、元素级对比抖动（评审实测） |
| 2026-10-16 | §2.1 新增 | 成本（Params/GFLOPs）+ TRT-fp16 b1/b4 首测 | A2 透明基准补全（`scripts/measure_spectrum_benchmarks.py`） |
