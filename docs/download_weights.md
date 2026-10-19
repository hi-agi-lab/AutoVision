# 权重下载与缓存

AutoVision 的 wheel / sdist **不包含权重**；权重按许可与体积独立分发，首次使用或
显式预取时下载。本文给出各类权重的来源、自动下载命令、缓存位置与排错方法。

> 先分清“可下载的上游权重”和“尚未发布的自研权重”。当前 `detr/n`、`detr/s`
> 仍是 spec-frozen，没有公开权重 URL；训练与发布流程见
> [模型发布](model_publish.md)。

## 1. 权重类别

| 类别 | 代表 | 来源 | 获取方式 |
|---|---|---|---|
| 上游完整检测器 | `rfdetr/rf-detr-{nano,small,medium,large}` | HF `Roboflow/rf-detr-*` | 首次 `pretrained=True` 自动下载 |
| 上游完整检测器 | `detr/gtrvit-{s,m,l,x}` | HF `Phoenix8125/GTR` | 运行 GTR 下载脚本 |
| 上游骨干 | `dinov2/*`、`dinov3/*` | HF `facebook/dinov2-*`、`facebook/dinov3-*` | 首次构建时自动下载；需许可门控 |
| 装配 warm-start | `d-fine/dfine_{s,m,l}_coco` | HF `Peterande/D-FINE`、GitHub Release、hf-mirror | `resolve_downloadable` / `detector_checkpoint=` 自动下载 |
| 自研完整检测器 | `detr/n`、`detr/s` | 尚未发布 | 自行训练、导出并按发布流程上传 |
| 用户自持检测器 | `dinov3-*` 7 个 | 仓库外私有存储 | **不提供下载**；3 个 STA-era 资产已拒绝 |

`autovision-cli download` 只下载**数据集**；模型权重使用本文的命令或首次加载时
自动获取。

## 2. 通用准备

当前以源码安装为主：

```bash
# 仓库根目录
pip install -e ".[rfdetr,export]"      # RF-DETR + ONNX 导出
pip install -e ".[transformers]"       # DINOv2 / DINOv3
pip install -e ".[download]"           # GTR 权重下载（huggingface_hub）
pip install -e ".[gtr]"                # GTR 前向（Triton/CUDA）
```

常用环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `HF_HOME` | HuggingFace snapshot 缓存根目录 | `~/.cache/huggingface` |
| `AUTOVISION_CACHE_DIR` | 可下载资产（D-FINE）缓存根目录；未设置时使用其下的 `downloadable_models/` | `~/.cache/autovision` |
| `AUTOVISION_GTR_ROOT` | GTR 权重目录 | `~/Research/Model/gtr` |
| `AUTOVISION_USER_HELD_ROOT` | 用户自持权重目录 | `~/Research/Model/AutoVision` |
| `AUTOVISION_MIRROR_URL` | HF 权重镜像；进程启动前设置 | — |
| `HF_ENDPOINT` | 直接指定 HuggingFace 端点 | 官方 Hub |
| `AUTOVISION_OFFLINE=1` | HF 权重、可下载模型与数据集下载只读本地缓存/拒绝联网 | — |
| `AUTOVISION_REQUIRE_DATASET_HASHES=1` | 数据集条目没有钉 sha256 时拒绝下载（数据集完整性门禁） | — |
| `AUTOVISION_DINOV2_WEIGHTS=1` | 接受 DINOv2 CC-BY-NC-4.0 后允许加载其权重 | — |
| `AUTOVISION_DINOV3_WEIGHTS=1` | 接受 Meta DINOv3 License 后允许加载其权重 | — |
| `AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1` | 允许加载未在 manifest 钉 sha256 的 EoMT hub 权重（可信镜像/离线副本；有风险） | — |
| `AUTOVISION_WEIGHT_SOURCE` | HF 骨干 / RF-DETR 下载源：`auto` / `modelscope` / `huggingface` | `auto` |

镜像变量必须在 Python 进程启动前设置（`huggingface_hub` 在导入时固定 endpoint）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或
export AUTOVISION_MIRROR_URL=https://hf-mirror.com
```

## 3. RF-DETR（自动下载）

```bash
pip install -e ".[rfdetr,export]"

python - <<'PY'
import autovision

# 首次调用会下载 HF snapshot 并按 pinned sha256 校验
model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
model.export("rfdetr-small.onnx")
PY
```

预取当前四个发布档：

```python
import autovision

for key in (
    "rfdetr/rf-detr-nano",
    "rfdetr/rf-detr-small",
    "rfdetr/rf-detr-medium",
    "rfdetr/rf-detr-large",
):
    autovision.detection(key, pretrained=True)
```

- 来源：HF `Roboflow/rf-detr-{nano,small,medium,large}`
- 许可：Apache-2.0，无需环境门控
- 缓存：`HF_HOME` 下的 snapshot 缓存
- 权重校验：`src/autovision/encoders/weight_hashes.py` 中的 pinned sha256

## 4. DINOv2 / DINOv3 骨干（自动下载 + 许可门控）

DINOv2 权重是 CC-BY-NC 4.0，DINOv3 权重受 DINOv3 License 约束。框架默认不下载；
设置环境变量表示你已自行接受对应条款。

### DINOv2

```bash
export AUTOVISION_DINOV2_WEIGHTS=1

python - <<'PY'
from autovision.encoders import get_wrapped_model

get_wrapped_model("dinov2/vits14", load_weights=True)
get_wrapped_model("dinov2/vitb14", load_weights=True)
PY
```

- 带 register：`facebook/dinov2-with-registers-*`
- 不带 register：`facebook/dinov2-*`（模型名加 `-noreg`）
- 配置构建、不下载权重：使用 `dinov2/<size>-notpretrained`

### DINOv3

```bash
export AUTOVISION_DINOV3_WEIGHTS=1

python - <<'PY'
from autovision.encoders import get_wrapped_model

get_wrapped_model("dinov3/convnext-tiny", load_weights=True)
get_wrapped_model("dinov3/vits16", load_weights=True)
PY
```

- 来源：HF `facebook/dinov3-*`
- `vitt16` / `vitt16plus` 没有公开 HF checkpoint，只能用
  `dinov3/vitt16-notpretrained` 配置构建，再加载自有权重
- 旧的 STA-era ViT 检查点（含 `backbone.sta.*`）已被加载器拒绝

这里的 `get_wrapped_model` 只预取骨干；`detr/dinov2-*`、`detr/dinov3-*` 的检测头
默认随机初始化，不等于完整 COCO 检测器。

## 5. GTR（显式脚本下载）

GTR 26 个官方权重不会在加载时自动联网，需要先运行下载脚本：

```bash
pip install -e ".[download]"

# 按任务下载（可重复 --task）
python scripts/download_gtr_assets.py --task det
python scripts/download_gtr_assets.py --task pose
python scripts/download_gtr_assets.py --task seg --task semseg --task depth --task obb

# 单个资产
python scripts/download_gtr_assets.py --asset gtrpose_s_coco

# 全部 26 个
python scripts/download_gtr_assets.py --all

# 仅校验本地文件
python scripts/download_gtr_assets.py --all --verify-only

# 只打印计划，不联网、不写文件
python scripts/download_gtr_assets.py --task pose --dry-run
```

常用选项：

```bash
export AUTOVISION_GTR_ROOT=/data/models/gtr
python scripts/download_gtr_assets.py --task pose --root /data/models/gtr
```

上游仓库若为 gated，先登录：

```bash
hf auth login            # 或设置 HF_TOKEN
```

脚本会逐文件核验 `src/autovision/registry/gtr_assets.py` 中 pinned sha256，并在
首次解析后打印上游 commit；需要可复现 URL 时用 `--revision <commit>`，并把该
commit 固定到 `GTR_HF_REVISION`。

下载完成后即可在训练/推理中引用：

```python
autovision.train_pose_estimation(
    ...,
    model="gtr/s",
    model_args={"checkpoint": "gtrpose_s_coco"},
)
```

GTR 前向另需 `gtr` extra：

```bash
pip install -e ".[gtr]"
```

## 6. D-FINE warm-start（自动下载）

D-FINE 权重用于 `detr/s` 等装配，不是完整 AutoVision 检测器：

```python
from autovision.registry import resolve_downloadable

path = resolve_downloadable("d-fine/dfine_s_coco")
print(path)
```

也可以直接传给检测构建：

```python
model_args={"detector_checkpoint": "d-fine/dfine_s_coco"}
```

- 可下载：`d-fine/dfine_s_coco`、`d-fine/dfine_m_coco`、`d-fine/dfine_l_coco`
- 来源：HF `Peterande/D-FINE`、hf-mirror、GitHub Release
- 许可：Apache-2.0
- 缓存：`AUTOVISION_CACHE_DIR` 下的 `d-fine/` 子目录；默认
  `~/.cache/autovision/downloadable_models/d-fine/`
- 下载后按 `src/autovision/registry/apache_assets.py` 的 pinned sha256 校验

## 7. 自研 `detr/n`、`detr/s`

当前 `RELEASED_SPECTRUM_WEIGHTS` 为空，`detr/n`、`detr/s` 是 spec-frozen，
**没有可下载权重**；`detr/m/l/x` 为 planned。`detr/dinov2-*` / `detr/dinov3-*` 只会
下载上游骨干，检测头仍需训练或提供检查点。

自研权重需要先训练、评测、冻结，再按[模型发布](model_publish.md)上传到 HF 并登记
`downloadable`。首次发布前仍需补齐 `spectrum_assets.py`、`RELEASED_SPECTRUM_WEIGHTS`
条目和 `detr/*` 的默认权重解析。

冒烟训练示例：

```bash
python scripts/train_spectrum.py --tier n --smoke --out runs/n-smoke
```

完整 COCO 训练、评测与发布命令见[模型发布](model_publish.md)。

## 8. 用户自持检测器

7 个第三方训练的检测器不在 `ASSET_MANIFEST`，不能通过 AutoVision 下载：

- 位置：`AUTOVISION_USER_HELD_ROOT`（默认 `~/Research/Model/AutoVision`）
- 其中 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 为 STA-era，
  STA 删除后加载器会明确拒绝
- 4 个 `dinov3-convnext-*` 资产在文件存在时可加载
- 许可为 `DINOv3-License+NOASSERTION`，禁再分发 / 商用；需自行从原始来源取得

详见[用户自持权重](user_held_weights.md)。

## 9. 校验与排错

查看当前真正带完整权重的模型：

```bash
autovision-cli list-models --task detection --pretrained --format names
```

当前应输出 4 个 RF-DETR 和 4 个 `detr/gtrvit-*`。`model-info <key>` 可查许可、
权重状态与导出能力。

常见问题：

- **401 / gated**：`hf auth login` 或设置 `HF_TOKEN`，然后重试 GTR/RF-DETR。
- **镜像不生效**：确认 `HF_ENDPOINT` 或 `AUTOVISION_MIRROR_URL` 在进程启动前设置；
  Xet 协议会被自动禁用，改走普通 HTTP。
- **sha256 不匹配**：HF 权重会自动删缓存重试一次；GTR 可加 `--force` 重新下载；
  D-FINE 删除对应缓存文件后重试。
- **离线环境**：`AUTOVISION_OFFLINE=1` 时 HF 权重只读缓存，GTR 脚本需 `--offline`
  或 `--verify-only`；D-FINE 缓存命中可离线使用，未命中仍会尝试下载。
- **缓存占满磁盘**：用 `HF_HOME`、`AUTOVISION_CACHE_DIR`、`AUTOVISION_GTR_ROOT`
  把三类缓存迁到数据盘。
- **不要用 `autovision-cli download` 下模型权重**：它只处理注册数据集。

## 10. 最小预取示例

公共、可再分发的一组权重（RF-DETR + GTR 检测 + D-FINE warm-start）：

```bash
pip install -e ".[rfdetr,export,download]"
python scripts/download_gtr_assets.py --task det

python - <<'PY'
import autovision
from autovision.registry import resolve_downloadable

autovision.detection("rfdetr/rf-detr-small", pretrained=True)
resolve_downloadable("d-fine/dfine_s_coco")
PY
```

需要 DINOv2 / DINOv3 时，在确认许可后单独设置门控并预取对应骨干。
