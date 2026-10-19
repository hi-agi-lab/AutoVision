# 模型发布指南

本文说明 AutoVision 的**权重/模型发布**流程：许可审计、冻结与读数、模型卡、HuggingFace
上传、可下载注册表、发布门禁、模型库登记与验证。

> **先分清两件事**：PyPI 发布的是框架代码（见[PyPI 发布指南](pypi_publish.md)）；
> 模型权重是独立发布物，单独许可、单独版本、单独钉 sha256，不进入框架的 wheel/sdist。

当前仓库**没有已发布的自研权重**：`detr/n`、`detr/s` 为 `spec-frozen`，
`RELEASED_SPECTRUM_WEIGHTS` 为空。第一次模型发布需要先把下文的“发布接线”补齐。

## 0. 发布对象与资产类别

| 类别 | 能否公开 | 发布方式 |
|---|---|---|
| 自研权重（Apache/MIT 兼容） | 可以 | 公开 HuggingFace repo + `downloadable` 注册表 |
| GTR 官方权重（MIT） | 可以 | 默认从上游 HF 按需下载；自建镜像需保留 LICENSE/NOTICE 并登记 revision + sha256 |
| warm-start 素材（Apache-2.0，如 d-fine） | 可以 | 已有 `ASSET_MANIFEST` / `resolve_downloadable` 模式 |
| 用户自持 7 个 DINOv3 检测器 | **不可以** | 仓库外私有库；边界见[用户自持权重](user_held_weights.md) |
| 受限许可权重（CC-BY-NC / DINOv3 License） | **不可以** | 不公开、不随导出产物分发 |

### 0.1 当前代码状态

已具备：

- 模型卡 schema：`src/autovision/registry/model_card.py`
- 谱系发布门禁：`src/autovision/registry/spectrum_release.py`
- 可下载资产缓存 + sha256 复核：`src/autovision/registry/downloadable.py`
- 资产许可元数据：`src/autovision/registry/assets.py`
- 模型库生成器：`scripts/gen_model_zoo.py`（`--check` 防漂移）
- Apache-2.0 检测器权重库准备：`scripts/prepare_apache_detector_weights.py`
- 训练 / 评测 / 延迟脚本：`scripts/train_spectrum.py`、
  `scripts/eval_spectrum_coco.py`、`scripts/measure_spectrum_benchmarks.py`
- 发布门禁测试：`tests/test_spectrum_release.py`

尚缺（首次发布前要补）：

- registry 包下的 `spectrum_assets.py`：把已发布权重注册进 `downloadable`；
- `RELEASED_SPECTRUM_WEIGHTS` 的条目与 `SPECTRUM` 状态翻转；
- `detr/*` 构建入口对 released 权重的默认解析/加载；
- `gen_model_zoo.py` 的“已发布档位权重”表；
- 第一版权重本身（n/s 或 m/l/x）。

## 1. 发布前决策

### 1.1 许可审计（第一步，不是最后一步）

必须同时核对：

- **代码**：框架 Apache-2.0，detr/gtr 等模块的许可证；
- **教师权重**：RF-DETR（Apache-2.0）、D-FINE（Apache-2.0）、GTR（MIT）是干净路径；
  用户自持 DINOv3 权重（`DINOv3-License+NOASSERTION`）不可再分发，公开前必须做法律评估；
- **训练数据**：COCO 图像/标注的再分发条款；蒸馏/微调不会自动“洗掉”上游限制；
- **检测头 / 第三方组件**：是否声明保留全部权利；
- **发布形态**：权重、导出件、模型卡、NOTICE 是否覆盖所有义务。

结论只能是两类：

- **可公开再分发** → HuggingFace 公开 repo + 注册表；
- **不可再分发** → 用户自持库/私有 repo；仓库只登记许可边界，不登记 URL。

### 1.2 发布格式

建议一次发布包含：

| 文件 | 用途 |
|---|---|
| `weights.pt` | 权重/推理用导出版（按加载器约定） |
| `warm_start.pt` | 可选：微调初始化（AVWarmStart） |
| `weights.pt.modelcard.json` | `ModelCard.save()` 产出的机器可读卡 |
| `README.md` | HuggingFace 模型卡正文（YAML front matter） |
| `LICENSE` / `NOTICE` | 许可与第三方溯源 |
| `config.json` | `model_init_args`、输入尺寸、类别数等 |

格式必须与 `detr/*` 的加载语义一致；`exported_best.pt` 可直接测 AP/预测，
`warm_start` 用于继续微调。模型卡里要写清楚“哪个文件用哪种加载方式”。

### 1.3 版本

- 模型版本与框架版本**独立**：例如框架 `0.3.0`，权重 `v1.0.0`。
- HF 上不要覆盖已发布文件；新权重 = 新 commit / 新 tag。
- 注册表钉 **commit + sha256**，同时写进模型卡 `version`。
- 发布后不要重新训练同一版本；“修一下再传”也必须升版本。

## 2. 冻结权重并产出可信读数

### 2.1 训练与评测

```bash
# 以 n 档为例（s 档按配方传 --warm-start）
python scripts/train_spectrum.py --tier n --data coco.yaml \
    --teacher rfdetr/rf-detr-nano --batch-size 16 --steps 88715 \
    --out runs/detr_n_coco

# 全量 COCO val2017（5000 图）
python scripts/eval_spectrum_coco.py \
    --weights runs/detr_n_coco/exported_models/exported_best.pt \
    --data coco.yaml --split val --out runs/detr_n_coco/eval_val2017.json \
    --max-images 5000
```

要求：

- 固定 seed，记录在模型卡；
- 全量 val2017，不用短程子集代替发布读数；
- 同时记录 params / GFLOPs 与延迟（engine / 全链、硬件、精度、batch）；
- 延迟复现脚本用 `scripts/measure_spectrum_benchmarks.py`；
- 读数的口径必须写清楚，不能把官方数字混进自家读数（见[基准](benchmarks.md)）。

### 2.2 冻结

评测通过后：

- 不再改权重、不重新训练；
- 记录训练 commit、教师资产、warm-start、数据配置；
- 计算 sha256；
- 保存 `model_init_args` / `config.json`。

## 3. 模型卡

使用 `src/autovision/registry/model_card.py` 的 `ModelCard`。最低字段：

| 字段 | 要求 |
|---|---|
| `model` | 必须是 `detr/{tier}` |
| `name` | 人类可读名称 |
| `train_dataset` | 数据集与 split |
| `schedule` | 如 `12ep distill` / `6ep finetune` |
| `seed` | **自研权重必填** |
| `metrics` | 至少 `coco_map`；建议 `coco_map50`、`params_m` |
| `latency` | 至少一条；含 backend / precision / hardware / batch / p50 / script |
| `license` | SPDX id，必填 |
| `weights_sha256` | 文件 → 64 hex，必填 |
| `provenance` | 教师、warm-start、训练 run |
| `known_limitations` | 适用范围与已知缺口 |
| `version` | 权重版本或 HF revision |

```python
from autovision.registry.model_card import ModelCard

card = ModelCard(
    model="detr/n",
    name="DETR-N (HGNetV2-B0 + D-FINE decoder)",
    train_dataset="COCO 2017 train2017",
    schedule="12ep distill (RF-DETR-n teacher)",
    seed=0,
    metrics={"coco_map": 0.0, "coco_map50": 0.0, "params_m": 17.8},
    license="Apache-2.0",
    weights_sha256={"weights.pt": "<64-hex>"},
)
card.validate()
card.save("weights.pt")   # 生成 weights.pt.modelcard.json
```

发布门禁比 `validate()` 更严：还要求全量 AP > 0、至少一条延迟、延迟脚本真实存在、
seed 非空、资产已注册且 sha256 与卡一致。

## 4. 上传 HuggingFace

### 4.1 准备

```bash
pip install -U huggingface_hub
hf auth login
```

公开仓库只放可再分发的权重；私有/用户自持走私有 repo 或本地库。

```bash
hf repo create ORG/autovision-detr-n-coco --repo-type model --public

# release/ 目录示例：
#   README.md  LICENSE  NOTICE  weights.pt  warm_start.pt
#   weights.pt.modelcard.json  config.json
hf upload ORG/autovision-detr-n-coco ./release . --repo-type model
```

### 4.2 模型卡正文

HuggingFace 的 `README.md` 建议包含 YAML front matter：

```yaml
---
license: apache-2.0
library_name: autovision
datasets:
  - coco
tags:
  - object-detection
  - detr
---
```

正文至少写：

- 模型是什么、对应 AutoVision 的哪个 model 键；
- 训练数据 / 教师 / schedule / seed；
- COCO val2017 AP / AP50 与评测脚本；
- 延迟协议、硬件、精度、batch；
- 一行加载 / 微调代码；
- 许可与已知限制。

### 4.3 钉住 revision 与 sha256

```bash
sha256sum release/weights.pt
python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("ORG/autovision-detr-n-coco")
print(info.sha)
PY
```

下载 URL 用 commit 固定，不要依赖会移动的 `main`：

```text
https://huggingface.co/ORG/autovision-detr-n-coco/resolve/<commit>/weights.pt
```

## 5. 注册到框架

### 5.1 新增 spectrum 资产注册

现有 `downloadable` 注册表已经提供 URL + sha256 + license + mirror + 缓存复核。
在 registry 包下新增 `spectrum_assets.py`（当前尚未创建）：

```python
from autovision.registry.assets import LicenseRef
from autovision.registry.downloadable import register_downloadable

_HF = "https://huggingface.co/ORG/autovision-detr-n-coco/resolve/<commit>"
_HF_MIRROR = "https://hf-mirror.com/ORG/autovision-detr-n-coco/resolve/<commit>"

def register_spectrum_assets() -> None:
    register_downloadable(
        "detr_n_coco",  # 建议无斜杠；见 5.2 的缓存路径注意
        url=f"{_HF}/weights.pt",
        sha256="<64-hex>",
        license=LicenseRef(
            id="Apache-2.0",
            origin="https://huggingface.co/ORG/autovision-detr-n-coco@<commit>",
        ),
        mirrors=(f"{_HF_MIRROR}/weights.pt",),
    )
```

在 `src/autovision/registry/__init__.py` 里 import 该模块，让 `ASSET_MANIFEST` 在
import 时完成注册（与 `apache_assets` 同样的模式）。

### 5.2 缓存路径注意

`resolve_downloadable` 会把缓存文件写成 `cache_root / f"{name}.pt"`；资产名带 `/`
（如 `d-fine/dfine_s_coco`）时，缓存父目录会被自动创建。新自研资产仍建议使用
无斜杠名（`detr_n_coco`），便于跨平台迁移和人工查找。

### 5.3 发布门禁与状态

在 `spectrum_release.py` 里登记：

```python
RELEASED_SPECTRUM_WEIGHTS["n"] = ReleasedSpectrumWeight(
    tier="n",
    asset="detr_n_coco",
    weights_file="weights.pt",
    card=card,
)
```

并把 `SPECTRUM["n"].status` 改为 `"released"`。当前门禁会检查卡字段、seed、AP、
延迟、脚本与文件名；**首次发布前还应补一条检查**：`asset` 必须存在于
`ASSET_MANIFEST`，且 `ASSET_MANIFEST[asset].sha256 == card.weights_sha256[weights_file]`。

### 5.4 默认加载

发布门禁只证明“可发布”，不自动让用户一行加载。还需要在 `detr/*` 的构建入口把
released 资产解析成对应的 `checkpoint` / `warm_start`，并在用户没有显式传权重时默认
加载；否则用户仍要手工写 `model_args`，不算完成发布。

### 5.5 模型库与测试

- `scripts/gen_model_zoo.py`：增加“已发布档位权重”表，列出 tier、HF repo、license、
  AP、延迟、sha256；数据来自 `ModelCard`，不要手写。
- `docs/benchmarks.md`：新增自研行（AP / AP50 / params / GFLOPs / 延迟），与模型卡一致。
- `tests/test_spectrum_release.py`：更新“尚无 released 档位”的断言；新增资产注册、
  sha256 一致性、门禁通过测试。
- `tests/test_model_zoo.py`：`gen_model_zoo.py --check` 必须通过。

## 6. 验证

```bash
# 冷缓存自动下载 + sha256 + 加载
rm -rf ~/.cache/autovision/downloadable_models/detr_n_coco.pt
python - <<'PY'
import autovision
model = autovision.detection("detr/n")
print(model)
PY

# 评测/预测 smoke
python scripts/eval_spectrum_coco.py \
    --weights runs/detr_n_coco/exported_models/exported_best.pt \
    --data coco.yaml --max-images 8
```

发布前门禁：

```bash
pytest tests/test_spectrum_release.py tests/test_model_zoo.py -q
python scripts/gen_model_zoo.py --check
python scripts/check_docs_links.py
mkdocs build --strict
```

同时确认：

- `docs/model_zoo.md` 中状态是 `released`，并有权重表；
- README / known_limits 不再把该档位写成 spec-frozen；
- CHANGELOG 记录权重版本与训练 commit；
- 离线环境能用缓存或本地文件加载（`AUTOVISION_OFFLINE`）。

## 7. 第三方与受限权重

- **RF-DETR**：权重由 Roboflow 托管；我们只维护模型卡/元数据，不重复上传。
- **GTR 官方权重（MIT）**：二进制不随仓库分发；用
  `scripts/download_gtr_assets.py` 从上游 HF `Phoenix8125/GTR` 按需下载到
  `AUTOVISION_GTR_ROOT`（需 `pip install -e ".[download]"`），逐文件核验
  sha256；上游 gated 时先 `hf auth login`，区域镜像用 `HF_ENDPOINT`。
  若要自建镜像，MIT 允许，但需保留上游 LICENSE/NOTICE 并登记 revision+sha256。

  ```bash
  pip install -e ".[download]"
  python scripts/download_gtr_assets.py --task det
  python scripts/download_gtr_assets.py --all --verify-only
  ```
- **用户自持 7 个 DINOv3 检测器**：`DINOv3-License+NOASSERTION`，禁再分发/禁商用；
  只登记资产名、sha256 与许可边界，绝不进入公开注册表或下载清单。
- **受限许可权重**（DINOv2 CC-BY-NC / DINOv3 License）：默认不下载，需显式自证；
  不随仓库/导出产物分发。

## 8. 发布检查清单

- [ ] 许可审计完成（代码 / 数据 / 教师 / 头 / 第三方）
- [ ] 权重冻结，训练 commit 与配置已记录
- [ ] 全量 COCO val2017 AP / AP50 已产出
- [ ] params / GFLOPs / 延迟（engine + 全链）已产出
- [ ] 复现脚本路径真实存在
- [ ] `ModelCard.validate()` 通过
- [ ] HF 公开/私有仓库已上传
- [ ] revision + sha256 已钉住，URL 使用 commit
- [ ] `spectrum_assets.py` 注册完成
- [ ] 门禁通过，`SPECTRUM` 状态改为 `released`
- [ ] `detr/*` 默认加载已接通
- [ ] `gen_model_zoo.py` 重新生成并通过 `--check`
- [ ] `docs/benchmarks.md` / README / known_limits / CHANGELOG 已更新
- [ ] 冷缓存下载 + 加载 + 评测 smoke 通过
- [ ] 权重版本 tag 与模型卡 `version` 一致

## 9. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 门禁报 “no published weight asset” | `RELEASED_SPECTRUM_WEIGHTS` 未登记 | 先注册资产，再翻状态 |
| 门禁报 “no full-COCO AP” | 卡里没有有效 `coco_map` | 跑全量 val2017 并回填 |
| 门禁报 “latency script does not exist” | 脚本路径写错/未入库 | 用仓库内真实脚本路径 |
| 门禁报 sha 不一致 | HF 文件变了或卡写错 | 重新计算 sha256，重新上传新版本 |
| 下载 404 | 用了会移动的 `main` 或文件名错 | 改用 commit URL，核对文件名 |
| 下载后 sha 校验失败 | 镜像/缓存旧文件 | 清缓存后重试，核对主源 |
| 冷缓存带 `/` 资产名失败 | 父目录未创建 | 用无斜杠资产名，或补 parent mkdir |
| GTR 下载 401 / gated | 上游需要认证 | `hf auth login` 或设置 `HF_TOKEN` |
| GTR 下载 sha 不符 | revision 漂移 / 镜像旧文件 | 用 `--revision` 钉住，清理缓存重试 |
| 用户仍要手写 model_args | 默认加载未接线 | 在 `detr/*` 构建入口接 released 资产 |
| 模型库页面漂移 | 手改了 `docs/model_zoo.md` | 重新运行 `scripts/gen_model_zoo.py` |
| 误把受限权重公开 | 许可审计缺失 | 立即撤下/yank，保留本地私有库 |
| 覆盖了已发布权重 | 直接改 HF main | 不要覆盖；发新 commit/tag/版本 |

## 10. 一句话

模型发布不是“上传一个 `.pt`”，而是：

> **权重 + SPDX 许可 + 模型卡 + 全量 AP/延迟 + 可复现脚本 + 注册表钉 sha256 +
> 一键加载，全部齐了才算 `released`。**

关键文件：`src/autovision/registry/model_card.py`、
`src/autovision/registry/spectrum_release.py`、
`src/autovision/registry/downloadable.py`、
`src/autovision/registry/assets.py`、
`src/autovision/registry/gtr_assets.py`、
`src/autovision/registry/gtr_download.py`、
`scripts/gen_model_zoo.py`、`scripts/train_spectrum.py`、
`scripts/eval_spectrum_coco.py`、`scripts/measure_spectrum_benchmarks.py`、
`scripts/download_gtr_assets.py`。
