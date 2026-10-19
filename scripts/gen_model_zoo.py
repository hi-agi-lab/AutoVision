#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Generate ``docs/model_zoo.md`` from the weight registries (E2 / release package).

The page is machine-generated so the documented identity columns
(license / sha256 / official metric / model key) cannot drift from what
the code enforces. Regenerate after any registry change:

    python scripts/gen_model_zoo.py           # rewrite docs/model_zoo.md
    python scripts/gen_model_zoo.py --check   # exit 1 if the page is stale

``--check`` runs in CI and in ``tests/test_model_zoo.py``.

Three inputs, three kinds of data:

- **Derived** (single source, never curated here): asset names, sha256,
  licenses, official metrics and the GTR ↔ task mapping from
  ``autovision.registry.gtr_assets``; the HF pull list from
  ``autovision.encoders.weight_hashes``; the downloadable warm-start set
  from ``autovision.registry.apache_assets``; the user-held store from
  ``autovision.registry.user_held_assets``; product tiers from the DETR
  ``SPECTRUM`` table; open-vocab text-tower keys from
  ``tasks.detection.shared.open_vocab``.
- **Curated and cited** (belongs to the *page*, not to code): the official
  GTR detection latencies (GTR paper table, mirrored in
  ``docs/benchmarks.md`` §3, 引用日期 2026-10-15) and the one-line role
  text for every HF manifest entry (completeness asserted below, so a new
  manifest entry fails generation until it gets a role).
- **Not on this page**: our own reproduced readings — those live in
  ``docs/benchmarks.md`` under its immutable protocol.

Deliberately absent: any local filesystem path or environment probe
(the page must be byte-identical across machines).
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "docs" / "model_zoo.md"

from autovision.encoders.weight_hashes import WEIGHT_MANIFEST  # noqa: E402
from autovision.registry.apache_assets import register_apache_assets  # noqa: E402
from autovision.registry.downloadable import ASSET_MANIFEST  # noqa: E402
from autovision.registry.gtr_assets import GTR_ASSETS  # noqa: E402
from autovision.registry.model_cards import rfdetr_model_cards  # noqa: E402
from autovision.registry.spectrum_release import (  # noqa: E402
    validate_spectrum_release,
)
from autovision.registry.user_held_assets import USER_HELD_ASSETS  # noqa: E402
from autovision.tasks.detection.detr.variants import SPECTRUM  # noqa: E402
from autovision.tasks.detection.shared.open_vocab import (  # noqa: E402
    _TEXT_ENCODER_SPECS,
)

register_apache_assets()

#: RF-DETR official cards (license / sha256 / COCO AP / params / input).
_RF_DETR_CARDS = rfdetr_model_cards(WEIGHT_MANIFEST)

#: Fail closed: a tier marked released without its asset/card/AP/latency
#: must not be rendered as a public model.
_RELEASE_PROBLEMS = validate_spectrum_release(SPECTRUM)
if _RELEASE_PROBLEMS:
    raise SystemExit(
        "spectrum release gate failed:\n  - " + "\n  - ".join(_RELEASE_PROBLEMS)
    )

#: Official published latency for the GTR detection tiers — GTR paper
#: table (RTX 4090 / FP16 / batch 1 / torch.compile + CUDA Graphs,
#: 640x640), mirrored in docs/benchmarks.md §3 (引用日期 2026-10-15).
#: Keyed by registry asset name; asserted complete below.
_GTR_DET_LATENCY_MS: dict[str, float] = {
    "gtr_s_coco": 1.225,
    "gtr_m_coco": 1.462,
    "gtr_l_coco": 1.908,
    "gtr_x_coco": 2.115,
}

#: One-line role for every WEIGHT_MANIFEST entry: repo_id -> (接入键, 说明).
#: "未接线" means the pin exists (sha256 verified) but no public model key
#: consumes it yet; external converted weights may still need a per-card
#: license review (DEIMv2 upstream is non-commercial).
_HF_ROLES: dict[str, tuple[str, str]] = {
    "Roboflow/rf-detr-nano": ("rfdetr/rf-detr-nano", "检测变体（HF transformers 路径；`-notpretrained` 可离线构建）"),
    "Roboflow/rf-detr-small": ("rfdetr/rf-detr-small", "同上"),
    "Roboflow/rf-detr-medium": ("rfdetr/rf-detr-medium", "同上"),
    "Roboflow/rf-detr-base": ("rfdetr/rf-detr-base", "同上"),
    "Roboflow/rf-detr-large": ("rfdetr/rf-detr-large", "同上"),
    "google/siglip-base-patch16-224": ('open_vocab.text_encoder="siglip-base"', "开放词表文本塔（SigLIP）"),
    "harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers": ("（未接线）", "外部转换权重；DEIMv2 上游非商用，许可需按卡复核"),
    "harshaljanjani/DEIMv2_HGNetv2_PICO_COCO_Transformers": ("（未接线）", "外部转换权重（PICO 档）；许可需按卡复核"),
    "harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers": ("（未接线）", "外部转换权重（DINOv3 骨干）；DINOv3/DEIMv2 条款需按卡复核"),
    "openai/clip-vit-base-patch32": ('open_vocab.text_encoder="clip-b32"', "开放词表文本塔（CLIP）"),
    "openai/clip-vit-base-patch16": ('open_vocab.text_encoder="clip-b16"', "开放词表文本塔（CLIP）"),
    "facebook/dinov2-small": ("dinov2/vits14-noreg", "骨干（不带 register）"),
    "facebook/dinov2-base": ("dinov2/vitb14-noreg", "骨干（不带 register）"),
    "facebook/dinov2-large": ("dinov2/vitl14-noreg", "骨干（不带 register）"),
    "facebook/dinov2-giant": ("dinov2/vitg14-noreg", "骨干（不带 register）"),
    "facebook/dinov2-with-registers-small": ("dinov2/vits14", "骨干（默认带 register）"),
    "facebook/dinov2-with-registers-base": ("dinov2/vitb14", "骨干（默认带 register）"),
    "facebook/dinov2-with-registers-large": ("dinov2/vitl14", "骨干（默认带 register）"),
    "facebook/dinov2-with-registers-giant": ("dinov2/vitg14", "骨干（默认带 register）"),
    "facebook/dinov3-vits16-pretrain-lvd1689m": ("dinov3/vits16", "骨干（LVD-1689M 预训练）"),
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": ("dinov3/vits16plus", "骨干（+ 蒸馏版）"),
    "facebook/dinov3-vitb16-pretrain-lvd1689m": ("dinov3/vitb16", "骨干"),
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": ("dinov3/vith16plus", "骨干（H 档 + 蒸馏版）"),
    "facebook/dinov3-vitl16-pretrain-lvd1689m": ("dinov3/vitl16", "骨干（L 档）"),
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": ("dinov3/vit7b16", "骨干（7B，多文件分片）"),
    "facebook/dinov3-vitl16-pretrain-sat493m": ("dinov3/vitl16-sat493m", "骨干（卫星预训练 SAT-493M）"),
    "facebook/dinov3-vit7b16-pretrain-sat493m": ("dinov3/vit7b16-sat493m", "骨干（卫星预训练，7B）"),
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": ("dinov3/convnext-tiny", "骨干（ConvNeXt）"),
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": ("dinov3/convnext-small", "骨干（ConvNeXt）"),
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": ("dinov3/convnext-base", "骨干（ConvNeXt）"),
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": ("dinov3/convnext-large", "骨干（ConvNeXt）"),
}

#: License group order for the HF table (page presentation).
_LICENSE_GROUPS: list[tuple[str, str]] = [
    ("apache-2.0", "Apache-2.0（可自由再分发）"),
    ("mit", "MIT（可自由再分发）"),
    ("cc-by-nc-4.0", "CC-BY-NC-4.0（禁商用；默认不下载，需显式自证）"),
    ("dinov3-custom (see repository card)", "DINOv3 License（默认不下载，需显式自证；条款以仓库卡片为准）"),
]

_TASK_ORDER = ["det", "det-obj365", "seg", "semseg", "depth", "obb", "pose"]
_TASK_LABEL = {
    "det": "检测（COCO 2017，AP）",
    "det-obj365": "检测初始化（Objects365 预训练）",
    "seg": "实例分割（COCO，mask AP）",
    "semseg": "语义分割（Cityscapes，mIoU）",
    "depth": "单目深度（NYU，absRel）",
    "obb": "旋转框（DOTA，AP50）",
    "pose": "姿态（COCO 关键点，kp AP）",
}
_METRIC_LABEL = {
    "coco_ap": "AP",
    "coco_mask_ap": "mask AP",
    "coco_kp_ap": "kp AP",
    "nyu_absrel": "absRel",
    "dota_ap50": "AP50",
    "cityscapes_miou": "mIoU",
}
_LOADERS = {
    "det": "train_object_detection",
    "seg": "train_instance_segmentation",
    "semseg": "train_semantic_segmentation",
    "depth": "train_depth_estimation",
    "obb": "train_oriented_object_detection",
    "pose": "train_pose_estimation",
}


def _variant_of(asset_name: str) -> str:
    """``gtrseg_m_coco`` -> ``m`` (registry naming: the size token after the task)."""
    tokens = asset_name.split("_")
    for token in tokens:
        if token in ("s", "m", "l", "x"):
            return token
    raise AssertionError(f"Cannot derive the variant size from asset '{asset_name}'")


def _fmt_metric(asset) -> str:
    if asset.metric_name is None or asset.metric_value is None:
        return "—（预训练权重，无基准行）" if asset.task == "det-obj365" else "—"
    label = _METRIC_LABEL[asset.metric_name]
    digits = 3 if asset.metric_name == "nyu_absrel" else 1
    return f"{label} {asset.metric_value:.{digits}f}"


def _check_inputs() -> None:
    det_assets = {n for n, a in GTR_ASSETS.items() if a.task == "det"}
    assert det_assets == set(_GTR_DET_LATENCY_MS), (
        "official GTR detection latency table out of sync with the registry: "
        f"{sorted(det_assets)} vs {sorted(_GTR_DET_LATENCY_MS)}"
    )
    missing = sorted(set(WEIGHT_MANIFEST) - set(_HF_ROLES))
    assert not missing, (
        "WEIGHT_MANIFEST entries without a curated zoo role (add to _HF_ROLES): "
        f"{missing}"
    )
    stale = sorted(set(_HF_ROLES) - set(WEIGHT_MANIFEST))
    assert not stale, f"_HF_ROLES entries absent from WEIGHT_MANIFEST: {stale}"
    unknown_metrics = {
        a.metric_name
        for a in GTR_ASSETS.values()
        if a.metric_name is not None and a.metric_name not in _METRIC_LABEL
    }
    assert not unknown_metrics, f"unmapped GTR metric names: {sorted(unknown_metrics)}"
    # Presentation coverage: a new task group or license family must get a
    # heading/order entry, otherwise its assets would silently vanish from
    # the tables while the overview count still includes them.
    unlisted_tasks = {a.task for a in GTR_ASSETS.values()} - set(_TASK_ORDER)
    assert not unlisted_tasks, f"GTR task groups missing from _TASK_ORDER: {unlisted_tasks}"
    unlisted_licenses = {a.license for a in WEIGHT_MANIFEST.values()} - {
        license_id for license_id, _ in _LICENSE_GROUPS
    }
    assert not unlisted_licenses, (
        f"WEIGHT_MANIFEST license families missing from _LICENSE_GROUPS: "
        f"{unlisted_licenses}"
    )
    # Open-vocab text towers must stay sha256-pinned (doc 27 gate G1) — the
    # page documents them as pinned assets, so an unpinned tower is a hard
    # inconsistency, not a cosmetic one.
    for name, spec in _TEXT_ENCODER_SPECS.items():
        repo_id = spec["repo_id"]
        if repo_id is None:
            continue
        assert repo_id in WEIGHT_MANIFEST, (
            f"open-vocab text tower '{name}' ({repo_id}) is not pinned in "
            "WEIGHT_MANIFEST"
        )


def _hf_main_file(artifact) -> str:
    return sorted(artifact.sha256)[0]


def _render() -> str:
    _check_inputs()
    out: list[str] = []
    add = out.append

    add("# 模型库\n")
    add(
        "> 本页由 `scripts/gen_model_zoo.py` 从权重注册表机器生成（许可 / sha256 / "
        "官方指标 / 接入键），**勿手改**；注册表变更后重新生成，漂移由\n"
        "> `scripts/gen_model_zoo.py --check` 与 `tests/test_model_zoo.py` 拦截。\n"
        ">\n"
        "> 仓库**不携带权重二进制**：注册表钉身份（sha256），不钉分发。表中数字为"
        "**官方发表值**（出处见各节）；我方复现读数与协议见[基准](benchmarks.md)，"
        "许可红线见仓库根 `NOTICE`。\n"
    )

    # ── §1 overview ────────────────────────────────────────────────────
    add("## 1. 资产概览\n")
    add("| 类别 | 数量 | 许可 | 分发形态 | 注册表 |")
    add("|---|---|---|---|---|")
    add(
        f"| GTR 官方家族 | {len(GTR_ASSETS)} | MIT | 来源 HF `Phoenix8125/GTR`，"
        "按需下载到 `AUTOVISION_GTR_ROOT`（仓库不携带二进制） "
        "| `src/autovision/registry/gtr_assets.py` |"
    )
    add(
        f"| HF 权重清单 | {len(WEIGHT_MANIFEST)} | Apache-2.0 / MIT / CC-BY-NC-4.0 / "
        "DINOv3 License | HF 下载 + sha256 校验（受限许可门控） | "
        "`src/autovision/encoders/weight_hashes.py` |"
    )
    add(
        f"| 可下载资产（warm-start） | {len(ASSET_MANIFEST)} | Apache-2.0 | "
        "HF / GitHub 镜像下载 + sha256 校验 | `src/autovision/registry/apache_assets.py` |"
    )
    add(
        f"| 用户自持库 | {len(USER_HELD_ASSETS)} | DINOv3-License+NOASSERTION | "
        "仓库外存储，**不可再分发** | `src/autovision/registry/user_held_assets.py` |"
    )
    add("")
    add(
        "当前环境可用的模型键（按已安装 extras 过滤）：`autovision-cli list-models`。\n"
    )

    # ── §2 GTR family ──────────────────────────────────────────────────
    add("## 2. GTR 多任务家族（MIT，旗舰）\n")
    add(
        "整族共享一套 GLA 骨干（`ViTAdapterSpatialSwiGLU`）+ `GTREncoder`，"
        "各任务只换头 / 解码器；权重为官方 **EMA 口径**（官方所有发表数字均用 EMA）。\n"
    )
    add(
        "权重二进制不随仓库分发：用 `scripts/download_gtr_assets.py` 从上游 HF "
        "`Phoenix8125/GTR` 按需下载到 `AUTOVISION_GTR_ROOT`（默认 "
        "`~/Research/Model/gtr`），逐文件核验 sha256；上游 revision 可钉住"
        "（下载器会在首次解析后打印实际 commit）。上游若为 gated，先 "
        "`hf auth login`；区域镜像用 `HF_ENDPOINT`。\n"
    )
    add("| 任务 | 变体 | 资产 | 官方指标 | 官方延迟 (ms) | sha256 |")
    add("|---|---|---|---|---|---|")
    for task in _TASK_ORDER:
        rows = [a for a in GTR_ASSETS.values() if a.task == task]
        rows.sort(key=lambda a: _variant_of(a.name))
        for asset in rows:
            latency = _GTR_DET_LATENCY_MS.get(asset.name)
            latency_cell = f"{latency:.3f}" if latency is not None else "—"
            add(
                f"| {_TASK_LABEL[task]} | {_variant_of(asset.name)} | `{asset.name}` "
                f"| {_fmt_metric(asset)} | {latency_cell} | `{asset.sha256[:12]}` |"
            )
    add("")
    add(
        "- 官方延迟 = GTR 论文协议（RTX 4090 / FP16 / batch 1 / torch.compile + "
        "CUDA Graphs / 640×640），仅检测档发表；其余任务官方未发表延迟"
        "（我方 torch 口径读数见[基准](benchmarks.md) §2）。引用日期 2026-10-15。\n"
        "- sha256 为完整摘要前 12 位；完整值以注册表为准。\n"
    )
    add("### 2.1 `gtr/*` 任务族（五任务直载）\n")
    add(
        "`model_args` 的 `checkpoint` 接受注册资产名或路径（`resolve_gtr_asset` "
        "解析 + sha256 核验 + 加载覆盖门：encoder/decoder 覆盖率不足即报错）：\n"
    )
    add("```python")
    add("import autovision")
    add("")
    add(
        '# 姿态；其余四个任务族同为 `model="gtr/{s,m,l,x}"` + '
        '`model_args={"checkpoint": "<资产名>"}`'
    )
    add(
        'autovision.train_pose_estimation(out="runs/pose-s", data="pose.yaml", '
        'model="gtr/s",\n'
        '    model_args={"checkpoint": "gtrpose_s_coco"})'
    )
    for task in ("semseg", "depth", "seg", "obb"):
        asset = next(a for a in GTR_ASSETS.values() if a.task == task)
        add(
            f'autovision.{_LOADERS[task]}(..., model="gtr/s", '
            f'model_args={{"checkpoint": "{asset.name}"}})   # {_TASK_LABEL[task]}'
        )
    add("```\n")
    add("离线核验（不加载模型，仅对文件做 sha256 核验）：\n")
    add("```python")
    add("from autovision.registry.gtr_assets import resolve_gtr_asset")
    add(
        'resolve_gtr_asset("gtrpose_s_coco")   # 文件不存在 / sha256 不符即抛 GTRAssetError'
    )
    add("```\n")
    add("### 2.2 检测线（`detr/gtrvit-s`）\n")
    add(
        "检测的官方权重经一步**装配**（`build_dfine_official_warm_start`，"
        "EMA 口径 + 解码器覆盖率门）后以 `model_args.warm_start` 注入；"
        "`detr/gtrvit-s` 的 GLA 骨干另由 `extract_gtr_backbone` 提取为 "
        "`backbone_weights`。以下写法在 v0.2.0 已实机验证（装配报告："
        "`decoder_coverage=1.0`、`skipped_shape=0`，短程训练闭环通过）：\n"
    )
    add("```python")
    add("import autovision")
    add("from autovision.registry.gtr_assets import resolve_gtr_asset")
    add(
        "from autovision.tasks.detection.detr.dfine_official_warm_start import (\n"
        "    build_dfine_official_warm_start,\n"
        ")"
    )
    add("")
    add('asset = resolve_gtr_asset("gtr_s_coco")            # MIT；sha256 核验')
    add("# `detr/gtrvit-s`（GLA 骨干线，需 `gtr` extra + CUDA）：装配 encoder+decoder，")
    add("# 骨干状态单独提取")
    add("import torch")
    add("from autovision.tasks.detection.detr.gtr_teacher import (")
    add("    extract_gtr_backbone,")
    add("    load_gtr_teacher,")
    add(")")
    add("")
    add('build_dfine_official_warm_start("gtrvit-s", out="ws_gtrvit_s.pt", checkpoint=asset.path,')
    add('    backbone_pretrained=False, license_id="MIT", origin=asset.source,')
    add('    source_format="gtr")')
    add('torch.save(extract_gtr_backbone(load_gtr_teacher(asset.path)), "gtr_s_backbone.pth")')
    add('autovision.train_object_detection(out="runs/gtrvit-s", data="coco.yaml",')
    add('    model="detr/gtrvit-s", model_args={"warm_start": "ws_gtrvit_s.pt",')
    add('    "backbone_weights": "gtr_s_backbone.pth"})')
    add("```\n")
    add("GTR 变体补充（2026-10-17）：\n")
    add(
        "- `gtrvit-{s,m,l,x}`：完整 GTR ViT 检测目标；M/L/X 已注册，装配后 "
        "encoder/decoder coverage 1.0（L/X 的 ViT 384 宽，encoder/decoder 256 宽，"
        "由 `backbone_embed_dim` 区分）。推理应走这一线。"
    )
    add(
        "- Objects365 检测头是 **366** 类（365 数据集类 + no-object）；装配时用 "
        "`GTRAsset.model_num_classes`，不要传 365（会丢掉 class-dependent tensors）。"
    )
    add("- `gtr/*` 实例分割任务族现支持 `s/m/l/x`（此前只注册了 `s`）。\n")

    # ── §3 spectrum ────────────────────────────────────────────────────
    add("## 3. 检测档位谱系（`detr/n` … `detr/x`）\n")
    add(
        "产品档位的冻结状态与骨干线方向记录在 `tasks/detection/detr/variants.py` "
        "的 `SPECTRUM` 表（机器可读）：`released` 为已发布档（权重 + 模型卡 + "
        "全量 AP + 延迟齐备），`spec-frozen` 为规格冻结档（组件与配方已承诺，"
        "预训练权重 / AP 待发布），`planned` 为方向已定、实现待落地。\n"
    )
    add("| 档位 | 变体键 | 路线 | 状态 | 基准锚点 |")
    add("|---|---|---|---|---|")
    _ROUTE = {"distill": "蒸馏", "assembly-finetune": "装配微调"}
    _STATUS = {
        "released": "**released**（已发布）",
        "spec-frozen": "**spec-frozen**（规格冻结，权重/AP 待发布）",
        "planned": "planned",
    }
    for tier in ("n", "s", "m", "l", "x"):
        entry = SPECTRUM[tier]
        key = f"`{entry.variant_key}`" if entry.variant_key else "—"
        add(
            f"| `detr/{tier}` | {key} | {_ROUTE[entry.route]} | {_STATUS[entry.status]} "
            f"| {entry.benchmark_anchor} |"
        )
    add("")
    add(
        "`released` 是唯一带预训练权重的状态，由 "
        "`autovision.registry.spectrum_release` 的发布门禁校验；当前 `n/s` 为 "
        "`spec-frozen`，需用 `scripts/train_spectrum.py` 自行训练、"
        "`scripts/eval_spectrum_coco.py` 产出全量 AP。\n"
    )
    add("```python")
    add(
        'model = autovision.detection("detr/s")   # 规格冻结档；n/s 权重/AP 待发布，'
        "m/l/x 为 planned"
    )
    add("```\n")

    # ── §4 HF manifest ─────────────────────────────────────────────────
    add("## 4. HF 权重清单（revision + sha256 钉住）\n")
    add(
        "预训练骨干、RF-DETR 检测权重与开放词表文本塔，按许可分组。CC-BY-NC / DINOv3 License 默认"
        "**不下载**：设 `AUTOVISION_DINOV2_WEIGHTS=1` / `AUTOVISION_DINOV3_WEIGHTS=1` "
        "表示已自行接受对应条款。sha256 列为该资产按文件名排序的首个权重文件"
        "（双格式资产另有 `pytorch_model.bin`、多分片资产另有其余分片，完整清单见注册表）。"
        "`revision` 为 8 位短哈希，完整值见注册表。\n"
    )
    add("| 资产（HF repo） | 许可 | 接入键 / 角色 | 官方指标 | revision | sha256 |")
    add("|---|---|---|---|---|---|")
    for license_id, label in _LICENSE_GROUPS:
        add(f"| **{label}** | | | | | |")
        for repo_id in WEIGHT_MANIFEST:
            artifact = WEIGHT_MANIFEST[repo_id]
            if artifact.license != license_id:
                continue
            key, role = _HF_ROLES[repo_id]
            card = _RF_DETR_CARDS.get(key)
            if card is None:
                metrics_cell = "—"
            elif card.metrics:
                values = card.metrics
                metrics_cell = (
                    f"AP {values['coco_map']:.1f} / {values['coco_map50']:.1f} · "
                    f"{values['params_m']:.1f}M · {int(values['input_px'])}²"
                )
            else:
                metrics_cell = "—（上游弃用）"
            file_name = _hf_main_file(artifact)
            add(
                f"| `{repo_id}` | {license_id} | `{key}`（{role}） "
                f"| {metrics_cell} "
                f"| `{artifact.revision[:8]}` | `{artifact.sha256[file_name][:12]}` |"
            )
    add("")
    add(
        "RF-DETR 行官方指标为 COCO val2017 AP / AP50、Params（M）与输入尺寸，"
        "来源为 Roboflow 官方口径；T4 / TensorRT 延迟与许可说明见"
        "[基准](benchmarks.md) §3.1。Base 上游已弃用，当前官方表不再发布指标。\n"
    )
    add("一行加载（骨干走模型键；开放词表文本塔经 `model_args` 接线）：\n")
    add("```python")
    add(
        'autovision.pretrain(out="runs/ssl1", data="images/", model="dinov2/vits14")'
        "   # 骨干；CC-BY-NC / DINOv3 需先按上表自证"
    )
    add('model = autovision.detection("rfdetr/rf-detr-base")                          # 检测变体')
    add(
        "autovision.train_object_detection(..., num_classes=2, model_args={\"open_vocab\": {"
    )
    add(
        '    "text_encoder": "siglip-base", "class_names": ("class-a", "class-b")}})'
    )
    add("```\n")

    # ── §5 downloadable ────────────────────────────────────────────────
    add("## 5. 可下载资产（warm-start / 蒸馏素材）\n")
    add(
        "经 `resolve_downloadable`（`autovision.registry.downloadable`）解析："
        "下载到本地缓存并核验 sha256（缓存命中同样复核），主源失败时回退镜像"
        "（hf-mirror / GitHub Release）。\n"
    )
    add("| 资产 | 许可 | sha256 | 用途 |")
    add("|---|---|---|---|")
    _DOWNLOAD_ROLE = {
        "d-fine/dfine_s_coco": "`detr/s` 装配暖启（Apache-2.0 上游权重）",
        "d-fine/dfine_m_coco": "m 档装配素材",
        "d-fine/dfine_l_coco": "l 档装配素材",
    }
    for name in sorted(ASSET_MANIFEST):
        model = ASSET_MANIFEST[name]
        license_id = model.license.id if model.license is not None else "—"
        role = _DOWNLOAD_ROLE.get(name, "—")
        add(f"| `{name}` | {license_id} | `{model.sha256[:12]}` | {role} |")
    add("")

    # ── §6 user-held ───────────────────────────────────────────────────
    add("## 6. 用户自持库（仓库外，不可再分发）\n")
    add(
        "7 个第三方训练的 COCO 检测器（DINOv3 骨干 + 第三方检测头）经合规重打包，"
        "逐位忠实性对拍后存于**仓库之外**（默认 `~/Research/Model/AutoVision`，"
        "`AUTOVISION_USER_HELD_ROOT` 覆盖）；不进可下载清单、不随仓库或导出产物分发。"
        "**禁再分发、禁商用**，使用即视为已自行接受相应条款。其中 3 个 ViT 资产为 "
        "STA-era（含 `backbone.sta.*`），DEIMv2 派生 spatial prior 删除后不再受支持；"
        "4 个 ConvNeXt 资产不受影响。\n"
    )
    add("| 资产 | 加载 model 键 | 构建覆盖（加载必需） | weights sha256 |")
    add("|---|---|---|---|")
    for name in USER_HELD_ASSETS:
        asset = USER_HELD_ASSETS[name]
        if asset.unsupported_reason is not None:
            override = "不可用（STA 已删除；需重训 no-STA 检查点）"
        elif asset.dec_layers is not None:
            override = f"`dec_layers={asset.dec_layers}` + `decoder_hidden_dim={asset.decoder_hidden_dim}`"
        else:
            override = "无（变体规格配方）"
        add(
            f"| `{name}` | `{asset.model_name}` | {override} "
            f"| `{asset.weights_sha256[:12]}` |"
        )
    add("")
    add("```python")
    add("from autovision.registry.user_held_assets import user_held_model_args")
    add("")
    add(
        'autovision.train_object_detection(out="runs/d1", data="data.yaml", '
        'model="detr/dinov3-convnext-base",\n'
        '    model_args=user_held_model_args("dinov3-convnext-base"))'
    )
    add("```\n")

    # ── §7 gating ──────────────────────────────────────────────────────
    add("## 7. 门控与环境变量\n")
    add("| 变量 | 作用 |")
    add("|---|---|")
    add("| `AUTOVISION_DINOV2_WEIGHTS=1` | 显式接受 DINOv2 权重条款（CC-BY-NC）后允许下载 |")
    add("| `AUTOVISION_DINOV3_WEIGHTS=1` | 显式接受 DINOv3 License 后允许下载 |")
    add("| `AUTOVISION_GTR_ROOT` | GTR 检查点目录（`scripts/download_gtr_assets.py` 写入；默认 `~/Research/Model/gtr`） |")
    add("| `AUTOVISION_USER_HELD_ROOT` | 用户自持库目录（默认 `~/Research/Model/AutoVision`） |")
    add("| `AUTOVISION_OFFLINE=1` | 禁止联网（只用本地资产） |")
    add("| `AUTOVISION_MIRROR_URL` / `HF_ENDPOINT` | 下载镜像端点 |")
    add("")
    add(
        "框架级环境变量总表见 `src/autovision/runtime/env.py`（`AUTOVISION_*`，遥测"
        "默认关闭）；上表的权重门控与资产根变量定义在各自的注册表模块"
        "（`encoders/dinov2|dinov3/`、`registry/gtr_assets.py`、"
        "`registry/user_held_assets.py`）。\n"
    )

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if docs/model_zoo.md differs from the generated page",
    )
    args = parser.parse_args()

    rendered = _render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            diff = difflib.unified_diff(
                current.splitlines(),
                rendered.splitlines(),
                fromfile="docs/model_zoo.md (on disk)",
                tofile="docs/model_zoo.md (generated)",
                lineterm="",
            )
            head = list(diff)[:40]
            print("model zoo page is stale — run scripts/gen_model_zoo.py")
            print("\n".join(head))
            return 1
        print("model zoo page is in sync")
        return 0

    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO_ROOT)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
