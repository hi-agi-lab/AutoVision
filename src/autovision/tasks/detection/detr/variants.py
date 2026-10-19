# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR variant registry: ``detr/<backbone>`` architecture specs.

Model names follow ``detr/<backbone>`` where ``<backbone>`` resolves
against the backbone packages (``dinov2``, ``dinov3``),
e.g. ``detr/dinov2-vits14`` or ``detr/dinov3-convnext-tiny``. All
DINOv2 sizes accept the package suffixes (``-noreg``, ``-notpretrained``).
``detr/dinov2-vittest-notpretrained`` is a tiny offline test variant
(mirrors the baseline ``ViTTest``).

Variants mirror the published baseline registry (lineage:
the provenance ledger):

- **DINOv2 ViT** (patch 14): NoSTA wrapper — no spatial-prior fusion, the
  semantic taps are projected straight to ``hidden_dim``.
- **DINOv3 ViT** (patch 16, incl. the tiny ``vitt16``/``vitt16plus``
  distillations): no-STA adapter (the DEIMv2-derived spatial prior was
  removed for licensing; old STA checkpoints are unsupported).
- **DINOv3 ConvNeXt**: last-three-stage taps, whole backbone at backbone LR.
  The published ``-eupe`` names are architecture aliases of these configs.

Encoder/decoder recipes (hidden dims, FFN widths, expansion/depth
multipliers, sampling points, layer counts) are carried per variant in
:class:`DETRVariantSpec` so the assembled network matches the baseline
configs instead of a single global default.

Train-argument dataclasses live in :mod:`.config`, which re-exports the
registry helpers so ``config`` stays the single import surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DETR_MODEL_REGISTRY",
    "DETRVariantSpec",
    "FAMILY_PREFIX",
    "SPECTRUM",
    "SpectrumTier",
    "check_image_size",
    "detr_default_batch_size",
    "detr_image_size_divisor",
    "list_detr_model_names",
    "list_spectrum_tiers",
    "resolve_detr_variant",
    "resolve_spectrum_tier",
]


@dataclass(frozen=True)
class DETRVariantSpec:
    """Static per-variant architecture recipe (baseline-faithful)."""

    backbone_package: str  # "dinov2" | "dinov3" | "timm" | "gtr"
    backbone_name: str  # name inside the backbone package (suffixes allowed)
    family: Literal["vit", "convnext", "timm_cnn", "gtr"]
    patch_size: int
    #: 0-based ViT block taps; unused for convnext (stage taps).
    interaction_indexes: tuple[int, ...]
    #: Shared projection target: encoder hidden, decoder hidden, and the
    #: backbone wrapper's 1x1 projection all run at this width.
    hidden_dim: int
    #: HybridEncoder transformer FFN width.
    enc_dim_feedforward: int
    enc_expansion: float = 1.0
    enc_depth_mult: float = 1.0
    #: Decoder depth (per variant; baseline uses 4 for ViT, 3/4/6 for CNN).
    dec_layers: int = 4
    #: Decoder FFN width (baseline rtdetrv2 decoder default is 1024).
    dec_dim_feedforward: int = 1024
    #: Deformable-attention sampling points per pyramid level.
    dec_num_points: tuple[int, ...] = (3, 6, 3)
    dec_num_heads: int = 8
    num_queries: int = 300
    project_features: bool = True
    #: convnext/vit dinov3 variants need transformers to ship DINOv3.
    requires_dinov3: bool = False
    #: Default decoder; "dfine_official" ships the official D-FINE decoder
    #: (weight-compatible with the published baseline D-FINE checkpoints),
    #: the fork's Dome-line variants "dfine" (RT-DETRv2-based port).
    decoder_default: Literal["rtdetrv2", "dfine", "dfine_official"] = "rtdetrv2"
    #: Which lineage's selection stage the ``dfine_official`` decoder
    #: must reproduce: "dfine" (published D-FINE checkpoints) or "gtr"
    #: (GTR's modified GTRTransformer — 3-layer query_pos_head, per-group
    #: selection heads with group-0 at eval, no enc_output projection,
    #: silu activations; 2026-10-16 gold-check). Load-correctness
    #: contract, not cosmetic: the axes are shape-invisible.
    dec_lineage: Literal["dfine", "gtr"] = "dfine"
    #: Per-variant default for ``backbone_lr_factor`` ("auto" resolution).
    backbone_lr_factor: float = 0.05
    #: GTR backbone width when it is wider than the shared encoder/decoder
    #: width. The official GTR L/X detection checkpoints are 384-wide ViTs
    #: with 256-wide encoder/decoder stacks; ``None`` keeps ``hidden_dim``.
    backbone_embed_dim: int | None = None
    #: GTR backbone FFN ratio (official X: 6.0; S/M/L: 4.0).
    backbone_ffn_ratio: float = 4.0


# the DEIMv2-derived spatial prior was removed, so the no-STA adapter ignores
# it. DINOv2 uses the no-STA path; DINOv3 ViT now uses the same path.
_VIT_SIZES: dict[str, dict[str, Any]] = {
    "vits14": dict(
        interaction_indexes=(5, 8, 11),
        hidden_dim=224,
        enc_dim_feedforward=896,
        dec_dim_feedforward=1792,
    ),
    "vitb14": dict(
        interaction_indexes=(5, 8, 11),
        hidden_dim=768,
        enc_dim_feedforward=3072,
        dec_dim_feedforward=6144,
    ),
    "vitl14": dict(
        interaction_indexes=(11, 17, 23),
        hidden_dim=1024,
        enc_dim_feedforward=4096,
        dec_dim_feedforward=8192,
    ),
    "vitg14": dict(
        interaction_indexes=(19, 29, 39),
        hidden_dim=1536,
        enc_dim_feedforward=6144,
        dec_dim_feedforward=12288,
    ),
}

# DINOv3 ViT recipes (no-STA adapter, patch 16). vitt16/vitt16plus are the
# tiny distillations: expansion/depth_mult shrink the HybridEncoder CNN.
_DINOV3_VIT_SIZES: dict[str, dict[str, Any]] = {
    "vitt16": dict(
        interaction_indexes=(3, 7, 11),
        hidden_dim=192,
        enc_dim_feedforward=512,
        enc_expansion=0.34,
        enc_depth_mult=0.67,
        dec_dim_feedforward=512,
    ),
    "vitt16plus": dict(
        interaction_indexes=(3, 7, 11),
        hidden_dim=256,
        enc_dim_feedforward=512,
        enc_expansion=0.67,
        enc_depth_mult=1.0,
        dec_dim_feedforward=512,
    ),
    "vits16": dict(
        interaction_indexes=(5, 8, 11),
        hidden_dim=224,
        enc_dim_feedforward=896,
        dec_dim_feedforward=1792,
    ),
    "vitb16": dict(
        interaction_indexes=(5, 8, 11),
        hidden_dim=768,
        enc_dim_feedforward=3072,
        dec_dim_feedforward=6144,
    ),
    "vitl16": dict(
        interaction_indexes=(11, 17, 23),
        hidden_dim=1024,
        enc_dim_feedforward=4096,
        dec_dim_feedforward=8192,
    ),
}

_CONVNEXT_SIZES: dict[str, dict[str, Any]] = {
    # Baseline CNN recipes: hidden 384 across sizes; decoder depth grows
    # tiny/small 3 -> base 4 -> large 6. Decoder FFN keeps the rtdetrv2
    # default (1024); sampling points stay (4, 4, 4).
    "convnext-tiny": dict(hidden_dim=384, dec_layers=3),
    "convnext-small": dict(hidden_dim=384, dec_layers=3),
    "convnext-base": dict(hidden_dim=384, dec_layers=4),
    "convnext-large": dict(hidden_dim=384, dec_layers=6),
}

# Spectrum width recipes for the timm CNN line (HGNetV2), spec-frozen per
# roadmap design doc 29 §3.1 (D1/D1b) — the widths are NOT guesses: they
# were read off the cached official D-FINE-S COCO checkpoint
# (hidden 256, encoder FFN 1024, decoder 3 layers, 8 heads), so the
# ``dfine_official`` decoder assembles weight-compatible with the
# Apache-2.0 official assets. Sampling points: **[3, 6, 3]** — the value
# both official lineages ship (D-FINE ``configs/dfine/include/
# dfine_hgnetv2.yml``:62 and GTR ``configs/det/gtr_base.yml``:52). NOTE:
# the flattened offsets make ``sum(num_points)`` (== 12 for both
# [3,6,3] and [4,4,4]) the only shape-visible quantity — a (4,4,4)
# spec silently mis-splits the checkpoint's per-level offsets and
# mis-scales the sampling grid, so the split, not the sum, is the
# load-correctness contract (2026-10-16 gold-check vs the official GTR
# code). backbone_lr_factor 0.1 = pretrained-CNN fine-tune convention
# (RT-DETR trains HGNetV2 from scratch at 1.0; our spectrum starts from
# timm ImageNet weights).
_HGNETV2_SIZES: dict[str, dict[str, Any]] = {
    "hgnetv2-b0": dict(backbone_name="hgnetv2_b0", dec_layers=3),
    "hgnetv2-b1": dict(backbone_name="hgnetv2_b1", dec_layers=3),
}
_HGNETV2_RECIPE: dict[str, Any] = dict(
    hidden_dim=256,
    enc_dim_feedforward=1024,
    dec_layers=3,
    dec_dim_feedforward=1024,
    dec_num_points=(3, 6, 3),
    backbone_lr_factor=0.1,
)



def _build_registry() -> dict[str, DETRVariantSpec]:
    registry: dict[str, DETRVariantSpec] = {}

    # DINOv2 ViT variants (NoSTA per baseline): base + package suffixes
    # (-noreg, -notpretrained).
    for size, info in _VIT_SIZES.items():
        for suffix in ("", "-noreg", "-notpretrained"):
            registry[f"dinov2-{size}{suffix}"] = DETRVariantSpec(
                backbone_package="dinov2",
                backbone_name=f"{size}{suffix}",
                family="vit",
                patch_size=14,
                interaction_indexes=info["interaction_indexes"],
                hidden_dim=info["hidden_dim"],
                enc_dim_feedforward=info["enc_dim_feedforward"],
                enc_expansion=info.get("enc_expansion", 1.0),
                enc_depth_mult=info.get("enc_depth_mult", 1.0),
                dec_dim_feedforward=info["dec_dim_feedforward"],
            )
    # Tiny offline test variant (baseline ViTTest analogue): built directly
    # from a local HF config, random init; exercises the no-STA ViT path.
    registry["dinov2-vittest-notpretrained"] = DETRVariantSpec(
        backbone_package="dinov2",
        backbone_name="vittest-notpretrained",
        family="vit",
        patch_size=14,
        interaction_indexes=(1, 2, 3),
        hidden_dim=64,
        enc_dim_feedforward=512,
        dec_layers=2,
        dec_dim_feedforward=512,
    )
    # DINOv3 ViT variants (no-STA adapter; usable once transformers ships
    # DINOv3). vitt16/vitt16plus have no public HF checkpoint: only their
    # -notpretrained builds resolve (see the dinov3 package).
    for size, info in _DINOV3_VIT_SIZES.items():
        for suffix in ("", "-notpretrained"):
            name = f"dinov3-{size}{suffix}"
            registry[name] = DETRVariantSpec(
                backbone_package="dinov3",
                backbone_name=f"{size}{suffix}",
                family="vit",
                patch_size=16,
                interaction_indexes=info["interaction_indexes"],
                hidden_dim=info["hidden_dim"],
                enc_dim_feedforward=info["enc_dim_feedforward"],
                enc_expansion=info.get("enc_expansion", 1.0),
                enc_depth_mult=info.get("enc_depth_mult", 1.0),
                dec_dim_feedforward=info["dec_dim_feedforward"],
                requires_dinov3=True,
            )
    # DINOv3 ConvNeXt variants (usable once transformers ships DINOv3).
    for size, info in _CONVNEXT_SIZES.items():
        registry[f"dinov3-{size}"] = DETRVariantSpec(
            backbone_package="dinov3",
            backbone_name=size,
            family="convnext",
            patch_size=16,
            interaction_indexes=(1, 2, 3),  # unused for convnext (stage taps)
            hidden_dim=info["hidden_dim"],
            enc_dim_feedforward=2048,
            dec_layers=info["dec_layers"],
            dec_num_points=(4, 4, 4),
            requires_dinov3=True,
        )
    # Spectrum CNN line: timm HGNetV2 (Apache-2.0 upstream) — the P1
    # backbone for the N/S tiers (roadmap doc 29 §3.1). Widths are
    # dfine_s-compatible (see _HGNETV2_RECIPE) so the official decoder
    # assembles weight-compatible from the Apache-2.0 D-FINE assets.
    for size, info in _HGNETV2_SIZES.items():
        registry[size] = DETRVariantSpec(
            backbone_package="timm",
            backbone_name=info["backbone_name"],
            family="timm_cnn",
            patch_size=32,
            interaction_indexes=(),  # unused for CNN stage taps
            hidden_dim=_HGNETV2_RECIPE["hidden_dim"],
            enc_dim_feedforward=_HGNETV2_RECIPE["enc_dim_feedforward"],
            dec_layers=info["dec_layers"],
            dec_dim_feedforward=_HGNETV2_RECIPE["dec_dim_feedforward"],
            dec_num_points=_HGNETV2_RECIPE["dec_num_points"],
            decoder_default="dfine_official",
            backbone_lr_factor=_HGNETV2_RECIPE["backbone_lr_factor"],
        )
    # GTR GLA-backbone line (roadmap design doc 31 G2, MIT re-port under
    # tasks/detection/shared/gtr/): the REAL GTR backbone (Gated Linear
    # Attention + Spatial SwiGLU + bidirectional scan, DEIMv2-vit-tiny
    # lineage) + GTREncoder pyramid. Requires the "gtr" extra
    # (flash-linear-attention) at construction; Triton kernels need CUDA.
    # Widths from the official coco_finetune configs (S: 192/3 heads).
    # NOT a spec-frozen spectrum tier — G2 acceptance pending: COCO trend open,
    # and ONNX/TRT export is blocked on the GLA kernel lowering (see
    # GTRViTBackbone.export_blocker). This is the only GTR detection route:
    # the former timm HGNetV2 ``gtr-*`` decoder-only targets were removed.
    registry["gtrvit-s"] = DETRVariantSpec(
        backbone_package="gtr",
        backbone_name="gtr_tiny_spatial_swiglu",
        family="gtr",
        patch_size=16,
        interaction_indexes=(3, 7, 11),
        hidden_dim=192,
        backbone_embed_dim=192,
        backbone_ffn_ratio=4.0,
        enc_dim_feedforward=1024,  # unused: GTREncoder has no AIFI FFN
        dec_layers=4,
        dec_dim_feedforward=512,
        dec_num_points=(3, 6, 3),
        decoder_default="dfine_official",
        dec_lineage="gtr",
        backbone_lr_factor=0.1,
    )
    # GTR ViT M/L/X. Widths read off the official det/seg checkpoints:
    # backbone 256/384/384 with FFN 4/4/6, shared encoder+decoder hidden
    # 256, decoder FFN 1024/1024/2048, 4 layers. L/X keep the 256-wide
    # encoder/decoder while the ViT is 384, hence ``backbone_embed_dim``.
    for name, embed, ffn, dec_ffn in (
        ("m", 256, 4.0, 1024),
        ("l", 384, 4.0, 1024),
        ("x", 384, 6.0, 2048),
    ):
        registry[f"gtrvit-{name}"] = DETRVariantSpec(
            backbone_package="gtr",
            backbone_name=f"gtr_{name}_spatial_swiglu",
            family="gtr",
            patch_size=16,
            interaction_indexes=(3, 7, 11),
            hidden_dim=256,
            backbone_embed_dim=embed,
            backbone_ffn_ratio=ffn,
            enc_dim_feedforward=1024,  # unused: GTREncoder has no AIFI FFN
            dec_layers=4,
            dec_dim_feedforward=dec_ffn,
            dec_num_points=(3, 6, 3),
            decoder_default="dfine_official",
            dec_lineage="gtr",
            backbone_lr_factor=0.1,
        )
    # Published "-eupe" names are architecture aliases (baseline registers
    # them against the same configs).
    aliases = {
        "dinov3-convnext-tiny-eupe": "dinov3-convnext-tiny",
        "dinov3-convnext-small-eupe": "dinov3-convnext-small",
        "dinov3-convnext-base-eupe": "dinov3-convnext-base",
        "dinov3-vitt16-eupe": "dinov3-vitt16",
        "dinov3-vits16-eupe": "dinov3-vits16",
        "dinov3-vitb16-eupe": "dinov3-vitb16",
    }
    for alias, target in aliases.items():
        registry[alias] = registry[target]
    return registry


DETR_MODEL_REGISTRY: dict[str, DETRVariantSpec] = _build_registry()


@dataclass(frozen=True)
class SpectrumTier:
    """One product tier of the DETR spectrum (roadmap doc 29 §3.1).

    ``status`` separates a specification promise from a shipped model:

    - ``spec-frozen``: components and recipe are product commitments
      (changes require a minor version + new weight entry), but weights,
      full-COCO AP and latency are not published yet;
    - ``released``: downloadable weights + model card + full-COCO AP +
      reproducible latency are all present (the release gate in
      ``registry/spectrum_release.py`` enforces this);
    - ``planned``: declared target still awaiting implementation.
    """

    tier: str
    variant_key: str  # key into DETR_MODEL_REGISTRY ("" when planned)
    route: Literal["distill", "assembly-finetune"]
    benchmark_anchor: str
    status: Literal["planned", "spec-frozen", "released"] = "planned"
    note: str = ""


#: Product spectrum, P1 state (design doc 29 §3.1, spec-frozen 2026-10-15).
SPECTRUM: dict[str, SpectrumTier] = {
    "n": SpectrumTier(
        tier="n",
        variant_key="hgnetv2-b0",
        route="distill",
        benchmark_anchor="YOLOv8n / YOLO11n",
        status="spec-frozen",
        note=(
            "D1b: dfine_n official weights do not exist — N is the "
            "distillation-first tier (12ep). Teacher candidates: RF-DETR-n "
            "(Apache, HF), dfine_s decoder knowledge, or GTR-S COCO (MIT, "
            "user-held store — see registry/gtr_assets.py — Intellindust "
            "Gated-Token-Recurrence, decoder shape-compatible with "
            "dfine_official at hidden 192). Phase 0 (2026-10-17): "
            "spec-frozen only — no pretrained weights or COCO AP yet."
        ),
    ),
    "s": SpectrumTier(
        tier="s",
        variant_key="hgnetv2-b1",
        route="assembly-finetune",
        benchmark_anchor="YOLOv8s / YOLO11s",
        status="spec-frozen",
        note=(
            "Widths are dfine_s-assembly-compatible; warm-start from the "
            "Apache-2.0 d-fine/dfine_s_coco asset. Phase 0 (2026-10-17): "
            "spec-frozen only — no pretrained weights or COCO AP yet."
        ),
    ),
    "m": SpectrumTier(
        tier="m",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="YOLOv8m / RT-DETR-R18",
        note=(
            "Backbone-line decision (2026-10-16): M/L/X run the timm "
            "ConvNeXt family (tiny→m / small→l / base→x, family "
            "\"timm_cnn\") — chosen for ONNX/TRT exportability (plain "
            "CNN, no custom kernels) and license cleanliness. The GTR "
            "GLA line does NOT carry M/L/X (its export is blocked on the "
            "kernel lowering, doc 31 G2). Implementation pending; the "
            "current convnext-* entries are DINOv3-ConvNeXt (user-held "
            "weights, not the product line)."
        ),
    ),
    "l": SpectrumTier(
        tier="l",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="RT-DETR-R34 / YOLOv8l",
        note="Same backbone line as M (see the M note); implementation pending.",
    ),
    "x": SpectrumTier(
        tier="x",
        variant_key="",
        route="assembly-finetune",
        benchmark_anchor="RT-DETR-R50 / YOLOv8x",
        note="Optional tier on the same timm ConvNeXt line (see the M note).",
    ),
}


def list_spectrum_tiers() -> list[str]:
    return sorted(SPECTRUM)


#: D10 public model strings: spec-frozen and released tiers become
#: first-class registry keys so ``detr/{n,s}`` flow through the standard
#: resolution/training/export plumbing unchanged. Planned tiers
#: deliberately get NO registry entry.
for _tier_key, _tier_entry in SPECTRUM.items():
    if (
        _tier_entry.status in {"spec-frozen", "released"}
        and _tier_entry.variant_key
    ):
        DETR_MODEL_REGISTRY[_tier_key] = DETR_MODEL_REGISTRY[_tier_entry.variant_key]


def resolve_spectrum_tier(tier: str) -> SpectrumTier:
    """Resolve a spectrum tier name (``"n"`` or ``"detr/n"``)."""
    key = tier[len(FAMILY_PREFIX) :] if tier.startswith(FAMILY_PREFIX) else tier
    entry = SPECTRUM.get(key)
    if entry is None:
        supported = ", ".join(sorted(SPECTRUM))
        raise ValueError(f"Unknown spectrum tier: '{tier}'. Supported: {supported}.")
    return entry


FAMILY_PREFIX = "detr/"


def resolve_detr_variant(name: str) -> tuple[str, DETRVariantSpec]:
    """Split ``detr/<backbone>`` into (backbone key, spec).

    Idempotent: accepts ``detr/<backbone>`` (public model name) and
    ``<backbone>`` (init-args form, e.g. from exported checkpoints). The
    pre-rename family-key aliases completed their deprecation window and are
    no longer accepted here; artifacts that embed one are rewritten by the
    export loader's model-name migration before reaching this resolver.
    """
    key = name[len(FAMILY_PREFIX) :] if name.startswith(FAMILY_PREFIX) else name
    if not key or key.startswith(FAMILY_PREFIX) or "/" in key:
        raise ValueError(
            f"Unknown DETR model: '{name}' (expected '{FAMILY_PREFIX}<backbone>')."
        )
    spec = DETR_MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(sorted(DETR_MODEL_REGISTRY))
        raise ValueError(f"Unknown DETR backbone: '{key}'. Supported: {supported}.")
    return key, spec


def list_detr_model_names() -> list[str]:
    return sorted(f"{FAMILY_PREFIX}{key}" for key in DETR_MODEL_REGISTRY)


def detr_default_batch_size(backbone: str) -> int:
    """Baseline reference batch: 32 across the family, 16 for dinov2.

    Early resolution for ``batch_size="auto"`` — the train command needs
    the value before the step-based ``resolve_auto_fields`` runs.
    """
    _, spec = resolve_detr_variant(f"detr/{backbone}")
    return 16 if spec.backbone_package == "dinov2" else 32


def detr_image_size_divisor(spec: DETRVariantSpec) -> int:
    """Image sizes must divide by ``2 × patch_size`` (baseline rule)."""
    return 2 * spec.patch_size


def check_image_size(image_size: tuple[int, int], spec: DETRVariantSpec) -> None:
    """Raise unless ``image_size`` is square and divides by ``2 × patch``.

    The detection pipeline is square-only end to end (the scale-jitter
    ladder and the batch resize are square), and the ViT patch grid plus
    the encoder pyramid assume the 2 × patch divisibility.
    """
    divisor = detr_image_size_divisor(spec)
    height, width = image_size
    if height != width:
        raise ValueError(
            f"DETR image_size {(height, width)} must be square (the "
            "detection pipeline resizes to square batches)."
        )
    if height % divisor or width % divisor:
        raise ValueError(
            f"DETR image_size {(height, width)} must be divisible by "
            f"{divisor} (= 2 x patch_size for the '{spec.backbone_name}' "
            f"backbone); e.g. use ({-(-height // divisor) * divisor}, "
            f"{-(-width // divisor) * divisor})."
        )
