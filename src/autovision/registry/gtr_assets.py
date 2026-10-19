# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR assets: upstream MIT checkpoints for every GTR task surface (doc 31).

The Intellindust GTR checkpoints ({model, ema, last_epoch} payloads) are
registered here with pinned sha256 digests so consumption is verifiable
end to end (``verify_gtr_asset``). Checkpoints are MIT-licensed
(HuggingFace ``Phoenix8125/GTR`` front matter); the binary files are
fetched on demand from the upstream repo into the GTR store
(:data:`GTR_ASSET_ROOT`: ``AUTOVISION_GTR_ROOT`` override) by
``scripts/download_gtr_assets.py`` — this registry pins identity, not
distribution.

Task groups (roadmap doc 31 §11): the whole family shares one backbone
(``ViTAdapterSpatialSwiGLU``) + ``GTREncoder``; per-task deltas are the
heads/decoders (key-set relationship to ``gtr_s_coco`` measured
2026-10-16: obb/obj365 identical 502 keys, seg = det + 35 mask-head keys,
semseg/depth = small heads, pose = own decoder).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from autovision.errors import AutoVisionError

__all__ = [
    "GTRAsset",
    "GTR_ASSET_ROOT",
    "GTR_ASSETS",
    "GTR_HF_REPO",
    "GTR_HF_REVISION",
    "GTR_SEG_VARIANTS",
    "GTR_SEMSEG_VARIANTS",
    "gtr_assets_for_task",
    "resolve_gtr_asset",
]

#: Known user-held location for the GTR checkpoints. Resolution:
#: ``AUTOVISION_GTR_ROOT`` env override, else this path (error messages point
#: at it). The historical research-tree fallback is intentionally not shipped:
#: it is a private, machine-specific location.
_GTR_ROOT_CANDIDATES = (Path.home() / "Research/Model/gtr",)


def _resolve_asset_root() -> Path:
    override = os.environ.get("AUTOVISION_GTR_ROOT")
    if override:
        return Path(override).expanduser()
    for candidate in _GTR_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _GTR_ROOT_CANDIDATES[0]


GTR_ASSET_ROOT = _resolve_asset_root()
GTR_HF_REPO = "Phoenix8125/GTR"
#: Upstream HF commit for the GTR family. ``None`` means the downloader
#: resolves the current ``main`` commit (and prints it); pin the printed
#: commit here after the first download for a reproducible URL. Every
#: asset's sha256 is the byte-level identity and is always enforced.
GTR_HF_REVISION: str | None = None
_GTR_HF = f"https://huggingface.co/{GTR_HF_REPO}"


@dataclass(frozen=True)
class GTRAsset:
    name: str
    path: Path
    sha256: str
    task: str = "det"  # det | det-obj365 | seg | pose | depth | obb | semseg
    license: str = "MIT"
    format: str = "gtr_ckpt_v1"  # {model, ema, last_epoch}; EMA is the
    # benchmark口径 (GTR README: all reported numbers use EMA weights)
    num_classes: int = 80
    num_keypoints: int | None = None
    source: str = _GTR_HF
    #: HF revision (commit hash) for this checkpoint; ``None`` lets the
    #: downloader resolve and print the current upstream commit.
    revision: str | None = None
    #: Benchmark value from the GTR README for this checkpoint.
    metric_name: str | None = None
    metric_value: float | None = None
    #: Detection-AP alias kept for the pre-multi-task consumers.
    coco_ap: float | None = None

    @property
    def hf_filename(self) -> str:
        """Path of this checkpoint inside the upstream HF repository."""
        return self.path.relative_to(GTR_ASSET_ROOT).as_posix()

    @property
    def model_num_classes(self) -> int:
        """Class-head width the checkpoint expects.

        The det-obj365 checkpoints ship 365 dataset classes plus a
        no-object slot (366-wide score heads, 367-wide denoising embed);
        building the detection target with 365 silently drops the
        class-dependent tensors (coverage 0.8281). Other GTR tasks use
        ``num_classes`` directly.
        """
        return self.num_classes + 1 if self.task == "det-obj365" else self.num_classes


#: Semantic-segmentation widths (configs/semseg/cityscapes_finetune +
#: checkpoint shapes, 2026-10-16): variant -> (embed_dim, num_heads,
#: ffn_ratio, hidden) where ``hidden`` is the stride-8 encoder output AND
#: the head width. NOTE: L/X keep a 256-wide encoder/head while their ViT
#: is 384 — the encoder down-projects (the official configs set
#: ``GTREncoder.hidden_dim: 256`` for both), so ``hidden != embed_dim``
#: there. Getting this wrong is shape-VISIBLE (the coverage gate refuses),
#: but only per-variant: S/M coincide with ``embed_dim``.
GTR_SEMSEG_VARIANTS: dict[str, tuple[int, int, float, int]] = {
    "s": (192, 3, 4.0, 192),
    "m": (256, 4, 4.0, 256),
    "l": (384, 6, 4.0, 256),
    "x": (384, 6, 6.0, 256),
}


#: Instance-segmentation backbone widths (configs/seg/coco_seg_finetune +
#: checkpoint shapes, 2026-10-16): variant -> (embed_dim, num_heads,
#: ffn_ratio). The encoder/decoder hidden width follows the detection
#: stack (S 192/512; M 256/1024; L/X 256/1024/2048) and is carried by the
#: ``gtrvit-*`` detection specs (``backbone_embed_dim`` keeps the L/X
#: 384-wide ViT separate). All four variants are registered as task
#: variants now.
GTR_SEG_VARIANTS: dict[str, tuple[int, int, float]] = {
    "s": (192, 3, 4.0),
    "m": (256, 4, 4.0),
    "l": (384, 6, 4.0),
    "x": (384, 6, 6.0),
}


def _asset(relative: str, sha256: str, *, task: str = "det", **kwargs) -> GTRAsset:
    return GTRAsset(
        name=Path(relative).stem,
        path=GTR_ASSET_ROOT / relative,
        sha256=sha256,
        task=task,
        revision=GTR_HF_REVISION,
        **kwargs,
    )


#: (stem, sha256, README benchmark value) per task group.
_DET_ROWS = [
    ("gtr_s_coco", "69b8bed4d2bdddd8c965170195a6e40e6b848d13c9bb72f73776dfa0f5dd1979", 53.6),
    ("gtr_m_coco", "a1a90111bb75286bbda7f30ff87a99cc156ca9b75a779ec8f995156cd926a6dd", 57.3),
    ("gtr_l_coco", "d7a3e07b2159a6e4953c038aa2a1f6ccfe67a2e3d7564862a230acba7276d298", 58.9),
    ("gtr_x_coco", "547cc2c2d29b32315fa9d533431e35ed0b17807e82f0803779a91f268ff878fb", 59.4),
]
_OBJ365_ROWS = [
    ("gtr_s_obj365", "49be6e81614950cff553989edeafc707c4343086be1b58f2068ab24c4c97d677"),
    ("gtr_m_obj365", "5338fcf02f18487de349a9f9741f677f5fa1efb5e948424eeb0b624ab2f4572e"),
    ("gtr_l_obj365", "364c76be3c21b3dd4e188a01d6c28f1bdc5c5fd735a7cbd3eb02ab734b8680ed"),
    ("gtr_x_obj365", "a3bdf1b3771ef14676bab36c129adbedfee57a0b975b524ad8ee16fd4512c625"),
]
_SEG_ROWS = [
    ("gtrseg_s_coco", "14113f7b7cf367411a2977085043cecd327205aa1b77d5ed7a60f0a474281a24", 45.0),
    ("gtrseg_m_coco", "60963e2f22d149c9957e8d364fc7479718eb63ec21bd32677d424ce1e12e11ca", 47.7),
    ("gtrseg_l_coco", "33cabf1f81192e8d45bc05db8bb79e3d4b672dd644d86b8790c5e6213ce46276", 49.5),
    ("gtrseg_x_coco", "63f7b3e32e5dbe9e72cfa853b3900d736286e1c05abb4ba1c5e228419f9c4cd6", 49.8),
]
_POSE_ROWS = [
    ("gtrpose_s_coco", "ead7a3958a1103c2af5049c9cc6ad2572704c9fe7af3535f4828a00881c51fe8", 70.1),
    ("gtrpose_m_coco", "e93c957dd1301aae648a34ac9525f79f02433fa880b8f4d2b1a6d4f1e927a6ef", 74.1),
    ("gtrpose_l_coco", "aa1396b9d4585ccad17c320d01698845a2e9ab050dd796f908798727de7d82d2", 74.7),
    ("gtrpose_x_coco", "dbfb964f3bc6f9cd3a8786144761e1ec015829d9d081d1d46324f8e61671e9c3", 75.5),
]
_DEPTH_ROWS = [
    ("gtrdepth_s", "e798c6347bc2228b2e42fb951da1afcca96425cd14fb34571145480412e8bb6c", 0.074),
    ("gtrdepth_m", "39b3bcd0576831b1343e6f124a4b2c2f12b5b05574f1833f0593d561fd31266d", 0.069),
    ("gtrdepth_l", "15cee2d0fd373bb60456bc77739f7f6cad63f697e33787046924a2053bc92079", 0.069),
    ("gtrdepth_x", "fef32854352d627e3ca887bd86e58414c52f769d4cba4d4b32771376b12e294d", 0.067),
]
_OBB_ROWS = [
    ("gtrobb_s_dota", "15a57e2ecab084489c756dfda719665224770c70450ab897ddcac26edec08f59", 80.0),
    ("gtrobb_x_dota", "deb785e2d6489fdd0083addfdb10493786f8e47200308a18bd58d60b00b4e58c", 81.3),
]
_SEMSEG_ROWS = [
    ("gtrsemseg_s_cityscapes", "365ed33e3ba8450a1c16f78b4231486c68cc28fd4b7a819639fdfb1ea58f7204", 81.5),
    ("gtrsemseg_m_cityscapes", "f5206bca92cb9a85376875160e54c6ebde81cc7618c9207aaa4801b0ae400675", 83.0),
    ("gtrsemseg_l_cityscapes", "42dec87b707fb07053d6932f7f80737678e83bcfdeb1e8dff1b2bb501a291546", 83.2),
    ("gtrsemseg_x_cityscapes", "ee4ade38e2e6e398110cbe909604566b221dbec87743d8f35e09ca2ca6093b55", 83.3),
]

GTR_ASSETS: dict[str, GTRAsset] = {
    **{
        stem: _asset(f"det/{stem}.pth", sha, metric_name="coco_ap",
                     metric_value=ap, coco_ap=ap)
        for stem, sha, ap in _DET_ROWS
    },
    **{
        # The README publishes no COCO AP for the pretrain stage — these
        # weights are a fine-tune init (``-t``), not a benchmark row.
        stem: _asset(f"obj365/{stem}.pth", sha, task="det-obj365", num_classes=365)
        for stem, sha in _OBJ365_ROWS
    },
    **{
        stem: _asset(f"seg/{stem}.pth", sha, task="seg",
                     metric_name="coco_mask_ap", metric_value=ap)
        for stem, sha, ap in _SEG_ROWS
    },
    **{
        stem: _asset(f"pose/{stem}.pth", sha, task="pose", num_classes=1,
                     num_keypoints=17, metric_name="coco_kp_ap", metric_value=ap)
        for stem, sha, ap in _POSE_ROWS
    },
    **{
        stem: _asset(f"depth/{stem}.pth", sha, task="depth", num_classes=0,
                     metric_name="nyu_absrel", metric_value=value)
        for stem, sha, value in _DEPTH_ROWS
    },
    **{
        stem: _asset(f"obb/{stem}.pth", sha, task="obb", num_classes=15,
                     metric_name="dota_ap50", metric_value=ap)
        for stem, sha, ap in _OBB_ROWS
    },
    **{
        stem: _asset(f"semseg/{stem}.pth", sha, task="semseg", num_classes=19,
                     metric_name="cityscapes_miou", metric_value=miou)
        for stem, sha, miou in _SEMSEG_ROWS
    },
}


def gtr_assets_for_task(task: str) -> dict[str, GTRAsset]:
    """All registered assets of one task group (e.g. ``"semseg"``)."""
    return {name: asset for name, asset in GTR_ASSETS.items() if asset.task == task}


class GTRAssetError(AutoVisionError):
    pass


def verify_gtr_asset(asset: GTRAsset, chunk_size: int = 1 << 22) -> None:
    """Fail unless the checkpoint exists and matches its pinned sha256."""
    if not asset.path.exists():
        raise GTRAssetError(
            f"GTR asset '{asset.name}' not found at '{asset.path}'. "
            f"Source: {asset.source} (license {asset.license})."
        )
    digest = hashlib.sha256()
    with asset.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset.sha256:
        raise GTRAssetError(
            f"GTR asset '{asset.name}' sha256 mismatch — refusing to use "
            f"unverified teacher weights (pinned {asset.sha256[:12]}…, "
            f"got {digest.hexdigest()[:12]}…)."
        )


def resolve_gtr_asset(name: str, verify: bool = True) -> GTRAsset:
    """Return the registered asset, optionally verifying its sha256."""
    asset = GTR_ASSETS.get(name)
    if asset is None:
        supported = ", ".join(sorted(GTR_ASSETS))
        raise GTRAssetError(f"Unknown GTR asset '{name}'. Registered: {supported}.")
    if verify:
        verify_gtr_asset(asset)
    return asset
