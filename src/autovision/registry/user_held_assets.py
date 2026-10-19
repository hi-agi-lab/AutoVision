# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""User-held DINOv3 detector checkpoints (policy D1-a, design doc 28 §6).

Seven repackaged COCO detectors — third-party trained heads on Meta
DINOv3 backbones — live in a store OUTSIDE the git tree
(``~/Research/Model/AutoVision`` by default; ``AUTOVISION_USER_HELD_ROOT``
overrides) as ``<variant>/{weights.pt, warm_start.pt, ASSET.json}`` or,
after the store was split by task, ``detection/<variant>/{...}``
(packing records: the store's ``RECORDS-2026-10-16.md`` §B). This registry
pins IDENTITY (sha256 of both containers, mirroring
:mod:`autovision.registry.gtr_assets`), never distribution: the entries are
deliberately absent from ``ASSET_MANIFEST``, carry
``redistributable=False``, and must never ship with the repository or
exported products. Four ConvNeXt entries are loadable; the three STA-era ViT
entries are audit-only and are rejected by every load path (the DEIMv2-derived
spatial prior was removed for licensing).

Both loading routes are resolvable by registered NAME from the training
arguments:

- ``model_args={"detector_checkpoint": "<name>"}`` — the full detector
  warm-start (backbone + encoder + decoder, family-remapped) via
  :func:`autovision.tasks.detection.detr.coco_checkpoint.resolve_detector_checkpoint`.
- ``model_args={"warm_start": "<name>"}`` — the AVWarmStart v1 container
  (our key layout, mandatory license field) via
  :func:`resolve_avwarm_start_source`.

Build contract: the ConvNeXt checkpoints were trained with a 6-layer,
256-wide decoder — NOT the variant spec defaults (3/4 layers, 384 width) —
so loading them without the overrides fails the strict decoder load.
:meth:`UserHeldAsset.required_model_args` returns exactly the ``model_args``
the load contract needs for each asset.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError

__all__ = [
    "USER_HELD_ASSETS",
    "USER_HELD_LOADABLE_ASSETS",
    "USER_HELD_ROOT",
    "USER_HELD_UNSUPPORTED_ASSETS",
    "UserHeldAsset",
    "UserHeldAssetError",
    "resolve_avwarm_start_source",
    "resolve_user_held_asset",
    "user_held_model_args",
    "verify_user_held_asset",
]

#: Known store locations, in preference order (one today; the tuple keeps
#: the resolution contract identical to ``gtr_assets``). Resolution:
#: ``AUTOVISION_USER_HELD_ROOT`` env override, else the first existing
#: candidate, else the first candidate as the reported default.
_USER_HELD_ROOT_CANDIDATES = (Path.home() / "Research/Model/AutoVision",)


def _resolve_asset_root() -> Path:
    override = os.environ.get("AUTOVISION_USER_HELD_ROOT")
    if override:
        return Path(override).expanduser()
    for candidate in _USER_HELD_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _USER_HELD_ROOT_CANDIDATES[0]


USER_HELD_ROOT = _resolve_asset_root()


class UserHeldAssetError(AutoVisionError):
    pass


@dataclass(frozen=True)
class UserHeldAsset:
    """One repackaged detector in the user-held store."""

    name: str
    #: ``detr/*`` registry key the asset loads into.
    model_name: str
    #: Original third-party file name (provenance, store ASSET.json).
    origin: str
    source_sha256: str
    weights_sha256: str
    warm_start_sha256: str
    #: Load-contract build overrides. ConvNeXt checkpoints carry a 6-layer
    #: / 256-wide decoder; ``None`` (ViT) = the variant spec recipe.
    dec_layers: int | None = None
    #: This checkpoint contains ``backbone.sta.*`` weights. The DEIMv2-derived
    #: spatial prior was removed for licensing, so the no-STA ViT adapter
    #: cannot load it; loading is rejected rather than silently dropping the
    #: fusion tensors.
    requires_removed_sta: bool = False
    decoder_hidden_dim: int | None = None
    image_size: tuple[int, int] = (640, 640)
    num_classes: int = 80
    license_id: str = "DINOv3-License+NOASSERTION"
    redistributable: bool = False

    @property
    def directory(self) -> Path:
        # The store may keep the detector category in ``detection/`` (the
        # task-split layout) or directly under the root (the original
        # layout). Prefer the nested directory when it exists; an
        # ``AUTOVISION_USER_HELD_ROOT`` pointing straight at ``detection/``
        # falls through to the root form.
        nested = USER_HELD_ROOT / "detection" / self.name
        if nested.is_dir():
            return nested
        return USER_HELD_ROOT / self.name

    @property
    def weights_path(self) -> Path:
        return self.directory / "weights.pt"

    @property
    def warm_start_path(self) -> Path:
        return self.directory / "warm_start.pt"

    @property
    def unsupported_reason(self) -> str | None:
        """Why this asset cannot be loaded, or ``None`` when loadable.

        Every load path (registry args, ``detector_checkpoint``,
        ``warm_start``) raises this message instead of silently dropping the
        removed spatial-prior tensors. The message names the four ConvNeXt
        alternatives so callers have an actionable fallback.
        """
        if not self.requires_removed_sta:
            return None
        alternatives = ", ".join(sorted(USER_HELD_LOADABLE_ASSETS))
        return (
            f"User-held asset '{self.name}' contains backbone.sta.* weights "
            "from the DEIMv2-derived spatial prior, which was removed for "
            "licensing. The no-STA ViT adapter cannot load it faithfully; "
            "retrain/export a no-STA checkpoint, or use a loadable ConvNeXt "
            f"asset ({alternatives})."
        )

    def require_loadable(self) -> None:
        """Raise :class:`UserHeldAssetError` when this entry is audit-only."""
        reason = self.unsupported_reason
        if reason is not None:
            raise UserHeldAssetError(reason)

    def required_model_args(self) -> dict[str, Any]:
        """The ``model_args`` the detector-checkpoint load contract needs.

        Returns the registered asset NAME as ``detector_checkpoint``
        (resolved against the store root at build time, so the dict stays
        portable) together with the build-time overrides the checkpoint's
        shapes require (``backbone_pretrained=False``: the store
        checkpoint carries the backbone; ``dec_layers`` /
        ``decoder_hidden_dim`` for ConvNeXt). STA-era ViT checkpoints are
        rejected because the spatial prior was removed for licensing.
        """
        self.require_loadable()
        args: dict[str, Any] = {
            "detector_checkpoint": self.name,
            "backbone_pretrained": False,
            "image_size": self.image_size,
        }
        if self.dec_layers is not None:
            args["dec_layers"] = self.dec_layers
        if self.decoder_hidden_dim is not None:
            args["decoder_hidden_dim"] = self.decoder_hidden_dim
        return args


# ConvNeXt entries: encoder 384 + FIXED decoder 256 at 6 layers (the
# published checkpoints' shape; see coco_checkpoint's width contract). The
# overrides are repeated per entry so the dataclass constructor stays typed.

USER_HELD_ASSETS: dict[str, UserHeldAsset] = {
    "dinov3-vitt16": UserHeldAsset(
        name="dinov3-vitt16",
        model_name="detr/dinov3-vitt16-notpretrained",
        origin="dinov3_vitt16_ltdetr_coco_251218_dfd34210.pt",
        source_sha256="dfd34210a1a3375793d149a55d9b49e6e8b783458bdd4cd76fd28fa2d61dbb37",
        weights_sha256="e7153c95379cb27e695aa1899041300dcb6b9dfc62c4cf913f158ebe057c3839",
        warm_start_sha256="5e303fba99e28aa99e574a8ef2c0cb87fa78f8c362bf80d8cb1d37e2065b410c",
        requires_removed_sta=True,
    ),
    "dinov3-vitt16plus": UserHeldAsset(
        name="dinov3-vitt16plus",
        model_name="detr/dinov3-vitt16plus-notpretrained",
        origin="dinov3_vitt16plus_ltdetr_coco_251218_af499c82.pt",
        source_sha256="af499c825436013098a77a028ff5cf08dbf31118f4d68b15eefa6fdd9635f5d2",
        weights_sha256="96603bff98fb7c49ada16f2d2eb0b86324737c8b8de36cfed2215a8b25919202",
        warm_start_sha256="8b93a15f0193de4587bc5e4029bbcc7a61b03259ad08089b09d851bad22bd08f",
        requires_removed_sta=True,
    ),
    "dinov3-vits16": UserHeldAsset(
        name="dinov3-vits16",
        model_name="detr/dinov3-vits16-notpretrained",
        origin="dinov3_vits16_ltdetr_coco_251218_4812416b.pt",
        source_sha256="4812416b861a80f305889cf1408775044c8b05f1baf9be45cd4b1d0edd5d4532",
        weights_sha256="9c38f900d1a5472b3b2b062acf5f841df298abc8f85f38eee912e82ff40d9f88",
        warm_start_sha256="589f593cf41f588444b0adc637e5cf29c25b934af21980e7d2f5a25ebc2befc5",
        requires_removed_sta=True,
    ),
    "dinov3-convnext-tiny": UserHeldAsset(
        name="dinov3-convnext-tiny",
        model_name="detr/dinov3-convnext-tiny",
        origin="dinov3_convnext_tiny_ltdetr_coco_full.pt",
        source_sha256="35bbc4fbec3bb9fa113a33f1013abaab1952edf3335f98624b5914812d63d26c",
        weights_sha256="2925ef6afacf2008098893bded94c0a6dd1db33757d03c182d9dc7f13e4f3e48",
        warm_start_sha256="7ab8e536eda01fe709d594dac5757254f902c2bc85bf9bc4f78baa068640aba8",
        dec_layers=6,
        decoder_hidden_dim=256,
    ),
    "dinov3-convnext-small": UserHeldAsset(
        name="dinov3-convnext-small",
        model_name="detr/dinov3-convnext-small",
        origin="dinov3_convnext_small_ltdetr_coco_251218_8f7109ab.pt",
        source_sha256="8f7109ab406aa92791e4e4ca6249ab9a863734795676c81b91dbd4cc4b1ef387",
        weights_sha256="72e37da6bdfd1b0c65c9d803fe7dfdd2a5624d897c75c8e0fbd406d3f87c1710",
        warm_start_sha256="d122d32f8496cc832665f7b7ab84e396d27b0b1b03c1daa3afb89f8cf3c30691",
        dec_layers=6,
        decoder_hidden_dim=256,
    ),
    "dinov3-convnext-base": UserHeldAsset(
        name="dinov3-convnext-base",
        model_name="detr/dinov3-convnext-base",
        origin="dinov3_convnext_base_ltdetr_coco_251218_836adb6b.pt",
        source_sha256="836adb6b5122665a24b6da3ee1720b9f3d0fc3c30cee44cfbd98dcb79fe0809a",
        weights_sha256="503eaea85c855b508b187f2a0f24f4f3098d6745b682e340fa8342410b856053",
        warm_start_sha256="303aba38979c813ffe54d43bb742fb58b1a35ab6b6368e678396758be30ea570",
        dec_layers=6,
        decoder_hidden_dim=256,
    ),
    "dinov3-convnext-large": UserHeldAsset(
        name="dinov3-convnext-large",
        model_name="detr/dinov3-convnext-large",
        origin="dinov3_convnext_large_ltdetr_coco_251218_03fe6750.pt",
        source_sha256="03fe6750392daf3ecd32bbab3f144bd5c4d6cdc8bd75635f9e1c5e296e7dd8b0",
        weights_sha256="7e617f3fbf691a3c80c80238ae807c941491c23acb5b59337c5b3b8804c71382",
        warm_start_sha256="dc5c58a72409f5d24fac95dfbba078392e43b38f6a8b34c2d042598efd940baa",
        dec_layers=6,
        decoder_hidden_dim=256,
    ),
}

#: Entries that can be built/loaded today.
USER_HELD_LOADABLE_ASSETS: dict[str, UserHeldAsset] = {
    name: asset
    for name, asset in USER_HELD_ASSETS.items()
    if not asset.requires_removed_sta
}

#: Audit-only entries retained for identity/provenance; every load path
#: rejects them because their spatial-prior tensors were removed.
USER_HELD_UNSUPPORTED_ASSETS: dict[str, UserHeldAsset] = {
    name: asset
    for name, asset in USER_HELD_ASSETS.items()
    if asset.requires_removed_sta
}


_VERIFIED_FILES: dict[tuple[str, int, int, str], bool] = {}


def verify_user_held_asset(
    asset: UserHeldAsset,
    files: tuple[str, ...] = ("weights.pt", "warm_start.pt"),
    chunk_size: int = 1 << 22,
) -> None:
    """Fail unless the pinned container file(s) match their sha256.

    Results are cached per (path, size, mtime, expected digest), so repeated
    resolves in one process do not re-hash a 40 MB–920 MB container; any file
    change invalidates the cache entry.

    This is an identity check only. Audit-only assets may be verified, but
    that does not make them loadable; use :func:`resolve_user_held_asset`
    with ``allow_unsupported=True`` to reach their metadata.
    """
    pinned = {
        "weights.pt": asset.weights_sha256,
        "warm_start.pt": asset.warm_start_sha256,
    }
    for fname in files:
        if fname not in pinned:
            raise UserHeldAssetError(
                f"Unknown file '{fname}' for asset '{asset.name}' "
                f"(pinned: {sorted(pinned)})."
            )
        path = asset.directory / fname
        if not path.is_file():
            raise UserHeldAssetError(
                f"User-held asset '{asset.name}' file missing: '{path}' "
                f"(store root {USER_HELD_ROOT}; source: {asset.origin}; "
                f"license {asset.license_id}, not redistributable)."
            )
        stat = path.stat()
        cache_key = (
            str(path.resolve()), stat.st_size, stat.st_mtime_ns, pinned[fname]
        )
        if _VERIFIED_FILES.get(cache_key):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        if digest.hexdigest() != pinned[fname]:
            raise UserHeldAssetError(
                f"User-held asset '{asset.name}' {fname} sha256 mismatch — "
                f"refusing to use unverified weights (pinned "
                f"{pinned[fname][:12]}…, got {digest.hexdigest()[:12]}…)."
            )
        _VERIFIED_FILES[cache_key] = True


def resolve_user_held_asset(
    name: str, *, verify: bool = False, allow_unsupported: bool = False
) -> UserHeldAsset:
    """Return the registered asset by name, optionally verifying sha256.

    Loadable entries are returned normally. Audit-only entries raise before
    ``verify`` runs, so a 40 MB–920 MB container is never hashed just to reach
    a licensing error; pass ``allow_unsupported=True`` for metadata/audit
    tooling that only needs the pinned identity.
    """
    asset = USER_HELD_ASSETS.get(str(name))
    if asset is None:
        loadable = ", ".join(sorted(USER_HELD_LOADABLE_ASSETS))
        unsupported = ", ".join(sorted(USER_HELD_UNSUPPORTED_ASSETS))
        raise UserHeldAssetError(
            f"Unknown user-held asset '{name}'. Loadable: {loadable}. "
            f"Registered but unsupported (STA removed): {unsupported}. "
            f"(store root: {USER_HELD_ROOT}; policy D1-a: user-held, "
            f"not redistributable)."
        )
    if not allow_unsupported:
        asset.require_loadable()
    if verify:
        verify_user_held_asset(asset)
    return asset


def user_held_model_args(name: str, *, verify: bool = False) -> dict[str, Any]:
    """``model_args`` that build and load the named asset (see the class).

    Audit-only STA-era assets raise before ``verify`` hashes their container.
    """
    return resolve_user_held_asset(name, verify=verify).required_model_args()


def resolve_avwarm_start_source(source: str | Path) -> Path:
    """Resolve a ``warm_start`` argument to a local file.

    An existing file passes through untouched; a registered user-held name
    maps to its ``warm_start.pt``. Any other value also passes through so
    the loader raises its own not-found error (no behavior change for
    plain paths).
    """
    path = Path(source)
    if path.is_file():
        return path
    asset = USER_HELD_ASSETS.get(str(source))
    if asset is None:
        return path
    asset.require_loadable()
    if not asset.warm_start_path.is_file():
        raise UserHeldAssetError(
            f"User-held asset '{asset.name}' is registered but its "
            f"warm_start.pt is not present at '{asset.warm_start_path}' "
            f"(store root {USER_HELD_ROOT}; AUTOVISION_USER_HELD_ROOT "
            f"overrides)."
        )
    verify_user_held_asset(asset, files=("warm_start.pt",))
    return asset.warm_start_path
