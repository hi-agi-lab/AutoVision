# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Apache-licensed warm-start component assets (design doc 28, section 6.3).

Phase 4 asset switch: the COCO warm-start chain is rebuilt from Apache-2.0
upstream weights instead of the unlicensed legacy reference checkpoints
(ledger P3-1). Every entry carries its license and mirrors for regional
access; importing this module (via :mod:`autovision.registry`) registers.

DINOv3-backbone warm-start variants additionally require the DINOv3
weights opt-in (``AUTOVISION_DINOV3_WEIGHTS=1``, decision D1-a) at load
time — the assets below are all Apache-2.0 and un-gated.
"""

from __future__ import annotations

from autovision.registry.assets import LicenseRef
from autovision.registry.downloadable import register_downloadable

__all__ = ["register_apache_assets"]

_DFINE_LICENSE = LicenseRef(
    id="Apache-2.0",
    origin="https://huggingface.co/Peterande/D-FINE (Peterande/D-FINE @ 956d170)",
)

_HF = "https://huggingface.co/Peterande/D-FINE/resolve/main"
_HF_MIRROR = "https://hf-mirror.com/Peterande/D-FINE/resolve/main"
_GH_STORAGE = "https://github.com/Peterande/storage/releases/download/dfinev1.0"


def register_apache_assets() -> None:
    """Register the Apache warm-start component weights (idempotent)."""
    register_downloadable(
        "d-fine/dfine_s_coco",
        url=f"{_HF}/dfine_s_coco.pth",
        sha256="48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89",
        license=_DFINE_LICENSE,
        mirrors=(f"{_HF_MIRROR}/dfine_s_coco.pth", f"{_GH_STORAGE}/dfine_s_coco.pth"),
    )
    register_downloadable(
        "d-fine/dfine_m_coco",
        url=f"{_HF}/dfine_m_coco.pth",
        sha256="b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195",
        license=_DFINE_LICENSE,
        mirrors=(f"{_HF_MIRROR}/dfine_m_coco.pth", f"{_GH_STORAGE}/dfine_m_coco.pth"),
    )
    register_downloadable(
        "d-fine/dfine_l_coco",
        url=f"{_HF}/dfine_l_coco.pth",
        sha256="f1d76b20c06610e03fa833ac9524337635e5e74204f22f4f8721174f19303d78",
        license=_DFINE_LICENSE,
        mirrors=(f"{_HF_MIRROR}/dfine_l_coco.pth", f"{_GH_STORAGE}/dfine_l_coco.pth"),
    )


register_apache_assets()
