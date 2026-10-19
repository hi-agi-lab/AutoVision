# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR teacher bundle: checkpoint contract loader (roadmap doc 31, G0).

Loads the Intellindust GTR detection checkpoints with **EMA-first
semantics** (the GTR README measures all published numbers with the EMA
weights — ``ck["ema"]["module"]``; ``ck["model"]`` is the raw final
step). The bundle exposes a state-dict level contract:

- key inventory (top modules, decoder hidden/classes, train epoch);
- backbone extraction with the wrapper prefix stripped (``backbone.``);
- the detection-teacher handoff used by the assembler
  (``decoder_decoder_state`` — decoder keys as-is, they are already
  name-compatible with our ``dfine_official`` decoder at GTR widths).

**Forward is intentionally not implemented**: the GLA inference backend
(flash-linear-attention 0.5.0 + the GTR vit_adapter line) belongs to
roadmap doc 31 **G2** (needs a direction decision + dependency extras).
Requesting a forward raises :class:`GTRBackendUnavailableError` with the
pointer, so a half-wired teacher can never silently train.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autovision.errors import AutoVisionError
from autovision.registry.gtr_assets import GTRAsset, verify_gtr_asset

__all__ = [
    "GTRBackendUnavailableError",
    "GTRTeacherBundle",
    "extract_gtr_backbone",
    "load_gtr_teacher",
]


class GTRBackendUnavailableError(AutoVisionError):
    pass


@dataclass
class GTRTeacherBundle:
    """A verified, inventory-described GTR checkpoint (see module doc)."""

    path: Path
    license: str
    use_ema: bool
    state: dict[str, torch.Tensor]
    last_epoch: int | None
    inventory: dict[str, Any]

    @property
    def decoder_decoder_state(self) -> dict[str, torch.Tensor]:
        """Decoder keys as-is (``decoder.decoder.*``), for the assembler."""
        return {k: v for k, v in self.state.items() if k.startswith("decoder.")}


def load_gtr_teacher(
    path: Path | str,
    *,
    use_ema: bool = True,
    asset: GTRAsset | None = None,
    verify: bool = True,
) -> GTRTeacherBundle:
    """Load and describe a GTR checkpoint (optionally sha256-verified)."""
    path = Path(path)
    if asset is not None and verify:
        verify_gtr_asset(asset)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"'{path}' is not a GTR checkpoint (expected a dict with "
            "'model'/'ema'/'last_epoch')."
        )
    ema_module = (checkpoint.get("ema") or {}).get("module")
    if use_ema and ema_module is not None:
        weights = ema_module
        source = "ema.module"
    else:
        weights = checkpoint["model"]
        source = "model"

    hidden: int | None = None
    num_class_embeds: int | None = None
    for key, tensor in weights.items():
        if key.endswith("self_attn.in_proj_weight") and hidden is None:
            hidden = tensor.shape[-1]
        if key.endswith("denoising_class_embed.weight"):
            num_class_embeds = tensor.shape[0]

    tops: dict[str, int] = {}
    for key in weights:
        tops[key.split(".")[0]] = tops.get(key.split(".")[0], 0) + 1

    return GTRTeacherBundle(
        path=path,
        license=asset.license if asset else "MIT",
        use_ema=use_ema and ema_module is not None,
        state=weights,
        last_epoch=checkpoint.get("last_epoch"),
        inventory={
            "weight_source": source,
            "num_keys": len(weights),
            "top_modules": dict(sorted(tops.items())),
            "decoder_hidden": hidden,
            "class_embed_rows": num_class_embeds,
        },
    )


def extract_gtr_backbone(bundle: GTRTeacherBundle) -> dict[str, torch.Tensor]:
    """Backbone state with the ``backbone.`` wrapper prefix stripped.

    GTR checkpoints nest the vision backbone under
    ``backbone.backbone._model.<timm-style keys>``; this returns the
    inner keys verbatim (the G2 backbone family consumes this layout).
    """
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in bundle.state.items():
        if not key.startswith("backbone."):
            continue
        inner = key[len("backbone.") :]
        if inner.startswith("backbone._model."):
            inner = inner[len("backbone._model.") :]
        stripped[inner] = tensor
    if not stripped:
        raise ValueError(
            f"No backbone keys found in '{bundle.path}' — unexpected "
            "checkpoint layout."
        )
    return stripped


def teacher_forward(bundle: GTRTeacherBundle, images: Any) -> Any:
    """Detection-teacher forward (doc 23 protocol hook).

    Deliberately unimplemented until the GLA inference backend lands
    (roadmap doc 31 G2: fla extras + re-ported vit_adapter line). The
    stub exists so the doc-23 teacher protocol has a stable target and
    so a partially wired pipeline fails loudly instead of silently
    training without a teacher.
    """
    raise GTRBackendUnavailableError(
        "GTR teacher forward requires the GLA inference backend (roadmap "
        "doc 31 G2: flash-linear-attention extras + re-ported vit_adapter "
        "line). State-dict level teacher usage (assembly / backbone "
        "extraction) works today via GTRTeacherBundle."
    )
