# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Adapter that routes timm backbones through the EncoderProtocol surface.

timm backbones are a heterogeneous zoo, and this module hides the heterogeneity
from the rest of the framework:

* **Feature hook selection.** Every timm generation exposes a different hook
  for grabbing an intermediate stage (``forward_intermediates`` on newer
  releases, ``get_intermediate_layers`` on 0.9.x, the plain forward otherwise).
  The best available hook is picked once, when the wrapper is built.
* **Output normalization.** Depending on the family, a stage arrives either as
  an NCHW tensor or as an NLC token stream that still carries class/register
  tokens in front. Prefix tokens are sliced off and token streams are folded
  back into a square spatial grid, so every backbone yields NCHW maps.
* **Pooling normalization.** timm parks the global pooling op in different
  attributes depending on the family; the wrapper locates it (or falls back to
  adaptive average pooling) and squeezes pooled output to four dimensions.
* **Architecture traits.** Family-dependent decisions — norm-layer-aware
  weight decay, for example — are derived from the architecture name recorded
  in ``pretrained_cfg["architecture"]``, via the prefix registry below.
"""

from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from math import isqrt
from typing import Literal

import torch

from autovision.encoders.model_wrapper import (
    ArchTraits,
    FeatureMapsOut,
    PoolOut,
)

__all__ = ["TIMMModelWrapper", "arch_traits_for_name"]

# Shorthand for "a timm backbone module" in the helpers below.
_Backbone = torch.nn.Module

_logger = getLogger("autovision")

# --- architecture families ---------------------------------------------------
#
# timm records only the architecture name per model, so traits have to be
# derived from that name. The registry groups lowercase architecture-name
# prefixes (verbatim timm registry spellings, so grepping a model name lands
# right here) into families that share ``ArchTraits``. Families are consulted
# in declaration order and the first family owning a matching prefix decides.

_FAMILY_REGISTRY = """
    transformer layernorm
        vit_ deit3_ deit_ swinv2_ swin_ beitv2_ beit3_ beit_ eva02_ eva_ xcit_
        flexivit_ tiny_vit_ cait_ crossvit_ pit_ tnt_ twins_ nest_ sequencer2d_
        aimv2_ convit_ gmixer_ gmlp_ hieradet_ hiera_ mixer_ mvitv2_ naflexvit_
        pvt_v2_ resmlp_ sam2_hiera_ samvit_ visformer_ vitamin_ volo_ test_

    hybrid layernorm
        maxvit_ coat_ cvt_ levit_ cmt_ mix_transformer_ caformer_ coatnet_
        coatnext_ davit_ edgenext_ efficientformerv2_ efficientformer_
        efficientvit_ fastvit_ gcvit_ maxxvitv2_ maxxvit_ mobilevitv2_ mobilevit_
        nextvit_ repvit_ shvit_ swiftformer_

    hybrid batchnorm
        bat_ botnet eca_botnext eca_halonext halo lambda_resnet lamhalobotnet
        sebotnet sehalonet

    convolutional layernorm
        convnextv2_ convnext_ convmixer_ poolformerv2_ poolformer_ convformer_
        fasternet_ focalnet_ inception_next_ mambaout_ rdnet_ starnet_

    convolutional batchnorm
        resnet resnext resnest seresnet seresnext regnetx_ regnety_ regnetv_
        regnetz_ efficientnetv2_ efficientnet_ tf_efficientnet mobilenetv5_
        mobilenetv4_ mobilenetv3_ mobilenetv2_ mobilenetv1_ mobilenet_edgetpu
        mobileone_ shufflenet_ densenet dpn nfnet_ eca_nfnet_ dm_nfnet_ nf_ vgg
        squeezenet tresnet_ hardcorenas_ cs3 csat cspdarknet cspresnet cspresnext
        darknet dla eca_resnet eca_resnext eca_vovnet ecaresnet ecaresnext
        ese_vovnet fbnetc_ fbnetv3_ gc_efficientnetv2_ gcresnet gcresnext gernet_
        ghostnet hgnetv2_ hgnet_ hrnet_ inception_resnet_ inception_v lcnet_
        legacy_ mixnet_ mnasnet_ nasnetalarge pnasnet repghostnet_ repvgg_ res2net
        res2next rexnetr_ rexnet_ sedarknet selecsls semnasnet_ senet skresnet
        skresnext spnasnet_ tf_mixnet_ tf_mobilenetv3_ tinynet_ vovnet wide_resnet
        xception
"""

_ModelType = Literal["transformer", "hybrid", "convolutional"]
_NormType = Literal["batchnorm", "layernorm"]

_MODEL_TYPE_CHOICES: tuple[str, ...] = "transformer hybrid convolutional".split()
_NORM_TYPE_CHOICES: tuple[str, ...] = "batchnorm layernorm".split()


@dataclass(frozen=True)
class _Family:
    """Architecture prefixes that share one set of ``ArchTraits``."""

    model_type: _ModelType
    norm_type: _NormType
    prefixes: tuple[str, ...]

    @property
    def traits(self) -> ArchTraits:
        return {"model_type": self.model_type, "norm_type": self.norm_type}

    def matches(self, arch_name: str) -> bool:
        """Whether ``arch_name`` starts with any of this family's prefixes."""
        return any(map(arch_name.startswith, self.prefixes))


def _parse_family_registry(blob: str) -> list[_Family]:
    """Read the whitespace registry into families, in declaration order.

    Each blank-line-separated stanza starts with a header naming the model
    type and norm type; the remaining words are that family's architecture
    name prefixes. Malformed stanzas are an authoring bug and raise.
    """
    families: list[_Family] = []
    for stanza in blob.split("\n\n"):
        words = stanza.split()
        if not words:
            continue
        if (
            len(words) < 3
            or words[0] not in _MODEL_TYPE_CHOICES
            or words[1] not in _NORM_TYPE_CHOICES
        ):
            raise ValueError(f"Malformed timm family stanza: {stanza!r}")
        families.append(_Family(words[0], words[1], tuple(words[2:])))
    return families


class _FamilyRegistry:
    """Prefix-indexed lookup over the parsed families.

    Exact-name hits bypass the prefix scan entirely; that shortcut is only
    order-safe because no registered prefix is a prefix of another, which
    :meth:`shadowing` verifies.
    """

    def __init__(self, families: list[_Family]) -> None:
        self._families = families
        self._exact: dict[str, ArchTraits] = {
            prefix: family.traits
            for family in families
            for prefix in family.prefixes
        }

    def traits_for(self, arch_name: str) -> ArchTraits | None:
        """Traits of the first family owning a matching prefix, if any."""
        exact = self._exact.get(arch_name)
        if exact is not None:
            return exact
        family = next((f for f in self._families if f.matches(arch_name)), None)
        if family is None:
            return None
        return family.traits

    def shadowing(self) -> tuple[str, str] | None:
        """First (prefix, shadowed-by) pair, or ``None`` when order-safe.

        Two registered prefixes must never be prefixes of one another;
        otherwise first-match order would silently hide one of them and the
        exact-name shortcut above would disagree with the scan.
        """
        registered: list[str] = [
            prefix for family in self._families for prefix in family.prefixes
        ]
        for candidate in registered:
            for other in registered:
                if candidate != other and candidate.startswith(other):
                    return candidate, other
        return None


_FAMILIES = _FamilyRegistry(_parse_family_registry(_FAMILY_REGISTRY))

_DEFAULT_TRAITS: ArchTraits = {"model_type": "transformer", "norm_type": "layernorm"}

_SHADOWED = _FAMILIES.shadowing()
if _SHADOWED is not None:  # pragma: no cover - guarded by the shipped registry
    raise ValueError(f"timm family registry has shadowed prefixes: {_SHADOWED}")

def arch_traits_for_name(arch_name: str) -> ArchTraits:
    """Family traits for a timm architecture name; first matching prefix wins.

    Unknown names are classified as transformer/LayerNorm and logged, since
    that is the least surprising default for recent timm additions.
    """
    traits = _FAMILIES.traits_for(arch_name)
    if traits is not None:
        return traits
    _logger.warning(
        "Could not infer architecture info for timm model '%s'; classifying "
        "as transformer (LayerNorm) — set weight_decay explicitly if this is "
        "wrong.", arch_name,
    )
    return _DEFAULT_TRAITS


def pretrained_architecture(owner: _Backbone, *, fold_case=False) -> str:
    """Architecture label recorded on a model's pretrained config (or ``''``)."""
    config = getattr(owner, "pretrained_cfg", {})
    if not isinstance(config, dict):
        return ""
    name = str(config.get("architecture", ""))
    return name.lower() if fold_case else name


# --- tensor shaping -----------------------------------------------------------


def _num_prefix_tokens(owner: _Backbone) -> int:
    """How many class/register tokens the model prepends to token streams."""
    declared = getattr(owner, "num_prefix_tokens", None)
    if declared is not None:
        return declared
    return int(hasattr(owner, "cls_token"))


def _strip_prefix_tokens(owner: _Backbone, sequence: torch.Tensor) -> torch.Tensor:
    """Slice class/register tokens off the front of an NLC token stream."""
    if sequence.ndim != 3:
        return sequence
    skip = _num_prefix_tokens(owner)
    return sequence[:, skip:]


def _sequence_patch_size(owner: _Backbone) -> int | None:
    """The model's patch size when discoverable (timm ``PatchEmbed``)."""
    size = getattr(getattr(owner, "patch_embed", None), "patch_size", None)
    if size is None:
        return None
    if isinstance(size, (tuple, list)):
        unique = set(int(value) for value in size)
        return unique.pop() if len(unique) == 1 else None
    return int(size)


def _sequence_to_grid(
    sequence: torch.Tensor, owner: _Backbone, images: torch.Tensor
) -> torch.Tensor:
    """Fold an NLC token stream into NCHW.

    The grid is derived from the INPUT size and the model's patch size
    (dynamic-H/W safe, roadmap D2: the token count is symbolic there and an
    ``isqrt`` reconstruction cannot be traced). Models without a discoverable
    patch size keep the square ``isqrt`` fallback — adequate for the static
    shapes they are usable with, and a loud error otherwise.
    """
    if sequence.ndim != 3:
        return sequence
    batch_size, token_count, token_width = sequence.shape
    patch_size = _sequence_patch_size(owner)
    if patch_size is not None:
        grid_h = images.shape[-2] // patch_size
        grid_w = images.shape[-1] // patch_size
    else:
        grid_side = isqrt(token_count)
        if grid_side * grid_side != token_count:
            raise ValueError(
                f"Non-square timm token grid ({token_count} tokens) is not "
                "supported without a discoverable patch size."
            )
        grid_h = grid_w = grid_side
    rows_first = sequence.reshape(batch_size, grid_h, grid_w, token_width)
    return rows_first.movedim(-1, 1).contiguous()


# --- single-stage feature extraction -------------------------------------------
#
# Adapters are tried newest-API first; the table maps the model attribute that
# hosts each hook to the adapter able to serve one NCHW stage through it.


def _stage_via_intermediates(net: _Backbone, images: torch.Tensor) -> torch.Tensor:
    """timm >= 1.0 hook; already returns NCHW."""
    stages = net.forward_intermediates(
        images, indices=1, output_fmt="NCHW", intermediates_only=True, norm=True
    )
    return stages[0]


def _stage_via_legacy_layers(net: _Backbone, images: torch.Tensor) -> torch.Tensor:
    """timm 0.9.x hook; ``reshape=True`` keeps the output NCHW."""
    stages = net.get_intermediate_layers(images, n=1, reshape=True, norm=True)
    return stages[0]


def _stage_via_forward(net: _Backbone, images: torch.Tensor) -> torch.Tensor:
    """Last resort: the plain forward output, cleaned up manually."""
    raw_output = net.forward_features(images)
    trimmed = _strip_prefix_tokens(net, raw_output)
    return _sequence_to_grid(trimmed, net, images)


_StageAdapter = Callable[[_Backbone, torch.Tensor], torch.Tensor]
_AdapterTable = dict[str, _StageAdapter]

_STAGE_ADAPTERS: _AdapterTable = {
    "forward_intermediates": _stage_via_intermediates,
    "get_intermediate_layers": _stage_via_legacy_layers,
}


def _pick_stage_adapter(net: _Backbone) -> _StageAdapter:
    """Best available adapter for ``net`` (the forward fallback always fits)."""
    for attribute, adapter in _STAGE_ADAPTERS.items():
        if hasattr(net, attribute):
            return adapter
    return _stage_via_forward


# --- pooling -------------------------------------------------------------------


def _pool_from(holder: _Backbone | str) -> _Backbone | str:
    """The model's global pooling op parked on ``holder``, if it is callable."""
    candidate = getattr(holder, "global_pool", "")
    return candidate if callable(candidate) else ""


def _locate_pool(net: _Backbone) -> _Backbone:
    """Find the model's global pooling op; timm parks it in different spots."""
    for holder in (net, getattr(net, "head", "")):
        pooling = _pool_from(holder)
        if callable(pooling):
            return pooling
    _logger.warning(
        "Could not find a pooling layer on the timm model; defaulting to "
        "AdaptiveAvgPool2d."
    )
    return torch.nn.AdaptiveAvgPool2d(1)


# --- the wrapper ---------------------------------------------------------------

# Attributes a timm module must expose to be wrappable.
_WRAPPABLE_ATTRIBUTES = "forward_features num_features".split()


class TIMMModelWrapper(_Backbone):
    """Presents a timm backbone through the EncoderProtocol surface."""

    _backbone: _Backbone
    _stage_extractor: _StageAdapter
    _pool_op: _Backbone

    def __init__(self, backbone: _Backbone):
        for member in _WRAPPABLE_ATTRIBUTES:
            if not hasattr(backbone, member):
                raise ValueError(f"A timm model must expose '{member}'.")
        stage_extractor = _pick_stage_adapter(backbone)
        pool_op = _locate_pool(backbone)
        super().__init__()
        self._backbone = backbone
        self._stage_extractor = stage_extractor
        self._pool_op = pool_op

    def raw_module(self) -> _Backbone:
        """The timm module handed to the constructor, unwrapped."""
        return self._backbone

    def encoder_dim(self) -> int:
        width = self.raw_module().num_features
        return int(width)

    def feature_maps(self, batch: torch.Tensor) -> FeatureMapsOut:
        """One normalized NCHW stage for the input batch."""
        return FeatureMapsOut(features=self._stage_extractor(self._backbone, batch))

    def pool_features(self, batch: FeatureMapsOut) -> PoolOut:
        pooled = self._pool_op(batch["features"])
        if pooled.ndim < 4:
            padded_shape = pooled.shape + (1,) * (4 - pooled.ndim)
            pooled = pooled.reshape(padded_shape)
        return PoolOut(pooled_features=pooled)

    def arch_traits(self) -> ArchTraits:
        return arch_traits_for_name(
            pretrained_architecture(self._backbone, fold_case=True)
        )
