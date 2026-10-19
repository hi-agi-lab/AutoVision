# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Student-side embedding model: a backbone adapter plus a width projection.

This is the object pretraining methods actually train and the one exports
store. It is deliberately separate from the encoder protocol itself: the
protocol says what a backbone adapter must expose, while this class couples
such an adapter to an optional 1x1 projection that rewidths the adapter's
native channels to the requested embedding size.
"""

import torch.nn as _tn

_TorchModule = _tn.Module
_PassThrough = _tn.Identity
_Conv = _tn.Conv2d

#: Names of the dict entries the protocol exchanges, kept as constants so
#: the string keys live in exactly one place each.
_FEATURES = "features"
_POOLED = "pooled_features"
_TAP_HOOK = "stage_feature_maps"


def _embed_head_for(native_width, requested_width):
    """Choose pass-through or a kernel-1 rewidthing conv for the head.

    ``None`` (unset) or the native channel count keep backbone features as
    they are; any other requested width projects them with a 1x1 conv.
    """
    if requested_width is None or requested_width == native_width:
        return _PassThrough()
    return _Conv(native_width, requested_width, 1)


def _require_tap_hook(adapter):
    """Return the adapter's intermediate-layer tap callable.

    Raises when the adapter cannot serve intermediate-layer taps; those
    need a ViT backbone (dinov2/*, dinov3/*, ...).
    """
    if not hasattr(adapter, _TAP_HOOK):
        raise ValueError(
            f"Backbone adapter '{adapter.__class__.__name__}' does not "
            "implement stage_feature_maps; intermediate-layer methods "
            "need a ViT backbone (dinov2/*, dinov3/*, ...)."
        )
    return getattr(adapter, _TAP_HOOK)


def _projected_stage_maps(adapter, head, inp, layer_indices):
    """Collect intermediate-layer taps and project their feature maps.

    Each tap's ``features`` entry passes through ``head`` while every other
    entry is handed back unchanged.
    """
    tap = _require_tap_hook(adapter)
    reworked = []
    for single_tap in tap(inp, layer_indices):
        channelled = {}
        for entry_name, entry_value in single_tap.items():
            if entry_name == _FEATURES:
                channelled[entry_name] = head(entry_value)
            else:
                channelled[entry_name] = entry_value
        reworked.append(channelled)
    return reworked


class FeatureExtractor(_TorchModule):
    """Backbone adapter plus an optional linear embed head.

    Calling the model yields pooled embedding vectors by default; with
    ``pool`` disabled it yields raw spatial feature maps, in both cases
    after the embed head. Intermediate-layer taps for ViT backbones are
    available through ``stage_feature_maps``. Members are grouped
    call-facing first, construction last.
    """

    def forward(self, inp, pool=True, **extra):
        adapter = self._inner
        maps = adapter.feature_maps(inp, **extra)
        chosen = (
            maps[_FEATURES]
            if not pool
            else adapter.pool_features(maps)[_POOLED]
        )
        head = self._embed
        return head(chosen)

    def stage_feature_maps(self, inp, layer_indices):
        """Intermediate-layer taps, each projected through the embed head.

        Requires an adapter implementing ``stage_feature_maps`` (ViT
        backbones); see :func:`_projected_stage_maps` for the details.
        """
        return _projected_stage_maps(self._inner, self._embed, inp, layer_indices)

    def _embed_width(self):
        """Channel count of whatever the embed head emits."""
        head = self._embed
        if isinstance(head, _Conv):
            return head.out_channels
        return self._inner.encoder_dim()

    embed_dim = property(_embed_width)

    def __init__(self, wrapped_model, embed_dim=None):
        # Explicit base-class call: this wrapper has no cooperative
        # multiple-inheritance expectations. The adapter is kept verbatim
        # so method code can reach protocol members directly.
        _TorchModule.__init__(self)
        adapter = wrapped_model
        native = adapter.encoder_dim()
        chosen_head = _embed_head_for(native, embed_dim)
        self._inner = adapter
        self._embed = chosen_head
