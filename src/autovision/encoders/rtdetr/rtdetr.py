# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RT-DETR adapter for bare ``nn.Module``\ s that expose a ``.backbone``.

This is an *internal* adapter (no package, no model list): it wraps a raw
detection module whose ``backbone`` yields a list of multi-scale feature
maps and publishes ``out_channels``. Task-model components build on it, and
users can reach it as ``autovision.encoders.rtdetr.RTDETRModelWrapper``.
Members are grouped call-facing first (the encoder-protocol surface), then
adapter accessors, construction last.
"""

import torch.nn as _tn

#: Base class reached through an alias, like the other encoder modules.
_Base = _tn.Module

#: Module attributes this adapter inspects, each spelled in one place.
_BACKBONE_ATTR = "backbone"
_CHANNEL_ATTR = "out_channels"
#: Dict entries exchanged with the encoder protocol.
_FEATURES = "features"
_POOLED = "pooled_features"
#: Container types a multi-scale backbone may legally return.
_SCALE_CONTAINERS = list, tuple
#: Spatial squeeze used to pool feature maps down to one cell.
_avg_pool2d = _tn.functional.adaptive_avg_pool2d


class RTDETRModelWrapper(_Base):
    """Adapts a raw RT-DETR-like module (with ``.backbone``) to ``EncoderProtocol``."""

    def feature_maps(self, inp):
        scales = self.backbone(inp)
        picked = scales[-1] if isinstance(scales, _SCALE_CONTAINERS) else scales
        return {_FEATURES: picked}

    def pool_features(self, mapping):
        squeezed = _avg_pool2d(mapping[_FEATURES], 1)
        return {_POOLED: squeezed}

    def encoder_dim(self):
        widths = self._source.backbone.out_channels
        return int(widths[-1])

    def arch_traits(self):
        traits = dict(model_type="convolutional", norm_type="batchnorm")
        return traits

    @property
    def backbone(self) -> _tn.Module:
        """The raw module's backbone sub-module."""
        holder = self._source
        return holder.backbone

    def raw_module(self) -> _tn.Module:
        """The wrapped detection module, exactly as given."""
        held = self._source
        return held

    def __init__(self, model: _tn.Module):
        # Explicit base-class call; the raw module is kept verbatim so
        # ``raw_module`` can hand it back untouched.
        _Base.__init__(self)
        _require_detection_shape(model)
        self._source = model


def _require_detection_shape(module: _tn.Module):
    """Reject modules that lack a backbone or a per-scale channel listing.

    A usable RT-DETR module exposes ``.backbone`` and that backbone must
    itself publish ``out_channels`` (one entry per feature-map scale).
    """
    probe = module
    for attribute_name in (_BACKBONE_ATTR, _CHANNEL_ATTR):
        probe = getattr(probe, attribute_name, None)
        if probe is None:
            raise ValueError(
                f"RTDETRModelWrapper needs a module with a '.backbone' whose "
                f"'out_channels' lists per-scale channels; '{attribute_name}' "
                f"is missing on {type(probe).__name__}."
            )
