# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR wrapper on top of ``transformers``.

Per framework policy (see the DINOv2/DINOv3 packages), RF-DETR is provided
through HuggingFace ``transformers`` (``RfDetrModel``) instead of the
Roboflow ``rfdetr`` package. The wrapper exposes only the distilled
DINOv2 encoder: in ``RfDetrModel`` the
ViT lives one level down (``model.backbone`` is the conv encoder
``RfDetrConvEncoder``, whose ``.backbone`` is the ``RfDetrDinov2Backbone``
returning one 4D feature map per ``out_features`` stage). The rest of the
detector (projector, decoder, heads) stays in the module tree so state
dicts and exports remain full-model, but receives no gradients from
feature extraction (the engine's ``find_unused_parameters=True`` covers
DDP).

Pretraining features come from the backbone only: the last feature map is
the spatial feature (the encoder's final stage output). The wrapper does
not support
masked-token forwards (the HF backbone drops ``bool_masked_pos``), so
DINOv2-style masked methods fail fast with an informative error.

transformers gained RF-DETR support in 5.9.0 (2026-05), which the package
gates on at runtime (see ``rfdetr_source``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from autovision.encoders.model_wrapper import (
    ArchTraits,
    FeatureMapsOut,
    PatchSize,
    PoolOut,
)

__all__ = ["RFDETRModelWrapper"]


def _dinov2_backbone(model: nn.Module) -> nn.Module:
    """Return the ``RfDetrDinov2Backbone`` nested inside an RF-DETR model.

    ``RfDetrModel.backbone`` is the detection conv encoder; the ViT is its
    ``.backbone`` child. Reaching past the encoder matters: the encoder's
    ``forward(pixel_values, pixel_mask)`` returns a plain
    ``(features, mask)`` tuple, while the ViT returns
    ``BackboneOutput.feature_maps`` (one 4D map per stage).
    """
    conv_encoder = getattr(model, "backbone", None)
    vit_backbone = getattr(conv_encoder, "backbone", None)
    if vit_backbone is None:
        raise ValueError(
            f"RFDETRModelWrapper requires a transformers RF-DETR model with "
            f"a '.backbone.backbone' (RfDetrConvEncoder -> RfDetrDinov2Backbone) "
            f"attribute chain, got '{type(model).__name__}'."
        )
    return vit_backbone


class RFDETRModelWrapper(nn.Module, PatchSize):
    """Adapts a ``transformers`` RF-DETR model to the ``EncoderProtocol`` protocol."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        _dinov2_backbone(model)  # fail fast on unsupported model shapes.
        self._model = model  # transformers models are nn.Modules: register.

    @property
    def hf_model(self) -> nn.Module:
        return self._model

    def raw_module(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    @property
    def backbone(self) -> nn.Module:
        """The nested ``RfDetrDinov2Backbone`` (not the conv encoder)."""
        return _dinov2_backbone(self._model)

    # ── EncoderProtocol protocol ───────────────────────────────────────────
    def feature_maps(self, x: Tensor, **kwargs: Any) -> FeatureMapsOut:
        if kwargs.get("bool_masked_pos") is not None:
            raise ValueError(
                "RF-DETR models do not support masked-token forwards: the "
                "transformers backbone ignores 'bool_masked_pos'. Masked "
                "image modeling (DINOv2/DINOv3 methods) requires a "
                "dinov2/* or dinov3/* backbone."
            )
        outputs = self.backbone(pixel_values=x)
        features = outputs.feature_maps[-1]
        if features.dim() != 4:
            raise ValueError(
                "Expected 4D (B, C, H, W) backbone feature maps (build the "
                f"backbone with reshape_hidden_states=True); got shape "
                f"{tuple(features.shape)}."
            )
        return {"features": features}

    def pool_features(self, x: FeatureMapsOut) -> PoolOut:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            x["features"], output_size=1
        )
        return {"pooled_features": pooled}

    def encoder_dim(self) -> int:
        backbone_config = getattr(self.config, "backbone_config", None)
        hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Cannot determine the RF-DETR feature dim: the model config "
                "has no backbone_config.hidden_size."
            )
        return int(hidden_size)

    def patch_size(self) -> int:
        return int(self.config.backbone_config.patch_size)

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "transformer", "norm_type": "layernorm"}

    def make_teacher(self) -> "RFDETRModelWrapper":
        """Prepare for teacher duty: eval mode and no stochastic dropout.

        Also zeroes DropPath-style stochastic depth: the HF DINOv2 drop-path
        subclasses ``nn.Module`` (attribute ``drop_prob``), not ``nn.Dropout``,
        so ``eval()`` alone would only keep it dormant and a later ``train()``
        call would silently re-enable it.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
        return self
