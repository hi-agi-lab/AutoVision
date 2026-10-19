# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection backbone wrappers for the AV-DETR family.

Ports the baseline detector backbone adapters:

- :class:`ViTDetectionBackbone` (baseline ViT adapter): semantic features
  from intermediate ViT blocks, bilinearly resized into a stride-8/16/32
  pyramid and projected to a shared ``hidden_dim`` via 1x1 convs. Only the
  pretrained ViT counts as "backbone" for LR grouping — the projection
  layers are randomly initialized and train at detector LR. The DEIMv2-
  derived spatial prior (STA) was removed for licensing; DINOv3 ViT uses
  this no-STA adapter.
- :class:`ConvNeXtDetectionBackbone`: taps the last three ConvNeXt
  stages (strides 8/16/32); the whole backbone counts as "backbone".
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Conv2d,
    Module,
    ModuleList,
    SyncBatchNorm,
)

from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.encoders.dinov3.dinov3_wrapper import TransformersDinov3ConvNextWrapper

__all__ = [
    "CONVNEXT_DETECTION_PATCH_SIZE",
    "ConvNeXtDetectionBackbone",
    "GTRViTBackbone",
    "TimmCNNBackbone",
    "ViTDetectionBackbone",
]

#: DINOv3 ConvNeXt has no patch embedding; the detection pipeline still
#: treats the family as patch-16 for image-size arithmetic. Default only —
#: :class:`ConvNeXtDetectionBackbone` receives the authoritative value
#: from the variant spec.
CONVNEXT_DETECTION_PATCH_SIZE = 16


def _ceil_half(value: int) -> int:
    """Output size of a stride-2, kernel-3 pad-1 conv on ``value``."""
    return -(-value // 2)


class ViTDetectionBackbone(Module):
    """ViT -> stride-8/16/32 detection features (no-STA adapter).

    The DEIMv2-derived spatial prior was removed for licensing reasons;
    DINOv3 ViT variants now use the same no-STA adapter as DINOv2. Old
    checkpoints containing ``backbone.sta.*`` are unsupported and rejected
    by the loaders (see docs/provenance/LEDGER.md).
    """

    def __init__(
        self,
        model_wrapper: TransformersDinov2Wrapper,
        interaction_indexes: tuple[int, int, int] = (5, 8, 11),
        finetune: bool = True,
        hidden_dim: int | None = None,
        project_features: bool = True,
        resize_features: bool = True,
    ) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        embed_dim = model_wrapper.encoder_dim()

        assert len(interaction_indexes) == 3
        self.interaction_indexes = tuple(interaction_indexes)
        self.patch_size = model_wrapper.patch_size()
        self.resize_features = resize_features

        if not finetune:
            model_wrapper.eval()
            model_wrapper.requires_grad_(False)

        self.project_features = project_features
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        if not self.project_features and hidden_dim != embed_dim:
            raise ValueError(
                "project_features=False requires hidden_dim to match the "
                f"backbone embed_dim ({embed_dim}), but got "
                f"hidden_dim={hidden_dim}."
            )
        if not self.project_features:
            return

        self.convs = ModuleList(
            [
                Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
                Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
                Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            ]
        )
        self.norms = ModuleList(
            [
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
            ]
        )

    @property
    def backbone_model(self) -> Module:
        """The pretrained ViT (only this part trains at backbone LR)."""
        return self._model_wrapper.raw_module()

    @property
    def in_channels(self) -> list[int]:
        hidden_dim = (
            self.convs[0].out_channels
            if self.project_features
            else self._model_wrapper.encoder_dim()
        )
        return [hidden_dim] * 3

    def forward(
        self, x: Tensor, window_keep_masks: list[Tensor] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        h_c, w_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        if window_keep_masks is not None:
            # In-backbone background token dropping (design doc 17): the
            # wrapper runs one encoder pass per image over its kept patch
            # tokens and zero-fills the dropped positions, so the taps keep
            # the dense (B, D, H', W') shapes and everything below — pyramid
            # resize and projection — is untouched.
            sem_feats = self._model_wrapper.stage_feature_maps_masked(
                x, self.interaction_indexes, window_keep_masks
            )
        else:
            sem_feats = self._model_wrapper.stage_feature_maps(
                x, self.interaction_indexes
            )

        # Target pyramid: shapes that chain through the encoder's stride-2
        # convolutions. The semantic taps (patch-grid based, e.g. grid 46 at
        # 644) are resized here rather than driving the shapes themselves.
        # Level 0 is input//8; each next level ceil-halves the PREVIOUS
        # level, exactly what a stride-2 3x3-pad-1 conv produces. Sizing
        # the lower levels independently from the input (e.g. ceil(x/16))
        # disagrees with the chain by one pixel at sizes like 644
        # (80 -> 40 vs ceil(644/16) = 41) and breaks the FPN/PAN fusion.
        if self.resize_features:
            h8, w8 = x.shape[2] // 8, x.shape[3] // 8
            sizes = [
                (h8, w8),
                (_ceil_half(h8), _ceil_half(w8)),
                (_ceil_half(_ceil_half(h8)), _ceil_half(_ceil_half(w8))),
            ]
        else:
            # All taps already share the patch-grid resolution.
            sizes = [(h_c, w_c)] * 3

        resized_feats = [
            F.interpolate(
                sem_feat["features"],
                size=list(size),
                mode="bilinear",
                align_corners=False,
            )
            for sem_feat, size in zip(sem_feats, sizes)
        ]

        fused_feats = resized_feats

        if not self.project_features:
            return fused_feats[0], fused_feats[1], fused_feats[2]

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        return c2, c3, c4


class ConvNeXtDetectionBackbone(Module):
    """DINOv3 ConvNeXt -> detection pyramid from consecutive stages.

    ``num_stages=3`` (default) taps stages 1/2/3 (strides 8/16/32);
    ``num_stages=4`` taps all four stages (strides 4/8/16/32) — the
    Dome-faithful pyramid for tiny-object workloads (design doc 13 §5/D1).

    Tap semantics (baseline ConvNeXt detection wrapper): every tap runs
    through the baseline's ``norms[i]`` — ``[Identity, Identity, Identity,
    self.norm]`` — so ONLY the final stage's feature map is LayerNormed;
    the earlier stages stay raw. Applied here to the last tap for both
    pyramid depths (the last tap IS the final stage in either case); the
    un-normed alternative measured 0.414 -> 0.293 COCO mAP on the published
    convnext-tiny detector.
    """

    #: 0-based stage indices for the default 3-level pyramid.
    STAGE_INDICES = (1, 2, 3)

    def __init__(
        self,
        model_wrapper: TransformersDinov3ConvNextWrapper,
        finetune: bool = True,
        patch_size: int = CONVNEXT_DETECTION_PATCH_SIZE,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = patch_size
        hidden_sizes = list(model_wrapper.config.hidden_sizes)
        if num_stages not in (3, 4):
            raise ValueError(
                f"ConvNeXtDetectionBackbone num_stages must be 3 or 4, "
                f"got {num_stages}."
            )
        if num_stages > len(hidden_sizes):
            raise ValueError(
                f"Backbone has {len(hidden_sizes)} stages; {num_stages} requested."
            )
        self.num_stages = num_stages
        self.stage_indices = tuple(
            range(len(hidden_sizes) - num_stages, len(hidden_sizes))
        )
        if not finetune:
            model_wrapper.eval()
            model_wrapper.requires_grad_(False)

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper.raw_module()

    @property
    def in_channels(self) -> list[int]:
        hidden_sizes = list(self._model_wrapper.config.hidden_sizes)
        return [hidden_sizes[i] for i in self.stage_indices]

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outputs = self._model_wrapper.raw_module()(
            pixel_values=x, output_hidden_states=True
        )
        hidden_states = outputs.hidden_states  # (embeddings, stage_0, ...)
        feats = tuple(hidden_states[i + 1] for i in self.stage_indices)
        if len(feats) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} ConvNeXt stage features, got {len(feats)}."
            )
        # Baseline norms[-1] == self.norm on the LAST stage only (the
        # earlier taps are Identity), in both 3- and 4-level pyramids.
        return feats[:-1] + (self._model_wrapper.final_stage_norm(feats[-1]),)


class TimmCNNBackbone(Module):
    """timm CNN (HGNetV2 line) -> detection pyramid from stage taps.

    Uses timm's ``features_only`` mode: ``out_indices=(1, 2, 3)`` yields
    the stride-8/16/32 pyramid the HybridEncoder consumes, mirroring the
    RT-DETR/D-FINE backbone treatment. Pretrained weights come from timm
    (ImageNet, Apache-2.0-friendly provenance); ``AUTOVISION_OFFLINE`` /
    ``AUTOVISION_MIRROR_URL`` are bridged to the HF hub env vars before
    the timm import so cache-only / mirror resolution applies (roadmap
    design doc 29, P1 mirror/offline 收口).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        finetune: bool = True,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        if num_stages != 3:
            raise ValueError(
                f"TimmCNNBackbone num_stages must be 3, got {num_stages}."
            )
        from autovision.encoders.weight_verification import bridge_hub_env

        bridge_hub_env()
        import timm

        self._model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self._channels = list(self._model.feature_info.channels())
        if not finetune:
            self._model.eval()
            self._model.requires_grad_(False)

    @property
    def backbone_model(self) -> Module:
        return self._model

    @property
    def in_channels(self) -> list[int]:
        return list(self._channels)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self._model(x)
        if len(feats) != 3:
            raise ValueError(
                f"Expected 3 pyramid features from '{type(self._model).__name__}', "
                f"got {len(feats)}."
            )
        return tuple(feats)


class GTRViTBackbone(Module):
    """GTR ViTAdapter (GLA + Spatial SwiGLU + bidirectional scan) wrapper.

    Same-resolution taps (stride = patch_size, uniform ``embed_dim``
    channels); the 8/16/32 pyramid comes from GTREncoder downstream
    (roadmap design doc 31 G2). Requires the ``gtr`` extra
    (flash-linear-attention) at construction — Triton GLA kernels are
    CUDA-only, so forward needs a CUDA tensor.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        interaction_indexes: tuple[int, ...] = (3, 7, 11),
        patch_size: int = 16,
        finetune: bool = True,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        from autovision.errors import AutoVisionError
        from autovision.tasks.detection.shared.gtr import (
            GTR_ONNX_EXPORT_BLOCKER,
            ViTAdapterSpatialSwiGLU,
            gtr_available,
        )

        if not gtr_available():
            raise AutoVisionError(
                "The GTR backbone line requires flash-linear-attention "
                "(roadmap design doc 31 G2): pip install -e '.[gtr]'"
            )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Declared for the export mixin's module-tree scan: the GLA
        # kernels have no ONNX lowering (see the constant's docstring).
        self.export_blocker = GTR_ONNX_EXPORT_BLOCKER
        self._adapter = ViTAdapterSpatialSwiGLU(
            embed_dim=embed_dim,
            num_heads=num_heads,
            interaction_indexes=list(interaction_indexes),
            multi_layer_same_res=True,
            skip_weights_warning=True,
            patch_size=patch_size,
            ffn_ratio=ffn_ratio,
        )
        if not finetune:
            self._adapter.eval()
            self._adapter.requires_grad_(False)

    @property
    def backbone_model(self) -> Module:
        # The inner VisionTransformer._model: its key layout matches
        # gtr_teacher.extract_gtr_backbone output verbatim, so
        # ``backbone_weights`` (task_model._load_backbone_weights) loads
        # extracted GTR backbone states without remapping.
        return self._adapter.backbone._model

    @property
    def in_channels(self) -> list[int]:
        return [self.embed_dim] * 3

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self._adapter(x)
        if len(feats) != 3:
            raise ValueError(
                f"Expected 3 GTR tap features, got {len(feats)}."
            )
        return tuple(feats)
