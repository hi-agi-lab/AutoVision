# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Arguments for EOMT semantic segmentation training (DINOv3 variant)."""

from __future__ import annotations

from pydantic import Field, field_validator

from autovision.tasks.semantic_segmentation.dinov2_eomt.config import (
    Dinov2EomtSemanticSegmentationTrainArgs,
)

__all__ = ["Dinov3EomtSemanticSegmentationTrainArgs"]


class Dinov3EomtSemanticSegmentationTrainArgs(Dinov2EomtSemanticSegmentationTrainArgs):
    """DINOv3 variant of the EOMT semantic segmentation recipe.

    Shares the DINOv2 recipe (queries, interaction blocks, layer decay,
    transforms); the backbone differs via the HF ``EomtDinov3Config``
    (RoPE position encoding, storage tokens, gated-MLP option). ``mlp_ratio``
    maps to the DINOv3 ``intermediate_size`` (= ratio x hidden_size), while
    ``use_swiglu_ffn`` has no DINOv3 counterpart and must stay False.

    ``layer_norm_eps`` is applied to the backbone and the scale-block
    LayerNorm2d; DINOv3 EoMT checkpoints trained with ``layernormbf16`` use
    1e-5, while HF's config defaults to 1e-6.
    """

    use_gated_mlp: bool = False
    layer_norm_eps: float = Field(default=1e-6, gt=0)

    @field_validator("use_swiglu_ffn")
    @classmethod
    def _no_swiglu_ffn(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "use_swiglu_ffn has no effect in the DINOv3 EOMT variant "
                "(EomtDinov3Config has no SwiGLU switch); use use_gated_mlp."
            )
        return value
