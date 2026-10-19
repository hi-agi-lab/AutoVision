# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation task model, DINOv3 variant (inference/export
facing)."""

from __future__ import annotations

from typing import Any

import torch

from autovision.augment.transform_args import NormalizeArgs
from autovision.tasks.semantic_segmentation.dinov2_eomt.task_model import (
    Dinov2EomtSemanticSegmentation,
)
from autovision.tasks.shared.eomt.modeling import ROPE_PERIODS_KEY, build_eomt_model
from autovision.tasks.task_model import TaskNetwork

__all__ = ["Dinov3EomtSemanticSegmentation"]


class _Dinov3EomtCheckpointMixin:
    """Checkpoint loading shared by the DINOv3 EoMT task models.

    The semantic, instance and panoptic variants derive from different
    DINOv2 parents, so an override on the semantic class alone would not be
    in the panoptic/instance MRO. The mixin is listed first in each MRO and
    delegates the strict load to the DINOv2 EoMT loader.
    """

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a DINOv3 EoMT checkpoint.

        The optional :data:`ROPE_PERIODS_KEY` entry carries the checkpoint's
        trained RoPE periods; it is removed before the strict HF load and
        installed on the rope module afterwards (``inv_freq`` is a
        non-persistent HF buffer, so it cannot ride the normal state dict).
        """
        state_dict = dict(state_dict)
        periods = state_dict.pop(ROPE_PERIODS_KEY, None)
        super().load_train_state_dict(state_dict)
        if periods is not None:
            self._apply_rope_periods(periods)

    def _apply_rope_periods(self, periods: Any) -> None:
        model = getattr(self, "model", None)
        rope = getattr(model, "rope_embeddings", None)
        inv_freq = getattr(rope, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor):
            raise ValueError(
                "Checkpoint carries RoPE periods but the built model exposes "
                "no rope_embeddings.inv_freq buffer."
            )
        periods_tensor = torch.as_tensor(periods, dtype=torch.float32)
        if periods_tensor.ndim != 1 or periods_tensor.numel() == 0:
            raise ValueError(
                f"RoPE periods must be a non-empty 1-D tensor, got shape "
                f"{tuple(periods_tensor.shape)}."
            )
        if not bool(torch.isfinite(periods_tensor).all()) or bool(
            torch.any(periods_tensor <= 0)
        ):
            raise ValueError("RoPE periods must be finite and positive.")
        if periods_tensor.numel() != inv_freq.numel():
            raise ValueError(
                f"RoPE periods length {periods_tensor.numel()} does not match "
                f"the model's inv_freq length {inv_freq.numel()}."
            )
        inverse = periods_tensor.reciprocal().to(
            device=inv_freq.device, dtype=inv_freq.dtype
        )
        with torch.no_grad():
            inv_freq.copy_(inverse)
            original = getattr(rope, "original_inv_freq", None)
            if isinstance(original, torch.Tensor):
                if original.numel() != inverse.numel():
                    raise ValueError(
                        "Model exposes an original_inv_freq buffer with a "
                        "different length than inv_freq."
                    )
                original.copy_(
                    inverse.to(device=original.device, dtype=original.dtype)
                )


class Dinov3EomtSemanticSegmentation(
    _Dinov3EomtCheckpointMixin, Dinov2EomtSemanticSegmentation
):
    """DINOv3 variant: same encoder-only mask transformer contract, backed
    by the HF ``EomtDinov3ForUniversalSegmentation`` (RoPE, storage tokens).
    Preprocessing, forward, decode and export behavior are inherited from
    the DINOv2 variant; construction adds ``layer_norm_eps`` and checkpoint
    loading transports the trained RoPE periods instead of relying on HF's
    recomputation from ``rope_theta``.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        num_queries: int = 100,
        num_blocks: int = 4,
        num_upscale_blocks: int = 2,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: int = 4,
        use_gated_mlp: bool = False,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        load_weights: bool = False,
        weight_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        # The DINOv2 parent __init__ pins the DINOv2 HF model class, so
        # skip it and replicate its TaskNetwork bookkeeping here.
        TaskNetwork.__init__(
            self,
            init_args=locals(),
            ignore_args={"load_weights", "kwargs", "self"},
        )
        self.num_classes = num_classes
        self.image_size = int(image_size)
        self.normalize = NormalizeArgs()
        self.model = build_eomt_model(
            num_classes,
            variant="dinov3",
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_queries=num_queries,
            num_blocks=num_blocks,
            num_upscale_blocks=num_upscale_blocks,
            image_size=self.image_size,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            use_gated_mlp=use_gated_mlp,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
            load_weights=load_weights,
            weight_source=weight_source,
        )
