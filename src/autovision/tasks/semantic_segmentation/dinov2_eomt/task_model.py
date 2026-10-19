# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""EOMT semantic segmentation task model (inference/export facing)."""

from __future__ import annotations

import pathlib
from typing import Any

import torch

from autovision.augment.transform_args import NormalizeArgs
from autovision.export import ONNXExportMixin
from autovision.tasks.shared.eomt.modeling import build_eomt_model
from autovision.tasks.shared.eomt.postprocess import semantic_segmentation_decode
from autovision.tasks.task_model import TaskNetwork

__all__ = ["Dinov2EomtSemanticSegmentation"]


class Dinov2EomtSemanticSegmentation(ONNXExportMixin, TaskNetwork):
    """Encoder-only mask transformer (HF ``EomtForUniversalSegmentation``).

    ``forward_backend`` returns the last-layer query outputs
    ``(masks_queries_logits, class_queries_logits)``; the semantic decode
    (class probability x mask probability -> argmax) happens in
    ``postprocess``, keeping the export graph minimal.
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
        use_swiglu_ffn: bool = False,
        drop_path_rate: float = 0.0,
        load_weights: bool = False,
        weight_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights", "kwargs", "self"})
        self.num_classes = num_classes
        self.image_size = int(image_size)
        self.normalize = NormalizeArgs()
        self.model = build_eomt_model(
            num_classes,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_queries=num_queries,
            num_blocks=num_blocks,
            num_upscale_blocks=num_upscale_blocks,
            image_size=self.image_size,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            use_swiglu_ffn=use_swiglu_ffn,
            drop_path_rate=drop_path_rate,
            load_weights=load_weights,
            weight_source=weight_source,
        )

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as TF

        from autovision.data.file_helpers import open_image

        if isinstance(image, (str, pathlib.Path)):
            tensor = open_image(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(image).float() / 255.0
        _, height, width = tensor.shape
        resized = TF.resize(
            tensor,
            [self.image_size, self.image_size],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def forward_train(self, x: torch.Tensor) -> Any:
        """Full HF forward (used by training; labels are applied there)."""
        return self.model(pixel_values=x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Export/inference entry point (ONNX traces this single-input call)."""
        return self.forward_backend(x)

    def forward_backend(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=x)
        return outputs.masks_queries_logits, outputs.class_queries_logits

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        if isinstance(raw_outputs, tuple):
            masks_logits, class_logits = raw_outputs
        else:
            masks_logits = raw_outputs.masks_queries_logits
            class_logits = raw_outputs.class_queries_logits
        return semantic_segmentation_decode(masks_logits, class_logits, metadata)

    # ── checkpoint loading ──────────────────────────────────────────────
    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Unlike flat task models, the HF model lives in the ``model.``
        # submodule, so that prefix is structural and must NOT be stripped
        # (the base class strips it). Only EMA prefixes are removed.
        for prefix in ("ema_model.model.", "ema_model."):
            stripped = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if stripped:
                state_dict = stripped
                break
        self.load_state_dict(state_dict, strict=True)

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, self.image_size, self.image_size),
                    dtype=torch.float32,
                    is_batched=True,
                )
            }
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: Any, onnx_outputs: list
    ) -> None:
        for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
            torch.testing.assert_close(
                torch_output.sum(dim=1),
                torch.as_tensor(onnx_output).float().sum(dim=1),
                atol=5e-3,
                rtol=1e-1,
                equal_nan=True,
                check_device=False,
                check_dtype=False,
            )
