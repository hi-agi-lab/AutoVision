# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Linear semantic segmentation task model (inference/export facing)."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from autovision.augment.transform_args import NormalizeArgs
from autovision.encoders import FeatureExtractor, get_wrapped_model
from autovision.export import ONNXExportMixin
from autovision.tasks.task_model import TaskNetwork

__all__ = ["LinearSemanticSegmentation"]


class LinearSemanticSegmentation(ONNXExportMixin, TaskNetwork):
    """Frozen-backbone feature map + 1x1 classification head + upsample.

    ``forward_backend`` returns the per-pixel logits (B, num_classes, H, W)
    at the *network input* resolution; the export graph contains no argmax
    or resize, keeping it minimal.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "torchvision/resnet18",
        image_size: tuple[int, int] = (512, 512),
        load_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights", "kwargs", "self"})
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.normalize = NormalizeArgs()
        # FeatureExtractor with an Identity head exposes the unified
        # ``forward(x, pool=...)`` interface over any backbone wrapper.
        self.backbone = FeatureExtractor(
            get_wrapped_model(backbone, num_input_channels=3, load_weights=load_weights),
            embed_dim=None,
        )
        self.head = nn.Conv2d(self.backbone.embed_dim, num_classes, kernel_size=1)

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(
        self, image: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
            [self.image_size[0], self.image_size[1]],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def forward_backend(self, x: torch.Tensor) -> torch.Tensor:
        # FeatureExtractor(pool=False) returns the feature map tensor (B, C, h, w).
        features = self.backbone(x, pool=False)
        logits = self.head(features)
        return F.interpolate(
            logits, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_backend(x)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        del threshold
        labels = raw_outputs.argmax(dim=1)  # (B, H, W) at input resolution
        results = []
        for index, meta in enumerate(metadata):
            in_h, in_w = labels.shape[-2:]
            orig_w, orig_h = float(meta["orig_w"]), float(meta["orig_h"])
            mask = labels[index].float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask, size=(int(orig_h), int(orig_w)), mode="nearest"
            ).squeeze(0).squeeze(0).long()
            results.append(
                {
                    "mask": mask,
                    "input_w": in_w,
                    "input_h": in_h,
                }
            )
        return results

    # ── checkpoint loading ──────────────────────────────────────────────
    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        prefixes = ("ema_model.model.", "ema_model.", "model.")
        for prefix in prefixes:
            stripped = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if stripped:
                state_dict = stripped
                break
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint contains unknown keys: {incompatible.unexpected_keys}."
            )
        if incompatible.missing_keys:
            import logging

            logging.getLogger("autovision").info(
                "Reinitialized %d parameter tensors.", len(incompatible.missing_keys)
            )

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, *self.image_size), dtype=torch.float32, is_batched=True
                )
            }
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: Any, onnx_outputs: list
    ) -> None:
        torch.testing.assert_close(
            torch_outputs.sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
