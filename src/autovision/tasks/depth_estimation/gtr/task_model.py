# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR metric-depth task model (inference/export facing, doc 31 §11 S4).

Architecture: GTR ViT backbone → 3-level ``GTREncoder``
(``force_contiguous=False`` — the official depth weights were trained
WITHOUT that copy; measured bit-exact against the official net) → DPT head
with ``log_depth=True`` decode (``exp(clamp(logit, -4, 5))``, meters).
Official MIT checkpoints load end to end (``load_official_checkpoint``).
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn.functional as F

from autovision.augment.transform_args import NormalizeArgs
from autovision.export import ONNXExportMixin
from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.depth_estimation.gtr.dpt_head import DPTDepthHead
from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone
from autovision.tasks.detection.shared.gtr import GTREncoder
from autovision.tasks.task_model import TaskNetwork

__all__ = ["GTRDepthEstimation", "GTR_DEPTH_VARIANTS"]

#: variant -> (embed_dim, num_heads, ffn_ratio, hidden); ``hidden`` is the
#: encoder/DPT input width. S/M coincide with the ViT width; the official
#: L/X depth checkpoints keep a 256-wide encoder/head while the ViT is 384
#: (same pattern as semseg L/X), so the width cannot be derived from
#: ``embed_dim``.
GTR_DEPTH_VARIANTS: dict[str, tuple[int, int, float, int]] = {
    "s": (192, 3, 4.0, 192),
    "m": (256, 4, 4.0, 256),
    "l": (384, 6, 4.0, 256),
    "x": (384, 6, 6.0, 256),
}


class GTRDepthEstimation(ONNXExportMixin, TaskNetwork):
    """GTR backbone + encoder + DPT head (metric depth, meters)."""

    def __init__(
        self,
        variant: str = "s",
        image_size: tuple[int, int] = (640, 640),
        features: int = 128,
        load_checkpoint: bool = True,
        checkpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(),
            ignore_args={"kwargs", "self", "checkpoint", "load_checkpoint"},
        )
        if variant not in GTR_DEPTH_VARIANTS:
            raise ValueError(
                f"Unknown GTR depth variant '{variant}'; expected one of "
                f"{sorted(GTR_DEPTH_VARIANTS)}."
            )
        embed_dim, num_heads, ffn_ratio, hidden = GTR_DEPTH_VARIANTS[variant]
        self.variant = variant
        self.image_size = tuple(image_size)
        self.normalize = NormalizeArgs()
        self.backbone = GTRViTBackbone(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            interaction_indexes=(3, 7, 11),
        )
        self.encoder = GTREncoder(
            in_channels=[embed_dim] * 3,
            hidden_dim=hidden,
            scale_factors=[2.0, 1.0, 0.5],
            feat_strides=[8, 16, 32],
            num_blocks=3,
            layer_norm=True,
            act="silu",
            force_contiguous=False,
        )
        self.decoder = DPTDepthHead(
            in_channels=[hidden] * 3,
            features=features,
            use_bn=False,
            log_depth=True,  # released weights decode via exp(clamp)
        )
        self.load_report: dict[str, Any] | None = None
        if checkpoint and load_checkpoint:
            self.load_report = self.load_official_checkpoint(checkpoint)

    # ── official checkpoint loading ─────────────────────────────────────
    def load_official_checkpoint(self, checkpoint: str | pathlib.Path) -> dict[str, Any]:
        from autovision.tasks.detection.detr.gtr_teacher import (
            extract_gtr_backbone,
            load_gtr_teacher,
        )

        text = str(checkpoint)
        asset = None
        if "/" not in text and not pathlib.Path(text).exists():
            asset = resolve_gtr_asset(text)
            path: pathlib.Path = asset.path
        else:
            path = pathlib.Path(text)
        bundle = load_gtr_teacher(path, asset=asset)
        backbone_state = extract_gtr_backbone(bundle)
        target_backbone = self.backbone.backbone_model.state_dict()
        mismatched = [
            key
            for key, tensor in backbone_state.items()
            if key in target_backbone and target_backbone[key].shape != tensor.shape
        ]
        if mismatched:
            raise ValueError(
                f"GTR depth checkpoint '{path}' does not match variant "
                f"'{self.variant}': {len(mismatched)} backbone tensors differ."
            )
        try:
            self.backbone.backbone_model.load_state_dict(backbone_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"GTR depth checkpoint '{path}' backbone does not load into "
                f"variant '{self.variant}' strictly: {str(error)[:400]}"
            ) from error

        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        for key, tensor in bundle.state.items():
            if not key.startswith(("encoder.", "decoder.")):
                continue
            if key not in target:
                continue
            if tensor.shape != target[key].shape:
                skipped_shape.append(key)
                continue
            transferred[key] = tensor
        missing, unexpected = self.load_state_dict(transferred, strict=False)
        side_keys = [
            key for key in target if key.startswith(("encoder.", "decoder."))
        ]
        expected = sum(target[key].numel() for key in side_keys)
        loaded = sum(target[key].numel() for key in transferred if key in side_keys)
        report = {
            "checkpoint": str(path),
            "weights_source": bundle.inventory["weight_source"],
            "transferred": len(transferred),
            "skipped_shape": len(skipped_shape),
            "encoder_decoder_coverage": round(loaded / expected, 4) if expected else 0.0,
            "unexpected": len(unexpected),
        }
        if unexpected or skipped_shape or report["encoder_decoder_coverage"] < 1.0:
            raise ValueError(
                f"GTR depth checkpoint '{path}' did not load cleanly: {report}"
            )
        del missing
        return report

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
            tensor, [self.image_size[0], self.image_size[1]], antialias=True
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def forward_backend(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(self.backbone(x))
        return self.decoder(feats, x.shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_backend(x)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        del threshold
        depth = raw_outputs if torch.is_tensor(raw_outputs) else raw_outputs.depth
        results = []
        for index, meta in enumerate(metadata):
            orig_w, orig_h = float(meta["orig_w"]), float(meta["orig_h"])
            per_image = depth[index : index + 1].unsqueeze(1)
            if tuple(per_image.shape[-2:]) != (int(orig_h), int(orig_w)):
                per_image = F.interpolate(
                    per_image,
                    size=(int(orig_h), int(orig_w)),
                    mode="bilinear",
                    align_corners=False,
                )
            results.append({"depth": per_image[0, 0]})
        return results

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
        # Single graph output: the dense depth map (B, H, W).
        torch.testing.assert_close(
            torch_outputs.sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
