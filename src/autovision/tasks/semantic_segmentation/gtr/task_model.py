# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR semantic segmentation task model (inference/export facing).

Architecture per the official GTR semseg configs (roadmap design doc 31
§11 S1): GTR ViT backbone (same taps as the detection family) → a
single-level stride-8 ``GTREncoder`` → :class:`SemSegHead` → bilinear
upsample to the network input resolution. Official MIT checkpoints load
end to end (``load_official_checkpoint``): backbone via the teacher
extractor, ``encoder.*``/``head.*`` by exact name.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn.functional as F

from autovision.augment.transform_args import NormalizeArgs
from autovision.export import ONNXExportMixin
from autovision.registry.gtr_assets import GTR_SEMSEG_VARIANTS, resolve_gtr_asset
from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone
from autovision.tasks.detection.shared.gtr import GTREncoder
from autovision.tasks.semantic_segmentation.gtr.segmentation_head import (
    SemSegHead,
)
from autovision.tasks.task_model import TaskNetwork

__all__ = ["GTRSemanticSegmentation"]

#: GTR ViT tap layers (all sizes ship a 12-layer ViT at patch 16).
_INTERACTION_INDEXES = (3, 7, 11)


class GTRSemanticSegmentation(ONNXExportMixin, TaskNetwork):
    """GTR backbone + stride-8 encoder + FCN head."""

    def __init__(
        self,
        num_classes: int = 19,
        variant: str = "s",
        image_size: tuple[int, int] = (1024, 1024),
        dropout: float = 0.1,
        checkpoint: str | None = None,
        load_checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(),
            ignore_args={"kwargs", "self", "checkpoint", "load_checkpoint"},
        )
        if variant not in GTR_SEMSEG_VARIANTS:
            raise ValueError(
                f"Unknown GTR semseg variant '{variant}'; expected one of "
                f"{sorted(GTR_SEMSEG_VARIANTS)}."
            )
        embed_dim, num_heads, ffn_ratio, hidden = GTR_SEMSEG_VARIANTS[variant]
        self.num_classes = num_classes
        self.variant = variant
        self.image_size = tuple(image_size)
        self.normalize = NormalizeArgs()
        self.backbone = GTRViTBackbone(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            interaction_indexes=_INTERACTION_INDEXES,
        )
        # Official semseg encoder: ONE stride-8 level (base config
        # ``scale_factors: [2.0], feat_strides: [8]``). Its output width
        # is ``hidden`` — equal to the ViT width for S/M, a 256-wide
        # projection for L/X (official configs: ``hidden_dim: 256``).
        self.encoder = GTREncoder(
            in_channels=[embed_dim] * 3,
            hidden_dim=hidden,
            scale_factors=[2.0],
            feat_strides=[8],
            num_blocks=3,
            layer_norm=True,
            act="silu",
        )
        self.head = SemSegHead(
            in_channels=hidden,
            hidden_dim=hidden,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.load_report: dict[str, Any] | None = None
        if checkpoint and load_checkpoint:
            self.load_report = self.load_official_checkpoint(checkpoint)

    # ── official checkpoint loading ─────────────────────────────────────
    def load_official_checkpoint(self, checkpoint: str | pathlib.Path) -> dict[str, Any]:
        """Load an official GTR semseg checkpoint (EMA口径, MIT).

        ``checkpoint`` is either a registered asset name (e.g.
        ``"gtrsemseg_s_cityscapes"`` — sha256-verified) or a file path.
        Returns a load report (backbone / encoder+head coverage).
        """
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
        backbone_mismatch = [
            key
            for key, tensor in backbone_state.items()
            if key in target_backbone and target_backbone[key].shape != tensor.shape
        ]
        if backbone_mismatch:
            # Wrong-variant checkpoints (e.g. an S checkpoint into an
            # L-width net) mismatch on the very first block; surface a
            # curated error instead of torch's raw per-tensor spew.
            raise ValueError(
                f"GTR semseg checkpoint '{path}' does not match variant "
                f"'{self.variant}': {len(backbone_mismatch)} backbone tensors "
                f"differ in shape (first: {backbone_mismatch[0]} "
                f"{tuple(target_backbone[backbone_mismatch[0]].shape)} vs "
                f"{tuple(backbone_state[backbone_mismatch[0]].shape)}). "
                "Use the checkpoint's own variant."
            )
        try:
            self.backbone.backbone_model.load_state_dict(backbone_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"GTR semseg checkpoint '{path}' backbone does not load into "
                f"variant '{self.variant}' strictly: {str(error)[:400]}"
            ) from error
        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        for key, tensor in bundle.state.items():
            if not key.startswith(("encoder.", "head.")):
                continue
            if key not in target:
                continue
            if tensor.shape != target[key].shape:
                skipped_shape.append(key)
                continue
            transferred[key] = tensor
        missing, unexpected = self.load_state_dict(transferred, strict=False)
        encoder_head_keys = [
            key for key in target if key.startswith(("encoder.", "head."))
        ]
        expected = sum(target[key].numel() for key in encoder_head_keys)
        loaded = sum(
            target[key].numel() for key in transferred if key in encoder_head_keys
        )
        report = {
            "checkpoint": str(path),
            "weights_source": bundle.inventory["weight_source"],
            "transferred": len(transferred),
            "skipped_shape": len(skipped_shape),
            "encoder_head_coverage": round(loaded / expected, 4) if expected else 0.0,
            "unexpected": len(unexpected),
        }
        if unexpected or skipped_shape or report["encoder_head_coverage"] < 1.0:
            raise ValueError(
                f"GTR semseg checkpoint '{path}' did not load cleanly: {report} "
                f"(skipped_shape[:5]={skipped_shape[:5]}, "
                f"unexpected[:5]={list(unexpected)[:5]})."
            )
        del missing  # backbone wrapper keys are loaded separately
        return report

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
        feats = self.encoder(self.backbone(x))
        logits = self.head(feats[0])
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
        labels = raw_outputs.argmax(dim=1)
        results = []
        for index, meta in enumerate(metadata):
            in_h, in_w = labels.shape[-2:]
            orig_w, orig_h = float(meta["orig_w"]), float(meta["orig_h"])
            mask = labels[index].float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(
                mask, size=(int(orig_h), int(orig_w)), mode="nearest"
            ).squeeze(0).squeeze(0).long()
            results.append({"mask": mask, "input_w": in_w, "input_h": in_h})
        return results

    # ── checkpoint loading (our own training artifacts) ────────────────
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
        # Single graph output: the per-class logits (B, C, H, W).
        torch.testing.assert_close(
            torch_outputs.sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
