# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR oriented-detection task model (inference/export facing, doc 31 §11 S3).

Official lineage: GTR ViT backbone → 3-level ``GTREncoder`` → the
``OBBGTRTransformer`` (rotation-aware deformable cross-attention + Angle
Distribution Refinement over 6 distributions). Official MIT checkpoints
(``gtrobb_s_dota``) load end to end (``load_official_checkpoint``):
backbone via the teacher extractor, ``encoder.*``/``decoder.*`` by exact
name (this lineage names the post-FFN norm ``norm2`` — no rename applies),
with the stale stored ``anchors``/``valid_mask`` skipped per the official
solver convention (the decoder regenerates them from ``eval_spatial_size``,
which the official OBB protocol pins to the 1024 eval canvas).

Training (oriented criterion/matcher/oriented-CDN) is roadmap doc 31 §11
**S3b**; this module ships the verified load + inference path.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch

from autovision.augment.transform_args import NormalizeArgs
from autovision.export import ONNXExportMixin
from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone
from autovision.tasks.detection.shared.gtr import GTREncoder
from autovision.tasks.detection.shared.gtr.obb_decoder import OBBGTRTransformer
from autovision.tasks.detection.shared.gtr.rbox_ops import PI
from autovision.tasks.oriented_detection.shared.obb_decoder import (
    decode_oriented_detection_outputs,
)
from autovision.tasks.task_model import TaskNetwork

__all__ = ["GTRROBBOrientedDetection", "GTR_OBB_VARIANTS"]

#: variant -> (embed_dim, num_heads, ffn_ratio, hidden, dim_feedforward,
#: num_layers). The decoder runs at ``hidden`` (not the ViT width): the
#: official X checkpoint is a 384-wide ViT with a 256-wide encoder/decoder,
#: a 2048 FFN and 4 layers; S coincides at 192 with FFN 512.
GTR_OBB_VARIANTS: dict[str, tuple[int, int, float, int, int, int]] = {
    "s": (192, 3, 4.0, 192, 512, 4),
    "x": (384, 6, 6.0, 256, 2048, 4),
}


class GTRROBBOrientedDetection(ONNXExportMixin, TaskNetwork):
    """GTR backbone + encoder + OBB decoder (oriented boxes)."""

    def __init__(
        self,
        num_classes: int = 15,  # DOTA-v1.0 classes
        variant: str = "s",
        image_size: tuple[int, int] = (1024, 1024),  # official OBB eval canvas
        num_queries: int = 300,
        num_top_queries: int = 300,
        num_denoising: int = 100,
        nms_iou_threshold: float = 1.0,  # 1.0 = NMS-free (official protocol)
        load_checkpoint: bool = True,
        checkpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(),
            ignore_args={"kwargs", "self", "checkpoint", "load_checkpoint"},
        )
        if variant not in GTR_OBB_VARIANTS:
            raise ValueError(
                f"Unknown GTR OBB variant '{variant}'; expected one of "
                f"{sorted(GTR_OBB_VARIANTS)}."
            )
        embed_dim, num_heads, ffn_ratio, hidden, dim_feedforward, num_layers = (
            GTR_OBB_VARIANTS[variant]
        )
        self.num_classes = num_classes
        self.variant = variant
        self.image_size = tuple(image_size)
        self.num_top_queries = num_top_queries
        self.nms_iou_threshold = nms_iou_threshold
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
        )
        self.decoder = OBBGTRTransformer(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_queries=num_queries,
            feat_channels=[hidden] * 3,
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[3, 6, 3],
            # Decoder attention heads are the official ctor default (8),
            # INDEPENDENT of the backbone's head count (measured: the S
            # checkpoint's sampling_offsets are 8 x sum(points) wide).
            nhead=8,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation="silu",
            num_denoising=num_denoising,
            eval_spatial_size=self.image_size,
            eval_idx=-1,
            group_detr=3,
        )
        self.load_report: dict[str, Any] | None = None
        if checkpoint and load_checkpoint:
            self.load_report = self.load_official_checkpoint(checkpoint)

    # ── official checkpoint loading ─────────────────────────────────────
    def load_official_checkpoint(self, checkpoint: str | pathlib.Path) -> dict[str, Any]:
        """Load an official GTR OBB checkpoint (EMA口径, MIT).

        ``checkpoint`` is a registered asset name (sha256-verified) or a
        file path. ``anchors``/``valid_mask`` are skipped (stale in the
        checkpoints; the official solver keeps the freshly generated
        buffers, and so does this loader).
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
        mismatched = [
            key
            for key, tensor in backbone_state.items()
            if key in target_backbone and target_backbone[key].shape != tensor.shape
        ]
        if mismatched:
            raise ValueError(
                f"GTR OBB checkpoint '{path}' does not match variant "
                f"'{self.variant}': {len(mismatched)} backbone tensors differ."
            )
        try:
            self.backbone.backbone_model.load_state_dict(backbone_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"GTR OBB checkpoint '{path}' backbone does not load into "
                f"variant '{self.variant}' strictly: {str(error)[:400]}"
            ) from error

        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        for key, tensor in bundle.state.items():
            if not key.startswith(("encoder.", "decoder.")):
                continue
            if key.endswith(("anchors", "valid_mask")):
                continue  # stale; regenerated from eval_spatial_size
            if key not in target:
                continue
            if tensor.shape != target[key].shape:
                skipped_shape.append(key)
                continue
            transferred[key] = tensor
        missing, unexpected = self.load_state_dict(transferred, strict=False)
        side_keys = [
            key
            for key in target
            if key.startswith(("encoder.", "decoder."))
            and not key.endswith(("anchors", "valid_mask"))
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
                f"GTR OBB checkpoint '{path}' did not load cleanly: {report} "
                f"(skipped_shape[:5]={skipped_shape[:5]})"
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

    def _forward_train(self, x: torch.Tensor, targets=None) -> dict:
        """Full forward; in training mode the OBB decoder runs the oriented
        CDN internally and emits the criterion-ready dict (upstream layout)."""
        feats = self.encoder(self.backbone(x))
        return self.decoder(feats, targets=targets)

    def forward_backend(self, x: torch.Tensor):
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor):
        return self.forward_backend(x)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decode to per-image rotated polygons in original pixels.

        ``pred_boxes`` carry (cx, cy, w, h, a) sigmoid-domain rboxes
        (a = theta / pi, long-edge convention); the decode helper consumes
        normalized cxcywh + radians, so the angle is converted here. The
        default ``nms_iou_threshold=1.0`` keeps the official NMS-free
        protocol semantics.
        """
        if isinstance(raw_outputs, tuple):
            logits, boxes = raw_outputs
        else:
            logits, boxes = raw_outputs.logits, raw_outputs.boxes
        return decode_oriented_detection_outputs(
            (logits, boxes[..., :4], boxes[..., 4:5] * PI),
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
            iou_threshold=self.nms_iou_threshold,
        )

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
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        # Two graph outputs: pred_logits, then pred_boxes (cx, cy, w, h, a).
        torch.testing.assert_close(
            torch_outputs[0].sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[1].sum(dim=1),
            torch.as_tensor(onnx_outputs[1]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
