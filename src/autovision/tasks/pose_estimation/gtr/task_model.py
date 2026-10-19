# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose-estimation task model (inference/export facing, doc 31 §11 S5).

Official lineage: GTR ViT backbone → 3-level ``GTREncoder`` → the
``GTRPoseTransformer`` (DETRPose-style: within/across-instance self
attention + MSDeformAttn + per-keypoint FDR over 2*(reg_max+1) bins).
Official MIT checkpoints (``gtrpose_*_coco``) load end to end
(``load_official_checkpoint``): backbone via the teacher extractor,
``encoder.*``/``decoder.*`` by exact name (this lineage names the
post-FFN norm ``norm2``; stale stored ``anchors``/``valid_mask`` skipped
per the official solver convention).

Training (CDN + keypoint loss/matcher) is roadmap doc 31 §11 **S5b**.
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
from autovision.tasks.detection.shared.gtr.pose_decoder import GTRPoseTransformer
from autovision.tasks.task_model import TaskNetwork

__all__ = ["GTRPoseEstimation", "GTR_POSE_VARIANTS"]

#: variant -> (embed_dim, num_heads, ffn_ratio, hidden, decoder_layers,
#: decoder_ffn) from the official per-variant finetune configs
#: (``configs/pose/coco_pose_finetune/gtrpose_*.yml``), cross-checked
#: against the checkpoint shapes (2026-10-16): L/X keep a **256-wide
#: encoder+decoder** while their ViT is 384 (the encoder down-projects —
#: same trap class as semseg L/X), decoder depth is 3 for S and 4 for
#: M/L/X, and FFN widths are 512/512/1024/2048.
GTR_POSE_VARIANTS: dict[str, tuple[int, int, float, int, int, int]] = {
    "s": (192, 3, 4.0, 192, 3, 512),
    "m": (256, 4, 4.0, 256, 4, 512),
    "l": (384, 6, 4.0, 256, 4, 1024),
    "x": (384, 6, 6.0, 256, 4, 2048),
}


class GTRPoseEstimation(ONNXExportMixin, TaskNetwork):
    """GTR backbone + encoder + pose decoder (person keypoints)."""

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "s",
        image_size: tuple[int, int] = (640, 640),
        num_queries: int = 60,
        num_decoder_layers: int | None = None,
        dim_feedforward: int | None = None,
        num_body_points: int = 17,
        num_select: int = 60,
        load_checkpoint: bool = True,
        checkpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(),
            ignore_args={"kwargs", "self", "checkpoint", "load_checkpoint"},
        )
        if variant not in GTR_POSE_VARIANTS:
            raise ValueError(
                f"Unknown GTR pose variant '{variant}'; expected one of "
                f"{sorted(GTR_POSE_VARIANTS)}."
            )
        embed_dim, num_heads, ffn_ratio, hidden, official_layers, official_ffn = (
            GTR_POSE_VARIANTS[variant]
        )
        # None keeps the variant's official shape (the shape gate would
        # refuse an official checkpoint against anything else).
        if num_decoder_layers is None:
            num_decoder_layers = official_layers
        if dim_feedforward is None:
            dim_feedforward = official_ffn
        self.num_classes = num_classes
        self.variant = variant
        self.image_size = tuple(image_size)
        self.num_body_points = num_body_points
        self.num_select = num_select
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
        self.decoder = GTRPoseTransformer(
            hidden_dim=hidden,
            nhead=8,  # official pose-decoder default (independent of the backbone)
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            activation="relu",
            num_feature_levels=3,
            dec_n_points=4,
            learnable_tgt_init=True,
            two_stage_type="standard",
            num_classes=num_classes,
            num_body_points=num_body_points,
            feat_strides=[8, 16, 32],
            eval_spatial_size=self.image_size,
            reg_max=32,
            reg_scale=4,
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
                f"GTR pose checkpoint '{path}' does not match variant "
                f"'{self.variant}': {len(mismatched)} backbone tensors differ "
                f"(first: {mismatched[0]})."
            )
        try:
            self.backbone.backbone_model.load_state_dict(backbone_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"GTR pose checkpoint '{path}' backbone does not load into "
                f"variant '{self.variant}' strictly: {str(error)[:400]}"
            ) from error

        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        checkpoint_only: list[str] = []
        for key, tensor in bundle.state.items():
            if not key.startswith(("encoder.", "decoder.")):
                continue
            if key.endswith(("anchors", "valid_mask")):
                continue  # stale; regenerated from eval_spatial_size
            if key not in target:
                # No counterpart at all — e.g. a decoder deeper than ours
                # (official M/L/X decode 4 layers, S decodes 3). Coverage
                # counting alone cannot see this direction.
                checkpoint_only.append(key)
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
            "checkpoint_only": len(checkpoint_only),
            "encoder_decoder_coverage": round(loaded / expected, 4) if expected else 0.0,
            "unexpected": len(unexpected),
        }
        if unexpected or skipped_shape or checkpoint_only or report["encoder_decoder_coverage"] < 1.0:
            raise ValueError(
                f"GTR pose checkpoint '{path}' did not load cleanly: {report} "
                f"(skipped_shape[:5]={skipped_shape[:5]}, "
                f"checkpoint_only[:5]={checkpoint_only[:5]})"
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
        """Full forward; in training mode the decoder runs the pose CDN
        internally (upstream passes the image for the DN noise scaling)."""
        feats = self.encoder(self.backbone(x))
        return self.decoder(feats, targets=targets, samples=x if self.training else None)

    def forward_backend(self, x: torch.Tensor):
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_keypoints"]

    def forward(self, x: torch.Tensor):
        return self.forward_backend(x)

    @torch.no_grad()
    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
        target_sizes: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """Port of ``GTRPosePostProcessor``: top-k over the flattened
        (query × class) scores; keypoints scaled to the original frame and
        returned COCO-style as ``[x, y, v]`` triplets with ``v = 1``.

        ``target_sizes`` (B, 2) as (w, h) overrides the metadata-derived
        sizes (export/CLI paths pass explicit sizes); metadata entries
        otherwise carry ``orig_w``/``orig_h``.
        """
        if isinstance(raw_outputs, tuple):
            out_logits, out_keypoints = raw_outputs
        else:
            out_logits, out_keypoints = raw_outputs.logits, raw_outputs.keypoints
        if target_sizes is None:
            target_sizes = torch.tensor(
                [[float(m["orig_w"]), float(m["orig_h"])] for m in metadata],
                device=out_logits.device,
            )
        num_select = min(self.num_select, out_logits.shape[1] * out_logits.shape[2])
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_keypoints = (topk_indexes.float() // out_logits.shape[2]).long()
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(
            out_keypoints,
            1,
            topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points * 2),
        )
        keypoints = keypoints * target_sizes.repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = keypoints.unflatten(-1, (-1, 2))
        keypoints_res = torch.cat(
            [keypoints_res, torch.ones_like(keypoints_res[..., 0:1])], dim=-1
        ).flatten(-2)
        results = []
        for image_index in range(scores.shape[0]):
            keep = scores[image_index] > threshold
            results.append(
                {
                    "scores": scores[image_index][keep],
                    "labels": labels[image_index][keep],
                    "keypoints": keypoints_res[image_index][keep],
                }
            )
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
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        # Two graph outputs: pred_logits, then pred_keypoints.
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
