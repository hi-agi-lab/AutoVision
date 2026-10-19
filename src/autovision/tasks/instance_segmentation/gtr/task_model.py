# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR instance-segmentation task model (inference/export facing).

Official lineage (roadmap design doc 31 §11 S2): the full detection stack
(GTR ViT backbone → stride-8/16/32 GTREncoder → ``dfine_official`` GTR
decoder) plus the per-query mask head (``mask_downsample_ratio=4``) the
official ``GTRSeg`` finetunes ship. Official MIT checkpoints load end to
end (``load_official_checkpoint``): backbone via the teacher extractor,
``encoder.*``/``decoder.*`` with the GTR ``norm2→norm3`` key remap;
``anchors``/``valid_mask`` buffers stay model-generated (official solver
convention).
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn.functional as F

from autovision.registry.gtr_assets import GTR_SEG_VARIANTS, resolve_gtr_asset
from autovision.tasks.detection.detr.dfine_official_warm_start import (
    _normalize_key,
)
from autovision.tasks.detection.detr.task_model import DETRObjectDetection

__all__ = ["GTRSegObjectDetection"]


class GTRSegObjectDetection(DETRObjectDetection):
    """GTR backbone+encoder+decoder + mask head (``gtrseg`` lineage)."""

    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "s",
        image_size: tuple[int, int] = (640, 640),
        num_queries: int = 300,
        num_top_queries: int = 300,
        num_denoising: int = 100,
        mask_downsample_ratio: int = 4,
        checkpoint: str | None = None,
        load_checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        if variant not in GTR_SEG_VARIANTS:
            raise ValueError(
                f"Unknown GTR seg variant '{variant}'; expected one of "
                f"{sorted(GTR_SEG_VARIANTS)}."
            )
        super().__init__(
            num_classes=num_classes,
            model_name=f"gtrvit-{variant}",
            image_size=image_size,
            decoder="dfine_official",
            num_queries=num_queries,
            num_top_queries=num_top_queries,
            num_denoising=num_denoising,
            mask_downsample_ratio=mask_downsample_ratio,
            **kwargs,
        )
        self.variant = variant
        self.load_report: dict[str, Any] | None = None
        if checkpoint and load_checkpoint:
            self.load_report = self.load_official_checkpoint(checkpoint)

    # ── official checkpoint loading ─────────────────────────────────────
    def load_official_checkpoint(self, checkpoint: str | pathlib.Path) -> dict[str, Any]:
        """Load an official GTR seg checkpoint (EMA口径, MIT).

        ``checkpoint`` is a registered asset name (sha256-verified) or a
        file path. Returns a load report with the encoder+decoder coverage
        (includes the 35-key mask head; ``anchors``/``valid_mask`` are
        model-generated).
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
                f"GTR seg checkpoint '{path}' does not match variant "
                f"'{self.variant}': {len(mismatched)} backbone tensors differ "
                f"in shape (first: {mismatched[0]})."
            )
        try:
            self.backbone.backbone_model.load_state_dict(backbone_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"GTR seg checkpoint '{path}' backbone does not load into "
                f"variant '{self.variant}' strictly: {str(error)[:400]}"
            ) from error

        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        for key, tensor in bundle.state.items():
            if not key.startswith(("encoder.", "decoder.")):
                continue
            if key.endswith(("anchors", "valid_mask")):
                continue  # model-generated (official solver convention)
            normalized = _normalize_key(key, "gtr")
            if normalized not in target:
                continue
            if tensor.shape != target[normalized].shape:
                skipped_shape.append(key)
                continue
            transferred[normalized] = tensor
        missing, unexpected = self.load_state_dict(transferred, strict=False)
        side_keys = [
            key
            for key in target
            if key.startswith(("encoder.", "decoder."))
            and not key.endswith(("anchors", "valid_mask"))
        ]
        expected = sum(target[key].numel() for key in side_keys)
        loaded = sum(
            target[key].numel() for key in transferred if key in side_keys
        )
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
                f"GTR seg checkpoint '{path}' did not load cleanly: {report} "
                f"(skipped_shape[:5]={skipped_shape[:5]}, "
                f"unexpected[:5]={list(unexpected)[:5]})."
            )
        del missing  # backbone wrapper keys load separately
        return report

    # ── three-stage contract ────────────────────────────────────────────
    def forward_backend(self, x: torch.Tensor):
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"], out["pred_masks"]

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Detection decode + per-instance masks at the original frame.

        Top-k over the flattened (query × class) score matrix — identical
        to :func:`decode_detection_outputs` — then the matched queries'
        masks are sigmoid'd, bilinearly upsampled to the original size and
        binarized at 0.5 (the COCO mask-AP protocol consumes booleans).
        """
        if isinstance(raw_outputs, tuple):
            logits, boxes, masks = raw_outputs
        else:
            logits, boxes, masks = (
                raw_outputs.logits,
                raw_outputs.boxes,
                raw_outputs.masks,
            )
        scores = logits.sigmoid()
        flat_scores = scores.flatten(1)
        num_top = min(self.num_top_queries, flat_scores.shape[1])
        scores_flat, indices = flat_scores.topk(num_top, dim=1)
        classes = indices % scores.shape[2]
        query_indices = indices // scores.shape[2]

        results = []
        for batch_index, meta in enumerate(metadata):
            keep = scores_flat[batch_index] > threshold
            query_index = query_indices[batch_index][keep]
            box = boxes[batch_index, query_index]
            from autovision.geometry.box_ops import box_cxcywh_norm_to_xyxy

            width, height = float(meta["orig_w"]), float(meta["orig_h"])
            xyxy = box_cxcywh_norm_to_xyxy(box, (width, height))
            if masks is not None:
                # Official order: bilinear upsample of the LOGITS, then
                # sigmoid (their postprocessor interpolates before the
                # activation — sigmoid-then-interpolate gives different
                # mask boundaries and, measured, ~1.3 mask-AP on val500).
                instance_masks = F.interpolate(
                    masks[batch_index, query_index].float().unsqueeze(1),
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                instance_masks = instance_masks.sigmoid() > 0.5
            else:
                instance_masks = torch.zeros(
                    (query_index.shape[0], int(height), int(width)), dtype=torch.bool
                )
            results.append(
                {
                    "labels": classes[batch_index][keep],
                    "bboxes": xyxy,
                    "scores": scores_flat[batch_index][keep],
                    # CPU: per-instance masks at the original frame are the
                    # heavy output (~80 MB/image at 300 instances) — keeping
                    # them on the accelerator accumulates the whole val set.
                    "masks": instance_masks.cpu(),
                }
            )
        return results
