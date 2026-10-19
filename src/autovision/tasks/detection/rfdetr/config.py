# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR object detection: model configuration."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from autovision.config.base import PydanticConfig
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLParams,
)
from autovision.tasks.train_model import TrainModelArgs
from autovision.types import ImageSizeTuple

__all__ = [
    "RFDETR_MODEL_REGISTRY",
    "RFDETREvalEraseArgs",
    "RFDETRObjectDetectionTrainArgs",
    "RFDETRSSLArgs",
    "resolve_rfdetr_variant",
]

# Per-variant metadata from the official rf-detr configs: native input
# resolution and the input-size unit (patch_size * num_windows) that any
# custom resolution must be divisible by.
RFDETR_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rf-detr-base": {"resolution": 560, "size_divisor": 56},  # patch 14 x 4 windows
    "rf-detr-nano": {"resolution": 384, "size_divisor": 32},  # patch 16 x 2 windows
    "rf-detr-small": {"resolution": 512, "size_divisor": 32},
    "rf-detr-medium": {"resolution": 576, "size_divisor": 32},
    "rf-detr-large": {"resolution": 704, "size_divisor": 32},
}


def resolve_rfdetr_variant(model: str) -> dict[str, Any]:
    """Registry entry for a variant name (``rf-detr-nano`` etc.)."""
    if model not in RFDETR_MODEL_REGISTRY:
        supported = ", ".join(sorted(RFDETR_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown rfdetr detection model '{model}'. Supported: {supported}."
        )
    return RFDETR_MODEL_REGISTRY[model]


class RFDETRSSLArgs(PydanticConfig):
    """Per-object contrastive SSL branch on the rfdetr family (doc 18).

    Port of design doc 14's matched-pool objective onto the HF
    RfDetrForObjectDetection: the pool is the final decoder layer's
    Hungarian match (recomputed with the SAME matcher/costs the HF box
    loss consumes), projected through a train-only SimCSE-style head.
    ``None`` (the default on the train args) keeps the baseline contract
    bit-for-bit. Upstream M4 recommended start: ``unsup`` / ``two_crop``
    / ``loss_coef=0.1`` / ``temp=0.05`` / ``iou_match=0.55``.

    v1 boundaries (mirrors AV-DETR doc 14 §3.3): ``view`` supports
    ``two_crop`` (the measured setting) and ``dropout`` (the zero-data
    ablation control); ``instance_source`` is fixed to the matched pool —
    the label-free Route-B pretraining is an AV-DETR decoder feature.
    """

    model_config = {"extra": "forbid"}

    #: Contrastive objective: SimCSE InfoNCE ("unsup"), the SupCon-style
    #: same-label extension ("sup"), or their sum ("both"). "off" disables
    #: the loss while keeping the view's doubled data pipeline (the
    #: batch-matched baseline).
    mode: Literal["off", "unsup", "sup", "both"] = "unsup"
    #: Weight λ of ``loss_ssl`` in the total loss. Upstream M4: 0.1–0.25 is
    #: the robust range; 1.0 was harmful. Must be > 0 whenever mode is active.
    loss_coef: float = Field(default=0.1, ge=0.0)
    #: Temperature of the cosine logits.
    temp: float = Field(default=0.05, gt=0.0)
    #: "two_crop" pairs every image with a second independent augmentation
    #: (doubled batch); "dropout" draws two stochastic samples through the
    #: projection head with zero data-pipeline changes (ablation control).
    view: Literal["two_crop", "dropout"] = "two_crop"
    #: Projection head architecture: SimCSE's Linear+Tanh or the paper's
    #: two-layer MLP ablation.
    projector: Literal["simcse", "mlp2"] = "simcse"
    #: Contrastive embedding width.
    projector_dim: int = Field(default=256, ge=1)
    #: Dropout applied to the head input (the dropout view's noise source).
    projector_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    #: Cross-view IoU threshold for instance correspondence (two_crop).
    iou_match: float = Field(default=0.55, gt=0.0, le=1.0)
    #: Weight λ_w of the IoU soft-alignment logit bias (positives only).
    iou_bias: float = Field(default=0.5, ge=0.0)
    #: Exponent on the alignment IoU before the log bias.
    iou_gamma: float = Field(default=1.0, gt=0.0)
    #: Restrict same-label soft positives to same-image pairs (sup modes).
    same_image_positive_only: bool = False
    #: Gather the contrastive pool across DDP ranks (padded gather).
    gather_distributed: bool = True
    #: Apply the SSL objective only every k-th step. Controls the SSL
    #: gradient frequency only — the two_crop pipeline still runs 2B images.
    every_n_steps: int = Field(default=1, ge=1)
    #: Also try the horizontally-mirrored alignment when pairing cross-view
    #: instances (views flip independently; the same object lands on
    #: opposite sides in ~50% of pairs). Doc 14 D13.
    flip_invariant_pairing: bool = True
    #: Feed matched instances whose cross-view counterpart failed validation
    #: back into the pool as extra candidates (negatives under "unsup")
    #: instead of dropping them (doc 14 D14).
    use_unpaired_as_negatives: bool = True
    #: SIGReg distribution regularizer (doc 16 R1-A1) on the RAW projector
    #: outputs of the full query pool. 0 disables it. Reference finding:
    #: mainly buys λ_ssl robustness rather than raw mAP.
    sigreg_coef: float = Field(default=0.0, ge=0.0)
    #: Random unit directions per SIGReg evaluation (resampled each step).
    sigreg_slices: int = Field(default=1024, ge=1)
    #: Trapezoidal quadrature knots for the Epps–Pulley integral.
    sigreg_knots: int = Field(default=17, ge=5)

    @model_validator(mode="after")
    def _validate_ssl_configuration(self) -> "RFDETRSSLArgs":
        if self.mode != "off" and self.loss_coef <= 0.0:
            raise ValueError(
                f"ssl mode={self.mode!r} requires loss_coef > 0, got {self.loss_coef}."
            )
        if self.same_image_positive_only and self.view != "two_crop":
            raise ValueError(
                "ssl same_image_positive_only=True requires view='two_crop'."
            )
        if self.sigreg_coef > 0:
            if self.mode == "off":
                raise ValueError(
                    "ssl sigreg_coef>0 requires an active mode "
                    "(mode='off' builds the head for nobody)."
                )
            if self.view != "two_crop":
                raise ValueError(
                    "ssl sigreg_coef>0 requires view='two_crop' (v1 scope)."
                )
        return self

    def to_ssl_params(self) -> SSLParams:
        """Map onto the shared-component :class:`SSLParams` contract."""
        return SSLParams(
            mode=self.mode,
            loss_coef=self.loss_coef,
            temp=self.temp,
            view=self.view,
            iou_match=self.iou_match,
            iou_bias=self.iou_bias,
            iou_gamma=self.iou_gamma,
            same_image_positive_only=self.same_image_positive_only,
            gather_distributed=self.gather_distributed,
            every_n_steps=self.every_n_steps,
            flip_invariant_pairing=self.flip_invariant_pairing,
            use_unpaired_as_negatives=self.use_unpaired_as_negatives,
            instance_source="matched_gt",
            sigreg_coef=self.sigreg_coef,
            sigreg_slices=self.sigreg_slices,
            sigreg_knots=self.sigreg_knots,
        )


class RFDETREvalEraseArgs(PydanticConfig):
    """Box-aware erasure for the secondary eval channel (doc 16 P1.6).

    ``eval_erase`` adds a SECOND validation pass on erased views: every
    val image is box-aware erased (same mechanism as the train-side
    erasure, V2 constrained-interior defaults) before the forward, decoded,
    and evaluated against the UNCHANGED dense ground truth. The primary
    dense channel (``val_metric/*``) is untouched — the eval protocol stays
    baseline-comparable — and the erased channel logs ``val_erased_metric/*``.
    Unlike deployment inference, validation has GT boxes, which is exactly
    the protection source the box-aware eraser requires.

    Deterministic: per-image masks are seeded from the image path, so the
    same image draws the same cells at every validation and every
    checkpoint (metrics comparable across runs).
    """

    model_config = {"extra": "forbid"}

    #: Fraction of erasable (non-protected) cells zero-filled, (0, 1].
    ratio: float = Field(default=0.4, gt=0.0, le=1.0)
    #: GT-box dilation fraction per side (2px floor); 0 disables dilation.
    padding: float = Field(default=0.15, ge=0.0)
    #: Cell edge in px, aligned with the backbone patch size.
    cell: int = Field(default=16, ge=1)
    #: V2 constrained-interior mode: box interiors erasable behind the
    #: per-box reservation (defaults mirror the --erase training recipe).
    interior: bool = True
    #: Per-box reserved interior cells under interior mode.
    min_keep: int = Field(default=2, ge=1)


class RFDETRObjectDetectionTrainArgs(TrainModelArgs, PydanticConfig):
    """Training args for the RF-DETR detection family (HF transformers).

    The optimizer/schedule/EMA defaults follow the official rf-detr recipe
    (AdamW lr 1e-4, encoder lr 1.5e-4 with ViT layer-wise decay 0.8,
    decoder component decay 0.7, clip 0.1, EMA 0.993 over ~100 updates).
    The loss (IA-BCE + L1 + GIoU, Hungarian matcher costs 2/5/2) is the one
    inside HF ``RfDetrForObjectDetection`` — a faithful port of the official
    criterion — so it is consumed, not re-implemented.
    """

    model_config = {"extra": "forbid"}

    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 12_000

    model: str = "rf-detr-nano"
    num_classes: int = 80
    # "auto" resolves to the variant's native resolution (base 560, nano
    # 384, small 512, medium 576, large 704); custom sizes must be divisible
    # by patch_size * num_windows.
    image_size: Literal["auto"] | ImageSizeTuple = "auto"
    load_weights: bool = True  # pretrained COCO checkpoint via the verified loader
    num_queries: int = 300
    num_select: int = 300  # eval decode width (official num_select)

    # optimizer / schedule — official rf-detr recipe
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    lr_component_decay: float = 0.7
    lr_vit_layer_decay: float = 0.8
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.1
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: Literal["auto", "step", "cosine"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"
    lr_milestones: tuple[float, ...] = (0.75,)  # fractions of total steps
    lr_gamma: float = 0.1

    # EMA — official: decay 0.993 with tau=100 warmup (same exponential
    # warmup formula as AutoVision's ModelEMA)
    use_ema_model: bool = True
    ema_momentum: float = 0.993
    ema_warmup_steps: int = 100

    eval_max_dets: int = 100

    #: IoU term of the box loss (design doc 15, single-variable ablation
    #: switch — expected gain is noise-level). "giou" keeps the HF
    #: criterion's GIoU emission; "siou" binds a loss_function mirror that
    #: emits SIoU (Scylla-IoU, arXiv:2205.12740) instead, inheriting the
    #: GIoU weight. Matcher costs stay GIoU in both modes.
    box_regression_iou: Literal["giou", "siou"] = "giou"
    #: SIoU shape-cost exponent (paper experience value 4; no per-dataset search).
    siou_theta: float = 4.0

    #: Per-object contrastive SSL branch (design doc 14, rfdetr port —
    #: doc 18). ``None`` keeps the baseline contract bit-for-bit;
    #: ``mode="off"`` + ``view="two_crop"`` is the batch-matched no-SSL
    #: baseline (doubled pipeline, no loss). The command layer reads this
    #: field to wire the two_crop data pipeline.
    ssl: RFDETRSSLArgs | None = None

    #: Secondary eval channel on box-aware erased views (doc 16 P1.6).
    #: ``None`` (default) keeps single-channel dense validation bit-for-bit.
    eval_erase: RFDETREvalEraseArgs | None = None

    def resolve_variant(self) -> dict[str, Any]:
        return resolve_rfdetr_variant(self.model)

    def resolve_image_size(self) -> None:
        """Pin ``image_size`` to a concrete (h, w) tuple ("auto" -> variant
        native resolution) and validate the patch-window divisibility."""
        info = self.resolve_variant()
        if self.image_size == "auto":
            resolution = int(info["resolution"])
            self.image_size = (resolution, resolution)
            return
        height, width = self.image_size
        divisor = int(info["size_divisor"])
        for name, value in (("height", height), ("width", width)):
            if value % divisor:
                suggestion = max(divisor, round(value / divisor) * divisor)
                raise ValueError(
                    f"rfdetr image_size {self.image_size} is invalid: {name} "
                    f"{value} is not divisible by patch_size * num_windows "
                    f"({divisor} for '{self.model}'). Try {suggestion}."
                )

    def resolve_auto_fields(
        self, total_steps: int, train_num_batches: int, gradient_accumulation_steps: int
    ) -> None:
        self.resolve_image_size()
        if self.lr_scheduler == "auto":
            self.lr_scheduler = "cosine"
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = max(100, min(500, total_steps // 10))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
