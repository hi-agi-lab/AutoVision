# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv3: a DINOv2 post-train method with a PaKA dense-relational loss.

- [0]: 2025, PaKA: https://arxiv.org/abs/2509.05606

DINOv3 continues ("post-trains") a pretrained backbone with the full DINOv2
objective (DINO + iBOT + KoLeo) PLUS an auxiliary Patch Kernel Alignment
(PaKA / linear CKA) loss that aligns the relational structure of student and
teacher dense patch tokens. It is a thin subclass of :class:`DINOv2`; the
PaKA term is added on top of the DINOv2 loss.

The transform's view layout::

    [global0, global1, dino_local0..L-1,
     clean_global0, clean_global1, paka_local0..K-1]

The two trailing clean globals (augmentation-free renders of the globals)
feed an aug-free EMA teacher (PaKA only; DINO/iBOT keep the augmented
teacher); the K trailing high-overlap locals feed the PaKA student. Each
student local is paired with its parent global only and ROI-aligned onto
the local grid before CKA.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Any

import torch
from torch import Tensor, nn

from autovision.augment.method_transform import MultiViewTransform
from autovision.augment.transform_args import NormalizeArgs
from autovision.optim import OptimizationTargets, get_weight_decay_parameters
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.pretext.dino.dino import (
    DINOTransformArgs,
    _cosine_schedule,
    _update_momentum,
)
from autovision.pretext.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWArgs,
    DINOv2Args,
)
from autovision.pretext.dinov3.constrained_crop import (
    _box_and_flips_from_geometry,
    _GeometryViewTransform,
    render_augmented_local,
    render_clean_global,
    sample_high_overlap_box,
)
from autovision.pretext.dinov3.paika import _linear_cka_loss, _roi_resample_to_grid
from autovision.pretext.recipe import StepOutput

__all__ = ["DINOv3", "DINOv3Args", "DINOv3Transform", "DINOv3TransformArgs"]

logger = logging.getLogger("autovision")


def _restore_orientation(tokens: Tensor, hflip: Tensor, vflip: Tensor) -> Tensor:
    """Undo per-sample recorded flips so tokens share the image orientation.

    Groups samples by flip combination (none / h / v / hv) and flips each
    group back; the masked scatter keeps the autograd path intact for the
    student side. Spec: provenance spec S2.
    """
    out = torch.empty_like(tokens)
    flip_codes = hflip.int() + 2 * vflip.int()
    for code in range(4):
        group = flip_codes == code
        if not bool(group.any()):
            continue
        piece = tokens[group]
        if code & 1:
            piece = piece.flip(-1)
        if code & 2:
            piece = piece.flip(-2)
        out[group] = piece
    return out


def _shared_region(
    geom_a: Tensor,
    geom_b: Tensor,
    out_h: int,
    out_w: int,
) -> tuple[Tensor, Tensor]:
    """Intersection of two crop boxes plus an overlap-validity mask.

    Returns ``(region, valid)``: ``region`` is ``(B, 4)`` xyxy in original
    image pixels — the intersection where it is meaningful, otherwise
    ``geom_a``'s own box (the loss masks invalid rows out, but the
    coordinates must stay well-defined for the resample). ``valid`` is
    ``(B,)`` bool: the intersection must span at least one output cell of
    A's grid in both dimensions; slimmer overlaps carry no relational
    signal. Spec: provenance spec S3.
    """
    lo_x = torch.maximum(geom_a[:, 0], geom_b[:, 0])
    lo_y = torch.maximum(geom_a[:, 1], geom_b[:, 1])
    hi_x = torch.minimum(geom_a[:, 2], geom_b[:, 2])
    hi_y = torch.minimum(geom_a[:, 3], geom_b[:, 3])
    cell_w = (geom_a[:, 2] - geom_a[:, 0]) / max(out_w, 1)
    cell_h = (geom_a[:, 3] - geom_a[:, 1]) / max(out_h, 1)
    valid = ((hi_x - lo_x) >= cell_w) & ((hi_y - lo_y) >= cell_h)
    region = torch.stack(
        (
            torch.where(valid, lo_x, geom_a[:, 0]),
            torch.where(valid, lo_y, geom_a[:, 1]),
            torch.where(valid, hi_x, geom_a[:, 2]),
            torch.where(valid, hi_y, geom_a[:, 3]),
        ),
        dim=1,
    )
    return region, valid


class DINOv3Args(DINOv2Args):
    """Arguments for the DINOv3 (PaKA post-train) method."""

    # PaKA auxiliary loss weight.
    paka_weight: float = 1.0
    # PaKA is skipped until this global step (0 = active for the whole run).
    paka_start_step: int = 0
    # Number of dedicated high-overlap student local crops (must match the
    # transform's paka_num_local).
    paka_num_local: int = 8
    # Per-image token subsample before the O(N^2) CKA kernel.
    paka_max_tokens: int = 512

    def resolve_auto_fields(
        self,
        scaling: Any = None,
        optimizer_args: Any = None,
        wrapped_model: Any = None,
    ) -> None:
        super().resolve_auto_fields(
            scaling=scaling,
            optimizer_args=optimizer_args,
            wrapped_model=wrapped_model,
        )
        if self.paka_num_local < 0:
            raise ValueError(
                f"paka_num_local must be >= 0, got {self.paka_num_local}."
            )
        if not 0.0 < self.paka_weight:
            raise ValueError(f"paka_weight must be > 0, got {self.paka_weight}.")


class DINOv3TransformArgs(DINOTransformArgs):
    """Transform args for the DINOv3 post-train method.

    ``record_geometry`` is on by default (the cross-view PaKA loss needs the
    crop boxes to ROI-align student locals with teacher globals).
    ``paka_num_local`` high-overlap student local crops are sampled inside
    the globals at ``paka_high_overlap_iou`` containment.
    """

    record_geometry: bool = True
    paka_num_local: int = 8
    paka_high_overlap_iou: float = 0.9


class DINOv3Transform(MultiViewTransform):
    """DINO multi-crop transform extended with PaKA clean globals + locals.

    Produces the view layout the DINOv3 method indexes by position::

        [global0, global1, dino_local0..L-1,
         clean_global0, clean_global1, paka_local0..K-1]

    DINO/iBOT see only the leading ``[global, local]`` views unchanged (all
    views carry geometry; the extra keys are ignored by the method slicing).
    """

    def __init__(self, transform_args: DINOv3TransformArgs) -> None:
        flip = transform_args.random_flip
        flip_h = flip.horizontal_prob if flip is not None else 0.0
        flip_v = flip.vertical_prob if flip is not None else 0.0
        global_view_0 = _GeometryViewTransform(
            image_size=transform_args.image_size,
            random_resize=(
                transform_args.random_resize.min_scale,
                transform_args.random_resize.max_scale,
            ),
            random_flip_h=flip_h,
            random_flip_v=flip_v,
            color_jitter=transform_args.color_jitter,
            random_gray_scale=transform_args.random_gray_scale,
            gaussian_blur=transform_args.gaussian_blur,
            solarize=transform_args.solarize,
            normalize=transform_args.normalize,
        )
        global_view_1 = _GeometryViewTransform(
            image_size=transform_args.image_size,
            random_resize=(
                transform_args.random_resize.min_scale,
                transform_args.random_resize.max_scale,
            ),
            random_flip_h=flip_h,
            random_flip_v=flip_v,
            color_jitter=transform_args.color_jitter,
            random_gray_scale=transform_args.random_gray_scale,
            gaussian_blur=transform_args.global_view_1_blur,
            solarize=transform_args.global_view_1_solarize,
            normalize=transform_args.normalize,
        )
        local_view = _GeometryViewTransform(
            image_size=transform_args.local_view_size,
            random_resize=(
                transform_args.local_resize.min_scale,
                transform_args.local_resize.max_scale,
            ),
            random_flip_h=flip_h,
            random_flip_v=flip_v,
            color_jitter=transform_args.color_jitter,
            random_gray_scale=transform_args.random_gray_scale,
            gaussian_blur=transform_args.local_view_blur,
            solarize=None,
            normalize=transform_args.normalize,
        )
        transforms = [global_view_0, global_view_1] + [local_view] * (
            transform_args.num_local_views
        )
        super().__init__(transforms=transforms, transform_args=transform_args)
        # Stored for the clean-teacher / high-overlap-local renders.
        self._image_size = tuple(transform_args.image_size)
        self._normalize: NormalizeArgs = transform_args.normalize
        self._paka_num_local = transform_args.paka_num_local
        self._paka_high_overlap_iou = transform_args.paka_high_overlap_iou
        self._paka_local_size = tuple(transform_args.local_view_size)
        self._paka_color_jitter = transform_args.color_jitter
        self._paka_random_gray_scale = transform_args.random_gray_scale
        self._paka_local_blur = transform_args.local_view_blur
        self._paka_solarize = transform_args.solarize
        self._paka_flip_h = flip_h
        self._paka_flip_v = flip_v

    def __call__(self, input: dict) -> list[dict]:
        image = input["image"]
        views = [transform(image) for transform in self.transforms]

        # Two clean (augmentation-free) renders of the global crops, re-using
        # their recorded geometry, for the PaKA clean teacher.
        for g in range(2):
            views.append(
                render_clean_global(
                    image=image,
                    geometry=views[g]["geometry"],
                    size=self._image_size,
                    normalize=self._normalize,
                )
            )

        # PaKA high-overlap student locals, appended last. Each is sampled
        # inside an alternating global crop (parent = local_idx % 2).
        for local_idx in range(self._paka_num_local):
            parent_geom = views[local_idx % 2]["geometry"]
            # Seed from the torch RNG (which the DataLoader seeds per
            # worker) so seeded runs are reproducible; a bare
            # ``random.Random()`` would draw from OS entropy and break it.
            rng = random.Random(
                int(torch.randint(0, 2**31 - 1, (1,)).item())
            )
            parent_box, _, _ = _box_and_flips_from_geometry(parent_geom)
            local_box = sample_high_overlap_box(
                parent_box=parent_box,
                min_iou=self._paka_high_overlap_iou,
                rng=rng,
            )
            local_hflip = rng.random() < self._paka_flip_h
            local_vflip = rng.random() < self._paka_flip_v
            views.append(
                render_augmented_local(
                    image=image,
                    box=local_box,
                    size=self._paka_local_size,
                    hflip=local_hflip,
                    vflip=local_vflip,
                    color_jitter=self._paka_color_jitter,
                    random_gray_scale=self._paka_random_gray_scale,
                    gaussian_blur=self._paka_local_blur,
                    solarize=self._paka_solarize,
                    normalize=self._normalize,
                )
            )
        return views

    @staticmethod
    def transform_args_cls() -> type[DINOv3TransformArgs]:
        return DINOv3TransformArgs


class _PakaHead(nn.Module):
    """3-layer MLP (embed -> 2048 -> 2048 -> 256, GELU) for PaKA tokens.

    Applied AFTER the ROI alignment (nonlinear head must not distort the
    grid alignment); no weight norm — the PaKA loss is scale-invariant CKA.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class DINOv3(DINOv2):
    """DINOv2 post-train method with a PaKA dense-relational auxiliary loss."""

    def __init__(
        self,
        method_args: DINOv3Args,
        optimizer_args: Any,
        embedding_model: Any,
        global_batch_size: int,
        num_input_channels: int,
    ) -> None:
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
            num_input_channels=num_input_channels,
        )
        embed_dim = embedding_model.embed_dim
        self.student_paka_head = _PakaHead(embed_dim)
        # Deepcopy so student and teacher start identical and parameter order
        # matches for the EMA update.
        self.teacher_paka_head = copy.deepcopy(self.student_paka_head)
        self.teacher_paka_head.eval()
        for parameter in self.teacher_paka_head.parameters():
            parameter.requires_grad_(False)

    # ── contract ────────────────────────────────────────────────────────
    @staticmethod
    def method_args_cls() -> type[DINOv3Args]:
        return DINOv3Args

    @staticmethod
    def optimizer_args_cls(optim_type: str) -> type:
        if optim_type in ("auto", "adamw"):
            return DINOv2AdamWArgs
        from autovision.optim import get_optimizer_args_cls

        return get_optimizer_args_cls(optim_type)

    @staticmethod
    def transform_cls() -> type[DINOv3Transform]:
        return DINOv3Transform

    def train(self, mode: bool = True) -> "DINOv3":
        super().train(mode)
        self.teacher_paka_head.eval()
        return self

    def trainable_modules(self) -> OptimizationTargets:
        return OptimizationTargets(
            targets=[
                self.embedding_model,
                self.student_projection_head,
                self.student_paka_head,
            ]
        )

    # ── forward ─────────────────────────────────────────────────────────
    def training_step_impl(self, batch: dict, batch_idx: int) -> StepOutput:
        args = self.method_args
        views = batch["views"]
        geometries = batch.get("geometries")
        n_clean_global = 2
        n_paka_local = args.paka_num_local
        n_dino = len(views) - n_clean_global - n_paka_local
        if n_dino < 2:
            raise ValueError(
                f"DINOv3 expected at least 2 global views before the "
                f"{n_clean_global} clean globals + {n_paka_local} paka locals, "
                f"but got {len(views)} views."
            )
        # PaKA head EMA follows the same momentum schedule as the backbone.
        momentum = _cosine_schedule(
            self.global_step, self.total_steps, args.momentum_start, args.momentum_end
        )
        _update_momentum(self.student_paka_head, self.teacher_paka_head, momentum)

        # super() sees only the leading [global0, global1, dino_local..] views,
        # so the DINO/iBOT path is the DINOv2 path unchanged.
        result = super().training_step_impl(
            {"views": views[:n_dino]}, batch_idx
        )
        if (
            self.global_step < args.paka_start_step
            or n_paka_local == 0
            or geometries is None
        ):
            return result
        paka = self._paka_loss(
            views=views,
            geometries=geometries,
            n_dino=n_dino,
            n_clean_global=n_clean_global,
            n_paka_local=n_paka_local,
        )
        log_dict = dict(result.log_dict) if result.log_dict else {}
        log_dict["train_loss/paka"] = float(paka.detach())
        return StepOutput(
            loss=result.loss + args.paka_weight * paka,
            log_dict=log_dict,
        )

    # ── PaKA ────────────────────────────────────────────────────────────
    def _paka_loss(
        self,
        views: list[Tensor],
        geometries: list[Tensor],
        n_dino: int,
        n_clean_global: int,
        n_paka_local: int,
    ) -> Tensor:
        """Cross-view PaKA: clean-teacher globals vs high-overlap student locals."""
        clean_views = torch.cat(views[n_dino : n_dino + n_clean_global])
        paka_views = torch.cat(views[n_dino + n_clean_global :])

        # Raw backbone patch tokens (the head is applied AFTER the ROI
        # alignment because of its nonlinear GELU).
        with torch.no_grad():
            teacher_globals = self.teacher_embedding_model(
                clean_views, pool=False
            )  # (2B, C, h, w)
        student_locals = self.embedding_model(
            paka_views, pool=False
        )  # (KB, C, h', w')

        teacher_h, teacher_w = teacher_globals.shape[-2:]
        student_h, student_w = student_locals.shape[-2:]
        n_global = 2
        teacher_per_global = teacher_globals.chunk(n_global)
        student_per_local = student_locals.chunk(n_paka_local)
        global_geoms = list(geometries[:n_global])
        local_geoms = geometries[
            n_dino + n_clean_global : n_dino + n_clean_global + n_paka_local
        ]

        # Parent-only pairing: local k aligns against global (k mod 2).
        parent_of_local = [k % n_global for k in range(n_paka_local)]
        aligned = [
            self._align_cross_view_pair(
                student_tokens=student_per_local[k],
                student_geom=local_geoms[k],
                student_h=student_h,
                student_w=student_w,
                teacher_tokens=teacher_per_global[parent_of_local[k]],
                teacher_geom=global_geoms[parent_of_local[k]],
                teacher_h=teacher_h,
                teacher_w=teacher_w,
                out_h=student_h,
                out_w=student_w,
            )
            for k in range(n_paka_local)
        ]
        student_all = torch.cat([pair[0] for pair in aligned], dim=0)
        teacher_all = torch.cat([pair[1] for pair in aligned], dim=0)
        valid_all = torch.cat([pair[2] for pair in aligned], dim=0)
        batch_size, channels = student_all.shape[:2]
        out_dim = self.student_paka_head.mlp[-1].out_features
        # (M, C, h, w) -> (M*N, C) for the head, then back to (M, N, D).
        student_tokens = self.student_paka_head(
            student_all.flatten(2).permute(0, 2, 1).reshape(-1, channels)
        ).view(batch_size, -1, out_dim)
        with torch.no_grad():
            teacher_tokens = self.teacher_paka_head(
                teacher_all.flatten(2).permute(0, 2, 1).reshape(-1, channels)
            ).view(batch_size, -1, out_dim)
        return _linear_cka_loss(
            student_tokens, teacher_tokens, valid_all, self.method_args.paka_max_tokens
        )

    def _align_cross_view_pair(
        self,
        student_tokens: Tensor,
        student_geom: Tensor,
        student_h: int,
        student_w: int,
        teacher_tokens: Tensor,
        teacher_geom: Tensor,
        teacher_h: int,
        teacher_w: int,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ROI-align a student local with its parent teacher global.

        Both feature maps are resampled over the two crops' shared region
        onto the ``(out_h, out_w)`` grid. Returns ``(student_aligned,
        teacher_aligned, valid)`` with shapes ``(B, C, out_h, out_w)`` and
        ``(B, out_h*out_w)``; ``valid`` is False for whole images whose
        crops do not overlap meaningfully (spec:
        provenance spec S3).
        """
        region, valid = _shared_region(student_geom, teacher_geom, out_h, out_w)
        student_aligned = self._roi_align_view(
            student_tokens,
            student_geom,
            region,
            in_h=student_h,
            in_w=student_w,
            out_h=out_h,
            out_w=out_w,
        )
        with torch.no_grad():
            teacher_aligned = self._roi_align_view(
                teacher_tokens,
                teacher_geom,
                region,
                in_h=teacher_h,
                in_w=teacher_w,
                out_h=out_h,
                out_w=out_w,
            )
        return student_aligned, teacher_aligned, valid[:, None].expand(
            -1, out_h * out_w
        )

    @staticmethod
    def _roi_align_view(
        tokens: Tensor,
        geom: Tensor,
        region: Tensor,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
    ) -> Tensor:
        """Resample one crop's tokens over ``region`` onto a common grid.

        ``region`` is ``(B, 4)`` xyxy in original-image pixels; it is mapped
        into this crop's feature-grid frame (undoing the recorded flips
        first so grid coordinates match the original orientation), then
        resampled with the shared bilinear primitive (spec:
        provenance spec S2).
        """
        feat = _restore_orientation(tokens, geom[:, 6] > 0.5, geom[:, 7] > 0.5)
        crop_w = (geom[:, 2] - geom[:, 0]).clamp(min=1e-6)
        crop_h = (geom[:, 3] - geom[:, 1]).clamp(min=1e-6)
        # Divide-then-scale (not reciprocal-multiply) so a shared region
        # equal to the whole crop maps to the exact full-grid identity box.
        grid_boxes = torch.stack(
            (
                ((region[:, 0] - geom[:, 0]) / crop_w * in_w).clamp(0.0, float(in_w)),
                ((region[:, 1] - geom[:, 1]) / crop_h * in_h).clamp(0.0, float(in_h)),
                ((region[:, 2] - geom[:, 0]) / crop_w * in_w).clamp(0.0, float(in_w)),
                ((region[:, 3] - geom[:, 1]) / crop_h * in_h).clamp(0.0, float(in_h)),
            ),
            dim=1,
        )
        return _roi_resample_to_grid(feat, grid_boxes, out_h=out_h, out_w=out_w)

    # ── optimization ────────────────────────────────────────────────────
    def configure_optimizers(
        self, total_steps: int, num_epochs: int
    ) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)

        param_groups, lr_factors = self._layerwise_param_groups()
        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
            [self.student_projection_head, self.student_paka_head]
        )
        last_layer_params = list(self.student_projection_head.last_layer.parameters())
        last_layer_ids = {id(p) for p in last_layer_params}
        params_weight_decay = [
            p for p in params_weight_decay if id(p) not in last_layer_ids
        ]
        params_no_weight_decay = [
            p for p in params_no_weight_decay if id(p) not in last_layer_ids
        ]
        if params_weight_decay:
            param_groups.append({"name": "params", "params": params_weight_decay})
        if params_no_weight_decay:
            param_groups.append(
                {
                    "name": "params_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                }
            )
        param_groups.append(
            {"name": "params_last_layer", "params": last_layer_params}
        )
        optimizer = self.optimizer_args.get_optimizer(
            params=param_groups, lr_scale=lr_scale
        )
        for group in optimizer.param_groups:
            factor = lr_factors.get(group.get("name") or "", 1.0)
            if factor != 1.0:
                group["lr"] = group["lr"] * factor
        warmup_steps = min(
            int(self.method_args.warmup_steps),
            int(total_steps * self.method_args.warmup_max_steps_fraction),
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, max_epochs=int(total_steps)
        )
        return optimizer, scheduler

    # ── checkpoint: strip teacher weights ───────────────────────────────
    _TEACHER_PREFIXES = (
        "teacher_embedding_model.",
        "teacher_projection_head.",
        "teacher_paka_head.",
    )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        live_modules = {
            self._TEACHER_PREFIXES[0]: self.teacher_embedding_model,
            self._TEACHER_PREFIXES[1]: self.teacher_projection_head,
            self._TEACHER_PREFIXES[2]: self.teacher_paka_head,
        }
        for prefix, module in live_modules.items():
            if not any(key.startswith(prefix) for key in state_dict):
                logger.warning(
                    "Checkpoint has no '%s*' weights; the teacher was "
                    "rebuilt from the current student and needs time to "
                    "re-converge.", prefix,
                )
            for key, value in module.state_dict().items():
                state_dict.setdefault(f"{prefix}{key}", value)
