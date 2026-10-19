# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Density-guided DETR encoder: DeFE + MWAS wiring (doc 13 §4.2/§4.6).

Subclasses the shared :class:`HybridEncoder` and mirrors the baseline Dome
``HybridEncoder.forward`` density branch (``src/zoo/dome/hybrid_encoder.py:
511-556``):

- the :class:`~...defe.LiteDeFE` head consumes the shallowest projected
  feature (stride 8 in this pyramid — baseline stride 4, D1) and emits the
  stride-2 density map plus the per-image count estimate;
- the density map is max-pooled to the MWAS window grid and binarized by
  descending threshold (:func:`adaptive_window_mask`); the window mask
  drives both MWAS feature attention and the decoder's query gating;
- when enabled, MWAS window attention enhances the stride-16 level
  in-place before AIFI/FPN/PAN (which run through the inherited,
  behavior-identical helpers);
- in training, per-image Gaussian heatmaps from the GT boxes supervise
  the density map (targets are already normalized cxcywh in this
  framework — no baseline-style coordinate conversion needed).

Returns ``(outs, defe_info)``; ``defe_info`` carries everything the
criterion and the dynamic-query decoder need. ``convert_to_deploy``
switches to the verbatim baseline path for static ONNX export (D8).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autovision.tasks.detection.detr.config import (
    DETRDensityArgs,
)
from autovision.tasks.detection.shared.defe import (
    GaussHeatmapGenerator,
    LiteDeFE,
)
from autovision.tasks.detection.shared.hybrid_encoder import (
    HybridEncoder,
)
from autovision.tasks.detection.shared.mwas import (
    WindowProcessor,
    adaptive_window_mask,
    window_grid_size,
)

__all__ = ["DomeHybridEncoder"]


class DomeHybridEncoder(HybridEncoder):
    """HybridEncoder + DeFE density head + (optional) MWAS attention."""

    def __init__(
        self,
        in_channels,
        hidden_dim: int,
        dim_feedforward: int,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        use_encoder_idx=None,
        density: DETRDensityArgs | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            hidden_dim=hidden_dim,
            dim_feedforward=dim_feedforward,
            expansion=expansion,
            depth_mult=depth_mult,
            use_encoder_idx=use_encoder_idx,
        )
        if density is None:
            raise ValueError("DomeHybridEncoder requires density args.")
        self.density_args = density
        self.defe = LiteDeFE(hidden_dim)
        self.mwas_processor: WindowProcessor | None = (
            WindowProcessor(
                embed_dim=hidden_dim,
                dim_feedforward=dim_feedforward,
            )
            if density.use_mwas
            else None
        )
        self._static_export = False

    def convert_to_deploy(self) -> None:
        """Static-export mode: drop DeFE/MWAS, verbatim baseline path."""
        self._static_export = True

    def forward(self, feats, targets=None):
        if self._static_export:
            return super().forward(feats), None

        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        density, reg_value = self.defe(proj_feats[0])
        stride16_map = proj_feats[1]
        window_size = self.density_args.mwas_window_size
        n_h, n_w = window_grid_size(
            stride16_map.shape[-2], stride16_map.shape[-1], window_size
        )
        pooled_density = F.adaptive_max_pool2d(density, (n_h, n_w))
        window_mask = adaptive_window_mask(
            pooled_density,
            threshold=self.density_args.window_threshold,
            step=self.density_args.window_threshold_step,
        )
        defe_info: dict = {
            "density_map": density,
            "density_pooled": pooled_density,
            "window_mask": window_mask,
            "reg_value": reg_value,
        }

        if self.mwas_processor is not None:
            glob_pos_embed = self._global_position_embedding(
                stride16_map.shape[-1],
                stride16_map.shape[-2],
                self.hidden_dim,
                stride16_map.device,
            )
            proj_feats[1] = self.mwas_processor(
                stride16_map, window_mask, window_size, glob_pos_embed
            )

        if self.training and targets is not None:
            generator = GaussHeatmapGenerator(
                density.shape[-2:], self.density_args.gaussian_sigma_ratio
            )
            heatmaps = [generator(t["boxes"]) for t in targets]
            defe_info["gt_heatmap"] = torch.stack(heatmaps).to(
                device=density.device, dtype=torch.float32
            )

        proj_feats = self._run_aifi(proj_feats)
        outs = self._run_fpn_pan(proj_feats)
        return outs, defe_info

    @staticmethod
    def _global_position_embedding(
        width: int, height: int, embed_dim: int, device: torch.device
    ) -> torch.Tensor:
        """Absolute sincos embedding as a (C, H, W) grid for MWAS.

        ``build_2d_sincos_position_embedding`` flattens its (w, h) mesh in
        w-major order; calling it with (height, width) makes the flatten
        order match the feature map's (y, x) row-major layout (D4). The
        sin/cos frequency halves then encode (y, x) instead of (x, y) —
        an inconsequential relabeling for attention.
        """
        position = HybridEncoder.build_2d_sincos_position_embedding(
            height, width, embed_dim
        )
        return position.view(height, width, -1).permute(2, 0, 1).to(device)
