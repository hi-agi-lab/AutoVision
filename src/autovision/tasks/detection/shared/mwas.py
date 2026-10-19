# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""MWAS: Masked Window Attention Sparsification (Dome-DETR port, doc 13 §4.2).

Port of the baseline ``src/zoo/dome/get_roi_features.py`` (Dome-DETR,
Apache-2.0). Given a binary window mask derived from the DeFE density
map, attention is computed **only inside the selected windows**; enhanced
window features are written back residually, unselected regions pass
through untouched.

Each :class:`AxisPermutedEncoder` layer runs the *same* weights twice:
once across positions within every window (sequence = window tokens,
batch = windows), then once across windows at the same relative position
(sequence = windows, batch = positions) — a 1-D-attention approximation
of dense window-grid interaction. Positional information (absolute
sincos + window-relative MLP coords) keeps both passes location-aware.

Deviations from the baseline (design doc 13 §5):

- D3: images are zero-padded to a whole number of windows instead of
  asserting ``H % window_size == 0`` — the scale-jitter ladder produces
  stride-16 grids (36..64 px) that do not divide by the window size;
- D4: all grid indexing is consistently (y, x);
- D7: the baseline's visualize switches and the ``no_valid_windows.log``
  side channel are dropped — :func:`adaptive_window_mask` guarantees at
  least one selected window per image, so the empty-selection branch is
  unreachable.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    get_activation,
)

__all__ = [
    "AxisPermutedEncoder",
    "AxisPermutedEncoderLayer",
    "WindowProcessor",
    "adaptive_window_mask",
    "window_grid_size",
]


def window_grid_size(height: int, width: int, window_size: int) -> tuple[int, int]:
    """Number of windows covering a (height, width) grid: ceiling division."""
    return (
        max(1, math.ceil(height / window_size)),
        max(1, math.ceil(width / window_size)),
    )


def adaptive_window_mask(
    pooled_density: torch.Tensor,
    *,
    threshold: float = 0.05,
    step: float = 0.01,
) -> torch.Tensor:
    """Binarize the window-pooled density map, guaranteeing a non-empty mask.

    Descends the threshold by ``step`` until at least one window is
    selected; if the density map is entirely flat-zero the argmax window
    is forced (unreachable in practice — the density map is per-image max
    normalized, so its hottest window is 1.0 > threshold). Baseline
    ``adaptive_defe_filter`` works on the pixel-level map and re-pools;
    at window granularity ``max(pooled) > t`` is the same predicate
    (design doc 13 §5, last row).

    Args:
        pooled_density: (B, 1, n_h, n_w) window-pooled density.
    Returns:
        (B, n_h, n_w) bool mask, at least one ``True`` per image.
    """
    masks = pooled_density[:, 0] > threshold
    # Single host sync for the common case (every image non-empty); the
    # per-image descent below only runs for degenerate flat-zero maps.
    if bool(masks.flatten(1).any(dim=1).all()):
        return masks
    for b in range(masks.shape[0]):
        current = threshold - step
        while current > 0 and not masks[b].any():
            masks[b] = pooled_density[b, 0] > current
            current -= step
        if not masks[b].any():
            masks[b].view(-1)[pooled_density[b].argmax()] = True
    return masks


class AxisPermutedEncoderLayer(nn.Module):
    """Post-norm self-attention + FFN block with externally supplied q/k.

    The window attention needs the same weights applied to two different
    sequence arrangements, so q/k are built by the caller and only v is
    implicit (``value=src``).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        src = residual = v
        attn_out, _ = self.self_attn(q, k, value=src)
        src = residual + self.dropout1(attn_out)
        src = self.norm1(src)

        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        return self.norm2(src)


class AxisPermutedEncoder(nn.Module):
    """Window attention with an axis-permuted second pass per layer.

    Inputs (per image, N = selected windows, L = window_size^2):
        src: (N, L, C) window token features;
        pos_embed: (1, L, C) window-relative positional embedding;
        glob_pos_embeds: (N, L, C) absolute positional embedding slices.

    Pass 1 attends within windows (batch = windows); pass 2 transposes the
    sequence so attention crosses windows at matching relative positions
    (batch = positions, sequence = windows).
    """

    def __init__(
        self, encoder_layer: AxisPermutedEncoderLayer, num_layers: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor,
        glob_pos_embeds: torch.Tensor,
    ) -> torch.Tensor:
        output = src
        position = glob_pos_embeds + pos_embed.to(glob_pos_embeds.dtype)
        for layer in self.layers:
            q = k = output + position
            output = layer(q, k, output)
            # Axis-permuted pass: sequence becomes the window axis.
            output_t = output.transpose(0, 1)
            q = k = output_t + position.transpose(0, 1)
            output = layer(q, k, output_t).transpose(0, 1)
        return output


class WindowProcessor(nn.Module):
    """Select windows by density mask, run window attention, write back.

    ``forward(memory, window_mask, window_size, glob_pos_embed)``:

    - ``memory``: (B, C, H, W) stride-16 features;
    - ``window_mask``: (B, n_h, n_w) bool from :func:`adaptive_window_mask`;
    - ``glob_pos_embed``: (C, H, W) absolute sincos embedding (y-major).

    Returns (B, C, H, W): a copy of ``memory`` with attention-encoded
    features *added* inside the selected windows (baseline residual
    semantics), zero-padding cropped away.
    """

    def __init__(
        self,
        embed_dim: int,
        dim_feedforward: int = 1024,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        encoder_layer = AxisPermutedEncoderLayer(
            embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.window_encoder = AxisPermutedEncoder(encoder_layer, num_layers)

    def forward(
        self,
        memory: torch.Tensor,
        window_mask: torch.Tensor,
        window_size: int,
        glob_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = memory.shape
        n_h, n_w = window_grid_size(height, width, window_size)
        pad_h = n_h * window_size - height
        pad_w = n_w * window_size - width

        padded = F.pad(memory, (0, pad_w, 0, pad_h))
        enhanced: torch.Tensor = padded.clone()
        windows = (
            padded.view(
                batch, channels, n_h, window_size, n_w, window_size
            )
            .permute(0, 2, 4, 1, 3, 5)
        )  # (B, n_h, n_w, C, win, win)

        # Absolute position grid aligned with the padded feature layout.
        glob_padded = F.pad(glob_pos_embed, (0, pad_w, 0, pad_h))
        glob_grid = (
            glob_padded.view(channels, n_h, window_size, n_w, window_size)
            .permute(1, 3, 0, 2, 4)
        )  # (n_h, n_w, C, win, win)

        rel_pos = self._relative_embedding(window_size, memory.device).to(
            memory.dtype
        )  # (C, win, win)
        seq_len = window_size * window_size
        rel_pos_flat = rel_pos.reshape(channels, seq_len).permute(1, 0).unsqueeze(0)

        # One host sync for the whole batch: all selected window positions
        # come from a single batched nonzero (repeated `.nonzero()` per
        # image, plus `int(ys[k])` per window, used to pay two pipeline
        # syncs per window — ~1600 of the ~2400 syncs per training step at
        # VisDrone window counts).
        sel_b, sel_y, sel_x = window_mask.nonzero(as_tuple=True)
        sel_b_host = sel_b.cpu().tolist()
        rows_by_image: list[list[int]] = [[] for _ in range(batch)]
        for row, image_id in enumerate(sel_b_host):
            rows_by_image[image_id].append(row)

        blocks: list[torch.Tensor] = []
        put_indices: list[torch.Tensor] = []
        for b in range(batch):
            rows = rows_by_image[b]
            if not rows:
                continue
            row_idx = torch.tensor(rows, device=sel_y.device, dtype=torch.long)
            y_idx = sel_y[row_idx]
            x_idx = sel_x[row_idx]
            # Index-based gather (no boolean indexing → no extra syncs).
            selected_windows = torch.stack([y_idx, x_idx], dim=1)
            window_feats = windows[b][
                selected_windows[:, 0], selected_windows[:, 1]
            ]  # (N, C, win, win)
            window_pos = (
                glob_grid[selected_windows[:, 0], selected_windows[:, 1]]
                .to(memory.dtype)
            )  # (N, C, win, win)
            num_win = window_feats.shape[0]

            feats_flat = window_feats.reshape(num_win, channels, seq_len).permute(
                0, 2, 1
            )  # (N, L, C)
            pos_flat = window_pos.reshape(num_win, channels, seq_len).permute(0, 2, 1)

            encoded = self.window_encoder(
                feats_flat, pos_embed=rel_pos_flat, glob_pos_embeds=pos_flat
            )
            encoded = encoded.permute(0, 2, 1).reshape(
                num_win, channels, window_size, window_size
            )
            blocks.append(encoded)
            put_indices.append(
                torch.stack(
                    [
                        torch.full_like(y_idx, b),
                        y_idx,
                        x_idx,
                    ],
                    dim=1,
                )
            )

        if blocks:
            # One batched residual write. The old loop did, per window,
            # `bf16_slice += encoded` — i.e. bf16(f32(memory) + encoded).
            # Doing the same accumulation on an f32 view of the padded
            # memory and casting back once is value-identical (the bf16 →
            # f32 → bf16 round trip of untouched cells is lossless), and
            # index_put_(accumulate=True) rejects bf16 destinations, which
            # the encoder's fp32 output would trip.
            grid = (
                padded.to(torch.float32)
                .view(batch, channels, n_h, window_size, n_w, window_size)
                .permute(0, 2, 4, 1, 3, 5)
            )  # (B, n_h, n_w, C, win, win), a view of the f32 copy
            index = torch.cat(put_indices, dim=0)
            grid.index_put_(
                (index[:, 0], index[:, 1], index[:, 2]),
                torch.cat(blocks, dim=0),
                accumulate=True,
            )
            enhanced = (
                grid.permute(0, 3, 1, 4, 2, 5)
                .reshape(batch, channels, n_h * window_size, n_w * window_size)
                .to(memory.dtype)
            )

        return enhanced[:, :, :height, :width]

    def _relative_embedding(self, window_size: int, device: torch.device) -> torch.Tensor:
        """Window-relative MLP coordinate embedding, (C, win, win)."""
        axis = torch.arange(window_size, device=device, dtype=torch.float32)
        denom = float(max(window_size - 1, 1))
        grid_y, grid_x = torch.meshgrid(axis / denom, axis / denom, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        embedding = self.rel_pos_encoder(coords)
        return (
            embedding.view(window_size, window_size, -1).permute(2, 0, 1).contiguous()
        )
