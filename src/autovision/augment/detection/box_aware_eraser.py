# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Box-aware background erasure for detection training views.

Foreground-preserving erasure ported from the rf-detr-ssl-v2 design
(docs/learn/train/levjepa-detr-integration.md section 8): the image is
divided into patch-sized cells, every cell overlapping a (margin-dilated)
ground-truth box is protected, and a uniform random subset of the
remaining background cells is zero-filled. Object evidence survives every
draw — Hungarian matching never sees an evidence-free box — while the
constraint mode (``interior=True``) additionally lets box interiors
participate behind a per-box reservation so every object keeps at least
``min_keep`` patches (boxes with at most ``min_keep`` cells stay fully
protected).

Erasure operates on the post-transform pipeline output (normalized float
image, normalized cxcywh boxes), so it is independent of the augmentation
backend and fills with ``0.0``, which is the dataset-mean gray in
normalized space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from autovision.geometry.box_ops import (
    box_cxcywh_norm_to_xyxy,
)

__all__ = [
    "erase_background",
    "protected_cell_mask",
    "reserve_interior_cells",
    "sample_erase_mask",
]

#: Fill value for erased cells. ``0.0`` in ImageNet-normalized space is the dataset-mean gray.
_FILL_VALUE = 0.0

#: Floor for the per-side margin so a tiny box still protects its immediate boundary context.
_MIN_MARGIN_PX = 2.0


def protected_cell_mask(
    boxes: Tensor,
    height: int,
    width: int,
    cell: int = 16,
    padding: float = 0.15,
) -> Tensor:
    """Compute the grid of cells that overlap any margin-dilated ground-truth box.

    A cell is protected when its pixel rectangle shares any overlap with a dilated box; the
    any-overlap rule (rather than cell-center containment) keeps single-cell objects fully safe.

    Args:
        boxes: Normalized cxcywh boxes of shape ``(N, 4)``; may be empty.
        height: Image height in pixels.
        width: Image width in pixels.
        cell: Cell edge in pixels, aligned with the backbone patch size.
        padding: Fraction of the box's own side used to dilate each side, floored at 2px, so
            boundary-context cells survive too.

    Returns:
        Boolean mask of shape ``(height // cell, width // cell)``, ``True`` where protected.
    """
    rows, cols = height // cell, width // cell
    if rows < 1 or cols < 1:
        raise ValueError(f"Image {height}x{width} smaller than one {cell}px cell.")
    mask = torch.zeros(rows, cols, dtype=torch.bool)
    if boxes is None or boxes.numel() == 0:
        return mask

    # Round to the pixel grid: normalized coords carry float noise that
    # would flip the strict overlap inequalities on knife-edge boxes.
    xyxy = torch.round(
        box_cxcywh_norm_to_xyxy(boxes.detach().cpu().float(), (width, height))
    )
    side = torch.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], dim=-1)
    # The 2px floor guards the context ring of tiny boxes when dilation is enabled; padding=0
    # explicitly disables dilation, so no floor applies there.
    margin = (padding * side).clamp(min=_MIN_MARGIN_PX if padding > 0.0 else 0.0)
    x1 = xyxy[:, 0] - margin[:, 0]
    y1 = xyxy[:, 1] - margin[:, 1]
    x2 = xyxy[:, 2] + margin[:, 0]
    y2 = xyxy[:, 3] + margin[:, 1]

    starts = torch.arange(cols, dtype=torch.float32) * cell
    hits_x = (x1[:, None] < (starts + cell)[None, :]) & (x2[:, None] > starts[None, :])
    starts = torch.arange(rows, dtype=torch.float32) * cell
    hits_y = (y1[:, None] < (starts + cell)[None, :]) & (y2[:, None] > starts[None, :])
    return (hits_y[:, :, None] & hits_x[:, None, :]).any(dim=0)


def sample_erase_mask(
    protected: Tensor, ratio: float, generator: torch.Generator | None = None
) -> Tensor:
    """Sample exactly ``round(ratio * |background|)`` background cells to erase.

    Sampling is uniform without replacement over non-protected cells — random
    complementary subsets beat structured patterns at cell granularity.

    Args:
        protected: Boolean cell mask, ``True`` where protected.
        ratio: Target fraction of background cells to erase in ``[0, 1]``.
        generator: Optional RNG for reproducible sampling; ``None`` uses the global generator.

    Returns:
        Boolean mask of the same shape, ``True`` where a background cell should be erased.
    """
    background = ~protected
    count = int(background.sum().item())
    erase = torch.zeros_like(protected)
    if count == 0 or ratio <= 0.0:
        return erase
    selected = min(int(round(ratio * count)), count)
    candidates = background.flatten().nonzero(as_tuple=True)[0]
    picked = torch.randperm(
        candidates.numel(), generator=generator, device=protected.device
    )[:selected]
    erase.view(-1)[candidates[picked]] = True
    return erase


def reserve_interior_cells(
    boxes: Tensor,
    height: int,
    width: int,
    cell: int,
    min_keep: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select the per-box ``min_keep`` interior cells that must survive erasure.

    Each box randomly reserves ``min_keep`` of its own cells (a box with at most ``min_keep``
    cells reserves everything, which is exactly how tiny boxes stay fully protected in interior
    mode). The reservation is a *pre*-constraint: reserved cells enter the protected mask before
    sampling, so every box satisfies the floor by construction and overlapping boxes each keep
    their own floor.

    Args:
        boxes: Normalized cxcywh boxes of shape ``(N, 4)``; may be empty.
        height: Image height in pixels.
        width: Image width in pixels.
        cell: Cell edge in pixels.
        min_keep: Minimum number of interior cells each box must retain (>= 1).
        generator: Optional RNG for reproducible reservation.

    Returns:
        Boolean mask of shape ``(height // cell, width // cell)``, ``True`` where reserved.
    """
    rows, cols = height // cell, width // cell
    reserved = torch.zeros(rows, cols, dtype=torch.bool)
    if boxes is None or boxes.numel() == 0:
        return reserved

    xyxy = torch.round(
        box_cxcywh_norm_to_xyxy(boxes.detach().cpu().float(), (width, height))
    )
    for b in range(xyxy.shape[0]):
        c1 = max(0, min(int(xyxy[b, 0] // cell), cols - 1))
        c2 = max(0, min(int(torch.ceil(xyxy[b, 2] / cell).item()) - 1, cols - 1))
        r1 = max(0, min(int(xyxy[b, 1] // cell), rows - 1))
        r2 = max(0, min(int(torch.ceil(xyxy[b, 3] / cell).item()) - 1, rows - 1))
        if c2 < c1 or r2 < r1:
            continue
        rect = reserved[r1 : r2 + 1, c1 : c2 + 1]
        count = rect.numel()
        keep = min(min_keep, count)
        if keep <= 0:
            continue
        picked = torch.randperm(count, generator=generator)[:keep]
        # Advanced indexing writes through the slice view (a flat .view(-1) would fail on the
        # non-contiguous interior rect whenever the box does not span the full canvas width).
        rect[picked // rect.shape[1], picked % rect.shape[1]] = True
    return reserved


def erase_background(
    image: Tensor,
    boxes: Tensor | None,
    ratio: float,
    padding: float = 0.15,
    cell: int = 16,
    generator: torch.Generator | None = None,
    interior: bool = False,
    min_keep: int = 2,
) -> Tensor:
    """Zero-fill a uniform random subset of background cells, protecting all box-overlapping cells.

    With ``interior=True`` (constrained foreground erasure), box *interiors* become
    erasable behind a per-box reservation: each box randomly reserves ``min_keep`` interior
    cells that are never erased, while the dilation ring outside box interiors stays fully
    protected so boundary-localization evidence survives. Tiny boxes (<= ``min_keep`` cells)
    reserve everything and are therefore untouched.

    Args:
        image: Normalized float image of shape ``(3, H, W)``.
        boxes: Normalized cxcywh boxes of shape ``(N, 4)``; ``None`` or an empty tensor skips
            erasure entirely (conservative: images without annotations stay untouched).
        ratio: Target fraction of erasable cells to erase.
        padding: Box dilation fraction forwarded to :func:`protected_cell_mask`.
        cell: Cell edge in pixels forwarded to :func:`protected_cell_mask`.
        generator: Optional RNG forwarded to :func:`sample_erase_mask`.
        interior: Whether box interiors may be erased behind the ``min_keep`` reservation.
        min_keep: Per-box minimum retained interior cells when ``interior`` is on.

    Returns:
        Image copy with the sampled cells set to the fill value; the input tensor is never mutated.
    """
    if ratio <= 0.0 or boxes is None or boxes.numel() == 0:
        return image
    height, width = int(image.shape[-2]), int(image.shape[-1])
    protected = protected_cell_mask(boxes, height, width, cell=cell, padding=padding)
    if interior:
        interior_cells = protected_cell_mask(
            boxes, height, width, cell=cell, padding=0.0
        )
        reserved = reserve_interior_cells(
            boxes, height, width, cell=cell, min_keep=min_keep, generator=generator
        )
        protected = (protected & ~interior_cells) | reserved
    erase = sample_erase_mask(protected, ratio, generator=generator)
    if not bool(erase.any()):
        return image
    cell_mask = erase.repeat_interleave(cell, dim=0).repeat_interleave(cell, dim=1)[
        :height, :width
    ]
    out = image.clone()
    out[:, cell_mask] = _FILL_VALUE
    return out
