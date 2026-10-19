# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Label conversion for the EOMT criterion.

The HF EOMT loss consumes per-instance binary masks (Mask2Former
convention). Semantic datasets provide class-index maps, so every present
class becomes one binary mask instance.
"""

from __future__ import annotations

import torch

__all__ = ["semantic_masks_to_instances"]


def semantic_masks_to_instances(
    masks: torch.Tensor,
    ignore_index: int = 255,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Class-index semantic maps -> per-image instance lists.

    ``masks``: (B, H, W) long class-index maps. Returns
    ``(mask_labels, class_labels)`` where ``mask_labels[i]`` is
    (K_i, H, W) float binary masks (one per present class) and
    ``class_labels[i]`` is (K_i,) the corresponding class ids, both on the
    input device. Pixels equal to ``ignore_index`` never spawn instances
    but stay inside the binary masks as 0 (they are masked out downstream
    by the loss semantics, matching the baseline conventions).
    """
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for mask in masks:
        classes = torch.unique(mask)
        classes = classes[(classes >= 0) & (classes != ignore_index)]
        mask_labels.append(
            (mask[None, :, :] == classes[:, None, None]).to(torch.float)
        )
        class_labels.append(classes.to(torch.long))
    return mask_labels, class_labels
