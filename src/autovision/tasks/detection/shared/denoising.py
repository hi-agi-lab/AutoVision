# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Contrastive denoising (CDN) training group for the RT-DETR family.

Source lineage: RT-DETR contrastive denoising (``lyuwenyu/RT-DETR`` @ ``df2a96c``,
Apache-2.0; also vendored in D-FINE ``@ 956d170``), re-implemented by AutoVision
with the ``max_gt_cap`` guard; provenance ledger entry (P1-4).

Port of the baseline ``denoising.py`` semantics (design doc §8.2). During
training, every GT box is repeated ``2 × num_group`` times — once as a
*positive* query (small noise within the box size) and once as a
*negative* query (noise beyond the box, forcing the model to reject
near-miss localizations). The noised boxes enter the decoder as extra
queries with a known assignment, which is what gives DETR-family models
their fast early convergence: box regression gets dense, well-localized
gradient signal from step one instead of having to discover objects with
random anchors.

Returns denoising queries in **inverse-sigmoid space** plus a block
attention mask: matching queries never see the denoising queries, and
denoising groups never see each other (no information leak).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "get_contrastive_denoising_training_group",
    "get_cdn_matched_indices",
    "inverse_sigmoid",
]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def get_contrastive_denoising_training_group(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    batch_queries_num: list[int] | None = None,
    num_heads: int | None = None,
    max_gt_cap: int | None = None,
):
    """Build the contrastive denoising query group.

    Returns ``(query_content, query_boxes_unact, attn_mask, dn_meta)``:

    - ``query_content``: ``(B, dn, hidden)`` class embeddings for the noised
      queries (``class_embed`` maps class index → embedding; index
      ``num_classes`` is the padding slot).
    - ``query_boxes_unact``: ``(B, dn, 4)`` noised boxes in inverse-sigmoid
      space, ready to be used as decoder reference boxes.
    - ``attn_mask``: ``(dn + num_queries, dn + num_queries)`` bool mask,
      ``True`` = masked out. With per-image query counts (see below) it is
      3-D: ``(B * num_heads, dn + num_queries, ...)``, ``True`` = masked.
    - ``dn_meta``: ``{"dn_positive_idx", "dn_num_group", "dn_num_split"}``
      consumed by the criterion (:func:`get_cdn_matched_indices`) and by the
      decoder's output split.

    Returns ``(None, None, None, None)`` when denoising is disabled for the
    batch (``num_denoising <= 0`` or no GT boxes anywhere).

    Density-guided dynamic queries (design doc 13 §4.4): when
    ``batch_queries_num`` (per-image valid query counts, padded to
    ``num_queries``) and ``num_heads`` are given, the mask is expanded
    per batch head so that valid queries and padding queries are mutually
    invisible (padding sees only itself, keeping the softmax finite).
    With no actual padding the 2-D mask is returned unchanged.
    """
    if num_denoising <= 0 or not targets:
        return None, None, None, None

    num_gts = [int(len(t["labels"])) for t in targets]
    device = targets[0]["labels"].device
    batch_size = len(num_gts)
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    # Optional crash guard for extreme-dense batches (e.g. VisDrone's 900-GT
    # images): cap the per-image DN slot count so the decoder's self-attention
    # over (DN + learned) queries stays bounded on a 24 GB card. Only batches
    # whose max GT count exceeds the cap change behavior — ``num_group`` is
    # already floored to 1 there, and DN supervision then covers the first
    # ``cap`` GTs of those images instead of all of them.
    group_size = min(max_gt_num, max_gt_cap) if max_gt_cap else max_gt_num
    num_group = max(1, num_denoising // group_size)

    # Pad every image's GT list up to the (capped) group size.
    input_query_class = torch.full(
        [batch_size, group_size], num_classes, dtype=torch.int64, device=device
    )
    input_query_bbox = torch.zeros([batch_size, group_size, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, group_size], dtype=torch.bool, device=device)
    for image_index, num_gt in enumerate(num_gts):
        fill = min(num_gt, group_size)
        if fill > 0:
            input_query_class[image_index, :fill] = targets[image_index]["labels"][:fill]
            input_query_bbox[image_index, :fill] = targets[image_index]["boxes"][:fill]
            pad_gt_mask[image_index, :fill] = True

    # Each GT appears as one positive + one negative query per group.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    negative_gt_mask = torch.zeros([batch_size, group_size * 2, 1], device=device)
    negative_gt_mask[:, group_size:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(
        dn_positive_idx,
        [min(n, group_size) * num_group for n in num_gts],
    )

    num_denoising_queries = int(group_size * 2 * num_group)

    # Label noise: with prob label_noise_ratio/2 a positive slot gets a
    # random wrong class (the model must recover the class from the image).
    if label_noise_ratio > 0:
        noise_mask = torch.rand_like(input_query_class, dtype=torch.float) < (
            label_noise_ratio * 0.5
        )
        new_label = torch.randint_like(
            noise_mask, 0, num_classes, dtype=input_query_class.dtype
        )
        input_query_class = torch.where(
            noise_mask & pad_gt_mask, new_label, input_query_class
        )

    # Box noise: positives jitter within the box size, negatives are pushed
    # 1-2 box-sizes away (rand in [1, 2] instead of [0, 1]).
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        # (w, h) tiled to (w, h, w, h): xyxy needs a per-dimension noise
        # magnitude — x-noise from w, y-noise from h.
        diff = (input_query_bbox[..., 2:] * 0.5).tile([1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask
        )
        known_bbox = known_bbox + rand_sign * rand_part * diff
        known_bbox = known_bbox.clamp(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    else:
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    query_content = class_embed(input_query_class)

    # Attention mask: the matching queries cannot see the denoising queries;
    # denoising groups cannot see each other (each GT must be reconstructed
    # from its own noisy version only).
    tgt_size = num_denoising_queries + num_queries
    attn_mask = torch.full(
        [tgt_size, tgt_size], False, dtype=torch.bool, device=device
    )
    attn_mask[num_denoising_queries:, :num_denoising_queries] = True
    for group in range(num_group):
        start = group_size * 2 * group
        end = group_size * 2 * (group + 1)
        if group == num_group - 1:
            attn_mask[start:end, : start] = True
        else:
            attn_mask[start:end, end:num_denoising_queries] = True
            attn_mask[start:end, :start] = True

    if (
        batch_queries_num is not None
        and num_heads is not None
        and any(n < num_queries for n in batch_queries_num)
    ):
        assert max(batch_queries_num) == num_queries, (
            "num_queries must be the padded maximum of batch_queries_num"
        )
        attn_mask = attn_mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
        for image_index, valid_queries in enumerate(batch_queries_num):
            padding_start = num_denoising_queries + valid_queries
            if padding_start >= tgt_size:
                continue
            head_start = image_index * num_heads
            head_end = (image_index + 1) * num_heads
            # Valid queries cannot see padding, padding cannot see valid;
            # padding queries attend to themselves so softmax stays finite.
            attn_mask[
                head_start:head_end, :padding_start, padding_start:
            ] = True
            attn_mask[
                head_start:head_end, padding_start:, :padding_start
            ] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising_queries, num_queries],
        "dn_group_size": group_size,
    }
    return query_content, input_query_bbox_unact, attn_mask, dn_meta


def get_cdn_matched_indices(
    dn_meta: dict, targets: list[dict[str, torch.Tensor]]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Matching assignment for the denoising queries: by construction each
    positive query corresponds to a known GT, no Hungarian matching needed."""
    dn_positive_idx = dn_meta["dn_positive_idx"]
    num_group = dn_meta["dn_num_group"]
    device = targets[0]["labels"].device

    dn_match_indices = []
    group_size = dn_meta.get("dn_group_size")
    for image_index, target in enumerate(targets):
        num_gt = int(len(target["labels"]))
        supervised = min(num_gt, group_size) if group_size else num_gt
        if supervised > 0:
            gt_idx = torch.arange(supervised, dtype=torch.int64, device=device)
            gt_idx = gt_idx.tile(num_group)
            assert len(dn_positive_idx[image_index]) == len(gt_idx)
            dn_match_indices.append((dn_positive_idx[image_index], gt_idx))
        else:
            dn_match_indices.append(
                (
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device),
                )
            )
    return dn_match_indices
