# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Slicing-aided inference (SAHI) for large images.

``tile_image`` tiles an image with overlap (always appending the final
aligned tile); ``combine_object_detection_tiles`` merges tile predictions
with a global (full-image) prediction: per-label NMS inside tiles, offset
boxes to original coordinates, then global-vs-local same-label IoU
suppression (baseline parameters: nms 0.3, global/local 0.1).
"""

from __future__ import annotations

import torch
from torchvision.ops import nms

__all__ = ["combine_object_detection_tiles", "tile_image"]


def tile_image(
    image: torch.Tensor,
    overlap: float,
    tile_size: int,
    padding_mode: str = "resize",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile a (C, H, W) image into (N, C, tile, tile).

    Returns ``(tiles, tiles_coordinates)`` with coordinates (x_start,
    y_start). Images smaller than the tile are returned as a single tile.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if padding_mode not in ("resize", "pad"):
        raise ValueError(f"padding_mode must be 'resize' or 'pad', got '{padding_mode}'")
    _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        if height < tile_size or width < tile_size:
            if padding_mode == "pad":
                image = torch.nn.functional.pad(
                    image, (0, tile_size - width, 0, tile_size - height)
                )
            else:
                image = torch.nn.functional.interpolate(
                    image[None].float(), size=(tile_size, tile_size),
                    mode="bilinear", align_corners=False,
                )[0].to(image.dtype)
        return image[None], torch.zeros(1, 2, dtype=torch.int64)

    step = max(1, int((1.0 - overlap) * tile_size))
    max_y = max(height - tile_size, 0)
    max_x = max(width - tile_size, 0)
    y_starts = list(range(0, max_y + 1, step)) or [0]
    x_starts = list(range(0, max_x + 1, step)) or [0]
    if y_starts[-1] != max_y:
        y_starts.append(max_y)
    if x_starts[-1] != max_x:
        x_starts.append(max_x)

    tiles, coordinates = [], []
    for y in y_starts:
        for x in x_starts:
            tile = image[:, y : y + tile_size, x : x + tile_size]
            _, tile_h, tile_w = tile.shape
            if tile_h < tile_size or tile_w < tile_size:
                if padding_mode == "resize":
                    tile = torch.nn.functional.interpolate(
                        tile[None].float(), size=(tile_size, tile_size),
                        mode="bilinear", align_corners=False,
                    )[0].to(image.dtype)
                else:
                    tile = torch.nn.functional.pad(
                        tile, (0, tile_size - tile_w, 0, tile_size - tile_h)
                    )
            tiles.append(tile)
            coordinates.append((x, y))
    return torch.stack(tiles), torch.tensor(coordinates, dtype=torch.int64)


def _overlap_matrix(
    boxes1: torch.Tensor, boxes2: torch.Tensor, metric: str
) -> torch.Tensor:
    """Pairwise IoU or intersection-over-smaller for two box sets."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
             * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
             * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    if metric == "iou":
        denominator = area1 + area2 - intersection
    else:
        denominator = torch.minimum(area1, area2)
    return intersection / denominator.clamp(min=1e-9)


def _nms_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    metric: str,
) -> torch.Tensor:
    """Greedy per-label NMS indices, highest score first."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel():
        winner = order[0]
        keep.append(winner)
        if order.numel() == 1:
            break
        overlaps = _overlap_matrix(
            boxes[winner : winner + 1], boxes[order[1:]], metric
        )[0]
        order = order[1:][overlaps <= threshold]
    return torch.stack(keep)


def _nmm_groups(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    metric: str,
) -> list[list[int]]:
    """Greedy overlap groups for non-maximum merging (highest score seeds)."""
    if boxes.numel() == 0:
        return []
    order = scores.argsort(descending=True).tolist()
    assigned: set[int] = set()
    groups: list[list[int]] = []
    for index in order:
        if index in assigned:
            continue
        overlaps = _overlap_matrix(
            boxes[index : index + 1], boxes, metric
        )[0] > threshold
        group = [
            candidate
            for candidate in range(boxes.shape[0])
            if bool(overlaps[candidate]) and candidate not in assigned
        ]
        if not group:
            group = [index]
        assigned.update(group)
        groups.append(group)
    return groups


def _merge_group(
    boxes: torch.Tensor, scores: torch.Tensor, group: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Union box with an area-weighted confidence for one merge group."""
    members = boxes[torch.tensor(group, dtype=torch.long, device=boxes.device)]
    merged = torch.stack(
        [
            members[:, 0].min(),
            members[:, 1].min(),
            members[:, 2].max(),
            members[:, 3].max(),
        ]
    )
    weights = (
        (members[:, 2] - members[:, 0]).clamp(min=0)
        * (members[:, 3] - members[:, 1]).clamp(min=0)
    )
    member_scores = scores[torch.tensor(group, dtype=torch.long, device=scores.device)]
    total = weights.sum()
    score = (
        (member_scores * weights).sum() / total
        if float(total) > 0
        else member_scores.mean()
    )
    return merged, score


def _filter_per_label(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
    metric: str,
    merge: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NMS or NMM per class and sort the survivors by score."""
    if boxes.numel() == 0:
        return (
            boxes.new_zeros((0, 4)),
            scores.new_zeros((0,)),
            labels.new_zeros((0,)),
        )
    out_boxes: list[torch.Tensor] = []
    out_scores: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    for label in labels.unique():
        mask = labels == label
        label_boxes = boxes[mask]
        label_scores = scores[mask]
        if merge == "nmm":
            for group in _nmm_groups(label_boxes, label_scores, threshold, metric):
                merged_box, merged_score = _merge_group(
                    label_boxes, label_scores, group
                )
                out_boxes.append(merged_box[None])
                out_scores.append(merged_score[None])
                out_labels.append(label[None])
        else:
            keep = (
                nms(label_boxes, label_scores, threshold)
                if metric == "iou"
                else _nms_indices(label_boxes, label_scores, threshold, metric)
            )
            out_boxes.append(label_boxes[keep])
            out_scores.append(label_scores[keep])
            out_labels.append(
                label_boxes.new_full(
                    (keep.numel(),), int(label), dtype=torch.long
                )
            )
    boxes_out = torch.cat(out_boxes)
    scores_out = torch.cat(out_scores)
    labels_out = torch.cat(out_labels)
    order = scores_out.argsort(descending=True)
    return boxes_out[order], scores_out[order], labels_out[order]


def combine_object_detection_tiles(
    global_output: dict[str, torch.Tensor],
    tile_outputs: list[dict[str, torch.Tensor]],
    tile_coordinates: torch.Tensor,
    original_size: tuple[int, int],
    nms_iou_threshold: float = 0.3,
    global_local_iou_threshold: float = 0.1,
    merge: str = "nms",
    overlap_metric: str = "iou",
) -> dict[str, torch.Tensor]:
    """Merge a full-image prediction with per-tile predictions.

    ``tile_outputs[i]``: {"bboxes": (n_i, 4) xyxy **in tile coordinates**,
    "scores", "labels"}; boxes are offset to original coordinates by tile i.

    ``merge="nms"`` (default) keeps the highest-scoring box and suppresses
    same-label duplicates; ``merge="nmm"`` unions each overlap group and
    area-weights its score, which keeps adjacent/truncated fragments as one
    box. ``overlap_metric`` is ``"iou"`` or ``"ios"`` (intersection over
    smaller, more sensitive to nested/truncated boxes). Tiles are filtered
    jointly, so overlap-region duplicates are removed before they reach the
    global/local stage.
    """
    if merge not in {"nms", "nmm"}:
        raise ValueError(f"merge must be 'nms' or 'nmm', got {merge!r}.")
    if overlap_metric not in {"iou", "ios"}:
        raise ValueError(
            f"overlap_metric must be 'iou' or 'ios', got {overlap_metric!r}."
        )
    width, height = original_size

    global_boxes, global_scores, global_labels = _filter_per_label(
        global_output["bboxes"],
        global_output["scores"],
        global_output["labels"],
        threshold=nms_iou_threshold,
        metric=overlap_metric,
        merge=merge,
    )

    tile_boxes_all: list[torch.Tensor] = []
    tile_scores_all: list[torch.Tensor] = []
    tile_labels_all: list[torch.Tensor] = []
    for tile_index, output in enumerate(tile_outputs):
        boxes = output["bboxes"]
        if boxes.numel() == 0:
            continue
        x_start = float(tile_coordinates[tile_index][0])
        y_start = float(tile_coordinates[tile_index][1])
        boxes = boxes.clone()
        boxes[:, 0] += x_start
        boxes[:, 2] += x_start
        boxes[:, 1] += y_start
        boxes[:, 3] += y_start
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        tile_boxes_all.append(boxes)
        tile_scores_all.append(output["scores"])
        tile_labels_all.append(output["labels"])

    if not tile_boxes_all:
        return {
            "bboxes": global_boxes,
            "scores": global_scores,
            "labels": global_labels,
        }
    tile_boxes, tile_scores, tile_labels = _filter_per_label(
        torch.cat(tile_boxes_all),
        torch.cat(tile_scores_all),
        torch.cat(tile_labels_all),
        threshold=nms_iou_threshold,
        metric=overlap_metric,
        merge=merge,
    )

    if merge == "nmm":
        merged_boxes, merged_scores, merged_labels = _filter_per_label(
            torch.cat([global_boxes, tile_boxes]),
            torch.cat([global_scores, tile_scores]),
            torch.cat([global_labels, tile_labels]),
            threshold=nms_iou_threshold,
            metric=overlap_metric,
            merge="nmm",
        )
        return {
            "bboxes": merged_boxes,
            "scores": merged_scores,
            "labels": merged_labels,
        }

    # NMS: global boxes win; drop same-label tile duplicates.
    keep = torch.ones(tile_boxes.shape[0], dtype=torch.bool, device=tile_boxes.device)
    for label in tile_labels.unique():
        tile_mask = tile_labels == label
        global_mask = global_labels == label
        if not global_mask.any():
            continue
        overlaps = _overlap_matrix(
            tile_boxes[tile_mask], global_boxes[global_mask], overlap_metric
        )
        keep[tile_mask] = ~(overlaps > global_local_iou_threshold).any(dim=1)
    merged_boxes = torch.cat([global_boxes, tile_boxes[keep]])
    merged_scores = torch.cat([global_scores, tile_scores[keep]])
    merged_labels = torch.cat([global_labels, tile_labels[keep]])
    order = merged_scores.argsort(descending=True)
    return {
        "bboxes": merged_boxes[order],
        "scores": merged_scores[order],
        "labels": merged_labels[order],
    }
