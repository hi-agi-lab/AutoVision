#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Target-side parity: compare reference and target JSONL predictions.

Each line: ``{"predictions": {"bboxes": [[x1,y1,x2,y2],...],
"scores": [...], "labels": [...]}}``. Matching is label + IoU greedy; the
verdict uses a pixel box tolerance (same spirit as
``autovision.export.verify_end_to_end_parity``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare(reference: list[dict], candidate: list[dict], tolerance: float) -> list[str]:
    problems: list[str] = []
    if len(reference) != len(candidate):
        return [f"line count {len(reference)} != {len(candidate)}"]
    for index, (ref_record, cand_record) in enumerate(zip(reference, candidate)):
        ref = ref_record["predictions"]
        cand = cand_record["predictions"]
        ref_labels = [int(v) for v in ref["labels"]]
        cand_labels = [int(v) for v in cand["labels"]]
        if len(ref_labels) != len(cand_labels):
            problems.append(
                f"line {index}: detection count {len(ref_labels)} != {len(cand_labels)}"
            )
            continue
        used: set[int] = set()
        for cand_index, (label, box) in enumerate(zip(cand_labels, cand["bboxes"])):
            best = None
            best_iou = 0.0
            for ref_index, ref_label in enumerate(ref_labels):
                if ref_index in used or ref_label != label:
                    continue
                value = iou(box, ref["bboxes"][ref_index])
                if value > best_iou:
                    best_iou, best = value, ref_index
            if best is None or best_iou < 0.5:
                problems.append(
                    f"line {index}: candidate {cand_index} label={label} "
                    "has no matching reference"
                )
                continue
            used.add(best)
            diff = max(
                abs(float(a) - float(b))
                for a, b in zip(box, ref["bboxes"][best])
            )
            if diff > tolerance:
                problems.append(
                    f"line {index}: box diff {diff:.3f}px exceeds {tolerance}px"
                )
        if len(used) != len(ref_labels):
            problems.append(f"line {index}: reference detections missing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--box-tolerance", type=float, default=1.0)
    args = parser.parse_args()
    problems = compare(
        read_jsonl(Path(args.reference)),
        read_jsonl(Path(args.candidate)),
        args.box_tolerance,
    )
    if problems:
        for problem in problems[:20]:
            print(f"FAIL {problem}")
        return 1
    print("parity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
