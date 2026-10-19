#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end detection self-training loop (roadmap C1, phase 3).

One round of the pseudo-labeling closed loop, optionally repeated:

    teacher (.pt)  --generate_pseudo_labels-->  COCO pseudo labels
                   --(+ optional labeled train set, merged)-->
                   student training (train_object_detection)
                   --> exported model
                   --val on the labeled val split-->  metrics (the read)

With ``--rounds > 1`` the improved student becomes the next round's
teacher; a round that fails ``--min-improvement`` is recorded but not
promoted, and the loop stops. Artifacts live under ``--out/rounds/r{i}/``
with a top-level ``summary.json`` / ``summary.md``. ``--rounds 1`` keeps
the historical layout (``out/pseudo``, ``out/student``, ``out/report.json``).

Usage::

    python scripts/self_train_detect.py \
        --teacher runs/teacher/exported_models/exported_best.pt \
        --images /data/unlabeled_pool \
        --val-data coco_val.yaml \
        --out runs/self_train_r1 \
        --student rtdetr/resnet18 --steps 2000 --batch-size 8 \
        --threshold 0.5 [--sahi] [--labeled coco_train.yaml] \
        [--rounds 3 --min-improvement 0.2]

``--val-data`` is the labeled validation spec (yaml path or a dict literal)
whose ``val`` entry carries the annotations json + images dir; its category
list is what the pseudo labels are written against (the teacher's class
list). The val split is evaluation-only: promote a student only when its
AP improves, never train on the validation images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

import autovision  # noqa: E402
from autovision.data.convert import merge_coco  # noqa: E402
from autovision.tasks.detection.shared.pseudo_label import (  # noqa: E402
    generate_pseudo_labels,
)


def _load_data_spec(value: str) -> dict[str, Any]:
    if value.strip().startswith("{"):
        return json.loads(value)
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"data spec not found: '{value}'")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"data spec must be a mapping: '{value}'")
    return loaded


def _categories_of(spec: dict[str, Any], split: str) -> dict[int, str]:
    entry = spec.get(split)
    if not isinstance(entry, dict) or "annotations" not in entry:
        raise SystemExit(f"the data spec needs a '{split}' entry with annotations.")
    payload = json.loads(Path(entry["annotations"]).read_text())
    return {int(c["id"]): str(c["name"]) for c in payload.get("categories", [])}


def _absolutize(annotations_json: Path, images_root: Path, out_json: Path) -> Path:
    """Rewrite ``file_name``s to absolute paths (for cross-root merging)."""
    payload = json.loads(annotations_json.read_text())
    for image in payload.get("images", []):
        name = Path(str(image["file_name"]))
        image["file_name"] = (
            name.as_posix() if name.is_absolute() else (images_root / name).resolve().as_posix()
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return out_json


def _run_round(
    args: argparse.Namespace,
    *,
    round_index: int,
    round_dir: Path,
    teacher_path: Path,
    val_spec: dict[str, Any],
    categories: dict[int, str],
) -> dict[str, Any]:
    round_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. pseudo labels from the teacher ───────────────────────────────
    teacher = autovision.load_task_model(teacher_path, device=None)
    pseudo = generate_pseudo_labels(
        teacher,
        args.images,
        round_dir / "pseudo" / "train.json",
        threshold=args.threshold,
        class_thresholds=json.loads(args.class_thresholds) if args.class_thresholds else None,
        categories=categories,
        sahi=args.sahi,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.pseudo_batch_size,
        max_images=args.max_images,
    )
    (round_dir / "pseudo" / "meta.json").write_text(
        json.dumps(pseudo.to_dict(), indent=2)
    )
    print(
        f"[pseudo] round {round_index}: {pseudo.annotations} labels over "
        f"{pseudo.images} images (threshold {args.threshold}, sahi={args.sahi}) "
        f"-> {pseudo.out_json}"
    )
    if pseudo.annotations == 0:
        raise SystemExit(
            f"No pseudo labels above threshold {args.threshold} over "
            f"{pseudo.images} images — lower --threshold, check the teacher, "
            "or use --class-thresholds. Refusing to train on empty labels."
        )

    # ── 2. student train spec: pseudo (+ optional labeled, merged) ──────
    train_annotations = pseudo.out_json
    if args.labeled:
        labeled_spec = _load_data_spec(args.labeled)
        labeled_entry = labeled_spec.get("train")
        if not isinstance(labeled_entry, dict) or "annotations" not in labeled_entry:
            raise SystemExit("--labeled needs a 'train' entry with annotations.")
        labeled_abs = _absolutize(
            Path(labeled_entry["annotations"]),
            Path(labeled_entry.get("images") or Path(labeled_entry["annotations"]).parent),
            round_dir / "labeled_absolute.json",
        )
        train_annotations = merge_coco(
            [labeled_abs, pseudo.out_json], round_dir / "train_merged.json"
        )
        print(f"[merge] labeled + pseudo -> {train_annotations}")
    train_spec = {
        "format": "coco",
        "train": {
            "annotations": str(train_annotations),
            "images": str(Path(args.images).resolve()),
        },
        "val": val_spec["val"],
    }

    # ── 3. train the student ────────────────────────────────────────────
    student_kwargs: dict[str, Any] = {}
    if args.model_args:
        student_kwargs["model_args"] = json.loads(args.model_args)
    result = autovision.train_object_detection(
        out=round_dir / "student",
        data=train_spec,
        model=args.student,
        steps=args.steps,
        batch_size=args.batch_size,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        **student_kwargs,
    )
    print(
        f"[train] round {round_index}: student finished: {result.steps_run} steps -> "
        f"{result.best_exported_model_path}"
    )

    # ── 4. evaluate the produced model (the round's read) ───────────────
    student = autovision.load_model(result.best_exported_model_path, device=None)
    metrics = student.val(data=val_spec, split="val")
    print(
        f"[val] round {round_index}: "
        + " | ".join(f"{key} {value:.4f}" for key, value in metrics.items())
    )

    report = {
        "round": round_index,
        "teacher": str(teacher_path),
        "student": args.student,
        "pseudo": pseudo.to_dict(),
        "labeled_merged": bool(args.labeled),
        "train_annotations": str(train_annotations),
        "steps": result.steps_run,
        "exported_model": str(result.best_exported_model_path),
        "val_metrics": {key: float(value) for key, value in metrics.items()},
    }
    (round_dir / "report.json").write_text(json.dumps(report, indent=2))
    return {
        "round": round_index,
        "teacher": str(teacher_path),
        "student": str(result.best_exported_model_path),
        "pseudo_annotations": pseudo.annotations,
        "pseudo_images": pseudo.images,
        "val_metrics": report["val_metrics"],
        "report": str(round_dir / "report.json"),
    }


def _ap(metrics: dict[str, float]) -> float:
    return float(metrics.get("map", metrics.get("map_50", 0.0)))


def _write_summary(
    out_dir: Path,
    rounds: list[dict[str, Any]],
    *,
    stop_reason: str,
    min_improvement: float,
    best: dict[str, Any] | None,
) -> None:
    best_model = best["student"] if best is not None else None
    payload = {
        "rounds": rounds,
        "stop_reason": stop_reason,
        "min_improvement": min_improvement,
        "best_model": best_model,
        "best_map": _ap(best["val_metrics"]) if best is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    lines = ["# 自训练轮次汇总\n"]
    lines.append(f"- 停止原因：`{stop_reason}`；最小提升：{min_improvement}\n")
    lines.append("| round | pseudo | AP | student |")
    lines.append("|---|---|---|---|")
    for entry in rounds:
        lines.append(
            f"| {entry['round']} | {entry['pseudo_annotations']} | "
            f"{_ap(entry['val_metrics']):.4f} | {entry['student']} |"
        )
    lines.append(f"\n最优模型：`{best_model}`\n")
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", required=True, help="exported teacher .pt (or registry key path)")
    parser.add_argument("--images", required=True, help="unlabeled image directory")
    parser.add_argument("--val-data", required=True, help="labeled val spec (yaml path or json literal)")
    parser.add_argument("--out", required=True, help="round output directory")
    parser.add_argument("--student", default="rtdetr/resnet18", help="student model key")
    parser.add_argument("--labeled", default=None, help="optional labeled train spec to merge in")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class-thresholds", default=None,
                        help='per-class overrides, e.g. \'{"person": 0.7}\'')
    parser.add_argument("--sahi", action="store_true", help="SAHI slicing for the teacher pass")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--pseudo-batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--steps", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--devices", default="auto", help="train/accelerator devices (auto/cpu/1/...)")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--model-args", default=None, help="student model_args as a json literal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1, help="maximum self-training rounds (1-10)")
    parser.add_argument(
        "--min-improvement", type=float, default=0.0,
        help="minimum AP gain (points) required to promote the student",
    )
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        raise SystemExit("--rounds must be between 1 and 10.")
    if args.min_improvement < 0:
        raise SystemExit("--min-improvement must be >= 0.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_spec = _load_data_spec(args.val_data)
    categories = _categories_of(val_spec, "val")

    teacher_path = Path(args.teacher)
    rounds: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_ap = float("-inf")
    stop_reason = "completed"
    for round_index in range(args.rounds):
        round_dir = (
            out_dir
            if args.rounds == 1
            else out_dir / "rounds" / f"r{round_index}"
        )
        record = _run_round(
            args,
            round_index=round_index,
            round_dir=round_dir,
            teacher_path=teacher_path,
            val_spec=val_spec,
            categories=categories,
        )
        current_ap = _ap(record["val_metrics"])
        rounds.append(record)
        if round_index > 0 and current_ap - best_ap < args.min_improvement:
            stop_reason = "no_improvement"
            print(
                f"[stop] round {round_index}: AP {current_ap:.4f} did not improve "
                f"on {best_ap:.4f} by {args.min_improvement}."
            )
            break
        if current_ap > best_ap:
            best_ap = current_ap
            best = record
        if round_index + 1 < args.rounds:
            teacher_path = Path(record["student"])
            print(f"[promote] next teacher: {teacher_path}")

    _write_summary(
        out_dir,
        rounds,
        stop_reason=stop_reason,
        min_improvement=args.min_improvement,
        best=best,
    )
    best_model = best["student"] if best is not None else None
    print(f"[report] {out_dir / 'summary.json'}")
    print(f"[model] 可评测模型: {best_model}")


if __name__ == "__main__":
    main()
