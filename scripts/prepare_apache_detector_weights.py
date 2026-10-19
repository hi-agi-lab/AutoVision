#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Prepare the Apache-2.0 detector weight store.

Builds a self-contained, license-clean detector store from:

* **D-FINE** official COCO checkpoints (``d-fine/dfine_{s,m,l}_coco``) —
  upstream complete detectors; AutoVision consumes their decoder/encoder
  as a warm-start for the ``detr/s`` / ``detr/n`` spectrum tiers.
* **RF-DETR** official COCO checkpoints (``Roboflow/rf-detr-*``, served by
  the pinned ModelScope mirrors when configured) — complete detectors,
  exported in AutoVision's ``weights.pt`` task-model format.
* **AVWarmStart v1** files for ``detr/n`` (HGNetV2-B0) and ``detr/s``
  (HGNetV2-B1): Apache-2.0 timm ImageNet backbones + the D-FINE-S
  decoder/encoder transfer. These are fine-tuning starting points, not
  complete COCO detectors.

Everything written here is Apache-2.0. The store is intentionally outside
the git tree and carries ``LICENSE`` + ``NOTICE`` + per-asset ``ASSET.json``
so it can be moved or published without losing attribution.

Usage::

    # uses the local cache; ModelScope is the default RF-DETR source
    python scripts/prepare_apache_detector_weights.py

    # explicit store / mirror
    python scripts/prepare_apache_detector_weights.py \
        --store ~/Research/Model/AutoVision/apache_detection \
        --mirror https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Must be set before autovision imports the weight-verification layer:
# the ModelScope cache is the pinned, byte-identical mirror for RF-DETR.
os.environ.setdefault("AUTOVISION_WEIGHT_SOURCE", "modelscope")

import torch  # noqa: E402

from autovision.encoders.weight_hashes import WEIGHT_MANIFEST  # noqa: E402
from autovision.registry.apache_assets import register_apache_assets  # noqa: E402
from autovision.registry.assets import LicenseRef  # noqa: E402
from autovision.registry.downloadable import (  # noqa: E402
    ASSET_MANIFEST,
    resolve_downloadable,
)
from autovision.tasks.detection.detr.dfine_official_warm_start import (  # noqa: E402
    build_dfine_official_warm_start,
)
from autovision.tasks.detection.detr.task_model import (  # noqa: E402
    DETRObjectDetection,
    load_warm_start_into_network,
)
from autovision.tasks.detection.rfdetr.task_model import (  # noqa: E402
    RFDETRObjectDetection,
)

register_apache_assets()

APACHE = LicenseRef(
    id="Apache-2.0",
    origin="AutoVision Apache detector store (D-FINE / RF-DETR upstreams)",
)

DFINE_VARIANTS = ("s", "m", "l")
RF_DETR_VARIANTS = ("nano", "small", "medium", "base", "large")
RF_DETR_NATIVE = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "base": 560,
    "large": 704,
}
RF_DETR_METRICS = {
    "nano": {"coco_map": 48.4, "coco_map50": 67.6, "params_m": 30.5, "resolution": 384},
    "small": {"coco_map": 53.0, "coco_map50": 72.1, "params_m": 32.1, "resolution": 512},
    "medium": {"coco_map": 54.7, "coco_map50": 73.6, "params_m": 33.7, "resolution": 576},
    "base": {"note": "upstream deprecated; no official COCO reading"},
    "large": {"coco_map": 56.5, "coco_map50": 75.1, "params_m": 33.9, "resolution": 704},
}
DFINE_METRICS = {
    "s": {"coco_map": 48.5, "params_m": 10.0, "resolution": 640},
    "m": {"coco_map": 52.3, "params_m": 19.0, "resolution": 640},
    "l": {"coco_map": 54.0, "params_m": 31.0, "resolution": 640},
}

STORE_FILES = ("LICENSE", "NOTICE", "README.md", "manifest.json")


def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(src: Path, dst: Path) -> None:
    """Hard-link (preferred) or copy a verified file into the store."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _dfine_source(variant: str) -> Path:
    name = f"d-fine/dfine_{variant}_coco"
    candidates = (
        Path.home() / ".cache/autovision/downloadable_models" / f"{name}.pt",
        Path.home() / f".cache/autovision/d-fine_dfine_{variant}_coco.pth",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return resolve_downloadable(name)


def prepare_dfine(store: Path, assets: list[dict]) -> None:
    for variant in DFINE_VARIANTS:
        name = f"d-fine/dfine_{variant}_coco"
        entry = ASSET_MANIFEST[name]
        src = _dfine_source(variant)
        digest = sha256_file(src)
        if digest != entry.sha256:
            raise RuntimeError(
                f"{name}: sha256 mismatch (expected {entry.sha256}, got {digest})"
            )
        asset_dir = store / "dfine" / f"dfine_{variant}_coco"
        weights = asset_dir / "weights.pth"
        materialize(src, weights)
        asset = {
            "name": f"dfine_{variant}_coco",
            "family": "dfine",
            "role": "official_checkpoint",
            "variant": variant,
            "weights": str(weights.relative_to(store)),
            "sha256": digest,
            "bytes": weights.stat().st_size,
            "license": "Apache-2.0",
            "source": entry.url,
            "upstream": "Peterande/D-FINE @ 956d170",
            "native_resolution": 640,
            "num_classes": 80,
            "metrics": DFINE_METRICS[variant],
            "load": {
                "upstream": (
                    "D-FINE configs/dfine/dfine_hgnetv2_{v}_coco.yml".format(
                        v=variant
                    )
                ),
                "autovision": (
                    "decoder/encoder warm-start only: use the prepared "
                    "'detr_s_hgnetv2b1_dfine_s.pt' / "
                    "'detr_n_hgnetv2b0_dfine_s.pt' warm-starts for "
                    "autovision.train_object_detection(model='detr/s' or 'detr/n')"
                ),
            },
        }
        _write_json(asset_dir / "ASSET.json", asset)
        assets.append(asset)
        print(f"D-FINE {variant}: {weights} ({weights.stat().st_size:,} bytes)")


def _rfdetr_cache_dir(modelscope_id: str) -> Path:
    return Path.home() / ".cache/autovision/modelscope" / modelscope_id.replace("/", "--")


def prepare_rfdetr(store: Path, assets: list[dict]) -> None:
    for variant in RF_DETR_VARIANTS:
        name = f"rf-detr-{variant}"
        repo = f"Roboflow/{name}"
        entry = WEIGHT_MANIFEST[repo]
        size = RF_DETR_NATIVE[variant]
        # RF-DETR's native COCO head is 91-way (0 = N/A, 1..90 are the
        # COCO category ids); using 80 here would retile/truncate the head
        # and shift every label (e.g. person -> bicycle).
        task_model = RFDETRObjectDetection(
            num_classes=91,
            model=name,
            image_size=(size, size),
            load_weights=True,
        ).eval()
        asset_dir = store / "rfdetr" / name
        asset_dir.mkdir(parents=True, exist_ok=True)
        weights = asset_dir / "weights.pt"
        torch.save(
            {
                "model_class_path": (
                    "autovision.tasks.detection.rfdetr.task_model."
                    "RFDETRObjectDetection"
                ),
                "model_init_args": {
                    "num_classes": 91,
                    "model": name,
                    "image_size": [size, size],
                    "num_queries": 300,
                    "num_select": 300,
                    "box_regression_iou": "giou",
                    "siou_theta": 4.0,
                },
                "train_model": task_model.state_dict(),
            },
            weights,
        )
        cache = _rfdetr_cache_dir(entry.modelscope_id or "")
        raw_sha = None
        if cache.is_dir():
            raw = cache / "model.safetensors"
            config = cache / "config.json"
            if raw.is_file():
                materialize(raw, asset_dir / "model.safetensors")
                raw_sha = sha256_file(asset_dir / "model.safetensors")
                expected = entry.sha256.get("model.safetensors")
                if expected and raw_sha != expected:
                    raise RuntimeError(
                        f"{repo}: ModelScope sha256 mismatch "
                        f"(expected {expected}, got {raw_sha})"
                    )
            if config.is_file():
                materialize(config, asset_dir / "config.json")
        digest = sha256_file(weights)
        asset = {
            "name": name,
            "family": "rfdetr",
            "role": "complete_detector",
            "variant": variant,
            "weights": str(weights.relative_to(store)),
            "sha256": digest,
            "bytes": weights.stat().st_size,
            "upstream_model_safetensors": (
                str((asset_dir / "model.safetensors").relative_to(store))
                if raw_sha
                else None
            ),
            "upstream_model_safetensors_sha256": raw_sha,
            "license": "Apache-2.0",
            "source": f"https://huggingface.co/{repo}",
            "modelscope": entry.modelscope_id,
            "revision": entry.revision,
            "native_resolution": size,
            "num_classes": 91,
            "label_space": (
                "COCO 91-way: 0=N/A, 1..90 are COCO category ids; "
                "use the checkpoint config.id2label"
            ),
            "params_m": sum(p.numel() for p in task_model.parameters()) / 1e6,
            "metrics": RF_DETR_METRICS[variant],
            "load": {
                "autovision": (
                    "from autovision import load_task_model; "
                    f"model = load_task_model('{weights}')"
                ),
                "hf": (
                    "RfDetrForObjectDetection.from_pretrained('<asset_dir>')"
                ),
            },
        }
        _write_json(asset_dir / "ASSET.json", asset)
        assets.append(asset)
        print(f"RF-DETR {variant}: {weights} ({weights.stat().st_size:,} bytes)")
        del task_model


def prepare_warmstarts(store: Path, assets: list[dict]) -> None:
    dfine_s = _dfine_source("s")
    for model_name, label in (
        ("hgnetv2-b0", "n"),
        ("hgnetv2-b1", "s"),
    ):
        out = store / "warmstart" / f"detr_{label}_{model_name}_dfine_s.pt"
        report = build_dfine_official_warm_start(
            model_name,
            out=out,
            checkpoint=dfine_s,
            backbone_pretrained=True,
            license_id="Apache-2.0",
            origin=(
                "D-FINE dfine_s_coco (Peterande/D-FINE @ 956d170) + "
                f"timm {model_name}.ssld_stage2_ft_in1k (Apache-2.0)"
            ),
        )
        network = DETRObjectDetection(
            num_classes=80,
            model_name=model_name,
            image_size=(640, 640),
            decoder="dfine_official",
            backbone_pretrained=True,
            load_weights=True,
        ).eval()
        load_report = load_warm_start_into_network(network, out)
        with torch.no_grad():
            logits, boxes = network(torch.randn(1, 3, 640, 640))
        if tuple(logits.shape) != (1, 300, 80) or tuple(boxes.shape) != (1, 300, 4):
            raise RuntimeError(f"{out}: unexpected forward shapes")
        digest = sha256_file(out)
        assembly = dict(report)
        assembly["source_checkpoint"] = "d-fine/dfine_s_coco (Apache-2.0)"
        asset = {
            "name": f"detr_{label}_{model_name}_dfine_s",
            "family": "detr",
            "role": "warmstart",
            "variant": label,
            "model_key": f"detr/{label}",
            "weights": str(out.relative_to(store)),
            "sha256": digest,
            "bytes": out.stat().st_size,
            "license": "Apache-2.0",
            "source": "D-FINE dfine_s_coco + timm HGNetV2 ImageNet",
            "assembly": assembly,
            "load_report": load_report,
            "spec": {
                "family": "detr",
                "backbone": model_name,
                "classes": 80,
                "decoder_coverage": report["decoder_coverage"],
                "dec_num_points": [3, 6, 3],
                "dec_lineage": "dfine",
            },
            "load": {
                "autovision": (
                    "autovision.train_object_detection(..., model='detr/"
                    f"{label}', model_args={{'warm_start': '{out}', "
                    "'backbone_pretrained': True}})"
                )
            },
        }
        _write_json(out.with_suffix(".ASSET.json"), asset)
        assets.append(asset)
        print(
            f"warm-start {label}: {out} ({out.stat().st_size:,} bytes, "
            f"decoder coverage {report['decoder_coverage']})"
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_docs(store: Path, assets: list[dict]) -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    (store / "LICENSE").write_text(license_text, encoding="utf-8")
    notice = f"""AutoVision Apache detector weight store
Copyright 2026 AutoVision contributors

All weights in this store are Apache-2.0. Retain this NOTICE and LICENSE
when redistributing.

D-FINE
  Copyright (c) 2024 Peterande
  Source: https://github.com/Peterande/D-FINE (956d170)
  Weights: https://huggingface.co/Peterande/D-FINE
  License: Apache-2.0
  Role: official COCO checkpoints (s/m/l); AutoVision also uses the
  D-FINE-S decoder/encoder as the detr/n and detr/s warm-start.

RF-DETR
  Copyright (c) Roboflow
  Source: https://github.com/roboflow/rf-detr
  Weights: https://huggingface.co/Roboflow/rf-detr-{{nano,small,medium,base,large}}
  License: Apache-2.0
  Role: complete COCO detectors, exported as AutoVision task models.

HGNetV2 (inside the warm-start files)
  Copyright (c) PaddlePaddle authors
  Source: timm/hgnetv2_b0|b1.ssld_stage2_ft_in1k
  License: Apache-2.0

Generated: {date.today().isoformat()}
"""
    (store / "NOTICE").write_text(notice, encoding="utf-8")
    readme = """# Apache-2.0 detector weights

This store contains only Apache-2.0 detector weights. It is deliberately
separate from the user-held DINOv3 store and from the AutoVision git tree.

## Contents

| Family | Files | Role |
|---|---|---|
| D-FINE | `dfine/dfine_{s,m,l}_coco/weights.pth` | official COCO checkpoints; warm-start source for `detr/n` and `detr/s` |
| RF-DETR | `rfdetr/rf-detr-{nano,small,medium,base,large}/weights.pt` | complete COCO detectors, loadable with `autovision.load_task_model` |
| DETR warm-start | `warmstart/detr_{n,s}_hgnetv2b{0,1}_dfine_s.pt` | AVWarmStart v1 for `detr/n` / `detr/s` fine-tuning |

Every directory has an `ASSET.json` with the license, upstream source,
sha256, size, and load recipe. `manifest.json` aggregates them.

## Load

Complete RF-DETR detectors:

```python
from autovision import load_task_model
model = load_task_model(
    "__STORE__/"
    "rfdetr/rf-detr-nano/weights.pt",
    device="cuda",
)
```

RF-DETR uses its native **91-way COCO label space**: `0 = N/A`,
`1 = person`, `2 = bicycle`, ... `90 = toothbrush`. Read names from
`model._model.config.id2label`; do **not** map with the 80-class
contiguous list. The exported `weights.pt` keeps the full 91-way head.

DETR spectrum warm-start:

```python
import autovision
autovision.train_object_detection(
    out="runs/detr_s", data="coco.yaml", model="detr/s",
    model_args={
        "warm_start": "__STORE__/"
        "warmstart/detr_s_hgnetv2b1_dfine_s.pt",
        # The warm-start carries decoder/encoder tensors only; this loads
        # the Apache-2.0 timm HGNetV2 ImageNet backbone.
        "backbone_pretrained": True,
    },
)
```

The warm-start files transfer the D-FINE-S decoder fully (coverage 1.0)
plus 46 encoder tensors; the backbone is the Apache-2.0 timm HGNetV2
ImageNet initialization. They are **fine-tuning starting points**, not
complete COCO detectors. Run the standard COCO fine-tune recipe before
measuring AP.

## D-FINE direct use

The `weights.pth` files are the upstream D-FINE checkpoints (not an
AutoVision export). Use the D-FINE repository configs for direct inference;
inside AutoVision use the prepared warm-start files.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. No DINOv2, DINOv3, AGPL, or
NOASSERTION weights are present.
"""
    readme = readme.replace(
        "__STORE__/",
        f"{store.as_posix().rstrip('/')}/",
    )
    (store / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "store": "apache_detection",
        "generated_at": date.today().isoformat(),
        "policy": {
            "license": "Apache-2.0",
            "redistributable": True,
            "excluded": ["DINOv2 (CC-BY-NC)", "DINOv3 (custom)", "AGPL", "NOASSERTION"],
        },
        "assets": sorted(assets, key=lambda item: item["name"]),
    }
    _write_json(store / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store",
        type=Path,
        default=Path.home() / "Research/Model/AutoVision/apache_detection",
        help="target store directory",
    )
    parser.add_argument(
        "--mirror",
        default=os.environ.get("AUTOVISION_MIRROR_URL", ""),
        help="optional HF mirror (e.g. https://hf-mirror.com) for timm/upstream",
    )
    parser.add_argument(
        "--skip-warmstart",
        action="store_true",
        help="skip the DETR n/s warm-start assembly",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", args.mirror)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    store = args.store.expanduser()
    store.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    prepare_dfine(store, assets)
    prepare_rfdetr(store, assets)
    if not args.skip_warmstart:
        prepare_warmstarts(store, assets)
    write_docs(store, assets)
    print(f"\nstore: {store}")
    print(f"assets: {len(assets)}")
    print(f"manifest: {store / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
