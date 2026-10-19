[简体中文](README.md) | **English**

# AutoVision

An enterprise-grade visual model training framework: self-supervised
pretraining / distillation → finetuning → evaluation → export and deployment.
The Python API and `autovision-cli` (30 commands) share one strict Pydantic
validation surface across detection, oriented detection, instance / panoptic /
semantic segmentation, pose estimation, depth estimation and classification.

Current version: **0.3.0 (Beta)**. It targets teams that need self-hosted
training, evaluation, export and private deployment; it is not a hosted API and
does not replace a production gateway. Core code is Apache-2.0; restricted-license
weights are not downloaded by default.

> This is the English entry point. The Chinese [README](README.md) is the
> canonical long-form document. Some deep documentation pages currently fall
> back to the Chinese original until translated.

- **Clean licensing**: Apache-2.0 core; repository-wide provenance gates; weight
  license gates (DINOv2 / DINOv3 restricted weights are opt-in, AGPL components
  are isolated behind extras). The [model zoo](docs/model_zoo.md) lists license
  and sha256 per asset.
- **Verifiable consistency**: export-time numerical parity gates for ONNX /
  TensorRT / OpenVINO; official-weight loading and forward parity (including
  bit-equal checks); golden loss-trajectory snapshots.
- **Protocolized benchmarks**: the [benchmark table](docs/benchmarks.md) cites
  only published values, requires reproduction scripts for in-house numbers, and
  is append-only with an errata section.

Documentation: [Task examples](docs/task_examples.en.md) · [Model zoo](docs/model_zoo.md) ·
[Weight downloads](docs/download_weights.md) · [Benchmarks](docs/benchmarks.md) ·
[Architecture](docs/framework_design.md) · [Export & deployment](docs/export.md) ·
[Plugin protocol](docs/plugins.md) · [Private registry](docs/registry.md) ·
[Compliance](docs/compliance.md) · [Hardware packs](docs/hardware_packs.md) ·
[User-held weights](docs/user_held_weights.en.md) · [LTS policy](docs/lts_policy.md) ·
[Support policy](docs/support_policy.md) · [Known limits](docs/known_limits.en.md) ·
[Contributing](CONTRIBUTING.md)

## Installation

Python ≥ 3.10. Source installation is the current supported path; after the PyPI
release, `pip install autovision-framework` will be available (the distribution
name is `autovision-framework`, the import name remains `autovision`):

```bash
# Common: HuggingFace backbones + timm + ONNX export
pip install -e ".[transformers,timm,export]"

# Pretrained RF-DETR inference
pip install -e ".[rfdetr,export]"

# Reference video / RTSP runtime (OpenCV)
pip install -e ".[video]"

# Development / full test suite
pip install -e ".[dev,export,serving,video,transformers]"
```

Model families are loaded lazily per extra; a missing extra produces an empty
registry instead of an import error. GTR (MIT) requires the `gtr` extra. See
[docs/download_weights.md](docs/download_weights.md) for weight download, cache
and mirror configuration. Restricted licenses are opt-in: DINOv2 CC-BY-NC 4.0
and DINOv3 weights are not downloaded by default.

## Quick start

Inference with released weights, no training:

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # parity gate + postprocess spec
```

List models and inspect export boundaries:

```bash
autovision-cli list-models --task detection --pretrained
autovision-cli model-info rfdetr/rf-detr-small
```

Object detection: train → validate → predict → export:

```python
import autovision

data = autovision.ensure_coco8()   # synthetic smoke dataset; use COCO/YOLO YAML or "coco8"
model = autovision.detection("rtdetr/resnet18").train(
    out="runs/det1", data=data, steps=2000, batch_size="auto",
)
model.val(data=data)          # {'map': ..., 'map_50': ...}
model.predict("image.jpg")    # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")    # dispatches by .onnx / .engine / .pt suffix
```

Equivalent CLI (same Pydantic validation):

```bash
autovision-cli train-object-detection out=runs/det1 data=data.yaml steps=2000
autovision-cli val     model=runs/det1/exported_models/exported_best.pt data=coco8
autovision-cli export  model=runs/det1/exported_models/exported_best.pt out=model.onnx precision=fp16
autovision-cli predict model=runs/det1/exported_models/exported_best.pt source=clip.mp4 out=preds/ batch_size=4
```

## Task entry points

| Task | Training API / CLI | Default recipe | Data form | Notes |
|---|---|---|---|---|
| Detection | `train_object_detection` / `train-object-detection` | `rtdetr/resnet18` | COCO / YOLO | Full task surface |
| Oriented detection | `train_oriented_object_detection` / `train-oriented-object-detection` | `rtdetr/obb-resnet18` | DOTA / YOLO-OBB | Rotated boxes |
| Instance segmentation | `train_instance_segmentation` / `train-instance-segmentation` | `dinov2/eomt` | COCO | EOMT / GTR |
| Panoptic segmentation | `train_panoptic_segmentation` / `train-panoptic-segmentation` | `dinov2/eomt` | COCO panoptic | EOMT |
| Semantic segmentation | `train_semantic_segmentation` / `train-semantic-segmentation` | `dinov2/eomt` | Cityscapes / VOC | EOMT / GTR / linear head |
| Pose estimation | `train_pose_estimation` / `train-pose-estimation` | `gtr/pose-s` | COCO keypoints | GTR CUDA stack |
| Depth estimation | `train_depth_estimation` / `train-depth-estimation` | `gtr/s` | images + depth (`.npy` meters or 16-bit PNG) | GTR CUDA stack |
| Classification | `train_image_classification` / `train-image-classification` | `torchvision/resnet18` | ImageFolder `train/`, `val/` | — |

Each task has an inference and a training example under `examples/tasks/`;
`--dry-run` prints the data contract without training. See
[docs/task_examples.md](docs/task_examples.md).

## Capabilities and subsystems

- **Config and registry**: strict Pydantic validation with did-you-mean errors;
  CLI and Python API share the same schema; model/asset/plugin registries.
- **Model layer**: torchvision, timm, super-gradients, Ultralytics (AGPL-isolated),
  DINOv2, DINOv3, RF-DETR, plus in-tree RT-DETR, DETR/GTR, D-FINE, PicoDet,
  RF-DETR and GTR families.
- **Data and transforms**: 11 dataset families; mmap manifests with content
  fingerprints and atomic publication; mosaic / mixup / copyblend / box-aware
  eraser; VOC, YOLO-pose, YOLO-seg conversion and COCO merging; data quality and
  hard-example loops.
- **Pretraining / distillation**: DINO, DINOv2, DINOv3, SimCLR, DenseCL, DetCon,
  DistillationV3, in-training kNN probes.
- **Training runtime**: DDP, mixed precision, atomic checkpoints + resume,
  gradient accumulation, gradient checkpointing, FSDP, early stopping and
  callbacks, deterministic mode, run manifests, NaN capture, TensorBoard /
  MLflow / W&B / JSONL, OTel (opt-in).
- **Evaluation and analysis**: mAP, task metrics with DDP aggregation, error
  analysis with FP attribution and confusion matrices, hard-example manifests,
  HPO (random / grid / ASHA), pseudo-labeling and self-training.
- **Inference and serving**: `InferenceResult` at the API boundary; task-aware
  annotators; video/RTSP runtime with a reader/infer/writer pipeline; ByteTrack
  tracking; LineZone / PolygonZone analytics; reference FastAPI serving with
  TLS, workers, rate limiting, drift and A/B/canary support.
- **Export and deployment**: ONNX (fp32/fp16/int8), TensorRT, OpenVINO;
  `postprocess.json` sidecars; end-to-end box-level parity; C++ ONNX Runtime +
  OpenCV reference runtime under `examples/cpp/`.

## Models and metrics

The [model zoo](docs/model_zoo.md) is generated from the registries and lists
every asset with license, sha256, official metric and a loading snippet. It
covers RF-DETR, GTR (detection / segmentation / pose / depth / OBB / semantic),
in-tree DETR / RT-DETR / D-FINE / PicoDet, and the user-held DINOv3 detectors
(out-of-tree, non-redistributable).

## Export and deployment

```python
model.export("model.onnx")     # ONNX + parity gate + postprocess spec
model.export("model.engine")   # TensorRT (requires CUDA + TensorRT)
model.export("model.xml")      # OpenVINO IR (requires openvino extra)
```

Exports run numerical parity gates by default; dynamic spatial inputs require
the dynamo exporter and are verified at unseen sizes. See
[docs/export.md](docs/export.md) for backend details, int8, TensorRT 11 and GTR
deployment notes.

## Inference runtime and serving

```python
from autovision.inference import run_stream
run_stream("clip.mp4", "preds/", model="best.pt", track=True)
```

The stream runtime includes a prefetched decode / infer / render / write
pipeline, JSONL output, optional tracking and zone analytics. The reference
serving layer provides `/predict`, `/healthz`, `/readyz`, `/metrics`, `/drift`,
optional API-key auth, request limits, TLS, workers, rate limiting and
A/B/canary routing. It is a reference implementation, not a complete production
gateway.

## Development

```bash
pip install -e ".[dev,export,serving,video,transformers]"
pytest tests
ruff check src tests scripts
bandit -r src/autovision -ll -ii -q
python scripts/check_provenance.py && python scripts/check_layering.py
python scripts/check_docs_links.py && mkdocs build --strict
```

The full test suite is CPU-compatible; gated tests (TensorRT, HF downloads,
private user-held stores, multi-process DDP) are enabled by their documented
environment variables. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/release_process.md](docs/release_process.md).

## Known limits

Highlights: no DeepSpeed/ZeRO/elastic training; multi-node NCCL and fault
injection are not covered by public CI; TensorRT/OpenVINO heavy tests are
gated; GTR export is blocked by the GLA kernel lowering; STA-era ViT user-held
checkpoints are rejected because the DEIMv2 spatial prior was removed. See
[docs/known_limits.md](docs/known_limits.md) for the full list.

## Industry solutions

Vertical scenario solutions built on AutoVision (solution engines consume the
framework's stable seams via `pip install autovision-framework`; see the
[industry solutions page](https://hi-agi-lab.github.io/AutoVision/en/ecosystem/)):

| Repository | Scenario |
| --- | --- |
| [ai-traffic](https://github.com/hi-agi-lab/ai-traffic) | Road incidents: FOD + composite accident scoring + congestion |
| [ai-wildfire](https://github.com/hi-agi-lab/ai-wildfire) | Wildfire + flood/typhoon video analytics |
| [ai-illop](https://github.com/hi-agi-lab/ai-illop) | SOP violation monitoring (checklist-driven acceptance) |
| [ai-conflict](https://github.com/hi-agi-lab/ai-conflict) | Public-order conflict analytics (long-tail T3) |
| [ai-perimeter](https://github.com/hi-agi-lab/ai-perimeter) | Perimeter intrusion video analytics |
| [ai-security](https://github.com/hi-agi-lab/ai-security) | General security video analytics |
| [ai-life-line](https://github.com/hi-agi-lab/ai-life-line) | Lifeline infrastructure video analytics |
| [ai-urban](https://github.com/hi-agi-lab/ai-urban) | Urban governance video analytics |

## License and provenance

Core code is Apache-2.0; the full text is in repository `LICENSE` and at
<https://www.apache.org/licenses/LICENSE-2.0>. Third-party components and weight
provenance are listed in `NOTICE`, `THIRD_PARTY_LICENSES.md` and
[docs/provenance/LEDGER.md](docs/provenance/LEDGER.md).

Restricted weights (DINOv2 CC-BY-NC 4.0 / DINOv3 License) are opt-in and not
redistributed. The user-held DINOv3 detector store is out-of-tree,
non-redistributable and not part of `ASSET_MANIFEST`.

## Acknowledgements

AutoVision benefits from PyTorch, torchvision, HuggingFace Transformers, ONNX
Runtime, NVIDIA TensorRT, OpenVINO, FastAPI, MLflow, Weights & Biases,
TensorBoard, RT-DETR, D-FINE, DEIM, GTR, ByteTrack, SAHI, Roboflow RF-DETR,
PicoDet and supervision. See [NOTICE](NOTICE) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for legal attribution.

> Translation status: this English README covers the main entry points. The
> Chinese README is the default/canonical long-form document; deep docs pages
> without an English translation fall back to Chinese until translated.
