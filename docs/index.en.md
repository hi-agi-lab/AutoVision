# AutoVision

An enterprise-grade visual model training framework: **self-supervised
pretraining / distillation → finetuning → evaluation → export** in one chain.
The Python API and `autovision-cli` (30 commands) share the same strict Pydantic
validation surface, covering detection, oriented detection, instance / panoptic /
semantic segmentation, pose estimation, depth estimation and classification.

Three differentiating pillars:

- **Clean licensing**: Apache-2.0; repository-wide provenance gates and weight
  license gating (restricted weights are not downloaded by default; AGPL
  components are isolated behind extras). The model zoo lists the license and
  sha256 for every weight.
- **Verifiable consistency**: ONNX / TensorRT / OpenVINO exports run numerical
  parity gates by default and emit a portable postprocess specification; official
  weight loading and forward parity (including bit-equal checks); golden loss
  trajectory snapshots.
- **Protocolized benchmarks**: the [benchmark table](benchmarks.md) cites only
  published values, requires a reproduction script for every in-house number, and
  is append-only with an errata section.

## Quick start

Inference with released weights, no training:

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # parity gate + postprocess spec
```

List available models and export boundaries:

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

## Installation

Python ≥ 3.10. Source installation is the current supported path; after the
PyPI release, `pip install autovision-framework` will be available (model
packages are lazily loaded per extra; a missing extra yields an empty registry
instead of an import error):

```bash
pip install -e ".[transformers,timm,export]"    # common: HF backbones + timm + ONNX export
pip install -e ".[rfdetr,export]"               # pretrained RF-DETR inference
pip install -e ".[video]"                       # reference video/RTSP runtime (OpenCV)
pip install -e ".[dev,export,serving,video,transformers]"  # development/full tests
```

The planned PyPI distribution name is `autovision-framework`.

The GTR family (MIT) requires the `gtr` extra; see the repository README for the
full extras table. Pretrained weights are downloaded from HuggingFace; restricted
networks can configure a mirror. **Restricted-license weights are not downloaded
by default** (DINOv2 CC-BY-NC 4.0 / DINOv3 License require explicit acceptance).
See [weight downloads](download_weights.md) for download commands, caches and
mirrors.

## Capabilities

| Subsystem | Capabilities |
|---|---|
| Config / registry | Strict Pydantic validation, did-you-mean errors, registries; CLI and API share one schema |
| Model layer | 7 registered sources (torchvision / timm / super_gradients / ultralytics (AGPL-isolated) / dinov2 / dinov3 / rfdetr) plus in-tree `rtdetr/*`, `detr/*`, `dfine/*`, `picodet/*`, `rfdetr/*`, `gtr/*` |
| Data / transforms | 11 dataset families; mmap file manifests (content fingerprints / atomic publication); full detection augmentation set (mosaic / mixup / copyblend / box-aware eraser); VOC / YOLO-pose / YOLO-seg conversion + COCO merge |
| Pretraining methods | DistillationV3, SimCLR, DINO, DINOv2, DINOv3, DenseCL, DetCon B/S, in-training kNN probes |
| Runtime | DDP, mixed precision, atomic checkpoints + resume, NaN capture, JSONL / TB / MLflow / W&B, OTel (opt-in) |
| Task layer | 8 task surfaces; metrics support cross-rank aggregation (DDP validation); error analysis (FP attribution + confusion matrix + top-FP/FN gallery) + pseudo-labeling / self-training loop; HPO (random / grid / ASHA + reports); task-aware visualization (boxes / masks / rotated boxes / pose / semantic and panoptic segmentation / depth) |
| Deployment | ONNX (FP16 blacklist + numerical parity gate), TensorRT, int8 QDQ, three-backend benchmark parity, reference inference runtime (video / RTSP + three-stage pipeline + batching, same pre/post-processing; optional ByteTrack tracking, LineZone / PolygonZone counting, track smoothing; RLE masks by default), reference serving (FastAPI + TLS/multi-worker + rate limiting + drift + A/B + `/metrics`) |

Per-model licenses / sha256 / official metrics / one-line loading snippets are in
the [model zoo](model_zoo.md).

## Documentation map

| Page | Contents |
|---|---|
| [Model zoo](model_zoo.md) | Per model: license / sha256 / official metrics / loading code (generated) |
| [Benchmarks](benchmarks.md) | Accuracy and latency benchmarks (immutable protocol + errata) |
| [Export & deployment](export.md) | Dynamic H/W / int8 / TensorRT 11 / parity / GTR deployment notes |
| [Plugin protocol](plugins.md) | Entry-point plugins, API version, model_source integration and template |
| [Private registry](registry.md) | current / rollback / audit / deployment gates |
| [Compliance report](compliance.md) | Dependency licenses / model cards / data snapshot evidence pack |
| [Hardware partner packs](hardware_packs.md) | One pack per partner, conversion and target-side parity contract |
| [LTS policy](lts_policy.md) | LTS cadence, support windows, security backports |
| [Support policy](support_policy.md) | Community/enterprise support, P1–P3 SLA, support matrix |
| [User-held weights](user_held_weights.md) | Loading routes and license boundaries for out-of-tree DINOv3 detectors |
| [Known limits](known_limits.md) | Scope, adaptation gaps, lineage status, export boundaries |
| [Architecture](framework_design.md) | As-built layering contract / module map / quality gates |
| [Contributing](contributing.md) | Development environment / pre-submit gates / licensing and provenance rules |
| [Release process](release_process.md) | Version policy / release checklist / weight publishing policy |
| [PyPI publishing](pypi_publish.md) | Distribution name / TestPyPI / PyPI / trusted publishing / troubleshooting |
| [Model publishing](model_publish.md) | License audit / model cards / HF upload / registry / release gates |
| [Changelog](changelog.md) | Keep a Changelog-format external change log |

## License

Apache-2.0 (full text:
<https://www.apache.org/licenses/LICENSE-2.0>; repository `LICENSE`). Third-party
components and weight provenance are listed in repository `NOTICE` and
`THIRD_PARTY_LICENSES.md`.

```bash
autovision-cli --help
autovision-cli list-models    # models available in this environment (filtered by installed extras)
```

> Translation status: this landing page and `README.en.md` are translated.
> Deep reference pages without an English translation fall back to the Chinese
> original until a translated page is added.
