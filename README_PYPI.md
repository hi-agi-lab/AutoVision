# AutoVision

Enterprise-grade visual model training framework: self-supervised pretrain /
distill → finetune → evaluate → export in one chain. Covers detection, rotated
boxes, instance / panoptic / semantic segmentation, pose, depth, and
classification; the Python API and `autovision-cli` share the same strict
Pydantic validation.

> **Distribution name**: the PyPI package is `autovision-framework`, while the
> import package stays `autovision` and the CLI stays `autovision-cli`. This
> project is unrelated to the unrelated `autovision` 0.0.1 package on PyPI;
> make sure you install `autovision-framework`.

**Links**: [Source repository](https://github.com/hi-agi-lab/AutoVision) ·
[Documentation](https://hi-agi-lab.github.io/AutoVision/) ·
[Changelog](https://github.com/hi-agi-lab/AutoVision/blob/main/CHANGELOG.md)

## Installation

Python ≥ 3.10:

```bash
pip install autovision-framework

# Common extras: HF backbones + timm + ONNX export
pip install "autovision-framework[transformers,timm,export]"

# Pretrained RF-DETR direct inference
pip install "autovision-framework[rfdetr,export]"
```

Model packages load lazily per extra; a missing extra yields an empty registry
listing instead of an error. See the
[repository README](https://github.com/hi-agi-lab/AutoVision#installation)
for the full extras table.

## Quick start

No training — load released weights and predict:

```python
import autovision

model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # parity gate + postprocess spec
```

Object detection, train → validate → predict → export:

```python
import autovision

data = autovision.ensure_coco8()   # synthetic smoke dataset; swap in COCO/YOLO configs
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
autovision-cli predict model=runs/det1/exported_models/exported_best.pt images=dir/ out=preds/ threshold=0.5
```

## Common extras

| extra | capability |
|---|---|
| `transformers` | DINOv2 / DINOv3 backbones (HuggingFace weights) |
| `timm` | full timm backbone zoo (arbitrary input channels) |
| `export` | ONNX export + numerical parity gate |
| `openvino` | OpenVINO IR export and runtime |
| `video` | video / RTSP reference inference runtime |
| `serving` | FastAPI reference serving (`/predict`, `/metrics`) |
| `gtr` | GTR GLA backbone (Triton, CUDA required) |
| `download` | on-demand GTR upstream weight download (huggingface_hub) |
| `dev` | pytest / ruff (same as CI) |

## License

Apache-2.0 (AutoVision original code; full text:
<https://www.apache.org/licenses/LICENSE-2.0>). Third-party component and
weight provenance ship in `NOTICE` and `THIRD_PARTY_LICENSES.md`.
Restricted-license weights (DINOv2 CC-BY-NC 4.0 / DINOv3 License) are not
downloaded by default; user-held weights may not be redistributed.

Full documentation (model zoo, benchmarks, export details, known limits) is
available on the
[documentation site](https://hi-agi-lab.github.io/AutoVision/) and in the
`docs/` directory of the source repository.
