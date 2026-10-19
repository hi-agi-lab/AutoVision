# C++ ONNX Runtime reference runtime

Status: **reference example**, not an SDK and not an ABI-stable API. It shows
the smallest useful device-side path for an AutoVision export:

1. read the image and resize/normalize exactly like the training preprocess;
2. run the exported ONNX graph with ONNX Runtime;
3. decode `(logits, boxes)` with the same contract as
   `<artifact>.postprocess.json` (DETR top-k or PicoDet class-offset NMS).

The C++ decode is mirrored from `autovision.export.decode_detections_numpy`;
`postprocess_test.cpp` is a host-only test (no ONNX Runtime/OpenCV needed).

## Files

```text
examples/cpp/
  CMakeLists.txt
  README.md
  src/postprocess.hpp       # header-only decode + class-offset NMS
  src/postprocess_test.cpp  # host test (bare g++)
  src/main.cpp              # ONNX Runtime + OpenCV reference runtime
```

## Host test (works with only a C++ compiler)

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexamples/cpp/src \
    examples/cpp/src/postprocess_test.cpp -o /tmp/postprocess_test
/tmp/postprocess_test
# postprocess ok
```

The host test covers DETR resize, stable top-k ties, multi-image original
sizes, PicoDet class-aware and class-agnostic NMS, `max_detections`, and the
strict score threshold. No ONNX Runtime or OpenCV headers are required.

## Build the ONNX Runtime example

Requirements: OpenCV development files, an extracted ONNX Runtime release
(C++ headers + `libonnxruntime`), CMake ≥ 3.16.

```bash
export ONNXRUNTIME_ROOT=/opt/onnxruntime
cmake -S examples/cpp -B build/cpp -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build build/cpp
(cd build/cpp && ctest --output-on-failure)
```

If ONNX Runtime or OpenCV is missing, CMake still builds the host test and
prints that `autovision_infer` was skipped.

```bash
./build/cpp/autovision_infer \
    --model model.onnx --image demo.jpg \
    --num-classes 80 --top-k 300 --max-detections 300 --threshold 0.5 \
    --input-width 640 --input-height 640 \
    --family detr
```

For PicoDet exports:

```bash
./build/cpp/autovision_infer --model picodet.onnx --image demo.jpg \
    --num-classes 80 --family picodet --nms-iou 0.6
```

`--max-detections 0` (the default) uses `--top-k` as the post-NMS cap. Use
`--help` for the full flag list. `--family` accepts only `detr` or `picodet`;
unknown values are rejected instead of silently falling back.

Output is JSON-lines with `label`, `score`, `box` in original-frame pixels.

## Mapping from `postprocess.json`

| Spec field | CLI flag |
|---|---|
| `num_classes` | `--num-classes` |
| `top_k` | `--top-k` |
| `score_threshold` | `--threshold` |
| `family` | `--family detr\|picodet` |
| `nms.iou_threshold` | `--nms-iou` |
| `nms.max_detections` | `--max-detections` (0 = use `top_k`) |
| `nms.class_aware` | `--class-agnostic` (inverts) |
| `input_size` | graph input dims; `--input-width/--input-height` fallback |

The example assumes the graph outputs are `(logits, boxes)` in that order and
uses the ImageNet mean/std (the framework default). Letterbox / custom
normalization requires changing `preprocess()`.

## Decode parity

`postprocess.hpp` mirrors `autovision.export.decode_detections_numpy`:

- top-k uses a stable descending score order (ties keep the lower flat
  `(query, class)` index);
- DETR applies sigmoid, top-k, threshold, then one rescale to the original
  frame;
- PicoDet applies class-offset NMS with `offset = max(original_w,
  original_h) + 1` when class-aware and `offset = 0` when class-agnostic.

The runtime validates that the model exposes at least two outputs, that the
outputs are float32, and that the shapes are `(B,Q,C)` and `(B,Q,4)`; invalid
models fail with a non-zero exit instead of reading unchecked memory.

## Boundaries

- No dynamic spatial input support beyond what the graph advertises.
- No tracking, batching across requests, or vendor backends (TensorRT,
  OpenVINO, RKNN, …); this is the ONNX Runtime path only.
- Not a performance reference; keep preprocessing and NMS on the device for a
  real product.
