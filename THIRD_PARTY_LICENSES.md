# Third-party licenses

AutoVision-authored code is licensed under Apache-2.0 (see `LICENSE`). This
file contains the full texts of non-Apache terms for components distributed
with the source tree, plus attribution notes for Apache-2.0 components whose
license text is already covered by `LICENSE`.

## Component map

| Component / file | Upstream | License | Mode |
|---|---|---|---|
| `tasks/shared/flat_cosine.py` | Intellindust DEIM | Apache-2.0 | re-implemented |
| D-FINE decoder / criterion / utils | Peterande/D-FINE | Apache-2.0 | ported |
| RT-DETR hybrid encoder / decoder / criterion / matcher | lyuwenyu/RT-DETR | Apache-2.0 | ported |
| `_msda_src/` deformable attention | Detectron2 lineage | Apache-2.0 | vendored |
| GTR backbone / heads / CUDA kernels | Intellindust-AI-Lab/GTR | MIT | ported / vendored |
| `matcher.py` optional LSAP solver | ivan-chai/torch-linear-assignment | Apache-2.0 | optional dependency |
| `tasks/detection/picodet/` | PaddleDetection PP-PicoDet | Apache-2.0 | architecture reference; AutoVision implementation |
| `tracking/byte_track.py` | ByteTrack (ECCV 2022) | MIT (upstream) | independent implementation |
| `tasks/shared/sahi.py`, `tiling_utils.py` | SAHI (slicing inference) | MIT (upstream) | independent implementation |
| `visualize/effects.py`, `visualize/skeletons.py`, `geometry/boxes.py`, `geometry/masks.py`, `inference/video.py` | Roboflow supervision 0.30 | MIT | API/algorithm reference; AutoVision implementation |
| `encoders/rfdetr/`, `tasks/detection/rfdetr/` | Roboflow RF-DETR / HF transformers | Apache-2.0 | wrapper; weights downloaded |
| `examples/tasks/assets/airport_aerial.jpg` | Wikimedia Commons / Mertbiol | CC0-1.0 | redistributed in the sdist |

Apache-2.0 components are distributed under the repository's `LICENSE`;
their copyright notices are retained in source headers and/or `NOTICE`.
The full Apache-2.0 text is in `LICENSE`.

DEIM attribution: Copyright (C) INTELLINDUST INFORMATION TECHNOLOGY
(SHENZHEN) CO., LTD. and affiliates; DEIM is licensed under the Apache
License 2.0 (see `LICENSE`).

## GTR — MIT License

Copyright (c) 2026 The GTR Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

GTR model weights, when redistributed, carry the same MIT notice.

## supervision — MIT License

Copyright (c) 2022 Roboflow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The AutoVision modules listed in the component map above are independent
implementations written against supervision's public API and algorithms; the
MIT notice is retained here because the API/algorithm reference is derived
from the upstream project. No supervision package is required at runtime.
