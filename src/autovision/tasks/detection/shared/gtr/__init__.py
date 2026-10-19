# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR integration package (roadmap design doc 31 G2).

MIT re-ports of the Intellindust GTR backbone line + multi-scale
encoder. Importing this package is safe without the ``gtr`` extra; the
fla dependency is required only to *construct* the attention modules
(:func:`gtr_available` reports readiness).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the compiled GLA CUDA extension (csrc/, built in-place via
# ``csrc/build.sh`` or setup.py) so vit_adapter's plain
# ``import gla_torch_ext`` resolves. Must run BEFORE the submodule
# imports below — vit_adapter probes the extension at module load.
_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
if _CSRC_DIR.is_dir() and str(_CSRC_DIR) not in sys.path:
    sys.path.append(str(_CSRC_DIR))

from autovision.tasks.detection.shared.gtr.encoder import (  # noqa: E402
    ConvNormLayer_fuse,
    GTREncoder,
    get_activation,
)
from autovision.tasks.detection.shared.gtr.spatial_swiglu import (  # noqa: E402
    ViTAdapterSpatialSwiGLU,
)
from autovision.tasks.detection.shared.gtr.vit_adapter import ViTAdapter  # noqa: E402

__all__ = [
    "GTR_ONNX_EXPORT_BLOCKER",
    "GTREncoder",
    "ViTAdapter",
    "ViTAdapterSpatialSwiGLU",
    "ConvNormLayer_fuse",
    "get_activation",
    "gtr_available",
]

#: Export surface of the GLA backbone. ``GTRViTBackbone`` carries this as
#: its ``export_blocker``, so any model composed with it (the ``gtr/*``
#: task families and ``detr/gtrvit-*``) is refused by the ONNX/TensorRT
#: export pipeline with this message instead of an opaque tracer error.
GTR_ONNX_EXPORT_BLOCKER = (
    "ONNX/TensorRT export is unavailable for the GTR GLA backbone: its "
    "chunk_gla and fused RMSNorm+gate kernels are custom CUDA ops without "
    "an ONNX lowering yet (deployment engineering, design doc 31 G2; "
    "TensorRT additionally needs the official gla_plugin.cu)."
)


def gtr_available() -> bool:
    """True when flash-linear-attention is importable (the gtr extra)."""
    try:
        import fla  # noqa: F401

        return True
    except ImportError:
        return False
