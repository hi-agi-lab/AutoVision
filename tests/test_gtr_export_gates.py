# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR family export gates: verify hooks + the GLA export blocker.

The GLA backbone runs custom CUDA kernels (``chunk_gla``, fused
RMSNorm+gate) without an ONNX lowering, so every model composed with
``GTRViTBackbone`` is refused by the ONNX/TensorRT pipeline with an
explicit error (design doc 31 G2). The four task families implement the
verification-hook contract regardless, so the numerical gate is in place
the moment the kernel-side lowering lands — the hooks are exercised here
against synthetic output pairs because a real export cannot run yet.
"""

from __future__ import annotations

import pytest
import torch

from autovision.errors import AutoVisionError

try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_gtr_extra = pytest.mark.skipif(
    not _fla, reason="GTR construction needs flash-linear-attention (pip install -e '.[gtr]')"
)


def _build_pose():
    from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

    return GTRPoseEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_depth():
    from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

    return GTRDepthEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_semseg():
    from autovision.tasks.semantic_segmentation.gtr import GTRSemanticSegmentation

    return GTRSemanticSegmentation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_obb():
    from autovision.tasks.oriented_detection.gtr import GTRROBBOrientedDetection

    return GTRROBBOrientedDetection(
        variant="s", image_size=(64, 64), load_checkpoint=False
    )


_TASK_MODELS = [
    pytest.param(_build_pose, ((2, 60, 2), (2, 60, 34)), id="pose"),
    pytest.param(_build_depth, ((2, 64, 64),), id="depth"),
    pytest.param(_build_semseg, ((2, 19, 64, 64),), id="semseg"),
    pytest.param(_build_obb, ((2, 300, 15), (2, 300, 5)), id="obb"),
]

_TASK_MODEL_BUILDERS = [_build_pose, _build_depth, _build_semseg, _build_obb]


def _as_forward_output(tensors: list[torch.Tensor]):
    """Single-output models return a bare tensor; multi-output ones a tuple."""
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


@requires_gtr_extra
@pytest.mark.parametrize(("build", "shapes"), _TASK_MODELS)
def test_verify_hook_contract(build, shapes):
    torch.manual_seed(0)
    model = build()
    torch_outputs = [torch.randn(shape) for shape in shapes]
    onnx_outputs = [tensor.numpy() for tensor in torch_outputs]

    model.verify_onnx_export_outputs(_as_forward_output(torch_outputs), onnx_outputs)

    perturbed = [array.copy() for array in onnx_outputs]
    perturbed[-1] += 1.0
    with pytest.raises(AssertionError):
        model.verify_onnx_export_outputs(
            _as_forward_output(torch_outputs), perturbed
        )


@requires_gtr_extra
def test_gtr_backbone_declares_export_blocker():
    from autovision.tasks.detection.detr.backbone_wrappers import GTRViTBackbone

    backbone = GTRViTBackbone(embed_dim=192, num_heads=3)
    assert backbone.export_blocker is not None
    assert "GTR GLA" in backbone.export_blocker


@requires_gtr_extra
@pytest.mark.parametrize(
    "build", _TASK_MODEL_BUILDERS, ids=("pose", "depth", "semseg", "obb")
)
def test_gtr_task_models_refuse_export(build, tmp_path):
    # Construction/blocking only; no forward runs (needs CUDA).
    model = build()
    assert model._export_blocker() is not None
    out = tmp_path / "model.onnx"
    with pytest.raises(AutoVisionError, match="GTR GLA"):
        model.export_onnx(out)
    assert not out.exists()


@requires_gtr_extra
def test_gtrvit_detection_variant_refuses_export(tmp_path):
    """The module-tree scan covers composition: ``detr/gtrvit-s`` carries
    the blocker via its backbone, without any per-model declaration."""
    from autovision.tasks.detection.detr.task_model import DETRObjectDetection

    network = DETRObjectDetection(
        num_classes=2,
        model_name="gtrvit-s",
        image_size=(64, 64),
        decoder="dfine_official",
        load_weights=False,
        backbone_pretrained=False,
    )
    with pytest.raises(AutoVisionError, match="GTR GLA"):
        network.export_onnx(tmp_path / "model.onnx")


def test_plain_models_have_no_blocker():
    from autovision.tasks import RTDETRObjectDetection

    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    assert model._export_blocker() is None
