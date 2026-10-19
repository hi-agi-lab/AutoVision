# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR oriented detection family (roadmap design doc 31 §11 S3a).

Gold parity (2026-10-16, RTX 4090D): official ``GTR`` with the
``OBBGTRTransformer`` decoder (their engine, OBB-S config, official EMA
weights, stale anchors skipped per the official solver convention) vs
this family, same inputs at the 1024 canvas — ``pred_logits`` and the
5-parameter rotated boxes (cx, cy, w, h, a) **bit-exact** on random 1024
and a real image; predict() returns rotated polygons.

Training (oriented criterion/matcher/oriented-CDN) is §11 S3b.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.oriented_detection.gtr.task_model import GTR_OBB_VARIANTS

_fla = True
try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_gla = pytest.mark.skipif(
    not (_fla and torch.cuda.is_available()),
    reason="GTR GLA forward needs flash-linear-attention + CUDA",
)

requires_fla = pytest.mark.skipif(
    not _fla,
    reason="GTR load tests need flash-linear-attention (pip install -e '.[gtr]')",
)


def _obb_checkpoint() -> Path:
    try:
        return resolve_gtr_asset("gtrobb_s_dota").path
    except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
        pytest.skip("local GTR OBB weights not present")


@pytest.fixture(scope="module")
def obb_checkpoint() -> Path:
    return _obb_checkpoint()


class TestGtrObbLoad:
    def test_variant_table(self):
        assert set(GTR_OBB_VARIANTS) == {"s", "x"}
        assert GTR_OBB_VARIANTS["s"] == (192, 3, 4.0, 192, 512, 4)
        assert GTR_OBB_VARIANTS["x"] == (384, 6, 6.0, 256, 2048, 4)

    @requires_fla
    def test_x_official_checkpoint_loads_completely(self):
        try:
            resolve_gtr_asset("gtrobb_x_dota")
        except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
            pytest.skip("local GTR OBB X weights not present")
        from autovision.tasks.oriented_detection.gtr import (
            GTRROBBOrientedDetection,
        )

        net = GTRROBBOrientedDetection(
            num_classes=15, variant="x", image_size=(1024, 1024),
            checkpoint="gtrobb_x_dota",
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 274
        assert net.decoder.decoder.layers[0].linear1.weight.shape == (2048, 256)
        offsets = net.decoder.decoder.layers[0].cross_attn.sampling_offsets
        assert offsets.weight.shape == (8 * 12 * 2, 256)

    @requires_fla
    def test_official_checkpoint_loads_completely(self, obb_checkpoint):
        from autovision.tasks.oriented_detection.gtr import (
            GTRROBBOrientedDetection,
        )

        net = GTRROBBOrientedDetection(
            num_classes=15, variant="s", image_size=(1024, 1024),
            checkpoint="gtrobb_s_dota",
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 274
        # Stale stored anchors were skipped; the decoder regenerated the
        # official 1024-canvas 5-channel anchors (incl. the angle channel).
        assert tuple(net.decoder.anchors.shape) == (1, 21504, 5)
        # Decoder attention heads are the official default (8), independent
        # of the backbone's 3 heads: 8 * sum([3,6,3]) * 2 = 192-wide offsets.
        offsets = net.decoder.decoder.layers[0].cross_attn.sampling_offsets
        assert offsets.weight.shape == (8 * 12 * 2, 192)
        # The OBB lineage names the post-FFN norm norm2 (no norm3 rename).
        assert hasattr(net.decoder.decoder.layers[0], "norm2")
        assert not hasattr(net.decoder.decoder.layers[0], "norm3")


class TestRboxOps:
    def test_canonical_periodicity_and_longedge(self):
        from autovision.tasks.detection.shared.gtr import rbox_ops

        torch.manual_seed(0)
        boxes = torch.rand(16, 5)
        boxes[..., 2:4] = boxes[..., 2:4] * 0.3 + 0.05
        boxes[..., 4] = boxes[..., 4] * 3.141592653589793

        canonical = rbox_ops.rbox_rad_to_canonical(boxes)
        corners = rbox_ops.rbox_to_corners(canonical)
        # 90-degree periodicity: shifting theta by k*pi/2 and swapping w/h
        # gives the same corner set (as a set, up to order).
        shifted = boxes.clone()
        shifted[..., 4] = shifted[..., 4] + 3.141592653589793 / 2
        shifted[..., 2], shifted[..., 3] = boxes[..., 3], boxes[..., 2]
        corners_shifted = rbox_ops.rbox_to_corners(
            rbox_ops.rbox_rad_to_canonical(shifted)
        )
        assert torch.allclose(corners, corners_shifted, atol=1e-5)

        longedge = rbox_ops.rbox_rad_to_longedge(boxes)
        assert bool((longedge[..., 2] >= longedge[..., 3] - 1e-6).all())
        assert bool((longedge[..., 4] >= -1e-6).all())
        assert bool((longedge[..., 4] < 3.141592653589793 + 1e-6).all())
        # Corner SETS must coincide (the canonical/long-edge transforms
        # may rotate the vertex order) — the upstream Chamfer cost is the
        # set-distance check these helpers ship with; per-sample pairs are
        # the matrix diagonal.
        pair_distance = rbox_ops.chamfer_cost_pairwise(longedge, boxes).diagonal()
        assert float(pair_distance.max()) < 1e-4
        canonical_distance = rbox_ops.chamfer_cost_pairwise(canonical, boxes).diagonal()
        assert float(canonical_distance.max()) < 1e-4

    def test_extbox_round_trip_and_hbox(self):
        from autovision.tasks.detection.shared.gtr import rbox_ops

        torch.manual_seed(1)
        boxes = torch.rand(64, 5)
        boxes[..., 2] = boxes[..., 2] * 0.4 + 0.1
        boxes[..., 3] = boxes[..., 3] * 0.3 + 0.05
        boxes[..., 4] = boxes[..., 4] * (3.141592653589793 / 2)
        cx, cy, Wr, Hr, eps, eta = rbox_ops.rbox_to_extbox(boxes)
        back = rbox_ops.extbox_to_rbox(cx, cy, Wr, Hr, eps, eta)
        assert torch.allclose(back[..., :2], boxes[..., :2], atol=1e-5)
        assert torch.allclose(back[..., 2:4], boxes[..., 2:4], atol=1e-5)
        assert torch.allclose(back[..., 4], boxes[..., 4], atol=1e-5)
        # Enclosing hbox contains the rotated corners.
        hbox = rbox_ops.rbox_to_hbox_xyxy(boxes)
        corners = rbox_ops.rbox_to_corners(boxes)
        assert bool((corners[..., 0] >= hbox[..., 0:1] - 1e-5).all())
        assert bool((corners[..., 1] >= hbox[..., 1:2] - 1e-5).all())
        assert bool((corners[..., 0] <= hbox[..., 2:3] + 1e-5).all())
        assert bool((corners[..., 1] <= hbox[..., 3:4] + 1e-5).all())

    def test_distance2rbox_keeps_valid_rbox(self):
        from autovision.tasks.detection.shared.gtr import rbox_ops

        torch.manual_seed(2)
        ref = torch.rand(8, 5)
        ref[..., 2:4] = ref[..., 2:4] * 0.4 + 0.1
        ref[..., 4] = ref[..., 4] / 2
        distance = torch.randn(8, 6) * 0.1
        decoded = rbox_ops.distance2rbox(ref, distance, 4.0)
        assert decoded.shape == (8, 5)
        assert torch.isfinite(decoded).all()
        assert bool((decoded[..., 2] > 0).all()) and bool((decoded[..., 3] > 0).all())
        assert bool((decoded[..., 4] >= 0).all()) and bool((decoded[..., 4] <= 1).all())


class TestGtrObbForward:
    @requires_gla
    def test_forward_shapes_and_predict_polygons(self, tmp_path: Path):
        import numpy as np
        from PIL import Image as PILImage

        from autovision.tasks.oriented_detection.gtr import (
            GTRROBBOrientedDetection,
        )

        net = GTRROBBOrientedDetection(
            num_classes=15, variant="s", image_size=(1024, 1024), load_checkpoint=False
        ).cuda().eval()
        with torch.no_grad():
            logits, boxes = net.forward_backend(torch.randn(1, 3, 1024, 1024).cuda())
        assert logits.shape == (1, 300, 15)
        assert boxes.shape == (1, 300, 5)
        assert bool((boxes[..., 4] >= 0).all()) and bool((boxes[..., 4] <= 1).all())

        image_path = tmp_path / "scene.png"
        PILImage.fromarray(
            np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
        ).save(image_path)
        prediction = net.predict(str(image_path), threshold=0.05)
        polygons = prediction["polygons"]
        assert polygons.ndim == 3 and polygons.shape[1:] == (4, 2)
        assert polygons.shape[0] == prediction["labels"].shape[0]
        assert prediction["scores"].shape[0] == polygons.shape[0]
