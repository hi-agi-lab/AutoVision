# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose-estimation family (roadmap design doc 31 §11 S5a).

Gold parity (2026-10-16, RTX 4090D): official ``GTRPose`` (their engine,
pose-S config, official EMA weights, stale anchors skipped) vs this
family, same inputs — ``pred_logits`` and ``pred_keypoints`` **bit-exact**
on random 640 and a real image; predict() returns COCO ``[x, y, v]``
triplets at the original frame.

Training (CDN + keypoint loss/matcher) is §11 S5b.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.pose_estimation.gtr.task_model import GTR_POSE_VARIANTS

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


def _pose_checkpoint() -> Path:
    try:
        return resolve_gtr_asset("gtrpose_s_coco").path
    except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
        pytest.skip("local GTR pose weights not present")


@pytest.fixture(scope="module")
def pose_checkpoint() -> Path:
    return _pose_checkpoint()


class TestGtrPoseLoad:
    def test_variant_table(self):
        # (embed_dim, num_heads, ffn_ratio, hidden, decoder_layers,
        # decoder_ffn) — official per-variant shapes. L/X down-project to a
        # 256-wide encoder+decoder (the semseg-lineage width trap) and
        # decode 4 layers; S decodes 3.
        assert GTR_POSE_VARIANTS == {
            "s": (192, 3, 4.0, 192, 3, 512),
            "m": (256, 4, 4.0, 256, 4, 512),
            "l": (384, 6, 4.0, 256, 4, 1024),
            "x": (384, 6, 6.0, 256, 4, 2048),
        }

    @pytest.mark.parametrize(
        ("variant", "transferred", "layers", "ffn", "hidden"),
        [
            ("s", 239, 3, 512, 192),
            ("m", 277, 4, 512, 256),
            ("l", 277, 4, 1024, 256),
            ("x", 277, 4, 2048, 256),
        ],
    )
    def test_every_variant_loads_cleanly(
        self, variant, transferred, layers, ffn, hidden
    ):
        """Every official checkpoint loads with zero shape skips AND zero
        checkpoint-only keys (the gate that catches a decoder shallower
        than the checkpoint — official M/L/X decode 4 levels)."""
        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        name = f"gtrpose_{variant}_coco"
        try:
            resolve_gtr_asset(name)
        except Exception:  # noqa: BLE001
            pytest.skip("local GTR pose weights not present")
        net = GTRPoseEstimation(
            num_classes=2, variant=variant, image_size=(640, 640), checkpoint=name
        )
        report = net.load_report
        assert report["transferred"] == transferred, report
        assert report["skipped_shape"] == 0, report
        assert report["checkpoint_only"] == 0, report
        assert report["encoder_decoder_coverage"] == 1.0, report
        assert len(net.decoder.decoder.layers) == layers
        assert net.decoder.decoder.layers[0].linear1.weight.shape == (ffn, hidden)
        assert net.encoder.stages[0][0].cv1.conv.weight.shape[0] == hidden

    def test_decoder_depth_mismatch_refused(self):
        """A network shallower than the checkpoint (extra layers on the
        checkpoint side) must be refused — coverage counting alone is blind
        to that direction."""
        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        try:
            resolve_gtr_asset("gtrpose_s_coco")
        except Exception:  # noqa: BLE001
            pytest.skip("local GTR pose weights not present")
        with pytest.raises(ValueError, match="checkpoint_only"):
            GTRPoseEstimation(
                num_classes=2,
                variant="s",
                num_decoder_layers=2,
                image_size=(640, 640),
                checkpoint="gtrpose_s_coco",
            )

    def test_official_checkpoint_loads_completely(self, pose_checkpoint):
        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        net = GTRPoseEstimation(
            num_classes=2, variant="s", image_size=(640, 640),
            checkpoint="gtrpose_s_coco",
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 239
        # Stale stored anchors were skipped; regenerated at the 640 canvas
        # as 2-channel (xy) proposals. Upstream registers the INVERTED
        # mask (True = padding).
        assert tuple(net.decoder.anchors.shape) == (1, 8400, 2)
        assert net.decoder.valid_mask.dtype == torch.bool
        # Upstream hard-codes the CDN label embedding to 81 rows and the
        # LQE to a 256 hidden width — both load verbatim.
        assert net.decoder.label_enc.weight.shape == (81, 192)
        assert net.decoder.lqe_embed[0].reg_conf.layers[0].weight.shape[0] == 256

    def test_pose_weighting_function_is_the_pose_lineage_one(self):
        """The pose decoder carries its OWN weighting function (measured
        9.5e-07 apart from the shared D-FINE one) — the port must keep it;
        swapping in the shared version costs bit-exactness."""
        from autovision.tasks.detection.shared.dfine_utils import (
            weighting_function as shared,
        )
        from autovision.tasks.detection.shared.gtr.pose_decoder import (
            weighting_function as pose,
        )

        up = torch.tensor([0.5])
        reg_scale = torch.tensor([4.0])
        theirs = pose(32, up, reg_scale)
        ours = shared(32, float(up), float(reg_scale))
        assert theirs.shape == ours.shape
        assert not torch.equal(theirs, ours)  # deliberately distinct lineages


class TestGtrPoseForward:
    @requires_gla
    def test_forward_shapes_and_predict_keypoints(self, tmp_path: Path):
        import numpy as np
        from PIL import Image as PILImage

        from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

        net = GTRPoseEstimation(
            num_classes=2, variant="s", image_size=(640, 640), load_checkpoint=False
        ).cuda().eval()
        with torch.no_grad():
            logits, keypoints = net.forward_backend(torch.randn(1, 3, 640, 640).cuda())
        assert logits.shape == (1, 60, 2)
        assert keypoints.shape == (1, 60, 17 * 2)

        image_path = tmp_path / "crowd.png"
        PILImage.fromarray(
            np.random.default_rng(0).integers(0, 255, (80, 120, 3), dtype=np.uint8)
        ).save(image_path)
        prediction = net.predict(str(image_path), threshold=0.0)
        kpts = prediction["keypoints"]
        assert kpts.ndim == 2 and kpts.shape[1] == 17 * 3  # [x, y, v] triplets
        assert bool((kpts[:, 2::3] == 1).all())  # visibility flag = 1
        # Coordinates live in the ORIGINAL frame (0..w / 0..h).
        assert float(kpts[:, 0::3].max()) <= 120.0
        assert float(kpts[:, 1::3].max()) <= 80.0
        assert prediction["scores"].shape[0] == kpts.shape[0]
