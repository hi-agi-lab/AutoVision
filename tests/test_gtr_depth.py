# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR metric-depth family (roadmap design doc 31 §11 S4).

Gold parity (2026-10-16, RTX 4090D): official ``GTRDepth`` (their engine,
their pretrain-S config, official EMA weights) vs this family, same
inputs — ``pred_depth`` **bit-exact** on random 640 and a real COCO image;
``predict()`` returns the depth map at the original frame (values in
meters). ``force_contiguous=False`` on the encoder was validated in the
same run (it is the official training-time configuration).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from autovision.registry.gtr_assets import resolve_gtr_asset
from autovision.tasks.depth_estimation.gtr.task_model import GTR_DEPTH_VARIANTS

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


def _depth_checkpoint() -> Path:
    try:
        return resolve_gtr_asset("gtrdepth_s").path
    except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
        pytest.skip("local GTR depth weights not present")


@pytest.fixture(scope="module")
def depth_checkpoint() -> Path:
    return _depth_checkpoint()


class TestGtrDepthLoad:
    def test_variant_table(self):
        assert GTR_DEPTH_VARIANTS == {
            "s": (192, 3, 4.0, 192),
            "m": (256, 4, 4.0, 256),
            "l": (384, 6, 4.0, 256),
            "x": (384, 6, 6.0, 256),
        }

    @pytest.mark.parametrize("variant", ["l", "x"])
    @requires_fla
    def test_lx_official_checkpoint_loads_completely(self, variant):
        try:
            resolve_gtr_asset(f"gtrdepth_{variant}")
        except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
            pytest.skip("local GTR depth L/X weights not present")
        from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

        net = GTRDepthEstimation(variant=variant, checkpoint=f"gtrdepth_{variant}")
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 132

    @requires_fla
    def test_official_checkpoint_loads_completely(self, depth_checkpoint):
        from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

        net = GTRDepthEstimation(
            variant="s", image_size=(640, 640), checkpoint="gtrdepth_s"
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 132
        # Encoder must keep the official depth training configuration.
        assert net.encoder.force_contiguous is False
        head_keys = [k for k in net.state_dict() if k.startswith("decoder.")]
        assert len(head_keys) == 39


class TestSilogAndMetrics:
    def test_silog_zero_on_perfect_prediction(self):
        import types

        from autovision.tasks.depth_estimation.gtr.config import (
            GTRDepthEstimationTrainArgs,
        )
        from autovision.tasks.depth_estimation.gtr.train_model import (
            GTRDepthEstimationTrain,
        )

        # _silog is pure math; bind it to an args-carrying stub to avoid
        # constructing the whole (GLA-requiring) model.
        stub = types.SimpleNamespace(model_args=GTRDepthEstimationTrainArgs())
        pred = torch.tensor([[1.0, 2.0, 4.0]])
        assert float(GTRDepthEstimationTrain._silog(stub, pred, pred)) == pytest.approx(
            0.0, abs=1e-6
        )
        # Constant log-diff d -> sqrt(d^2 - lambda * d^2).
        gt = pred * float(np.exp(0.5))
        expected = 0.5 * (1 - 0.5) ** 0.5
        assert float(GTRDepthEstimationTrain._silog(stub, pred, gt)) == pytest.approx(
            expected, abs=1e-6
        )

    def test_metrics_known_values(self):
        from autovision.metrics.depth_estimation import DepthMetricCollector

        gt = torch.full((1, 4, 4), 4.0)
        perfect = DepthMetricCollector(alignment="none")
        perfect.update(gt.clone(), gt)
        metrics = perfect.compute()
        assert metrics["abs_rel"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["delta1"] == 1.0

        # A pure scale error is removed by median/log-affine alignment but
        # not by alignment="none".
        scaled = gt * 2.0
        raw = DepthMetricCollector(alignment="none")
        raw.update(scaled, gt)
        assert raw.compute()["abs_rel"] == pytest.approx(1.0, abs=1e-4)
        aligned = DepthMetricCollector(alignment="median")
        aligned.update(scaled, gt)
        assert aligned.compute()["abs_rel"] == pytest.approx(0.0, abs=1e-4)

        # Invalid (out-of-range / zero) pixels are excluded.
        mixed = gt.clone()
        mixed[0, 0, 0] = 0.0  # invalid GT
        collector = DepthMetricCollector(alignment="none")
        collector.update(torch.where(mixed > 0, mixed, torch.full_like(mixed, 100.0)), mixed)
        assert collector.compute()["abs_rel"] == pytest.approx(0.0, abs=1e-6)


class TestDepthDataset:
    def test_pairs_and_readers(self, tmp_path: Path):
        from PIL import Image as PILImage

        from autovision.data.depth_estimation_dataset import (
            DepthEstimationDataset,
            list_depth_pairs,
        )

        images_dir = tmp_path / "images"
        depths_dir = tmp_path / "depths"
        images_dir.mkdir()
        depths_dir.mkdir()
        for index in range(2):
            PILImage.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(images_dir / f"img_{index}.png")
        # one .npy (meters) and one 16-bit png (millimetres)
        np.save(depths_dir / "img_0.npy", np.full((32, 32), 2.5, dtype=np.float32))
        PILImage.fromarray(
            np.full((32, 32), 1500, dtype=np.uint16)
        ).save(depths_dir / "img_1.png")

        pairs = list_depth_pairs(images_dir, depths_dir)
        assert len(pairs) == 2
        dataset = DepthEstimationDataset(pairs)
        item = dataset[0]
        assert item["depth"].dtype == torch.float32
        assert float(item["depth"].mean()) == pytest.approx(2.5, abs=1e-5)
        item = dataset[1]
        assert float(item["depth"].mean()) == pytest.approx(1.5, abs=1e-3)
        assert item["original_size"] == (32, 32)


class TestGtrDepthClosure:
    @requires_gla
    def test_training_closure_and_metrics(self, tmp_path: Path):
        """gtr/s dispatch → dataset → SiLog loss → delta1 validation →
        export, on two synthetic pairs."""
        from PIL import Image as PILImage

        from autovision import train_depth_estimation

        images_dir = tmp_path / "images"
        depths_dir = tmp_path / "depths"
        images_dir.mkdir()
        depths_dir.mkdir()
        rng = np.random.default_rng(0)
        for index in range(2):
            PILImage.fromarray(
                rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            ).save(images_dir / f"img_{index}.png")
            depth = rng.uniform(1.0, 6.0, (64, 64)).astype(np.float32)
            np.save(depths_dir / f"img_{index}.npy", depth)

        result = train_depth_estimation(
            out=tmp_path / "run",
            data={
                "train": {"images": images_dir, "depths": depths_dir},
                "val": {"images": images_dir, "depths": depths_dir},
            },
            model="gtr/s",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "load_checkpoint": False,
                "lr_warmup_steps": 2,
            },
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
        assert "val_metric/delta1" in (result.best_metrics or {}) or result.steps_run == 4

    @requires_gla
    def test_predict_returns_original_frame_depth(self, tmp_path: Path):
        from PIL import Image as PILImage

        from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

        image_path = tmp_path / "scene.png"
        array = np.random.default_rng(1).integers(0, 255, (48, 72, 3), dtype=np.uint8)
        PILImage.fromarray(array).save(image_path)

        net = GTRDepthEstimation(
            variant="s", image_size=(64, 64), load_checkpoint=False
        ).cuda().eval()
        prediction = net.predict(str(image_path))
        assert tuple(prediction["depth"].shape) == (48, 72)
        assert torch.isfinite(prediction["depth"]).all()
        assert float(prediction["depth"].min()) > 0


class TestDepthValidationResolution:
    def test_validation_uses_gt_canvas_resolution(self):
        # Regression: validation used to upsample predictions to the
        # original frame and compare them against canvas-resized GT, which
        # crashes (or silently mis-scores) when the two sizes differ.
        from types import SimpleNamespace

        from torch import nn

        from autovision.tasks.depth_estimation.gtr.train_model import (
            GTRDepthEstimationTrain,
        )

        class _FakeDepthModel(nn.Module):
            def forward_backend(self, x):
                return torch.ones(x.shape[0], 8, 8)

        train = object.__new__(GTRDepthEstimationTrain)
        nn.Module.__init__(train)
        train.model = _FakeDepthModel()
        train.model_args = SimpleNamespace(
            min_depth=0.001, max_depth=10.0, silog_lambd=0.15
        )
        train._evaluators = {}
        batch = {
            "image": torch.randn(1, 3, 16, 24),
            "depth": torch.full((1, 16, 24), 2.0),
            "original_size": [(48, 32)],
        }
        result = train.validation_step(batch, step=0)
        assert torch.isfinite(result.loss)
        metrics = train.compute_metrics("val")
        assert "val_metric/abs_rel" in metrics
        assert np.isfinite(metrics["val_metric/abs_rel"])
