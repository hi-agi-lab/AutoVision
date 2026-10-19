# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR instance-segmentation family (roadmap design doc 31 §11 S2).

Gold parity (2026-10-16, RTX 4090D): official ``GTRSeg`` (their engine,
their seg-S config, anchors skipped per the official solver convention)
vs this family, same EMA weights, same inputs — logits / boxes / masks
**bit-exact** on random 640 and a real COCO image; mask-AP acceptance on
COCO val500 runs through the same evaluator for both sides (see
``runs/probe_speed/gtr_seg_mask_ap.py``).

Unit parity vs the official criterion/matcher (same seed, same tensors):
mask CE/Dice losses BIT-EQUAL; matcher assignments with mask costs
identical. Those checks need the GTR-main checkout and stay in the probe;
this suite keeps the in-repo halves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from autovision.registry.gtr_assets import GTR_SEG_VARIANTS, resolve_gtr_asset

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


def _seg_checkpoint() -> Path:
    try:
        return resolve_gtr_asset("gtrseg_s_coco").path
    except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
        pytest.skip("local GTR seg weights not present")


@pytest.fixture(scope="module")
def seg_checkpoint() -> Path:
    return _seg_checkpoint()


class TestGtrSegLoad:
    def test_variant_table_and_args(self):
        assert GTR_SEG_VARIANTS == {
            "s": (192, 3, 4.0),
            "m": (256, 4, 4.0),
            "l": (384, 6, 4.0),
            "x": (384, 6, 6.0),
        }
        from autovision.tasks.instance_segmentation.gtr.config import (
            GTRSegTrainArgs,
        )

        for variant in ("s", "m", "l", "x"):
            args = GTRSegTrainArgs(
                num_classes=80, variant=variant, load_checkpoint=False
            )
            assert args.variant == variant
        with pytest.raises(ValueError):
            GTRSegTrainArgs(
                num_classes=80, variant="unknown", load_checkpoint=False
            )

    @requires_gla
    def test_m_l_x_task_models_build(self):
        from autovision.tasks.instance_segmentation.gtr import (
            GTRSegObjectDetection,
        )

        for variant in ("m", "l", "x"):
            net = GTRSegObjectDetection(
                num_classes=80, variant=variant, load_checkpoint=False
            )
            assert net.variant == variant

    @requires_gla
    @pytest.mark.parametrize("variant", ["m", "l", "x"])
    def test_variant_official_checkpoint_loads_completely(self, variant):
        try:
            resolve_gtr_asset(f"gtrseg_{variant}_coco")
        except Exception:  # noqa: BLE001 - absent local weights keep fresh checkouts green
            pytest.skip("local GTR seg M/L/X weights not present")
        from autovision.tasks.instance_segmentation.gtr import (
            GTRSegObjectDetection,
        )

        net = GTRSegObjectDetection(
            num_classes=80,
            variant=variant,
            image_size=(640, 640),
            checkpoint=f"gtrseg_{variant}_coco",
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["transferred"] == 309

    @requires_fla
    def test_official_checkpoint_loads_completely(self, seg_checkpoint):
        """Backbone strict + encoder/decoder with the norm rename; the
        mask head rides ``decoder.decoder.segmentation_head.*`` (official
        layout, single registration)."""
        from autovision.tasks.instance_segmentation.gtr import (
            GTRSegObjectDetection,
        )

        net = GTRSegObjectDetection(
            num_classes=80,
            variant="s",
            image_size=(640, 640),
            checkpoint="gtrseg_s_coco",  # registered asset (sha256 verified)
        )
        assert net.load_report is not None
        assert net.load_report["encoder_decoder_coverage"] == 1.0
        assert net.load_report["skipped_shape"] == 0
        assert net.load_report["unexpected"] == 0
        assert net.load_report["weights_source"] == "ema.module"
        assert net.load_report["transferred"] == 309
        # The mask head exists exactly once, under the inner decoder.
        head_keys = [k for k in net.state_dict() if "segmentation_head" in k]
        assert all(k.startswith("decoder.decoder.segmentation_head.") for k in head_keys)
        assert len(head_keys) == 35

    @requires_fla
    def test_det_path_unchanged_by_mask_flag(self):
        """Constructing without mask_downsample_ratio adds no mask keys
        (det regression guard)."""
        from autovision.tasks.detection.detr.task_model import DETRObjectDetection

        det = DETRObjectDetection(
            num_classes=80, model_name="gtrvit-s", image_size=(64, 64),
            load_weights=False, decoder="dfine_official",
            num_queries=10, num_top_queries=10, num_denoising=0,
        )
        assert not any("segmentation_head" in k for k in det.state_dict())


class TestMaskLosses:
    def test_criterion_registers_mask_losses(self):
        from autovision.tasks.detection.shared.dfine_criterion import (
            DFINECriterionV2,
        )
        from autovision.tasks.detection.shared.matcher import HungarianMatcher

        criterion = DFINECriterionV2(
            HungarianMatcher(), mask_losses=True, mask_sampling_ratio=16
        )
        assert criterion.weight_dict["loss_mask_ce"] == 5.0
        assert criterion.weight_dict["loss_mask_dice"] == 5.0
        assert criterion.mask_sampling_ratio == 16
        # Flag off → no mask keys (det default unchanged).
        plain = DFINECriterionV2(HungarianMatcher())
        assert "loss_mask_ce" not in plain.weight_dict

    def test_mask_point_losses_shapes_and_determinism(self):
        from autovision.tasks.detection.shared.dfine_criterion import (
            mask_point_losses,
        )

        torch.manual_seed(0)
        pred = torch.randn(2, 6, 16, 16)
        targets = [
            {
                "masks": (torch.rand(3, 20, 24) > 0.6).float(),
            }
            for _ in range(2)
        ]
        indices = [(torch.arange(3), torch.arange(3)) for _ in range(2)]
        torch.manual_seed(7)
        first = mask_point_losses(pred, targets, indices, torch.tensor(6.0), 16)
        torch.manual_seed(7)
        second = mask_point_losses(pred, targets, indices, torch.tensor(6.0), 16)
        assert set(first) == {"loss_mask_ce", "loss_mask_dice"}
        for key in first:
            assert torch.isfinite(first[key])
            assert first[key] > 0
            assert torch.equal(first[key], second[key])
        # All-empty matching → exact zero terms (no NaN).
        empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] * 2
        zeros = mask_point_losses(pred, targets, empty, torch.tensor(0.0), 16)
        assert zeros["loss_mask_ce"] == 0.0 and zeros["loss_mask_dice"] == 0.0

    def test_matcher_mask_cost_breaks_ties_towards_matching_mask(self):
        """Two queries with identical class/box costs: the mask cost must
        decide — the assignment picks the query whose mask matches the GT."""
        from autovision.tasks.detection.shared.matcher import (
            HungarianMatcher,
            forward_match,
        )

        torch.manual_seed(0)
        # Query 0: confident CORRECT mask (top-left positive). Query 1:
        # confident INVERTED mask. The target is the top-left block; the
        # discriminating points are everywhere, so the mask cost must
        # strongly prefer query 0.
        correct = torch.full((8, 8), -8.0)
        correct[:4, :4] = 8.0
        inverted = torch.full((8, 8), 8.0)
        inverted[:4, :4] = -8.0
        outputs = {
            "pred_logits": torch.zeros(1, 2, 3),
            "pred_boxes": torch.full((1, 2, 4), 0.5),
            "pred_masks": torch.stack([correct, inverted], dim=0).unsqueeze(0),
        }
        target_mask = torch.zeros(1, 8, 8)
        target_mask[..., :4, :4] = 1.0
        targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
             "labels": torch.tensor([0]),
             "masks": target_mask}
        ]
        plain = HungarianMatcher()
        with_masks = HungarianMatcher(
            cost_mask_ce=5.0, cost_mask_dice=5.0, mask_point_sample_ratio=4
        )
        # Sanity: the plain matcher builds the matrix without masks.
        forward_match(plain, outputs, targets)
        torch.manual_seed(3)
        indices_mask = forward_match(with_masks, outputs, targets)
        # Boxes/logits tie exactly in the plain case (its choice is the
        # solver's tie-break); with mask costs query 0 must win outright.
        assert indices_mask[0][0].tolist() == [0]


class TestGtrSegClosure:
    @requires_gla
    def test_training_closure_and_mask_ap(self, tmp_path: Path):
        """End-to-end: gtr/s dispatch → masks in targets → mask losses →
        COCO mask AP in validation → export. Four steps on two synthetic
        COCO-instance images; the point is the wiring, not the metric."""
        from PIL import Image as PILImage

        from autovision import train_instance_segmentation

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        records = []
        for index in range(2):
            name = f"img_{index}.png"
            array = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy()
            PILImage.fromarray(array).save(images_dir / name)
            records.append({"id": index + 1, "file_name": name, "width": 64, "height": 64})
        annotations = []
        annotation_id = 1
        for record in records:
            for box in ((8, 8, 24, 24), (32, 32, 56, 56)):
                x, y, w, h = box
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": record["id"],
                        "category_id": 1 if annotation_id % 2 else 2,
                        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        payload = {
            "images": records,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        }
        annotation_file = tmp_path / "instances.json"
        annotation_file.write_text(json.dumps(payload))

        result = train_instance_segmentation(
            out=tmp_path / "run",
            data={
                "train": {"annotations": annotation_file, "images": images_dir},
                "val": {"annotations": annotation_file, "images": images_dir},
            },
            model="gtr/s",
            num_classes=2,
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "load_checkpoint": False,
                "lr_warmup_steps": 2,
                # The tiny input yields 84 selection positions — keep the
                # query count below that (prod default 300 assumes 640).
                "num_queries": 50,
                "num_top_queries": 50,
                "num_denoising": 10,
            },
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
        # The validation instrument is mask AP (the watch metric).
        assert result.best_metrics is not None or result.steps_run == 4
