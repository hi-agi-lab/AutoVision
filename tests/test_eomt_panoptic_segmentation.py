"""Tests for the EOMT panoptic segmentation family (HF-backed)."""

import json
import math

import numpy as np
import pytest
import torch

from autovision import load_task_model as load_model
from autovision import train_panoptic_segmentation
from autovision.data.mask_panoptic_segmentation_dataset import (
    MaskPanopticSegmentationDataset,
    list_panoptic_pairs,
    load_panoptic_segments,
    split_panoptic_classes,
)
from autovision.metrics.panoptic_segmentation import PanopticMetricCollector
from autovision.tasks.panoptic_segmentation.dinov2_eomt import (
    Dinov2EomtPanopticSegmentation,
    Dinov2EomtPanopticSegmentationTrain,
    Dinov2EomtPanopticSegmentationTrainArgs,
)
from autovision.tasks.shared.eomt.postprocess import panoptic_segmentation_decode

MODEL_KWARGS = {
    "image_size": 64,
    "num_queries": 6,
    "num_blocks": 2,
    "num_upscale_blocks": 1,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}


def make_train_model(**overrides) -> Dinov2EomtPanopticSegmentationTrain:
    torch.manual_seed(0)
    args = Dinov2EomtPanopticSegmentationTrainArgs(
        num_classes=2,
        num_thing_classes=1,
        lr_warmup_steps=2,
        **{**MODEL_KWARGS, **overrides},
    )
    return Dinov2EomtPanopticSegmentationTrain(args)


def make_batch(size: int = 64, images: int = 2):
    """One thing (internal 0) top-left + one stuff (internal 1) bottom-right."""
    masks = torch.zeros(images, size, size, 2, dtype=torch.long)
    binary_masks = []
    for index in range(images):
        masks[index, : size // 2, : size // 2, 0] = 0
        masks[index, : size // 2, : size // 2, 1] = 0
        masks[index, size // 2 :, size // 2 :, 0] = 1
        masks[index, size // 2 :, size // 2 :, 1] = 1
        masks[index, : size // 2, size // 2 :, 0] = 2  # ignore class
        masks[index, : size // 2, size // 2 :, 1] = -1
        thing = torch.zeros(1, size, size)
        thing[:, : size // 2, : size // 2] = 1.0
        stuff = torch.zeros(1, size, size)
        stuff[:, size // 2 :, size // 2 :] = 1.0
        binary_masks.append(
            {
                "masks": torch.cat([thing, stuff]),
                "labels": torch.tensor([0, 1]),
                "iscrowd": torch.tensor([False, False]),
            }
        )
    return {
        "image": torch.randn(images, 3, size, size),
        "masks": masks,
        "binary_masks": binary_masks,
    }


# ── panoptic quality metric ─────────────────────────────────────────────


class TestPanopticQuality:
    def make_metric(self, **kwargs) -> PanopticMetricCollector:
        return PanopticMetricCollector([0], [1], **kwargs)

    @staticmethod
    def blank_target(*shape) -> torch.Tensor:
        """(class, segment) map whose background is void (ignore class)."""
        target = torch.full(shape, -1, dtype=torch.long)
        target[..., 0] = 2  # ignore class, not declared -> void
        return target

    def test_perfect_prediction(self):
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])  # thing
        target[0, 2:, 2:] = torch.tensor([1, 1])  # stuff
        metric = self.make_metric()
        metric.update(target.clone(), target)
        result = metric.compute()
        assert result["pq"] == 1.0 and result["sq"] == 1.0 and result["rq"] == 1.0

    def test_iou_three_quarters_match(self):
        # GT thing = 4 px; pred thing covers 3 of them -> IoU 0.75 (match).
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])
        pred = target.clone()
        pred[0, 1, 1] = torch.tensor([2, -1])  # unassigned (void for pred)
        metric = self.make_metric()
        metric.update(pred, target)
        result = metric.compute()
        assert math.isclose(result["pq_0"], 0.75)
        assert math.isclose(result["sq_0"], 0.75)
        assert result["rq_0"] == 1.0

    def test_missed_ground_truth_counts_fn(self):
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])
        pred = self.blank_target(1, 4, 4, 2)  # nothing predicted
        metric = self.make_metric()
        metric.update(pred, target)
        result = metric.compute()
        assert result["rq_0"] == 0.0 and result["pq_0"] == 0.0

    def test_void_only_prediction_is_not_fp(self):
        # A prediction whose pixels are all void (ignore class) must not
        # create a false positive.
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])
        pred = target.clone()
        pred[0, 2:, 2:] = torch.tensor([2, 5])  # ignore label + segment
        metric = self.make_metric()
        metric.update(pred, target)
        result = metric.compute()
        assert result["rq_0"] == 1.0 and result["pq"] == 1.0

    def test_crowd_ground_truth_is_void(self):
        # Crowd pixels (label -1) are void: predicting there is not FP.
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])
        target[0, 2:, 2:, 0] = -1  # crowd region, no valid GT segment
        pred = target.clone()
        pred[0, 2:, 2:] = torch.tensor([1, 0])
        metric = self.make_metric()
        metric.update(pred, target)
        result = metric.compute()
        # Class 1 has no GT segment; the prediction there is void-excused.
        assert result["rq_0"] == 1.0
        assert result["pq_1"] == 0.0 and result["rq_1"] == 0.0

    def test_stuff_merging_pred_side(self):
        # GT has two segments of the same stuff class; the prediction merges
        # them into one segment id -> only one can match, the other is FN.
        target = self.blank_target(1, 4, 2, 2)
        target[0, :2, :, 0] = 1
        target[0, :2, :, 1] = 0  # segment 0 (top half)
        target[0, 2:, :, 0] = 1
        target[0, 2:, :, 1] = 1  # segment 1 (bottom half)
        pred = self.blank_target(1, 4, 2, 2)
        pred[..., 0] = 1
        pred[..., 1] = 0  # single merged stuff segment covering both
        metric = self.make_metric()
        metric.update(pred, target)
        metric.compute()
        # Each half IoU = 0.5 (not > 0.5), so nothing matches.
        state = metric.export_state()
        assert state["tp"][1] == 0.0
        assert state["fn"][1] == 2.0 and state["fp"][1] == 1.0

    def test_stuff_merging_exact_match(self):
        # Same setup but the merged pred segment covers one GT half exactly.
        target = self.blank_target(1, 4, 2, 2)
        target[0, :2, :, 0] = 1
        target[0, :2, :, 1] = 0
        target[0, 2:, :, 0] = 1
        target[0, 2:, :, 1] = 1
        pred = self.blank_target(1, 4, 2, 2)
        pred[0, :2, :, 0] = 1
        pred[0, :2, :, 1] = 3  # merged single segment over the top half
        metric = self.make_metric()
        metric.update(pred, target)
        state = metric.export_state()
        assert state["tp"][1] == 1.0 and state["fn"][1] == 1.0
        # PQ divides by (TP + 0.5*FN) = 1.5; SQ divides by TP only.
        assert math.isclose(metric.compute()["pq_1"], 2 / 3)
        assert metric.compute()["sq_1"] == 1.0

    def test_state_roundtrip_matches_joint_update(self):
        target_a = self.blank_target(1, 4, 4, 2)
        target_a[0, :2, :2] = torch.tensor([0, 0])
        target_b = self.blank_target(1, 4, 4, 2)
        target_b[0, 2:, 2:] = torch.tensor([1, 1])
        pred_a = target_a.clone()
        pred_a[0, 0, 0] = torch.tensor([2, -1])  # drop one pixel of the thing
        empty_pred = self.blank_target(1, 4, 4, 2)
        joint = self.make_metric()
        joint.update(pred_a, target_a)
        joint.update(empty_pred, target_b)
        split = self.make_metric()
        split.update(pred_a, target_a)
        other = self.make_metric()
        other.update(empty_pred, target_b)
        split.merge_state(other.export_state())
        for key in joint.export_state():
            assert torch.equal(split.export_state()[key], joint.export_state()[key])

    def test_empty_inputs(self):
        metric = self.make_metric()
        assert metric.compute() == {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "pq_0": 0.0,
            "sq_0": 0.0,
            "rq_0": 0.0,
            "pq_1": 0.0,
            "sq_1": 0.0,
            "rq_1": 0.0,
        }

    def test_reset(self):
        target = self.blank_target(1, 4, 4, 2)
        target[0, :2, :2] = torch.tensor([0, 0])
        metric = self.make_metric()
        metric.update(target.clone(), target)
        metric.reset()
        assert metric.compute()["pq"] == 0.0


# ── panoptic decode ─────────────────────────────────────────────────────


class TestPanopticDecode:
    def make_logits(self, num_labels: int = 2):
        masks = torch.full((1, 3, 4, 4), -10.0)
        classes = torch.full((1, 3, num_labels + 1), -10.0)
        return masks, classes

    def test_confident_query_wins(self):
        masks, classes = self.make_logits()
        masks[0, 0] = 10.0
        classes[0, 0, 0] = 10.0
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
        )
        result = results[0]
        assert result["masks"].shape == (4, 4, 2)
        assert result["segment_ids"].tolist() == [0]
        assert result["scores"].numel() == 1
        # Every pixel confidently claimed by query 0 (class 0, segment 0).
        assert (result["masks"][..., 0] == 0).all()
        assert (result["masks"][..., 1] == 0).all()

    def test_null_class_queries_dropped(self):
        masks, classes = self.make_logits()
        masks[0, 0] = 10.0
        classes[0, 0, 2] = 10.0  # null class wins -> dropped
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
        )
        assert results[0]["segment_ids"].numel() == 0
        # Unassigned pixels carry the ignore class and no segment.
        assert (results[0]["masks"][..., 0] == 2).all()
        assert (results[0]["masks"][..., 1] == -1).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_decode_keeps_device(self):
        # Regression: the contiguous-id rescan built a CPU `present` tensor
        # (crashed on every non-empty GPU decode) and the empty branch
        # created a CPU mask map alongside GPU labels/scores.
        masks, classes = self.make_logits()
        masks, classes = masks.cuda(), classes.cuda()
        meta = [{"orig_h": 4, "orig_w": 4}]
        kwargs = dict(num_labels=2, num_thing_classes=1)
        kept = panoptic_segmentation_decode(masks, classes, meta, **kwargs)[0]
        assert kept["masks"].device.type == "cuda"
        assert kept["segment_ids"].device.type == "cuda"
        assert kept["scores"].device.type == "cuda"
        null_classes = classes.clone()
        null_classes[..., -1] = 10.0  # every query maps to the null class
        empty = panoptic_segmentation_decode(masks, null_classes, meta, **kwargs)[0]
        assert empty["masks"].device.type == "cuda"
        assert empty["segment_ids"].device.type == "cuda"

    def test_low_confidence_dropped(self):
        masks, classes = self.make_logits()
        masks[0, 0] = 10.0
        classes[0, 0, 0] = 0.0  # ties with the null class -> score ~0.33 < 0.8
        classes[0, 0, 2] = 0.0
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
        )
        assert results[0]["segment_ids"].numel() == 0

    def test_stuff_merging_shares_segment_id(self):
        masks, classes = self.make_logits()
        masks[0, 0, :, :2] = 10.0  # query 0: left half
        masks[0, 1, :, 2:] = 10.0  # query 1: right half
        classes[0, 0, 1] = 10.0  # both stuff (class 1)
        classes[0, 1, 1] = 10.0
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
        )
        result = results[0]
        assert result["segment_ids"].tolist() == [0, 0]  # merged
        assert (result["masks"][..., 1][result["masks"][..., 1] >= 0] == 0).all()

    def test_thing_queries_keep_distinct_ids(self):
        masks, classes = self.make_logits()
        masks[0, 0, :, :2] = 10.0
        masks[0, 1, :, 2:] = 10.0
        classes[0, 0, 0] = 10.0  # thing (class 0)
        classes[0, 1, 0] = 10.0
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
        )
        assert results[0]["segment_ids"].tolist() == [0, 1]

    def test_overlap_filter_drops_shrunk_masks(self):
        # Query mask covers the full image but the low-score half is claimed
        # by a stronger query: the loser keeps < overlap_threshold of its
        # area and must be filtered out.
        masks, classes = self.make_logits()
        masks[0, 0] = 10.0
        masks[0, 1] = 9.0
        classes[0, 0, 0] = 12.0
        classes[0, 1, 0] = 10.0
        results = panoptic_segmentation_decode(
            masks, classes, [{"orig_h": 4, "orig_w": 4}],
            num_labels=2, num_thing_classes=1,
            mask_overlap_threshold=0.8,
        )
        # Both queries claim every pixel; only the winner survives.
        assert results[0]["segment_ids"].numel() == 1


# ── dataset ─────────────────────────────────────────────────────────────


def write_panoptic_data(tmp_path, num_images: int = 2, size: int = 64):
    from PIL import Image

    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "masks").mkdir(parents=True)
        annotations = {}
        for i in range(num_images):
            stem = f"img_{split[0]}{i}"
            Image.fromarray(
                np.random.default_rng(hash(stem) % 2**32).integers(
                    0, 256, (size, size, 3), dtype=np.uint8
                )
            ).save(tmp_path / split / "images" / f"{stem}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[: size // 2, : size // 2] = 1  # thing segment
            mask[size // 2 :, size // 2 :] = 2  # stuff segment
            Image.fromarray(mask).save(tmp_path / split / "masks" / f"{stem}.png")
            annotations[stem] = [
                {"id": 1, "category_id": 1, "iscrowd": 0},
                {"id": 2, "category_id": 2, "iscrowd": 0},
            ]
        path = tmp_path / split / "segments.json"
        path.write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": 1, "name": "car", "isthing": 1},
                        {"id": 2, "name": "sky", "isthing": 0},
                    ],
                    "annotations": annotations,
                }
            )
        )
    return {
        "train": {
            "images": tmp_path / "train" / "images",
            "masks": tmp_path / "train" / "masks",
            "segments": tmp_path / "train" / "segments.json",
        },
        "val": {
            "images": tmp_path / "val" / "images",
            "masks": tmp_path / "val" / "masks",
            "segments": tmp_path / "val" / "segments.json",
        },
    }


class TestPanopticDataset:
    def test_split_classes_things_first(self):
        mapping, things, stuffs = split_panoptic_classes(
            {2: {"name": "sky", "isthing": 0}, 1: {"name": "car", "isthing": 1}}
        )
        assert mapping == {1: 0, 2: 1}
        assert things == {0: "car"} and stuffs == {1: "sky"}

    def test_dataset_items(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        categories, segments = load_panoptic_segments(data["train"]["segments"])
        pairs = list_panoptic_pairs(data["train"]["images"], data["train"]["masks"])
        assert len(pairs) == 2
        dataset = MaskPanopticSegmentationDataset(
            pairs, categories, segments, num_channels=3
        )
        assert dataset.num_classes == 2
        assert dataset.internal_ignore_class_id == 2
        assert dataset.thing_classes == {0: "car"}
        assert dataset.stuff_classes == {1: "sky"}
        item = dataset[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["masks"].shape == (64, 64, 2)
        # Top-left thing, bottom-right stuff, unassigned -> ignore class.
        assert item["masks"][0, 0].tolist() == [0, 0]
        assert item["masks"][63, 63].tolist() == [1, 1]
        assert item["masks"][0, 63].tolist() == [2, -1]
        assert item["binary_masks"]["masks"].shape == (2, 64, 64)
        assert item["binary_masks"]["labels"].tolist() == [0, 1]
        assert item["binary_masks"]["iscrowd"].tolist() == [False, False]

    def test_crowd_handling(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        categories, segments = load_panoptic_segments(data["val"]["segments"])
        # Mark the stuff segment as crowd.
        for stems in segments.values():
            stems[1]["iscrowd"] = 1
        pairs = list_panoptic_pairs(data["val"]["images"], data["val"]["masks"])
        keep = MaskPanopticSegmentationDataset(pairs, categories, segments)
        drop = MaskPanopticSegmentationDataset(
            pairs, categories, segments, drop_crowd=True
        )
        item = keep[0]
        assert item["masks"][63, 63].tolist() == [-1, 1]  # crowd -> void in metrics
        assert item["binary_masks"]["labels"].tolist() == [0, 1]
        assert item["binary_masks"]["iscrowd"].tolist() == [False, True]
        dropped = drop[0]
        assert dropped["binary_masks"]["labels"].tolist() == [0]
        assert dropped["masks"][63, 63].tolist() == [2, -1]  # back to ignore

    def test_unknown_segment_and_class_skipped(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        categories, segments = load_panoptic_segments(data["train"]["segments"])
        segments["img_t0"].append({"id": 7, "category_id": 9, "iscrowd": 0})
        pairs = list_panoptic_pairs(data["train"]["images"], data["train"]["masks"])
        dataset = MaskPanopticSegmentationDataset(pairs, categories, segments)
        item = dataset[0]
        # Only the two valid segments survive; unknown id 7 is unlabeled.
        assert item["binary_masks"]["labels"].tolist() == [0, 1]


# ── train model ─────────────────────────────────────────────────────────


class TestPanopticTrainModel:
    def test_training_backward(self):
        train = make_train_model()
        result = train.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert train.model.model.class_predictor.weight.grad is not None

    def test_validation_updates_pq(self):
        train = make_train_model()
        train.model.eval()
        result = train.validation_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        metrics = train.compute_metrics("val")
        assert set(metrics) == {
            "val_metric/pq",
            "val_metric/sq",
            "val_metric/rq",
            "val_metric/pq_0",
            "val_metric/sq_0",
            "val_metric/rq_0",
            "val_metric/pq_1",
            "val_metric/sq_1",
            "val_metric/rq_1",
        }
        assert 0.0 <= metrics["val_metric/pq"] <= 1.0

    def test_metric_state_roundtrip(self):
        train = make_train_model()
        train.model.eval()
        train.validation_step(make_batch(), step=0)
        other = make_train_model()
        other.merge_metric_state("val", train.export_metric_state("val"))
        assert 0.0 <= other.compute_metrics("val")["val_metric/pq"] <= 1.0

    def test_task_model_decode_remaps_user_classes(self):
        task_model = Dinov2EomtPanopticSegmentation(
            num_classes=2, image_size=64, num_queries=6, num_blocks=2,
            num_upscale_blocks=1, hidden_size=32, num_hidden_layers=4,
            num_attention_heads=4, num_thing_classes=1,
        )
        assert task_model.internal_class_to_class.tolist() == [0, 1, 1]
        assert task_model.init_args["num_thing_classes"] == 1

    def test_class_var_defaults(self):
        assert Dinov2EomtPanopticSegmentationTrainArgs.default_batch_size == 16
        assert Dinov2EomtPanopticSegmentationTrainArgs.default_steps == 90_000


# ── end to end ──────────────────────────────────────────────────────────


class TestPanopticEndToEnd:
    def test_train_and_predict(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        result = train_panoptic_segmentation(
            out=tmp_path / "run",
            data=data,
            model="dinov2/eomt",
            steps=3,
            batch_size=2,
            num_workers=0,
            model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
        )
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
        assert result.best_metrics is not None
        assert 0.0 <= result.best_metrics["val_metric/pq"] <= 1.0
        assert result.classes == {0: "car", 1: "sky"}

        loaded = load_model(result.exported_model_path, device="cpu")
        prediction = loaded.predict(torch.rand(3, 64, 64))
        assert prediction["masks"].shape == (64, 64, 2)
        assert prediction["masks"].dtype == torch.long
        assert prediction["segment_ids"].ndim == 1
        assert prediction["scores"].ndim == 1
        assert prediction["masks"][..., 0].max() <= 1  # user class ids only

    def test_training_is_deterministic(self, tmp_path):
        import random as _random

        data = write_panoptic_data(tmp_path)
        runs = []
        for i in range(2):
            # Model weights are drawn at construction time inside the
            # command, so seed the process RNGs before each run.
            torch.manual_seed(0)
            _random.seed(0)
            np.random.seed(0)
            result = train_panoptic_segmentation(
                out=tmp_path / f"run_{i}",
                data=data,
                model="dinov2/eomt",
                steps=3,
                batch_size=2,
                num_workers=0,
                # Bit-exact comparison: CPU avoids GPU bf16 reduction-order
                # differences (GPU determinism is the opt-in
                # AUTOVISION_DETERMINISTIC path, not this seed test).
                devices="cpu",
                model_args={**MODEL_KWARGS, "lr_warmup_steps": 2},
            )
            runs.append(result.best_metrics)
        assert runs[0] == runs[1]

    def test_rejects_num_classes_in_model_args(self, tmp_path):
        data = write_panoptic_data(tmp_path)
        with pytest.raises(ValueError, match="num_classes"):
            train_panoptic_segmentation(
                out=tmp_path / "run",
                data=data,
                steps=3,
                num_workers=0,
                model_args={"num_classes": 2},
            )
