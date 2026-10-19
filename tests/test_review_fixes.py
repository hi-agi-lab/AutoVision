# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Regression tests for code-review findings."""

import pytest
import torch
from tests.test_methods_distillationv3 import make_batch, make_method

from autovision.cli.train import PretrainConfig
from autovision.data.serialize.memory_mapped_sequence import (
    MemoryMappedSequence,
    write_items_to_file,
)
from autovision.pretext.distillationv3 import DistillationV3Args


class TestP0StringPaths:
    """P0: the public API must accept plain strings (CLI/YAML inputs)."""

    def test_pretrain_config_accepts_str_paths(self):
        config = PretrainConfig(out="/tmp/x", data="/tmp/y", model="m")
        assert config.out.name == "x"
        assert config.data.name == "y"

    def test_pretrain_config_accepts_str_list(self):
        config = PretrainConfig(out="/tmp/x", data=["/tmp/a", "/tmp/b"], model="m")
        assert [p.name for p in config.data] == ["a", "b"]

    def test_pretrain_config_accepts_path_objects(self):
        from pathlib import Path

        config = PretrainConfig(out=Path("/x"), data=Path("/y"), model="m")
        assert isinstance(config.out, Path)

    def test_precision_is_validated(self):
        with pytest.raises(Exception, match="precision"):
            PretrainConfig(out="/x", data="/y", model="m", precision="fp16-typo")


class TestP0DDPForward:
    """P0: PretextRecipe needs a forward for DDP wrapping."""

    def test_forward_is_training_step(self):
        # The method is stateful (queue updates), so compare two identically
        # seeded instances: one via forward (the DDP entry), one via
        # training_step.
        torch.manual_seed(0)
        method_forward = make_method()
        torch.manual_seed(0)
        method_step = make_method()
        batch = make_batch()
        loss_forward = method_forward(batch, batch_idx=0)
        loss_step = method_step.training_step(batch, batch_idx=0)
        assert torch.allclose(loss_forward, loss_step)


class TestP1MmapRefcount:
    """P1: attaching to an existing mmap must register a reference."""

    def test_exists_path_increments_refcount(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOVISION_DATA_CACHE_DIR", str(tmp_path / "cache"))
        from autovision.data.mmap_helpers import get_dataset_mmap_file

        def items():
            return [{"filenames": f"i{i}"} for i in range(3)]

        seq1 = get_dataset_mmap_file(items, tmp_path / "pre.mmap")
        seq2 = get_dataset_mmap_file(items, tmp_path / "pre.mmap")
        ref_file = tmp_path / "pre.ref_count"
        assert ref_file.exists()
        import json

        assert json.loads(ref_file.read_text()) == 2
        # First consumer exits -> file survives for the second.
        from autovision.data.mmap_helpers import _decrement_and_cleanup_if_zero
        from autovision.runtime.env import Env

        _decrement_and_cleanup_if_zero(seq1.path, reuse=Env.MMAP_REUSE_FILE.value)
        assert seq2.path.exists()
        assert len(seq2) == 3
        _decrement_and_cleanup_if_zero(seq2.path, reuse=Env.MMAP_REUSE_FILE.value)
        assert not seq2.path.exists()


class TestP1TeacherEval:
    """P1: method.train() must not flip the frozen teacher into train mode."""

    def test_teacher_stays_eval(self):
        method = make_method()
        method.train()
        assert not method.teacher_embedding_model.training
        assert method.training  # the method itself is in train mode
        method.eval()
        assert not method.teacher_embedding_model.training


class TestP1TeacherPretrained:
    def test_default_requests_pretrained(self):
        args = DistillationV3Args()
        assert args.teacher_pretrained is True

    def test_untrained_flag_available(self):
        args = DistillationV3Args(teacher_pretrained=False)
        assert args.teacher_pretrained is False


class TestP2SequenceContract:
    def test_negative_index(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file([{"v": i} for i in range(5)], path)
        seq = MemoryMappedSequence.from_file(path)
        assert seq[-1] == {"v": 4}
        assert seq[-2] == {"v": 3}
        with pytest.raises(IndexError):
            seq[-6]
        with pytest.raises(IndexError):
            seq[5]

    def test_reversed_slice(self, tmp_path):
        path = tmp_path / "data.mmap"
        write_items_to_file([{"v": i} for i in range(5)], path)
        seq = MemoryMappedSequence.from_file(path)
        assert [row["v"] for row in seq[::-1]] == [4, 3, 2, 1, 0]


class TestP2EngineValidation:
    def test_zero_epochs_raises(self, tmp_path):
        from autovision.engine import Engine

        engine = Engine()
        with pytest.raises(ValueError, match="max_epochs"):
            engine.fit(
                make_method(),
                dataloader=_EmptyLoader(),
                out_dir=tmp_path,
                max_epochs=0,
            )


class _EmptyLoader:
    dataset = [1, 2]
    batch_size = 1

    def __len__(self):
        return 2

    def __iter__(self):
        return iter([])


# ── visdrone_convergence_v5 review: eval-path coordinate mapping ─────────


def _tiny_detection_model():
    from autovision.tasks.detection.rtdetr.task_model import (
        RTDETRObjectDetection,
    )

    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=5,
        dec_layers=1, num_heads=4, num_top_queries=10,
        num_denoising=10, load_weights=False,
    )


def _perfect_outputs(boxes_px: torch.Tensor, labels: torch.Tensor, orig_wh):
    """Model outputs that decode exactly to ``boxes_px`` (original pixels)."""
    orig_w, orig_h = orig_wh
    cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2 / orig_w
    cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2 / orig_h
    w = (boxes_px[:, 2] - boxes_px[:, 0]) / orig_w
    h = (boxes_px[:, 3] - boxes_px[:, 1]) / orig_h
    boxes_norm = torch.stack([cx, cy, w, h], -1).unsqueeze(0)
    logits = torch.full((1, boxes_px.shape[0], 2), -8.0)
    logits[0, torch.arange(boxes_px.shape[0]), labels] = 6.0
    return logits, boxes_norm


class TestP0PostprocessCoordinateMapping:
    """P0: normalized pred boxes map to original pixels exactly once.

    The validation path used to rescale by (orig/input) *and* by the
    original size, double-scaling every box and collapsing the mAP to 0
    even for perfect predictions (visdrone_convergence_v5 finding).
    """

    ORIG = (136, 76)  # non-square, != network input
    GT_BOXES = torch.tensor([[10.0, 10.0, 32.0, 32.0], [50.0, 30.0, 90.0, 60.0]])
    GT_LABELS = torch.tensor([0, 1])

    def test_boxes_land_on_original_pixels(self):
        model = _tiny_detection_model().eval()
        logits, boxes_norm = _perfect_outputs(self.GT_BOXES, self.GT_LABELS, self.ORIG)
        # Legacy metadata (input_w/input_h present) must decode identically:
        # normalized coordinates are canvas-independent.
        meta = [{"orig_w": self.ORIG[0], "orig_h": self.ORIG[1],
                 "input_w": 64, "input_h": 64}]
        (result,) = model.postprocess((logits, boxes_norm), meta, threshold=0.05)
        order = torch.argsort(result["scores"], descending=True)
        assert torch.allclose(result["bboxes"][order], self.GT_BOXES.flip(0), atol=1e-4)
        assert torch.equal(result["labels"][order], self.GT_LABELS.flip(0))

    def test_predict_path_metadata_decodes_identically(self):
        model = _tiny_detection_model().eval()
        logits, boxes_norm = _perfect_outputs(self.GT_BOXES, self.GT_LABELS, self.ORIG)
        meta = [{"orig_w": self.ORIG[0], "orig_h": self.ORIG[1]}]
        (result,) = model.postprocess((logits, boxes_norm), meta, threshold=0.05)
        order = torch.argsort(result["scores"], descending=True)
        assert torch.allclose(result["bboxes"][order], self.GT_BOXES.flip(0), atol=1e-4)

    def test_perfect_val_path_scores_map_one(self):
        """End-to-end val-path sanity: perfect predictions → mAP 1.0."""
        from autovision.metrics.map_evaluator import COCOMAPEvaluator

        model = _tiny_detection_model().eval()
        evaluator = COCOMAPEvaluator({0: "cat", 1: "dog"})
        for index in range(3):
            orig_w, orig_h = 136 + 8 * index, 76 + 4 * index
            logits, boxes_norm = _perfect_outputs(
                self.GT_BOXES, self.GT_LABELS, (orig_w, orig_h)
            )
            (decoded,) = model.postprocess(
                (logits, boxes_norm),
                [{"orig_w": orig_w, "orig_h": orig_h}],
                threshold=0.05,
            )
            evaluator.update([decoded])
            evaluator.update_targets(
                [{"boxes": boxes_norm[0], "labels": self.GT_LABELS}],
                original_sizes=[(orig_w, orig_h)],
            )
        metrics = evaluator.compute()
        assert metrics["map"] == pytest.approx(1.0, abs=1e-6)


class TestP1EvalKeepsTinyGroundTruth:
    """The small-box filter is a train-time policy; eval must keep tiny GT."""

    def test_val_args_lift_min_bbox_size(self):
        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
            build_val_transform_args,
        )

        args = RTDETRDetectionTransformArgs(image_size=(32, 32), scale_jitter=None)
        val_args = build_val_transform_args(args)
        # torchvision's floor is min_size=1; the *training* threshold (4px)
        # must not silently delete tiny GT at eval time.
        assert val_args.min_bbox_size_px == 1.0
        assert args.min_bbox_size_px == 4.0  # train policy unchanged

        from torchvision import tv_tensors

        item = {
            "image": torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8),
            "bboxes": tv_tensors.BoundingBoxes(
                torch.tensor([[2.0, 2.0, 3.0, 3.0]]),  # 1px box: train would drop it
                format="XYXY",
                canvas_size=(24, 24),
            ),
            "classes": torch.tensor([0]),
            "image_path": "x.jpg",
            "original_size": (24, 24),
        }
        val_transform = RTDETRDetectionTransform(val_args)
        val_transform.set_step(0)
        out = val_transform(item)
        assert out["bboxes"].shape[0] == 1  # tiny GT survives val processing


class TestP1EffectiveBatchLrScaling:
    """The detection LR recipe scales with the *effective* batch, which
    includes gradient accumulation (v5 ran at sqrt(8/16) of the recipe LR)."""

    def _train_model(self):
        from autovision.tasks.detection.rtdetr.config import (
            RTDETRObjectDetectionTrainArgs,
        )
        from autovision.tasks.detection.rtdetr.train_model import (
            RTDETRObjectDetectionTrain,
        )

        args = RTDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=5, dec_layers=1, num_heads=4,
            num_top_queries=5, backbone_pretrained=False,
        )
        return RTDETRObjectDetectionTrain(args)

    def test_get_optimizer_uses_effective_batch_for_lr(self):
        wrapper = self._train_model()
        optimizer, _ = wrapper.get_optimizer(
            total_steps=100, global_batch_size=16, gradient_accumulation_steps=2
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        # 16 == default_batch_size → recipe LR untouched; accumulation was
        # already applied by the caller when computing global_batch_size.
        assert groups["detector"]["lr"] == pytest.approx(5e-4)
        assert groups["backbone"]["lr"] == pytest.approx(5e-4 * 0.05)

    def test_engine_passes_accumulated_batch(self, tmp_path, monkeypatch):
        from tests.test_train_object_detection import (
            data_config,
            make_coco_dataset,
            model_args,
            transform_args,
        )

        train_json, val_json, images_dir = make_coco_dataset(
            tmp_path, num_images=4
        )
        from autovision.tasks.detection.rtdetr.train_model import (
            RTDETRObjectDetectionTrain,
        )

        captured = {}
        original = RTDETRObjectDetectionTrain.get_optimizer

        def spy(self, total_steps, global_batch_size, **kwargs):
            captured["global_batch_size"] = global_batch_size
            return original(self, total_steps, global_batch_size, **kwargs)

        monkeypatch.setattr(RTDETRObjectDetectionTrain, "get_optimizer", spy)
        from autovision import train_object_detection

        train_object_detection(
            out=tmp_path / "det_accum",
            data=data_config(train_json, val_json, images_dir),
            steps=2,
            batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=0,
            model_args=model_args(),
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        # batch_size=2 × accumulation 2 × 1 device = 4.
        assert captured["global_batch_size"] == 4


class TestP2DetectionExampleRendering:
    """Normalized batches must un-normalize before the ">1.5 means raw
    0-255" heuristic can fire — otherwise the example renders as a flat
    mean-color image (bright normalized pixels reach ~2.6)."""

    def test_normalized_batch_renders_structure(self, tmp_path):
        from autovision.visualize.detection import save_detection_examples

        raw = torch.rand(3, 32, 32)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalized = (raw - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(
            std
        ).view(-1, 1, 1)
        batch = {
            "image": normalized.unsqueeze(0),
            "bboxes": [torch.tensor([[0.25, 0.25, 0.2, 0.2]])],
            "classes": [torch.tensor([1])],
            "original_size": [(32, 32)],
        }
        (path,) = save_detection_examples(
            tmp_path, batch, step=0,
            normalize_args={"mean": mean, "std": std},
        )
        rendered = torch.from_numpy(
            __import__("numpy").array(__import__("PIL.Image", fromlist=["Image"]).open(path))
        ).float()
        # A flat mean-color render has ~zero channel spread; the fixed
        # render preserves the random-texture spread of the source image.
        assert rendered.std() > 20.0
