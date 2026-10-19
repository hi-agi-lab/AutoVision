# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the cross-model distillation branch (design doc 23 §8).

Hermetic, CPU-only: criterion-level cases run on synthetic tensors with
hand-built union matches, wrapper-level cases on the tiny
``dinov2-vittest-notpretrained`` variant (the test conventions of
``tests/test_detr_ssl.py``).
"""


import pytest
import torch
import torch.nn.functional as F
from tests.test_detr_ssl import VARIANT_ARGS

from autovision.tasks.detection.detr import (
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.detr.config import DETRDistillArgs
from autovision.tasks.detection.detr.train_model import (
    _build_gt_to_teacher_query,
)
from autovision.tasks.detection.shared.dfine_criterion import (
    DFINECriterionV2,
)
from autovision.tasks.detection.shared.distillation import (
    DistillParams,
    cross_model_ddf_loss,
    cross_model_distill_losses,
)
from autovision.tasks.detection.shared.matcher import (
    HungarianMatcher,
)

REG_MAX = 8
B, Q, C = 2, 20, 2


# ── fixtures ────────────────────────────────────────────────────────────
def _harness(seed: int):
    torch.manual_seed(seed)
    targets = [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])},
        {"boxes": torch.rand(2, 4) * 0.3 + 0.3, "labels": torch.tensor([1, 0])},
    ]
    outputs = [
        {"pred_logits": torch.randn(B, Q, C), "pred_boxes": torch.rand(B, Q, 4) * 0.6 + 0.2}
        for _ in range(2)
    ]
    corners = [torch.randn(B, Q, 4 * (REG_MAX + 1)) for _ in range(2)]
    refs = [torch.rand(B, Q, 4) * 0.6 + 0.2 for _ in range(2)]
    return targets, outputs, corners, refs


def _teacher_payload(harness, coverage="full"):
    """Teacher payload over a hand-built GT→teacher-query map.

    ``full`` covers every GT (teacher query = GT index, well inside Q);
    ``miss`` leaves image 0's last GT unmatched (a teacher false negative).
    """
    targets = harness[0]
    b_size = len(targets)
    max_gt = max(int(t["boxes"].shape[0]) for t in targets)
    t_map = torch.full((b_size, max_gt), -1, dtype=torch.long)
    for b, t in enumerate(targets):
        limit = int(t["boxes"].shape[0])
        if coverage == "miss" and b == 0:
            limit -= 1
        for gt_index in range(max(limit, 0)):
            t_map[b, gt_index] = gt_index % Q
    return {
        "teacher_logits": harness[1][-1]["pred_logits"],
        "teacher_boxes": harness[1][-1]["pred_boxes"],
        "teacher_corners": harness[2][-1],
        "gt_to_teacher_query": t_map,
    }


def _run_criterion(harness, distill_params=None, distill_targets=None, train=True):
    targets, outputs, corners, refs = harness
    criterion = DFINECriterionV2(
        HungarianMatcher(), reg_max=REG_MAX, distill_params=distill_params
    )
    criterion.train(train)
    kwargs = {"distill_targets": distill_targets} if distill_params is not None else {}
    losses = criterion(
        outputs, targets, world_size=1,
        corners_list=corners, ref_boxes_list=refs, **kwargs,
    )
    return criterion, losses


def make_distill_train_model(**distill_overrides) -> DETRObjectDetectionTrain:
    distill = {"teacher_model": "self:ema", "start_step": 0}
    distill.update(distill_overrides)
    args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS, distill=distill)
    return DETRObjectDetectionTrain(args)


def make_single_view_batch(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    images, batch_boxes, batch_classes = [], [], []
    for _ in range(2):
        n = int(torch.randint(1, 4, (1,), generator=generator))
        batch_boxes.append(torch.rand(n, 4, generator=generator) * 0.3 + 0.3)
        batch_classes.append(torch.randint(0, 2, (n,), generator=generator))
        images.append(torch.randn(3, 224, 224, generator=generator))
    return {"image": torch.stack(images), "bboxes": batch_boxes, "classes": batch_classes}


def _export_payload(wrapper: DETRObjectDetectionTrain) -> dict:
    return {
        "schema_version": 1,
        "kind": "task_model",
        "model_class_path": (
            f"{type(wrapper.model).__module__}.{type(wrapper.model).__name__}"
        ),
        "model_init_args": dict(wrapper.model._init_args),
        "train_model": wrapper.get_export_state_dict(),
    }


@pytest.fixture(scope="module")
def distill_train_model() -> DETRObjectDetectionTrain:
    return make_distill_train_model()


@pytest.fixture(scope="module")
def baseline_train_model() -> DETRObjectDetectionTrain:
    return DETRObjectDetectionTrain(DETRObjectDetectionTrainArgs(**VARIANT_ARGS))


# ── config ──────────────────────────────────────────────────────────────
class TestDistillConfig:
    def test_off_by_default(self):
        args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS)
        assert args.distill is None

    def test_empty_teacher_source_rejected(self):
        with pytest.raises(ValueError, match="teacher_checkpoint"):
            DETRDistillArgs(teacher_model="")

    def test_nonpositive_temperature_rejected(self):
        with pytest.raises(ValueError, match="ddf_temperature"):
            DETRDistillArgs(teacher_model="self:ema", ddf_temperature=0.0)

    def test_class_space_rules(self):
        args = DETRDistillArgs(teacher_model="self:ema", cls_coef=0.7)
        assert args.resolved_cls_coef(2, 2) == 0.7
        with pytest.raises(ValueError, match="require_same_class_space"):
            args.resolved_cls_coef(5, 2)
        relaxed = DETRDistillArgs(
            teacher_model="self:ema", require_same_class_space=False
        )
        assert relaxed.resolved_cls_coef(5, 2) == 0.0

    def test_args_round_trip(self):
        args = DETRObjectDetectionTrainArgs(
            **VARIANT_ARGS,
            distill={"teacher_model": "self:ema", "cls_coef": 0.5},
        )
        dumped = args.model_dump()["distill"]
        re_coerced = DETRDistillArgs(**dumped)
        assert re_coerced.cls_coef == 0.5
        assert re_coerced.teacher_model == "self:ema"


# ── criterion-level legs ────────────────────────────────────────────────
class TestDistillLegs:
    def test_legs_present_and_finite(self):
        harness = _harness(0)
        params = DistillParams()
        criterion, losses = _run_criterion(harness, params, _teacher_payload(harness))
        for key in ("loss_distill_cls", "loss_distill_box", "loss_distill_ddf"):
            assert key in losses
            assert torch.isfinite(losses[key])
        for key in ("loss_distill_cls", "loss_distill_box", "loss_distill_ddf"):
            assert key in criterion.weight_dict
        assert criterion.last_distill_stats["union"] > 0
        assert criterion.last_distill_stats["covered"] == criterion.last_distill_stats["union"]

    def test_cls_leg_hand_computed(self):
        """Direct call with a hand-built union match and exact BCE math."""
        torch.manual_seed(1)
        targets = [
            {"boxes": torch.rand(2, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1])},
            {"boxes": torch.rand(1, 4) * 0.3 + 0.3, "labels": torch.tensor([1])},
        ]
        final = {
            "pred_logits": torch.randn(B, Q, C),
            "pred_boxes": torch.rand(B, Q, 4) * 0.6 + 0.2,
        }
        indices = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.tensor([2]), torch.tensor([0])),
        ]
        t_map = torch.tensor([[0, 1], [3, -1]])
        teacher_logits = torch.randn(B, Q, C)
        payload = {
            "teacher_logits": teacher_logits,
            "teacher_boxes": final["pred_boxes"],
            "teacher_corners": None,
            "gt_to_teacher_query": t_map,
        }
        params = DistillParams(cls_coef=2.0, box_coef=0.0, ddf_coef=0.0)
        stats: dict = {}
        losses = cross_model_distill_losses(
            final, None, targets, indices, torch.tensor(3.0), payload, params, stats
        )
        pairs = [(0, 0, 0, 0), (0, 1, 1, 1), (1, 2, 0, 3)]  # (b, q, gt, t_q)
        expected_terms = []
        for b, q, gt, t_q in pairs:
            label = targets[b]["labels"][gt]
            z = final["pred_logits"][b, q, label]
            p = torch.sigmoid(teacher_logits[b, t_q, label]).clamp(0.01, 0.99)
            expected_terms.append(F.binary_cross_entropy_with_logits(z, p))
        expected = torch.stack(expected_terms).sum() / 3.0
        assert stats == {"union": 3, "covered": 3}
        torch.testing.assert_close(losses["loss_distill_cls"], expected, rtol=1e-6, atol=1e-7)

    def test_teacher_misses_are_skipped(self):
        harness = _harness(2)
        criterion, losses = _run_criterion(
            harness, DistillParams(), _teacher_payload(harness, coverage="miss")
        )
        stats = criterion.last_distill_stats
        assert stats["covered"] < stats["union"]
        for key in ("loss_distill_cls", "loss_distill_box", "loss_distill_ddf"):
            assert torch.isfinite(losses[key])

    def test_uncovered_teacher_connected_zero(self):
        harness = _harness(3)
        payload = _teacher_payload(harness)
        payload["gt_to_teacher_query"] = torch.full_like(
            payload["gt_to_teacher_query"], -1
        )
        criterion, losses = _run_criterion(harness, DistillParams(), payload)
        assert criterion.last_distill_stats["covered"] == 0
        for key in ("loss_distill_cls", "loss_distill_box", "loss_distill_ddf"):
            assert float(losses[key]) == 0.0

    def test_teacher_corners_absent_drops_only_ddf(self):
        harness = _harness(4)
        payload = _teacher_payload(harness)
        payload["teacher_corners"] = None
        criterion, losses = _run_criterion(harness, DistillParams(), payload)
        assert "loss_distill_ddf" not in losses
        assert "loss_distill_cls" in losses
        assert "loss_distill_box" in losses

    def test_ddf_self_zero(self):
        """Teacher corners identical to the student's → the KL is exactly 0."""
        torch.manual_seed(4)
        corners = torch.randn(6, 4 * (REG_MAX + 1))
        quality = torch.rand(6)
        loss = cross_model_ddf_loss(corners, corners.clone(), quality, temperature=5.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_gt_to_teacher_query_map(self):
        device = torch.device("cpu")
        indices = [
            (torch.tensor([5]), torch.tensor([1])),
            (torch.tensor([]), torch.tensor([])),
        ]
        targets = [{"boxes": torch.rand(2, 4)}, {"boxes": torch.rand(1, 4)}]
        t_map = _build_gt_to_teacher_query(indices, targets, device)
        assert t_map.shape == (2, 2)
        assert t_map[0, 1].item() == 5
        assert t_map[0, 0].item() == -1
        assert t_map[1].tolist() == [-1, -1]

    def test_gated_out_step(self):
        harness = _harness(6)
        criterion, losses = _run_criterion(harness, DistillParams(), {"gated_out": True})
        assert not any("distill" in key for key in losses)
        assert criterion.last_distill_stats is None

    def test_missing_targets_raise_in_training(self):
        harness = _harness(7)
        with pytest.raises(ValueError, match="distill_targets"):
            _run_criterion(harness, DistillParams(), None)

    def test_eval_never_enters_distill(self):
        harness = _harness(8)
        criterion, losses = _run_criterion(harness, DistillParams(), None, train=False)
        assert not any("distill" in key for key in losses)
        assert criterion.last_distill_stats is None


# ── wrapper-level ───────────────────────────────────────────────────────
class TestDistillWrapper:
    def test_off_by_default(self):
        args = DETRObjectDetectionTrainArgs(**VARIANT_ARGS)
        assert args.distill is None

    def test_self_ema_teacher_aliases_ema_copy(
        self, distill_train_model, baseline_train_model
    ):
        teacher = distill_train_model._distill_teacher
        assert teacher is distill_train_model.ema_model.model
        assert not any(p.requires_grad for p in teacher.parameters())
        # The teacher is an alias of an already-registered module (and the
        # matcher holds no parameters): no new state_dict keys may appear.
        assert set(distill_train_model.state_dict().keys()) == set(
            baseline_train_model.state_dict().keys()
        )

    def test_training_step_runs_distill_legs(self, distill_train_model):
        result = distill_train_model.training_step(make_single_view_batch(), step=0)
        assert torch.isfinite(result.loss)
        assert any("loss_distill_cls" in key for key in result.log_dict)
        assert 0.0 <= result.log_dict["distill_coverage"] <= 1.0

    def test_start_step_gate(self):
        wrapper = make_distill_train_model(start_step=5)
        result = wrapper.training_step(make_single_view_batch(), step=0)
        assert not any("loss_distill" in key for key in result.log_dict)
        assert wrapper.criterion.last_distill_stats is None

    def test_every_n_steps_gate(self):
        wrapper = make_distill_train_model(every_n_steps=2)
        wrapper.training_step(make_single_view_batch(), step=1)
        assert wrapper.criterion.last_distill_stats is None
        wrapper.training_step(make_single_view_batch(), step=2)
        assert wrapper.criterion.last_distill_stats is not None

    def test_exported_checkpoint_teacher(self, baseline_train_model, tmp_path):
        checkpoint = tmp_path / "exported_teacher.pt"
        torch.save(_export_payload(baseline_train_model), checkpoint)
        wrapper = make_distill_train_model(
            teacher_model="", teacher_checkpoint=checkpoint
        )
        teacher = wrapper._distill_teacher
        assert teacher is not None
        assert not any(p.requires_grad for p in teacher.parameters())
        # Teacher weights must be exactly the exported EMA weights.
        exported = baseline_train_model.get_export_state_dict()
        ema_state = {
            key[len("ema_model."):]: value
            for key, value in exported.items()
            if key.startswith("ema_model.")
        }
        assert set(teacher.state_dict().keys()) == set(ema_state.keys())

    def test_class_space_mismatch_rejected(self, baseline_train_model, tmp_path):
        three_class = DETRObjectDetectionTrain(
            DETRObjectDetectionTrainArgs(**{**VARIANT_ARGS, "num_classes": 3})
        )
        checkpoint = tmp_path / "teacher3.pt"
        torch.save(_export_payload(three_class), checkpoint)
        student_args = DETRObjectDetectionTrainArgs(
            **VARIANT_ARGS,
            distill={"teacher_model": "", "teacher_checkpoint": checkpoint},
        )
        with pytest.raises(ValueError, match="require_same_class_space"):
            DETRObjectDetectionTrain(student_args)

    def test_class_space_relaxed_drops_cls_leg(self, baseline_train_model, tmp_path):
        three_class = DETRObjectDetectionTrain(
            DETRObjectDetectionTrainArgs(**{**VARIANT_ARGS, "num_classes": 3})
        )
        checkpoint = tmp_path / "teacher3.pt"
        torch.save(_export_payload(three_class), checkpoint)
        wrapper = make_distill_train_model(
            teacher_model="",
            teacher_checkpoint=checkpoint,
            require_same_class_space=False,
        )
        assert wrapper.criterion.distill_params.cls_coef == 0.0
        assert "loss_distill_cls" not in wrapper.criterion.weight_dict
        assert wrapper.criterion.weight_dict["loss_distill_box"] > 0

    def test_self_ema_requires_ema(self):
        args = DETRObjectDetectionTrainArgs(
            **VARIANT_ARGS,
            use_ema_model=False,
            distill={"teacher_model": "self:ema"},
        )
        with pytest.raises(ValueError, match="use_ema_model"):
            DETRObjectDetectionTrain(args)

    def test_decoder_gate(self):
        args = DETRObjectDetectionTrainArgs(
            **{**VARIANT_ARGS, "decoder": "rtdetrv2"},
            distill={"teacher_model": "self:ema"},
        )
        with pytest.raises(ValueError, match="D-FINE decoder"):
            DETRObjectDetectionTrain(args)

    def test_teacher_not_in_export(self, distill_train_model):
        export = distill_train_model.get_export_state_dict()
        assert not any("_distill_teacher" in key for key in export)

    def test_public_weight_teacher_not_implemented(self):
        args = DETRObjectDetectionTrainArgs(
            **VARIANT_ARGS,
            distill={"teacher_model": "deimv2/deimv2-dinov3-s"},
        )
        with pytest.raises(NotImplementedError, match="teacher_model"):
            DETRObjectDetectionTrain(args)


class TestLegacyTeacherCheckpointGate:
    def test_legacy_export_requires_explicit_trust(self, baseline_train_model, tmp_path, monkeypatch):
        import pathlib

        from autovision.errors import CheckpointError
        from autovision.tasks.detection.detr.train_model import (
            _load_teacher_from_export,
        )

        payload = _export_payload(baseline_train_model)
        # Simulate a legacy export: a non-safe-global object makes
        # weights_only=True fail, exactly like the old __class__ cell.
        payload["legacy_marker"] = pathlib.Path(".")
        checkpoint = tmp_path / "legacy_teacher.pt"
        torch.save(payload, checkpoint)

        with pytest.raises(CheckpointError, match="TRUST_TEACHER"):
            _load_teacher_from_export(checkpoint)

        monkeypatch.setenv("AUTOVISION_TRUST_TEACHER_CHECKPOINT", "1")
        teacher = _load_teacher_from_export(checkpoint)
        assert isinstance(teacher, torch.nn.Module)
