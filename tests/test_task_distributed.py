# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for distributed task-training support (engine + metric gathering).

Multi-process end-to-end DDP runs need a torchrun launcher and are not part
of this suite (see README known limitations); these tests cover the
single-process building blocks the DDP path relies on.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine, _InfiniteCycleIterator
from autovision.metrics.map_evaluator import COCOMAPEvaluator
from autovision.tasks.train_model import TaskStepResult, TrainModel

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class _TinyTrainModel(TrainModel):
    task = "unit_test"

    def __init__(self) -> None:
        super().__init__(task_model=nn.Linear(4, 2))
        self.seen_batches = []

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        self.seen_batches.append(step)
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(loss=loss, log_dict={"unit/loss": float(loss)})


class _TensorDataset(Dataset):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.arange(4, dtype=torch.float32) + index


class _FakeSampler:
    def __init__(self) -> None:
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class TestTrainModelForwardAlias:
    def test_forward_delegates_to_training_step(self):
        model = _TinyTrainModel()
        result = model({"x": torch.randn(2, 4)}, step=3)
        assert isinstance(result, TaskStepResult)
        assert model.seen_batches == [3]
        assert result.loss.ndim == 0

    def test_watch_metric_defaults_to_none(self):
        assert _TinyTrainModel.watch_metric is None
        from autovision.tasks.detection.rtdetr import (
            RTDETRObjectDetectionTrain,
        )

        assert RTDETRObjectDetectionTrain.watch_metric == "val_metric/map"


class TestMetricStateGather:
    def test_export_merge_roundtrip(self):
        categories = {1: "cat", 2: "dog"}
        first = COCOMAPEvaluator(categories)
        first.update(
            [
                {
                    "bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]
        )
        first.update_targets(
            [{"boxes": torch.tensor([[0.1, 0.1, 0.4, 0.4]]), "labels": torch.tensor([0])}],
            original_sizes=[(100, 100)],
        )
        second = COCOMAPEvaluator(categories)
        second.merge_state(first.export_state())
        assert len(second._predictions) == 1
        assert len(second._ground_truth) == 1
        # The evaluator states are independent after merging.
        first.reset()
        assert len(second._predictions) == 1

    def test_base_train_model_metric_state_is_none(self):
        model = _TinyTrainModel()
        assert model.export_metric_state("val") is None
        model.merge_metric_state("val", None)  # no-op


class TestInfiniteCycleSampler:
    def test_sampler_epoch_advances_on_restart(self):
        loader = DataLoader(_TensorDataset(4), batch_size=2)
        sampler = _FakeSampler()
        iterator = _InfiniteCycleIterator(loader, sampler=sampler)
        batches = [next(iterator) for _ in range(4)]  # two full passes
        assert len(batches) == 4
        assert sampler.epochs == [1]  # one natural restart

    def test_reset_advances_sampler_epoch(self):
        loader = DataLoader(_TensorDataset(4), batch_size=2)
        sampler = _FakeSampler()
        iterator = _InfiniteCycleIterator(loader, sampler=sampler)
        next(iterator)
        iterator.reset()
        iterator.reset()
        assert sampler.epochs == [1, 2]

    def test_without_sampler_no_epoch_tracking(self):
        loader = DataLoader(_TensorDataset(4), batch_size=2)
        iterator = _InfiniteCycleIterator(loader)
        assert [next(iterator) for _ in range(4)]
        iterator.reset()
        assert iterator._epoch == 0  # nothing tracked without a sampler


class TestWithSampler:
    def test_rebuilt_loader_preserves_contract(self):
        class _XY(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return torch.arange(4, dtype=torch.float32) + index, index

        def _collate(batch):
            xs, ys = zip(*batch)
            return {"x": torch.stack(xs), "y": torch.tensor(ys)}

        loader = DataLoader(
            _XY(), batch_size=2, shuffle=True, num_workers=0, collate_fn=_collate,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            loader.dataset, shuffle=True, drop_last=True, num_replicas=1, rank=0
        )
        rebuilt = TaskEngine._with_sampler(loader, sampler)
        assert rebuilt.batch_size == loader.batch_size
        assert rebuilt.collate_fn is _collate
        # With a sampler, the loader's own shuffle flag is always False —
        # the sampler drives the order.
        assert rebuilt.sampler is sampler
        sampler.set_epoch(0)
        batch = next(iter(rebuilt))
        assert batch["x"].shape == (2, 4)


class TestTaskEngineAcceptsDistributed:
    def test_no_early_refusal_without_distributed_env(self, tmp_path):
        """fit_task must no longer refuse WORLD_SIZE>1 up front (DDP is set
        up later, after device resolution)."""
        import inspect

        source = inspect.getsource(TaskEngine.fit_task)
        assert "Distributed task training is not supported" not in source
