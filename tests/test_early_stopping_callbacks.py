# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Early stopping and trainer callback contracts for task training."""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.tasks.train_model import TaskStepResult, TrainModel


class _MetricModel(TrainModel):
    task = "unit_test"
    watch_metric = "val_metric/map"

    def __init__(self, values: list[float]) -> None:
        super().__init__(task_model=nn.Linear(4, 2))
        self.values = values
        self.validation_calls = 0

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(
            loss=self.model(batch["x"]).sum(),
            log_dict={"loss_component": torch.tensor(1.0)},
        )

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def compute_metrics(self, split: str) -> dict[str, float]:
        value = self.values[min(self.validation_calls, len(self.values) - 1)]
        self.validation_calls += 1
        return {"val_metric/map": value}

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0
        )


class _Dataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_train_start(self, steps, out_dir):
        self.events.append(("start", steps))

    def on_step_end(self, step, log_dict):
        self.events.append(("step", step))

    def on_validation_end(self, step, metrics):
        self.events.append(("validation", step, metrics["val_metric/map"]))

    def on_early_stop(self, step, best_metric, patience):
        self.events.append(("early", step, best_metric, patience))

    def on_train_end(self, result):
        self.events.append(("end", result.stopped_early))


def _fit(model, tmp_path, **kwargs):
    loader = DataLoader(_Dataset(), batch_size=2)
    return TaskEngine().fit_task(
        model,
        loader,
        loader,
        out_dir=tmp_path,
        steps=kwargs.pop("steps", 6),
        val_every_num_steps=kwargs.pop("val_every_num_steps", 1),
        save_every_num_steps=100,
        save_last=False,
        save_best=False,
        devices="cpu",
        visualize_examples=False,
        **kwargs,
    )


class TestEarlyStopping:
    def test_stops_after_patience_without_improvement(self, tmp_path):
        result = _fit(
            _MetricModel([0.5] * 10), tmp_path,
            early_stopping_patience=2,
        )
        assert result.stopped_early is True
        assert result.stop_step == 3
        assert result.steps_run == 3

    def test_improvement_resets_the_counter(self, tmp_path):
        result = _fit(
            _MetricModel([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]), tmp_path,
            steps=8, early_stopping_patience=2,
        )
        assert result.stopped_early is True
        assert result.steps_run == 6

    def test_disabled_patience_runs_all_steps(self, tmp_path):
        result = _fit(
            _MetricModel([0.5] * 10), tmp_path,
            steps=4, early_stopping_patience=None,
        )
        assert result.stopped_early is False
        assert result.stop_step is None
        assert result.steps_run == 4

    def test_requires_periodic_validation(self, tmp_path):
        model = _MetricModel([0.5])
        with pytest.raises(ValueError, match="val_dataloader"):
            TaskEngine().fit_task(
                model,
                DataLoader(_Dataset(), batch_size=2),
                None,
                out_dir=tmp_path,
                steps=2,
                devices="cpu",
                early_stopping_patience=1,
            )


class TestCallbacks:
    def test_hooks_fire_in_order(self, tmp_path):
        recorder = _Recorder()
        result = _fit(
            _MetricModel([0.5] * 10), tmp_path,
            early_stopping_patience=2, callbacks=[recorder],
        )
        names = [event[0] for event in recorder.events]
        assert names[0] == "start"
        assert names[-1] == "end"
        assert names.count("validation") == 3
        assert "early" in names
        assert names.index("early") < names.index("end")
        assert result.stopped_early is True

    def test_callback_exception_does_not_abort_training(self, tmp_path, caplog):
        class _Broken:
            def on_step_end(self, step, log_dict):
                raise RuntimeError("callback exploded")

        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = _fit(
                _MetricModel([0.5] * 10), tmp_path,
                steps=2, callbacks=[_Broken()],
            )
        assert result.steps_run == 2
        assert "callback" in caplog.text.lower()

    def test_callback_without_a_hook_is_ignored(self, tmp_path):
        result = _fit(
            _MetricModel([0.5] * 10), tmp_path,
            steps=2, callbacks=[object()],
        )
        assert result.steps_run == 2


class TestEngineRegressions:
    def test_steps_run_counts_steps_without_validation(self, tmp_path):
        loader = DataLoader(_Dataset(), batch_size=2)
        result = TaskEngine().fit_task(
            _MetricModel([0.5]),
            loader,
            None,
            out_dir=tmp_path,
            steps=3,
            devices="cpu",
            save_last=False,
            save_best=False,
            visualize_examples=False,
        )
        assert result.steps_run == 3

    def test_rejects_non_positive_gradient_accumulation(self, tmp_path):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            TaskEngine().fit_task(
                _MetricModel([0.5]),
                DataLoader(_Dataset(), batch_size=2),
                None,
                out_dir=tmp_path,
                steps=1,
                gradient_accumulation_steps=0,
                devices="cpu",
                save_last=False,
                save_best=False,
                visualize_examples=False,
            )

    def test_save_last_false_writes_no_last_checkpoint(self, tmp_path):
        result = _fit(
            _MetricModel([0.5] * 10), tmp_path,
            steps=2,
        )
        assert result.steps_run == 2
        assert not (tmp_path / "checkpoints" / "last.ckpt").exists()
        assert not (tmp_path / "exported_models" / "exported_last.pt").exists()

    def test_logger_failure_does_not_abort_training(self, tmp_path, caplog):
        def broken_logger(metrics, step):
            raise RuntimeError("tracking backend down")

        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = _fit(
                _MetricModel([0.5] * 10), tmp_path,
                steps=2, log_every_n_steps=1, logger_fn=broken_logger,
            )
        assert result.steps_run == 2
        assert "logger_fn failed" in caplog.text

    def test_out_dir_accepts_string(self, tmp_path):
        result = _fit(_MetricModel([0.5] * 10), str(tmp_path), steps=1)
        assert result.out_dir == tmp_path

    def test_set_accelerator_device_selects_index(self, monkeypatch):
        from autovision.engine.engine import _set_accelerator_device

        calls = []
        monkeypatch.setattr(torch.cuda, "set_device", calls.append)
        _set_accelerator_device(torch.device("cuda", 1))
        assert calls == [torch.device("cuda", 1)]


class TestGradientCheckpointingFlag:
    def test_flag_is_accepted_and_degrades_for_models_without_blocks(
        self, tmp_path, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = _fit(
                _MetricModel([0.5] * 10), tmp_path,
                steps=2, gradient_checkpointing=True,
            )
        assert result.steps_run == 2
        assert "no checkpointable blocks" in caplog.text


class TestParallelStrategyValidation:
    def test_unsupported_strategies_fail_loud(self, tmp_path):
        with pytest.raises(NotImplementedError, match="DeepSpeed"):
            TaskEngine().fit_task(
                _MetricModel([0.5]), DataLoader(_Dataset(), batch_size=2), None,
                out_dir=tmp_path, steps=1, devices="cpu",
                parallel="deepspeed", visualize_examples=False,
            )

    def test_fsdp_requires_multiple_ranks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "1")
        with pytest.raises(ValueError, match="WORLD_SIZE"):
            TaskEngine().fit_task(
                _MetricModel([0.5]), DataLoader(_Dataset(), batch_size=2), None,
                out_dir=tmp_path, steps=1, devices="cpu",
                parallel="fsdp", visualize_examples=False,
            )
