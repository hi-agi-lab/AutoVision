# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""fp16 loss-scaling contract: build, checkpoint, resume, and GPU smoke."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from autovision.engine.engine import Engine, TaskEngine, _build_grad_scaler
from autovision.runtime.checkpoint import Checkpoint
from autovision.tasks.train_model import TaskStepResult, TrainModel


class _Method(nn.Linear):
    """Minimal training method with the checkpoint hooks the engine calls."""

    def __init__(self) -> None:
        super().__init__(4, 2)
        self.loaded_checkpoint = None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        pass

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.loaded_checkpoint = checkpoint


class _FakeScaler:
    def __init__(self) -> None:
        self.state = {"scale": torch.tensor(128.0)}
        self.loaded = None

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


def _checkpoint_inputs():
    method = _Method()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return method, optimizer, scheduler


class TestGradScalerSelection:
    def test_fp32_and_bf16_do_not_need_scaling(self):
        assert _build_grad_scaler("32-true", "cuda") is None
        assert _build_grad_scaler("bf16-mixed", "cuda") is None
        assert _build_grad_scaler("16-mixed", "cpu") is None

    def test_fp16_accelerator_builds_a_scaler(self):
        # On a CPU-only build torch returns a disabled scaler instead of
        # raising; the engine still stores its (empty) state for symmetry.
        scaler = _build_grad_scaler("16-mixed", "cuda")
        assert scaler is not None


class TestScalerCheckpoint:
    def test_save_checkpoint_persists_scaler_state(self, tmp_path):
        method, optimizer, scheduler = _checkpoint_inputs()
        scaler = _FakeScaler()
        path = tmp_path / "checkpoints" / "last.ckpt"
        Engine.save_checkpoint(
            path,
            method=method,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=3,
            scaler=scaler,
        )
        ckpt = Checkpoint.from_path(path)
        assert ckpt.autovision.engine_state["scaler"]["scale"].item() == 128.0

    def test_resume_restores_scaler_state(self, tmp_path):
        method, optimizer, scheduler = _checkpoint_inputs()
        path = tmp_path / "checkpoints" / "last.ckpt"
        Engine.save_checkpoint(
            path,
            method=method,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=3,
            scaler=_FakeScaler(),
        )
        restored = _FakeScaler()
        start_epoch, step = Engine._resume(
            path, method, optimizer, scheduler, torch.device("cpu"),
            scaler=restored,
        )
        assert start_epoch == 1
        assert step == 3
        assert restored.loaded["scale"].item() == 128.0

    def test_old_checkpoints_without_scaler_are_still_loadable(self, tmp_path):
        """Backward compatibility: checkpoints written before this change
        carry no ``scaler`` key and must resume unchanged."""
        method, optimizer, scheduler = _checkpoint_inputs()
        path = tmp_path / "checkpoints" / "last.ckpt"
        Engine.save_checkpoint(
            path,
            method=method,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=1,
        )
        restored = _FakeScaler()
        Engine._resume(
            path, method, optimizer, scheduler, torch.device("cpu"),
            scaler=restored,
        )
        assert restored.loaded is None


class _ProbeTrainModel(TrainModel):
    """Tiny task model exercising the task-engine fp16 path on CUDA."""

    task = "unit_test"

    def __init__(self) -> None:
        super().__init__(task_model=nn.Linear(4, 2))

    def training_step(self, batch: dict, step: int):
        return TaskStepResult(loss=self.model(batch["x"]).sum(), log_dict={})

    def validation_step(self, batch: dict, step: int):
        return TaskStepResult(loss=torch.zeros(()), log_dict={})

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


class _TensorDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


_test_cuda = torch.cuda.is_available()


class TestFp16TrainingSmoke:
    def test_task_fit_writes_scaler_state(self, tmp_path):
        if not _test_cuda:
            import pytest

            pytest.skip("fp16 autocast needs CUDA")

        model = _ProbeTrainModel()
        loader = DataLoader(_TensorDataset(), batch_size=2)
        TaskEngine().fit_task(
            model,
            loader,
            None,
            out_dir=tmp_path,
            steps=1,
            save_last=True,
            save_every_num_steps=1,
            devices="cuda",
            precision="16-mixed",
            visualize_examples=False,
        )
        ckpt = Checkpoint.from_path(tmp_path / "checkpoints" / "last.ckpt")
        assert "scaler" in ckpt.autovision.engine_state
