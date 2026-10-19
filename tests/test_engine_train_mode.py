# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Regression test: the engine must force train mode UP FRONT.

HF ``from_pretrained``-built task models start in eval mode, and the only
other ``train()`` call in the fit loop sits at the end of the validation
block — so before the fix, every step before the FIRST validation ran in
eval mode (no denoising groups, truncated aux losses, frozen BN stats).
Observed live as the DEIMv2 baseline's train_loss jumping 3.8 → 25.6 and
its step time 0.34 → 0.53 s exactly at the first validation (design doc
22 v1.2 analysis, 2026-10-11).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.tasks.train_model import TaskStepResult, TrainModel


class _ModeProbeTrainModel(TrainModel):
    """Records the module's train/eval state during each training step."""

    task = "unit_test"

    def __init__(self, start_in_eval: bool) -> None:
        super().__init__(task_model=nn.Linear(4, 2))
        self.seen_modes: list[bool] = []
        if start_in_eval:
            # Mirror the HF from_pretrained contract: deserialized models
            # come back in eval mode.
            self.eval()

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        self.seen_modes.append(self.model.training)
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(loss=loss, log_dict={})

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
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


class _TensorDataset(Dataset):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


def _fit_two_steps(train_model: _ModeProbeTrainModel, tmp_path) -> None:
    engine = TaskEngine()
    loader = DataLoader(_TensorDataset(4), batch_size=2)
    engine.fit_task(
        train_model,
        loader,
        loader,
        out_dir=tmp_path,
        steps=2,
        val_every_num_steps=2,
        save_every_num_steps=100,
        save_last=False,
        save_best=False,
        devices="cpu",
        visualize_examples=False,
    )


class TestTrainModeUpFront:
    def test_eval_initialized_model_trains_in_train_mode(self, tmp_path):
        """A from_pretrained-style (eval-mode) model must see train mode at
        step 0 — BEFORE any validation round-trip has run."""
        model = _ModeProbeTrainModel(start_in_eval=True)
        _fit_two_steps(model, tmp_path)
        assert model.seen_modes, "training steps did not run"
        assert all(model.seen_modes), (
            f"training steps ran in eval mode: {model.seen_modes}"
        )

    def test_constructed_model_unaffected(self, tmp_path):
        """Pure nn.Module constructors (train by default) stay on the
        fixed path — no regression for the AV-DETR style families."""
        model = _ModeProbeTrainModel(start_in_eval=False)
        _fit_two_steps(model, tmp_path)
        assert all(model.seen_modes)
