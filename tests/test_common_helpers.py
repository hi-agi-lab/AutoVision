# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest

from autovision.cli.common_helpers import (
    get_global_batch_size,
    get_num_workers,
    get_out_dir,
)


class TestGetOutDir:
    def test_creates_dir(self, tmp_path):
        out = tmp_path / "run"
        assert get_out_dir(out) == out
        assert out.is_dir()

    def test_existing_empty_ok(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        get_out_dir(out)

    def test_existing_nonempty_raises(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        (out / "file.txt").write_text("x")
        with pytest.raises(ValueError, match="not empty"):
            get_out_dir(out)

    def test_overwrite_replaces_nonempty(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        (out / "file.txt").write_text("x")
        get_out_dir(out, overwrite=True)
        assert out.is_dir()
        assert not (out / "file.txt").exists()  # stale artifacts are wiped

    def test_resume_keeps_previous_contents(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        (out / "file.txt").write_text("x")
        get_out_dir(out, resume_interrupted=True)
        assert (out / "file.txt").exists()  # resume must keep last.ckpt etc.

    def test_resume_tolerates_nonempty(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        (out / "file.txt").write_text("x")
        get_out_dir(out, resume_interrupted=True)


class TestGetGlobalBatchSize:
    def test_passthrough(self):
        assert get_global_batch_size(32, 1000, 1) == 32
        assert get_global_batch_size(32, 1000, 4) == 32

    def test_clamps_to_dataset_size(self):
        assert get_global_batch_size(128, 64, 1) == 64

    def test_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            get_global_batch_size(10, 1000, 4)

    def test_batch_one_device_ok(self):
        assert get_global_batch_size(1, 1000, 1) == 1


class TestGetNumWorkers:
    def test_explicit(self):
        assert get_num_workers(3, 1) == 3

    def test_auto_single_device(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        workers = get_num_workers("auto", 1)
        assert 0 <= workers <= 8

    def test_auto_slurm(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        assert get_num_workers("auto", 1) == 5  # cpus - 1, below the cap

    def test_auto_slurm_capped(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        assert get_num_workers("auto", 1) == 8  # capped by MAX_NUM_WORKERS_AUTO


def test_resolve_gradient_checkpointing():
    from autovision.cli.common_helpers import resolve_gradient_checkpointing

    assert resolve_gradient_checkpointing(None) is False
    assert resolve_gradient_checkpointing({"gradient_checkpointing": True}) is True
    with pytest.raises(ValueError, match="must be a bool"):
        resolve_gradient_checkpointing({"gradient_checkpointing": "yes"})


def test_resolve_parallel():
    from autovision.cli.common_helpers import resolve_parallel

    assert resolve_parallel(None) == "ddp"
    assert resolve_parallel({"parallel": "FSDP"}) == "fsdp"
    with pytest.raises(ValueError, match="must be 'ddp' or 'fsdp'"):
        resolve_parallel({"parallel": "deepspeed"})
