# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import errno
import logging
import os

import pytest
import torch

from autovision.errors import CheckpointError
from autovision.runtime.checkpoint import (
    CHECKPOINT_AUTOVISION_KEY,
    Checkpoint,
    CheckpointMetadata,
)


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "checkpoints" / "last.ckpt"
        metadata = CheckpointMetadata(epoch=3, global_step=42, normalize_args={"mean": [0.5]})
        checkpoint = Checkpoint(state_dict={"weight": torch.randn(4, 4)}, autovision=metadata)
        checkpoint.save(path)
        restored = Checkpoint.from_path(path)
        assert torch.allclose(restored.state_dict["weight"], checkpoint.state_dict["weight"])
        assert restored.autovision.epoch == 3
        assert restored.autovision.global_step == 42
        assert restored.autovision.license_info
        assert restored.autovision.version

    def test_atomic_save_leaves_no_tmp(self, tmp_path):
        path = tmp_path / "last.ckpt"
        Checkpoint(state_dict={}, autovision=CheckpointMetadata()).save(path)
        assert path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_save_preserves_the_previous_checkpoint(self, tmp_path, monkeypatch):
        import autovision.runtime.checkpoint as checkpoint_module

        path = tmp_path / "last.ckpt"
        Checkpoint(
            state_dict={"weight": torch.ones(1)}, autovision=CheckpointMetadata()
        ).save(path)
        original = path.read_bytes()

        def failing_save(payload, target, *args, **kwargs):
            with open(target, "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("disk full")

        monkeypatch.setattr(checkpoint_module.torch, "save", failing_save)
        with pytest.raises(RuntimeError, match="disk full"):
            Checkpoint(
                state_dict={"weight": torch.zeros(1)}, autovision=CheckpointMetadata()
            ).save(path)
        assert path.read_bytes() == original
        assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    def test_enospc_raises_checkpoint_error_and_preserves_previous(
        self, tmp_path, monkeypatch
    ):
        """Disk full fails loud as CheckpointError; the old file is intact."""
        import autovision.runtime.checkpoint as checkpoint_module

        path = tmp_path / "last.ckpt"
        Checkpoint(
            state_dict={"weight": torch.ones(1)}, autovision=CheckpointMetadata()
        ).save(path)
        original = path.read_bytes()

        def failing_save(payload, target, *args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(checkpoint_module.torch, "save", failing_save)
        with pytest.raises(CheckpointError, match="No space left"):
            Checkpoint(
                state_dict={"weight": torch.zeros(1)},
                autovision=CheckpointMetadata(),
            ).save(path)
        assert path.read_bytes() == original
        assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    def test_save_sweeps_stale_temp_files(self, tmp_path):
        """A temp file orphaned by a crashed writer is removed by the next save."""
        path = tmp_path / "last.ckpt"
        stale = tmp_path / ".last.ckpt.deadbeef.tmp"
        stale.write_bytes(b"orphan")
        stat = stale.stat()
        os.utime(stale, (stat.st_atime - 7200, stat.st_mtime - 7200))
        Checkpoint(state_dict={}, autovision=CheckpointMetadata()).save(path)
        assert path.exists()
        assert not stale.exists()

    def test_save_sweeps_tmp_for_names_with_glob_metachars(self, tmp_path):
        """Filenames like 'run[1].ckpt' must not break the sweep glob."""
        path = tmp_path / "run[1].ckpt"
        stale = tmp_path / ".run[1].ckpt.deadbeef.tmp"
        stale.write_bytes(b"orphan")
        stat = stale.stat()
        os.utime(stale, (stat.st_atime - 7200, stat.st_mtime - 7200))
        Checkpoint(state_dict={}, autovision=CheckpointMetadata()).save(path)
        assert not stale.exists()
        assert path.exists()

    def test_cross_version_checkpoint_load_warns(self, tmp_path, caplog):
        """Loading a checkpoint from another AutoVision version warns."""
        path = tmp_path / "old.ckpt"
        Checkpoint(
            state_dict={}, autovision=CheckpointMetadata(version="0.0.1")
        ).save(path)
        with caplog.at_level(
            logging.WARNING, logger="autovision.runtime.checkpoint"
        ):
            Checkpoint.from_path(path)
        assert any("0.0.1" in record.message for record in caplog.records)

    def test_same_version_checkpoint_load_is_quiet(self, tmp_path, caplog):
        path = tmp_path / "current.ckpt"
        Checkpoint(state_dict={}, autovision=CheckpointMetadata()).save(path)
        with caplog.at_level(
            logging.WARNING, logger="autovision.runtime.checkpoint"
        ):
            Checkpoint.from_path(path)
        assert not [
            record for record in caplog.records if "cross-version" in record.message
        ]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="does not exist"):
            Checkpoint.from_path(tmp_path / "nope.ckpt")

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "bad.ckpt"
        path.write_bytes(b"not-a-checkpoint")
        with pytest.raises(CheckpointError, match="Failed to load"):
            Checkpoint.from_path(path)

    def test_non_autovision_file_raises(self, tmp_path):
        path = tmp_path / "foreign.ckpt"
        torch.save({"some_key": 1}, path)
        with pytest.raises(CheckpointError, match=CHECKPOINT_AUTOVISION_KEY):
            Checkpoint.from_path(path)

    def test_metadata_ignores_unknown_fields(self):
        metadata = CheckpointMetadata.from_dict(
            {"epoch": 2, "future_field": "whatever"}
        )
        assert metadata.epoch == 2

    def test_default_metadata(self):
        metadata = CheckpointMetadata()
        assert metadata.epoch == -1
        assert metadata.date  # ISO timestamp set
