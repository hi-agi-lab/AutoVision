# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Crash-residue sweeps: age-guarded, best-effort, pattern-scoped."""

import os

from autovision.runtime.fs_utils import sweep_stale_temp_files


def _backdate(path, seconds: float) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime - seconds, stat.st_mtime - seconds))


def test_old_matching_files_are_removed(tmp_path):
    stale = tmp_path / ".run.ckpt.123.tmp"
    stale.write_bytes(b"orphan of a crashed run")
    _backdate(stale, 7200.0)
    assert sweep_stale_temp_files(tmp_path, ".run.ckpt.*.tmp") == 1
    assert not stale.exists()


def test_fresh_temp_files_are_never_touched(tmp_path):
    """A concurrent writer's temp file is always younger than the guard."""
    fresh = tmp_path / ".run.ckpt.456.tmp"
    fresh.write_bytes(b"in-flight write")
    assert sweep_stale_temp_files(tmp_path, ".run.ckpt.*.tmp") == 0
    assert fresh.exists()


def test_min_age_zero_removes_everything_matching(tmp_path):
    fresh = tmp_path / ".run.ckpt.1.tmp"
    fresh.write_bytes(b"old enough under test")
    keep = tmp_path / "other.tmp"
    keep.write_bytes(b"different target")
    assert sweep_stale_temp_files(tmp_path, ".run.ckpt.*.tmp", min_age_s=0.0) == 1
    assert not fresh.exists()
    assert keep.exists()


def test_missing_directory_is_not_an_error(tmp_path):
    assert sweep_stale_temp_files(tmp_path / "absent", "*.tmp") == 0


def test_unlinkable_entry_is_skipped(tmp_path):
    """A directory matching the glob must not abort the sweep."""
    weird = tmp_path / ".run.ckpt.9.tmp"
    weird.mkdir()
    _backdate(weird, 7200.0)
    assert sweep_stale_temp_files(tmp_path, ".run.ckpt.*.tmp") == 0
    assert weird.exists()
