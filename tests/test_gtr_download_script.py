# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR downloader CLI: dry-run / help / usage errors (no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "download_gtr_assets.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


class TestGTRDownloadScript:
    def test_help(self):
        result = _run("--help")
        assert result.returncode == 0
        assert "--verify-only" in result.stdout
        assert "--revision" in result.stdout

    def test_all_dry_run_lists_the_family(self):
        result = _run("--all", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "assets: 26" in result.stdout
        assert "gtr_s_coco" in result.stdout
        assert "<resolve main at run>" in result.stdout

    def test_task_dry_run(self):
        result = _run("--task", "det", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "assets: 4" in result.stdout
        assert "gtrpose_s_coco" not in result.stdout

    def test_explicit_revision_in_dry_run(self):
        result = _run("--asset", "gtr_s_coco", "--revision", "abc123", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "revision: abc123" in result.stdout

    def test_unknown_asset_is_a_usage_error(self):
        result = _run("--asset", "nope", "--dry-run")
        assert result.returncode == 2
        assert "unknown GTR asset" in result.stderr

    def test_all_plus_filter_is_a_usage_error(self):
        result = _run("--all", "--asset", "gtr_s_coco", "--dry-run")
        assert result.returncode == 2
        assert "not both" in result.stderr

    def test_force_with_verify_only_is_a_usage_error(self):
        result = _run("--all", "--force", "--verify-only")
        assert result.returncode == 2
        assert "cannot be combined" in result.stderr
