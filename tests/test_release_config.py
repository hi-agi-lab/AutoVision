# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Release URL configuration tool: writes project.urls + mkdocs site/repo URLs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "configure_release.py"


def _root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        "\n# TODO(release): add [project.urls] once URLs are known\n"
        '\n[project.scripts]\ndemo = "demo:main"\n'
    )
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: Demo\nsite_description: demo docs\nnav:\n  - Home: index.md\n"
    )
    return tmp_path


def test_configure_release_writes_urls_and_check(tmp_path):
    root = _root(tmp_path)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--root", str(root),
            "--repo-url", "https://github.com/acme/AutoVision.git",
            "--docs-url", "https://acme.github.io/AutoVision",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    pyproject = (root / "pyproject.toml").read_text()
    mkdocs = (root / "mkdocs.yml").read_text()
    assert '[project.urls]' in pyproject
    assert 'Repository = "https://github.com/acme/AutoVision"' in pyproject
    assert "site_url: https://acme.github.io/AutoVision/" in mkdocs
    assert "repo_url: https://github.com/acme/AutoVision" in mkdocs

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
    assert "release URLs configured" in result.stdout


def test_check_fails_without_urls(tmp_path):
    root = _root(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "MISSING" in result.stdout
    assert "missing [project.urls]" in result.stdout
