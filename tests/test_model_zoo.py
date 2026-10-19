# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model zoo page contract (E2 release package): docs/model_zoo.md must be
byte-identical to what ``scripts/gen_model_zoo.py`` renders from the weight
registries — the documented sha256 / license / official metric columns
cannot drift from what the code enforces."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
GENERATOR = ROOT / "scripts" / "gen_model_zoo.py"
PAGE = ROOT / "docs" / "model_zoo.md"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


class TestModelZoo:
    def test_page_exists_and_is_referenced_by_the_site(self):
        assert PAGE.is_file()
        mkdocs = (ROOT / "mkdocs.yml").read_text()
        assert "model_zoo.md" in mkdocs

    def test_page_matches_the_registries(self):
        """Drift gate: regenerate and compare (the CI form of this check)."""
        result = _run("--check")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_every_registered_asset_is_documented(self):
        """Belt and braces beyond --check: every pinned asset name appears."""
        from autovision.encoders.weight_hashes import WEIGHT_MANIFEST
        from autovision.registry.gtr_assets import GTR_ASSETS
        from autovision.registry.user_held_assets import USER_HELD_ASSETS

        page = PAGE.read_text()
        for name in GTR_ASSETS:
            assert f"`{name}`" in page, f"GTR asset missing from the zoo: {name}"
        for repo_id in WEIGHT_MANIFEST:
            assert repo_id in page, f"HF weight missing from the zoo: {repo_id}"
        for name in USER_HELD_ASSETS:
            assert f"`{name}`" in page, f"user-held asset missing from the zoo: {name}"

    def test_gtr_metric_column_matches_the_registry(self):
        """The official metric values on the page are the pinned registry
        values (a wrong number here would be a compliance-grade error)."""
        from autovision.registry.gtr_assets import GTR_ASSETS

        page = PAGE.read_text()
        for asset in GTR_ASSETS.values():
            if asset.metric_value is None:
                continue
            digits = 3 if asset.metric_name == "nyu_absrel" else 1
            rendered = f"{asset.metric_value:.{digits}f}"
            assert rendered in page, (
                f"metric {asset.metric_name}={rendered} for {asset.name} "
                "missing from the zoo page"
            )

    def test_licenses_are_stated_per_asset(self):
        from autovision.encoders.weight_hashes import WEIGHT_MANIFEST
        from autovision.registry.gtr_assets import GTR_ASSETS

        page = PAGE.read_text()
        for asset in GTR_ASSETS.values():
            assert asset.license in page, f"license {asset.license} not on the page"
        for repo_id, artifact in WEIGHT_MANIFEST.items():
            rows = [line for line in page.splitlines() if f"`{repo_id}`" in line]
            assert len(rows) == 1, f"{repo_id}: expected exactly one row, got {len(rows)}"
            row = rows[0]
            assert artifact.license in row, f"{repo_id}: license missing from its row"
            assert artifact.revision[:8] in row, f"{repo_id}: revision missing from its row"
