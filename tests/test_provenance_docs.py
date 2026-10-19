# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Attribution/provenance docs stay truthful and build-relative links valid.

The DEIMv2-derived spatial prior was removed for licensing. These tests
pin the post-removal state so neither the old "DEIMv2 = Apache-2.0" claim
nor a stale release blocker can return.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_has_acknowledgements_and_links():
    readme = (ROOT / "README.md").read_text()
    assert "## 致谢（Acknowledgements）" in readme
    assert "THIRD_PARTY_LICENSES.md" in readme
    assert "docs/provenance/LEDGER.md" in readme
    assert "DEIMv2" in readme
    assert "no-STA" in readme
    assert "不得商用" not in readme
    assert "https://github.com/Peterande/D-FINE" in readme
    assert "https://github.com/roboflow/rf-detr" in readme


def test_third_party_licenses_has_full_texts_without_deimv2():
    text = (ROOT / "THIRD_PARTY_LICENSES.md").read_text()
    assert "MIT License" in text
    assert "The GTR Authors" in text
    assert "Permission is hereby granted" in text
    # The DEIMv2 component (and therefore its non-commercial text) is gone.
    assert "DEIMv2 License" not in text
    assert "Non-Commercial Purpose" not in text
    assert "**BLOCKER**" not in text
    assert "SpatialPriorModule" not in text


def test_notice_points_to_existing_sources_and_records_removal():
    notice = (ROOT / "NOTICE").read_text()
    assert "docs/provenance/LEDGER.md" in notice
    assert "THIRD_PARTY_LICENSES.md" in notice
    assert "no DEIMv2 code" in notice
    assert "RELEASE BLOCKER" not in notice
    assert "This component is NOT Apache-2.0" not in notice
    assert "grant does not extend" not in notice
    assert "docs/design/28-clean-provenance-refactor.md" not in notice
    assert "DEIM / DEIMv2 (Apache License 2.0)" not in notice
    assert (ROOT / "docs" / "provenance" / "LEDGER.md").is_file()


def test_ledger_records_disposal_and_checkpoint_migration():
    ledger = (ROOT / "docs" / "provenance" / "LEDGER.md").read_text()
    assert "DEIMv2" in ledger
    assert "DONE（已删除）" in ledger
    assert "backbone.sta.*" in ledger
    assert "no-STA" in ledger
    assert "dinov3-vitt16" in ledger
    assert "THIRD_PARTY_LICENSES.md" in ledger


def test_code_has_no_spatial_prior_class():
    wrappers = (
        ROOT
        / "src"
        / "autovision"
        / "tasks"
        / "detection"
        / "detr"
        / "backbone_wrappers.py"
    ).read_text()
    variants = (
        ROOT
        / "src"
        / "autovision"
        / "tasks"
        / "detection"
        / "detr"
        / "variants.py"
    ).read_text()
    assert "class SpatialPriorModule" not in wrappers
    assert "use_sta" not in variants
    assert "conv_inplane" not in variants
    assert "removed for licensing" in wrappers


def test_packaging_includes_third_party_license_file():
    pyproject = (ROOT / "pyproject.toml").read_text()
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert 'license-files = ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"]' in pyproject
    assert "include THIRD_PARTY_LICENSES.md" in manifest
    assert "include docs/provenance/LEDGER.md" in manifest


def test_known_limits_surfaces_sta_migration():
    limits = (ROOT / "docs" / "known_limits.md").read_text()
    assert "## 许可状态与 STA 迁移" in limits
    assert "DEIMv2 组件已删除" in limits
    assert "backbone.sta.*" in limits
    assert "ConvNeXt" in limits
    assert "## 许可阻塞" not in limits
