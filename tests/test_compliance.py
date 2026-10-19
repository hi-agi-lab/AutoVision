# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Compliance pack: license normalization, model card/sha, snapshot, CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from autovision.cli._app import app
from autovision.compliance import build_compliance_report, normalize_license
from autovision.registry.model_card import LatencyRecord, ModelCard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_with_card(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not-a-real-checkpoint" * 16)
    card = ModelCard(
        model="detr/s",
        name="Demo detector",
        train_dataset="COCO 2017 train2017",
        schedule="1 step smoke",
        seed=0,
        metrics={"coco_map": 50.0},
        latency=[
            LatencyRecord(
                backend="onnxruntime",
                precision="fp32",
                hardware="CPU",
                batch_size=1,
                p50_ms=10.0,
                script="scripts/measure_spectrum_benchmarks.py",
            )
        ],
        license="Apache-2.0",
        weights_sha256={path.name: _sha(path)},
        version="demo-v1",
    )
    card.save(path)
    return path


def test_license_normalization():
    assert normalize_license("Apache Software License") == "Apache-2.0"
    assert normalize_license("MIT License") == "MIT"
    assert normalize_license("BSD 3-Clause") == "BSD-3-Clause"
    assert normalize_license("some custom terms") == "UNKNOWN"
    assert normalize_license(None) == "UNKNOWN"


def test_license_normalization_atoms():
    expected = {
        "MIT": "MIT",
        "MIT-0": "MIT-0",
        "BSD-2-Clause": "BSD-2-Clause",
        "BSD 2-Clause": "BSD-2-Clause",
        "BSD-3-Clause": "BSD-3-Clause",
        "Apache-2.0": "Apache-2.0",
        "Apache License 2.0": "Apache-2.0",
        "Apache License, Version 2.0": "Apache-2.0",
        "ISC": "ISC",
        "MPL-2.0": "MPL-2.0",
        "GPL-2.0-only": "GPL-2.0-only",
        "GPL-2.0+": "GPL-2.0-or-later",
        "GPL-3.0-only": "GPL-3.0-only",
        "GPL-3.0+": "GPL-3.0-or-later",
        "GNU General Public License v3.0": "GPL-3.0-only",
        "LGPL-2.1-only": "LGPL-2.1-only",
        "LGPL-3.0-only": "LGPL-3.0-only",
        "AGPL-3.0-only": "AGPL-3.0-only",
        "Unlicense": "Unlicense",
        "CC0-1.0": "CC0-1.0",
        "CC-BY-4.0": "CC-BY-4.0",
        "CC-BY-NC-4.0": "CC-BY-NC-4.0",
        "CC-BY-SA-4.0": "CC-BY-SA-4.0",
        "PSF-2.0": "PSF-2.0",
        "Python-2.0": "Python-2.0",
        "DINOv3 License": "DINOv3 License",
        "The MIT License": "MIT",
    }
    for raw, spdx in expected.items():
        assert normalize_license(raw) == spdx, raw


def test_license_normalization_expressions_and_fail_closed():
    # Six previously broken cases.
    assert normalize_license("BSD-2-Clause") == "BSD-2-Clause"
    assert normalize_license("Apache License 2.0") == "Apache-2.0"
    assert normalize_license("Proprietary. Redistribution permitted") == "UNKNOWN"
    assert normalize_license("MIT OR Apache-2.0") == "MIT OR Apache-2.0"
    assert normalize_license("BSD-3-Clause AND MIT") == "BSD-3-Clause AND MIT"
    assert normalize_license("GPL-3.0-only OR MIT") == "GPL-3.0-only OR MIT"

    # Operators preserved in order, with case/whitespace normalized.
    assert normalize_license("mit or apache-2.0") == "MIT OR Apache-2.0"
    assert normalize_license("  MIT   AND   ISC  ") == "MIT AND ISC"

    # Any unknown atom fails the whole expression closed.
    assert normalize_license("MIT OR Proprietary") == "UNKNOWN"
    assert normalize_license("MIT AND SomeCustomTerms") == "UNKNOWN"

    # "permitted" must never be read as MIT.
    assert normalize_license("permitted") == "UNKNOWN"


def test_report_with_model_card_and_snapshot(tmp_path):
    model = _model_with_card(tmp_path)
    snapshot = tmp_path / "data.snap.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "splits": []}))
    report = build_compliance_report(
        model=model,
        snapshot=snapshot,
        include_dependencies=False,
        timestamp=False,
    )
    assert report.passed
    assert report.generated_at is None
    assert report.models[0]["license"] == "Apache-2.0"
    assert report.models[0]["sha256"] == _sha(model)
    assert report.datasets[0]["snapshot"]["schema_version"] == 1
    report.save(tmp_path / "compliance")
    payload = json.loads((tmp_path / "compliance" / "compliance.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert (tmp_path / "compliance" / "compliance.md").is_file()


def test_missing_model_card_is_an_error(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_bytes(b"bare")
    report = build_compliance_report(
        model=model, include_dependencies=False, timestamp=False
    )
    assert not report.passed
    assert any(issue.kind == "missing_model_card" for issue in report.issues)
    assert any(issue.kind == "unknown_model_license" for issue in report.issues)


def test_cli_compliance_report_and_strict(tmp_path):
    good = _model_with_card(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compliance-report", f"model={good}", f"out={tmp_path / 'ok'}",
            "include_dependencies=false", "no_timestamp=true",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ok" / "compliance.json").is_file()

    bad = tmp_path / "bare.pt"
    bad.write_bytes(b"bare")
    result = runner.invoke(
        app,
        [
            "compliance-report", f"model={bad}", f"out={tmp_path / 'bad'}",
            "include_dependencies=false", "strict=true",
        ],
    )
    assert result.exit_code == 1


def test_compliance_warning_gate_and_sbom(tmp_path):
    model = _model_with_card(tmp_path)
    snapshot = tmp_path / "data.snap.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "splits": []}))
    report = build_compliance_report(
        model=model, snapshot=snapshot, include_dependencies=False
    )
    # The fake snapshot is unverifiable -> warning, not error.
    assert report.passes("error")
    assert not report.passes("warning")
    report.save(tmp_path / "pack")
    sbom = json.loads((tmp_path / "pack" / "compliance.sbom.json").read_text())
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert any(c["type"] == "machine-learning-model" for c in sbom["components"])
    manifest = json.loads(
        (tmp_path / "pack" / "compliance.manifest.json").read_text()
    )
    for name, digest in manifest["sha256"].items():
        import hashlib

        assert hashlib.sha256((tmp_path / "pack" / name).read_bytes()).hexdigest() == digest


def test_compliance_detects_dataset_drift(tmp_path):
    from tests.test_data_versioning import _tree

    from autovision.data.versioning import dataset_snapshot

    spec = _tree(tmp_path / "data")
    snapshot = tmp_path / "data.snap.json"
    dataset_snapshot(spec, snapshot, deep=True)
    (tmp_path / "data" / "images" / "img_0.jpg").unlink()
    report = build_compliance_report(
        snapshot=snapshot, include_dependencies=False
    )
    assert not report.passed
    assert any(issue.kind == "dataset_snapshot_drift" for issue in report.issues)


def test_cli_fail_on_warning_promotes_warnings(tmp_path):
    good = _model_with_card(tmp_path)
    snapshot = tmp_path / "fake.snap.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "splits": []}))
    result = CliRunner().invoke(
        app,
        [
            "compliance-report",
            f"model={good}",
            f"snapshot={snapshot}",
            f"out={tmp_path / 'warn'}",
            "include_dependencies=false",
            "no_timestamp=true",
            "fail_on=warning",
        ],
    )
    assert result.exit_code == 1, result.output
