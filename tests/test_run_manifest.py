# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Run manifest: command/environment audit artifact for every training run."""

import json

from autovision.cli.common_helpers import get_out_dir
from autovision.runtime.manifest import build_run_manifest, write_run_manifest


def _read(path):
    return json.loads(path.read_text())


def test_manifest_records_stack_and_command(tmp_path):
    path = write_run_manifest(tmp_path)
    payload = _read(path)
    assert path.name == "run_manifest.json"
    assert payload["schema_version"] == 1
    assert payload["framework_version"]
    assert payload["python"]
    assert payload["command"]
    assert payload["cwd"]
    assert "torch" in payload["packages"]
    assert isinstance(payload["gpus"], list)
    assert payload["resume"] is False


def test_manifest_does_not_dump_the_whole_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "do-not-record")
    monkeypatch.setenv("AUTOVISION_MIRROR_URL", "https://user:pass@mirror.invalid")
    monkeypatch.setenv("AUTOVISION_SERVING_API_KEY", "also-secret")
    payload = build_run_manifest().to_dict()
    assert "SECRET_TOKEN" not in payload["env"]
    assert "AUTOVISION_MIRROR_URL" not in payload["env"]
    assert "AUTOVISION_SERVING_API_KEY" not in payload["env"]
    assert "do-not-record" not in json.dumps(payload)
    assert "also-secret" not in json.dumps(payload)


def test_get_out_dir_writes_manifest(tmp_path):
    out = tmp_path / "run"
    get_out_dir(out)
    manifest = out / "run_manifest.json"
    assert manifest.is_file()
    assert _read(manifest)["resume"] is False


def test_resume_manifest_is_separate_from_the_original(tmp_path):
    out = tmp_path / "run"
    get_out_dir(out)
    original = _read(out / "run_manifest.json")
    get_out_dir(out, resume_interrupted=True)
    resume_files = sorted(out.glob("run_manifest.resume-*.json"))
    assert len(resume_files) == 1
    assert _read(resume_files[0])["resume"] is True
    # The original is preserved.
    assert _read(out / "run_manifest.json") == original
