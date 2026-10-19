# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Infrastructure checks: CI configs, Docker, docs, release package, packaging."""

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: stdlib tomllib landed in 3.11
    import tomli as tomllib

ROOT = Path(__file__).parent.parent


class TestCI:
    def test_gitlab_ci_parses(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
        assert set(config["stages"]) == {"lint", "test"}
        assert "pytest" in config["test"]["script"][-1]
        assert "mkdocs" in str(config["docs"]["script"])

    def test_github_workflow_parses(self):
        pytest.importorskip("yaml")
        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert {"test", "lint", "docs"} <= set(workflow["jobs"])
        # PyYAML 1.1 parses the `on:` key as boolean True.
        trigger = workflow.get("on", workflow.get(True))
        # Release tags trigger the suite, which carries the tag ↔ version check.
        assert "tags" in trigger["push"]

    def test_telemetry_disabled_in_ci(self):
        import yaml

        config = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
        variables = config["test"].get("variables", {})
        # Compliance: CI must not enable telemetry.
        assert variables.get("AUTOVISION_TELEMETRY", "0") != "1"

    def test_ci_covers_the_supported_python_range(self):
        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
        # requires-python is >=3.10; CI must exercise both ends of the
        # supported range rather than only the release interpreter.
        assert {"3.10", "3.11"} <= set(map(str, matrix))


class TestSecurityPolicy:
    def test_security_policy_exists_and_names_the_private_channel(self):
        text = (ROOT / "SECURITY.md").read_text()
        assert "Reporting a vulnerability" in text
        assert "Security Advisories" in text
        assert "Do not open a public issue" in text

    def test_dependabot_covers_python_and_actions(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())
        ecosystems = {update["package-ecosystem"] for update in config["updates"]}
        assert {"pip", "github-actions"} <= ecosystems


class TestPep561Packaging:
    def test_py_typed_marker_is_present_and_shipped(self):
        assert (ROOT / "src" / "autovision" / "py.typed").is_file()
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        package_data = config["tool"]["setuptools"]["package-data"]
        assert "py.typed" in package_data["autovision"]


class TestDocker:
    def test_dockerfile_exists_and_smoke_wired(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        assert "smoke_test.sh" in dockerfile
        assert (ROOT / "docker" / "smoke_test.sh").exists()

    def test_smoke_script_exercises_pretrain(self):
        script = (ROOT / "docker" / "smoke_test.sh").read_text()
        assert "autovision.pretrain" in script
        assert "checkpoint" in script


def _nav_paths(nav, prefix: str = "") -> list[str]:
    """Flatten a mkdocs nav tree into the referenced page paths."""
    paths: list[str] = []
    if isinstance(nav, list):
        for entry in nav:
            paths.extend(_nav_paths(entry, prefix))
    elif isinstance(nav, dict):
        for entry in nav.values():
            paths.extend(_nav_paths(entry, prefix))
    elif isinstance(nav, str):
        paths.append(nav)
    return paths


class TestDocs:
    """Site contract (E1 release package): nav ⊆ docs/, docs/ ⊆ nav ∪ excluded.

    `mkdocs build --strict` is the renderer-level gate; these checks are the
    fast structural contract and run without mkdocs installed.
    """

    def test_mkdocs_config_nav_targets_exist(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        assert config["site_name"] == "AutoVision"
        nav_files = _nav_paths(config["nav"])
        assert nav_files, "nav must reference at least one page"
        for target in nav_files:
            assert (ROOT / "docs" / target).is_file(), (
                f"nav references missing page: {target}"
            )

    def test_every_docs_page_in_nav_or_excluded(self):
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        in_nav = set(_nav_paths(config["nav"]))
        patterns = [
            line.strip()
            for line in str(config.get("exclude_docs", "")).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        pages = {
            path.relative_to(ROOT / "docs").as_posix()
            for path in (ROOT / "docs").rglob("*.md")
            # Locale alternates (index.en.md, ...) are resolved by the i18n
            # plugin, not listed as separate nav entries.
            if not re.search(r"\.[a-z]{2}(?:-[A-Z]{2})?\.md$", path.name)
        }
        # mkdocs exclude_docs takes gitignore-style patterns (docs: "a
        # pattern may match a directory"). `dir/` excludes the whole dir;
        # anything else is fnmatch-style.
        excluded: set[str] = set()
        matched = {pattern: False for pattern in patterns}
        for page in pages:
            for pattern in patterns:
                hit = (
                    page.startswith(pattern.rstrip("/") + "/")
                    if pattern.endswith("/")
                    else fnmatch.fnmatch(page, pattern)
                )
                if hit:
                    excluded.add(page)
                    matched[pattern] = True
        orphans = pages - in_nav - excluded
        assert not orphans, f"pages neither in nav nor excluded: {sorted(orphans)}"
        # Exclusions must not be stale (they double as the "internal, not
        # published" register).
        stale = sorted(pattern for pattern, hit in matched.items() if not hit)
        assert not stale, f"stale exclude_docs entries: {stale}"

    def test_internal_strategy_docs_are_not_in_the_public_tree(self):
        """Internal roadmap/analysis material must not ship in the public tree."""
        assert not (ROOT / "docs" / "strategy").exists()
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        assert not any(
            "strategy/" in target for target in _nav_paths(config["nav"])
        )

    def test_license_references_avoid_blocked_extensionless_links(self):
        """Renderers such as PyPI block ``[LICENSE](LICENSE)`` targets.

        Use a code-path reference for the shipped files and a canonical
        Apache-2.0 URL for the license text.
        """
        readme = (ROOT / "README.md").read_text()
        assert "](LICENSE)" not in readme
        assert "](NOTICE)" not in readme
        assert "https://www.apache.org/licenses/LICENSE-2.0" in readme
        pypi_readme = (ROOT / "README_PYPI.md").read_text()
        assert "](LICENSE)" not in pypi_readme
        assert "https://www.apache.org/licenses/LICENSE-2.0" in pypi_readme

    def test_docs_i18n_defaults_to_chinese_with_english_alternate(self):
        """Chinese is the default locale; English is built under /en/."""
        pytest.importorskip("yaml")
        import yaml

        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        plugins = config.get("plugins", [])
        i18n = next(
            item["i18n"]
            for item in plugins
            if isinstance(item, dict) and "i18n" in item
        )
        assert i18n["docs_structure"] == "suffix"
        assert i18n["fallback_to_default"] is True
        languages = {item["locale"]: item for item in i18n["languages"]}
        assert languages["zh"]["default"] is True
        assert languages["zh"]["build"] is True
        assert languages["en"]["default"] is False
        assert languages["en"]["build"] is True
        assert languages["en"]["nav_translations"]["首页"] == "Home"
        assert (ROOT / "docs" / "index.en.md").is_file()
        assert (ROOT / "README.en.md").is_file()
        readme = (ROOT / "README.md").read_text()
        assert "README.en.md" in readme
        assert "](README.md)" in (ROOT / "README.en.md").read_text()

    def test_site_builds_strictly(self, tmp_path):
        pytest.importorskip("mkdocs")
        pytest.importorskip("mkdocs_static_i18n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-d",
                str(tmp_path / "site"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "site" / "index.html").is_file()
        english_index = tmp_path / "site" / "en" / "index.html"
        assert english_index.is_file()
        assert "An enterprise-grade visual model training framework" in (
            english_index.read_text()
        )

    def test_no_broken_references_in_public_tree(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_docs_links.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_link_checker_detects_breakage(self, tmp_path, monkeypatch):
        """The guard itself: fences skipped for links, inline code skipped,
        repo paths checked everywhere (root + ``src/autovision/``
        convention), `<!-- docpath-ok -->` exempts one line."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "check_docs_links", ROOT / "scripts" / "check_docs_links.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ok.md").write_text(
            "see [x](index.md) and `scripts/real.py`\n"
            "inline code is fine: `[y](nowhere.md)`\n"
            "angle target: [z](<index.md>)\n"
            "line suffix: `scripts/real.py:12-34`\n"
            "module-map path: `cli/real.py`\n"
            "illustrative output path: `report/report.md`\n"
            "comment is fine: <!-- [w](nowhere.md) -->\n"
        )
        (tmp_path / "docs" / "index.md").write_text("# home\n")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "real.py").write_text("")
        (tmp_path / "src" / "autovision" / "cli").mkdir(parents=True)
        (tmp_path / "src" / "autovision" / "cli" / "real.py").write_text("")
        (tmp_path / "docs" / "bad.md").write_text(
            "fenced is fine:\n```\n[x](nowhere.md)\n```\n"
            "but this is not: [y](missing.md)\n"
            "and `scripts/gone.py` too\n"
            "and `cli/gone.py` (package-map miss)\n"
            "history: `scripts/gone.py` <!-- docpath-ok -->\n"
        )
        problems = module.check()
        assert any("missing.md" in p for p in problems), problems
        assert any("scripts/gone.py" in p for p in problems), problems
        assert any("cli/gone.py" in p for p in problems), problems
        assert not any("nowhere.md" in p for p in problems), problems
        assert sum("scripts/gone.py" in p for p in problems) == 1, problems
        assert not any(p.startswith("docs/ok.md") for p in problems), problems


class TestRelease:
    """Release-package contract (E1): version source of truth, changelog,
    contributor entry points and issue/PR templates."""

    def test_version_has_a_single_source(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert "version" not in config["project"]
        assert config["project"]["dynamic"] == ["version"]
        assert (
            config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
            == "autovision._version.__version__"
        )
        import autovision

        assert re.fullmatch(r"\d+\.\d+\.\d+", autovision.__version__), (
            "release tags are vX.Y.Z; the version string must be plain semver"
        )

    def test_changelog_follows_keep_a_changelog(self):
        text = (ROOT / "CHANGELOG.md").read_text()
        assert "Keep a Changelog" in text
        assert "## [Unreleased]" in text
        import autovision

        assert f"## [{autovision.__version__}]" in text, (
            "the current version needs a dated CHANGELOG section"
        )

    def test_release_tag_matches_version(self):
        """A HEAD tagged with a release tag must carry the same version.

        Development commits ship untagged and are unaffected; the release
        checklist runs this against the tagged commit (docs/release_process.md).
        """
        import autovision

        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("git unavailable")
        release_tags = [
            tag
            for tag in result.stdout.split()
            if re.fullmatch(r"v\d+\.\d+\.\d+", tag)
        ]
        if not release_tags:
            pytest.skip("HEAD is not a release-tagged commit")
        assert f"v{autovision.__version__}" in release_tags, (
            f"release tag/version mismatch: HEAD has {release_tags}, "
            f"package version is {autovision.__version__}"
        )

    def test_release_tag_filters_exclude_legacy_tags(self):
        """Only vX.Y.Z tags may enter the release pipeline.

        ``v0.2-clean`` is a legacy non-release tag whose tree carries 0.1.0;
        a ``v*`` glob would let it trigger the PyPI publish job.
        """
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert 'tags: ["v*.*.*"]' in workflow
        assert 'tags: ["v*"]' not in workflow
        gitlab = (ROOT / ".gitlab-ci.yml").read_text()
        assert "$CI_COMMIT_TAG =~ /^v[0-9]+\\.[0-9]+\\.[0-9]+/" in gitlab

    def test_release_urls_are_configured(self):
        """The public repo/docs URLs must be set before a public snapshot."""
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "configure_release.py"), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_no_tracked_ignored_files(self):
        """Generated cache artifacts must not ship in the public snapshot."""
        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        result = subprocess.run(
            ["git", "ls-files", "-ci", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("git unavailable")
        assert not result.stdout.strip(), (
            "tracked files match .gitignore: " + result.stdout
        )

    def test_publish_snapshot_script_is_wired(self):
        """The release snapshot must be buildable without pushing private history."""
        script = ROOT / "scripts" / "publish_clean_cut.sh"
        assert script.is_file()
        assert script.stat().st_mode & 0o111
        hook = (ROOT / "scripts" / "hooks" / "pre-push").read_text()
        assert "publish_clean_cut.sh" in hook
        result = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "--push" in result.stdout

    def test_contributor_entry_points_exist(self):
        contributing = (ROOT / "CONTRIBUTING.md").read_text()
        for gate in (
            "ruff check src tests scripts",
            "bandit -r src/autovision",
            "mypy src/autovision/errors.py",
            "scripts/smoke_enterprise.py",
            "pytest tests -q",
            "scripts/check_provenance.py",
            "scripts/check_layering.py",
            "scripts/check_docs_links.py",
            "mkdocs build --strict",
        ):
            assert gate in contributing, f"CONTRIBUTING.md must document: {gate}"

    def test_issue_and_pr_templates_parse(self):
        pytest.importorskip("yaml")
        import yaml

        template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
        for name in ("bug_report.yml", "feature_request.yml"):
            data = yaml.safe_load((template_dir / name).read_text())
            assert data["name"] and data["description"] and data["body"]
            for field in data["body"]:
                if field.get("type") in ("input", "textarea", "checkboxes"):
                    assert field.get("id"), f"{name}: a field misses its id"
        template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
        assert "ruff check src tests scripts" in template
        assert "CHANGELOG.md" in template


class TestPackaging:
    def test_cli_entry_point_registered(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        scripts = config["project"].get("scripts", {})
        assert scripts.get("autovision-cli") == "autovision.cli._app:cli_entrypoint"

    def test_optional_extras_declared(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extras = config["project"]["optional-dependencies"]
        for name in ("tensorboard", "mlflow", "wandb", "bitsandbytes",
                     "ultralytics", "transformers", "rfdetr", "timm",
                     "video", "docs", "openvino"):
            assert name in extras, f"extra '{name}' missing"

    def test_docs_extra_pins_the_site_toolchain(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        docs_extra = config["project"]["optional-dependencies"]["docs"]
        assert any(dep.startswith("mkdocs-material") for dep in docs_extra)
        assert any(dep.startswith("mkdocs-static-i18n") for dep in docs_extra)

    def test_public_api_surface(self):
        import autovision

        for name in (
            "pretrain", "train_object_detection", "embed",
            "load_task_model", "load_embedding_model",
            "list_models", "list_methods", "extract_video_frames",
        ):
            assert callable(getattr(autovision, name, None)), f"{name} missing"

    def test_lockfile_uses_the_public_pypi_index(self):
        """The committed lockfile must install outside the project network."""
        text = (ROOT / "uv.lock").read_text()
        assert "agentsmirror" not in text
        assert 'registry = "https://pypi.org/simple"' in text


class TestCIInstallsTestDependencies:
    def test_optional_test_extras_are_installed(self):
        """Collection imports DETR/EoMT transformers and serves FastAPI.

        The CI test job must install the same optional packages the test
        modules import at module level; `[dev,export]` alone fails during
        collection.
        """
        from pathlib import Path

        import yaml

        workflow = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml")
            .read_text()
        )
        steps = workflow["jobs"]["test"]["steps"]
        install = " ".join(
            str(step.get("run", "")) for step in steps
            if "pip install" in str(step.get("run", ""))
        )
        for extra in ("dev", "export", "serving", "video", "transformers"):
            assert extra in install, f"CI test job misses extra '{extra}': {install}"


class TestCIQualityGates:
    def test_sast_type_and_coverage_gates_are_wired(self):
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load(
            (root / ".github" / "workflows" / "ci.yml").read_text()
        )
        commands = " ".join(
            str(step.get("run", ""))
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        )
        assert "bandit -r src/autovision" in commands
        assert "mypy src/autovision/errors.py" in commands
        assert "--cov-fail-under=" in commands
        dev = tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]["dev"]
        for package in ("bandit", "mypy", "pytest-cov"):
            assert any(item.startswith(package) for item in dev), package


class TestTensorRTNightly:
    def test_nightly_targets_self_hosted_gpu_and_gated_tests(self):
        import yaml

        root = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load(
            (root / ".github" / "workflows" / "trt-nightly.yml").read_text()
        )
        job = workflow["jobs"]["trt-parity"]
        assert "self-hosted" in job["runs-on"]
        assert "gpu" in job["runs-on"]
        commands = " ".join(
            str(step.get("run", "")) + " " + str(step.get("env", {}))
            for step in job["steps"]
        )
        assert "AUTOVISION_TRT_TESTS" in commands
        assert (root / "scripts" / "verify_tensorrt_engine.py").is_file()
