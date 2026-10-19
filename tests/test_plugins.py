# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Plugin protocol: registration, entry-point discovery, template package."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import autovision.plugins as plugins
from autovision.cli._app import app
from autovision.encoders.package_helpers import get_source
from autovision.plugins import (
    PLUGIN_API_VERSION,
    DiscoveryResult,
    PluginError,
    PluginInfo,
    PluginLoadError,
    discover_plugins,
    get_plugin,
    list_plugins,
    register_plugin,
)

ROOT = Path(__file__).parent.parent


def _info(name: str, kind: str = "metric", factory=None) -> PluginInfo:
    return PluginInfo(
        name=name,
        version="1.0",
        kind=kind,
        factory=factory or (lambda: object()),
    )


def test_register_list_get_and_duplicate():
    info = _info("unit-metric-a")
    register_plugin(info)
    assert info in list_plugins("metric")
    assert get_plugin("metric", "unit-metric-a") is info
    with pytest.raises(PluginError, match="already registered"):
        register_plugin(info)
    with pytest.raises(PluginError, match="No plugin"):
        get_plugin("metric", "missing")


def test_invalid_plugin_fields():
    with pytest.raises(PluginError, match="kind"):
        register_plugin(_info("bad-kind", kind="bogus"))
    with pytest.raises(PluginError, match="API"):
        register_plugin(
            PluginInfo(
                name="bad-api", version="1", kind="metric",
                factory=object, api_version=PLUGIN_API_VERSION + 1,
            )
        )
    with pytest.raises(PluginError, match="name/version"):
        register_plugin(_info(""))


def test_mixed_plugin_iterable_is_rejected():
    """A partially bad entry point must fail instead of silently dropping items."""
    with pytest.raises(PluginError, match="mixed iterable"):
        plugins._coerce([_info("mixed-metric"), "not-a-plugin"])


def test_model_source_plugin_registers_with_encoder_catalog():
    source = SimpleNamespace(name="unit-source-a")
    register_plugin(
        _info("unit-source-plugin", kind="model_source", factory=lambda: source)
    )
    assert get_source("unit-source-a") is source


def test_discovery_loads_entry_points(monkeypatch):
    class EntryPoint:
        name = "unit-discovery-a"

        def load(self):
            return _info("unit-discovery-a")

    class Points:
        def select(self, group):
            assert group == "autovision.plugins"
            return [EntryPoint()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: Points())
    result = discover_plugins()
    assert result.ok
    assert [plugin.name for plugin in result.loaded] == ["unit-discovery-a"]
    assert get_plugin("metric", "unit-discovery-a").version == "1.0"


def test_discovery_errors_and_strict(monkeypatch):
    class EntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    class Points:
        def select(self, group):
            return [EntryPoint()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: Points())
    result = discover_plugins()
    assert not result.ok
    assert result.errors[0].entry_point == "broken"
    with pytest.raises(PluginError, match="broken"):
        discover_plugins(strict=True)


def test_template_package_is_loadable():
    plugin_root = ROOT / "examples" / "plugin-demo"
    sys.path.insert(0, str(plugin_root))
    try:
        import autovision_demo_plugin

        infos = autovision_demo_plugin.plugin()
        assert {info.kind for info in infos} == {"metric", "dataset"}
        for info in infos:
            register_plugin(info)
        assert get_plugin("metric", "autovision-demo-metric").version == "0.1.0"
    finally:
        sys.path.remove(str(plugin_root))


def test_cli_plugins_list_and_check(monkeypatch):
    info = _info("unit-cli-a")
    monkeypatch.setattr(
        plugins, "discover_plugins", lambda: DiscoveryResult(loaded=[info])
    )
    monkeypatch.setattr(plugins, "list_plugins", lambda: [info])
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "action=list"])
    assert result.exit_code == 0, result.output
    assert "unit-cli-a" in result.output

    monkeypatch.setattr(
        plugins,
        "discover_plugins",
        lambda: DiscoveryResult(
            errors=[PluginLoadError(entry_point="broken", error="boom")]
        ),
    )
    result = runner.invoke(app, ["plugins", "action=check"])
    assert result.exit_code == 1
    assert "broken" in result.output
