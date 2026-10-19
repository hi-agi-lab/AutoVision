# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Plugin protocol (roadmap 90d, phase 4).

Plugins are discovered from the ``autovision.plugins`` entry-point group.
An entry point returns one :class:`PluginInfo` (or an iterable of them),
or registers itself during import. Discovery is explicit — importing
``autovision`` never executes third-party plugin code.

``model_source`` plugins are wired into the existing encoder catalog
(:func:`autovision.encoders.register_source`); the other kinds are stored
factories for callers (dataset/metric/loss/task/postprocess). The contract
is versioned with :data:`PLUGIN_API_VERSION`; a mismatching plugin is
rejected rather than guessed at.
"""

from __future__ import annotations

import importlib.metadata
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from autovision.errors import AutoVisionError

__all__ = [
    "PLUGIN_API_VERSION",
    "PLUGIN_KINDS",
    "PluginError",
    "PluginInfo",
    "PluginLoadError",
    "DiscoveryResult",
    "discover_plugins",
    "get_plugin",
    "list_plugins",
    "register_plugin",
]

PLUGIN_API_VERSION = 1
PLUGIN_KINDS = (
    "model_source",
    "dataset",
    "metric",
    "loss",
    "task",
    "postprocess",
)


class PluginError(AutoVisionError):
    pass


@dataclass(frozen=True)
class PluginInfo:
    name: str
    version: str
    kind: str
    factory: Callable[[], Any] | Any
    api_version: int = PLUGIN_API_VERSION
    description: str = ""


@dataclass(frozen=True)
class PluginLoadError:
    entry_point: str
    error: str


@dataclass
class DiscoveryResult:
    loaded: list[PluginInfo] = field(default_factory=list)
    errors: list[PluginLoadError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


_PLUGINS: dict[tuple[str, str], PluginInfo] = {}
_PLUGINS_LOCK = threading.RLock()


def _validate(info: PluginInfo) -> None:
    if not info.name or not info.version:
        raise PluginError("Plugin name/version must be non-empty.")
    if info.kind not in PLUGIN_KINDS:
        raise PluginError(
            f"Unknown plugin kind '{info.kind}'; expected one of "
            f"{list(PLUGIN_KINDS)}."
        )
    if info.api_version != PLUGIN_API_VERSION:
        raise PluginError(
            f"Plugin '{info.name}' targets API v{info.api_version}; this "
            f"framework supports v{PLUGIN_API_VERSION}."
        )


def register_plugin(info: PluginInfo) -> PluginInfo:
    """Register one plugin; ``model_source`` factories are loaded eagerly."""
    _validate(info)
    key = (info.kind, info.name)
    with _PLUGINS_LOCK:
        if key in _PLUGINS:
            raise PluginError(
                f"Plugin '{info.name}' of kind '{info.kind}' is already registered."
            )
        if info.kind == "model_source":
            from autovision.encoders.package_helpers import register_source

            source = info.factory() if callable(info.factory) else info.factory
            try:
                register_source(source)
            except Exception as error:  # noqa: BLE001 - source contract errors
                raise PluginError(
                    f"model_source plugin '{info.name}' failed to register: {error}"
                ) from error
        _PLUGINS[key] = info
    return info


def list_plugins(kind: str | None = None) -> list[PluginInfo]:
    if kind is not None and kind not in PLUGIN_KINDS:
        raise PluginError(f"Unknown plugin kind '{kind}'.")
    with _PLUGINS_LOCK:
        return [
            info
            for (entry_kind, _), info in sorted(_PLUGINS.items())
            if kind is None or entry_kind == kind
        ]


def get_plugin(kind: str, name: str) -> PluginInfo:
    with _PLUGINS_LOCK:
        try:
            return _PLUGINS[(kind, name)]
        except KeyError as error:
            raise PluginError(
                f"No plugin '{name}' of kind '{kind}' is registered."
            ) from error


def _entry_points() -> Iterable[Any]:
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return points.select(group="autovision.plugins")
    return points.get("autovision.plugins", [])  # pragma: no cover - py3.9


def _coerce(value: Any) -> list[PluginInfo]:
    if value is None:
        return []
    if isinstance(value, PluginInfo):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = list(value)
        invalid = [type(item).__name__ for item in items if not isinstance(item, PluginInfo)]
        if invalid:
            raise PluginError(
                "Plugin entry point returned a mixed iterable; every item "
                f"must be PluginInfo (got {invalid})."
            )
        return items
    raise PluginError(
        f"Plugin entry point returned {type(value).__name__}; expected "
        "PluginInfo or an iterable of PluginInfo."
    )


def discover_plugins(*, strict: bool = False) -> DiscoveryResult:
    """Load ``autovision.plugins`` entry points.

    With ``strict=True`` the first failure raises; otherwise failures are
    returned in ``result.errors`` and successful plugins stay registered.
    """
    result = DiscoveryResult()
    with _PLUGINS_LOCK:
        pass_start_keys = set(_PLUGINS)
    for entry_point in _entry_points():
        with _PLUGINS_LOCK:
            entry_start_keys = set(_PLUGINS)
        try:
            value = entry_point.load()
            if callable(value) and not isinstance(value, PluginInfo):
                value = value()
            plugins = _coerce(value)
            with _PLUGINS_LOCK:
                if not plugins and set(_PLUGINS) == entry_start_keys:
                    # The entry point may have registered itself during import.
                    raise PluginError(
                        f"entry point '{entry_point.name}' registered nothing"
                    )
            for plugin in plugins:
                register_plugin(plugin)
                result.loaded.append(plugin)
        except Exception as error:  # noqa: BLE001 - plugin boundary
            if strict:
                # Roll back this discovery pass so a partially loaded plugin
                # set cannot leak into callers after a strict failure.
                with _PLUGINS_LOCK:
                    for key in set(_PLUGINS) - pass_start_keys:
                        _PLUGINS.pop(key, None)
                raise PluginError(
                    f"Plugin entry point '{entry_point.name}' failed: {error}"
                ) from error
            result.errors.append(
                PluginLoadError(entry_point=str(entry_point.name), error=str(error))
            )
    return result
