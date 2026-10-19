# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Fake ``ultralytics`` module for tests (the real package is AGPL and not
installed in the test environment).

The fake reproduces exactly the surface our package uses: ``YOLO`` /
``RTDETR`` classes, ``utils.downloads.GITHUB_ASSETS_STEMS``,
``nn.tasks.yaml_model_load``, and ``__version__`` for version gating.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch import nn


class Conv(nn.Module):
    """ultralytics-style Conv: ``self.conv`` is the raw Conv2d."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.c2 = c2
        self.conv = Conv(c1, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.cv2 = Conv(c1, c2, k=1)  # no c2 attribute -> cv2 resolution path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(x)


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.c2 = c2
        self.conv = Conv(c1, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HGBlock(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.conv = Conv(c1, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class A2C2f(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.c2 = c2
        self.conv = Conv(c1, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Classify(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.linear = nn.Linear(c1, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=(2, 3)))


class FakeDetectionModel(nn.Module):
    def __init__(self, layers: list[nn.Module], args: dict | None = None) -> None:
        super().__init__()
        self.model = nn.Sequential(*layers)
        self.args = dict(args or {})
        self.yaml: dict = {}


def _v8_stack() -> list[nn.Module]:
    # Mirrors the real yolov8 adjacency: C2f immediately precedes SPPF.
    return [
        Conv(3, 8, k=3, s=2),
        Conv(8, 16, k=3, s=2),
        C2f(16, 16),
        SPPF(16, 32),
        torch.nn.Upsample(scale_factor=2),
        Conv(32, 16, k=1),
    ]


def _v11_stack() -> list[nn.Module]:
    # Mirrors the real yolo11 adjacency: C2f and SPPF are NOT adjacent (a
    # Conv sits between), so the (C2PSA, Upsample) pattern closes the
    # backbone at its first element.
    return [
        Conv(3, 8, k=3, s=2),
        C2f(8, 16),
        Conv(16, 16, k=1),
        SPPF(16, 16),
        C2PSA(16, 16),
        torch.nn.Upsample(scale_factor=2),
    ]


def _v12_stack() -> list[nn.Module]:
    return [
        Conv(3, 8, k=3, s=2),
        A2C2f(8, 16),
        torch.nn.Upsample(scale_factor=2),
    ]


def _rtdetr_stack() -> list[nn.Module]:
    return [
        Conv(3, 16, k=3, s=2),
        HGBlock(16, 16),
        Conv(16, 32, k=3, s=2),
    ]


def _build(name: str, constructed: list[str], kwargs: dict) -> Any:
    """Build a fake detection model for the given model name."""
    constructed.append(name)
    if "detr" in name.lower():
        layers = _rtdetr_stack()
    elif "yolo11" in name:
        layers = _v11_stack()
    elif "yolo12" in name:
        layers = _v12_stack()
    else:
        layers = _v8_stack()
    detection_model = FakeDetectionModel(layers, args={"freeze": kwargs.get("freeze")})
    if name.endswith(".pt"):
        # Checkpoint-loaded ultralytics models disable gradients.
        detection_model.requires_grad_(False)
    return detection_model


def install_fake_ultralytics(monkeypatch, version: str = "8.3.50") -> types.ModuleType:
    """Install the fake ``ultralytics`` module tree into sys.modules."""
    mod = types.ModuleType("ultralytics")
    mod.__version__ = version
    constructed: list[str] = []

    class FakeYOLO:
        def __init__(self, model: str = "yolov8n.yaml", **kwargs) -> None:
            self.model = _build(model, constructed, kwargs)
            self.save = lambda path: torch.save({"model": self.model}, path)

    class FakeRTDETR(FakeYOLO):
        pass

    mod.YOLO = FakeYOLO
    mod.RTDETR = FakeRTDETR
    mod._constructed_names = constructed

    utils = types.ModuleType("ultralytics.utils")
    downloads = types.ModuleType("ultralytics.utils.downloads")
    downloads.GITHUB_ASSETS_STEMS = {
        "yolov8n",
        "yolov8n-cls",
        "yolo11n",
        "yolo12n",
        "yolo26n",
        "rtdetr-l",
        "v8unmatched",
    }
    utils.downloads = downloads
    nn_mod = types.ModuleType("ultralytics.nn")
    tasks = types.ModuleType("ultralytics.nn.tasks")
    _KNOWN_YAMLS = {
        "yolov8n.yaml",
        "yolov8s.yaml",
        "yolov8m.yaml",
        "yolov8l.yaml",
        "yolov8x.yaml",
        "yolo11n.yaml",
        "yolo11s.yaml",
        "yolo11m.yaml",
        "yolo11l.yaml",
        "yolo11x.yaml",
        "yolo26m.yaml",
        "yolo26m-p2.yaml",
        "yolo26m-p6.yaml",
        "rtdetr-l.yaml",
        "rtdetr-x.yaml",
    }

    def yaml_model_load(name: str) -> dict:
        if name in _KNOWN_YAMLS:
            return {"scale": "n"}
        raise FileNotFoundError(name)

    tasks.yaml_model_load = yaml_model_load
    nn_mod.tasks = tasks
    mod.utils = utils
    mod.nn = nn_mod

    for name, module in [
        ("ultralytics", mod),
        ("ultralytics.utils", utils),
        ("ultralytics.utils.downloads", downloads),
        ("ultralytics.nn", nn_mod),
        ("ultralytics.nn.tasks", tasks),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return mod
