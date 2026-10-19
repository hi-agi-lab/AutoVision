# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-worker app factory.

uvicorn requires an import string (not an app object) when ``workers > 1``.
:func:`autovision.serving.server.serve` serializes the app configuration into
``AUTOVISION_SERVE_CONFIG``; every worker imports this module and rebuilds its
own app (each worker loads its own model copy, as usual for process workers).
"""

from __future__ import annotations

import json
import os
from typing import Any

from autovision.errors import AutoVisionError
from autovision.serving.app import create_app

__all__ = ["create_app_from_env", "SERVE_CONFIG_ENV"]

SERVE_CONFIG_ENV = "AUTOVISION_SERVE_CONFIG"


def create_app_from_env() -> Any:
    raw = os.environ.get(SERVE_CONFIG_ENV)
    if not raw:
        raise AutoVisionError(
            f"{SERVE_CONFIG_ENV} is empty; start multi-worker serving through "
            "autovision.serving.server.serve."
        )
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AutoVisionError(
            f"Invalid {SERVE_CONFIG_ENV} JSON: {error}"
        ) from error
    if not isinstance(config, dict):
        raise AutoVisionError(f"{SERVE_CONFIG_ENV} must hold a JSON object.")
    class_names = config.get("class_names")
    if isinstance(class_names, dict):
        # JSON object keys are strings; create_app expects int labels.
        config["class_names"] = {
            int(label): name for label, name in class_names.items()
        }
    return create_app(**config)
