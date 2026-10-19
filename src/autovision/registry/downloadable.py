# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Downloadable model registry: name -> (url, sha256[, license]) with
verified download + local cache.

Entries are usually populated by task-model packages at import time; tests
and plugins can add their own via :func:`register_downloadable`. License
metadata comes from :mod:`autovision.registry.assets`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.errors import AutoVisionError
from autovision.registry.assets import LicenseRef
from autovision.runtime.env import Env

__all__ = [
    "ASSET_MANIFEST",
    "DownloadableModelError",
    "DownloadableModel",
    "LicenseRef",
    "register_downloadable",
    "resolve_downloadable",
]

logger = logging.getLogger("autovision")


class DownloadableModelError(AutoVisionError):
    """Raised when a downloadable model is unknown or fails verification."""


@dataclass(frozen=True)
class DownloadableModel:
    """A hosted model checkpoint pinned by its sha256.

    ``license`` is optional until the Phase 4 asset switch (design doc 28,
    ledger P3-1) and mandatory afterwards: every downloadable binary must
    state its license (see :mod:`autovision.registry.assets`).

    ``mirrors`` lists fallback URLs tried in order when the primary ``url``
    fails (download error or sha256 mismatch) — e.g. a regional hub mirror
    next to an upstream release URL.
    """

    url: str
    sha256: str
    license: LicenseRef | None = None
    mirrors: tuple[str, ...] = ()


ASSET_MANIFEST: dict[str, DownloadableModel] = {}


def register_downloadable(
    name: str,
    url: str,
    sha256: str,
    license: LicenseRef | None = None,
    mirrors: tuple[str, ...] = (),
) -> DownloadableModel:
    """Register (or replace) a downloadable model entry."""
    if not name or not url or not sha256:
        raise DownloadableModelError(
            "register_downloadable requires name, url and sha256."
        )
    entry = DownloadableModel(url=url, sha256=sha256, license=license, mirrors=mirrors)
    ASSET_MANIFEST[name] = entry
    return entry


def resolve_downloadable(
    name: str,
    *,
    cache_dir: Path | str | None = None,
) -> Path:
    """Resolve a downloadable model name to a verified local file.

    Downloads the pinned URL into the cache (env ``AUTOVISION_CACHE_DIR`` or
    ~/.cache/autovision by default), verifies sha256, and returns the path.
    An existing cache hit is re-verified (corruption is healed by
    re-download).
    """
    if name not in ASSET_MANIFEST:
        known = ", ".join(sorted(ASSET_MANIFEST)) or "<none>"
        raise DownloadableModelError(
            f"Unknown downloadable model '{name}'; registered: {known}."
        )
    entry = ASSET_MANIFEST[name]
    if entry.license is None:
        # Legacy entries registered before the R7 license field existed
        # (design doc 28 §6.3); all of them are scheduled for replacement
        # in the Phase 4 asset switch.
        logger.warning(
            "Downloadable asset '%s' carries no license metadata (legacy "
            "entry, pending Phase 4 replacement); resolving anyway.",
            name,
        )
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _default_cache_dir()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{name}.pt"
    # Registered names may contain a namespace ("d-fine/dfine_s_coco"),
    # so the cache target can be nested.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and compute_file_sha256(target) == entry.sha256:
        return target
    if target.is_file():
        logger.warning(
            "Cached model '%s' failed sha256 verification; re-downloading.", name
        )
        target.unlink()
    if Env.OFFLINE.value:
        raise DownloadableModelError(
            f"AUTOVISION_OFFLINE=1 and no verified cache for '{name}' at "
            f"'{target}'. Run once without AUTOVISION_OFFLINE to populate "
            "the cache, or pre-seed it with a sha256-matching file."
        )
    last_error: DownloadableModelError | None = None
    for source in (entry.url, *entry.mirrors):
        try:
            _download(source, target)
        except DownloadableModelError as error:
            last_error = error
            logger.warning("Download source failed, trying next: %s", error)
            continue
        actual = compute_file_sha256(target)
        if actual != entry.sha256:
            target.unlink()
            last_error = DownloadableModelError(
                f"Downloaded model '{name}' from '{source}' sha256 mismatch: "
                f"expected {entry.sha256}, got {actual}."
            )
            logger.warning("sha256 mismatch, trying next source: %s", source)
            continue
        return target
    raise last_error or DownloadableModelError(
        f"No download source succeeded for '{name}'."
    )


def _default_cache_dir() -> Path:
    import os

    env = os.environ.get("AUTOVISION_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "autovision" / "downloadable_models"


def _download(url: str, target: Path) -> None:
    import urllib.request

    logger.info("Downloading model from %s", url)
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310
            url, timeout=Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value
        ) as response:
            target.write_bytes(response.read())
    except Exception as error:  # noqa: BLE001 (clear network failure surface)
        if target.exists():
            target.unlink()
        raise DownloadableModelError(
            f"Failed to download model from '{url}': {error}"
        ) from error
