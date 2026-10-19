# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""On-demand download of the upstream MIT GTR checkpoints (plan A).

The GTR weights are MIT-licensed and live in the upstream HuggingFace repo
``Phoenix8125/GTR``. AutoVision does not redistribute them: this module
fetches the selected checkpoints into the GTR store
(:data:`~autovision.registry.gtr_assets.GTR_ASSET_ROOT`, or a ``root``
override), then verifies the pinned sha256 before use.

The registry pins identity, not distribution. The upstream revision is
recorded per asset (``GTRAsset.revision``) when known; otherwise the
downloader resolves the current ``main`` commit and prints it so the caller
can pin it in ``GTR_HF_REVISION`` after the first download.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from autovision.errors import AutoVisionError
from autovision.registry.gtr_assets import (
    GTR_ASSET_ROOT,
    GTR_ASSETS,
    GTR_HF_REPO,
    GTR_HF_REVISION,
    GTRAsset,
    gtr_assets_for_task,
    verify_gtr_asset,
)

__all__ = [
    "GTRDownloadError",
    "download_gtr_asset",
    "resolve_revision",
    "select_assets",
    "target_path",
    "verify_asset",
]


class GTRDownloadError(AutoVisionError):
    """Raised when a GTR checkpoint cannot be selected, fetched or verified."""


def select_assets(
    *,
    names: Sequence[str] = (),
    tasks: Sequence[str] = (),
    all_assets: bool = False,
) -> list[GTRAsset]:
    """Select assets by name / task group, or all of them.

    Exactly one selection mode is allowed: ``all_assets`` or at least one
    of ``names`` / ``tasks``. Unknown names and empty task groups fail.
    """

    if all_assets:
        if names or tasks:
            raise GTRDownloadError(
                "choose --all or --asset/--task, not both"
            )
        return list(GTR_ASSETS.values())
    if not names and not tasks:
        raise GTRDownloadError("choose --all, --asset or --task")

    selected: dict[str, GTRAsset] = {}
    for name in names:
        asset = GTR_ASSETS.get(name)
        if asset is None:
            known = ", ".join(sorted(GTR_ASSETS))
            raise GTRDownloadError(
                f"unknown GTR asset '{name}'; registered: {known}"
            )
        selected[name] = asset
    for task in tasks:
        assets = gtr_assets_for_task(task)
        if not assets:
            known = ", ".join(sorted({asset.task for asset in GTR_ASSETS.values()}))
            raise GTRDownloadError(
                f"unknown GTR task '{task}'; registered: {known}"
            )
        selected.update(assets)
    return [selected[name] for name in sorted(selected)]


def resolve_revision(
    revision: str | None = None,
    *,
    repo_id: str = GTR_HF_REPO,
    api: Any | None = None,
) -> str:
    """Return an explicit revision, the pinned registry revision, or the
    current upstream commit (resolved through ``HfApi`` and printed by the
    CLI so it can be pinned)."""

    if revision:
        return revision
    if GTR_HF_REVISION:
        return GTR_HF_REVISION
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise GTRDownloadError(
                "huggingface_hub is required to resolve the upstream "
                "revision; install the 'download' extra "
                "(pip install 'autovision-framework[download]')"
            ) from error
        api = HfApi()
    try:
        return str(api.model_info(repo_id).sha)
    except Exception as error:  # noqa: BLE001 - one actionable error surface
        raise GTRDownloadError(
            f"failed to resolve the upstream revision for {repo_id}: {error}. "
            "If the repo is gated, run 'hf auth login' or set HF_TOKEN."
        ) from error


def target_path(asset: GTRAsset, root: Path | str | None = None) -> Path:
    """Local path for ``asset`` under ``root`` (default: registry root)."""

    if root is None:
        return asset.path
    root_path = Path(root).expanduser()
    return root_path / asset.path.relative_to(GTR_ASSET_ROOT)


def verify_asset(asset: GTRAsset, root: Path | str | None = None) -> Path:
    """Verify the checkpoint at ``root`` and return its path."""

    target = target_path(asset, root)
    verify_gtr_asset(replace(asset, path=target))
    return target


def _default_downloader(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: str,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise GTRDownloadError(
            "huggingface_hub is required to download GTR checkpoints; "
            "install the 'download' extra "
            "(pip install 'autovision-framework[download]')"
        ) from error
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
        )
    except Exception as error:  # noqa: BLE001 - one actionable error surface
        raise GTRDownloadError(
            f"failed to download '{filename}' from {repo_id}@{revision}: "
            f"{error}. If the upstream repo is gated, run 'hf auth login' "
            "and retry; for regional mirrors set HF_ENDPOINT."
        ) from error
    return Path(path)


Downloader = Callable[..., Path]


def download_gtr_asset(
    asset: GTRAsset,
    *,
    revision: str,
    root: Path | str | None = None,
    force: bool = False,
    offline: bool = False,
    repo_id: str = GTR_HF_REPO,
    downloader: Downloader | None = None,
) -> tuple[Path, bool]:
    """Fetch and verify one asset; return ``(path, downloaded)``.

    An existing file that matches the pinned sha256 is reused
    (``downloaded=False``). A corrupt existing file is re-downloaded.
    ``offline=True`` refuses all network access and fails if the file is
    missing or invalid.
    """

    target = target_path(asset, root)
    if target.is_file() and not force:
        try:
            verify_gtr_asset(replace(asset, path=target))
            return target, False
        except Exception:
            if offline:
                raise
            # Corrupt/incomplete cache: replace it below.

    if offline:
        raise GTRDownloadError(
            f"GTR asset '{asset.name}' is missing or invalid at '{target}' "
            "and offline mode forbids downloading it"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    # A stale/corrupt target or an explicit --force must not let
    # ``hf_hub_download`` reuse the local file; remove it first.
    target.unlink(missing_ok=True)
    fetch = downloader or _default_downloader
    try:
        fetch(
            repo_id=repo_id,
            filename=asset.hf_filename,
            revision=revision,
            local_dir=str(Path(root).expanduser() if root else GTR_ASSET_ROOT),
        )
    except GTRDownloadError:
        raise
    except Exception as error:  # noqa: BLE001 - injected downloader
        raise GTRDownloadError(
            f"failed to download '{asset.hf_filename}' from "
            f"{repo_id}@{revision}: {error}"
        ) from error

    try:
        verify_gtr_asset(replace(asset, path=target))
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, True
