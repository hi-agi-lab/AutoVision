#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Download the MIT GTR checkpoints from the upstream HuggingFace repo.

The weights are not redistributed by AutoVision. This script fetches the
selected files into the GTR store (``AUTOVISION_GTR_ROOT``, default
``~/Research/Model/gtr``), verifies the pinned sha256, and reuses valid
files. It needs the ``download`` extra (``huggingface_hub``):

    pip install -e ".[download]"

Usage::

    python scripts/download_gtr_assets.py --all
    python scripts/download_gtr_assets.py --task det
    python scripts/download_gtr_assets.py --asset gtr_s_coco --revision <commit>
    python scripts/download_gtr_assets.py --all --verify-only
    python scripts/download_gtr_assets.py --all --dry-run

The upstream repo may be gated: run ``hf auth login`` (or export
``HF_TOKEN``) first. For regional mirrors set ``HF_ENDPOINT`` (e.g.
``https://hf-mirror.com``). ``AUTOVISION_OFFLINE=1`` forbids network access.
The script prints the resolved upstream commit; pin it in
``GTR_HF_REVISION`` after the first download for a reproducible URL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autovision.registry.gtr_assets import (  # noqa: E402
    GTR_HF_REPO,
    GTR_HF_REVISION,
)
from autovision.registry.gtr_download import (  # noqa: E402
    GTRDownloadError,
    download_gtr_asset,
    resolve_revision,
    select_assets,
    target_path,
    verify_asset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--all",
        action="store_true",
        help="download/verify every registered GTR checkpoint (26 files)",
    )
    selection.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME",
        help="asset name, e.g. gtr_s_coco (repeatable)",
    )
    selection.add_argument(
        "--task",
        action="append",
        default=[],
        choices=("det", "det-obj365", "seg", "pose", "depth", "obb", "semseg"),
        help="task group (repeatable)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="target store (default: AUTOVISION_GTR_ROOT / registry root)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="upstream HF commit/tag (default: registry pin, else resolve main)",
    )
    parser.add_argument("--repo", default=GTR_HF_REPO, help="upstream HF repo id")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the local file already verifies",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not download; verify the selected local files",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no network; equivalent to a verification pass",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (no network, no writes)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.force and (args.verify_only or args.offline):
        parser.error("--force cannot be combined with --verify-only/--offline")

    try:
        assets = select_assets(
            names=args.asset,
            tasks=args.task,
            all_assets=args.all,
        )
    except GTRDownloadError as error:
        parser.error(str(error))

    offline = args.offline or os.environ.get("AUTOVISION_OFFLINE", "") == "1"
    verify_only = args.verify_only or offline

    revision = args.revision or GTR_HF_REVISION
    if args.dry_run:
        print(f"repo: {args.repo}")
        print(f"revision: {revision or '<resolve main at run>'}")
        for asset in assets:
            print(f"  {asset.name:24s} -> {target_path(asset, args.root)}")
        print(f"assets: {len(assets)}")
        return 0

    if not verify_only and not revision:
        if offline:
            parser.error("--offline needs a local revision or --verify-only")
        revision = resolve_revision(repo_id=args.repo)
        print(f"resolved upstream revision: {revision}")
        print("pin it in GTR_HF_REVISION for a reproducible URL")

    if verify_only:
        failures = 0
        missing = 0
        for asset in assets:
            try:
                verify_asset(asset, args.root)
                print(f"OK       {asset.name}")
            except Exception as error:  # noqa: BLE001 - report per asset
                failures += 1
                path = target_path(asset, args.root)
                if not path.is_file():
                    missing += 1
                print(f"FAILED   {asset.name}: {error}")
        print(
            f"verified={len(assets) - failures} failed={failures} "
            f"missing={missing}"
        )
        return 1 if failures else 0

    assert revision is not None
    downloaded = 0
    skipped = 0
    failures = 0
    for asset in assets:
        try:
            path, fetched = download_gtr_asset(
                asset,
                revision=revision,
                root=args.root,
                force=args.force,
                offline=offline,
                repo_id=args.repo,
            )
            if fetched:
                downloaded += 1
                print(f"DOWNLOAD {asset.name} -> {path}")
            else:
                skipped += 1
                print(f"OK       {asset.name} (already verified)")
        except Exception as error:  # noqa: BLE001 - report per asset
            failures += 1
            print(f"FAILED   {asset.name}: {error}")

    print(
        f"revision={revision} downloaded={downloaded} "
        f"skipped={skipped} failed={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
