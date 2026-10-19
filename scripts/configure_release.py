#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Apply the public release URLs (repo + docs site) to the metadata files.

The URLs cannot be guessed by the framework; run this once the public
repository and docs domain are known, then commit the resulting metadata.

    python scripts/configure_release.py \
        --repo-url https://github.com/ORG/AutoVision \
        --docs-url https://ORG.github.io/AutoVision/

    python scripts/configure_release.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalize(repo_url: str, docs_url: str) -> tuple[str, str]:
    repo = repo_url.strip().rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    docs = docs_url.strip()
    if not docs.endswith("/"):
        docs += "/"
    for name, value in (("repo", repo), ("docs", docs)):
        if not value.startswith("https://"):
            raise SystemExit(f"--{name}-url must start with https://")
    return repo, docs


def _project_urls_block(repo: str, docs: str) -> str:
    return (
        "[project.urls]\n"
        f'Homepage = "{repo}"\n'
        f'Documentation = "{docs}"\n'
        f'Repository = "{repo}"\n'
        f'Issues = "{repo}/issues"\n'
        f'Changelog = "{repo}/blob/main/CHANGELOG.md"\n'
    )


def _write_pyproject(text: str, repo: str, docs: str) -> str:
    block = _project_urls_block(repo, docs)
    pattern = re.compile(r"(?ms)^\[project\.urls\]\s*\n.*?(?=^\[|\Z)")
    if pattern.search(text):
        return pattern.sub(block + "\n", text, count=1)
    anchor = "[project.scripts]"
    if anchor not in text:
        raise SystemExit("pyproject.toml has no [project.scripts] anchor")
    return text.replace(anchor, block + "\n" + anchor, 1)


def _write_mkdocs(text: str, repo: str, docs: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith("site_url:") and not line.startswith("repo_url:")
    ]
    anchor = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("site_description:")
        ),
        None,
    )
    if anchor is None:
        anchor = next(
            (index for index, line in enumerate(lines) if line.startswith("site_name:")),
            None,
        )
    if anchor is None:
        raise SystemExit("mkdocs.yml has no site_name/site_description anchor")
    lines[anchor + 1 : anchor + 1] = [
        f"site_url: {docs}",
        f"repo_url: {repo}",
    ]
    return "\n".join(lines) + "\n"


def _configured(pyproject: str, mkdocs: str) -> list[str]:
    problems: list[str] = []
    match = re.search(
        r"(?ms)^\[project\.urls\]\s*\n(.*?)(?=^\[|\Z)", pyproject
    )
    if match is None:
        problems.append("pyproject.toml is missing [project.urls]")
    else:
        block = match.group(0)
        if "ORG" in block or "example." in block:
            problems.append("pyproject.toml still contains placeholder URLs")
        for key in ("Repository", "Documentation"):
            if key not in block:
                problems.append(f"[project.urls] is missing {key}")
    for key in ("site_url:", "repo_url:"):
        if not any(line.startswith(key) for line in mkdocs.splitlines()):
            problems.append(f"mkdocs.yml is missing {key}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--docs-url", default=None)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    pyproject_path = root / "pyproject.toml"
    mkdocs_path = root / "mkdocs.yml"
    if not pyproject_path.is_file() or not mkdocs_path.is_file():
        raise SystemExit(f"{root} is not a repository root")
    pyproject = pyproject_path.read_text()
    mkdocs = mkdocs_path.read_text()

    if args.check:
        problems = _configured(pyproject, mkdocs)
        if problems:
            for problem in problems:
                print(f"MISSING {problem}")
            return 1
        print("release URLs configured")
        return 0

    if not args.repo_url or not args.docs_url:
        parser.error("--repo-url and --docs-url are required (or use --check)")
    repo, docs = _normalize(args.repo_url, args.docs_url)
    pyproject_path.write_text(_write_pyproject(pyproject, repo, docs))
    mkdocs_path.write_text(_write_mkdocs(mkdocs, repo, docs))
    print(f"repository: {repo}")
    print(f"documentation: {docs}")
    print(f"updated: {pyproject_path}, {mkdocs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
