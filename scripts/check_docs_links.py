#!/usr/bin/env python3
"""Public-tree link guard for the AutoVision repository (E1 release package).

Two classes of broken references are failures:

1. **Relative Markdown links** — ``[text](target)`` / ``![alt](target)``
   must resolve to an existing file or directory. External URLs
   (``scheme:``), pure anchors (``#...``) and query strings are skipped;
   anchor fragments are stripped before resolution (anchor slugs are
   renderer-specific and deliberately out of scope). Inline-code spans and
   HTML comments on a line are not scanned for links (they never render).
2. **Backticked repository paths** — a ``X/...path.py|md`` token inside
   backticks must exist, resolving at the repository root first and then
   under ``src/autovision/`` (the module-map convention used by
   ``docs/framework_design.md``); only paths whose first segment anchors
   to one of those two roots are checked, so illustrative output paths
   pass (see ``_is_repo_path_reference``). A trailing ``:12`` / ``:12-34``
   line suffix is allowed. This keeps prose references honest, e.g. the
   benchmark protocol's "自家数字必附复现脚本" promise in
   ``docs/benchmarks.md``.

Fenced code blocks are skipped for class 1 (sample invocations inside
them are not links) but not for class 2 (a code sample that names a repo
script should still name a real one). A line carrying the literal marker
``<!-- docpath-ok -->`` is exempt from class 2 — for legitimate historical
mentions ("the old ``scripts/foo.py`` was renamed"), the same spirit as
the provenance allowlist: an explicit, reviewable exemption.

Deliberately NOT covered: reference-style links (``[text][ref]``), anchors,
external URL reachability, and bare file names without a directory
(``variants.py`` — under-determined, could be prose).

Scanned: every ``*.md`` in the tree except the skip list below (build
output, caches, virtualenvs, data/run directories).

Usage:
    python scripts/check_docs_links.py          # exit 1 on any broken reference
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directory names pruned from the scan (any path segment match).
SKIP_DIRS = {
    ".autovision-cache",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "node_modules",
    "output",
    "runs",
    "site",
}

#: ``[text](target)`` / ``![alt](target)`` with an optional title, target
#: either ``<angle-bracketed>`` or a plain token.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>\n]*)>|([^)\s\"]*))")
#: Backticked repo path with at least one slash, optional ``:line`` suffix.
_BACKTICK_PATH_RE = re.compile(
    r"`([\w][\w.-]*/[\w./-]*\.(?:py|md))(?::\d+(?:-\d+)?)?`"
)
#: Stripped before class-1 scanning (never rendered as links).
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->")
#: Per-line exemption marker for legitimate historical path mentions.
_DOCPATH_OK = "<!-- docpath-ok -->"
#: Package root for the module-map path convention.
_PACKAGE_ROOT = Path("src") / "autovision"


def _iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*.md")):
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        files.append(path)
    return files


def _strip_fences(text: str) -> list[tuple[int, str, bool]]:
    """``(line_number, line, inside_fence)`` for every line of ``text``."""
    rows: list[tuple[int, str, bool]] = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            rows.append((number, line, True))
            continue
        if fence is not None and stripped.startswith(fence):
            fence = None
            rows.append((number, line, True))
            continue
        rows.append((number, line, fence is not None))
    return rows


def _repo_path_exists(path: str) -> bool:
    """Repo-root path, else the ``src/autovision/`` module-map convention."""
    return (REPO_ROOT / path).exists() or (REPO_ROOT / _PACKAGE_ROOT / path).exists()


def _is_repo_path_reference(path: str) -> bool:
    """Whether a backticked path refers to the repository at all.

    The FIRST segment must exist at the repository root (``docs/...``) or
    under the package root (``cli/...`` -> ``src/autovision/cli``). That
    keeps the guard strict for real repository references — including
    files deleted under an existing directory — while letting illustrative
    paths pass (``report/report.md`` as an output example is not a
    repository path, and no ``report/`` tree exists to anchor it).
    """
    first = path.split("/", 1)[0]
    return (REPO_ROOT / first).exists() or (REPO_ROOT / _PACKAGE_ROOT / first).exists()


def _link_resolves(link_target: str, source: Path) -> bool:
    target = unquote(link_target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return True
    if "://" in target or target.startswith(("mailto:", "tel:")):
        return True
    candidate = (
        REPO_ROOT / target.lstrip("/") if target.startswith("/") else source.parent / target
    )
    return candidate.exists()


def check() -> list[str]:
    problems: list[str] = []
    for source in _iter_markdown_files():
        relative = source.relative_to(REPO_ROOT)
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            problems.append(f"{relative}:1: not valid UTF-8 ({error})")
            continue
        for number, line, inside_fence in _strip_fences(text):
            if not inside_fence:
                link_line = _INLINE_CODE_RE.sub("", _HTML_COMMENT_RE.sub("", line))
                for match in _LINK_RE.finditer(link_line):
                    target = match.group(1) if match.group(1) is not None else match.group(2)
                    if target.startswith("#"):
                        continue
                    if not _link_resolves(target, source):
                        problems.append(f"{relative}:{number}: broken link -> {target}")
            if _DOCPATH_OK in line:
                continue
            for match in _BACKTICK_PATH_RE.finditer(line):
                path = match.group(1)
                if not _is_repo_path_reference(path):
                    continue  # illustrative path, not a repository reference
                if not _repo_path_exists(path):
                    problems.append(f"{relative}:{number}: missing repo path -> {path}")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("Broken references in the public tree:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("public-tree links OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
