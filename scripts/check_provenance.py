#!/usr/bin/env python3
"""Provenance guard for the Autovision repository (design doc 28, section 7.2).

Two modes:

Mode 1 (default) — banned-marker scan.
    Scans ``src/``, ``tests/`` and ``scripts/`` for markers tied to the
    reference system (brand names, AGPL imports, reference asset hosts,
    historical registry constant names). Files exempted in
    ``provenance_allowlist.txt`` (repo root) are skipped; entries carry a
    reason and a review date, and stale entries (matching no file) fail
    the run unless ``--allow-unused-allowlist`` is given. Exit code 1 when
    unexempted hits exist.

Mode 2 (``--reference-dir DIR``) — token-similarity audit.
    Compares this repository's Python sources against a reference source
    tree after stripping comments, docstrings and string literals and
    normalizing package names. Reports every pair whose similarity exceeds
    ``--threshold`` (default 0.25). Pass ``--fail-over RATIO`` to make
    threshold exceedances fail the run (default: report only). Files
    exempted in the allowlist are skipped here too. A cheap multiset
    containment prefilter (a strict upper bound on the difflib ratio)
    keeps the full-tree sweep tractable.

Usage:
    python scripts/check_provenance.py
    python scripts/check_provenance.py --reference-dir /path/to/ref/src/pkg
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import re
import sys
import tokenize
from collections import Counter
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "tests", "scripts")
ALLOWLIST_PATH = REPO_ROOT / "provenance_allowlist.txt"

#: (pattern, reason) — matched case-insensitively over every scanned file.
#: The historical registry constant and asset host are spelled via
#: concatenation so this file never matches its own patterns.
BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"lightly[_-]?train", "reference-system name"),
    (r"\blightly\b", "reference-system brand"),
    (r"from\s+lightly\b|import\s+lightly\b", "AGPL package import"),
    ("DOWNLOADABLE_" + "MODEL_URL_AND_HASH", "historical registry constant"),
    ("lightly-" + "train-checkpoints", "reference asset host"),
    (r"s3\.us-east-1\.amazonaws\.com", "reference asset host"),
    (r"Copyright \(c\) Lightly", "reference copyright header"),
]

_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}
#: The guard's own pattern definitions contain banned markers by design.
_SELF_EXEMPT = "scripts/check_provenance.py"
_MAX_SCAN_BYTES = 2_000_000


def load_allowlist() -> list[tuple[str, str]]:
    """Parse ``path :: reason`` lines; blank lines and ``#`` comments skipped."""
    entries: list[tuple[str, str]] = []
    if not ALLOWLIST_PATH.exists():
        return entries
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        path, sep, reason = stripped.partition("::")
        entries.append((path.strip(), reason.strip() if sep else ""))
    return entries


def is_exempt(rel_path: str, allowlist: list[tuple[str, str]]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern, _ in allowlist)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if _SKIP_DIRS.intersection(path.parts):
                continue
            if any(part.endswith(".egg-info") for part in path.parts):
                continue  # build artifacts, not source
            if path.relative_to(REPO_ROOT).as_posix() == _SELF_EXEMPT:
                continue  # the guard's own pattern list matches by design
            if path.stat().st_size > _MAX_SCAN_BYTES:
                continue
            files.append(path)
    return files


def scan_markers(
    allowlist: list[tuple[str, str]], check_stale: bool = True
) -> list[str]:
    problems: list[str] = []
    scanned_rel: set[str] = set()
    for path in iter_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        scanned_rel.add(rel)
        exempt = is_exempt(rel, allowlist)
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, reason in BANNED_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    if exempt:
                        break
                    problems.append(
                        f"{rel}:{line_number}: banned marker ({reason}): "
                        f"{line.strip()[:120]}"
                    )
    if check_stale:
        for pattern, _ in allowlist:
            if not any(fnmatch.fnmatch(rel, pattern) for rel in scanned_rel):
                problems.append(
                    "provenance_allowlist.txt: entry matches no scanned "
                    f"file (remove it): {pattern}"
                )
    return problems


# ── mode 2: token-similarity audit ──────────────────────────────────────

_PACKAGE_NAMES = re.compile(r"lightly_train|lightlytrain|autovision")


def python_tokens(path: Path) -> list[str] | None:
    """Token stream minus comments/strings/whitespace; names normalized.

    Returns ``None`` when the file cannot be tokenized (reported, not
    silently dropped).
    """
    try:
        source = path.read_bytes()
        tokens = tokenize.tokenize(BytesIO(source).readline)
        out: list[str] = []
        for token in tokens:
            if token.type in (
                tokenize.COMMENT,
                tokenize.STRING,
                tokenize.ENCODING,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                continue
            out.append(_PACKAGE_NAMES.sub("PKG", token.string))
        return out
    except (tokenize.TokenError, OSError, UnicodeDecodeError, SyntaxError):
        return None


def _multiset_containment(a: Counter, b: Counter) -> float:
    """``|A ∩ B| / min(|A|, |B|)`` — a strict upper bound on the difflib
    ratio (SequenceMatcher can never match more tokens than the multiset
    intersection), so it is a lossless prefilter for a ratio threshold."""
    intersection = sum((a & b).values())
    smaller = min(sum(a.values()), sum(b.values()))
    return intersection / smaller if smaller else 0.0


def similar_pairs(
    ours: dict[Path, list[str]],
    theirs: dict[Path, list[str]],
    threshold: float,
) -> list[tuple[Path, Path, float]]:
    pairs: list[tuple[Path, Path, float]] = []
    ref_counters = {path: Counter(tokens) for path, tokens in theirs.items()}
    total = len(ours)
    for index, (our_path, our_tokens) in enumerate(sorted(ours.items()), start=1):
        if not our_tokens:
            continue
        if index % 25 == 0 or index == total:
            print(f"  ... {index}/{total} files", file=sys.stderr)
        our_counter = Counter(our_tokens)
        for ref_path in theirs:
            ref_tokens = theirs[ref_path]
            if not ref_tokens:
                continue
            if _multiset_containment(our_counter, ref_counters[ref_path]) < threshold:
                continue
            matcher = difflib.SequenceMatcher(
                None, our_tokens, ref_tokens, autojunk=False
            )
            if matcher.real_quick_ratio() < threshold:
                continue
            if matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                pairs.append((our_path, ref_path, ratio))
    return pairs


def run_reference_audit(
    reference_dir: Path,
    threshold: float,
    fail_over: float | None,
    allowlist: list[tuple[str, str]],
) -> int:
    skipped: list[str] = []

    def tokenize_tree(tree: Path) -> dict[Path, list[str]]:
        result: dict[Path, list[str]] = {}
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tokens = python_tokens(path)
            if tokens is None:
                skipped.append(str(path))
            elif tokens:
                result[path] = tokens
        return result

    ours = tokenize_tree(REPO_ROOT / "src")
    theirs = tokenize_tree(reference_dir)
    pairs = similar_pairs(ours, theirs, threshold)
    if allowlist:
        pairs = [
            (our_path, ref_path, ratio)
            for our_path, ref_path, ratio in pairs
            if not is_exempt(our_path.relative_to(REPO_ROOT).as_posix(), allowlist)
        ]
    pairs.sort(key=lambda item: item[2], reverse=True)
    print(f"comparing {len(ours)} files against {len(theirs)} reference files")
    if skipped:
        print(f"skipped {len(skipped)} untokenizable file(s):")
        for path in skipped[:10]:
            print(f"  {path}")
    for our_path, ref_path, ratio in pairs:
        print(f"{ratio:6.1%}  {our_path.relative_to(REPO_ROOT)}  <->  {ref_path}")
    if not pairs:
        print(f"no pair at or above {threshold:.0%}")
    if fail_over is not None and pairs and pairs[0][2] > fail_over:
        print(
            f"FAIL: highest similarity {pairs[0][2]:.1%} exceeds "
            f"--fail-over {fail_over:.0%}"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="reference source tree for the similarity audit (mode 2)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="similarity report threshold"
    )
    parser.add_argument(
        "--fail-over",
        type=float,
        default=None,
        metavar="RATIO",
        help="fail (exit 1) when any similarity pair exceeds RATIO",
    )
    parser.add_argument(
        "--allow-unused-allowlist",
        action="store_true",
        help="do not flag allowlist entries that match no scanned file",
    )
    options = parser.parse_args(argv)

    allowlist = load_allowlist()
    if options.reference_dir is not None:
        if not options.reference_dir.is_dir():
            parser.error(f"--reference-dir is not a directory: {options.reference_dir}")
        return run_reference_audit(
            options.reference_dir,
            options.threshold,
            options.fail_over,
            allowlist,
        )

    problems = scan_markers(
        allowlist, check_stale=not options.allow_unused_allowlist
    )
    if problems:
        print(f"{len(problems)} provenance problem(s):")
        for problem in problems:
            print(f"  {problem}")
        print(
            "Fix the source, or add a time-boxed entry to "
            "provenance_allowlist.txt if the marker documents pending "
            "remediation work."
        )
        return 1
    print("provenance scan clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
