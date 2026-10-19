# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Content-addressed dataset snapshots (roadmap 数据版本化).

The mmap cache key is a path+mtime fingerprint — cheap, but unstable across
copies and blind to content edits. A **snapshot** is the versioning-grade
artifact: for every split it records the referenced roots, the file
count/bytes and a sha256 over the sorted ``(relative path, size)`` manifest,
plus — with ``deep=True`` — a sha256 over the file CONTENTS.
:func:`verify_dataset_snapshot` re-derives the same records and reports
precise differences (added / removed / resized / content-changed), so a
training run can pin exactly the dataset it consumed and fail loudly when
the data drifts.

Portability: digests are computed over paths RELATIVE to each root, so a
snapshot verifies after the tree is moved or copied; ``deep=True``
additionally requires identical file contents. Relative paths in a data
spec resolve against the spec FILE's directory (the rule training uses —
the snapshot records that base directory too); inline dict specs must use
absolute paths.

Scope note: directory symbolic links are not descended (``Path.rglob``
semantics) and link identity is not recorded — snapshot the real trees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError

__all__ = [
    "dataset_snapshot",
    "load_snapshot",
    "verify_dataset_snapshot",
]

logger = logging.getLogger("autovision")

_CHUNK = 1 << 20
#: Trees with more files than this get their per-file manifest omitted from
#: the snapshot JSON (the digest still covers every file; the entry carries
#: ``manifest_omitted`` and verification then reports split-level
#: differences without naming the files).
_MAX_MANIFEST_ENTRIES = 200_000
#: Drift detail lists are capped at this many names (``truncated`` is set).
_MAX_DETAIL_ENTRIES = 100


def _coerce_bool(value: Any, *, name: str) -> bool:
    """Strict boolean — the CLI passes strings straight from ``deep=true``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise AutoVisionError(f"{name} must be a boolean (true/false), got {value!r}.")


def _load_spec(data: Any) -> tuple[dict[str, Any], Path | None]:
    """``(spec, base_dir)`` — ``base_dir`` is the spec file's directory.

    Relative paths inside a spec read from disk resolve against that
    directory (exactly like the training entry); the base directory is
    recorded in the snapshot so verification works from any CWD.
    """
    if isinstance(data, dict):
        return dict(data), None
    if isinstance(data, (str, Path)):
        if str(data) == "coco8":
            # The framework-wide literal for the built-in smoke dataset.
            from autovision.data.coco8 import ensure_coco8

            return dict(ensure_coco8()), None
        path = Path(data)
        if not path.is_file():
            raise AutoVisionError(
                f"data= must be a mapping or a path to an existing yaml/json "
                f"data spec; got '{data}'."
            )
        try:
            text = path.read_text()
        except OSError as error:
            raise AutoVisionError(f"Could not read data spec '{path}': {error}.") from error
        if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise AutoVisionError(
                    f"Data spec '{path}' is not valid JSON: {error}."
                ) from error
        else:
            import yaml

            try:
                payload = yaml.safe_load(text)
            except yaml.YAMLError as error:
                raise AutoVisionError(
                    f"Data spec '{path}' is not valid YAML: {error}."
                ) from error
        if not isinstance(payload, dict):
            raise AutoVisionError(f"Data spec '{path}' must hold a mapping.")
        return payload, path.parent.resolve()
    raise AutoVisionError(f"Unsupported data spec type: {type(data).__name__}.")


def _collect_roots(
    spec: dict[str, Any],
    *,
    base_dir: Path | None = None,
    strict: bool = False,
) -> dict[str, list[tuple[str, Path]]]:
    """Split name → [(field label, existing path)] from a data spec.

    Every string value that names an existing file or directory is a root;
    relative paths resolve against ``base_dir`` (the spec file's directory).
    Split names come from the top-level mapping entries; top-level path
    values group under ``"(shared)"``. Labels must be unique per split —
    the manifest key is ``label/relative-path``. With ``strict=True`` a path
    that does not exist is an error instead of being silently omitted.
    """
    groups: dict[str, list[tuple[str, Path]]] = {}
    missing: list[str] = []

    def collect(node: Any, split: str, label: str = "") -> None:
        if isinstance(node, str):
            candidate = Path(node)
            if not candidate.is_absolute() and base_dir is not None:
                candidate = base_dir / candidate
            if candidate.exists():
                groups.setdefault(split, []).append((label or node, candidate))
            else:
                missing.append(f"{split}/{label or node}: '{candidate}'")
            return
        if isinstance(node, dict):
            for key, value in node.items():
                collect(value, split, str(key))
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                collect(value, split, f"{label}[{index}]")

    for key, value in spec.items():
        if key == "format":
            continue
        if isinstance(value, dict):
            collect(value, str(key))
        else:
            collect(value, "(shared)")
    if not groups or all(not roots for roots in groups.values()):
        raise AutoVisionError(
            "The data spec references no existing paths — pass a spec whose "
            "annotations/images entries point at real files or directories "
            "(relative paths resolve against the spec file's directory)."
        )
    if strict and missing:
        raise AutoVisionError(
            "The data spec references missing path(s): "
            + "; ".join(missing[:10])
            + ("; ..." if len(missing) > 10 else "")
        )
    for name, roots in groups.items():
        labels = [label for label, _ in roots]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise AutoVisionError(
                f"Split '{name}' has duplicate root label(s) {duplicates}: the "
                "manifest key is '<label>/<relative path>', so labels must be "
                "unique — rename the spec keys (e.g. part_a/part_b)."
            )
    return groups


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file())


def _digest_files(
    roots: list[tuple[str, Path]], *, deep: bool
) -> tuple[int, int, str, str | None, list[list[Any]]]:
    """``(files, bytes, digest, content_digest, manifest)`` for the roots.

    Every manifest row carries its own resolved file path, so the deep
    content hash never has to re-derive a root from a label.
    """
    rows: list[tuple[str, int, Path]] = []
    total_bytes = 0
    digest = hashlib.sha256()
    content_digest = hashlib.sha256() if deep else None
    started = time.perf_counter()
    for label, root in sorted(roots, key=lambda item: item[0]):
        for path in _iter_files(root):
            size = path.stat().st_size
            relative = path.relative_to(root if root.is_dir() else root.parent)
            rows.append((f"{label}/{relative.as_posix()}", size, path))
            total_bytes += size
    rows.sort(key=lambda row: (row[0], row[1]))
    for key, size, _path in rows:
        digest.update(f"{key}\0{size}\n".encode())
    manifest = [[key, size] for key, size, _path in rows]
    if deep and content_digest is not None:
        for _key, _size, path in rows:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b""):
                    content_digest.update(chunk)
        logger.info(
            "Deep-hashed %d files (%.1f MB) in %.1fs.",
            len(rows), total_bytes / 1e6, time.perf_counter() - started,
        )
    return (
        len(rows),
        total_bytes,
        digest.hexdigest(),
        content_digest.hexdigest() if content_digest is not None else None,
        manifest,
    )


def dataset_snapshot(
    data: Any,
    out_json: "str | Path",
    *,
    deep: bool | str = False,
    note: str | None = None,
    overwrite: bool | str = False,
    strict: bool | str = True,
) -> dict[str, Any]:
    """Write a content-addressed snapshot of the dataset ``data`` refers to.

    ``deep=True`` additionally hashes every file's CONTENTS (slower; use it
    for versioning-grade pins of small/medium corpora). An existing
    ``out_json`` is never replaced silently — pass ``overwrite=true``.
    ``strict=True`` (default) fails on a declared path that does not exist,
    so a typo cannot produce a snapshot that silently omits a split.
    """
    from autovision import __version__

    deep_flag = _coerce_bool(deep, name="deep")
    overwrite_flag = _coerce_bool(overwrite, name="overwrite")
    strict_flag = _coerce_bool(strict, name="strict")
    spec, base_dir = _load_spec(data)
    out_path = Path(out_json)
    if out_path.exists() and not overwrite_flag:
        raise AutoVisionError(
            f"Snapshot '{out_path}' already exists — pass overwrite=true to "
            "replace it (a snapshot pins the data; replacing it is deliberate)."
        )
    groups = _collect_roots(spec, base_dir=base_dir, strict=strict_flag)
    splits: dict[str, Any] = {}
    for name, roots in sorted(groups.items()):
        if not roots:
            continue
        files, total_bytes, digest, content_digest, manifest = _digest_files(
            roots, deep=deep_flag
        )
        entry: dict[str, Any] = {
            "roots": [f"{label}:{path}" for label, path in sorted(roots)],
            "files": files,
            "bytes": total_bytes,
            "digest": digest,
        }
        if content_digest is not None:
            entry["content_digest"] = content_digest
        if len(manifest) <= _MAX_MANIFEST_ENTRIES:
            entry["manifest"] = manifest
        else:
            entry["manifest_omitted"] = True
            logger.info(
                "Split '%s' has %d files (> %d): per-file manifest omitted "
                "(the digest still covers every file).",
                name, len(manifest), _MAX_MANIFEST_ENTRIES,
            )
        splits[name] = entry
    payload = {
        "kind": "autovision-dataset-snapshot",
        "autovision_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "deep": deep_flag,
        "note": note,
        "spec": spec,
        "spec_base_dir": str(base_dir) if base_dir is not None else None,
        "splits": splits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Dataset snapshot written to '%s'.", out_path)
    return payload


def load_snapshot(snapshot: "str | Path | dict[str, Any]") -> dict[str, Any]:
    if isinstance(snapshot, dict):
        payload = snapshot
    else:
        path = Path(snapshot)
        if not path.is_file():
            raise AutoVisionError(f"Snapshot not found: '{path}'.")
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AutoVisionError(
                f"Snapshot '{path}' is not readable JSON: {error}."
            ) from error
    if not isinstance(payload, dict):
        raise AutoVisionError("A snapshot must be a JSON object.")
    if payload.get("kind") != "autovision-dataset-snapshot":
        raise AutoVisionError(
            "Not an AutoVision dataset snapshot (missing the 'kind' marker)."
        )
    return payload


def verify_dataset_snapshot(
    snapshot: "str | Path | dict[str, Any]",
    *,
    deep: bool | str | None = None,
    spec_override: Any = None,
) -> dict[str, Any]:
    """Re-derive the snapshot's records and report the differences.

    Returns ``{"ok": bool, "checked": ..., "differences": [...]}`` where a
    difference is ``{"split", "field", "expected", "actual"}`` — plus
    ``added``/``removed``/``resized`` name lists (field ``"manifest"``) when
    the snapshot carries a per-file manifest. ``deep`` defaults to the
    snapshot's own mode; requesting ``deep=True`` on a SHALLOW snapshot
    fails loudly (there are no recorded content digests, so the check would
    be vacuous — re-snapshot with ``deep=true``). ``spec_override`` verifies
    a MOVED or COPIED tree: pass the copy's own data spec — the digests are
    computed over root-relative paths, so an identical copy verifies clean.
    """
    payload = load_snapshot(snapshot)
    deep_mode = (
        _coerce_bool(payload.get("deep", False), name="snapshot 'deep'")
        if deep is None
        else _coerce_bool(deep, name="deep")
    )
    if deep_mode and not any(
        "content_digest" in entry for entry in payload["splits"].values()
    ):
        raise AutoVisionError(
            "deep verification needs a deep snapshot: this one records no "
            "content digests, so the content check would be vacuous — "
            "re-snapshot with deep=true to pin file contents."
        )
    if spec_override is not None:
        spec, base_dir = _load_spec(spec_override)
    else:
        spec, base_dir = _load_spec(payload["spec"])
        if base_dir is None and payload.get("spec_base_dir"):
            base_dir = Path(payload["spec_base_dir"])
    groups = _collect_roots(spec, base_dir=base_dir)
    differences: list[dict[str, Any]] = []
    checked: list[str] = []
    for name, recorded in sorted(payload["splits"].items()):
        checked.append(name)
        roots = groups.get(name)
        if not roots:
            differences.append(
                {"split": name, "field": "missing", "expected": "present", "actual": None}
            )
            continue
        files, total_bytes, digest, content_digest, manifest = _digest_files(
            roots, deep=deep_mode
        )
        for field_name, expected, actual in (
            ("files", recorded["files"], files),
            ("bytes", recorded["bytes"], total_bytes),
            ("digest", recorded["digest"], digest),
        ):
            if expected != actual:
                differences.append(
                    {"split": name, "field": field_name, "expected": expected, "actual": actual}
                )
        recorded_content = recorded.get("content_digest")
        if deep_mode and recorded_content is not None and recorded_content != content_digest:
            differences.append(
                {
                    "split": name,
                    "field": "content_digest",
                    "expected": recorded_content,
                    "actual": content_digest,
                }
            )
        recorded_manifest = recorded.get("manifest")
        if recorded_manifest:
            before = {row[0]: row[1] for row in recorded_manifest}
            after = {key: size for key, size in manifest}
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            resized = sorted(
                key for key in set(before) & set(after) if before[key] != after[key]
            )
            if added or removed or resized:
                detail: dict[str, Any] = {
                    "split": name,
                    "field": "manifest",
                    "added": added[:_MAX_DETAIL_ENTRIES],
                    "removed": removed[:_MAX_DETAIL_ENTRIES],
                    "resized": resized[:_MAX_DETAIL_ENTRIES],
                }
                if max(len(added), len(removed), len(resized)) > _MAX_DETAIL_ENTRIES:
                    detail["truncated"] = True
                differences.append(detail)
    return {"ok": not differences, "checked": checked, "differences": differences}
