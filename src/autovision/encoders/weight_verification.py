# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Verified pretrained-weight loading (supply-chain integrity, §3.2/§6.3).

Every pretrained download goes through :func:`load_pretrained_verified`:

1. ``AUTOVISION_MIRROR_URL`` is bridged to the HF endpoint mechanism when the
   process has not already configured ``HF_ENDPOINT`` (the mirror consumer).
   The bridge runs at import time of this module — ``huggingface_hub``
   freezes its endpoint at import, so later bridging cannot redirect the
   download it precedes.
2. ``AUTOVISION_OFFLINE=1`` resolves from the local HF cache only; a missing
   snapshot raises an actionable error instead of attempting network I/O.
3. The snapshot's weight files are sha256-verified against the manifest in
   :mod:`autovision.encoders.weight_hashes`. Pinned hashes verify fully
   offline; unpinned entries fall back to the LFS sha256 published in the
   registry metadata (transfer-integrity tier).
4. On a hash mismatch the repo cache is deleted and the download retried
   once; a second mismatch raises :class:`WeightVerificationError`.
5. ``AUTOVISION_WEIGHT_SOURCE`` selects the download source: ``auto``
   (default) uses the hub — honoring ``AUTOVISION_MIRROR_URL``, e.g.
   hf-mirror.com — and falls back to a pinned ModelScope mirror when the
   hub path fails; ``modelscope`` forces the mirror; ``huggingface``
   disables fallback. The sha256 flow is identical for every source
   because verified mirrors are byte-identical.

The manifest is machine-pinned via ``scripts/update_weight_hashes.py`` so
the pinned tier grows over time without trusting first use.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError, CheckpointError
from autovision.runtime.env import Env

__all__ = [
    "WeightArtifact",
    "WeightVerificationError",
    "get_weight_artifact",
    "load_pretrained_verified",
]

logger = logging.getLogger("autovision")


def _bridge_mirror_to_hf_endpoint() -> None:
    """Propagate ``AUTOVISION_MIRROR_URL`` to ``HF_ENDPOINT``.

    Must run before the first ``huggingface_hub`` import anywhere in the
    process: hub freezes its endpoint constant at import time (verified on
    hub 1.x — setting ``HF_ENDPOINT`` afterwards has no effect), so a
    bridge inside :func:`load_pretrained_verified` would be too late for
    the very download it is meant to route.
    """
    mirror_url = Env.MIRROR_URL.value
    if mirror_url and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror_url
        # Xet-backed repos bypass HF_ENDPOINT entirely (the CAS
        # reconstruction server is a separate host, which then 401s
        # without hub credentials). A mirror cannot serve Xet, so disable
        # the protocol and let downloads fall back to plain HTTP resolve —
        # which the mirror DOES serve (and the pinned sha256 still
        # verifies every byte).
        if not os.getenv("HF_HUB_DISABLE_XET"):
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        logger.info("Using weight mirror: %s", mirror_url)


def bridge_hub_env() -> None:
    """Point third-party hub consumers (timm/transformers) at the mirror.

    Complements :func:`_bridge_mirror_to_hf_endpoint` (already applied at
    import) with the offline switch: ``AUTOVISION_OFFLINE=1`` maps to
    ``HF_HUB_OFFLINE=1`` so ``timm.create_model(pretrained=True)`` and
    friends resolve cache-only instead of reaching the network. Callers
    must invoke this BEFORE the consumer library is first imported (same
    import-time-freeze caveat as the mirror bridge).
    """
    if Env.OFFLINE.value and not os.getenv("HF_HUB_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("AUTOVISION_OFFLINE=1 -> HF_HUB_OFFLINE=1 (cache-only hub)")


_bridge_mirror_to_hf_endpoint()

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")  # for str.endswith (no glob support)
_SNAPSHOT_PATTERNS = ("config.json", "*.safetensors", "*.bin")
_RETRY_MESSAGE = (
    "If the failure persists the download source may be compromised or "
    "truncated: set AUTOVISION_MIRROR_URL to a trusted mirror, or download the "
    "model manually and pass a local directory via model_args."
)


class WeightVerificationError(AutoVisionError):
    """A pretrained weight file failed sha256 verification after one retry."""


@dataclass(frozen=True)
class WeightArtifact:
    """Supply-chain metadata for one pretrained checkpoint repository.

    ``sha256`` maps weight filenames to pinned hashes; an empty dict means
    the entry is unpinned and verification falls back to the LFS sha256
    from the registry metadata (see module docstring). ``modelscope_id``
    optionally names a byte-identical mirror repository on ModelScope
    (modelscope.cn), selectable via ``AUTOVISION_WEIGHT_SOURCE=modelscope``
    — pinned hashes are source-agnostic, so verification is unchanged.
    """

    repo_id: str
    license: str
    source: str = "huggingface"
    revision: str | None = None
    sha256: dict[str, str] = field(default_factory=dict)
    modelscope_id: str | None = None


def get_weight_artifact(repo_id: str) -> WeightArtifact | None:
    """Manifest entry for ``repo_id`` (``None`` for unlisted repositories)."""
    # Late import: the manifest data module re-exports WeightArtifact from
    # here, so the dependency only points one way at call time.
    from autovision.encoders.weight_hashes import WEIGHT_MANIFEST

    return WEIGHT_MANIFEST.get(repo_id)


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file (constant memory for multi-GB weights)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_sha256(repo_id: str, revision: str | None) -> dict[str, str]:
    """Registry-published LFS sha256 per weight filename (network tier)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    hashes: dict[str, str] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if lfs and sibling.rfilename.endswith((".safetensors", ".bin")):
            hashes[sibling.rfilename] = lfs.sha256
    return hashes


def _delete_repo_cache(snapshot_dir: Path) -> None:
    """Delete the HF cache entry (``models--org--name``) for a snapshot."""
    for parent in snapshot_dir.parents:
        if parent.name.startswith("models--"):
            shutil.rmtree(parent, ignore_errors=True)
            logger.info("Deleted corrupted weight cache '%s'.", parent)
            return
    # Snapshot outside the HF cache layout (test/custom dir): remove the
    # snapshot itself so the retry re-downloads.
    shutil.rmtree(snapshot_dir, ignore_errors=True)


def _verify_snapshot(
    snapshot_dir: Path, artifact: WeightArtifact, source: str = "huggingface"
) -> str:
    """Verify every weight file in the snapshot.

    Returns the verification tier ("pinned" or "lfs-metadata"); raises
    :class:`WeightVerificationError` on the first mismatch. ``source``
    selects where unpinned hashes are fetched from (hub LFS metadata or
    the ModelScope file listing).
    """
    weight_files = sorted(
        path
        for pattern in _WEIGHT_PATTERNS
        for path in snapshot_dir.glob(pattern)
        if path.is_file()
    )
    if not weight_files:
        raise WeightVerificationError(
            f"Snapshot of '{artifact.repo_id}' contains no weight files "
            f"matching {_WEIGHT_PATTERNS}; the download is incomplete."
        )
    unpinned = [
        path.name for path in weight_files if path.name not in artifact.sha256
    ]
    registry_hashes: dict[str, str] = {}
    if unpinned:
        if source == "modelscope":
            if artifact.modelscope_id is None:
                raise WeightVerificationError(
                    f"'{artifact.repo_id}' has unpinned weight files but no "
                    "modelscope_id in the manifest; pin it with "
                    "scripts/update_weight_hashes.py."
                )
            registry_hashes = _modelscope_registry_sha256(artifact.modelscope_id)
        else:
            registry_hashes = _lfs_sha256(artifact.repo_id, artifact.revision)
    tier = "pinned" if not unpinned else "lfs-metadata"
    for path in weight_files:
        expected = artifact.sha256.get(path.name) or registry_hashes.get(path.name)
        if not expected:
            raise WeightVerificationError(
                f"No sha256 available for '{path.name}' of '{artifact.repo_id}' "
                "(not pinned in the manifest and not published in the "
                "registry metadata). Run scripts/update_weight_hashes.py to "
                "pin it."
            )
        actual = compute_file_sha256(path)
        if actual != expected:
            raise WeightVerificationError(
                f"sha256 mismatch for '{artifact.repo_id}/{path.name}': "
                f"expected {expected}, got {actual}. {_RETRY_MESSAGE}"
            )
        logger.info("Verified %s (%s tier).", path.name, tier)
    return tier


def _download_snapshot(
    repo_id: str, revision: str | None, local_files_only: bool = False
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(_SNAPSHOT_PATTERNS),
            local_files_only=local_files_only,
        )
    )


_MODELSCOPE_API = "https://modelscope.cn/api/v1/models/{repo}/repo"
_MODELSCOPE_CACHE = Path.home() / ".cache" / "autovision" / "modelscope"


def _modelscope_registry_sha256(ms_id: str) -> dict[str, str]:
    """ModelScope-published sha256 per weight filename (network tier)."""
    import json
    from urllib.request import Request, urlopen

    url = _MODELSCOPE_API.format(repo=ms_id) + "/files?Revision=master"
    with urlopen(Request(url), timeout=Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value) as resp:  # nosec B310
        payload = json.load(resp)
    hashes: dict[str, str] = {}
    for entry in payload.get("Data", {}).get("Files", []):
        name = entry.get("Path", "")
        sha = entry.get("Sha256")
        if sha and name.endswith(_WEIGHT_SUFFIXES):
            hashes[name] = sha
    return hashes


def _download_modelscope_snapshot(
    ms_id: str, local_files_only: bool = False
) -> Path:
    """Download a ModelScope repo snapshot (config + weight files).

    Cached under ``~/.cache/autovision/modelscope/<ms_id>``; sha256
    verification happens in :func:`_verify_snapshot` against the manifest,
    exactly like the hub path.
    """
    import json
    from urllib.request import Request, urlopen

    snapshot_dir = _MODELSCOPE_CACHE / ms_id.replace("/", "--")
    weight_files = (
        [
            path
            for pattern in _WEIGHT_PATTERNS
            for path in snapshot_dir.glob(pattern)
        ]
        if snapshot_dir.is_dir()
        else []
    )
    if weight_files and (snapshot_dir / "config.json").is_file():
        return snapshot_dir
    if local_files_only:
        raise CheckpointError(
            f"AUTOVISION_OFFLINE=1 and no cached ModelScope snapshot for "
            f"'{ms_id}'. Run once without AUTOVISION_OFFLINE to populate "
            "the cache."
        )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    listing_url = _MODELSCOPE_API.format(repo=ms_id) + "/files?Revision=master"
    with urlopen(Request(listing_url), timeout=60.0) as resp:  # nosec B310
        files = [
            entry["Path"]
            for entry in json.load(resp).get("Data", {}).get("Files", [])
            if entry.get("Path") == "config.json"
            or entry.get("Path", "").endswith(_WEIGHT_SUFFIXES)
        ]
    if not any(name.endswith(_WEIGHT_SUFFIXES) for name in files):
        raise WeightVerificationError(
            f"ModelScope repo '{ms_id}' lists no weight files matching "
            f"{_WEIGHT_PATTERNS}."
        )
    timeout = Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value
    for name in files:
        target = snapshot_dir / name
        url = _MODELSCOPE_API.format(repo=ms_id) + f"?Revision=master&FilePath={name}"
        with urlopen(Request(url), timeout=timeout) as resp, open(target, "wb") as out:  # nosec B310
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        logger.info("Downloaded %s from ModelScope.", name)
    return snapshot_dir


def _load_from_modelscope(
    model_cls: Any,
    artifact: WeightArtifact,
    config_kwargs: dict[str, Any],
) -> Any:
    """Verified load from the artifact's ModelScope mirror.

    Mirrors the hub flow: snapshot (cached), sha256 verify, one
    delete-and-retry on mismatch, then ``from_pretrained`` on the local
    directory. Requires a pinned manifest entry (``artifact.sha256``) so
    verification never depends on the network.
    """
    assert artifact.modelscope_id is not None  # gated by the caller
    if Env.OFFLINE.value:
        snapshot_dir = _download_modelscope_snapshot(
            artifact.modelscope_id, local_files_only=True
        )
        _verify_snapshot(snapshot_dir, artifact, source="modelscope")
        return model_cls.from_pretrained(snapshot_dir, **config_kwargs)

    snapshot_dir = _download_modelscope_snapshot(artifact.modelscope_id)
    try:
        _verify_snapshot(snapshot_dir, artifact, source="modelscope")
    except WeightVerificationError:
        logger.warning(
            "Hash mismatch for '%s' (ModelScope); retrying once.",
            artifact.repo_id,
        )
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        snapshot_dir = _download_modelscope_snapshot(artifact.modelscope_id)
        _verify_snapshot(snapshot_dir, artifact, source="modelscope")
    return model_cls.from_pretrained(snapshot_dir, **config_kwargs)


def _load_from_hub(
    model_cls: Any,
    artifact: WeightArtifact,
    config_kwargs: dict[str, Any],
) -> Any:
    """Hub flow: mirror bridge, offline resolution, verify with one retry."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # hub < 1.0 re-exports at the top level
        from huggingface_hub import LocalEntryNotFoundError  # noqa: F401

    repo_id = artifact.repo_id
    revision = artifact.revision

    # Mirror consumer (§3.2): AUTOVISION_MIRROR_URL -> HF endpoint, unless the
    # process already pinned one explicitly.
    mirror_url = Env.MIRROR_URL.value
    if mirror_url and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info("Using weight mirror: %s", mirror_url)

    if Env.OFFLINE.value:
        try:
            snapshot_dir = _download_snapshot(
                repo_id, revision, local_files_only=True
            )
        except LocalEntryNotFoundError as error:
            raise CheckpointError(
                f"AUTOVISION_OFFLINE=1 and no cached snapshot for '{repo_id}'. "
                "Run once without AUTOVISION_OFFLINE to populate the cache, or "
                "download the model manually and point model_args at it."
            ) from error
        if artifact.sha256:
            _verify_snapshot(snapshot_dir, artifact)  # offline-capable tier
        else:
            logger.warning(
                "Offline load of '%s' skipped sha256 verification (entry is "
                "unpinned; run scripts/update_weight_hashes.py).", repo_id,
            )
        return model_cls.from_pretrained(snapshot_dir, **config_kwargs)

    snapshot_dir = _download_snapshot(repo_id, revision)
    try:
        _verify_snapshot(snapshot_dir, artifact)
    except WeightVerificationError:
        # One delete-cache + re-download retry before failing hard.
        logger.warning("Hash mismatch for '%s'; retrying once.", repo_id)
        _delete_repo_cache(snapshot_dir)
        snapshot_dir = _download_snapshot(repo_id, revision)
        _verify_snapshot(snapshot_dir, artifact)  # raises on second failure
    return model_cls.from_pretrained(snapshot_dir, **config_kwargs)


#: License tags in the weight manifest that are NOT commercially
#: unrestricted. Loading still proceeds (the per-package gates decide
#: that); this warning makes the obligation visible at the moment of use,
#: including for downstream obligations such as distilling from these
#: weights as teachers.
_NON_PERMISSIVE_LICENSE_MARKERS = ("cc-by-nc", "dinov3-custom", "custom")


def _warn_non_permissive_license(repo_id: str, artifact: WeightArtifact) -> None:
    tag = artifact.license.lower()
    if any(marker in tag for marker in _NON_PERMISSIVE_LICENSE_MARKERS):
        logger.warning(
            "Pretrained weights '%s' carry license '%s' — not commercially "
            "unrestricted. Ensure your use (including models trained from "
            "these weights as initialization or distillation teachers) "
            "complies; see NOTICE for the framework's weights policy.",
            repo_id,
            artifact.license,
        )


def load_pretrained_verified(
    model_cls: Any,
    repo_id: str,
    config_kwargs: dict[str, Any] | None = None,
) -> Any:
    """``model_cls.from_pretrained`` wrapped with the verification flow."""
    artifact = get_weight_artifact(repo_id)
    if artifact is None:
        logger.warning(
            "Model '%s' is not in the weight manifest; loading WITHOUT "
            "sha256 verification.", repo_id,
        )
        return model_cls.from_pretrained(repo_id, **(config_kwargs or {}))

    _warn_non_permissive_license(repo_id, artifact)

    source = Env.WEIGHT_SOURCE.value
    mirrorable = artifact.modelscope_id is not None and bool(artifact.sha256)

    if source == "modelscope":
        if mirrorable:
            return _load_from_modelscope(model_cls, artifact, config_kwargs or {})
        logger.warning(
            "AUTOVISION_WEIGHT_SOURCE=modelscope but '%s' has no pinned "
            "ModelScope mirror in the manifest; falling back to the hub.",
            repo_id,
        )
    elif source == "auto" and mirrorable:
        try:
            return _load_from_hub(model_cls, artifact, config_kwargs or {})
        except Exception as hub_error:  # network failure, blocked endpoint, ...
            logger.warning(
                "Hub download for '%s' failed (%s); falling back to the "
                "ModelScope mirror '%s'.",
                repo_id, hub_error, artifact.modelscope_id,
            )
            return _load_from_modelscope(model_cls, artifact, config_kwargs or {})

    return _load_from_hub(model_cls, artifact, config_kwargs or {})


def _render_manifest(manifest: dict[str, WeightArtifact]) -> str:
    """Deterministic manifest source (single formatting source, used by
    ``scripts/update_weight_hashes.py`` to re-pin hashes in place)."""
    lines = [
        "# Copyright (c) 2026 AutoVision contributors. Apache-2.0.",
        '"""Pinned pretrained-weight manifest (machine-managed).',
        "",
        "Regenerate with scripts/update_weight_hashes.py; hand-edits are",
        "overwritten on the next pinning run.",
        '"""',
        "",
        "from autovision.encoders.weight_verification import WeightArtifact",
        "",
        "__all__ = ['WEIGHT_MANIFEST']",
        "",
        "WEIGHT_MANIFEST = {",
    ]
    for repo_id in sorted(manifest):
        artifact = manifest[repo_id]
        lines.append(f"    {repo_id!r}: WeightArtifact(")
        lines.append(f"        repo_id={repo_id!r},")
        lines.append(f"        license={artifact.license!r},")
        lines.append(f"        source={artifact.source!r},")
        lines.append(f"        revision={artifact.revision!r},")
        lines.append(f"        sha256={artifact.sha256!r},")
        if artifact.modelscope_id is not None:
            lines.append(f"        modelscope_id={artifact.modelscope_id!r},")
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines) + "\n"
