# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset download registry for the public detection datasets.

Model weights have their own pinned-manifest pipeline
(``encoders.weight_verification``); datasets are plain public artifacts.
The registry records official URLs and, when the source publishes them,
sizes and sha256 digests — entries without a digest WARN at download time
instead of silently skipping verification. Nothing here runs at import
time and no test requires network access.
"""

from __future__ import annotations

import hashlib
import logging
import os
import posixpath
import tarfile
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from autovision.errors import AutoVisionError
from autovision.runtime.env import Env

__all__ = ["DATASET_REGISTRY", "DatasetFile", "DatasetSpec", "download_dataset"]

logger = logging.getLogger("autovision")


@dataclass(frozen=True)
class DatasetFile:
    url: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    description: str
    homepage: str
    files: tuple[DatasetFile, ...] = field(default=())
    #: Set when the source requires manual acquisition (registration form,
    #: license click-through). ``download_dataset`` refuses with instructions.
    manual_instructions: str | None = None
    license_note: str = ""


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "coco2017-val": DatasetSpec(
        description="COCO 2017 validation images + annotations (detection).",
        homepage="https://cocodataset.org",
        files=(
            DatasetFile(url="https://images.cocodataset.org/zips/val2017.zip"),
            DatasetFile(
                url="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ),
        ),
        license_note="Images: Flickr terms of use; annotations: CC BY 4.0.",
    ),
    "coco2017-train": DatasetSpec(
        description="COCO 2017 train images (118k, ~18GB) + annotations.",
        homepage="https://cocodataset.org",
        files=(
            DatasetFile(url="https://images.cocodataset.org/zips/train2017.zip"),
            DatasetFile(
                url="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ),
        ),
        license_note="Images: Flickr terms of use; annotations: CC BY 4.0.",
    ),
    "voc2007-trainval": DatasetSpec(
        description="PASCAL VOC 2007 trainval (~450MB tar).",
        homepage="http://host.robots.ox.ac.uk/pascal/VOC/",
        files=(
            DatasetFile(
                url=(
                    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
                    "VOCtrainval_06-Nov-2007.tar"
                )
            ),
        ),
        license_note="CLUF at the VOC homepage; non-commercial research.",
    ),
    "visdrone2017-det": DatasetSpec(
        description="VisDrone-DET2017 (manual acquisition required).",
        homepage="https://github.com/VisDrone/VisDrone-Dataset",
        manual_instructions=(
            "VisDrone is distributed by the AISKYEYE set, Lab of MEDIS, "
            "Tianjin University via a request form / OneDrive+GoogleDrive "
            "mirrors listed at the homepage. Download DET-train/val and "
            "unpack under your dataset root, then point the train config's "
            "data dict at the annotation directories."
        ),
        license_note="Free for research; see homepage for terms.",
    ),
}


class UnknownDatasetError(AutoVisionError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - cleanup is best effort
        logger.warning("Could not remove partial download %s", path)


def _download_file(spec: DatasetFile, dest: Path) -> Path:
    """Download ``spec.url`` to ``dest`` atomically.

    The bytes are streamed to a unique ``*.part`` file in the destination
    directory, verified against the pinned size/sha256, and only then moved
    onto the final path with :func:`os.replace`. Any error removes the temp
    file; the final path never holds a partial download.
    """
    if Env.OFFLINE.value:
        raise AutoVisionError(
            f"AUTOVISION_OFFLINE=1; refusing to download '{dest.name}'. "
            "Pre-populate the archive or dataset root, or unset "
            "AUTOVISION_OFFLINE."
        )
    if Env.REQUIRE_DATASET_HASHES.value and spec.sha256 is None:
        raise AutoVisionError(
            f"Dataset artifact '{dest.name}' has no pinned sha256 and "
            "AUTOVISION_REQUIRE_DATASET_HASHES=1; refusing the download. "
            "Register a mirrored artifact with a digest, or unset the flag "
            "to accept unverified downloads."
        )
    if spec.sha256 is None:
        logger.warning(
            "Dataset artifact %s has no pinned sha256 in the registry; "
            "integrity is not verified for this download.",
            dest.name,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    scheme = urlparse(spec.url).scheme.lower()
    if scheme not in ("https", "http"):
        raise AutoVisionError(
            f"Dataset artifact '{dest.name}' uses unsupported URL scheme "
            f"'{scheme}'; only https/http are allowed."
        )
    temp = dest.parent / f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    timeout = Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value
    progress = None
    try:
        with (
            # scheme is restricted to https/http above.
            urllib.request.urlopen(spec.url, timeout=timeout) as response,  # nosec B310
            temp.open("xb") as handle,
        ):
            total = response.getheader("Content-Length")
            try:
                total_bytes = int(total) if total else None
            except (TypeError, ValueError):
                total_bytes = None
            try:
                from tqdm import tqdm

                progress = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                )
            except ImportError:  # pragma: no cover - tqdm is a core dependency
                progress = None
            received = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
            handle.flush()
            os.fsync(handle.fileno())
        if total_bytes is not None and received != total_bytes:
            raise AutoVisionError(
                f"Download of '{dest.name}' was truncated: received "
                f"{received} of {total_bytes} bytes."
            )
        size = temp.stat().st_size
        if spec.size_bytes is not None and size != spec.size_bytes:
            raise AutoVisionError(
                f"Downloaded '{dest.name}' has {size} bytes, expected "
                f"{spec.size_bytes}."
            )
        if spec.sha256 is not None:
            actual = _digest(temp)
            if actual != spec.sha256:
                raise AutoVisionError(
                    f"sha256 mismatch for '{dest.name}': expected "
                    f"{spec.sha256}, got {actual}; partial download deleted."
                )
        os.replace(temp, dest)
        return dest
    except BaseException as error:
        _discard(temp)
        if isinstance(error, Exception) and not isinstance(error, AutoVisionError):
            raise AutoVisionError(
                f"Failed to download '{dest.name}' from '{spec.url}': {error}"
            ) from error
        raise
    finally:
        if progress is not None:
            progress.close()


def _reject_unsafe_member(name: str, archive: Path) -> None:
    """Reject zip/tar members that are absolute or contain ``..`` traversal."""
    if not name or "\x00" in name:
        raise AutoVisionError(
            f"Archive '{archive}' contains an invalid member name {name!r}; "
            "refusing extraction."
        )
    normalized = name.replace("\\", "/")
    windows = PureWindowsPath(name)
    if (
        normalized.startswith("/")
        or PurePosixPath(normalized).is_absolute()
        or windows.drive
        or windows.is_absolute()
    ):
        raise AutoVisionError(
            f"Archive '{archive}' contains an absolute-path member {name!r}; "
            "refusing extraction."
        )
    if ".." in PurePosixPath(normalized).parts:
        raise AutoVisionError(
            f"Archive '{archive}' contains a path-traversal member {name!r}; "
            "refusing extraction."
        )


def _reject_unsafe_link(member: tarfile.TarInfo, archive: Path) -> None:
    linkname = member.linkname
    normalized = linkname.replace("\\", "/")
    windows = PureWindowsPath(linkname)
    if (
        not normalized
        or "\x00" in linkname
        or normalized.startswith("/")
        or windows.drive
        or windows.is_absolute()
    ):
        raise AutoVisionError(
            f"Archive '{archive}' member {member.name!r} has an absolute or "
            f"invalid link target {linkname!r}; refusing extraction."
        )
    # Symlink targets are relative to the member's directory, hard-link
    # targets to the archive root (tar semantics).
    if member.issym():
        base = PurePosixPath(member.name.replace("\\", "/")).parent
    else:
        base = PurePosixPath(".")
    resolved = posixpath.normpath(str(base / normalized))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        raise AutoVisionError(
            f"Archive '{archive}' member {member.name!r} links outside the "
            f"extraction root ({linkname!r}); refusing extraction."
        )


def _validate_tar_members(
    bundle: tarfile.TarFile, archive: Path
) -> list[tarfile.TarInfo]:
    """Equivalent of ``tarfile.data_filter`` for builds lacking ``filter``.

    Validates every member before extracting anything and strips setuid/
    setgid/sticky bits. Only regular files, directories, and links whose
    targets stay inside the extraction root are allowed.
    """
    members = bundle.getmembers()
    for member in members:
        _reject_unsafe_member(member.name, archive)
        if member.isreg() or member.isdir():
            pass
        elif member.issym() or member.islnk():
            _reject_unsafe_link(member, archive)
        else:
            raise AutoVisionError(
                f"Archive '{archive}' contains unsupported member "
                f"{member.name!r} (not a regular file, directory, or link); "
                "refusing extraction."
            )
        member.mode &= 0o777
    return members


def _extract(archive: Path, out_dir: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _reject_unsafe_member(member.filename, archive)
            bundle.extractall(out_dir)  # nosec B202
        return
    with tarfile.open(archive) as bundle:
        try:
            # tarfile data_filter validates members (links/
            # device files/traversal) before extraction.
            bundle.extractall(out_dir, filter="data")
        except TypeError:
            # Python builds without the tarfile filter backport (3.10 builds
            # older than 3.10.12/any build lacking the keyword). Applying the
            # equivalent checks by hand keeps extraction safe rather than
            # silently extracting an unvalidated member. Reopen the archive:
            # a mid-extraction failure may have consumed the stream.
            with tarfile.open(archive) as fallback:
                # Every member is validated by
                # _validate_tar_members before extraction.
                fallback.extractall(  # nosec B202
                    out_dir, members=_validate_tar_members(fallback, archive)
                )


def _verify_cached_archive(spec: DatasetFile, archive: Path) -> None:
    """Fail closed when a cached archive no longer matches its pins."""
    if spec.size_bytes is not None:
        actual_size = archive.stat().st_size
        if actual_size != spec.size_bytes:
            raise AutoVisionError(
                f"cached archive failed size verification for '{archive}': "
                f"{actual_size} bytes, expected {spec.size_bytes}; delete the "
                "archive and retry."
            )
    if spec.sha256 is not None:
        actual = _digest(archive)
        if actual != spec.sha256:
            raise AutoVisionError(
                f"cached archive failed sha256 verification for '{archive}': "
                f"expected {spec.sha256}, got {actual}; delete the archive "
                "and retry."
            )
    if spec.size_bytes is None and spec.sha256 is None:
        logger.warning(
            "Cached dataset archive %s has no pinned size_bytes or sha256; "
            "integrity is not verified before extraction.",
            archive.name,
        )


def download_dataset(name: str, out_dir: Path | str) -> Path:
    """Download + extract a registered dataset under ``out_dir/<name>``.

    Downloads are written to a unique temporary file, verified, and moved
    atomically onto the final path. Cached archives are re-verified against
    the registry's pinned size/sha256 before extraction. Already-extracted
    archives are skipped, so the call is idempotent at file granularity; it
    does not resume partial HTTP transfers.
    """
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        raise UnknownDatasetError(
            f"Unknown dataset '{name}'. Registered: {sorted(DATASET_REGISTRY)}"
        )
    if spec.manual_instructions is not None:
        raise AutoVisionError(
            f"Dataset '{name}' requires manual acquisition. {spec.manual_instructions}"
        )
    if Env.REQUIRE_DATASET_HASHES.value:
        unpinned = [Path(entry.url).name for entry in spec.files if entry.sha256 is None]
        if unpinned:
            raise AutoVisionError(
                f"AUTOVISION_REQUIRE_DATASET_HASHES=1 but dataset '{name}' "
                f"has no pinned sha256 for {unpinned}."
            )
    root = Path(out_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    for entry in spec.files:
        archive_name = Path(entry.url).name
        archive = root / archive_name
        if archive.exists():
            _verify_cached_archive(entry, archive)
        else:
            _download_file(entry, archive)
        _extract(archive, root)
    return root
