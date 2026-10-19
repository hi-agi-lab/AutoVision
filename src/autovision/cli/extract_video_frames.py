# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Video frame extraction via ffmpeg (baseline-compatible command)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from autovision.types import PathLike

__all__ = ["extract_video_frames"]

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_video_frames(
    data: PathLike | Sequence[PathLike],
    out: PathLike,
    *,
    overwrite: bool = False,
    frame_filename_format: str = "%09d.jpg",
    max_frames_per_video: int | None = None,
) -> list[Path]:
    """Extract JPEG frames from video file(s) using ffmpeg.

    ``data`` is a video file, a directory of videos, or a list of either.
    Each video writes frames to ``<out>/<video_stem>/``. Returns the created
    frame directories.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for extract_video_frames but was not found "
            "on PATH."
        )
    out_dir = Path(out)
    videos = _resolve_videos(data)
    if not videos:
        raise ValueError(f"No videos found in '{data}'.")
    created = []
    used_stems: set[str] = set()
    for video in videos:
        target = out_dir / video.stem
        if video.stem in used_stems:
            # Same-stem videos (a/clip.mp4, b/clip.mp4) must not merge.
            target = out_dir / f"{video.parent.name}_{video.stem}"
        used_stems.add(video.stem)
        if target.exists():
            if not overwrite:
                raise ValueError(
                    f"Frames directory '{target}' is not empty; use "
                    f"overwrite=True (stale frames are removed)."
                )
            shutil.rmtree(target)  # never mix old and new frames
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-loglevel", "error", "-i", str(video),
        ]
        if max_frames_per_video is not None:
            command += ["-frames:v", str(max_frames_per_video)]
        command += [str(target / frame_filename_format)]
        subprocess.run(command, check=True, capture_output=True)
        created.append(target)
    return created


def _resolve_videos(data: PathLike | Sequence[PathLike]) -> list[Path]:
    if not isinstance(data, (list, tuple)):
        data = [data]
    videos: list[Path] = []
    for item in data:
        path = Path(item)
        if path.is_dir():
            videos.extend(
                p for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
            videos.append(path)
    return videos
