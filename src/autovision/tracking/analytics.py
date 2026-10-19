# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Track-level analytics on top of ByteTrack: line/polygon zones and smoothing.

The zone counters consume :class:`~autovision.tracking.byte_track.Track`
rows (``track_id`` is the identity), which keeps counting stable across
frames. ``TrackSmoother`` averages recent boxes per track for rendering;
raw detections in the JSONL stay untouched.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autovision.errors import AutoVisionError
from autovision.tracking.byte_track import Track

__all__ = [
    "LineZone",
    "LineZoneCounts",
    "PolygonZone",
    "TrackSmoother",
    "ZoneCounts",
]


def _anchor(track: Track, name: str) -> tuple[float, float]:
    """Resolve a named anchor on the track's xyxy box."""
    x1, y1, x2, y2 = track.bbox
    if name == "bottom_center":
        return (x1 + x2) / 2.0, y2
    if name == "center":
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if name == "top_center":
        return (x1 + x2) / 2.0, y1
    if name == "bottom_left":
        return x1, y2
    if name == "bottom_right":
        return x2, y2
    if name == "center_left":
        return x1, (y1 + y2) / 2.0
    if name == "center_right":
        return x2, (y1 + y2) / 2.0
    raise AutoVisionError(f"Unknown anchor '{name}'.")


@dataclass
class ZoneCounts:
    """Current/cumulative count for one zone, with a per-class split."""

    total: int = 0
    per_class: dict[int, int] = field(default_factory=dict)

    def add(self, class_id: int, amount: int = 1) -> None:
        self.total += amount
        self.per_class[class_id] = self.per_class.get(class_id, 0) + amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "per_class": {str(key): value for key, value in sorted(self.per_class.items())},
        }


@dataclass
class LineZoneCounts:
    """Directional crossing totals for a line zone."""

    in_count: int = 0
    out_count: int = 0
    in_per_class: dict[int, int] = field(default_factory=dict)
    out_per_class: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "in": self.in_count,
            "out": self.out_count,
            "in_per_class": {
                str(key): value for key, value in sorted(self.in_per_class.items())
            },
            "out_per_class": {
                str(key): value for key, value in sorted(self.out_per_class.items())
            },
        }


class LineZone:
    """Count tracks crossing a directed line segment.

    ``in`` counts a negative-to-positive side change, ``out`` the opposite;
    the direction follows the line's ``start -> end`` orientation. A track
    contributes at most one crossing per frame and is forgotten after
    ``max_age`` frames without an update.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        name: str = "line",
        anchor: str = "bottom_center",
        max_age: int = 30,
    ) -> None:
        if start == end:
            raise AutoVisionError("LineZone start and end must differ.")
        if max_age < 1:
            raise AutoVisionError("LineZone max_age must be >= 1.")
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.name = name
        self.anchor = anchor
        self.max_age = int(max_age)
        self.counts = LineZoneCounts()
        self._side: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def _side_of(self, track: Track) -> int:
        px, py = _anchor(track, self.anchor)
        x1, y1 = self.start
        dx, dy = self.end[0] - x1, self.end[1] - y1
        cross = dx * (py - y1) - dy * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def update(self, tracks: Sequence[Track]) -> LineZoneCounts:
        """Update crossing history with one frame's tracks."""
        self._frame += 1
        # A track absent for more than max_age frames starts a fresh life.
        # Check the gap before matching so a reappearing id cannot produce a
        # spurious crossing from its previous side history.
        for track in tracks:
            track_id = int(track.track_id)
            if (
                self._frame - self._last_seen.get(track_id, 0)
                > self.max_age
            ):
                self._side.pop(track_id, None)
                self._last_seen.pop(track_id, None)
        for track in tracks:
            track_id = int(track.track_id)
            self._last_seen[track_id] = self._frame
            side = self._side_of(track)
            if side == 0:
                continue
            previous = self._side.get(track_id)
            self._side[track_id] = side
            if previous is None or previous == side:
                continue
            if previous < 0 and side > 0:
                self.counts.in_count += 1
                key = int(track.class_id)
                self.counts.in_per_class[key] = (
                    self.counts.in_per_class.get(key, 0) + 1
                )
            elif previous > 0 and side < 0:
                self.counts.out_count += 1
                key = int(track.class_id)
                self.counts.out_per_class[key] = (
                    self.counts.out_per_class.get(key, 0) + 1
                )
        # Evict tracks that were absent for max_age frames; current tracks
        # were refreshed above, so a gap of exactly max_age is still allowed.
        for track_id in list(self._side):
            if self._frame - self._last_seen.get(track_id, 0) >= self.max_age:
                self._side.pop(track_id, None)
                self._last_seen.pop(track_id, None)
        return self.counts

    def reset(self) -> None:
        """Clear crossing history and totals."""
        self.counts = LineZoneCounts()
        self._side.clear()
        self._last_seen.clear()
        self._frame = 0


def _points_in_polygon(
    points: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """Vectorized ray casting for ``(N, 2)`` points against one polygon."""
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x1 = polygon[:, 0][None, :]
    y1 = polygon[:, 1][None, :]
    x2 = np.roll(polygon[:, 0], -1)[None, :]
    y2 = np.roll(polygon[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    crossings = ((y1 > y) != (y2 > y)) & (x < x_intersect)
    return (crossings.sum(axis=1) % 2) == 1


class PolygonZone:
    """Count tracks whose anchor(s) fall inside a polygon."""

    def __init__(
        self,
        polygon: Sequence[tuple[float, float]],
        *,
        name: str = "zone",
        anchors: Sequence[str] = ("bottom_center",),
        require_all: bool = False,
    ) -> None:
        if len(polygon) < 3:
            raise AutoVisionError("PolygonZone needs at least three points.")
        if not anchors:
            raise AutoVisionError("PolygonZone needs at least one anchor.")
        self.polygon = np.asarray(polygon, dtype=np.float64)
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise AutoVisionError("PolygonZone points must be (x, y) pairs.")
        self.name = name
        self.anchors = tuple(anchors)
        self.require_all = bool(require_all)
        self.counts = ZoneCounts()

    def update(self, tracks: Sequence[Track]) -> ZoneCounts:
        """Recount the current frame's tracks inside the polygon."""
        counts = ZoneCounts()
        if not tracks:
            self.counts = counts
            return counts
        anchor_points = np.asarray(
            [_anchor(track, anchor) for track in tracks for anchor in self.anchors],
            dtype=np.float64,
        ).reshape(len(tracks), len(self.anchors), 2)
        inside_anchor = np.stack(
            [
                _points_in_polygon(anchor_points[:, index, :], self.polygon)
                for index in range(len(self.anchors))
            ],
            axis=1,
        )
        inside = (
            inside_anchor.all(axis=1)
            if self.require_all
            else inside_anchor.any(axis=1)
        )
        for track, hit in zip(tracks, inside):
            if hit:
                counts.add(int(track.class_id))
        self.counts = counts
        return counts

    def reset(self) -> None:
        """Reset the current-frame counts."""
        self.counts = ZoneCounts()


@dataclass
class _TrackHistory:
    boxes: deque[tuple[float, float, float, float]]
    scores: deque[float]
    last_seen: int


class TrackSmoother:
    """Average each track's box and score over the last ``length`` frames."""

    def __init__(self, length: int = 5, *, max_age: int = 30) -> None:
        if length < 1:
            raise AutoVisionError("TrackSmoother length must be >= 1.")
        if max_age < 1:
            raise AutoVisionError("TrackSmoother max_age must be >= 1.")
        self.length = int(length)
        self.max_age = int(max_age)
        self._history: dict[int, _TrackHistory] = {}
        self._frame = 0

    def update(self, tracks: Sequence[Track]) -> list[Track]:
        """Return tracks with averaged boxes/scores, preserving ids and classes."""
        self._frame += 1
        smoothed: list[Track] = []
        for track in tracks:
            track_id = int(track.track_id)
            history = self._history.get(track_id)
            # A gap longer than max_age starts a new history; a gap of exactly
            # max_age is still within the allowed tracking window.
            if (
                history is not None
                and self._frame - history.last_seen > self.max_age
            ):
                self._history.pop(track_id, None)
                history = None
            if history is None:
                history = _TrackHistory(
                    boxes=deque(maxlen=self.length),
                    scores=deque(maxlen=self.length),
                    last_seen=self._frame,
                )
                self._history[track_id] = history
            history.boxes.append(
                (
                    float(track.bbox[0]),
                    float(track.bbox[1]),
                    float(track.bbox[2]),
                    float(track.bbox[3]),
                )
            )
            history.scores.append(float(track.score))
            history.last_seen = self._frame
            boxes = np.asarray(history.boxes, dtype=np.float64)
            smoothed.append(
                Track(
                    track_id=track.track_id,
                    bbox=tuple(float(value) for value in boxes.mean(axis=0)),
                    score=float(np.mean(history.scores)),
                    class_id=track.class_id,
                    state=track.state,
                    age=track.age,
                )
            )
        for track_id in list(self._history):
            if self._frame - self._history[track_id].last_seen >= self.max_age:
                self._history.pop(track_id, None)
        return smoothed

    def reset(self) -> None:
        """Forget all track history."""
        self._history.clear()
        self._frame = 0
