# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ByteTrack: per-frame detection association with a Kalman motion model (A3).

Reference implementation of the ByteTrack association strategy (Zhang et
al., ECCV 2022 — "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box"), following the official ``BYTETracker`` (yolox) semantics:

- a constant-velocity Kalman filter per track over the ``(x, y, aspect,
  height)`` state, with the official process/measurement noise weights;
- every frame, detections split at ``track_thresh`` into the HIGH pool and
  the LOW-score pool ``0.1 < score < track_thresh`` — ByteTrack's core
  idea is that low-confidence boxes still carry motion evidence and must
  not be dropped;
- THREE association passes, with the official thresholds (IoU similarity):
  1. activated tracks (tracked + lost) vs the high pool —
     ``sim >= 1 - match_thresh`` with the detection score fused into the
     similarity (``sim = IoU * score``, the official non-mot20 path);
  2. still-tracked leftovers vs the low pool — plain IoU ``>= 0.5``;
  3. unconfirmed one-frame tracks vs the high detections left over —
     ``sim >= 0.3``; an unmatched unconfirmed track is removed WITHOUT
     ever being published (the confirmation gate);
- a detection starts a track only with ``score >= track_thresh + 0.1``
  (``det_thresh``), and the new track stays unconfirmed until its first
  successful match — the first frame is the official exception, where
  every new track is activated at once.

Differences from the official MOT17 implementation, documented rather
than hidden:

- **Class-aware**: boxes are associable only when their class ids match
  (the official tracker is class-agnostic for single-class MOT; the
  restriction stops ids swapping across classes on multi-class output).
- **Per-frame API** over absolute-pixel ``xyxy`` boxes (the framework's
  native detection output) instead of the official ``tlwh``; the state
  conversion happens inside.
- The ``mot20`` variant (unfused stage-1 cost) is not implemented — the
  official non-mot20 behavior is the only mode.

Deterministic: given the same inputs and ``frame_rate`` the ids are stable
(no RNG anywhere; one id counter per tracker instance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from autovision.errors import AutoVisionError

__all__ = ["BYTETracker", "Track"]

logger = logging.getLogger(__name__)

#: ByteTrack's Kalman std weights (official values).
_STD_WEIGHT_POSITION = 1.0 / 20.0
_STD_WEIGHT_VELOCITY = 1.0 / 160.0

#: Official gates (BYTETracker, non-mot20 defaults).
_DET_THRESH_MARGIN = 0.1  # new tracks need score >= track_thresh + 0.1
_LOW_SCORE_FLOOR = 0.1  # the low pool is 0.1 < score < track_thresh
_SECOND_MATCH_IOU = 0.5  # stage 2: plain-IoU similarity threshold
_UNCONFIRMED_DIST_THRESH = 0.7  # stage 3: fused similarity >= 1 - 0.7

_NEW = "new"
_TRACKED = "tracked"
_LOST = "lost"
_REMOVED = "removed"


@dataclass
class Track:
    """One live track as reported to the caller."""

    track_id: int
    bbox: tuple[float, float, float, float]  # xyxy absolute pixels
    score: float
    class_id: int
    state: str = _TRACKED
    #: Frames since the track was created (``frame_id - start_frame``).
    age: int = 0


class _KalmanFilter:
    """Constant-velocity filter over the ``(x, y, aspect, height)`` state."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim, dtype=np.float64)
        for index in range(ndim):
            self._motion[index, ndim + index] = dt
        self._update_matrix = np.eye(ndim, 2 * ndim, dtype=np.float64)

    def _std(self, position: float, velocity: float, height: float) -> np.ndarray:
        return np.array(
            [
                position * height, position * height, 1e-2, position * height,
                velocity * height, velocity * height, 1e-5, velocity * height,
            ],
            dtype=np.float64,
        )

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        std = self._std(_STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY, measurement[3])
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std = self._std(_STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY, mean[3])
        motion_cov = np.diag(np.square(std))
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        # Measurement noise is 4-dimensional (position-only weights, 1e-1 for
        # the aspect term) — the 8-dimensional vector above is the PROCESS
        # noise of ``predict`` (official ByteTrack parameterization).
        std = np.array(
            [
                _STD_WEIGHT_POSITION * mean[3],
                _STD_WEIGHT_POSITION * mean[3],
                1e-1,
                _STD_WEIGHT_POSITION * mean[3],
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_matrix @ mean
        projected_cov = self._update_matrix @ covariance @ self._update_matrix.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ):
        projected_mean, projected_cov = self.project(mean, covariance)
        # Kalman gain via an explicit solve (no scipy.linalg dependency).
        gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._update_matrix.T).T
        ).T
        new_mean = mean + gain @ (measurement - projected_mean)
        new_covariance = covariance - gain @ projected_cov @ gain.T
        return new_mean, new_covariance


def _xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return np.array(
        [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, width / max(height, 1e-6), height],
        dtype=np.float64,
    )


def _xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    height = state[3]
    width = state[2] * height
    return np.array(
        [
            state[0] - width / 2.0,
            state[1] - height / 2.0,
            state[0] + width / 2.0,
            state[1] + height / 2.0,
        ]
    )


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every ``a[i]``/``b[j]`` pair (``(N, M)``, numpy only)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = (np.minimum(ax2, bx2) - np.maximum(ax1, bx1)).clip(0) * (
        np.minimum(ay2, by2) - np.maximum(ay1, by1)
    ).clip(0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


class _STrack:
    """One Kalman track's bookkeeping (``new`` → ``tracked`` ⇄ ``lost``)."""

    def __init__(self, box: np.ndarray, score: float, class_id: int, *, frame_id: int) -> None:
        self.mean, self.covariance = _KALMAN.initiate(_xyxy_to_xyah(box))
        self.score = float(score)
        self.class_id = int(class_id)
        self.track_id = 0
        #: Unconfirmed until the first successful match (frame 1 excepted).
        self.state = _NEW
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = int(frame_id)

    @property
    def box(self) -> np.ndarray:
        return _xyah_to_xyxy(self.mean)

    def activate(self, track_id: int) -> None:
        self.track_id = track_id
        self.state = _TRACKED

    def predict(self) -> None:
        self.mean, self.covariance = _KALMAN.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float, class_id: int) -> None:
        """Kalman update from a matched detection.

        Carries the state transitions too: a ``new`` track becomes
        ``tracked`` (activation) and a ``lost`` track is re-found.
        """
        self.mean, self.covariance = _KALMAN.update(
            self.mean, self.covariance, _xyxy_to_xyah(box)
        )
        self.score = float(score)
        self.class_id = int(class_id)
        self.hits += 1
        self.time_since_update = 0
        self.state = _TRACKED


_KALMAN = _KalmanFilter()


@dataclass(frozen=True)
class _FrameDetections:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def take(self, indices: list[int]) -> "_FrameDetections":
        return _FrameDetections(
            self.boxes[indices], self.scores[indices], self.classes[indices]
        )


class BYTETracker:
    """ByteTrack association over a detection stream.

    ``update(boxes, scores, classes)`` takes absolute-pixel ``xyxy`` boxes
    for ONE frame and returns the ACTIVATED tracks (``Track``). New tracks
    need ``score >= track_thresh + 0.1`` and stay unconfirmed until their
    first successful match (frame 1 excepted) — a one-frame blip is never
    published. Parameters follow the official API: ``track_thresh`` splits
    the high/low detection pools, ``match_thresh`` is the stage-1
    similarity ceiling, ``track_buffer`` (in frames at ``frame_rate``)
    bounds how long a lost track survives.
    """

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        class_aware: bool = True,
    ) -> None:
        if not 0.0 <= track_thresh <= 1.0:
            raise AutoVisionError(f"track_thresh must be in [0, 1], got {track_thresh}.")
        if not 0.0 <= match_thresh <= 1.0:
            raise AutoVisionError(f"match_thresh must be in [0, 1], got {match_thresh}.")
        if track_buffer < 0:
            raise AutoVisionError(f"track_buffer must be >= 0, got {track_buffer}.")
        if frame_rate < 1:
            raise AutoVisionError(f"frame_rate must be >= 1, got {frame_rate}.")
        self.track_thresh = float(track_thresh)
        self.det_thresh = self.track_thresh + _DET_THRESH_MARGIN
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate = int(frame_rate)
        self.class_aware = bool(class_aware)
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        #: Every live track (the ``state`` field carries new/tracked/lost);
        #: the tracker owns the list so a re-found track cannot linger in a
        #: stale pool.
        self.tracks: list[_STrack] = []
        self._next_id = 1
        self.frame_id = 0

    # ── association ─────────────────────────────────────────────────────
    def _match(
        self,
        tracks: list[_STrack],
        detections: _FrameDetections,
        similarity_threshold: float,
        *,
        fuse_score: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian over the similarity cost.

        Returns ``(pairs, unmatched_tracks, unmatched_dets)``. With
        ``fuse_score`` the detection score multiplies the IoU (the official
        ``fuse_score``, non-mot20 path); a class mismatch zeroes the pair
        under ``class_aware``. Matching is 1:1 — every detection and every
        track is consumed at most once.
        """
        from scipy.optimize import linear_sum_assignment

        if not tracks or len(detections.boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(detections.boxes)))
        boxes = np.stack([track.box for track in tracks])
        similarity = _iou_matrix(boxes, detections.boxes)
        if self.class_aware:
            track_classes = np.array([track.class_id for track in tracks])[:, None]
            similarity = np.where(
                track_classes == detections.classes[None, :], similarity, 0.0
            )
        if fuse_score:
            similarity = similarity * detections.scores[None, :]
        cost = 1.0 - similarity
        rows, cols = linear_sum_assignment(cost)
        pairs, matched_tracks, matched_dets = [], set(), set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if similarity[row, col] >= similarity_threshold:
                pairs.append((row, col))
                matched_tracks.add(row)
                matched_dets.add(col)
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections.boxes)) if i not in matched_dets]
        return pairs, unmatched_tracks, unmatched_dets

    # ── per-frame API ───────────────────────────────────────────────────
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> list[Track]:
        """Associate one frame's detections; returns the activated tracks."""
        self.frame_id += 1
        boxes, scores, classes = self._validate(boxes, scores, classes)

        high_mask = scores > self.track_thresh
        low_mask = (scores > _LOW_SCORE_FLOOR) & (scores < self.track_thresh)
        high = _FrameDetections(boxes[high_mask], scores[high_mask], classes[high_mask])
        low = _FrameDetections(boxes[low_mask], scores[low_mask], classes[low_mask])

        for track in self.tracks:
            track.predict()
        pool = [track for track in self.tracks if track.state in (_TRACKED, _LOST)]
        unconfirmed = [track for track in self.tracks if track.state == _NEW]

        # Stage 1: activated tracks (tracked + lost) vs the high-score pool.
        pairs, unmatched_pool, unmatched_high = self._match(
            pool, high, 1.0 - self.match_thresh, fuse_score=True
        )
        for track_index, det_index in pairs:
            pool[track_index].update(
                high.boxes[det_index], high.scores[det_index], high.classes[det_index]
            )

        # Stage 2: still-tracked leftovers vs the LOW-score pool — ByteTrack's
        # "associate every detection box".
        tracked_left = [pool[i] for i in unmatched_pool if pool[i].state == _TRACKED]
        pairs2, unmatched_tracked, _ = self._match(
            tracked_left, low, _SECOND_MATCH_IOU
        )
        for track_index, det_index in pairs2:
            tracked_left[track_index].update(
                low.boxes[det_index], low.scores[det_index], low.classes[det_index]
            )
        for track_index in unmatched_tracked:
            tracked_left[track_index].state = _LOST

        # Stage 3: unconfirmed tracks vs the high detections left over by
        # stage 1; an unmatched unconfirmed track is removed unpublished.
        remaining = high.take(unmatched_high)
        pairs3, unmatched_new, unmatched_remaining = self._match(
            unconfirmed, remaining, 1.0 - _UNCONFIRMED_DIST_THRESH, fuse_score=True
        )
        for track_index, det_index in pairs3:
            unconfirmed[track_index].update(
                remaining.boxes[det_index],
                remaining.scores[det_index],
                remaining.classes[det_index],
            )
        for track_index in unmatched_new:
            unconfirmed[track_index].state = _REMOVED

        # New tracks from the high detections nobody claimed; the reference
        # gates them on ``det_thresh`` and activates frame-1 tracks at once.
        for det_index in unmatched_remaining:
            if remaining.scores[det_index] < self.det_thresh:
                continue
            track = _STrack(
                remaining.boxes[det_index],
                remaining.scores[det_index],
                remaining.classes[det_index],
                frame_id=self.frame_id,
            )
            track.track_id = self._next_id
            self._next_id += 1
            if self.frame_id == 1:
                track.state = _TRACKED
            self.tracks.append(track)

        # Retire tracks that have been missing beyond the buffer.
        for track in self.tracks:
            if track.state == _LOST and track.time_since_update > self.max_time_lost:
                track.state = _REMOVED
        self.tracks = [track for track in self.tracks if track.state != _REMOVED]

        return [
            Track(
                track_id=track.track_id,
                bbox=tuple(float(value) for value in track.box),
                score=track.score,
                class_id=track.class_id,
                state=track.state,
                age=self.frame_id - track.start_frame,
            )
            for track in self.tracks
            if track.state == _TRACKED
        ]

    # ── validation ──────────────────────────────────────────────────────
    def _validate(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strict input checks — bad geometry fails here, not in the Kalman."""
        boxes_array = np.asarray(boxes, dtype=np.float64)
        if boxes_array.size:
            if boxes_array.ndim == 1:
                boxes_array = boxes_array.reshape(1, -1)
            if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
                raise AutoVisionError(
                    "tracker.update: boxes must be (N, 4) xyxy, got shape "
                    f"{boxes_array.shape}."
                )
            if not np.isfinite(boxes_array).all():
                raise AutoVisionError("tracker.update: boxes contain non-finite values.")
        else:
            boxes_array = boxes_array.reshape(0, 4)
        scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores_array) != len(boxes_array):
            raise AutoVisionError(
                f"tracker.update: {len(boxes_array)} boxes but {len(scores_array)} scores."
            )
        if not np.isfinite(scores_array).all():
            raise AutoVisionError("tracker.update: scores contain non-finite values.")
        if classes is None:
            classes_array = np.zeros(len(boxes_array), dtype=np.int64)
        else:
            classes_array = np.asarray(classes).reshape(-1)
            if len(classes_array) != len(boxes_array):
                raise AutoVisionError(
                    f"tracker.update: {len(boxes_array)} boxes but "
                    f"{len(classes_array)} classes."
                )
            if classes_array.dtype.kind not in "iu":
                raise AutoVisionError(
                    "tracker.update: classes must be integer-valued, got dtype "
                    f"{classes_array.dtype}."
                )
        # Zero/negative-area boxes make the Kalman innovation covariance
        # singular. Drop them at the boundary instead of failing mid-stream.
        widths = boxes_array[:, 2] - boxes_array[:, 0]
        heights = boxes_array[:, 3] - boxes_array[:, 1]
        valid = (widths > 0) & (heights > 0)
        if not valid.all():
            logger.warning(
                "tracker.update: dropped %d degenerate boxes (non-positive "
                "width/height).",
                int((~valid).sum()),
            )
            boxes_array = boxes_array[valid]
            scores_array = scores_array[valid]
            classes_array = classes_array[valid]
        return boxes_array, scores_array, classes_array.astype(np.int64)
