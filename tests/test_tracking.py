# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""ByteTrack association (roadmap A3) + the inference-stream wiring.

The core tests are pure numpy (always run); the stream test needs the
``video`` extra (OpenCV) and a tiny RT-DETR forward (CPU).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from autovision.errors import AutoVisionError
from autovision.tracking import BYTETracker

try:  # pragma: no cover - import guard mirrors the inference tests
    import cv2  # noqa: F401

    _cv2_installed = True
except ImportError:  # pragma: no cover
    _cv2_installed = False

requires_video = pytest.mark.skipif(
    not _cv2_installed, reason="opencv not installed (pip install -e '.[video]')"
)

VIDEO_FRAMES = 8
VIDEO_SIZE = (96, 64)  # (width, height)


def _build_tiny_model():
    """A random-weight RT-DETR tiny model (CPU)."""
    import torch

    from autovision.tasks import RTDETRObjectDetection

    torch.manual_seed(0)
    model = RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=1, num_heads=4, num_top_queries=10,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    """A random-weight RT-DETR tiny model (static query grid, CPU)."""
    return _build_tiny_model()


@pytest.fixture(scope="module")
def score_boosted_model():
    """The tiny model with its detection scores lifted into a usable band.

    The random-weight fixture emits sigmoid scores around 0.02 — below
    ByteTrack's ``det_thresh`` gate (``track_thresh + 0.1``), so reference
    semantics would never start a single track. Only the recorded scores are
    rescaled here; the geometry is untouched, so the association behaves
    exactly as it would on a trained model.
    """
    import torch

    model = _build_tiny_model()
    original = model.postprocess

    def boosted(raw, metadata, *, threshold=0.5):
        records = original(raw, metadata, threshold=threshold)
        for record in records:
            scores = record.get("scores")
            if torch.is_tensor(scores):
                record["scores"] = torch.clamp(scores * 40.0, max=0.99)
        return records

    model.postprocess = boosted
    return model


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """A deterministic synthetic MP4 (a moving box over noise)."""
    import cv2

    path = tmp_path_factory.mktemp("track_clip") / "clip.mp4"
    width, height = VIDEO_SIZE
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for index in range(VIDEO_FRAMES):
        frame = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (40 + index, 40), (255, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def _box(x: float, y: float = 10.0, size: float = 30.0) -> list[float]:
    return [x, y, x + size, y + size]


def _empty() -> tuple[np.ndarray, np.ndarray, None]:
    return np.zeros((0, 4)), np.zeros(0), None


class TestBYTETrackerCore:
    def test_moving_box_keeps_one_id(self):
        tracker = BYTETracker()
        ids = set()
        for frame in range(12):
            x = 10 + frame * 4
            tracks = tracker.update(
                np.array([_box(x)]), np.array([0.9]), np.array([0])
            )
            ids.update(track.track_id for track in tracks)
        assert ids == {1}  # a single, stable identity

    def test_degenerate_boxes_are_dropped_not_fatal(self):
        """Zero-area boxes must not reach the Kalman solve."""
        tracker = BYTETracker()
        tracks = tracker.update(
            np.array([[10.0, 10.0, 10.0, 20.0], [20.0, 20.0, 30.0, 30.0]]),
            np.array([0.9, 0.9]),
            np.array([0, 0]),
        )
        assert len(tracks) == 1
        assert tracks[0].bbox == (20.0, 20.0, 30.0, 30.0)

    def test_two_parallel_boxes_keep_distinct_ids(self):
        tracker = BYTETracker()
        seen: dict[int, list[float]] = {}
        for frame in range(10):
            boxes = np.array([_box(10 + frame * 3), _box(10 + frame * 3, y=200.0)])
            tracks = tracker.update(boxes, np.array([0.9, 0.85]), np.array([0, 0]))
            assert len(tracks) == 2
            for track in tracks:
                seen.setdefault(track.track_id, []).append(track.bbox[1])
        assert len(seen) == 2
        # Each id stayed on its own lane (class/space association, no swap).
        centers = sorted(next(iter(values)) for values in seen.values())
        assert centers[0] < 100.0 < centers[1]

    def test_new_track_is_published_only_after_two_hits(self):
        """The reference confirmation gate: a one-frame blip never ships."""
        tracker = BYTETracker()
        tracker.update(*_empty())  # burn frame 1 (the activation exception)
        assert tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0])) == []
        # The second consecutive hit confirms it — and only then is it published.
        published = tracker.update(np.array([_box(11.0)]), np.array([0.9]), np.array([0]))
        assert len(published) == 1
        assert published[0].track_id == 1
        # A track appearing on frame 1 is the reference's own exception.
        first_frame = BYTETracker().update(
            np.array([_box(10.0)]), np.array([0.9]), np.array([0])
        )
        assert len(first_frame) == 1

    def test_class_aware_association(self):
        """A box that changes class starts a NEW id (a class-agnostic
        tracker would continue the old identity on the IoU match)."""
        tracker = BYTETracker()
        first = tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0]))
        original = first[0]
        # Frame 2: same place, class 1 — no match, and the new id is still
        # unconfirmed (nothing published this frame).
        assert tracker.update(np.array([_box(11.0)]), np.array([0.9]), np.array([1])) == []
        reclassified = tracker.update(
            np.array([_box(12.0)]), np.array([0.9]), np.array([1])
        )
        assert reclassified
        assert reclassified[0].track_id != original.track_id
        assert reclassified[0].class_id == 1

    def test_dropout_within_buffer_restores_the_id(self):
        tracker = BYTETracker(track_buffer=30)
        first = tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0]))
        track_id = first[0].track_id
        for _ in range(3):
            assert tracker.update(*_empty()) == []
        revived = tracker.update(np.array([_box(12.0)]), np.array([0.9]), np.array([0]))
        assert revived and revived[0].track_id == track_id

    def test_track_retires_after_the_buffer(self):
        tracker = BYTETracker(track_buffer=10)  # max_time_lost = 10 frames
        first = tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0]))
        old_id = first[0].track_id
        for _ in range(12):
            tracker.update(*_empty())
        assert tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0])) == []
        fresh = tracker.update(np.array([_box(11.0)]), np.array([0.9]), np.array([0]))
        assert fresh and fresh[0].track_id != old_id

    def test_low_score_detections_never_spawn_tracks(self):
        for score in (0.05, 0.3, 0.55):  # below floor / in the low pool / below det_thresh
            spawner = BYTETracker(track_thresh=0.5)
            for frame in range(3):
                published = spawner.update(
                    np.array([_box(10.0 + frame)]), np.array([score]), np.array([0])
                )
                assert published == [], f"score={score} spawned a track"

    def test_low_score_detection_maintains_an_existing_track(self):
        """Stage 2: a low-score box at the track's place keeps it alive."""
        tracker = BYTETracker(track_thresh=0.5)
        first = tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0]))
        track_id = first[0].track_id
        # The high-score box at 300 starts the confirmation clock (unpublished).
        second = tracker.update(
            np.array([_box(11.0), _box(300.0)]),
            np.array([0.3, 0.9]),
            np.array([0, 0]),
        )
        assert {track.track_id for track in second} == {track_id}
        third = tracker.update(
            np.array([_box(12.0), _box(301.0)]),
            np.array([0.3, 0.9]),
            np.array([0, 0]),
        )
        ids = {track.track_id for track in third}
        assert len(ids) == 2 and track_id in ids  # maintained + confirmed

    def test_low_score_box_is_consumed_once_and_never_revives_lost_tracks(self):
        """Regression: one low detection must not sustain/revive two tracks.

        (A previous stage-3 association reused the whole low-score pool, so a
        single low box could update both a tracked and a lost track.)
        """
        tracker = BYTETracker(track_thresh=0.5)
        both = tracker.update(
            np.array([_box(10.0), _box(13.0)]), np.array([0.9, 0.9]), np.array([0, 0])
        )
        assert len(both) == 2
        # Only the first track gets a high box: the second goes lost.
        second = tracker.update(np.array([_box(10.0)]), np.array([0.9]), np.array([0]))
        assert len(second) == 1
        # ONE low box between the two: it may sustain the tracked one and
        # must never revive the lost one.
        third = tracker.update(np.array([_box(11.5)]), np.array([0.3]), np.array([0]))
        assert len(third) == 1
        # And a sustained low-only stream never resurrects the lost track.
        for frame in range(3):
            fourth = tracker.update(
                np.array([_box(11.5 + frame)]), np.array([0.3]), np.array([0])
            )
            assert len(fourth) == 1

    def test_age_counts_frames_since_creation(self):
        tracker = BYTETracker()
        ages = []
        for frame in range(5):
            tracks = tracker.update(
                np.array([_box(10.0 + frame)]), np.array([0.9]), np.array([0])
            )
            ages.append(tracks[0].age if tracks else None)
        assert ages == [0, 1, 2, 3, 4]

    def test_frame_rate_scales_the_lost_buffer(self):
        slow = BYTETracker(track_buffer=30, frame_rate=10)  # 10 lost frames
        assert slow.max_time_lost == 10

    def test_determinism_across_fresh_instances(self):
        def run() -> list[list[int]]:
            tracker = BYTETracker()
            history = []
            for frame in range(8):
                boxes = np.array([_box(10 + frame * 3), _box(60 - frame * 3, y=50.0)])
                tracks = tracker.update(boxes, np.array([0.9, 0.8]), np.array([0, 1]))
                history.append(sorted(track.track_id for track in tracks))
            return history

        assert run() == run()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"track_thresh": 1.5}, "track_thresh"),
            ({"match_thresh": -0.1}, "match_thresh"),
            ({"track_buffer": -1}, "track_buffer"),
            ({"frame_rate": 0}, "frame_rate"),
        ],
    )
    def test_invalid_parameters_fail_loud(self, kwargs, match):
        with pytest.raises(AutoVisionError, match=match):
            BYTETracker(**kwargs)

    def test_box_score_count_mismatch_fails_loud(self):
        tracker = BYTETracker()
        with pytest.raises(AutoVisionError, match="boxes but"):
            tracker.update(np.array([_box(0.0)]), np.array([0.9, 0.9]), None)
        with pytest.raises(AutoVisionError, match="classes"):
            tracker.update(np.array([_box(0.0)]), np.array([0.9]), np.array([0, 1]))

    def test_bad_geometry_fails_loud(self):
        tracker = BYTETracker()
        with pytest.raises(AutoVisionError, match="shape"):
            tracker.update(np.zeros((2, 5)), np.ones(2), None)
        with pytest.raises(AutoVisionError, match="non-finite"):
            tracker.update(np.array([[np.nan, 0.0, 1.0, 1.0]]), np.ones(1), None)
        with pytest.raises(AutoVisionError, match="non-finite"):
            tracker.update(np.array([_box(0.0)]), np.array([np.inf]), None)
        with pytest.raises(AutoVisionError, match="integer-valued"):
            tracker.update(np.array([_box(0.0)]), np.array([0.9]), np.array([0.5]))


@requires_video
class TestStreamTracking:
    def test_run_stream_track_persists_ids(self, score_boosted_model, clip, tmp_path):
        from autovision.inference import run_stream

        result = run_stream(
            clip, tmp_path / "tracked", model=score_boosted_model, threshold=0.0,
            track=True, track_thresh=0.0,  # keep every boosted detection
            batch_size=2, class_names={0: "a", 1: "b"},
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert len(records) == VIDEO_FRAMES
        assert all("tracks" in record for record in records)
        ids_per_frame = [set(t["track_id"] for t in r["tracks"]) for r in records]
        assert ids_per_frame[0], "expected tracks from frame 0"
        assert result.tracks == len(set().union(*ids_per_frame))
        # The tiny model emits a static query grid: ids persist every frame.
        stable = set(ids_per_frame[0]).intersection(*ids_per_frame)
        assert stable, "expected at least one id across every frame"
        # ``age`` counts frames since creation: a track that spans the whole
        # clip starts at 0 and ends at ``VIDEO_FRAMES - 1``.
        stable_id = min(stable)
        ages = [
            {t["track_id"]: t["age"] for t in r["tracks"]}[stable_id]
            for r in records
        ]
        assert ages == list(range(VIDEO_FRAMES))
        # The overlay carries the ids.
        assert result.video_path is not None and result.video_path.exists()

    def test_run_stream_without_track_has_no_tracks_key(self, tiny_model, clip, tmp_path):
        from autovision.inference import run_stream

        result = run_stream(
            clip, tmp_path / "plain", model=tiny_model, threshold=0.0, batch_size=2
        )
        records = [
            json.loads(line)
            for line in result.predictions_path.read_text().splitlines()
        ]
        assert len(records) == VIDEO_FRAMES
        assert all("tracks" not in record for record in records)
        assert result.tracks == 0
