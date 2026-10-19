# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Track analytics: line/polygon zones and per-track smoothing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autovision.errors import AutoVisionError
from autovision.tracking import LineZone, PolygonZone, Track, TrackSmoother
from autovision.visualize.zones import draw_zones


def _track(
    track_id: int,
    box: tuple[float, float, float, float],
    *,
    class_id: int = 0,
    score: float = 0.9,
    age: int = 0,
) -> Track:
    return Track(
        track_id=track_id,
        bbox=box,
        score=score,
        class_id=class_id,
        age=age,
    )


class TestLineZone:
    def test_counts_a_directional_crossing(self):
        zone = LineZone((0, 10), (20, 10), name="gate")
        zone.update([_track(1, (5, 0, 10, 5), class_id=0)])  # above
        counts = zone.update([_track(1, (5, 10, 10, 15), class_id=0)])  # below
        assert counts.in_count == 1
        assert counts.out_count == 0
        assert counts.in_per_class == {0: 1}
        counts = zone.update([_track(1, (5, 0, 10, 5), class_id=0)])  # back above
        assert counts.out_count == 1
        assert counts.out_per_class == {0: 1}

    def test_one_track_counts_once_per_frame(self):
        zone = LineZone((0, 10), (20, 10))
        zone.update([_track(1, (5, 0, 10, 5))])
        counts = zone.update([_track(1, (5, 10, 10, 15)), _track(1, (5, 12, 10, 17))])
        assert counts.in_count == 1

    def test_stale_tracks_are_evicted(self):
        zone = LineZone((0, 10), (20, 10), max_age=2)
        zone.update([_track(1, (5, 0, 10, 5))])
        zone.update([])
        zone.update([])
        counts = zone.update([_track(1, (5, 10, 10, 15))])
        assert counts.in_count == 0

    def test_reappearance_at_max_age_keeps_side_history(self):
        """A track absent for exactly max_age frames is still the same life."""
        zone = LineZone((0, 10), (20, 10), max_age=3)
        zone.update([_track(1, (5, 0, 10, 5))])  # frame 1: above
        zone.update([])  # frame 2
        zone.update([])  # frame 3
        counts = zone.update([_track(1, (5, 10, 10, 15))])  # frame 4: below
        assert counts.in_count == 1

    def test_reset_clears_counts(self):
        zone = LineZone((0, 10), (20, 10))
        zone.update([_track(1, (5, 0, 10, 5))])
        zone.update([_track(1, (5, 10, 10, 15))])
        zone.reset()
        assert zone.counts.in_count == 0

    def test_degenerate_line_is_rejected(self):
        with pytest.raises(AutoVisionError, match="must differ"):
            LineZone((1, 1), (1, 1))


class TestPolygonZone:
    def test_counts_tracks_inside(self):
        zone = PolygonZone([(0, 0), (20, 0), (20, 20), (0, 20)], name="square")
        counts = zone.update(
            [
                _track(1, (5, 5, 10, 10), class_id=0),
                _track(2, (30, 30, 35, 35), class_id=1),
            ]
        )
        assert counts.total == 1
        assert counts.per_class == {0: 1}

    def test_require_all_anchors(self):
        zone = PolygonZone(
            [(0, 0), (20, 0), (20, 20), (0, 20)],
            anchors=("bottom_center", "top_center"),
            require_all=True,
        )
        counts = zone.update([_track(1, (5, -5, 10, 5))])  # top outside
        assert counts.total == 0

    def test_polygon_needs_three_points(self):
        with pytest.raises(AutoVisionError, match="three points"):
            PolygonZone([(0, 0), (1, 1)])


class TestTrackSmoother:
    def test_averages_boxes_and_scores(self):
        smoother = TrackSmoother(length=3)
        smoother.update([_track(1, (0, 0, 10, 10), score=0.6)])
        smoother.update([_track(1, (2, 2, 12, 12), score=0.8)])
        smoothed = smoother.update([_track(1, (4, 4, 14, 14), score=1.0)])
        assert len(smoothed) == 1
        assert smoothed[0].bbox == pytest.approx((2.0, 2.0, 12.0, 12.0))
        assert smoothed[0].score == pytest.approx(0.8)
        assert smoothed[0].track_id == 1

    def test_histories_are_per_track(self):
        smoother = TrackSmoother(length=2)
        smoother.update([_track(1, (0, 0, 10, 10)), _track(2, (20, 20, 30, 30))])
        smoothed = smoother.update([_track(2, (22, 22, 32, 32))])
        assert len(smoothed) == 1
        assert smoothed[0].track_id == 2

    def test_stale_history_is_dropped(self):
        smoother = TrackSmoother(length=2, max_age=1)
        smoother.update([_track(1, (0, 0, 10, 10))])
        smoother.update([])
        smoother.update([])
        smoothed = smoother.update([_track(1, (10, 10, 20, 20))])
        # The evicted track restarts its history at the current box.
        assert smoothed[0].bbox == (10.0, 10.0, 20.0, 20.0)


class TestDrawZones:
    def test_draws_line_and_polygon_with_counts(self):
        line = LineZone((0, 10), (32, 10), name="gate")
        line.update([_track(1, (5, 0, 10, 5))])
        line.update([_track(1, (5, 10, 10, 15))])
        polygon = PolygonZone([(2, 2), (20, 2), (20, 20), (2, 20)], name="zone")
        polygon.update([_track(1, (5, 5, 10, 10))])
        image = Image.new("RGB", (32, 24), (0, 0, 0))
        original = image.copy()
        rendered = draw_zones(image, [line], [polygon])
        assert not np.array_equal(np.asarray(original), np.asarray(rendered))
