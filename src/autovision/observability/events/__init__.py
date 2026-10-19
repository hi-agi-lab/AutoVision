# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Opt-in telemetry (disabled by default; see tracker module docstring)."""

from autovision.observability.events.tracker import (
    track_event,
    track_inference_started,
    track_training_started,
)

__all__ = ["track_event", "track_inference_started", "track_training_started"]
