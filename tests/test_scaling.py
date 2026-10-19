# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest

from autovision.scaling import (
    IMAGENET_SIZE,
    QUEUE_SIZE_BUCKETS,
    ScalingInfo,
    epochs_from_steps,
    get_bucket_value,
    interpolate,
)


class TestInterpolate:
    def test_linear(self):
        assert interpolate(50, 0, 100, 0, 10) == 5

    def test_clamped(self):
        assert interpolate(-10, 0, 100, 0, 10) == 0
        assert interpolate(200, 0, 100, 0, 10) == 10

    def test_rounding(self):
        assert interpolate(1, 0, 3, 0, 1, round_ndigits=2) == 0.33

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="input_end"):
            interpolate(1, 0, 0, 0, 10)


class TestBucketValue:
    def test_first_matching_threshold(self):
        assert get_bucket_value(3, [(5, "a"), (50, "b"), (float("inf"), "c")]) == "a"
        assert get_bucket_value(49, [(5, "a"), (50, "b"), (float("inf"), "c")]) == "b"
        assert get_bucket_value(10**9, [(5, "a"), (50, "b"), (float("inf"), "c")]) == "c"

    def test_above_all_thresholds_raises(self):
        with pytest.raises(ValueError, match="larger than all bucket"):
            get_bucket_value(10, [(5, "a")])


class TestQueueBuckets:
    """Golden table from the design doc §7.3 / baseline distillationv3."""

    @pytest.mark.parametrize(
        ("dataset_size", "expected"),
        [
            (1, 16),
            (49, 16),
            (50, 32),
            (99, 32),
            (100, 64),
            (250, 128),
            (500, 256),
            (1000, 512),
            (2000, 1024),
            (4000, 2048),
            (10000, 4096),
            (20000, 8192),
            (10**9, 8192),
        ],
    )
    def test_buckets(self, dataset_size, expected):
        assert get_bucket_value(dataset_size, QUEUE_SIZE_BUCKETS) == expected


class TestEpochsFromSteps:
    def test_dinov2_style(self):
        # 125k steps at global batch 1024 over 1M images -> ~128 epochs.
        assert epochs_from_steps(125_000, 1024, IMAGENET_SIZE) == 128

    def test_small_dataset_rounds_up(self):
        assert epochs_from_steps(100, 8, 100) == 8

    def test_minimum_one_epoch(self):
        assert epochs_from_steps(1, 64, 10_000_000) == 1

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            epochs_from_steps(10, 0, 100)
        with pytest.raises(ValueError):
            epochs_from_steps(10, 8, 0)


def test_scaling_info_frozen():
    info = ScalingInfo(dataset_size=100, epochs=10)
    try:
        info.dataset_size = 200  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("ScalingInfo must be frozen")


class TestGetEpochs:
    """The ``epochs="auto"`` resolution reads the recipe args' ClassVars
    (``auto_steps``/``auto_epochs``) — regression guard for the 2026-10-15
    rename, which a string-based getattr had silently detached."""

    def test_step_recipe_wins(self):
        from autovision.cli.train_helpers import get_epochs

        class Args:
            auto_steps: int = 125_000

        class Recipe:
            @staticmethod
            def method_args_cls():
                return Args

        assert get_epochs(Recipe, "auto", dataset_size=50_000, batch_size=64) == 160  # ceil(125_000 * 64 / 50_000)

    def test_falls_back_to_auto_epochs(self):
        from autovision.cli.train_helpers import get_epochs

        class Args:
            auto_epochs: int = 100

        class Recipe:
            @staticmethod
            def method_args_cls():
                return Args

        assert get_epochs(Recipe, "auto", dataset_size=50_000, batch_size=64) == 100

    def test_neither_defined_raises(self):
        from autovision.cli.train_helpers import get_epochs

        class Args:
            pass

        class Recipe:
            @staticmethod
            def method_args_cls():
                return Args

        with pytest.raises(ValueError, match="auto_steps"):
            get_epochs(Recipe, "auto", dataset_size=50_000, batch_size=64)
