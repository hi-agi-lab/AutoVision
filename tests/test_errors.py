# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
from autovision import errors as err


def test_error_hierarchy():
    for exc in (
        err.UnknownModelError,
        err.ConfigError,
        err.ConfigUnknownKeyError,
        err.ConfigValidationError,
        err.ConfigMissingKeysError,
        err.UnresolvedAutoError,
        err.CheckpointError,
    ):
        assert issubclass(exc, err.AutoVisionError)
    assert issubclass(err.ConfigValidationError, err.ConfigError)


def test_errors_are_raisable_with_message():
    try:
        raise err.ConfigValidationError("boom")
    except err.AutoVisionError as error:
        assert str(error) == "boom"
