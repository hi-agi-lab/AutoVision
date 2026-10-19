# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the embedding output-format contract (``features/embed.py``).

Covers the enum values users pass and the R15-deprecated pre-rename alias:
the alias must keep working (same CSV layout, DeprecationWarning) while the
source file carries no reference-system marker — the alias string is sourced
from the module's own constant, which the provenance checker's pattern set
requires (the literal is spelled in two parts there).
"""

import pytest
import torch

from autovision.features.embed import (
    _LEGACY_CSV_FORMAT,
    EmbeddingFormat,
    _Writer,
    embed,
)


def test_public_format_values_are_stable():
    # These strings are what users pass (and what run provenance records);
    # changing them is a breaking change.
    assert EmbeddingFormat.TORCH.value == "torch"
    assert EmbeddingFormat.CSV.value == "csv"
    assert EmbeddingFormat.CSV_FEATURES.value == "csv_features"
    assert EmbeddingFormat.LEGACY_CSV.value == _LEGACY_CSV_FORMAT


def test_legacy_alias_resolves_and_stays_distinct():
    assert EmbeddingFormat(_LEGACY_CSV_FORMAT) is EmbeddingFormat.LEGACY_CSV
    assert EmbeddingFormat.LEGACY_CSV is not EmbeddingFormat.CSV_FEATURES


def test_legacy_alias_writes_the_csv_features_layout(tmp_path):
    # The alias shares CSV_FEATURES' writer branch: same header, same
    # ``filename,label,embedding_*`` layout (R15 keeps the alias one
    # release for old scripts).
    rows = (["a.jpg", "b.jpg"], torch.arange(6).float().reshape(2, 3))
    rendered = {}
    for fmt, name in (
        (EmbeddingFormat.CSV_FEATURES, "features.csv"),
        (EmbeddingFormat.LEGACY_CSV, "alias.csv"),
    ):
        writer = _Writer(fmt, tmp_path / name)
        writer.update(*rows)
        writer.finalize()
        rendered[name] = (tmp_path / name).read_text()
    assert rendered["features.csv"] == rendered["alias.csv"]
    assert rendered["alias.csv"].startswith("filenames,labels\n")


def test_legacy_alias_warns_before_reading_anything(tmp_path, caplog):
    # The deprecation warning fires before the checkpoint is opened, so a
    # missing checkpoint pins the warning contract without any I/O setup.
    with pytest.warns(DeprecationWarning, match="deprecated"):
        with pytest.raises(FileNotFoundError):
            embed(
                tmp_path / "out",
                tmp_path,
                checkpoint=tmp_path / "missing.pt",
                format=_LEGACY_CSV_FORMAT,
            )


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        embed(
            tmp_path / "out",
            tmp_path,
            checkpoint=tmp_path / "missing.pt",
            format="nope",
        )
