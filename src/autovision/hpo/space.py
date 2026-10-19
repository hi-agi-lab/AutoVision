# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Hyperparameter search spaces (roadmap B2).

A space is a flat mapping of **dotted override keys** to distribution
specs; ``split_overrides`` turns one sampled point back into the keyword
arguments the training entry takes — i.e. the search rides the SAME
Pydantic-validated override surface as the CLI (``model_args.lr`` →
``model_args={"lr": ...}``, a bare ``batch_size`` → a top-level kwarg), so
a space that names a nonexistent field fails loudly at trial 1 instead of
being silently ignored.

Spec forms (yaml/json/dict)::

    model_args.lr:            {loguniform: [1.0e-5, 1.0e-3]}
    model_args.weight_decay:   {choice: [0.0, 1.0e-4, 5.0e-4]}
    model_args.ema_warmup_steps: {randint: [100, 1000]}
    batch_size:                [2, 4, 8]        # bare list = choice

Kinds: ``choice`` (list), ``uniform``/``loguniform`` ([lo, hi] floats),
``randint`` ([lo, hi] ints, inclusive). Sampling is seeded and
deterministic; ``grid`` enumerates the cartesian product and therefore
requires every axis to be a ``choice``.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError

__all__ = [
    "Axis",
    "SearchSpace",
    "load_space",
    "split_overrides",
]

_KINDS = ("choice", "uniform", "loguniform", "randint")


@dataclass(frozen=True)
class Axis:
    """One sampled hyperparameter axis."""

    key: str
    kind: str
    spec: Any

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            return rng.choice(self.spec)
        low, high = self.spec
        if self.kind == "uniform":
            return rng.uniform(low, high)
        if self.kind == "loguniform":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        if self.kind == "randint":
            return rng.randint(int(low), int(high))
        raise AssertionError(f"unhandled kind '{self.kind}'")

    def choices(self) -> list[Any] | None:
        return list(self.spec) if self.kind == "choice" else None

    def describe(self) -> str:
        if self.kind == "choice":
            return f"choice{list(self.spec)}"
        return f"{self.kind}{list(self.spec)}"


class SearchSpace:
    """Named axes with deterministic sampling and grid enumeration."""

    def __init__(self, axes: Sequence[Axis]) -> None:
        if not axes:
            raise AutoVisionError("A search space needs at least one axis.")
        keys = [axis.key for axis in axes]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise AutoVisionError(f"Duplicate search-space keys: {duplicates}.")
        self.axes = list(axes)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SearchSpace":
        axes = []
        for key, spec in mapping.items():
            if not isinstance(key, str) or not key:
                raise AutoVisionError(f"Search-space keys must be non-empty strings, got {key!r}.")
            if any(not segment for segment in key.split(".")):
                raise AutoVisionError(
                    f"Axis '{key}': dotted keys must not contain empty segments."
                )
            if isinstance(spec, Mapping):
                if len(spec) != 1:
                    raise AutoVisionError(
                        f"Axis '{key}': expected exactly one kind key from {list(_KINDS)}, "
                        f"got {sorted(spec)}."
                    )
                ((kind, value),) = spec.items()
                if kind not in _KINDS:
                    raise AutoVisionError(
                        f"Axis '{key}': unknown kind '{kind}'; supported: {list(_KINDS)}."
                    )
            else:
                kind, value = "choice", spec
            if isinstance(value, (list, tuple)):
                value = list(value)
            if kind == "choice":
                if not isinstance(value, list) or not value:
                    raise AutoVisionError(f"Axis '{key}': 'choice' needs a non-empty list.")
            elif not isinstance(value, list) or len(value) != 2:
                raise AutoVisionError(
                    f"Axis '{key}': '{kind}' needs exactly [low, high], got {value!r}."
                )
            elif kind == "randint":
                low, high = value
                if not all(float(v).is_integer() for v in (low, high)):
                    raise AutoVisionError(
                        f"Axis '{key}': randint bounds must be whole numbers, got "
                        f"[{low}, {high}]."
                    )
                if int(low) > int(high):
                    raise AutoVisionError(f"Axis '{key}': randint low {low} > high {high}.")
            else:
                low, high = (float(v) for v in value)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise AutoVisionError(
                        f"Axis '{key}': '{kind}' bounds must be finite, got [{low}, {high}]."
                    )
                if low > high:
                    raise AutoVisionError(
                        f"Axis '{key}': '{kind}' needs low <= high, got [{low}, {high}]."
                    )
                if kind == "loguniform" and low <= 0:
                    raise AutoVisionError(
                        f"Axis '{key}': 'loguniform' needs a positive lower bound, "
                        f"got [{low}, {high}]."
                    )
                value = (low, high)
            axes.append(Axis(key=key, kind=kind, spec=value))
        return cls(axes)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {axis.key: axis.sample(rng) for axis in self.axes}

    def sample_many(self, count: int, seed: int = 0) -> list[dict[str, Any]]:
        """``count`` seeded samples; continuous spaces are not deduplicated."""
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(count)]

    def grid(self) -> list[dict[str, Any]]:
        """Cartesian product; requires every axis to be a ``choice``."""
        continuous = [axis.key for axis in self.axes if axis.kind != "choice"]
        if continuous:
            raise AutoVisionError(
                f"strategy='grid' requires discrete axes; continuous axes: {continuous}. "
                "Use {choice: [...]} or strategy='random'."
            )
        points: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for combination in itertools.product(*(axis.choices() for axis in self.axes)):
            key = tuple(repr(item) for item in combination)
            if key in seen:
                continue  # a choice with repeated values
            seen.add(key)
            points.append(dict(zip((axis.key for axis in self.axes), combination)))
        return points

    def to_dict(self) -> dict[str, str]:
        return {axis.key: axis.describe() for axis in self.axes}

    def __len__(self) -> int:
        return len(self.axes)

    def __repr__(self) -> str:
        return f"SearchSpace({', '.join(f'{a.key}: {a.describe()}' for a in self.axes)})"


def load_space(source: "Mapping[str, Any] | str | Path") -> SearchSpace:
    """Build a space from a mapping or a yaml/json file."""
    if isinstance(source, Mapping):
        return SearchSpace.from_mapping(source)
    path = Path(source)
    if not path.exists():
        raise AutoVisionError(f"Search-space file not found: '{path}'.")
    text = path.read_text()
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        mapping = json.loads(text)
    else:
        import yaml

        mapping = yaml.safe_load(text)
    if not isinstance(mapping, Mapping):
        raise AutoVisionError(f"Search-space file '{path}' must hold a mapping.")
    return SearchSpace.from_mapping(mapping)


def split_overrides(params: Mapping[str, Any]) -> dict[str, Any]:
    """Dotted override keys → the training entry's keyword arguments.

    ``{"model_args.lr": 1e-4, "batch_size": 4}`` →
    ``{"model_args": {"lr": 1e-4}, "batch_size": 4}``. Passed verbatim to
    the entry, so the Pydantic validators (strict, extra-forbid) are what
    rejects a mistyped key — the same surface the CLI uses.
    """
    keys = list(params)
    for key in keys:
        for other in keys:
            if other != key and other.startswith(key + "."):
                raise AutoVisionError(
                    f"Override '{key}' is a dotted prefix of '{other}'; nested and "
                    "scalar overrides cannot coexist on the same path."
                )
    grouped: dict[str, Any] = {}
    for key, value in params.items():
        head, _, tail = key.partition(".")
        if tail:
            # The prefix check above guarantees ``head`` is not a scalar key.
            grouped.setdefault(head, {})[tail] = value
        else:
            grouped[head] = value
    return grouped
