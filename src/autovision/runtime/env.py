# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Framework configuration surface via environment variables.

Every environment variable is declared here as a frozen :class:`EnvVar`
descriptor so that defaults, types, and parsing behavior live in one place.
All variables use the ``AUTOVISION_`` prefix.

Parsing is deliberately lenient for booleans: any of ``1/true/yes/on``
(case-insensitive) enables a flag. This avoids the baseline pitfall where a
quoted ``"1"`` value silently failed the ``x == "1"`` check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

__all__ = ["Env", "EnvVar"]

T = TypeVar("T")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_optional_str(value: str) -> str | None:
    return value or None


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    """A typed environment variable descriptor."""

    name: str
    _default: Any = field(kw_only=True)
    _type: Callable[[Any], T] = field(kw_only=True)
    convert_empty_str_to_default: bool = True
    #: Redact from Env.values() diagnostics (keys, tokens, credentials).
    secret: bool = False

    def default(self) -> T:
        return self._default() if callable(self._default) else self._default

    @property
    def value(self) -> T:
        raw = os.getenv(self.name)
        if raw is None or (self.convert_empty_str_to_default and raw == ""):
            return self.default()
        try:
            return self._type(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Environment variable {self.name}={raw!r} is invalid: {error}"
            ) from error

    def set(self, value: T | None) -> None:
        """Set (or unset with ``None``) the variable in the process env."""
        if value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(value)


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "autovision"


class Env:
    """All framework environment variables."""

    LOG_LEVEL = EnvVar[str]("AUTOVISION_LOG_LEVEL", _default="INFO", _type=str)
    CACHE_DIR = EnvVar[Path]("AUTOVISION_CACHE_DIR", _default=_cache_dir, _type=Path)
    MODEL_CACHE_DIR = EnvVar[Path](
        "AUTOVISION_MODEL_CACHE_DIR", _default=lambda: Env.CACHE_DIR.value / "models", _type=Path
    )
    TMP_DIR = EnvVar[Path](
        "AUTOVISION_TMP_DIR", _default=lambda: Env.CACHE_DIR.value / "data", _type=Path
    )
    DATA_CACHE_DIR = EnvVar[Path](
        "AUTOVISION_DATA_CACHE_DIR", _default=lambda: Env.TMP_DIR.value, _type=Path
    )
    DATALOADER_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_DATALOADER_TIMEOUT_SEC", _default=180.0, _type=float
    )
    MAX_NUM_WORKERS_AUTO = EnvVar[int](
        "AUTOVISION_MAX_NUM_WORKERS_AUTO", _default=8, _type=int
    )
    DEFAULT_NUM_WORKERS_AUTO = EnvVar[int](
        "AUTOVISION_DEFAULT_NUM_WORKERS_AUTO", _default=8, _type=int
    )
    MMAP_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_MMAP_TIMEOUT_SEC", _default=300.0, _type=float
    )
    MMAP_REUSE_FILE = EnvVar[bool](
        "AUTOVISION_MMAP_REUSE_FILE", _default=False, _type=_parse_bool
    )
    VERIFY_OUT_DIR_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_VERIFY_OUT_DIR_TIMEOUT_SEC", _default=30.0, _type=float
    )
    DOWNLOAD_CHUNK_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_DOWNLOAD_CHUNK_TIMEOUT_SEC", _default=180.0, _type=float
    )
    OFFLINE = EnvVar[bool]("AUTOVISION_OFFLINE", _default=False, _type=_parse_bool)
    # Optional API key for the reference serving app (create_app(api_key=...)
    # takes precedence). Prefer this over putting the key on the CLI.
    SERVING_API_KEY = EnvVar[str | None](
        "AUTOVISION_SERVING_API_KEY", _default=None,
        _type=_parse_optional_str, secret=True,
    )
    # Opt-in deterministic training: torch determinism + cuDNN deterministic,
    # no benchmark autotuning. Some ops have no deterministic kernel and will
    # raise; slower. See engine.seed_everything.
    DETERMINISTIC = EnvVar[bool](
        "AUTOVISION_DETERMINISTIC", _default=False, _type=_parse_bool
    )
    # Dataset downloads only succeed when the registry entry pins a sha256.
    # The built-in COCO/VOC entries currently have no official digest, so this
    # is opt-in for shops that mirror and pin their own artifacts.
    REQUIRE_DATASET_HASHES = EnvVar[bool](
        "AUTOVISION_REQUIRE_DATASET_HASHES", _default=False, _type=_parse_bool
    )
    # EoMT (and any future adapter) fails closed when a hub checkpoint has
    # no pinned sha256 in WEIGHT_MANIFEST. Setting this is an explicit
    # acceptance of an unpinned download (trusted mirror / air-gapped copy).
    ALLOW_UNPINNED_WEIGHTS = EnvVar[bool](
        "AUTOVISION_ALLOW_UNPINNED_WEIGHTS", _default=False, _type=_parse_bool
    )
    MIRROR_URL = EnvVar[str | None](
        "AUTOVISION_MIRROR_URL", _default=None, _type=_parse_optional_str
    )
    # Batched GPU LSAP for the DETR matcher (optional extra ``lsap``,
    # torch-linear-assignment, Apache-2.0): "auto" (default) uses the
    # batched GPU solver when it imports and the problems fit the kernel,
    # with automatic scipy fallback otherwise (design/measurement notes in
    # tasks/detection/shared/matcher.py); "scipy"/"0"/"off" forces the
    # per-image scipy loop.
    LSAP = EnvVar[str]("AUTOVISION_LSAP", _default="auto", _type=str)
    # Weight download source: "auto" (default), "huggingface", or
    # "modelscope". "auto" uses the hub (honoring AUTOVISION_MIRROR_URL,
    # e.g. hf-mirror.com) and falls back to a pinned ModelScope mirror when
    # the hub path fails; explicit values force one source. Applies only to
    # manifest entries that declare a `modelscope_id`; everything else
    # stays on the hub. sha256 verification is source-agnostic (pinned
    # hashes are byte-level identities).
    WEIGHT_SOURCE = EnvVar[str](
        "AUTOVISION_WEIGHT_SOURCE", _default="auto", _type=str
    )
    # Compliance: telemetry is opt-in (default off), the inverse of the baseline.
    TELEMETRY = EnvVar[bool]("AUTOVISION_TELEMETRY", _default=False, _type=_parse_bool)
    TELEMETRY_ENDPOINT = EnvVar[str](
        "AUTOVISION_TELEMETRY_ENDPOINT",
        _default="https://telemetry.example.invalid/v0/e",
        _type=str,
    )
    # Optional OpenTelemetry span emission (see _observability.otel); the
    # opentelemetry-api package itself is supplied by the host application.
    OTEL = EnvVar[bool]("AUTOVISION_OTEL", _default=False, _type=_parse_bool)
    # Legacy exported teacher checkpoints may embed a ``__class__`` object
    # that the pickle-free loader rejects. The full unpickler is an
    # arbitrary-code-execution surface, so it is opt-in for a trusted local
    # artifact only; current exports are safe-loadable.
    TRUST_TEACHER_CHECKPOINT = EnvVar[bool](
        "AUTOVISION_TRUST_TEACHER_CHECKPOINT", _default=False, _type=_parse_bool
    )
    # Process-group watchdog: a rank that dies before a collective must fail
    # the job instead of hanging forever. Applies to DDP and FSDP init.
    DDP_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_DDP_TIMEOUT_SEC", _default=1800.0, _type=float
    )
    # FSDP keeps EMA as a full (unsharded) copy; updating it requires a
    # full-parameter gather. 1 updates every step (correct, slow); raise it
    # to trade EMA freshness for throughput on large models.
    FSDP_EMA_INTERVAL = EnvVar[int](
        "AUTOVISION_FSDP_EMA_INTERVAL", _default=1, _type=int
    )
    # DDP static-graph tuning: find_unused_parameters costs a full graph
    # traversal per step; queue/EMA-free methods with stable step shapes
    # can set AUTOVISION_DDP_FIND_UNUSED=0 to shed that tax.
    DDP_FIND_UNUSED = EnvVar[bool](
        "AUTOVISION_DDP_FIND_UNUSED", _default=True, _type=_parse_bool
    )
    # Route the DETR decoders' deformable cross-attention through the
    # vendored fused MSDA CUDA op (object_detection_components/_msda_src,
    # JIT-compiled on first use; needs nvcc) instead of the per-level
    # grid_sample path. Numerically equivalent, opt-in until benchmarked.
    MSDA_CROSS_ATTN = EnvVar[bool](
        "AUTOVISION_MSDA_CROSS_ATTN", _default=False, _type=_parse_bool
    )
    # Reserved for the per-step watchdog (design §7) — declared so the
    # config surface is discoverable, not yet consumed by the engine.
    STEP_TIMEOUT_SEC = EnvVar[float](
        "AUTOVISION_STEP_TIMEOUT_SEC", _default=0.0, _type=float  # 0 disables the watchdog
    )

    @staticmethod
    def values() -> dict[str, Any]:
        """Snapshot of all variable values (for run metadata / diagnostics).

        Variables declared ``secret=True`` are redacted; the live value is
        still available through the descriptor itself.
        """
        values: dict[str, Any] = {}
        for attr in Env.__dict__.values():
            if not isinstance(attr, EnvVar):
                continue
            value = attr.value
            if attr.secret and value is not None:
                value = "<redacted>"
            values[attr.name] = value
        return values
