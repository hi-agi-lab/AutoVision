# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dependency-free token-bucket rate limiting for the reference serving app.

Per-process limits: with multiple uvicorn workers each worker enforces its own
bucket, so the effective aggregate is ``workers × limit``. Global quotas belong
at the ingress/gateway layer; this guard protects an individual process from a
single noisy client.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

__all__ = ["TokenBucketRateLimiter"]


class TokenBucketRateLimiter:
    """Thread-safe per-key token bucket.

    ``per_minute`` is the sustained refill rate; ``burst`` is the bucket
    capacity (default: ten seconds worth, at least one request).
    """

    def __init__(
        self,
        per_minute: int,
        burst: int | None = None,
        *,
        clock: Callable[[], float] | None = None,
        max_buckets: int = 10_000,
    ) -> None:
        if per_minute <= 0:
            raise ValueError(f"per_minute must be > 0, got {per_minute}.")
        self.rate_per_second = per_minute / 60.0
        resolved_burst = (
            burst if burst is not None
            else max(1, math.ceil(self.rate_per_second * 10.0))
        )
        if resolved_burst < 1:
            raise ValueError(f"burst must be >= 1, got {resolved_burst}.")
        self.burst = float(resolved_burst)
        self._clock = clock or time.monotonic
        self._max_buckets = max_buckets
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """Consume one token for ``key``; return ``(allowed, retry_after_s)``."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(
                key, (self.burst, now)
            )
            tokens = min(
                self.burst, tokens + max(0.0, now - last) * self.rate_per_second
            )
            if tokens >= 1.0:
                tokens -= 1.0
                allowed = True
                retry_after = 0.0
            else:
                allowed = False
                retry_after = (1.0 - tokens) / self.rate_per_second
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self._max_buckets:
                self._buckets.popitem(last=False)
            return allowed, retry_after
