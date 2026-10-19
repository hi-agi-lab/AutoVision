# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Asset licensing metadata (design doc 28, section 6.3 / rule R7).

Every downloadable binary asset must carry an explicit license reference.
``LicenseRef`` is attached to :class:`~autovision.registry.downloadable.
DownloadableModel` entries; from the Phase 4 asset switch onwards the
field is mandatory and the guard fails registrations without one.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LicenseRef"]


@dataclass(frozen=True)
class LicenseRef:
    """License of a binary asset.

    Attributes:
        id: SPDX identifier (e.g. ``Apache-2.0``) or a short custom label
            for non-SPDX terms (e.g. ``DINOv3-License``).
        origin: provenance of the artifact — an upstream repository with
            commit (``https://github.com/...@<sha>``) for third-party
            weights, or a training run id for self-produced assets.
    """

    id: str
    origin: str
