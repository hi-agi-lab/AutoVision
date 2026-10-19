# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""``autovision-cli compliance-report`` (roadmap 90d, phase 4)."""

from __future__ import annotations

from pathlib import Path

from autovision.compliance import ComplianceReport, build_compliance_report

__all__ = ["compliance_report"]


def compliance_report(
    model: str | Path | None = None,
    snapshot: str | Path | None = None,
    out: str | Path = "compliance",
    include_dependencies: bool = True,
    no_timestamp: bool = False,
) -> ComplianceReport:
    """Build and write ``out/compliance.{json,md}``."""
    report = build_compliance_report(
        model=model,
        snapshot=snapshot,
        include_dependencies=bool(include_dependencies),
        timestamp=not bool(no_timestamp),
    )
    return report.save(out)
