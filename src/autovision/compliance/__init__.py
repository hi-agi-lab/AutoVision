# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Enterprise compliance inventory + report."""

from autovision.compliance.licenses import (
    DEFAULT_ALLOWLIST,
    LicenseEntry,
    dependency_inventory,
    normalize_license,
)
from autovision.compliance.report import (
    ComplianceIssue,
    ComplianceReport,
    build_compliance_report,
)

__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "DEFAULT_ALLOWLIST",
    "LicenseEntry",
    "build_compliance_report",
    "dependency_inventory",
    "normalize_license",
]
