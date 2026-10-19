# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Enterprise compliance report (roadmap 90d, phase 4).

Compiles the evidence an enterprise review asks for: framework identity,
installed dependency licenses, model card + weights sha256, dataset
snapshot, and explicit issues. It does not make legal conclusions; it makes
the evidence auditable and repeatable.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from autovision.compliance.inventory import (
    dataset_inventory,
    framework_inventory,
    model_inventory,
)
from autovision.compliance.licenses import (
    DEFAULT_ALLOWLIST,
    dependency_inventory,
)

__all__ = ["ComplianceIssue", "ComplianceReport", "build_compliance_report"]


@dataclass(frozen=True)
class ComplianceIssue:
    severity: str  # error | warning | info
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass
class ComplianceReport:
    framework: dict[str, Any]
    dependencies: list[dict[str, str]]
    models: list[dict[str, Any]]
    datasets: list[dict[str, Any]]
    issues: list[ComplianceIssue]
    generated_at: str | None = None
    report_json: Path | None = None
    report_md: Path | None = None
    report_sbom: Path | None = None
    report_manifest: Path | None = None

    @property
    def errors(self) -> list[ComplianceIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ComplianceIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def passes(self, fail_on: Literal["error", "warning", "none"] = "error") -> bool:
        """CI gate: ``error`` (default), ``warning`` or ``none``."""
        if fail_on == "none":
            return True
        if fail_on == "warning":
            return not self.errors and not self.warnings
        return not self.errors

    def summary_text(self) -> str:
        return (
            f"framework={self.framework.get('version')} "
            f"dependencies={len(self.dependencies)} models={len(self.models)} "
            f"errors={len(self.errors)} warnings={len(self.warnings)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "framework": dict(self.framework),
            "dependencies": [dict(entry) for entry in self.dependencies],
            "models": [dict(entry) for entry in self.models],
            "datasets": [dict(entry) for entry in self.datasets],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def save(self, out_dir: str | Path) -> "ComplianceReport":
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.report_json = directory / "compliance.json"
        self.report_md = directory / "compliance.md"
        self.report_sbom = directory / "compliance.sbom.json"
        self.report_manifest = directory / "compliance.manifest.json"
        self.report_json.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True)
        )
        self.report_md.write_text(self._render_markdown())
        self.report_sbom.write_text(
            json.dumps(self.to_cyclonedx(), indent=2, sort_keys=True)
        )
        # Integrity manifest: reviewers can re-hash the three artifacts and
        # compare without trusting the report's own "passed" field.
        self.report_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": self.generated_at,
                    "sha256": {
                        path.name: _file_sha256(path)
                        for path in (
                            self.report_json,
                            self.report_md,
                            self.report_sbom,
                        )
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return self

    def to_cyclonedx(self) -> dict[str, Any]:
        """Minimal CycloneDX 1.5 dependency/model/dataset SBOM."""
        components: list[dict[str, Any]] = []
        for entry in self.dependencies:
            component: dict[str, Any] = {
                "type": "library",
                "name": entry.get("name", "unknown"),
                "version": entry.get("version", ""),
            }
            license_id = entry.get("license")
            if license_id and license_id != "UNKNOWN":
                component["licenses"] = [{"license": {"id": license_id}}]
            components.append(component)
        for model in self.models:
            component = {
                "type": "machine-learning-model",
                "name": Path(str(model.get("path", "model"))).name,
                "version": str(model.get("version") or ""),
            }
            if model.get("sha256"):
                component["hashes"] = [
                    {"alg": "SHA-256", "content": str(model["sha256"])}
                ]
            components.append(component)
        for dataset in self.datasets:
            components.append(
                {
                    "type": "data",
                    "name": Path(str(dataset.get("path", "dataset"))).name,
                }
            )
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": self.generated_at,
                "tools": [
                    {
                        "vendor": "AutoVision",
                        "name": "compliance-report",
                        "version": str(self.framework.get("version", "")),
                    }
                ],
            },
            "components": components,
        }

    def _render_markdown(self) -> str:
        lines = ["# 合规报告\n"]
        lines.append(f"- 生成时间：{self.generated_at or '（未记录）'}")
        lines.append(
            f"- 结论：**{'PASS' if self.passed else 'REVIEW'}**"
            f"（errors={len(self.errors)} warnings={len(self.warnings)}）\n"
        )
        lines.append("## 框架\n")
        for key, value in sorted(self.framework.items()):
            lines.append(f"- {key}: {value}")
        lines.append("\n## 模型\n")
        if not self.models:
            lines.append("_（未提供模型）_\n")
        for model in self.models:
            lines.append(f"- path: {model.get('path')}")
            lines.append(f"- sha256: {model.get('sha256')}")
            lines.append(f"- license: {model.get('license', 'UNKNOWN')}")
        lines.append("\n## 数据集\n")
        if not self.datasets:
            lines.append("_（未提供快照）_\n")
        for dataset in self.datasets:
            lines.append(f"- path: {dataset.get('path')}")
        lines.append("\n## 依赖许可\n")
        if not self.dependencies:
            lines.append("_（未扫描依赖）_\n")
        else:
            lines.append("| package | version | license | status |")
            lines.append("|---|---|---|---|")
            for entry in self.dependencies:
                lines.append(
                    f"| {entry.get('name')} | {entry.get('version')} | "
                    f"{entry.get('license')} | {entry.get('status')} |"
                )
        lines.append("\n## 问题\n")
        if not self.issues:
            lines.append("无。\n")
        else:
            lines.append("| severity | kind | detail |")
            lines.append("|---|---|---|")
            for issue in self.issues:
                detail = issue.detail.replace("|", "\\|")
                lines.append(f"| {issue.severity} | {issue.kind} | {detail} |")
        return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_compliance_report(
    *,
    model: str | Path | None = None,
    snapshot: str | Path | None = None,
    include_dependencies: bool = True,
    allowlist: Iterable[str] | None = None,
    timestamp: bool = True,
) -> ComplianceReport:
    """Build the report; missing model cards/snapshots are explicit issues."""
    allowed = frozenset(allowlist) if allowlist is not None else DEFAULT_ALLOWLIST
    issues: list[ComplianceIssue] = []

    dependencies: list[dict[str, str]] = []
    if include_dependencies:
        for entry in dependency_inventory(skip=("autovision-framework",)):
            license_id = entry.license
            if license_id == "UNKNOWN":
                status = "unknown"
                issues.append(
                    ComplianceIssue(
                        "warning", "unknown_dependency_license",
                        f"{entry.name} {entry.version} has no recognized license.",
                    )
                )
            elif license_id in allowed:
                status = "allowed"
            else:
                status = "review"
                issues.append(
                    ComplianceIssue(
                        "warning", "dependency_license_review",
                        f"{entry.name} {entry.version} uses {license_id}.",
                    )
                )
            dependencies.append({**entry.to_dict(), "status": status})

    models: list[dict[str, Any]] = []
    if model is not None:
        inventory, model_issues = model_inventory(model)
        if inventory is not None:
            models.append(inventory)
            license_id = str(inventory.get("license", "UNKNOWN"))
            if license_id == "UNKNOWN":
                issues.append(
                    ComplianceIssue(
                        "error", "unknown_model_license",
                        "Model weights have no declared license.",
                    )
                )
            elif license_id not in allowed:
                issues.append(
                    ComplianceIssue(
                        "warning", "model_license_review",
                        f"Model weights use {license_id}.",
                    )
                )
        for issue in model_issues:
            issues.append(
                ComplianceIssue(
                    issue["severity"], issue["kind"], issue["detail"]
                )
            )

    datasets: list[dict[str, Any]] = []
    dataset_entry, dataset_issues = dataset_inventory(snapshot)
    if dataset_entry is not None:
        datasets.append(dataset_entry)
    for issue in dataset_issues:
        issues.append(
            ComplianceIssue(issue["severity"], issue["kind"], issue["detail"])
        )

    return ComplianceReport(
        framework=framework_inventory(),
        dependencies=dependencies,
        models=models,
        datasets=datasets,
        issues=issues,
        generated_at=(
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if timestamp
            else None
        ),
    )
