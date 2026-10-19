# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Reference Kubernetes manifests stay parseable and hardened."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "deploy" / "k8s"


def _docs():
    yaml = pytest.importorskip("yaml")
    docs = []
    for path in sorted(ROOT.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        docs.extend(
            doc for doc in yaml.safe_load_all(path.read_text()) if doc
        )
    return docs


def test_manifests_parse_and_cover_workload_surface():
    docs = _docs()
    kinds = {doc["kind"] for doc in docs}
    assert {"Deployment", "Service", "HorizontalPodAutoscaler", "PodDisruptionBudget"} <= kinds


def test_deployment_is_hardened_and_probed():
    deployment = next(doc for doc in _docs() if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["terminationGracePeriodSeconds"] >= 45
    container = pod["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    env = {item["name"]: item for item in container["env"]}
    assert env["AUTOVISION_SERVING_API_KEY"]["valueFrom"]["secretKeyRef"]["name"]


def test_kustomization_lists_every_manifest():
    yaml = pytest.importorskip("yaml")
    kustomization = yaml.safe_load((ROOT / "kustomization.yaml").read_text())
    listed = set(kustomization["resources"])
    present = {path.name for path in ROOT.glob("*.yaml")} - {"kustomization.yaml"}
    assert listed == present
