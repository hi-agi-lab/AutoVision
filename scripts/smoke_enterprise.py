#!/usr/bin/env python
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Enterprise smoke test: end-to-end functional checks in one command.

Quick profile (default, ~1 minute):

    python scripts/smoke_enterprise.py

Full profile (adds DDP, FSDP, OpenVINO, TensorRT; several minutes):

    python scripts/smoke_enterprise.py --profile full

The script is intentionally black-box: it trains a tiny detector, exports,
serves, registers, builds a compliance pack, and snapshots data. It does NOT
prove production scale, soak/load behavior, or multi-node NCCL — see
docs/smoke_test.md for the boundary. Exit code 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import autovision  # noqa: E402


@dataclass
class Result:
    name: str
    status: str  # PASS | SKIP | FAIL
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Smoke:
    profile: str
    tmp: Path
    results: list[Result] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def run(self, name: str, fn, *, required: bool = True) -> None:
        started = time.perf_counter()
        try:
            detail = fn()
            status = "PASS"
            detail = "" if detail is None else str(detail)
        except SkipCheck as error:
            status = "SKIP"
            detail = str(error)
        except Exception as error:  # noqa: BLE001 - smoke boundary
            status = "FAIL" if required else "SKIP"
            detail = f"{type(error).__name__}: {error}"
            if required:
                traceback.print_exc()
        elapsed = time.perf_counter() - started
        self.results.append(Result(name, status, detail, elapsed))
        print(f"[{status:4s}] {name:32s} {elapsed:6.1f}s  {detail}")


class SkipCheck(RuntimeError):
    pass


def _make_coco(root: Path) -> tuple[Path, Path, Path]:
    images = root / "images"
    images.mkdir(parents=True)
    train_rows, train_anns, val_rows, val_anns = [], [], [], []
    rng = np.random.default_rng(0)
    for index in range(8):
        array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(images / f"img_{index:03d}.jpg")
        row = {
            "id": index + 1,
            "file_name": f"img_{index:03d}.jpg",
            "width": 64,
            "height": 64,
        }
        ann = {
            "id": index + 1,
            "image_id": index + 1,
            "category_id": 1 + index % 2,
            "bbox": [8.0, 8.0, 32.0, 32.0],
            "area": 1024.0,
            "iscrowd": 0,
        }
        train_rows.append(row)
        train_anns.append(ann)
        if index < 4:
            val_rows.append(row)
            val_anns.append(ann)
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]

    def write(name: str, rows, anns) -> Path:
        path = root / name
        path.write_text(
            json.dumps(
                {"images": rows, "annotations": anns, "categories": categories}
            )
        )
        return path

    return write("train.json", train_rows, train_anns), write(
        "val.json", val_rows, val_anns
    ), images


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _env_check(smoke: Smoke) -> str:
    info = {
        "python": sys.version.split()[0],
        "autovision": autovision.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    optional = {}
    for module in (
        "onnx",
        "onnxruntime",
        "tensorrt",
        "openvino",
        "mlflow",
        "fastapi",
        "uvicorn",
        "cv2",
    ):
        try:
            __import__(module)
            optional[module] = True
        except Exception:  # noqa: BLE001
            optional[module] = False
    smoke.context["env"] = {**info, "optional": optional}
    missing = sorted(name for name, ok in optional.items() if not ok)
    return (
        f"python={info['python']} torch={info['torch']} cuda={info['cuda']} "
        f"gpu={info['gpu']}; missing optional: {missing or 'none'}"
    )


def _cli_check(smoke: Smoke) -> str:
    cli = Path(sys.executable).with_name("autovision-cli")
    if not cli.is_file():
        cli = Path(shutil.which("autovision-cli") or "")
    if not cli.is_file():
        raise SkipCheck("autovision-cli is not installed on PATH")

    def run(*args: str) -> None:
        proc = subprocess.run(
            [str(cli), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"autovision-cli {' '.join(args)} failed: "
                f"{(proc.stderr or proc.stdout)[-800:]}"
            )

    run("--help")
    run("list-models")
    run("model-info", "rtdetr/resnet18")
    return "help / list-models / model-info"


def _train_check(smoke: Smoke) -> str:
    train_json, val_json, images = _make_coco(smoke.tmp / "coco")
    data = {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images)},
        "val": {"annotations": str(val_json), "images": str(images)},
    }
    out = smoke.tmp / "run"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = autovision.train_object_detection(
        out=out,
        data=data,
        model="rtdetr/resnet18",
        steps=2,
        batch_size=2,
        num_workers=0,
        devices=device,
        precision="32-true",
        model_args={
            "image_size": (64, 64),
            "hidden_dim": 32,
            "num_queries": 10,
            "dec_layers": 2,
            "num_heads": 4,
            "num_top_queries": 10,
            "lr_warmup_steps": 1,
        },
        transform_args={
            "scale_jitter": None,
            "photometric_distort_prob": 0.0,
            "random_zoom_out_prob": None,
            "random_iou_crop_prob": None,
        },
        save_checkpoint_args={
            "save_every_num_steps": 1,
            "save_last": True,
            "save_best": True,
        },
        trainer_args={"val_every_num_steps": 1},
    )
    for name, path in (
        ("checkpoint", result.checkpoint_path),
        ("export", result.exported_model_path),
        ("manifest", out / "run_manifest.json"),
        ("metrics", out / "metrics.jsonl"),
    ):
        if path is None or not Path(path).is_file():
            raise RuntimeError(f"training artifact missing: {name} ({path})")
    net = autovision.load_task_model(result.exported_model_path, device="cpu")
    prediction = net.predict(images / "img_000.jpg")
    if not prediction:
        raise RuntimeError("exported model returned no prediction structure")
    smoke.context.update(
        {
            "exported": Path(result.exported_model_path),
            "dataset": (train_json, val_json, images),
            "run": out,
        }
    )
    return (
        f"steps={result.steps_run} device={device} "
        f"map={result.best_metrics.get('val_metric/map', 0.0):.4f}"
    )


def _onnx_check(smoke: Smoke) -> str:
    net = autovision.load_task_model(smoke.context["exported"], device="cpu")
    out = smoke.tmp / "model.onnx"
    net.export_onnx(
        out, batch_size=1, dynamic_batch_size=False, simplify=False, verify=True
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("ONNX artifact is empty")
    smoke.context["onnx"] = out
    return f"{out.name} {out.stat().st_size // 1024} KiB, ORT parity passed"


def _openvino_check(smoke: Smoke) -> str:
    try:
        import openvino  # noqa: F401
    except ImportError as error:
        raise SkipCheck("openvino is not installed") from error
    net = autovision.load_task_model(smoke.context["exported"], device="cpu")
    out = smoke.tmp / "model.xml"
    net.export_openvino(out, precision="fp32", device="CPU", verify=True)
    if not out.is_file():
        raise RuntimeError("OpenVINO IR missing")
    return f"{out.name}, compiled-model parity passed"


def _tensorrt_check(smoke: Smoke) -> str:
    if not torch.cuda.is_available():
        raise SkipCheck("TensorRT needs CUDA")
    try:
        import tensorrt  # noqa: F401
    except ImportError as error:
        raise SkipCheck("tensorrt is not installed") from error
    net = autovision.load_task_model(smoke.context["exported"], device="cuda")
    out = smoke.tmp / "model.engine"
    net.export_tensorrt(out, precision="fp16", verify=True)
    if not out.is_file():
        raise RuntimeError("TensorRT engine missing")
    return f"{out.name} {out.stat().st_size // (1024 * 1024)} MiB, engine parity passed"


def _serving_check(smoke: Smoke) -> str:
    try:
        from fastapi.testclient import TestClient
    except ImportError as error:
        raise SkipCheck("serving extra (fastapi/httpx) not installed") from error
    from autovision.serving import create_app

    net = autovision.load_task_model(smoke.context["exported"], device="cpu")
    app = create_app(net, threshold=0.0, max_batch=4)
    client = TestClient(app)
    image = next((smoke.context["run"].parent / "coco" / "images").glob("*.jpg"))
    payload = {"image": _b64_image(image)}

    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["count"] == 1
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and "autovision_requests_total" in metrics.text
    assert client.get("/docs").status_code == 404

    auth_client = TestClient(
        create_app(net, threshold=0.0, api_key="smoke-key")
    )
    assert auth_client.post("/predict", json=payload).status_code == 401
    assert (
        auth_client.post(
            "/predict", json=payload, headers={"X-API-Key": "smoke-key"}
        ).status_code
        == 200
    )
    assert auth_client.get("/metrics").status_code == 401
    return "health/ready/predict/metrics/auth/docs-off"


def _registry_check(smoke: Smoke) -> str:
    from autovision.encoders.weight_verification import compute_file_sha256
    from autovision.export.postprocess import write_postprocess_spec
    from autovision.registry import ModelRegistry
    from autovision.registry.model_card import LatencyRecord, ModelCard

    exported = smoke.context["exported"]
    net = autovision.load_task_model(exported, device="cpu")
    write_postprocess_spec(net, exported)
    card = ModelCard(
        model="rtdetr/resnet18",
        name="AutoVision smoke detector",
        train_dataset="synthetic smoke COCO",
        schedule="2-step smoke",
        metrics={"coco_map": 0.01},
        latency=[
            LatencyRecord(
                backend="torch",
                precision="fp32",
                hardware="smoke host",
                batch_size=1,
                p50_ms=1.0,
                script="scripts/smoke_enterprise.py",
            )
        ],
        license="Apache-2.0",
        weights_sha256={exported.name: compute_file_sha256(exported)},
    )
    card.save(exported)
    # ParityGate requires the report artifact (a metadata flag is not evidence).
    exported.with_name(exported.name + ".parity.json").write_text(
        '{"passed": true, "source": "smoke_enterprise"}'
    )

    registry = ModelRegistry(smoke.tmp / "registry")
    registry.register(
        exported,
        name="smoke-det",
        version="1.0.0",
        stage="dev",
    )
    staged = registry.promote(
        name="smoke-det", version="1.0.0", to_stage="staging"
    )
    prod = registry.promote(name="smoke-det", version="1.0.0", to_stage="prod")
    assert staged.sha256 == prod.sha256
    assert registry.current(name="smoke-det", stage="prod").version == "1.0.0"
    assert registry.audit_log()
    return "register -> staging -> prod (default gates), current + audit"


def _compliance_check(smoke: Smoke) -> str:
    from autovision.compliance.report import build_compliance_report
    from autovision.data.versioning import dataset_snapshot, verify_dataset_snapshot

    train_json, _, images = smoke.context["dataset"]
    spec = {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images)},
    }
    snapshot = smoke.tmp / "data.snap.json"
    dataset_snapshot(spec, snapshot, deep=True)
    assert verify_dataset_snapshot(snapshot)["ok"]

    report = build_compliance_report(
        model=smoke.context["exported"], snapshot=snapshot
    )
    if not report.passes("error"):
        raise RuntimeError(
            "compliance errors: "
            + "; ".join(issue.detail for issue in report.errors)
        )
    report.save(smoke.tmp / "compliance")
    for name in (
        "compliance.json",
        "compliance.md",
        "compliance.sbom.json",
        "compliance.manifest.json",
    ):
        if not (smoke.tmp / "compliance" / name).is_file():
            raise RuntimeError(f"compliance artifact missing: {name}")
    return (
        f"errors={len(report.errors)} warnings={len(report.warnings)}, "
        "SBOM + sha256 manifest"
    )


def _data_check(smoke: Smoke) -> str:
    from autovision.data.versioning import dataset_snapshot, verify_dataset_snapshot

    train_json, _, images = smoke.context["dataset"]
    spec = {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images)},
    }
    snap = smoke.tmp / "data-check.snap.json"
    dataset_snapshot(spec, snap, deep=True)
    result = verify_dataset_snapshot(snap)
    if not result["ok"]:
        raise RuntimeError(f"dataset snapshot drift: {result['differences'][:3]}")
    return f"snapshot files={result.get('checked', '?')}, deep verify OK"


def _k8s_check(smoke: Smoke) -> str:
    try:
        import yaml
    except ImportError as error:
        raise SkipCheck("PyYAML not installed") from error
    root = ROOT / "deploy" / "k8s"
    kinds = set()
    for path in root.glob("*.yaml"):
        if path.name == "kustomization.yaml":
            continue
        for doc in yaml.safe_load_all(path.read_text()):
            if doc:
                kinds.add(doc.get("kind"))
    required = {
        "Deployment",
        "Service",
        "HorizontalPodAutoscaler",
        "PodDisruptionBudget",
        "PersistentVolumeClaim",
    }
    missing = required - kinds
    if missing:
        raise RuntimeError(f"K8s manifests missing kinds: {sorted(missing)}")
    return "Deployment/Service/HPA/PDB/PVC parse"


def _pytest_command(smoke: Smoke, node: str, extra_env: dict[str, str]) -> str:
    env = dict(os.environ)
    env.update(extra_env)
    env.setdefault("AUTOVISION_TELEMETRY", "0")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-q", "--maxfail=1"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-1200:])
    tail = (proc.stdout or "").strip().splitlines()
    return tail[-1] if tail else "passed"


def _ddp_check(smoke: Smoke) -> str:
    return _pytest_command(
        smoke,
        "tests/test_torchrun_distributed.py::TestTaskTorchrunDistributed",
        {"AUTOVISION_DDP_TESTS": "1"},
    )


def _fsdp_check(smoke: Smoke) -> str:
    if not torch.cuda.is_available():
        raise SkipCheck("FSDP smoke needs CUDA")
    return _pytest_command(
        smoke,
        "tests/test_torchrun_distributed.py::TestFsdpTorchrunDistributed",
        {
            "AUTOVISION_DDP_TESTS": "1",
            "NCCL_MULTI_RANK_GPU_ENABLE": "1",
        },
    )


def _quality_check(smoke: Smoke) -> str:
    commands = []
    if shutil.which("ruff") or Path(sys.executable).with_name("ruff").is_file():
        commands.append([str(Path(sys.executable).with_name("ruff")), "check", "src", "tests", "scripts"])
    if not commands:
        raise SkipCheck("ruff is not installed in this environment")
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout)[-800:])
    return "ruff clean"


def _assess(smoke: Smoke) -> dict[str, str]:
    by_name = {result.name: result for result in smoke.results}
    env = smoke.context.get("env", {})
    optional = env.get("optional", {})

    def status(*names: str) -> str:
        missing = [name for name in names if name not in by_name]
        if len(missing) == len(names):
            return "NOT RUN"
        values = [by_name[name].status for name in names if name in by_name]
        if "FAIL" in values:
            return "FAIL"
        if missing or not all(value == "PASS" for value in values):
            return "PARTIAL"
        return "PASS"

    return {
        "功能完整性": status(
            "train_detection", "export_onnx", "serving", "registry", "compliance"
        ),
        "正确性/可复现": status("train_detection", "export_onnx", "serving"),
        "分布式与规模": status("ddp", "fsdp"),
        "安全": status("serving", "registry"),
        "合规与供应链": status("registry", "compliance"),
        "可观测性": status("train_detection", "serving"),
        "部署后端": status("export_onnx", "openvino", "tensorrt", "k8s_manifests"),
        "工程质量": status("cli_contract", "quality"),
        "MLflow 后端": "NOT INSTALLED" if not optional.get("mlflow") else "NOT RUN",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("quick", "full"), default="quick"
    )
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary work directory"
    )
    args = parser.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="autovision-smoke-"))
    smoke = Smoke(args.profile, tmp)
    print(f"AutoVision enterprise smoke | profile={args.profile} | {tmp}")

    smoke.run("environment", lambda: _env_check(smoke))
    smoke.run("cli_contract", lambda: _cli_check(smoke))
    smoke.run("train_detection", lambda: _train_check(smoke))
    smoke.run("export_onnx", lambda: _onnx_check(smoke))
    smoke.run("serving", lambda: _serving_check(smoke))
    smoke.run("registry", lambda: _registry_check(smoke))
    smoke.run("compliance", lambda: _compliance_check(smoke))
    smoke.run("data_snapshot", lambda: _data_check(smoke))
    smoke.run("k8s_manifests", lambda: _k8s_check(smoke))
    smoke.run("quality", lambda: _quality_check(smoke), required=False)
    if args.profile == "full":
        smoke.run("openvino", lambda: _openvino_check(smoke), required=False)
        smoke.run("tensorrt", lambda: _tensorrt_check(smoke), required=False)
        smoke.run("ddp", lambda: _ddp_check(smoke), required=False)
        smoke.run("fsdp", lambda: _fsdp_check(smoke), required=False)

    assessment = _assess(smoke)
    print("\nEnterprise dimension assessment (smoke scope):")
    for dimension, verdict in assessment.items():
        print(f"  {dimension:16s} {verdict}")

    failures = [result for result in smoke.results if result.status == "FAIL"]
    payload = {
        "profile": args.profile,
        "tmp": str(tmp),
        "environment": smoke.context.get("env", {}),
        "results": [result.__dict__ for result in smoke.results],
        "assessment": assessment,
        "passed": not failures,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"JSON report: {args.json}")

    print(
        "\nSmoke scope note: this proves the functional paths on this host. "
        "It does not prove production load/soak, multi-node NCCL, or "
        "DeepSpeed/ZeRO/elastic training; see docs/smoke_test.md."
    )
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
