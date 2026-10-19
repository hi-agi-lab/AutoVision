# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Command-line interface (typer thin shell over the public API).

Commands share the exact validation of the Python API: keyword-style
``key=value`` arguments are parsed by the API functions themselves via the
config validators.

    autovision-cli pretrain out=runs/e1 data=images/ model=torchvision/resnet18 epochs=2
    autovision-cli train-object-detection out=runs/d1 data=data.yaml model_args='{"image_size":[64,64]}'
    autovision-cli export model=runs/d1/exported_models/exported_best.pt out=model.onnx precision=fp16
    autovision-cli embed out=embeddings.pt checkpoint=runs/e1/exported_models/exported_last.pt
    autovision-cli list-models | list-methods | extract-video-frames ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(
    name="autovision-cli",
    help="AutoVision: enterprise visual model training framework.",
    no_args_is_help=True,
    add_completion=False,
)

_KEY_VALUE_ERROR = (
    "Arguments must be key=value pairs, e.g. out=runs/e1 model=torchvision/resnet18"
)


def _parse_kwargs(arguments: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` CLI arguments into API kwargs.

    Values are JSON-decoded when possible (numbers, booleans, lists, nested
    dicts); otherwise kept as strings (paths stay strings — the configs
    coerce them).
    """
    kwargs: dict[str, Any] = {}
    for argument in arguments:
        if "=" not in argument:
            raise typer.BadParameter(_KEY_VALUE_ERROR)
        key, _, value = argument.partition("=")
        # Conventional CLI spelling: --some-flag / some-flag=... maps to the
        # Python parameter some_flag.
        key = key.lstrip("-").replace("-", "_")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


def _run(fn, arguments: list[str]) -> None:
    kwargs = _parse_kwargs(arguments)
    fn(**kwargs)


def _reject_unknown_keys(
    leftovers: dict[str, Any], command: str, accepted: tuple[str, ...]
) -> None:
    """Typo guard: silently dropped kwargs hide mistakes (``deeep=true``)."""
    if leftovers:
        raise typer.BadParameter(
            f"Unknown {command} key(s) {sorted(leftovers)} — accepted: {sorted(accepted)}."
        )


@app.command("pretrain")
def pretrain_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Self-supervised pretraining / distillation."""
    from autovision import pretrain

    _run(pretrain, list(arguments or []))


@app.command("train-object-detection")
def train_object_detection_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Object detection finetuning (COCO/YOLO)."""
    from autovision import train_object_detection

    _run(train_object_detection, list(arguments or []))


@app.command("train-oriented-object-detection")
def train_oriented_object_detection_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Oriented (rotated-box) detection finetuning (YOLO-OBB)."""
    from autovision import train_oriented_object_detection

    _run(train_oriented_object_detection, list(arguments or []))


@app.command("train-instance-segmentation")
def train_instance_segmentation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Instance segmentation finetuning (EOMT, COCO instance json)."""
    from autovision import train_instance_segmentation

    _run(train_instance_segmentation, list(arguments or []))


@app.command("train-panoptic-segmentation")
def train_panoptic_segmentation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Panoptic segmentation finetuning (EOMT, mask PNGs + segments json)."""
    from autovision import train_panoptic_segmentation

    _run(train_panoptic_segmentation, list(arguments or []))


@app.command("train-image-classification")
def train_image_classification_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Image classification finetuning (ImageFolder layout)."""
    from autovision import train_image_classification

    _run(train_image_classification, list(arguments or []))


@app.command("train-depth-estimation")
def train_depth_estimation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Metric depth estimation finetuning (GTR depth family)."""
    from autovision import train_depth_estimation

    _run(train_depth_estimation, list(arguments or []))


@app.command("train-pose-estimation")
def train_pose_estimation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Person keypoint finetuning (GTR pose family, COCO person_keypoints)."""
    from autovision import train_pose_estimation

    _run(train_pose_estimation, list(arguments or []))


@app.command("train-semantic-segmentation")
def train_semantic_segmentation_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Semantic segmentation finetuning (linear head, image/mask dirs)."""
    from autovision import train_semantic_segmentation

    _run(train_semantic_segmentation, list(arguments or []))


@app.command("embed")
def embed_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Extract embeddings from a pretrained checkpoint."""
    from autovision import embed

    _run(embed, list(arguments or []))


@app.command("benchmark")
def benchmark_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Benchmark inference latency across torch/ONNX/TensorRT/OpenVINO backends.

        autovision-cli benchmark model=rtdetr/resnet18 out=bench/ backends='{"torch": {}, "onnx": {"provider": "cpu"}}'
        autovision-cli benchmark model=runs/d1/exported_models/exported_best.pt out=bench/ \
            backends='{"torch": {}, "openvino": {"ov_device": "CPU"}}' verify_parity=true
    """
    from autovision import benchmark

    _run(benchmark, list(arguments or []))


@app.command("extract-video-frames")
def extract_video_frames_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Extract JPEG frames from videos with ffmpeg."""
    from autovision.cli.extract_video_frames import extract_video_frames

    _run(extract_video_frames, list(arguments or []))


@app.command("list-models")
def list_models_command(
    task: str | None = typer.Option(
        None, "--task", help="Filter the task catalog (e.g. detection)."
    ),
    pretrained: bool = typer.Option(
        False, "--pretrained", help="Only models with full task weights."
    ),
    planned: bool = typer.Option(
        False, "--planned", help="Include planned catalog entries."
    ),
    format: str = typer.Option(
        "names", "--format", help="names | table | json (catalog views)."
    ),
) -> None:
    """Print available models.

    Without catalog flags this keeps the historical encoder-name listing.
    With ``--task`` / ``--pretrained`` / ``--format`` it prints the curated
    task catalog (roadmap phase 1).
    """
    if task is None and not pretrained and format == "names":
        from autovision import list_models

        for name in list_models():
            typer.echo(name)
        return

    from autovision.registry.catalog import (
        format_model_table,
        list_task_models,
    )

    entries = list_task_models(
        task=task, pretrained_only=pretrained, include_planned=planned
    )
    if format == "names":
        for entry in entries:
            typer.echo(entry.key)
    elif format == "json":
        typer.echo(
            json.dumps([entry.to_dict() for entry in entries], indent=2)
        )
    elif format == "table":
        typer.echo(format_model_table(entries))
    else:
        raise typer.BadParameter(
            "--format must be one of names/table/json (catalog views)."
        )


@app.command("model-info")
def model_info_command(
    key: str = typer.Argument(..., help="Task model key, e.g. rfdetr/rf-detr-small.")
) -> None:
    """Print license, weights, metrics and export surface for one model."""
    from autovision.registry.catalog import describe_model

    try:
        typer.echo(describe_model(key))
    except KeyError as error:
        raise typer.BadParameter(str(error)) from error


@app.command("data-check")
def data_check_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Audit a detection dataset (images / boxes / duplicates / leakage).

        autovision-cli data-check data=coco.yaml out=qa/ [near_duplicate=true]
        autovision-cli data-check data=coco.yaml out=qa/ fail_on=warning
    """
    from autovision.cli.data_check import data_check

    kwargs = _parse_kwargs(list(arguments or []))
    out = kwargs.pop("out", None)
    if out is None:
        raise typer.BadParameter("data-check needs out=<directory>.")
    if "data" not in kwargs:
        raise typer.BadParameter("data-check needs data=<yaml|dict|coco8>.")
    fail_on = str(kwargs.pop("fail_on", "error"))
    if fail_on not in {"error", "warning", "none"}:
        raise typer.BadParameter("fail_on must be error | warning | none.")
    report = data_check(out=out, **kwargs)
    typer.echo(report.summary_text())
    typer.echo(f"report: {report.report_md}")
    if report.has_errors and fail_on in {"error", "warning"}:
        raise typer.Exit(code=1)
    if report.has_warnings and fail_on == "warning":
        raise typer.Exit(code=1)


@app.command("plugins")
def plugins_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Discover and inspect ``autovision.plugins`` entry points.

        autovision-cli plugins action=list
        autovision-cli plugins action=check
    """
    from autovision.plugins import discover_plugins, list_plugins

    kwargs = _parse_kwargs(list(arguments or []))
    action = str(kwargs.pop("action", "list"))
    if kwargs:
        raise typer.BadParameter(
            f"Unknown plugins key(s) {sorted(kwargs)}; accepted: action."
        )
    if action not in {"list", "check"}:
        raise typer.BadParameter("plugins action must be list | check.")
    result = discover_plugins()
    for error in result.errors:
        typer.echo(f"error {error.entry_point}: {error.error}", err=True)
    if action == "check":
        if result.errors:
            raise typer.Exit(code=1)
        typer.echo(f"plugins ok ({len(result.loaded)} loaded)")
        return
    for plugin in list_plugins():
        typer.echo(
            f"{plugin.kind}\t{plugin.name}\t{plugin.version}\t"
            f"api={plugin.api_version}\t{plugin.description}"
        )


@app.command("compliance-report")
def compliance_report_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Enterprise compliance pack (dependency licenses / model / dataset).

        autovision-cli compliance-report model=best.pt snapshot=qa/data.snap.json out=compliance/
        autovision-cli compliance-report model=best.pt out=compliance/ include_dependencies=false
        autovision-cli compliance-report model=best.pt out=compliance/ fail_on=warning
    """
    from autovision.cli.compliance_report import compliance_report

    kwargs = _parse_kwargs(list(arguments or []))
    strict = bool(kwargs.pop("strict", False))
    fail_on = str(kwargs.pop("fail_on", "error" if strict else "none"))
    if fail_on not in ("error", "warning", "none"):
        raise typer.BadParameter(
            "fail_on must be one of error | warning | none."
        )
    report = compliance_report(**kwargs)
    typer.echo(report.summary_text())
    typer.echo(f"json: {report.report_json}")
    typer.echo(f"markdown: {report.report_md}")
    typer.echo(f"sbom: {report.report_sbom}")
    typer.echo(f"manifest: {report.report_manifest}")
    if not report.passes(fail_on):  # type: ignore[arg-type]
        raise typer.Exit(code=1)


@app.command("registry")
def registry_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Private model registry: register / promote / rollback / current / audit.

        autovision-cli registry action=register file=best.pt name=det v=1.0.0
        autovision-cli registry action=promote name=det v=1.0.0 to=staging gates=all
        autovision-cli registry action=rollback name=det stage=prod to=0.9.0 reason="regression"
        autovision-cli registry action=current name=det stage=prod
    """
    import os

    from autovision.registry import ModelRegistry, default_gates

    kwargs = _parse_kwargs(list(arguments or []))
    action = str(kwargs.pop("action", ""))
    root = kwargs.pop("root", os.environ.get("AUTOVISION_REGISTRY_ROOT", "runs/registry"))
    actor = kwargs.pop("actor", None)
    registry = ModelRegistry(root)
    if action == "register":
        file = kwargs.pop("file", None)
        name = kwargs.pop("name", None)
        version = kwargs.pop("v", kwargs.pop("version", None))
        if file is None or name is None or version is None:
            raise typer.BadParameter("register needs file= name= v=")
        entry = registry.register(
            file, name=str(name), version=str(version),
            stage=str(kwargs.pop("stage", "dev")),
            metadata=kwargs.pop("metadata", None),
            actor=actor,
        )
        typer.echo(json.dumps(entry.to_manifest(), indent=2, sort_keys=True))
    elif action == "promote":
        name = kwargs.pop("name", None)
        version = kwargs.pop("v", kwargs.pop("version", None))
        if name is None or version is None:
            raise typer.BadParameter("promote needs name= v=")
        gates_arg = str(kwargs.pop("gates", "all"))
        if gates_arg == "none":
            gates = []
        elif gates_arg == "all":
            gates = default_gates()
        else:
            raise typer.BadParameter("gates must be all | none.")
        entry = registry.promote(
            name=str(name), version=str(version),
            to_stage=str(kwargs.pop("to", "staging")),
            gates=gates, actor=actor, reason=kwargs.pop("reason", None),
        )
        typer.echo(json.dumps(entry.to_manifest(), indent=2, sort_keys=True))
    elif action == "rollback":
        name = kwargs.pop("name", None)
        to_version = kwargs.pop("to", None)
        if name is None or to_version is None:
            raise typer.BadParameter("rollback needs name= to=<version>")
        entry = registry.rollback(
            name=str(name), stage=str(kwargs.pop("stage", "prod")),
            to_version=str(to_version), reason=kwargs.pop("reason", None),
            actor=actor,
        )
        typer.echo(json.dumps(entry.to_manifest(), indent=2, sort_keys=True))
    elif action == "current":
        name = kwargs.pop("name", None)
        if name is None:
            raise typer.BadParameter("current needs name=")
        entry = registry.current(name=str(name), stage=str(kwargs.pop("stage", "prod")))
        typer.echo(json.dumps(entry.to_manifest(), indent=2, sort_keys=True))
    elif action == "list":
        stage = kwargs.pop("stage", None)
        for entry in registry.list_entries(stage=stage):
            typer.echo(
                f"{entry.stage}\t{entry.name}\t{entry.version}\t{entry.sha256[:12]}"
            )
    elif action == "audit":
        for record in registry.audit_log():
            typer.echo(json.dumps(record, sort_keys=True))
    else:
        raise typer.BadParameter(
            "registry action must be register | promote | rollback | current | list | audit."
        )
    if kwargs:
        raise typer.BadParameter(f"Unknown registry key(s): {sorted(kwargs)}")


@app.command("list-methods")
def list_methods_command() -> None:
    """Print all registered pretraining methods."""
    from autovision import list_methods

    for name in list_methods():
        typer.echo(name)


@app.command("val")
def val_command(arguments: list[str] = typer.Argument(None)) -> None:
    """COCO evaluation of an exported detection model.

        autovision-cli val model=runs/d1/exported_models/exported_best.pt data=coco8 [out=metrics.json]
    """
    from autovision.cli.model_handle import evaluate_object_detection

    kwargs = _parse_kwargs(list(arguments or []))
    out = kwargs.pop("out", None)
    metrics = evaluate_object_detection(**kwargs)
    if out is not None:
        Path(str(out)).write_text(json.dumps(metrics, indent=2))
    for key, value in sorted(metrics.items()):
        typer.echo(f"{key}\t{value:.4f}" if isinstance(value, float) else f"{key}\t{value}")


@app.command("predict")
def predict_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Run prediction over images, a video file, or an RTSP/HTTP stream.

        autovision-cli predict model=model.pt images=dir/ out=preds/ threshold=0.5
        autovision-cli predict model=model.pt source=clip.mp4 out=preds/ batch_size=4
        autovision-cli predict model=model.pt source=rtsp://host/live out=preds/ backend=onnx artifact=model.onnx
        autovision-cli predict model=model.pt sources=rtsp://a,rtsp://b out=preds/ batch_size=4
    """
    kwargs = _parse_kwargs(list(arguments or []))
    if "out" in kwargs:
        kwargs["out_dir"] = kwargs.pop("out")
    if "sources" in kwargs:
        if "out_dir" not in kwargs:
            raise typer.BadParameter("sources= requires out=<directory>")
        from autovision.inference import run_multi_stream

        typer.echo(run_multi_stream(**kwargs).summary())
        return
    if "source" in kwargs:
        if "out_dir" not in kwargs:
            raise typer.BadParameter("source= requires out=<directory>")
        from autovision.inference import run_stream

        typer.echo(run_stream(**kwargs).summary())
        return
    from autovision.cli.model_handle import predict_to_files

    typer.echo(str(predict_to_files(**kwargs)))


@app.command("pseudo-label")
def pseudo_label_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Pseudo-label an unlabeled image pool with a (teacher) detection model.

        autovision-cli pseudo-label model=teacher.pt images=pool/ out=pseudo.json \
            data=coco.yaml threshold=0.5 [sahi=true]
    """
    from autovision.cli.pseudo_label import pseudo_label

    result = pseudo_label(**_parse_kwargs(list(arguments or [])))
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("analyze-errors")
def analyze_errors_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Detection error report + top-FP/FN gallery.

        autovision-cli analyze-errors model=best.pt data=coco.yaml out=report/ iou=0.5
        autovision-cli analyze-errors predictions=preds.json annotations=val.json out=report/
    """
    from autovision.cli.analyze_errors import analyze_errors

    report = analyze_errors(**_parse_kwargs(list(arguments or [])))
    typer.echo(json.dumps(report.summary["totals"], indent=2))
    typer.echo(f"report: {report.report_md}")
    typer.echo(f"gallery: {len(report.gallery)} images under {report.out_dir / 'gallery'}")
    if report.hard_examples_json is not None:
        typer.echo(f"hard examples: {report.hard_examples_json}")


@app.command("export")
def export_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Export an exported ``.pt`` to ONNX or TensorRT.

        autovision-cli export model=runs/d1/exported_models/exported_best.pt out=model.onnx precision=fp16
        autovision-cli export model=exported_best.pt out=model.engine precision=fp16 max_batch_size=4
    """
    from autovision.cli.export import export_model

    typer.echo(str(export_model(**_parse_kwargs(list(arguments or [])))))


@app.command("download")
def download_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Download + extract a registered dataset.

        autovision-cli download dataset=coco2017-val out=datasets/
    """
    from autovision.data.downloads import download_dataset

    kwargs = _parse_kwargs(list(arguments or []))
    dataset = kwargs.pop("dataset", None)
    if dataset is None:
        raise typer.BadParameter("dataset=<name> is required")
    typer.echo(str(download_dataset(str(dataset), Path(kwargs.get("out", "datasets")))))


def _load_names(value: Any, source: str) -> Any:
    """""names=" accepts a yaml/json file path or an inline list."""
    if value is None:
        raise typer.BadParameter(f"from={source} requires names=<yaml|json list|inline list>")
    if isinstance(value, str) and not Path(value).exists() and "," not in value:
        raise typer.BadParameter(f"names file not found: '{value}'")
    if isinstance(value, str):
        if Path(value).exists():
            loaded = yaml.safe_load(Path(value).read_text())
            return loaded.get("names", loaded) if isinstance(loaded, dict) else loaded
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


@app.command("serve")
def serve_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Reference inference serving: FastAPI + batch + metrics.

        autovision-cli serve model=exported_best.pt port=8000 [backend=onnx artifact=m.onnx]
        autovision-cli serve model=exported_best.pt port=8443 workers=2 \
            ssl_keyfile=key.pem ssl_certfile=cert.pem
        autovision-cli serve model=exported_best.pt rate_limit_per_minute=600 \
            rate_limit_burst=20

    Endpoints: ``POST /predict`` ({"images": ["<base64>", ...]}),
    ``GET /healthz``, ``GET /readyz``, ``GET /drift``, ``GET /metrics``.
    Needs the ``serving`` extra (fastapi/uvicorn). TLS and multi-worker are
    transport options; canary/A-B variants are available through the Python
    API (``create_app(variants=...)``).
    """

    try:
        import uvicorn  # noqa: F401 - availability check for a friendly error
    except ImportError:
        raise typer.BadParameter(
            "serve needs the 'serving' extra: pip install 'autovision[serving]'."
        ) from None

    from autovision.errors import AutoVisionError

    try:
        from autovision.serving.server import serve
    except ImportError:
        raise typer.BadParameter(
            "serve needs the 'serving' extra: pip install 'autovision[serving]'."
        ) from None

    kwargs = _parse_kwargs(list(arguments or []))
    host = str(kwargs.pop("host", "127.0.0.1"))
    port = int(kwargs.pop("port", 8000))
    workers = int(kwargs.pop("workers", 1))
    ssl_keyfile = kwargs.pop("ssl_keyfile", None)
    ssl_certfile = kwargs.pop("ssl_certfile", None)
    ssl_ca_certs = kwargs.pop("ssl_ca_certs", None)
    ssl_cert_reqs = kwargs.pop("ssl_cert_reqs", None)
    if ssl_cert_reqs is not None:
        try:
            ssl_cert_reqs = int(ssl_cert_reqs)
        except (TypeError, ValueError):
            raise typer.BadParameter(
                "ssl_cert_reqs must be an integer "
                "(e.g. 2 for ssl.CERT_REQUIRED)."
            ) from None
    model = kwargs.pop("model", None)
    if model is None:
        raise typer.BadParameter("serve needs model=<exported .pt> (or engine kwargs).")
    scheme = "https" if ssl_keyfile or ssl_certfile else "http"
    typer.echo(
        f"serving on {scheme}://{host}:{port} with workers={workers} "
        "(endpoints: /predict /healthz /readyz /drift /metrics)"
    )
    try:
        serve(
            model,
            host=host,
            port=port,
            workers=workers,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs,
            ssl_cert_reqs=ssl_cert_reqs,
            **kwargs,
        )
    except AutoVisionError as error:
        raise typer.BadParameter(str(error)) from None


@app.command("data-snapshot")
def data_snapshot_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Content-addressed dataset snapshot / verification (数据版本化).

        autovision-cli data-snapshot data=coco8 out=snap.json [deep=true]
        autovision-cli data-snapshot data=spec.yaml out=snap.json overwrite=true
        autovision-cli data-snapshot snapshot=snap.json [deep=true] [spec=other.yaml]

    Without ``snapshot=`` a snapshot is written (an existing ``out`` needs
    ``overwrite=true``); with it the snapshot is RE-VERIFIED against the
    current tree (non-zero exit on drift). ``deep`` additionally hashes file
    contents (versioning-grade pins; on verification it must match the
    snapshot's own mode). ``spec=`` verifies a moved/copied tree against its
    own data spec.
    """
    from autovision.data.versioning import dataset_snapshot, verify_dataset_snapshot

    kwargs = _parse_kwargs(list(arguments or []))
    snapshot = kwargs.pop("snapshot", None)
    deep = kwargs.pop("deep", None)
    if snapshot is not None:
        spec_override = kwargs.pop("spec", None)
        _reject_unknown_keys(kwargs, "data-snapshot", ("deep", "spec", "snapshot"))
        report = verify_dataset_snapshot(
            snapshot, deep=deep, spec_override=spec_override
        )
        typer.echo(json.dumps(report, indent=2))
        if not report["ok"]:
            raise typer.Exit(code=1)
        return
    if "data" not in kwargs:
        raise typer.BadParameter("data-snapshot needs data=<spec> (or snapshot=<json> to verify).")
    if "out" not in kwargs:
        raise typer.BadParameter("data-snapshot needs out=<snapshot.json>.")
    data = kwargs.pop("data")
    out = kwargs.pop("out")
    note = kwargs.pop("note", None)
    overwrite = kwargs.pop("overwrite", False)
    _reject_unknown_keys(kwargs, "data-snapshot", ("data", "deep", "note", "out", "overwrite"))
    payload = dataset_snapshot(
        data,
        out,
        deep=deep if deep is not None else False,
        note=note,
        overwrite=overwrite,
    )
    typer.echo(
        f"snapshot: {len(payload['splits'])} split(s), deep={payload['deep']}"
    )


@app.command("hpo")
def hpo_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Hyperparameter search (random / grid / ASHA) with a report.

        autovision-cli hpo space=search.yaml out=runs/hpo task=object_detection \
            data=coco8 model=rtdetr/resnet18 trials=8 budget=500 strategy=random \
            objective=val_metric/map_50 seed=0
        autovision-cli hpo space=search.yaml out=runs/hpo task=object_detection \
            data=coco8 model=rtdetr/resnet18 strategy=asha trials=9 budget=200 \
            asha_max_budget=1800

    ``task`` names a training entry (``object_detection`` ->
    ``autovision-cli train-object-detection``'s kwargs, passed through
    unchanged); ``space`` is a yaml/json mapping of dotted override keys to
    distribution specs (see autovision.hpo.space). Artifacts:
    ``out/report.md``, ``out/trials.jsonl``, ``out/best.json``.
    """
    from autovision.hpo import load_space, resolve_task_entry, run_search, task_trial

    kwargs = _parse_kwargs(list(arguments or []))
    if "space" not in kwargs:
        raise typer.BadParameter(
            "hpo needs space=<search.yaml>; e.g. autovision-cli hpo "
            "space=search.yaml out=runs/hpo task=object_detection data=coco8 "
            "model=rtdetr/resnet18 trials=8 budget=300"
        )
    space = load_space(kwargs.pop("space"))
    out = kwargs.pop("out", "runs/hpo")
    task = str(kwargs.pop("task", "object_detection"))
    # Fail on a mistyped task BEFORE any trial runs (the entry resolution
    # inside task_trial would otherwise be swallowed as a failed trial).
    # Surface it as a usage error, not a traceback.
    from autovision.errors import AutoVisionError

    try:
        resolve_task_entry(task)
    except AutoVisionError as error:
        raise typer.BadParameter(str(error)) from None
    strategy = str(kwargs.pop("strategy", "random"))

    def _as_int(key: str, default: int) -> int:
        raw = kwargs.pop(key, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise typer.BadParameter(f"{key}= must be an integer, got {raw!r}") from None

    trials = _as_int("trials", 10)
    budget = _as_int("budget", 1000)
    objective = str(kwargs.pop("objective", "val_metric/map"))
    maximize = bool(kwargs.pop("maximize", True))
    seed = _as_int("seed", 0)
    force = bool(kwargs.pop("force", False))
    asha_kwargs = {
        key: kwargs.pop(key)
        for key in ("asha_min_budget", "asha_max_budget", "asha_eta")
        if key in kwargs
    }
    reserved = sorted(set(kwargs) & {"out", "steps", "checkpoint", "resume_interrupted"})
    if reserved:
        raise typer.BadParameter(
            f"{reserved} are owned by the search (a trial's steps/out are set per "
            "trial); remove them from the command line."
        )
    result = run_search(
        task_trial(task, kwargs),
        space,
        out,
        strategy=strategy,
        trials=trials,
        budget=budget,
        objective=objective,
        maximize=maximize,
        seed=seed,
        force=force,
        **asha_kwargs,
    )
    best = result.best
    if best is not None:
        typer.echo(
            f"best trial #{best.index}: {objective} = "
            f"{best.objective(objective):.6g} | params: {best.params}"
        )
    else:
        typer.echo("no successful trial produced the objective")
    typer.echo(f"report: {result.report_md}")


@app.command("convert-data")
def convert_data_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Convert annotations between formats (detection / keypoints / segmentation).

        autovision-cli convert-data from=yolo images=img/ names=a.yaml out=train.json
        autovision-cli convert-data from=coco annotations=train.json out-labels=labels/
        autovision-cli convert-data from=voc voc-root=VOC2007/ out=voc_train.json imagesets=train
        autovision-cli convert-data from=yolo-pose images=img/ names=person.yaml out=pose.json
        autovision-cli convert-data from=yolo-seg images=img/ names=a.yaml out=seg.json
        autovision-cli convert-data from=cvat cvat-xml=annotations.xml images=img/ out=train.json
        autovision-cli convert-data from=label-studio ls-json=export.json images=img/ out=train.json
        autovision-cli convert-data from=merge parts=a.json,b.json out=merged.json
    """
    from autovision.data.convert import (
        coco_to_yolo,
        merge_coco,
        voc_to_coco,
        yolo_pose_to_coco,
        yolo_seg_to_coco,
        yolo_to_coco,
    )
    from autovision.data.interchange import cvat_to_coco, label_studio_to_coco

    kwargs = _parse_kwargs(list(arguments or []))
    source = str(kwargs.pop("from", ""))
    # The converters name their output explicitly (out_json /
    # out_labels_dir); accept the documented short spellings too.
    if "images" in kwargs:
        kwargs["images_dir"] = kwargs.pop("images")
    if source == "coco" and "annotations" in kwargs:
        kwargs["annotations_json"] = kwargs.pop("annotations")
    if "out" in kwargs:
        kwargs["out_json" if source != "coco" else "out_labels_dir"] = kwargs.pop("out")
    if "out_labels" in kwargs:
        kwargs["out_labels_dir"] = kwargs.pop("out_labels")
    if source == "yolo":
        typer.echo(
            str(yolo_to_coco(names=_load_names(kwargs.pop("names", None), source), **kwargs))
        )
    elif source == "coco":
        typer.echo(str(coco_to_yolo(**kwargs)))
    elif source == "voc":
        typer.echo(str(voc_to_coco(**kwargs)))
    elif source == "yolo-pose":
        typer.echo(
            str(
                yolo_pose_to_coco(
                    names=_load_names(kwargs.pop("names", None), source), **kwargs
                )
            )
        )
    elif source == "yolo-seg":
        typer.echo(
            str(
                yolo_seg_to_coco(
                    names=_load_names(kwargs.pop("names", None), source), **kwargs
                )
            )
        )
    elif source == "merge":
        parts = kwargs.pop("parts", None)
        if not parts:
            raise typer.BadParameter("from=merge requires parts=a.json,b.json")
        if isinstance(parts, str):
            parts = [part for part in parts.split(",") if part]
        typer.echo(str(merge_coco(parts, **kwargs)))
    elif source == "cvat":
        xml_path = kwargs.pop("cvat_xml", None) or kwargs.pop("xml", None)
        if xml_path is None:
            raise typer.BadParameter("from=cvat requires cvat-xml=<annotations.xml>")
        typer.echo(str(cvat_to_coco(xml_path=xml_path, **kwargs)))
    elif source == "label-studio":
        json_path = kwargs.pop("ls_json", None) or kwargs.pop("json", None)
        if json_path is None:
            raise typer.BadParameter("from=label-studio requires ls-json=<export.json>")
        typer.echo(str(label_studio_to_coco(json_path=json_path, **kwargs)))
    else:
        raise typer.BadParameter(
            "from= must be one of 'yolo', 'coco', 'voc', 'yolo-pose', "
            "'yolo-seg', 'cvat', 'label-studio', 'merge'"
        )


@app.command("plot-curves")
def plot_curves_command(arguments: list[str] = typer.Argument(None)) -> None:
    """Plot training curves from a run's metrics.jsonl (needs matplotlib).

        autovision-cli plot-curves run=runs/d1
    """
    from autovision.cli.plot import plot_training_curves

    kwargs = _parse_kwargs(list(arguments or []))
    run = Path(str(kwargs.pop("run")))
    metrics = run / "metrics.jsonl" if run.is_dir() else run
    typer.echo(str(plot_training_curves(metrics, out_path=kwargs.get("out"))))


def cli_entrypoint() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    cli_entrypoint()
