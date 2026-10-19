# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests: Ascend NPU accelerator adaptation + optional OTel spans.

Two families (design §12.4, §15.2):

- ``TestOtel``: ``AUTOVISION_OTEL`` gate parsing, no-op degradation without
  the opentelemetry API, stub-tracer injection, and end-to-end span
  sequences for both training entry points (four-phase instrumentation).
- ``TestAccelerator``: device resolution / DDP backend / bf16 capability /
  mixed-precision guards under mocked CUDA/NPU availability. All NPU paths
  are exercised via mocks (real-hardware validation is an M4+ item).
"""

import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import autovision.engine.engine as engine_module
import autovision.runtime.accelerator as accelerator
from autovision import pretrain, train_object_detection
from autovision.engine.engine import resolve_device
from autovision.observability import otel as otel_module
from autovision.observability.otel import (
    TRACER_NAME,
    end_span,
    get_tracer,
    is_otel_enabled,
    otel_span,
    start_span,
)
from autovision.runtime.accelerator import (
    accelerator_device_count,
    accelerator_device_name,
    accelerator_supports_bf16,
    ddp_backend_for,
    import_torch_npu,
    npu_available,
    supports_amp,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubSpan:
    def __init__(self, records, name, attributes):
        self.name = name
        self.attributes = attributes or {}
        self.ended = False
        records.append(self)

    def end(self) -> None:
        self.ended = True


class _StubSpanContext:
    def __init__(self, records, name, attributes):
        self.span = StubSpan(records, name, attributes)

    def __enter__(self) -> StubSpan:
        return self.span

    def __exit__(self, *exc_info) -> bool:
        self.span.end()
        return False


class StubTracer:
    def __init__(self, records, name):
        self.records = records
        self.name = name

    def start_as_current_span(self, name, attributes=None):
        return _StubSpanContext(self.records, name, attributes)

    def start_span(self, name, attributes=None):
        return StubSpan(self.records, name, attributes)


class StubTraceModule:
    def __init__(self, records):
        self.records = records

    def get_tracer(self, name):
        return StubTracer(self.records, name)


def install_stub_tracer(monkeypatch) -> list[StubSpan]:
    """Force the OTel gate on and route the API import to a stub recorder."""
    records: list[StubSpan] = []
    monkeypatch.setenv("AUTOVISION_OTEL", "1")
    monkeypatch.setattr(
        otel_module, "_get_opentelemetry_trace", lambda: StubTraceModule(records)
    )
    return records


def _reset_npu_cache(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "_TORCH_NPU", None)
    monkeypatch.setattr(accelerator, "_NPU_IMPORT_WARNED", False)


def _fake_torch_npu(
    monkeypatch, *, available=True, device_count=2, bf16=True
) -> None:
    """Register a fake ``torch.npu`` namespace (as torch_npu would)."""
    namespace = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
        is_bf16_supported=lambda: bf16,
        get_device_name=lambda index: f"Ascend910B{index}",
    )
    monkeypatch.setattr(torch, "npu", namespace, raising=False)


def _fake_torch_device(monkeypatch) -> None:
    """Stand-in for ``torch.device`` accepting unregistered type names.

    Stock torch only parses device types a backend registered (torch_npu
    registers ``"npu"``); the resolve_device NPU branches are exercised
    without the real backend by stubbing the constructor. Scoped to the
    test via monkeypatch — no process-global side effects.
    """

    def fake_device(spec, index=None):
        text = f"{spec}:{index}" if index is not None else str(spec)
        device_type, _, device_index = text.partition(":")
        return SimpleNamespace(
            type=device_type, index=int(device_index) if device_index else None
        )

    monkeypatch.setattr(torch, "device", fake_device)


# ---------------------------------------------------------------------------
# OTel gate + spans
# ---------------------------------------------------------------------------


class TestOtel:
    def test_disabled_by_default_noop(self, monkeypatch):
        monkeypatch.delenv("AUTOVISION_OTEL", raising=False)
        assert is_otel_enabled() is False
        assert get_tracer() is None
        # Context-manager form degrades to nullcontext; explicit form to None.
        with otel_span("engine.step") as span:
            assert isinstance(span, type(nullcontext().__enter__()))
        assert start_span("pretrain.fit") is None
        end_span(None)  # None-safe by contract
        ended = StubSpan([], "x", None)
        end_span(ended)
        assert ended.ended

    @pytest.mark.parametrize(
        "value", ["1", "true", "TRUE", "Yes", "on", "  On  "]
    )
    def test_env_true_values(self, monkeypatch, value):
        monkeypatch.setenv("AUTOVISION_OTEL", value)
        assert is_otel_enabled() is True

    @pytest.mark.parametrize(
        "value", ["0", "", "no", "off", "garbage", "  "]
    )
    def test_env_false_values(self, monkeypatch, value):
        monkeypatch.setenv("AUTOVISION_OTEL", value)
        assert is_otel_enabled() is False

    def test_missing_api_warns_once(self, monkeypatch, caplog):
        try:
            import opentelemetry.trace  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("opentelemetry installed; stub path not applicable")
        monkeypatch.setenv("AUTOVISION_OTEL", "1")
        monkeypatch.setattr(otel_module, "_otel_disabled_logged", False)
        with caplog.at_level("WARNING", logger="autovision"):
            assert get_tracer() is None
            assert get_tracer() is None
        warnings = [
            r for r in caplog.records if "opentelemetry is not installed" in r.message
        ]
        assert len(warnings) == 1  # warn once, not per call

    def test_stub_tracer_unit(self, monkeypatch):
        records = install_stub_tracer(monkeypatch)
        tracer = get_tracer()
        assert isinstance(tracer, StubTracer)
        assert tracer.name == TRACER_NAME
        with otel_span("engine.dataload", {"step": 3}) as span:
            assert span.name == "engine.dataload"
            assert span.attributes == {"step": 3}
            assert not span.ended
        assert span.ended  # closed on context exit
        explicit = start_span("task.fit", {"steps": 4})
        end_span(explicit)
        assert explicit.ended
        assert [s.name for s in records] == [
            "engine.dataload",
            "task.fit",
        ]

    # -- End-to-end span sequences -----------------------------------------

    def test_pretrain_e2e_span_sequence(self, tmp_path, monkeypatch):
        from tests.conftest import create_images

        records = install_stub_tracer(monkeypatch)
        image_dir = tmp_path / "images"
        create_images(image_dir, num_images=8, size=(48, 48))
        pretrain(
            out=tmp_path / "otel_pretrain",
            data=image_dir,
            model="torchvision/resnet18",
            epochs=1,
            batch_size=4,
            num_workers=0,
            method_args={"queue_size": 4},
        )
        names = [s.name for s in records]
        # 8 images / batch 4 -> 2 steps per epoch, 1 epoch.
        assert names.count("engine.step") == 2
        assert names.count("engine.epoch") == 1
        assert names.count("engine.checkpoint") == 1
        # The fit span opens first and closes in the finally block.
        assert names[0] == "pretrain.fit"
        fit_span = records[0]
        assert fit_span.ended
        assert set(fit_span.attributes) >= {
            "device",
            "precision",
            "world_size",
            "steps_per_epoch",
            "max_epochs",
        }
        # Epoch wraps steps; checkpoint closes the epoch.
        assert (
            names.index("engine.epoch")
            < names.index("engine.step")
            < names.index("engine.checkpoint")
        )

    def test_task_e2e_four_phase_spans(self, tmp_path, monkeypatch):
        from tests.test_train_object_detection import (
            data_config,
            make_coco_dataset,
            model_args,
            transform_args,
        )

        records = install_stub_tracer(monkeypatch)
        train_json, val_json, images_dir = make_coco_dataset(tmp_path)
        result = train_object_detection(
            out=tmp_path / "otel_det",
            data=data_config(train_json, val_json, images_dir),
            steps=4,
            batch_size=4,
            num_workers=0,
            model_args=model_args(),
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 4
        names = [s.name for s in records]
        assert names[0] == "task.fit"
        assert records[0].ended
        assert set(records[0].attributes) >= {"device", "precision", "world_size"}
        # Four phases present with exact counts for this configuration.
        assert names.count("engine.step") == 4
        assert names.count("engine.dataload") == 4  # real dataload phase
        assert names.count("engine.validation") == 2  # steps 1 and 3
        assert names.count("engine.checkpoint") >= 1
        # Dataload is nested inside the step span.
        first_step = names.index("engine.step")
        first_dataload = names.index("engine.dataload")
        assert first_step < first_dataload
        dataload = next(s for s in records if s.name == "engine.dataload")
        assert "step" in dataload.attributes


# ---------------------------------------------------------------------------
# Ascend NPU accelerator adaptation (mocked, no hardware)
# ---------------------------------------------------------------------------


class TestAccelerator:
    def test_npu_unavailable_in_real_env(self, monkeypatch):
        _reset_npu_cache(monkeypatch)
        assert import_torch_npu() is None  # torch_npu not installed here
        assert npu_available() is False
        # The warn-once latch short-circuits subsequent imports.
        monkeypatch.setattr(accelerator, "_TORCH_NPU", None)
        assert import_torch_npu() is None

    def test_import_torch_npu_caches_module(self, monkeypatch):
        fake = types.ModuleType("torch_npu")
        _reset_npu_cache(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch_npu", fake)
        assert import_torch_npu() is fake
        # Cached even after the sys.modules entry is dropped.
        monkeypatch.delitem(sys.modules, "torch_npu")
        assert import_torch_npu() is fake
        # Once the latch is set, a missing module stays unresolved.
        monkeypatch.setattr(accelerator, "_TORCH_NPU", None)
        monkeypatch.setattr(accelerator, "_NPU_IMPORT_WARNED", True)
        monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
        assert import_torch_npu() is None

    def test_npu_available_mock(self, monkeypatch):
        _reset_npu_cache(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "torch_npu", types.ModuleType("torch_npu")
        )
        _fake_torch_npu(monkeypatch, available=True)
        assert npu_available() is True
        _fake_torch_npu(monkeypatch, available=False)
        assert npu_available() is False

        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=_boom), raising=False)
        assert npu_available() is False  # runtime errors degrade to False

    def test_accelerator_device_name_priority(self, monkeypatch):
        # CUDA wins even when NPU is also available.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(accelerator, "npu_available", lambda: True)
        assert accelerator_device_name() == "cuda"
        # NPU is the fallback accelerator.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert accelerator_device_name() == "npu"
        # CPU-only: no accelerator.
        monkeypatch.setattr(accelerator, "npu_available", lambda: False)
        assert accelerator_device_name() is None

    def test_accelerator_device_count(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        assert accelerator_device_count() == 3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(accelerator, "npu_available", lambda: True)
        _fake_torch_npu(monkeypatch, device_count=4)
        assert accelerator_device_count() == 4
        monkeypatch.setattr(accelerator, "npu_available", lambda: False)
        assert accelerator_device_count() == 0

    def test_supports_bf16(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert accelerator_supports_bf16() is True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(accelerator, "npu_available", lambda: True)
        _fake_torch_npu(monkeypatch, bf16=True)
        assert accelerator_supports_bf16() is True

        def _boom():
            raise RuntimeError("unsupported")

        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(is_bf16_supported=_boom), raising=False
        )
        assert accelerator_supports_bf16() is False
        monkeypatch.setattr(accelerator, "npu_available", lambda: False)
        assert accelerator_supports_bf16() is False  # CPU-only

    @pytest.mark.parametrize(
        ("device_type", "backend"),
        [("cuda", "nccl"), ("npu", "hccl"), ("cpu", "gloo"), ("mps", "gloo")],
    )
    def test_ddp_backend_for(self, device_type, backend):
        assert ddp_backend_for(device_type) == backend

    @pytest.mark.parametrize(
        ("device_type", "expected"),
        [("cuda", True), ("npu", True), ("cpu", False), ("mps", False)],
    )
    def test_supports_amp(self, device_type, expected):
        assert supports_amp(device_type) is expected

    def test_resolve_device_auto_prefers_npu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine_module, "npu_available", lambda: True)
        _fake_torch_npu(monkeypatch, device_count=4)
        _fake_torch_device(monkeypatch)
        device = resolve_device("auto")
        assert device.type == "npu" and device.index is None
        device = resolve_device(3)
        assert device.type == "npu" and device.index == 3
        device = resolve_device(7)
        assert device.type == "npu" and device.index == 3  # modulo wrap

    def test_resolve_device_honors_local_rank_on_npu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine_module, "npu_available", lambda: True)
        _fake_torch_npu(monkeypatch, device_count=2)
        _fake_torch_device(monkeypatch)
        monkeypatch.setenv("LOCAL_RANK", "3")
        device = resolve_device("auto")
        assert device.type == "npu" and device.index == 1
        # Explicit specs pass through untouched.
        device = resolve_device("npu:0")
        assert device.type == "npu" and device.index == 0
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_device_cpu_fallback(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine_module, "npu_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device(2) == torch.device("cpu")
        assert resolve_device("auto") == torch.device("cpu")
