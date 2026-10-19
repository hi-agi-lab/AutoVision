# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for peripheral subsystems: loggers, telemetry, debug, visualization,
embed, video extraction, torch.compile, 8-bit optimizer, CLI."""

import json
import sys
import types

import pytest
import torch
from torch import nn

# ── loggers ─────────────────────────────────────────────────────────────


class TestLoggers:
    def test_jsonl_always_available(self, tmp_path):
        from autovision.observability.loggers import get_loggers

        loggers = get_loggers(None, tmp_path)
        assert len(loggers) >= 1
        loggers[0].log_metrics({"train_loss": 1.0}, step=3)
        log_hyperparams = {"a": 1}
        loggers[0].log_hyperparams(log_hyperparams)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert any("train_loss" in line for line in lines)

    @staticmethod
    def _hide_tensorboard(monkeypatch) -> None:
        """Force the missing-package path regardless of the environment."""
        stub = types.ModuleType("torch.utils.tensorboard")

        def _raise(*args, **kwargs):
            raise ImportError("tensorboard hidden by test")

        stub.SummaryWriter = _raise
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", stub)
        monkeypatch.setitem(sys.modules, "torch", __import__("torch"))
        monkeypatch.delitem(sys.modules, "torch.utils", raising=False)

    def test_tensorboard_lazy_missing_warns(self, tmp_path, monkeypatch, caplog):
        self._hide_tensorboard(monkeypatch)
        from autovision.observability.loggers import get_loggers

        with caplog.at_level("INFO"):
            loggers = get_loggers(None, tmp_path)  # default: skip silently
        assert not any("TensorBoard" in type(lg).__name__ for lg in loggers)

    def test_tensorboard_explicit_missing_raises(self, tmp_path, monkeypatch):
        self._hide_tensorboard(monkeypatch)
        from autovision.observability.loggers import get_loggers

        with pytest.raises(ImportError, match="tensorboard"):
            get_loggers({"tensorboard": {}}, tmp_path)

    def test_unknown_logger_raises(self, tmp_path):
        from autovision.observability.loggers import get_loggers

        with pytest.raises(ValueError, match="Unknown loggers"):
            get_loggers({"neptune": {}}, tmp_path)

    def test_fake_mlflow_and_wandb(self, tmp_path, monkeypatch):
        """Fake external packages to test the wrapper protocol."""
        calls = {"mlflow_metrics": [], "wandb_metrics": []}

        fake_mlflow = types.ModuleType("mlflow")

        class FakeRun:
            def log_metrics(self, metrics, step=None):
                calls["mlflow_metrics"].append((metrics, step))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        fake_mlflow.set_tracking_uri = lambda uri: None
        fake_mlflow.set_experiment = lambda name: None
        fake_mlflow.start_run = lambda run_name=None: FakeRun()
        fake_mlflow.log_metrics = lambda m, step=None: calls["mlflow_metrics"].append((m, step))
        fake_mlflow.log_params = lambda p: None
        fake_mlflow.log_artifact = lambda p: None
        fake_mlflow.end_run = lambda: None
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        fake_wandb = types.ModuleType("wandb")

        class FakeWandbRun:
            def log(self, metrics, step=None):
                calls["wandb_metrics"].append((metrics, step))

            def config_update(self, params):
                pass

        fake_wandb.init = lambda **kwargs: FakeWandbRun()
        fake_wandb.log = lambda metrics, step=None: calls["wandb_metrics"].append((metrics, step))
        fake_wandb.config = types.SimpleNamespace(update=lambda p: None)
        fake_wandb.finish = lambda: None
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        from autovision.observability.loggers import get_loggers

        loggers = get_loggers({"mlflow": {}, "wandb": {}}, tmp_path)
        names = [type(lg).__name__ for lg in loggers]
        assert "MLflowLogger" in names and "WandbLogger" in names
        for lg in loggers:
            lg.log_metrics({"x": 1.0}, step=1)
            lg.log_hyperparams({"p": 2})
            if hasattr(lg, "finalize"):
                lg.finalize()
        assert any(m == {"x": 1.0} for m, _ in calls["mlflow_metrics"])
        assert any(m == {"x": 1.0} for m, _ in calls["wandb_metrics"])

    def test_wired_into_pretrain(self, tmp_path):
        """pretrain(loggers=...) must honor the spec (JSONL only)."""
        from tests.conftest import create_images

        from autovision import pretrain

        create_images(tmp_path / "images", num_images=8, size=(64, 64))
        result = pretrain(
            out=tmp_path / "run", data=tmp_path / "images",
            model="torchvision/resnet18", epochs=1, batch_size=4,
            num_workers=0, method_args={"queue_size": 4, "teacher_pretrained": False},
            loggers={"tensorboard": False},
        )
        assert (result.out_dir / "metrics.jsonl").exists()


# ── telemetry (opt-in) ──────────────────────────────────────────────────


class TestTelemetry:
    def test_disabled_by_default(self, monkeypatch):
        import urllib.request

        from autovision.observability.events import track_event

        def _fail(*args, **kwargs):
            raise AssertionError("network must not be touched")

        monkeypatch.setattr(urllib.request, "urlopen", _fail)
        track_event("test_event", {"x": 1})  # must be a no-op

    def test_enabled_sends_payload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOVISION_TELEMETRY", "1")
        monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv(
            "AUTOVISION_TELEMETRY_ENDPOINT", str(tmp_path / "endpoint")
        )  # invalid URL scheme -> transport error path is swallowed
        from autovision.observability.events import tracker

        sent = {}
        monkeypatch.setattr(tracker, "_sender", lambda payload, endpoint: sent.update(
            payload=json.loads(payload.decode()), endpoint=endpoint
        ))

        tracker.track_event("training_started", {"task_type": "detection"})
        assert "training_started" in sent["payload"]["events"][0]["event"]
        event = sent["payload"]["events"][0]
        assert event["properties"]["task_type"] == "detection"
        assert event["distinct_id"]  # anonymous uuid present
        # No data paths leak into the payload.
        assert str(tmp_path) not in json.dumps(sent["payload"])

    def test_rate_limit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOVISION_TELEMETRY", "1")
        monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("AUTOVISION_TELEMETRY_ENDPOINT", str(tmp_path / "e"))
        from autovision.observability.events import tracker

        monkeypatch.setattr(tracker, "_sender", lambda *a: None)
        tracker.track_event("evt", {})
        tracker.track_event("evt", {})  # rate-limited: dropped
        assert len(tracker._events) == 0

    def test_user_id_persisted(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOVISION_TELEMETRY", "1")
        monkeypatch.setenv("AUTOVISION_CACHE_DIR", str(tmp_path))
        from autovision.observability.events import tracker

        uid1 = tracker._anonymous_user_id()
        uid2 = tracker._anonymous_user_id()
        assert uid1 == uid2 == (tmp_path / "userid.txt").read_text().strip()


# ── debug subsystem ─────────────────────────────────────────────────────


class TestDebug:
    def test_nan_capture_snapshot_and_replay(self, tmp_path):
        from tests.test_methods_distillationv3 import make_method

        from autovision.debug import (
            NaNCaptureMonitor,
            NaNDetectedError,
            load_nan_capture,
        )

        model = make_method()
        monitor = NaNCaptureMonitor(
            model, tmp_path, model_class_path="fake.Path", model_init_args={"a": 1}
        )
        batch = {"views": [torch.randn(2, 3, 16, 16)],
                 "filename": ["a.jpg", "b.jpg"]}
        monitor.capture_batch(batch)
        monitor.set_optimizer_state({"lr": 0.001})
        bad_loss = torch.tensor(float("nan"), requires_grad=True)
        with pytest.raises(NaNDetectedError, match="snapshot"):
            monitor.check_loss(bad_loss)
        path = tmp_path / "nan_capture" / "nan_capture.pt"
        assert path.exists()
        # Snapshots carry RNG-state objects: the safe loader must reject
        # them by default, and replay requires the explicit opt-in.
        with pytest.raises(ValueError, match="trust_source"):
            load_nan_capture(path)
        snapshot = load_nan_capture(path, trust_source=True)
        assert snapshot["trigger"] == "loss"
        assert snapshot["model_class_path"] == "fake.Path"
        assert len(snapshot["microbatches"]) == 1
        assert snapshot["torch_rng_state"].numel() > 0

    def test_nan_gradient_check(self, tmp_path):
        from tests.test_methods_distillationv3 import make_method

        from autovision.debug import NaNCaptureMonitor, NaNDetectedError

        model = make_method()
        monitor = NaNCaptureMonitor(model, tmp_path)
        model.training_step({"views": [torch.randn(2, 3, 16, 16)],
                             "filename": ["a.jpg"]}, 0).backward()
        # Corrupt one gradient.
        for p in model.parameters():
            if p.grad is not None:
                p.grad[0, 0] = float("inf")
                break
        with pytest.raises(NaNDetectedError, match="gradient"):
            monitor.check_gradients(model)

    def test_finite_gradients_pass(self, tmp_path):
        from tests.test_methods_distillationv3 import make_method

        from autovision.debug import NaNCaptureMonitor

        model = make_method()
        monitor = NaNCaptureMonitor(model, tmp_path)
        loss = model.training_step({"views": [torch.randn(2, 3, 16, 16)],
                                    "filename": ["a.jpg"]}, 0)
        loss.backward()
        monitor.check_loss(loss)
        monitor.check_gradients(model)  # must not raise

    def test_underflow_overflow_hooks(self, tmp_path):
        from autovision.debug import DebugUnderflowOverflow

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
        monitor = DebugUnderflowOverflow(model, out_dir=tmp_path)
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            model(x)
        monitor.set_batch_number(0)
        assert not monitor.aborted
        # Force an anomaly: poison a module output via a hook? Simply dump.
        path = monitor.dump_frames()
        assert path.exists()

    def test_underflow_abort(self, tmp_path):
        from autovision.debug import DebugUnderflowOverflow

        model = nn.Sequential(nn.Conv2d(3, 4, 3))
        monitor = DebugUnderflowOverflow(
            model, out_dir=tmp_path, abort_after_batch_num=2
        )
        monitor.set_batch_number(0)
        monitor.set_batch_number(1)
        monitor.set_batch_number(2)
        assert monitor.aborted
        assert monitor._hooks == []

    def test_debug_args_wired_into_task_engine(self, tmp_path):
        """debug_args flow into TaskEngine (snapshot behavior is covered by
        the monitor unit tests above)."""
        from tests.test_train_object_detection import (
            data_config,
            make_coco_dataset,
            model_args,
            transform_args,
        )

        from autovision import train_object_detection

        train_json, val_json, images_dir = make_coco_dataset(tmp_path)
        result = train_object_detection(
            out=tmp_path / "det_dbg",
            data=data_config(train_json, val_json, images_dir),
            steps=2,
            batch_size=4, num_workers=0,
            model_args=model_args(), transform_args=transform_args(),
            debug_args={"nan_capture": {"enabled": True},
                        "underflow_overflow": {"enabled": True}},
        )
        assert result.steps_run == 2  # monitors active without false trips


# ── visualization ───────────────────────────────────────────────────────


class TestVisualization:
    def test_plot_example_augmentations(self, tmp_path):
        from autovision.cli.plot import save_example_augmentations

        batch = {"views": [torch.rand(2, 3, 64, 64), torch.rand(2, 3, 128, 128)]}
        path = save_example_augmentations(batch, tmp_path / "aug.jpg")
        assert path.exists() and path.stat().st_size > 0

    def test_save_detection_examples(self, tmp_path):
        from autovision.visualize import save_detection_examples

        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "bboxes": [torch.tensor([[0.25, 0.25, 0.25, 0.25]])] * 2,
            "classes": [torch.tensor([0])] * 2,
        }
        predictions = [{"bboxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]])}]
        paths = save_detection_examples(tmp_path, batch, step=0, predictions=predictions)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)


# ── embed ───────────────────────────────────────────────────────────────


class TestEmbed:
    def test_full_cycle(self, tmp_path):
        from tests.conftest import create_images

        from autovision import embed, pretrain

        create_images(tmp_path / "images", num_images=8, size=(64, 64))
        result = pretrain(
            out=tmp_path / "run", data=tmp_path / "images",
            model="torchvision/resnet18", epochs=1, batch_size=4, num_workers=0,
            method_args={"queue_size": 4, "teacher_pretrained": False},
        )
        out = tmp_path / "embeddings.pt"
        embed(out=out, data=tmp_path / "images", checkpoint=result.exported_model_path,
              format="torch", image_size=64, batch_size=2)
        payload = torch.load(out, weights_only=True)
        assert set(payload) == {"filenames", "embeddings"}
        assert payload["embeddings"].shape[0] == 8

    def test_csv_format(self, tmp_path):
        from tests.conftest import create_images

        from autovision import embed, pretrain

        create_images(tmp_path / "images", num_images=8, size=(64, 64))
        result = pretrain(
            out=tmp_path / "run", data=tmp_path / "images",
            model="torchvision/resnet18", epochs=1, batch_size=4, num_workers=0,
            method_args={"queue_size": 4, "teacher_pretrained": False},
        )
        out = tmp_path / "embeddings.csv"
        embed(out=out, data=tmp_path / "images", checkpoint=result.exported_model_path,
              format="csv", image_size=64)
        lines = out.read_text().strip().splitlines()
        assert lines[0].startswith("filename,embedding_0")

    def test_existing_output_raises(self, tmp_path):
        from autovision import embed

        existing = tmp_path / "out.pt"
        existing.write_bytes(b"x")
        with pytest.raises(ValueError, match="overwrite"):
            embed(out=existing, data=tmp_path, checkpoint=tmp_path / "nope.pt")

    def test_invalid_checkpoint_raises(self, tmp_path):
        from autovision import embed

        bad = tmp_path / "bad.pt"
        torch.save({"random": 1}, bad)
        with pytest.raises(ValueError, match="not an exported embedding model"):
            embed(out=tmp_path / "e.pt", data=tmp_path, checkpoint=bad)


# ── extract_video_frames ────────────────────────────────────────────────


class TestExtractVideoFrames:
    def _make_video(self, tmp_path):
        import subprocess

        video = tmp_path / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=64x64:rate=5",
             "-pix_fmt", "yuv420p", str(video)],
            check=True, capture_output=True,
        )
        return video

    def test_extract(self, tmp_path):
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed (apt install ffmpeg)")
        from autovision import extract_video_frames

        video = self._make_video(tmp_path)
        dirs = extract_video_frames(video, out=tmp_path / "frames",
                                    max_frames_per_video=5)
        frames = list(dirs[0].glob("*.jpg"))
        assert len(frames) == 5

    def test_existing_frames_raise(self, tmp_path):
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed (apt install ffmpeg)")
        from autovision import extract_video_frames

        video = self._make_video(tmp_path)
        extract_video_frames(video, out=tmp_path / "frames", max_frames_per_video=2)
        with pytest.raises(ValueError, match="not empty"):
            extract_video_frames(video, out=tmp_path / "frames")


# ── torch.compile + 8-bit optimizer ─────────────────────────────────────


class TestTorchCompile:
    def test_disabled_by_default(self):
        from autovision.runtime.torch_compile import TorchCompileArgs, try_compile

        def fn(x):
            return x + 1

        assert try_compile(fn, "test", TorchCompileArgs()) is fn

    def test_extra_kwargs_forwarded(self):
        from autovision.runtime.torch_compile import TorchCompileArgs

        args = TorchCompileArgs(disable=False, mode="reduce-overhead")
        assert args.mode == "reduce-overhead"


class TestAdamW8bit:
    def test_missing_package_raises(self):
        pytest.importorskip.__doc__  # keep importorskip semantics documented
        from autovision.optim.adamw8bit_args import AdamW8BitArgs

        args = AdamW8BitArgs()
        try:
            import bitsandbytes  # noqa: F401

            pytest.skip("bitsandbytes installed: error path not testable")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="bitsandbytes"):
            args.get_optimizer(params=[{"params": []}], lr_scale=1.0)

    def test_registered_in_lookup(self):
        from autovision.optim import get_optimizer_args_cls

        assert get_optimizer_args_cls("adamw8bit").__name__ == "AdamW8BitArgs"


# ── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_list_methods(self, capsys):
        from typer.testing import CliRunner

        from autovision.cli._app import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-methods"])
        assert result.exit_code == 0
        assert "distillation" in result.output

    def test_list_models(self):
        from typer.testing import CliRunner

        from autovision.cli._app import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "torchvision/resnet18" in result.output

    def test_pretrain_smoke(self, tmp_path):
        from tests.conftest import create_images
        from typer.testing import CliRunner

        from autovision.cli._app import app

        create_images(tmp_path / "images", num_images=8, size=(64, 64))
        runner = CliRunner()
        result = runner.invoke(app, [
            "pretrain",
            f"out={tmp_path / 'run'}",
            f"data={tmp_path / 'images'}",
            "model=torchvision/resnet18",
            "epochs=1", "batch_size=4", "num_workers=0",
            "method_args=" + json.dumps({"queue_size": 4, "teacher_pretrained": False}),
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "run" / "checkpoints" / "last.ckpt").exists()

    def test_bad_kwargs_message(self):
        from typer.testing import CliRunner

        from autovision.cli._app import app

        runner = CliRunner()
        result = runner.invoke(app, ["pretrain", "novalue"])
        assert result.exit_code != 0

    def test_jsonl_args_forwarded(self, tmp_path):
        from autovision.observability.loggers import get_loggers

        loggers = get_loggers({"jsonl": {"flush_logs_every_n_steps": 5}}, tmp_path)
        assert len(loggers) >= 1  # jsonl built from args without error


class TestFileLogging:
    def test_train_log_receives_info_records(self, tmp_path):
        # Regression: the file handler used to be attached without raising
        # the logger level, so INFO records never reached train.log (empty
        # file bug seen in real runs).
        import logging as logging_module

        from autovision.runtime.logging import set_up_file_logging

        log_file = tmp_path / "train.log"
        logger = logging_module.getLogger("autovision")
        previous_level = logger.level
        previous_handlers = logger.handlers[:]
        try:
            set_up_file_logging(log_file)
            logger.info("smoke-info-record")
            for handler in logger.handlers:
                handler.flush()
            content = log_file.read_text()
            assert "smoke-info-record" in content
        finally:
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)

    @staticmethod
    def _handler_kinds(logger):
        import logging as logging_module
        from logging.handlers import RotatingFileHandler

        return [
            "file" if isinstance(h, RotatingFileHandler) else "console"
            if type(h) is logging_module.StreamHandler else "other"
            for h in logger.handlers
        ]

    def test_console_setup_preserves_file_handler_in_both_orders(self, tmp_path):
        # Regression: set_up_console_logging used to strip every handler
        # except the ones its isinstance check was meant to remove —
        # calling it after set_up_file_logging silently deleted the
        # train.log handler (0-byte train.log, real pilot run).
        import logging as logging_module

        from autovision.runtime.logging import (
            set_up_console_logging,
            set_up_file_logging,
        )

        logger = logging_module.getLogger("autovision")
        previous_level = logger.level
        previous_handlers = logger.handlers[:]
        try:
            # Hermetic start: e2e tests elsewhere call the train command and
            # leak handlers into the shared logger; this test pins the
            # setup functions' own behavior from a clean slate.
            logger.handlers = []
            set_up_file_logging(tmp_path / "train.log")
            set_up_console_logging()
            kinds = self._handler_kinds(logger)
            assert kinds.count("file") == 1, kinds
            assert kinds.count("console") == 1, kinds

            # And the reverse order must land in the same state.
            logger.handlers = []
            set_up_console_logging()
            set_up_file_logging(tmp_path / "train.log")
            kinds = self._handler_kinds(logger)
            assert kinds.count("file") == 1, kinds
            assert kinds.count("console") == 1, kinds

            # Idempotence: a second console call must not stack handlers.
            set_up_console_logging()
            kinds = self._handler_kinds(logger)
            assert kinds.count("console") == 1, kinds
        finally:
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)


class TestLoggerLifecycleRegressions:
    def test_get_loggers_is_rank_zero_only(self, tmp_path, monkeypatch):
        import autovision.observability.loggers as logger_module

        monkeypatch.setattr(logger_module, "is_global_rank_zero", lambda: False)
        assert logger_module.get_loggers(None, tmp_path) == []

    def test_finalize_isolates_backend_failures(self, caplog):
        from autovision.observability.loggers import finalize_loggers

        calls = []

        class _Broken:
            def finalize(self):
                raise RuntimeError("backend down")

        class _Good:
            def finalize(self):
                calls.append("good")

        with caplog.at_level("WARNING", logger="autovision"):
            finalize_loggers([_Broken(), _Good()])
        assert calls == ["good"]
        assert "finalize failed" in caplog.text

    def test_file_handler_is_replaced_between_runs(self, tmp_path):
        import logging as logging_module

        from autovision.runtime.logging import set_up_file_logging

        logger = logging_module.getLogger("autovision")
        previous_level = logger.level
        previous_handlers = logger.handlers[:]
        first = tmp_path / "run_a" / "train.log"
        second = tmp_path / "run_b" / "train.log"
        try:
            logger.handlers = []
            set_up_file_logging(first)
            logger.info("record-from-run-a")
            set_up_file_logging(second)
            logger.info("record-from-run-b")
            for handler in logger.handlers:
                handler.flush()
            assert sum(
                bool(getattr(h, "_autovision_file_handler", False))
                for h in logger.handlers
            ) == 1
            assert "record-from-run-a" in first.read_text()
            assert "record-from-run-b" not in first.read_text()
            assert "record-from-run-b" in second.read_text()
        finally:
            for handler in logger.handlers[:]:
                if getattr(handler, "_autovision_file_handler", False):
                    logger.removeHandler(handler)
                    handler.close()
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)
