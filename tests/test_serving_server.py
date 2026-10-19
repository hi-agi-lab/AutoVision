# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Serving launcher: TLS options, multi-worker factory, CLI wiring."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from autovision.cli._app import app as cli_app
from autovision.errors import AutoVisionError
from autovision.serving import server, worker
from autovision.serving.server import serve
from autovision.serving.worker import SERVE_CONFIG_ENV, create_app_from_env


class _Capture:
    def __init__(self) -> None:
        self.args: tuple = ()
        self.kwargs: dict = {}

    def __call__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture()
def uvicorn_run(monkeypatch):
    capture = _Capture()
    monkeypatch.setattr("uvicorn.run", capture)
    return capture


class TestServeLauncher:
    def test_single_worker_builds_the_app_and_passes_tls(
        self, monkeypatch, uvicorn_run
    ):
        monkeypatch.setattr(
            server, "create_app", lambda model, **kwargs: ("APP", model, kwargs)
        )
        serve(
            "model.pt",
            host="0.0.0.0",
            port=9000,
            workers=1,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
            ssl_cert_reqs=2,
            threshold=0.3,
        )
        assert uvicorn_run.args[0] == ("APP", "model.pt", {"threshold": 0.3})
        assert uvicorn_run.kwargs["host"] == "0.0.0.0"
        assert uvicorn_run.kwargs["port"] == 9000
        assert uvicorn_run.kwargs["ssl_keyfile"] == "key.pem"
        assert uvicorn_run.kwargs["ssl_certfile"] == "cert.pem"
        assert uvicorn_run.kwargs["ssl_ca_certs"] == "ca.pem"
        assert uvicorn_run.kwargs["ssl_cert_reqs"] == 2
        assert "workers" not in uvicorn_run.kwargs

    def test_multi_worker_uses_the_import_string_factory(
        self, monkeypatch, uvicorn_run
    ):
        old = os.environ.get(SERVE_CONFIG_ENV)
        try:
            serve(
                "model.pt",
                workers=2,
                threshold=0.3,
                class_names={0: "cat"},
            )
            assert uvicorn_run.args[0] == (
                "autovision.serving.worker:create_app_from_env"
            )
            assert uvicorn_run.kwargs["factory"] is True
            assert uvicorn_run.kwargs["workers"] == 2
            payload = json.loads(os.environ[SERVE_CONFIG_ENV])
            assert payload == {
                "model": "model.pt",
                "threshold": 0.3,
                "class_names": {"0": "cat"},
            }
        finally:
            if old is None:
                os.environ.pop(SERVE_CONFIG_ENV, None)
            else:
                os.environ[SERVE_CONFIG_ENV] = old

    def test_invalid_launch_options_fail_loud(self):
        with pytest.raises(AutoVisionError, match="workers"):
            serve("model.pt", workers=0)
        with pytest.raises(AutoVisionError, match="ssl_keyfile"):
            serve("model.pt", ssl_keyfile="key.pem")
        with pytest.raises(AutoVisionError, match="ssl_cert_reqs"):
            serve("model.pt", ssl_cert_reqs=2)
        with pytest.raises(AutoVisionError, match="workers>1"):
            serve(object(), workers=2)
        with pytest.raises(AutoVisionError, match="variants"):
            serve("model.pt", workers=2, variants={"a": object()})

    def test_worker_factory_rebuilds_config(self, monkeypatch):
        monkeypatch.setenv(
            SERVE_CONFIG_ENV,
            json.dumps({"model": "m.pt", "class_names": {"0": "cat"}}),
        )
        monkeypatch.setattr(
            worker, "create_app", lambda **kwargs: kwargs
        )
        config = create_app_from_env()
        assert config["model"] == "m.pt"
        assert config["class_names"] == {0: "cat"}

    def test_worker_factory_requires_a_config(self, monkeypatch):
        monkeypatch.delenv(SERVE_CONFIG_ENV, raising=False)
        with pytest.raises(AutoVisionError, match=SERVE_CONFIG_ENV):
            create_app_from_env()


class TestServeCLI:
    def test_cli_forwards_transport_options(self, monkeypatch):
        captured: dict = {}

        def _fake_serve(model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

        monkeypatch.setattr(server, "serve", _fake_serve)
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            cli_app,
            [
                "serve", "model=m.pt", "port=9443", "workers=2",
                "ssl_keyfile=k.pem", "ssl_certfile=c.pem", "ssl_cert_reqs=2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["model"] == "m.pt"
        assert captured["port"] == 9443
        assert captured["workers"] == 2
        assert captured["ssl_keyfile"] == "k.pem"
        assert captured["ssl_certfile"] == "c.pem"
        assert captured["ssl_cert_reqs"] == 2


class TestGracefulShutdown:
    def test_timeout_is_forwarded_to_uvicorn(self, monkeypatch, uvicorn_run):
        monkeypatch.setattr(
            server, "create_app", lambda model, **kwargs: ("APP", model, kwargs)
        )
        serve("model.pt", timeout_graceful_shutdown=5.0)
        assert uvicorn_run.kwargs["timeout_graceful_shutdown"] == 5.0

    def test_non_positive_timeout_is_rejected(self):
        with pytest.raises(AutoVisionError, match="timeout_graceful_shutdown"):
            serve("model.pt", timeout_graceful_shutdown=0)
