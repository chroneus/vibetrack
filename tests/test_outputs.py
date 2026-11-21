"""Tests for CLI, viewer auto-discovery, listen server, and output backends."""

import asyncio
import io
import inspect
import importlib.util
import json
import logging
from pathlib import Path
from unittest import mock

import httpx
import pytest

from vibetrack.config import save_config
from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.viewers.console import ConsoleOutput, _sparkline
from vibetrack.viewers import mcp as mcp_module
from vibetrack.viewers.mcp import (
    _analyze_scalar_payload,
    _compare_image_lpips_payload,
    _compare_scalar_payload,
    _suppress_streamable_http_startup_log,
)
from vibetrack.writer import SummaryWriter


class _ASGIClient:
    def __init__(self, app):
        self._app = app

    def request(self, method: str, url: str, **kwargs):
        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:6116",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_run())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def close(self):
        return None


def _call_app_route(app, route_path: str, method: str = "GET", **kwargs):
    for route in app.router.routes:
        route_pattern = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set())
        if route_pattern == route_path and method in route_methods:
            result = route.endpoint(**kwargs)
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result
    raise LookupError(f"route not found: {method} {route_path}")


@pytest.fixture
def populated_project(tmp_path):
    project_folder = tmp_path / "project"
    for name, lr in [("run_a", 0.01), ("run_b", 0.001)]:
        run_dir = project_folder / name
        with SummaryWriter(
            str(run_dir), name=name, project_folder=str(project_folder)
        ) as w:
            for i in range(30):
                w.add_scalar("loss", lr * 10 / (i + 1), i)
                w.add_scalar("acc", 1.0 - lr * 10 / (i + 1), i)
    return str(project_folder)


class TestSparkline:
    def test_monotone_increasing_is_sorted_chars(self):
        """Strictly increasing input must produce non-decreasing unicode block chars."""
        line = _sparkline([0, 1, 2, 3, 4, 5, 6, 7])
        assert all(ord(line[i]) <= ord(line[i + 1]) for i in range(len(line) - 1))

    def test_downsampling_exact_width(self):
        line = _sparkline(list(range(100)), width=20)
        assert len(line) == 20

    def test_empty(self):
        assert _sparkline([]) == ""


class TestConsoleOutput:
    def test_show(self, populated_project):
        out = ConsoleOutput(populated_project)
        result = out.show()
        assert "run_a" in result
        assert "run_b" in result
        assert "loss" in result
        out.close()

    def test_summary(self, populated_project):
        out = ConsoleOutput(populated_project)
        result = out.summary()
        assert "run_a" in result
        out.close()

    def test_filter_experiments(self, populated_project):
        out = ConsoleOutput(populated_project)
        result = out.show(experiments=["run_a"])
        assert "run_a" in result
        assert "run_b" not in result
        out.close()

    def test_filter_tags(self, populated_project):
        out = ConsoleOutput(populated_project)
        result = out.show(tags=["loss"])
        assert "loss" in result
        lines = result.strip().split("\n")
        data_lines = [line for line in lines[2:] if line.strip()]
        for line in data_lines:
            assert "loss" in line
        out.close()

    def test_empty_dir(self, tmp_path):
        out = ConsoleOutput(str(tmp_path / "nope"))
        result = out.show()
        assert "No experiments" in result
        out.close()


class TestCLI:
    def test_no_args_defaults_to_web(self, tmp_path, monkeypatch):
        """vibetrack with no args should default to the 'web' viewer."""
        from vibetrack.cli import main

        central_path = tmp_path / ".vibetrack" / "vibetrack.db"
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_path)

        with mock.patch("vibetrack.viewers.web.WebOutput.show") as mock_show:
            main([])
            mock_show.assert_called_once()

    def test_viewer_console(self, populated_project, capsys):
        from vibetrack.cli import main

        main(["--project-folder", populated_project, "--viewer", "console"])
        captured = capsys.readouterr()
        assert "loss" in captured.out

    def test_unknown_viewer_errors(self):
        from vibetrack.cli import main

        with pytest.raises(ValueError, match="Unknown viewer"):
            main(["--viewer", "nonexistent"])

    def test_listen_flag_parsed(self, populated_project):
        """--listen should start the ingest server thread."""
        from vibetrack.cli import main

        with mock.patch(
            "vibetrack.cli._start_listen_server"
        ) as mock_listen, mock.patch("vibetrack.viewers.web.WebOutput.show"):
            main(
                [
                    "--project-folder",
                    populated_project,
                    "--viewer",
                    "web",
                    "--listen",
                    "0.0.0.0:9009",
                ]
            )
            mock_listen.assert_called_once_with(
                populated_project, "0.0.0.0", 9009, None
            )

    def test_listen_with_token(self, populated_project):
        from vibetrack.cli import main

        with mock.patch(
            "vibetrack.cli._start_listen_server"
        ) as mock_listen, mock.patch("vibetrack.viewers.web.WebOutput.show"):
            main(
                [
                    "--project-folder",
                    populated_project,
                    "--viewer",
                    "web",
                    "--listen",
                    "0.0.0.0:9009",
                    "--token",
                    "abc",
                ]
            )
            mock_listen.assert_called_once_with(
                populated_project, "0.0.0.0", 9009, "abc"
            )

    def test_migrate_command(self, tmp_path):
        from vibetrack.cli import main

        project_folder = tmp_path / "project"
        project_folder.mkdir()
        for name in ["legacy_a", "legacy_b"]:
            run_dir = project_folder / name
            run_dir.mkdir()
            db = Database(run_dir / "vibetrack.db")
            exp_id = db.create_experiment(
                name, project=project_folder.name, log_dir=str(run_dir)
            )
            db.add_scalar(exp_id, "loss", 0.5, 0)
            db.close()

        with pytest.raises(SystemExit) as exc:
            main(["migrate", str(project_folder)])
        assert exc.value.code == 0

        reader = RunReader(str(project_folder))
        names = {exp.name for exp in reader.experiments()}
        assert names == {"legacy_a", "legacy_b"}
        reader.close()

    def test_local_db_path_is_synonym_for_project_folder(self):
        from vibetrack.cli import _normalize_project_folder

        demo_dir = str((Path("/tmp") / "demo").resolve())
        demo_db = str((Path("/tmp") / "demo" / "vibetrack.db").resolve())
        assert _normalize_project_folder(demo_db) == demo_dir
        assert _normalize_project_folder(demo_dir) == demo_dir
        assert _normalize_project_folder(None) is None


class TestAutoDiscovery:
    def test_discover_finds_all_viewers(self):
        from vibetrack.viewers import discover_viewers

        viewers = discover_viewers()
        assert "web" in viewers
        assert "console" in viewers
        assert "gradio" in viewers
        assert "telegram" in viewers
        assert "mcp" in viewers

    def test_gradio_ui_stripped(self):
        """gradio_ui.py should map to 'gradio', not 'gradio_ui'."""
        from vibetrack.viewers import discover_viewers

        viewers = discover_viewers()
        assert "gradio" in viewers
        assert "gradio_ui" not in viewers

    def test_base_excluded(self):
        from vibetrack.viewers import discover_viewers

        viewers = discover_viewers()
        assert "base" not in viewers
        assert "__init__" not in viewers


class TestMCPOutput:
    def test_analyze_scalar_reports_extrema_trend_and_events(self):
        rows = [
            {"step": 0, "value": 5.0, "wall_time": 1.0},
            {"step": 1, "value": 3.0, "wall_time": 2.0},
            {"step": 2, "value": 2.0, "wall_time": 3.0},
            {"step": 3, "value": 2.01, "wall_time": 4.0},
            {"step": 4, "value": 2.0, "wall_time": 5.0},
        ]

        result = _analyze_scalar_payload("run_a", 7, "loss/val", rows)

        assert result["experiment"] == "run_a"
        assert result["count"] == 5
        assert result["min"]["step"] == 2
        assert result["max"]["step"] == 0
        assert result["best"]["value"] == 2.0
        assert "decreasing" in result["trend"]["direction"]
        assert result["plateau"]["start_step"] == 2
        assert {event["type"] for event in result["events"]} >= {
            "minimum",
            "maximum",
            "largest_drop",
            "largest_rise",
            "plateau",
        }

    def test_compare_scalar_ranks_experiments_and_reports_missing(self):
        class Exp:
            def __init__(self, name, experiment_id, series):
                self.name = name
                self.experiment_id = experiment_id
                self._series = series

            def scalars(self, tag):
                return self._series.get(tag, [])

        exps = [
            Exp(
                "baseline",
                1,
                {"loss": [{"step": 0, "value": 4.0}, {"step": 1, "value": 2.0}]},
            ),
            Exp(
                "candidate",
                2,
                {"loss": [{"step": 0, "value": 4.0}, {"step": 1, "value": 1.5}]},
            ),
            Exp("missing", 3, {}),
        ]

        result = _compare_scalar_payload(exps, "loss", objective="min")

        assert result["winner"]["experiment"] == "candidate"
        assert [row["experiment"] for row in result["ranking"]] == [
            "candidate",
            "baseline",
        ]
        assert result["missing"] == ["missing"]

    def test_compare_image_lpips_payload_reports_lpips_and_pixel_metrics(
        self, tmp_path, monkeypatch
    ):
        image_module = pytest.importorskip("PIL.Image")

        ref_path = tmp_path / "ref.png"
        cand_path = tmp_path / "cand.png"
        image_module.new("RGB", (4, 4), (255, 0, 0)).save(ref_path)
        image_module.new("RGB", (4, 4), (255, 0, 0)).save(cand_path)

        class Exp:
            def __init__(self, name, experiment_id, path):
                self.name = name
                self.experiment_id = experiment_id
                self._path = path

            def images(self, tag):
                return [
                    {
                        "step": 3,
                        "path": f"media/{tag}/3.png",
                        "abs_path": str(self._path),
                    }
                ]

        monkeypatch.setattr(
            mcp_module,
            "_lpips_metric",
            lambda _reference, _candidate: (0.123, None),
        )

        result = _compare_image_lpips_payload(
            Exp("reference", 1, ref_path),
            Exp("candidate", 2, cand_path),
            "samples",
        )

        assert result["selection"] == "latest_common_step"
        assert result["compared_step"] == 3
        assert result["lpips"]["available"] is True
        assert result["lpips"]["distance"] == 0.123
        assert result["pixel"]["mse"] == 0.0
        assert result["pixel"]["mae"] == 0.0

    def test_streamable_http_startup_log_is_suppressed(self):
        logger = logging.getLogger("mcp.server.streamable_http_manager")
        previous_level = logger.level
        previous_propagate = logger.propagate
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)

        try:
            logger.info("outside")
            assert "outside" in stream.getvalue()

            stream.seek(0)
            stream.truncate(0)

            with _suppress_streamable_http_startup_log():
                logger.info("inside")

            assert stream.getvalue() == ""

            logger.info("after")
            assert "after" in stream.getvalue()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate


class TestNotificationCredentials:
    @pytest.fixture
    def isolated_config(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: cfg_dir)
        monkeypatch.setattr(
            "vibetrack.config.config_path", lambda: cfg_dir / "config.json"
        )
        return cfg_dir

    def test_telegram_can_opt_into_config_credentials(
        self, tmp_path, monkeypatch, isolated_config
    ):
        from vibetrack.viewers.telegram import TelegramOutput

        monkeypatch.delenv("VIBETRACK_TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("VIBETRACK_TELEGRAM_CHAT_ID", raising=False)
        save_config(
            {"telegram": {"token": "cfg-token", "chat_id": "cfg-chat"}},
        )

        default = TelegramOutput(str(tmp_path / "project"))
        configured = TelegramOutput(
            str(tmp_path / "project"),
            use_config_credentials=True,
        )
        explicit = TelegramOutput(
            str(tmp_path / "project"),
            token="arg-token",
            chat_id="arg-chat",
            use_config_credentials=True,
        )

        assert default.token == ""
        assert default.chat_id == ""
        assert configured.token == "cfg-token"
        assert configured.chat_id == "cfg-chat"
        assert explicit.token == "arg-token"
        assert explicit.chat_id == "arg-chat"

    def test_slack_can_opt_into_config_credentials(
        self, tmp_path, monkeypatch, isolated_config
    ):
        from vibetrack.viewers.slack import SlackOutput

        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_CHANNEL", raising=False)
        save_config(
            {
                "slack": {
                    "webhook": "https://hooks.slack.test/config",
                    "bot_token": "xoxb-config",
                    "channel": "C123",
                }
            },
        )

        default = SlackOutput(str(tmp_path / "project"))
        configured = SlackOutput(
            str(tmp_path / "project"),
            use_config_credentials=True,
        )
        explicit = SlackOutput(
            str(tmp_path / "project"),
            webhook="https://hooks.slack.test/arg",
            bot_token="xoxb-arg",
            channel="C999",
            use_config_credentials=True,
        )

        assert default.webhook == ""
        assert default.bot_token == ""
        assert default.channel == ""
        assert configured.webhook == "https://hooks.slack.test/config"
        assert configured.bot_token == "xoxb-config"
        assert configured.channel == "C123"
        assert explicit.webhook == "https://hooks.slack.test/arg"
        assert explicit.bot_token == "xoxb-arg"
        assert explicit.channel == "C999"


class TestListenServer:
    @pytest.fixture
    def listen_app(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        app = _create_listen_app(str(tmp_path), token=None)
        return _ASGIClient(app), tmp_path

    @pytest.fixture
    def listen_app_with_token(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        app = _create_listen_app(str(tmp_path), token="secret123")
        return _ASGIClient(app)

    def test_log_scalars(self, listen_app):
        client, project_folder = listen_app
        resp = client.post(
            "/log",
            json={
                "experiment": "remote_run",
                "step": 0,
                "scalars": {"loss": 0.5, "acc": 0.9},
            },
        )
        assert resp.status_code == 200

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        assert any(e.name == "remote_run" for e in exps)
        exp = next(e for e in exps if e.name == "remote_run")
        tags = exp.scalar_tags()
        assert "loss" in tags and "acc" in tags
        reader.close()

    def test_log_texts(self, listen_app):
        client, project_folder = listen_app
        resp = client.post(
            "/log",
            json={
                "experiment": "text_run",
                "step": 0,
                "texts": {"note": "hello world"},
            },
        )
        assert resp.status_code == 200

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        exp = next(e for e in exps if e.name == "text_run")
        assert "note" in exp.text_tags()
        reader.close()

    def test_log_rejects_non_integer_step(self, listen_app):
        client, _ = listen_app
        resp = client.post(
            "/log",
            json={
                "experiment": "bad_step",
                "step": "<script>alert(1)</script>",
                "scalars": {"loss": 0.5},
            },
        )
        assert resp.status_code == 400

    def test_token_rejects_unauthorized(self, listen_app_with_token):
        resp = listen_app_with_token.post(
            "/log",
            json={
                "experiment": "x",
                "step": 0,
                "scalars": {},
            },
        )
        assert resp.status_code == 401

    def test_token_accepts_authorized(self, listen_app_with_token):
        resp = listen_app_with_token.post(
            "/log",
            json={"experiment": "x", "step": 0, "scalars": {"loss": 1.0}},
            headers={"Authorization": "Bearer secret123"},
        )
        assert resp.status_code == 200

    def test_media_upload_artifact(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        client = _ASGIClient(_create_listen_app(str(tmp_path), token=None))
        resp = client.post(
            "/media",
            data={
                "experiment": "media_run",
                "tag": "sample",
                "step": "0",
                "type": "artifact",
            },
            files={
                "file": (
                    "test.bin",
                    io.BytesIO(b"fake data"),
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 200

    def test_log_histogram(self, listen_app):
        client, project_folder = listen_app
        resp = client.post(
            "/log",
            json={
                "experiment": "hist_run",
                "step": 0,
                "histograms": {
                    "weights": {
                        "bin_edges": [-1.0, 0.0, 1.0],
                        "counts": [3.0, 7.0],
                    }
                },
            },
        )
        assert resp.status_code == 200

        reader = RunReader(str(project_folder))
        exp = next(e for e in reader.experiments() if e.name == "hist_run")
        assert "weights" in exp.histogram_tags()
        rows = exp.histograms("weights")
        assert rows and rows[0]["counts"] == [3.0, 7.0]
        reader.close()

    def test_hparams_endpoint(self, listen_app):
        client, project_folder = listen_app
        resp = client.post(
            "/hparams",
            json={
                "experiment": "hp_run",
                "hparams": {"lr": 1e-3, "batch_size": 32},
                "metrics": {"final_acc": 0.92},
            },
        )
        assert resp.status_code == 200

        reader = RunReader(str(project_folder))
        exp = next(e for e in reader.experiments() if e.name == "hp_run")
        assert exp.hparams() == {"lr": 1e-3, "batch_size": 32}
        assert "final_acc" in exp.scalar_tags()
        reader.close()

    def test_hparams_rejects_non_object(self, listen_app):
        client, _ = listen_app
        resp = client.post(
            "/hparams",
            json={"experiment": "bad", "hparams": "not-a-dict", "metrics": {}},
        )
        assert resp.status_code == 400


class TestRemoteOutput:
    """RemoteOutput adapter sends events to a stub server via .to('remote', ...)."""

    @pytest.fixture(autouse=True)
    def _isolate_central_db(self, tmp_path, monkeypatch):
        central_db = tmp_path / "_central" / "vibetrack.db"
        central_db.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)

    @pytest.fixture
    def stub_server(self):
        """A minimal threaded HTTPServer that records every request."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        recorded: list = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                recorded.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, *_a, **_kw):  # silence stderr noise
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            yield f"http://{host}:{port}", recorded
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_to_remote_dispatches_scalar(self, stub_server, tmp_path):
        url, recorded = stub_server
        with SummaryWriter(str(tmp_path / "run"), name="run") as w:
            w.to("remote", url=url, token="t1")
            w.add_scalar("loss", 0.25, 0)
        # Drain dispatch executor — close() already does this, just assert.
        log_posts = [r for r in recorded if r["path"] == "/log"]
        assert log_posts, f"expected at least one /log POST, got {recorded}"
        body = json.loads(log_posts[0]["body"])
        assert body["experiment"] == "run"
        assert body["scalars"] == {"loss": 0.25}
        assert log_posts[0]["headers"].get("Authorization") == "Bearer t1"

    def test_to_remote_dispatches_text_and_histogram(self, stub_server, tmp_path):
        import numpy as np

        url, recorded = stub_server
        with SummaryWriter(str(tmp_path / "r"), name="r") as w:
            w.to("remote", url=url)
            w.add_text("note", "hello", 0)
            w.add_histogram("weights", np.array([0.0, 0.5, 1.0, 1.0]), 0)

        log_posts = [r for r in recorded if r["path"] == "/log"]
        bodies = [json.loads(p["body"]) for p in log_posts]
        # Find one body that carries the text and one that carries the histogram.
        assert any(b.get("texts") == {"note": "hello"} for b in bodies)
        hist_body = next(b for b in bodies if b.get("histograms"))
        hist = hist_body["histograms"]["weights"]
        assert "bin_edges" in hist and "counts" in hist
        assert sum(hist["counts"]) == 4.0

    def test_to_remote_dispatches_media(self, stub_server, tmp_path):
        url, recorded = stub_server
        img = tmp_path / "frame.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        with SummaryWriter(str(tmp_path / "r2"), name="r2") as w:
            w.to("remote", url=url)
            w.add_image("samples", str(img), 0)

        media_posts = [r for r in recorded if r["path"] == "/media"]
        assert (
            media_posts
        ), f"expected /media POST, got paths={[r['path'] for r in recorded]}"
        assert (
            media_posts[0]["headers"]
            .get("Content-Type", "")
            .startswith("multipart/form-data")
        )
        assert b'name="experiment"' in media_posts[0]["body"]
        assert b"r2" in media_posts[0]["body"]
        assert b'name="type"' in media_posts[0]["body"]
        assert b"image" in media_posts[0]["body"]

    def test_to_remote_dispatches_hparams(self, stub_server, tmp_path):
        url, recorded = stub_server
        with SummaryWriter(str(tmp_path / "r3"), name="r3") as w:
            w.to("remote", url=url)
            w.add_hparams({"lr": 0.01}, {"final_loss": 0.1})

        hp_posts = [r for r in recorded if r["path"] == "/hparams"]
        assert (
            hp_posts
        ), f"expected /hparams POST, got paths={[r['path'] for r in recorded]}"
        body = json.loads(hp_posts[0]["body"])
        assert body["experiment"] == "r3"
        assert body["hparams"] == {"lr": 0.01}
        assert body["metrics"] == {"final_loss": 0.1}

    def test_to_remote_failure_does_not_raise(self, tmp_path):
        # Point at an unreachable port; .add_scalar must not raise.
        with SummaryWriter(str(tmp_path / "rf"), name="rf") as w:
            w.to("remote", url="http://127.0.0.1:1", timeout=0.5)
            w.add_scalar("loss", 0.5, 0)
        # No assertion on logs — just that the run finished cleanly.

    def test_remote_in_discovered_viewers(self):
        from vibetrack.viewers import discover_viewers, load_viewer
        from vibetrack.viewers.remote import RemoteOutput

        assert "remote" in discover_viewers()
        assert load_viewer("remote") is RemoteOutput


class TestUploadSecurity:
    @pytest.fixture
    def client(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        return _ASGIClient(_create_listen_app(str(tmp_path), token=None))

    def test_oversized_upload_rejected(self, client, monkeypatch):
        """Files larger than the configured limit must be rejected with 413."""
        monkeypatch.setattr("vibetrack.cli._MAX_UPLOAD_BYTES", 8)
        big = io.BytesIO(b"x" * 9)
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "artifact"},
            files={"file": ("big.bin", big, "application/octet-stream")},
        )
        assert resp.status_code == 413

    def test_disallowed_extension_for_image_rejected(self, client):
        """Uploading a .exe as an image must be rejected with 400."""
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "image"},
            files={
                "file": (
                    "evil.exe",
                    io.BytesIO(b"MZ\x90\x00"),
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 400

    def test_allowed_image_extension_accepted(self, client):
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "image"},
            files={
                "file": ("photo.png", io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "image/png")
            },
        )
        assert resp.status_code == 200

    def test_artifact_type_accepts_any_extension(self, client):
        """artifact type must accept arbitrary file extensions."""
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "artifact"},
            files={
                "file": ("model.pkl", io.BytesIO(b"data"), "application/octet-stream")
            },
        )
        assert resp.status_code == 200


class TestUploadHelpers:
    class _FakeUpload:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, _size):
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    def test_stream_upload_to_tempfile(self):
        from vibetrack.cli import _write_upload_to_tempfile

        upload = self._FakeUpload([b"abc", b"def"])
        path = asyncio.run(
            _write_upload_to_tempfile(upload, suffix=".bin", max_bytes=10, chunk_size=3)
        )
        try:
            assert Path(path).is_file()
            assert Path(path).read_bytes() == b"abcdef"
        finally:
            Path(path).unlink()

    def test_stream_upload_rejects_oversized_payload(self):
        from vibetrack.cli import _write_upload_to_tempfile

        upload = self._FakeUpload([b"abc", b"def"])
        with pytest.raises(ValueError, match="File too large"):
            asyncio.run(
                _write_upload_to_tempfile(
                    upload, suffix=".bin", max_bytes=5, chunk_size=3
                )
            )


class TestWebSerialization:
    def test_json_for_html_escapes_script_terminator(self):
        from vibetrack.viewers.web import _json_for_html

        payload = [{"value": "</script><script>alert(1)</script>"}]
        serialized = _json_for_html(payload)
        assert "</script>" not in serialized
        assert "\\u003c/script>" in serialized


class TestWebSecurity:
    @pytest.fixture
    def make_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from vibetrack.viewers import web
        from vibetrack.viewers.web import WebOutput

        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: cfg_dir)
        monkeypatch.setattr(
            "vibetrack.config.config_path", lambda: cfg_dir / "config.json"
        )
        web._mutation_rate_state.clear()
        project_folder = tmp_path / "project"
        with SummaryWriter(
            str(project_folder / "run_a"),
            name="run_a",
            project_folder=str(project_folder),
        ) as w:
            w.add_scalar("loss", 0.5, 0)
        outputs = []

        def _make(host="127.0.0.1", token=None):
            out = WebOutput(str(project_folder))
            outputs.append(out)
            return out._build_app(None, host, token=token)

        yield _make
        for out in outputs:
            out.close()

    def test_render_never_resubstitutes_placeholders(self):
        import re

        from vibetrack.viewers.web import _render

        data = [{"name": "__PROJECT_JSON__", "note": "__PROJECTS_JSON__ <!--<script>"}]
        html = _render(data, "-alert(1)-", ["-alert(2)-"])
        assert "<!--<script>" not in html
        assigned = dict(re.findall(r"window\.(VT_\w+) = (.*);\n", html))
        assert json.loads(assigned["VT_DATA"]) == data
        assert json.loads(assigned["VT_PROJECT"]) == "-alert(1)-"
        assert json.loads(assigned["VT_PROJECTS"]) == ["-alert(2)-"]

    def test_ingest_rejects_cross_origin_requests(self, make_app):
        client = _ASGIClient(make_app())
        body = json.dumps({"experiment": "e", "scalars": {"loss": 1.0}})
        forged = client.post(
            "/project/listen/log",
            content=body,
            headers={"Origin": "http://evil.example", "Content-Type": "text/plain"},
        )
        assert forged.status_code == 403
        forged_media = client.post(
            "/project/listen/media",
            headers={"Origin": "http://evil.example"},
            data={"experiment": "e", "type": "artifact"},
            files={"file": ("a.txt", b"x", "text/plain")},
        )
        assert forged_media.status_code == 403
        # Remote loggers send no Origin and must keep working.
        assert client.post("/project/listen/log", content=body).status_code == 200

    def test_listen_app_rejects_browser_requests(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        client = _ASGIClient(_create_listen_app(str(tmp_path), token=None))
        payload = {"experiment": "e", "scalars": {"loss": 1.0}}
        forged = client.post(
            "/log", json=payload, headers={"Origin": "http://evil.example"}
        )
        assert forged.status_code == 403
        assert client.post("/log", json=payload).status_code == 200

    def test_non_ascii_authorization_is_401_not_500(self, make_app):
        client = _ASGIClient(make_app(token="s3cret"))
        resp = client.get(
            "/api/projects", headers={"Authorization": "Bearer café".encode("latin-1")}
        )
        assert resp.status_code == 401

    def test_mcp_mount_requires_token(self, make_app):
        from fastapi.testclient import TestClient

        app = make_app(token="s3cret")
        if "/vibetrack_mcp" not in [getattr(r, "path", None) for r in app.routes]:
            pytest.skip("MCP sub-app not available")
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
        headers = {"Accept": "application/json, text/event-stream"}
        with TestClient(app, base_url="http://127.0.0.1:6116") as client:
            anon = client.post("/vibetrack_mcp/mcp", json=init, headers=headers)
            assert anon.status_code == 401
            authed = client.post(
                "/vibetrack_mcp/mcp",
                json=init,
                headers={**headers, "Authorization": "Bearer s3cret"},
            )
            assert authed.status_code == 200

    @pytest.mark.parametrize(
        "route,scope",
        [("/api/config", None), ("/api/config/project", "project")],
    )
    def test_config_api_hides_and_preserves_credentials(self, make_app, route, scope):
        """Credentials stored in either scope survive a UI round-trip.

        Both routes read/write the "project" scope here (the project folder's
        name); ``scope`` is only where the credentials were stored.
        """
        from vibetrack.config import config_path, load_config

        creds = {
            "telegram": {"token": "123:SECRET", "chat_id": "42"},
            "slack": {"bot_token": "xoxb-SECRET"},
        }
        save_config(creds, project=scope)
        client = _ASGIClient(make_app())

        loaded = client.get(route)
        assert loaded.status_code == 200
        assert "SECRET" not in loaded.text

        # The UI posts the whole (blanked) config back; an attacker would
        # try to redirect notifications. Neither may touch stored credentials.
        body = loaded.json()
        body["web"]["theme"] = "orange"
        body["telegram"] = {"token": "evil", "chat_id": "attacker"}
        saved = client.post(route, json=body)
        assert saved.status_code == 200
        assert "SECRET" not in saved.text

        cfg = load_config(project="project")
        assert cfg["web"]["theme"] == "orange"
        assert cfg["telegram"] == creds["telegram"]
        assert cfg["slack"] == creds["slack"]
        assert "evil" not in config_path().read_text()

        assert client.post(route, json=[1, 2]).status_code == 400

    def test_loopback_bind_rejects_foreign_host_header(self, make_app):
        client = _ASGIClient(make_app())
        rebound = client.get("/api/projects", headers={"Host": "attacker.example:6116"})
        assert rebound.status_code == 403
        rebound_static = client.get(
            "/static/js/core.js", headers={"Host": "attacker.example"}
        )
        assert rebound_static.status_code == 403
        for good in ("localhost:6116", "127.0.0.1", "[::1]:6116"):
            assert client.get("/api/projects", headers={"Host": good}).status_code == 200

    def test_lan_bind_does_not_restrict_host_header(self, make_app):
        client = _ASGIClient(make_app(host="192.168.1.10"))
        resp = client.get("/api/projects", headers={"Host": "trainbox.lan:6116"})
        assert resp.status_code == 200


class TestWebHardening:
    """Central-mode server behaviour: ingest layout, auth cookie, origins."""

    @pytest.fixture
    def central(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from vibetrack.viewers import web
        from vibetrack.viewers.web import WebOutput

        central_db = tmp_path / "home" / ".vibetrack" / "vibetrack.db"
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)
        cfg = tmp_path / "cfg"
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: cfg)
        monkeypatch.setattr("vibetrack.config.config_path", lambda: cfg / "config.json")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        web._mutation_rate_state.clear()
        outputs = []

        def _make(host="127.0.0.1", **kwargs):
            out = WebOutput()
            outputs.append(out)
            return _ASGIClient(out._build_app(None, host, **kwargs))

        yield _make, work
        for out in outputs:
            out.close()

    @staticmethod
    def _upload(client, project, experiment, payload, tag="img"):
        return client.post(
            f"/{project}/listen/media",
            data={"experiment": experiment, "tag": tag, "step": "0", "type": "image"},
            files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + payload, "image/png")},
        )

    def test_dashboard_picks_up_database_created_after_start(self, central):
        make, work = central
        client = make()
        assert client.get("/api/projects").json() == []
        with SummaryWriter(str(work / "late"), project="late", system_metrics_interval=0) as w:
            w.add_scalar("loss", 1.0, 0)
        assert client.get("/api/projects").json() == ["late"]
        assert client.get("/late").status_code == 200

    def test_same_run_name_in_two_projects_keeps_separate_media(self, central):
        make, work = central
        client = make()
        assert self._upload(client, "projA", "run", b"A").status_code == 200
        assert self._upload(client, "projB", "run", b"B").status_code == 200
        path_a = client.get("/api/data/projA").json()[0]["images"]["img"][0]["path"]
        path_b = client.get("/api/data/projB").json()[0]["images"]["img"][0]["path"]
        assert path_a != path_b
        assert client.get("/media", params={"path": path_a}).content.endswith(b"A")
        client.request("DELETE", "/api/experiment", json={"name": "run", "project": "projA"})
        assert Path(path_b).exists()

    def test_existing_run_keeps_its_log_dir(self, central):
        make, work = central
        legacy = work / "run"  # pre-fix layout: cwd/<experiment>
        with SummaryWriter(str(legacy), name="run", project="old", system_metrics_interval=0) as w:
            w.add_scalar("loss", 1.0, 0)
        client = make()
        assert self._upload(client, "old", "run", b"C").status_code == 200
        exp = client.get("/api/data/old").json()[0]
        assert exp["log_dir"] == str(legacy.resolve())
        assert exp["images"]["img"][0]["path"].startswith(str(legacy.resolve()))

    def test_delete_keeps_media_shared_with_another_project(self, central):
        make, work = central
        shared = work / "shared"
        for project, payload in (("a", b"A"), ("b", b"B")):
            src = work / f"{project}.png"
            src.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)
            with SummaryWriter(
                str(shared), name="run", project=project, system_metrics_interval=0
            ) as w:
                w.add_image("img", str(src), 0)
        client = make()
        resp = client.request("DELETE", "/api/experiment", json={"name": "run", "project": "a"})
        assert resp.status_code == 200
        assert (shared / "media" / "img" / "0.png").exists()

    def test_token_login_sets_cookie_and_strips_query(self, central):
        make, _ = central
        client = make(token="s3cret")
        assert client.get("/").status_code == 401
        assert client.get("/?token=nope").status_code == 401
        resp = client.get("/somewhere?token=s3cret&tab=text")
        assert resp.status_code == 303
        assert resp.headers["location"] == "/somewhere?tab=text"
        cookie = resp.headers["set-cookie"]
        assert "httponly" in cookie.lower() and "samesite=lax" in cookie.lower()
        assert "s3cret" not in cookie
        pair = cookie.split(";", 1)[0]
        assert client.get("/api/projects", headers={"Cookie": pair}).status_code == 200
        name = pair.split("=", 1)[0]
        assert client.get("/api/projects", headers={"Cookie": f"{name}=x"}).status_code == 401

    def test_origin_must_match_host_including_port(self, central):
        make, _ = central
        body = {"old_name": "missing", "new_name": "x", "project": "p"}
        lan = make(host="192.168.1.10")
        lan.post("/p/listen/log", json={"experiment": "seed", "scalars": {"x": 1}})
        same = lan.post("/api/rename", json=body,
                        headers={"Origin": "http://localhost:6116", "Host": "localhost:6116"})
        assert same.status_code == 404  # passed the CSRF gate
        local = make()
        other_port = local.post("/api/rename", json=body,
                                headers={"Origin": "http://127.0.0.1:3000"})
        assert other_port.status_code == 403

    def test_allow_host_for_reverse_proxy(self, central):
        make, _ = central
        client = make(allowed_hosts=["vt.example.com"])
        client.post("/p/listen/log", json={"experiment": "seed", "scalars": {"x": 1}})
        assert client.get("/api/projects", headers={"Host": "vt.example.com"}).status_code == 200
        assert client.get("/api/projects", headers={"Host": "other.example"}).status_code == 403
        proxied = client.post(
            "/api/rename",
            json={"old_name": "missing", "new_name": "x", "project": "p"},
            headers={"Origin": "https://vt.example.com"},
        )
        assert proxied.status_code == 404

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            "[1, 2]",
            '{"scalars": [1]}',
            '{"scalars": {"loss": "abc"}}',
            '{"texts": {"note": 5}}',
            '{"step": 1e400}',
            '{"step": 1180591620717411303424}',
            '{"experiment": "."}',
        ],
    )
    def test_malformed_ingest_is_400(self, central, content):
        make, _ = central
        resp = make().post(
            "/proj/listen/log", content=content, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 400

    def test_malformed_mutation_bodies_are_400(self, central):
        make, _ = central
        client = make()
        assert client.post("/api/rename", json=[1]).status_code == 400
        assert client.post("/api/rename", json={"old_name": 1, "new_name": "x"}).status_code == 400
        assert client.post("/api/move-logdir", json={"old": None, "new": "/x"}).status_code == 400
        huge = client.post(
            "/proj/listen/media",
            data={"experiment": "e", "step": str(2**70), "type": "artifact"},
            files={"file": ("a.txt", b"x", "text/plain")},
        )
        assert huge.status_code == 400

    def test_move_parent_folder_of_runs(self, central):
        make, work = central
        client = make()
        for name in ("a", "b"):
            client.post("/proj/listen/log", json={"experiment": name, "scalars": {"x": 1}})
        target = work / "moved"
        resp = client.post("/api/move-logdir", json={"old": str(work / "proj"), "new": str(target)})
        assert resp.status_code == 200, resp.text
        dirs = sorted(e["log_dir"] for e in client.get("/api/data/proj").json())
        assert dirs == [str((target / "a").resolve()), str((target / "b").resolve())]

    def test_move_refuses_unsafe_requests(self, central):
        make, work = central
        client = make()
        with SummaryWriter(str(work / "home" / "run1"), project="h", system_metrics_interval=0) as w:
            w.add_scalar("x", 1.0, 0)
        (work / "home" / "notes.txt").write_text("not vibetrack's")
        run = str(work / "home" / "run1")

        def move(old, new):
            return client.post("/api/move-logdir", json={"old": old, "new": new}).status_code

        assert move(str(work / "home"), str(work / "elsewhere")) == 403
        assert move(run, "relative/dest") == 400
        assert move(run, str(work / "no" / "such" / "parent" / "run1")) == 400
        assert move(run, run + "/inside") == 400
        assert (work / "home" / "run1").is_dir()

    def test_move_refuses_folder_with_open_database(self, populated_project):
        from vibetrack.viewers import web
        from vibetrack.viewers.web import WebOutput

        web._mutation_rate_state.clear()
        out = WebOutput(populated_project)
        client = _ASGIClient(out._build_app(None, "127.0.0.1"))
        resp = client.post(
            "/api/move-logdir",
            json={"old": populated_project, "new": populated_project + "_moved"},
        )
        out.close()
        assert resp.status_code == 409
        assert Path(populated_project, "vibetrack.db").exists()

    def test_listen_app_validates_bodies_and_project(self, tmp_path):
        from vibetrack.cli import _create_listen_app

        client = _ASGIClient(_create_listen_app(str(tmp_path), token=None))
        assert client.post("/log", json=[1]).status_code == 400
        assert client.post("/log", json={"histograms": {"h": {"bin_edges": ["a"], "counts": [1]}}}).status_code == 400
        assert client.post("/hparams", json={"project": "..", "hparams": {}}).status_code == 400

    def test_rate_limiter_forgets_idle_clients(self, monkeypatch):
        import types

        from vibetrack.viewers import web

        web._mutation_rate_state.clear()
        now = [1000.0]
        monkeypatch.setattr(web, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
        for i in range(web._MUTATION_RATE_MAX_TRACKED):
            web._rate_limit_mutation(f"10.0.{i // 256}.{i % 256}")
        now[0] += 5
        assert web._rate_limit_mutation("192.0.2.1")
        assert list(web._mutation_rate_state) == ["192.0.2.1"]


class TestNotificationFormatting:
    @staticmethod
    def _event(**kwargs):
        from vibetrack.viewers.event import LogEvent

        base = dict(kind="text", tag="t", step=1, value="", walltime=0.0,
                    run_name="resnet_50", project=None)
        base.update(kwargs)
        return LogEvent(**base)

    def test_telegram_sends_escaped_html(self, monkeypatch):
        from vibetrack.viewers import telegram

        sent = []

        class FakeBot:
            def __init__(self, token):
                pass

            async def send_message(self, **kwargs):
                sent.append(kwargs)

        monkeypatch.setattr(telegram, "_import_telegram_bot", lambda: FakeBot)
        telegram.TelegramOutput(token="t", chat_id="c").send([
            self._event(tag="train_loss<x>", value="lr_decay *on* & <b>"),
            self._event(kind="hparams", tag="hparams", value={"lr_sched": "cos_<1>"}),
        ])
        assert sent and sent[0]["parse_mode"] == "HTML"
        text = sent[0]["text"]
        assert "<b>resnet_50</b>" in text
        assert "train_loss&lt;x&gt;" in text
        assert "lr_decay *on* &amp; &lt;b&gt;" in text
        assert "lr_sched=cos_&lt;1&gt;" in text

    def test_telegram_clip_never_splits_entities(self):
        from vibetrack.viewers.telegram import _html_clip, _join_within

        assert _html_clip("ab&cd", 4) == "ab…"
        assert _html_clip("a<b", 10) == "a&lt;b"
        assert _join_within(["<b>x</b>", "y" * 10], limit=12) == "<b>x</b>\n…"

    def test_slack_escapes_mentions_and_links(self):
        from vibetrack.viewers.slack import SlackOutput

        summary = SlackOutput()._build_summary([
            self._event(value="<!channel> see <http://evil|docs> & more"),
        ])
        assert "<!channel>" not in summary and "<http" not in summary
        assert "&lt;!channel&gt;" in summary and "&amp; more" in summary


class TestWallTimesInPayload:
    def test_scalar_data_includes_wall_times(self, populated_project):
        from vibetrack.viewers.web import _build_data, WebOutput

        output = WebOutput(populated_project)
        exps = output._resolve_experiments()
        data = _build_data(exps)
        assert len(data) > 0
        exp = data[0]
        for tag, scalar_data in exp["scalars"].items():
            assert "wall_times" in scalar_data
            assert len(scalar_data["wall_times"]) == len(scalar_data["steps"])
            assert all(isinstance(t, float) for t in scalar_data["wall_times"])
        output.close()


class TestMediaPathTraversal:
    @pytest.fixture
    def web_app(self, tmp_path):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        project_folder = tmp_path / "project"
        src = project_folder / "seed.png"
        project_folder.mkdir()
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        with SummaryWriter(
            str(project_folder / "run_a"),
            name="run_a",
            project_folder=str(project_folder),
        ) as writer:
            writer.add_image("samples", str(src), 0)

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput(str(project_folder))
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return _ASGIClient(captured["app"]), project_folder

    def test_dotdot_traversal_rejected(self, web_app):
        client, _project_folder = web_app
        resp = _call_app_route(client._app, "/media", path="/etc/passwd")
        assert resp.status_code in (403, 404)

    def test_nested_traversal_rejected(self, web_app):
        client, project_folder = web_app
        attempted = (
            project_folder
            / "run_a"
            / "media"
            / "samples"
            / ".."
            / ".."
            / ".."
            / "etc"
            / "shadow"
        )
        resp = _call_app_route(client._app, "/media", path=str(attempted))
        assert resp.status_code in (403, 404)

    def test_valid_path_resolves_normally(self, web_app):
        client, project_folder = web_app
        media_path = project_folder / "run_a" / "media" / "samples" / "0.png"
        resp = _call_app_route(client._app, "/media", path=str(media_path))
        assert resp.status_code == 200


class TestMediaThumbnails:
    @pytest.fixture
    def thumb_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        Image = pytest.importorskip("PIL.Image")
        import uvicorn

        from vibetrack.viewers import web
        from vibetrack.viewers.web import WebOutput

        monkeypatch.setattr(web, "_THUMB_DIR", tmp_path / "thumbs")
        project_folder = tmp_path / "project"
        project_folder.mkdir()
        sources = {
            "big": Image.new("RGB", (3000, 750), "red"),
            "small": Image.new("RGB", (200, 100), "blue"),
            "alpha": Image.new("RGBA", (2000, 1000), (0, 255, 0, 64)),
        }
        with SummaryWriter(
            str(project_folder / "run_a"), name="run_a", project_folder=str(project_folder)
        ) as writer:
            for tag, img in sources.items():
                src = tmp_path / f"{tag}.png"
                img.save(src)
                writer.add_image(tag, str(src), 0)
            (tmp_path / "broken.mp4").write_bytes(b"not a video")
            writer.add_video("broken", str(tmp_path / "broken.mp4"), 0)

        captured: dict = {}
        output = WebOutput(str(project_folder))
        with mock.patch.object(uvicorn, "run", side_effect=lambda app, **kw: captured.update(app=app)):
            output._serve_uvicorn(None, "127.0.0.1", 6006)
        media = project_folder / "run_a" / "media"
        return captured["app"], media, Image

    def test_large_image_is_downscaled_to_bucket(self, thumb_app):
        app, media, Image = thumb_app
        resp = _call_app_route(app, "/media/thumb", path=str(media / "big" / "0.png"), w=500)
        assert resp.status_code == 200
        with Image.open(resp.path) as thumb:
            assert thumb.format == "JPEG"
            assert thumb.size == (640, 160)

    def test_small_image_served_as_original(self, thumb_app):
        app, media, _Image = thumb_app
        original = media / "small" / "0.png"
        resp = _call_app_route(app, "/media/thumb", path=str(original), w=640)
        assert Path(resp.path) == original

    def test_alpha_is_preserved(self, thumb_app):
        app, media, Image = thumb_app
        resp = _call_app_route(app, "/media/thumb", path=str(media / "alpha" / "0.png"), w=320)
        with Image.open(resp.path) as thumb:
            assert thumb.format == "PNG" and thumb.mode == "RGBA"
            assert thumb.size == (320, 160)

    def test_cache_reused_until_source_changes(self, thumb_app):
        app, media, Image = thumb_app
        src = media / "big" / "0.png"
        first = _call_app_route(app, "/media/thumb", path=str(src), w=640).path
        with mock.patch("PIL.Image.open", side_effect=AssertionError("cache miss")):
            assert _call_app_route(app, "/media/thumb", path=str(src), w=640).path == first
        Image.new("RGB", (2000, 1000), "green").save(src)
        changed = _call_app_route(app, "/media/thumb", path=str(src), w=640).path
        assert changed != first
        with Image.open(changed) as thumb:
            assert thumb.size == (640, 320)

    def test_rejects_paths_outside_media_roots(self, thumb_app, tmp_path):
        app, _media, _Image = thumb_app
        outside = tmp_path / "big.png"
        assert _call_app_route(app, "/media/thumb", path=str(outside)).status_code == 404
        assert _call_app_route(app, "/media/thumb", path="/etc/passwd").status_code == 404

    def test_undecodable_video_reports_unavailable(self, thumb_app):
        app, media, _Image = thumb_app
        resp = _call_app_route(app, "/media/thumb", path=str(media / "broken" / "0.mp4"))
        assert resp.status_code == 422

    def test_video_poster_from_first_frame(self, thumb_app, tmp_path):
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        from vibetrack.viewers.web import _media_thumbnail

        _app, _media, Image = thumb_app
        clip = tmp_path / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=5",
             "-t", "1", "-pix_fmt", "yuv420p", "-y", str(clip)], check=True)
        with Image.open(_media_thumbnail(str(clip), 640)) as poster:
            assert poster.format == "JPEG" and poster.size == (640, 360)


class TestWebProjectRouting:
    @pytest.fixture
    def central_web_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        config_root = tmp_path / "cfg"
        db = Database(central_db)
        alpha_id = db.create_experiment(
            "run_a",
            project="alpha",
            log_dir=str(tmp_path / "alpha" / "runs" / "run_a"),
        )
        beta_id = db.create_experiment(
            "run_b",
            project="beta",
            log_dir=str(tmp_path / "beta" / "runs" / "run_b"),
        )
        db.add_scalar(alpha_id, "loss", 0.5, 0)
        db.add_scalar(beta_id, "loss", 0.7, 0)
        db.close()

        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: config_root)
        monkeypatch.setattr(
            "vibetrack.config.config_path", lambda: config_root / "config.json"
        )

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput()
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return _ASGIClient(captured["app"])

    def test_root_redirects_to_project(self, central_web_app):
        resp = _call_app_route(central_web_app._app, "/")
        assert resp.status_code == 200
        # Root page auto-redirects via JS to last active or first project
        body = resp.body.decode()
        assert '"alpha"' in body
        assert '"beta"' in body
        assert "localStorage" in body

    def test_project_route_uses_project_scoped_api(self, central_web_app):
        resp = _call_app_route(central_web_app._app, "/{project}", project="alpha")
        assert resp.status_code == 200
        body = resp.body.decode()
        # Project is injected into the template as window.VT_PROJECT; the JS
        # apiUrl() helper suffixes it onto /api/* calls at runtime.
        assert 'window.VT_PROJECT = "alpha"' in body
        assert "run_a" in body
        assert "run_b" not in body

    def test_project_api_returns_only_selected_project(self, central_web_app):
        data = _call_app_route(
            central_web_app._app, "/api/data/{project}", project="alpha"
        )
        assert [item["name"] for item in data] == ["run_a"]

    def test_project_settings_are_isolated(self, central_web_app):
        alpha = central_web_app.post(
            "/api/config/alpha", json={"web": {"theme": "light"}}
        )
        beta = central_web_app.post(
            "/api/config/beta", json={"web": {"theme": "orange"}}
        )
        assert alpha.status_code == 200
        assert beta.status_code == 200
        assert (
            _call_app_route(
                central_web_app._app, "/api/config/{project}", project="alpha"
            )["web"]["theme"]
            == "light"
        )
        assert (
            _call_app_route(
                central_web_app._app, "/api/config/{project}", project="beta"
            )["web"]["theme"]
            == "orange"
        )

    @pytest.fixture
    def movable_web_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        config_root = tmp_path / "cfg"
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: config_root)
        monkeypatch.setattr(
            "vibetrack.config.config_path", lambda: config_root / "config.json"
        )

        old_dir = tmp_path / "alpha" / "runs" / "run_a"
        with SummaryWriter(
            str(old_dir),
            project="alpha",
            name="run_a",
            system_metrics_interval=0,
        ) as writer:
            writer.add_scalar("loss", 0.5, 0)

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput()
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return (
            _ASGIClient(captured["app"]),
            central_db,
            old_dir,
            tmp_path / "alpha" / "runs" / "run_a_moved",
        )

    def test_move_logdir_updates_database_and_api(self, movable_web_app):
        client, central_db, old_dir, new_dir = movable_web_app
        assert old_dir.exists()
        assert not new_dir.exists()

        resp = client.post(
            "/api/move-logdir",
            json={"old": str(old_dir), "new": str(new_dir)},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert not old_dir.exists()
        assert new_dir.exists()
        db = Database(central_db)
        exp = db.get_experiment_by_name("run_a", project="alpha")
        assert exp is not None
        assert exp["log_dir"] == str(new_dir)
        db.close()

    @pytest.fixture
    def deletable_central_web_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        config_root = tmp_path / "cfg"
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.config.config_dir", lambda: config_root)
        monkeypatch.setattr(
            "vibetrack.config.config_path", lambda: config_root / "config.json"
        )

        seed = tmp_path / "seed.bin"
        seed.write_bytes(b"checkpoint")

        alpha_run = tmp_path / "alpha" / "runs" / "run_a"
        with SummaryWriter(
            str(alpha_run),
            project="alpha",
            name="run_a",
            system_metrics_interval=0,
        ) as writer:
            writer.add_scalar("loss", 0.5, 0)
            writer.add_artifact("checkpoint", str(seed), 0)

        beta_run = tmp_path / "beta" / "runs" / "run_b"
        with SummaryWriter(
            str(beta_run),
            project="beta",
            name="run_b",
            system_metrics_interval=0,
        ) as writer:
            writer.add_scalar("loss", 0.7, 0)
            writer.add_artifact("checkpoint", str(seed), 0)

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput()
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return (
            _ASGIClient(captured["app"]),
            central_db,
            alpha_run / "media",
            beta_run / "media",
        )

    def test_delete_project_removes_rows_and_artifacts(self, deletable_central_web_app):
        client, central_db, alpha_media, beta_media = deletable_central_web_app
        assert alpha_media.exists()
        assert beta_media.exists()

        resp = _call_app_route(
            client._app, "/api/project/{project}", method="DELETE", project="alpha"
        )
        assert resp.status_code == 200
        assert resp.body and b'"ok":true' in resp.body

        db = Database(central_db)
        assert db.list_experiments(project="alpha") == []
        remaining = {row["project"] for row in db.list_experiments()}
        assert remaining == {"beta"}
        db.close()

        assert not alpha_media.exists()
        assert beta_media.exists()
        assert (
            _call_app_route(
                client._app, "/api/data/{project}", project="alpha"
            ).status_code
            == 404
        )


class TestUnifiedServer:
    """Test MCP mount and listen routes in the unified web app."""

    @pytest.fixture
    def unified_app(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        project_folder = tmp_path / "project"
        for name in ["run_a"]:
            run_dir = project_folder / name
            with SummaryWriter(
                str(run_dir), name=name, project_folder=str(project_folder)
            ) as w:
                for i in range(5):
                    w.add_scalar("loss", 1.0 / (i + 1), i)

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput(str(project_folder))
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return _ASGIClient(captured["app"]), project_folder

    @pytest.mark.skipif(
        importlib.util.find_spec("mcp") is None,
        reason="MCP optional dependency not installed",
    )
    def test_mcp_mounted(self, unified_app):
        """MCP sub-app should be mounted at /vibetrack_mcp."""
        client, _ = unified_app
        app = client._app
        mount_paths = [getattr(route, "path", None) for route in app.routes]
        assert "/vibetrack_mcp" in mount_paths

    def test_listen_log(self, unified_app):
        """POST /{project}/listen/log should accept scalar data."""
        client, project_folder = unified_app
        project_name = project_folder.name
        resp = client.post(
            f"/{project_name}/listen/log",
            json={
                "experiment": "remote_run",
                "step": 0,
                "scalars": {"loss": 0.5, "acc": 0.9},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_listen_log_with_token(self, tmp_path, monkeypatch):
        """Listen routes should enforce token auth when configured."""
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn

        from vibetrack.viewers.web import WebOutput

        project_folder = tmp_path / "project"
        project_folder.mkdir()

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput(str(project_folder))
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006, token="secret123")

        client = _ASGIClient(captured["app"])
        project_name = project_folder.name

        # Without token -> 401
        resp = client.post(
            f"/{project_name}/listen/log",
            json={"experiment": "x", "step": 0, "scalars": {}},
        )
        assert resp.status_code == 401

        # With token -> 200
        resp = client.post(
            f"/{project_name}/listen/log",
            json={"experiment": "x", "step": 0, "scalars": {"loss": 1.0}},
            headers={"Authorization": "Bearer secret123"},
        )
        assert resp.status_code == 200

    def test_listen_media_upload(self, unified_app):
        """POST /{project}/listen/media should accept file uploads."""
        client, project_folder = unified_app
        project_name = project_folder.name
        resp = client.post(
            f"/{project_name}/listen/media",
            data={
                "experiment": "media_run",
                "tag": "sample",
                "step": "0",
                "type": "artifact",
            },
            files={
                "file": (
                    "test.bin",
                    io.BytesIO(b"fake data"),
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 200

    def test_listen_log_rejects_path_traversal_experiment(self, unified_app):
        """Project-scoped listen routes must not write outside project_folder."""
        client, project_folder = unified_app
        project_name = project_folder.name
        resp = client.post(
            f"/{project_name}/listen/log",
            json={
                "experiment": "../escape",
                "step": 0,
                "scalars": {"loss": 0.5},
            },
        )
        assert resp.status_code == 400
        assert not (project_folder.parent / "escape").exists()

    def test_listen_log_rejects_non_integer_step(self, unified_app):
        """Remote JSON steps must be numeric before entering dashboard payloads."""
        client, project_folder = unified_app
        project_name = project_folder.name
        resp = client.post(
            f"/{project_name}/listen/log",
            json={
                "experiment": "remote_run",
                "step": "<img src=x onerror=alert(1)>",
                "scalars": {"loss": 0.5},
            },
        )
        assert resp.status_code == 400

    def test_listen_media_rejects_dangerous_artifact_extension(self, unified_app):
        """Remote artifact uploads should not create executable same-origin files."""
        client, project_folder = unified_app
        project_name = project_folder.name
        resp = client.post(
            f"/{project_name}/listen/media",
            data={
                "experiment": "media_run",
                "tag": "sample",
                "step": "0",
                "type": "artifact",
            },
            files={
                "file": (
                    "payload.html",
                    io.BytesIO(b"<script>alert(1)</script>"),
                    "text/html",
                )
            },
        )
        assert resp.status_code == 400
