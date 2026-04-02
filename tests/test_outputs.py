"""Tests for CLI, viewer auto-discovery, listen server, and output backends."""

import asyncio
import io
import inspect
import logging
from pathlib import Path
from unittest import mock

import httpx
import pytest

from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.viewers.console import ConsoleOutput, _sparkline
from vibetrack.viewers.mcp import _suppress_streamable_http_startup_log
from vibetrack.writer import SummaryWriter


class _ASGIClient:
    def __init__(self, app):
        self._app = app

    def request(self, method: str, url: str, **kwargs):
        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
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
        with SummaryWriter(str(run_dir), name=name, project_folder=str(project_folder)) as w:
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
        """vibetrack with no args should default to the 'web' viewer (web+mcp+ingest on one port)."""
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

        with mock.patch("vibetrack.cli._start_listen_server") as mock_listen, \
             mock.patch("vibetrack.viewers.web.WebOutput.show"):
            main(["--project-folder", populated_project, "--viewer", "web", "--listen", "0.0.0.0:9009"])
            mock_listen.assert_called_once_with(populated_project, "0.0.0.0", 9009, None)

    def test_listen_with_token(self, populated_project):
        from vibetrack.cli import main

        with mock.patch("vibetrack.cli._start_listen_server") as mock_listen, \
             mock.patch("vibetrack.viewers.web.WebOutput.show"):
            main([
                "--project-folder", populated_project,
                "--viewer", "web",
                "--listen", "0.0.0.0:9009",
                "--token", "abc",
            ])
            mock_listen.assert_called_once_with(populated_project, "0.0.0.0", 9009, "abc")

    def test_migrate_command(self, tmp_path):
        from vibetrack.cli import main

        project_folder = tmp_path / "project"
        project_folder.mkdir()
        for name in ["legacy_a", "legacy_b"]:
            run_dir = project_folder / name
            run_dir.mkdir()
            db = Database(run_dir / "vibetrack.db")
            exp_id = db.create_experiment(name, project=project_folder.name, log_dir=str(run_dir))
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
        resp = client.post("/log", json={
            "experiment": "remote_run",
            "step": 0,
            "scalars": {"loss": 0.5, "acc": 0.9},
        })
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
        resp = client.post("/log", json={
            "experiment": "text_run",
            "step": 0,
            "texts": {"note": "hello world"},
        })
        assert resp.status_code == 200

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        exp = next(e for e in exps if e.name == "text_run")
        assert "note" in exp.text_tags()
        reader.close()

    def test_token_rejects_unauthorized(self, listen_app_with_token):
        resp = listen_app_with_token.post("/log", json={
            "experiment": "x", "step": 0, "scalars": {},
        })
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
            data={"experiment": "media_run", "tag": "sample", "step": "0", "type": "artifact"},
            files={"file": ("test.bin", io.BytesIO(b"fake data"), "application/octet-stream")},
        )
        assert resp.status_code == 200


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
            files={"file": ("evil.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_allowed_image_extension_accepted(self, client):
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "image"},
            files={"file": ("photo.png", io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "image/png")},
        )
        assert resp.status_code == 200

    def test_artifact_type_accepts_any_extension(self, client):
        """artifact type must accept arbitrary file extensions."""
        resp = client.post(
            "/media",
            data={"experiment": "e", "tag": "t", "step": "0", "type": "artifact"},
            files={"file": ("model.pkl", io.BytesIO(b"data"), "application/octet-stream")},
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
                _write_upload_to_tempfile(upload, suffix=".bin", max_bytes=5, chunk_size=3)
            )


class TestWebSerialization:
    def test_json_for_html_escapes_script_terminator(self):
        from vibetrack.viewers.web import _json_for_html

        payload = [{"value": "</script><script>alert(1)</script>"}]
        serialized = _json_for_html(payload)
        assert "</script>" not in serialized
        assert "\\u003c/script>" in serialized


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
        attempted = project_folder / "run_a" / "media" / "samples" / ".." / ".." / ".." / "etc" / "shadow"
        resp = _call_app_route(client._app, "/media", path=str(attempted))
        assert resp.status_code in (403, 404)

    def test_valid_path_resolves_normally(self, web_app):
        client, project_folder = web_app
        media_path = project_folder / "run_a" / "media" / "samples" / "0.png"
        resp = _call_app_route(client._app, "/media", path=str(media_path))
        assert resp.status_code == 200


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
        monkeypatch.setattr("vibetrack.config.config_path", lambda: config_root / "config.json")

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
        assert "/api/data/alpha" in body
        assert "/api/rename/alpha" in body
        assert "/api/config/alpha" in body
        assert "run_a" in body
        assert "run_b" not in body

    def test_project_api_returns_only_selected_project(self, central_web_app):
        data = _call_app_route(central_web_app._app, "/api/data/{project}", project="alpha")
        assert [item["name"] for item in data] == ["run_a"]

    def test_project_settings_are_isolated(self, central_web_app):
        alpha = central_web_app.post("/api/config/alpha", json={"web": {"theme": "light"}})
        beta = central_web_app.post("/api/config/beta", json={"web": {"theme": "orange"}})
        assert alpha.status_code == 200
        assert beta.status_code == 200
        assert _call_app_route(central_web_app._app, "/api/config/{project}", project="alpha")["web"]["theme"] == "light"
        assert _call_app_route(central_web_app._app, "/api/config/{project}", project="beta")["web"]["theme"] == "orange"

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
        monkeypatch.setattr("vibetrack.config.config_path", lambda: config_root / "config.json")

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
        monkeypatch.setattr("vibetrack.config.config_path", lambda: config_root / "config.json")

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

        return _ASGIClient(captured["app"]), central_db, alpha_run / "media", beta_run / "media"

    def test_delete_project_removes_rows_and_artifacts(self, deletable_central_web_app):
        client, central_db, alpha_media, beta_media = deletable_central_web_app
        assert alpha_media.exists()
        assert beta_media.exists()

        resp = _call_app_route(client._app, "/api/project/{project}", method="DELETE", project="alpha")
        assert resp.status_code == 200
        assert resp.body and b'"ok":true' in resp.body

        db = Database(central_db)
        assert db.list_experiments(project="alpha") == []
        remaining = {row["project"] for row in db.list_experiments()}
        assert remaining == {"beta"}
        db.close()

        assert not alpha_media.exists()
        assert beta_media.exists()
        assert _call_app_route(client._app, "/api/data/{project}", project="alpha").status_code == 404


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
            with SummaryWriter(str(run_dir), name=name, project_folder=str(project_folder)) as w:
                for i in range(5):
                    w.add_scalar("loss", 1.0 / (i + 1), i)

        captured: dict = {}

        def _capture(app, **kwargs):
            captured["app"] = app

        output = WebOutput(str(project_folder))
        with mock.patch.object(uvicorn, "run", side_effect=_capture):
            output._serve_uvicorn(None, "127.0.0.1", 6006)

        return _ASGIClient(captured["app"]), project_folder

    def test_mcp_mounted(self, unified_app):
        """MCP sub-app should be mounted at /vibetrack_mcp."""
        client, _ = unified_app
        app = client._app
        mount_paths = [
            getattr(route, "path", None) for route in app.routes
        ]
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
            data={"experiment": "media_run", "tag": "sample", "step": "0", "type": "artifact"},
            files={"file": ("test.bin", io.BytesIO(b"fake data"), "application/octet-stream")},
        )
        assert resp.status_code == 200
