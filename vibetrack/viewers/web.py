"""Standalone web UI with interactive Chart.js graphs and media viewers.

Requires the base ``vibetrack`` install. MCP mounting is enabled when the
optional MCP dependency is installed via ``vibetrack[all]`` on Python 3.10+.

The browser-facing assets live under ``viewers/web/``:

- ``index.html``  — HTML shell (placeholders for injected JSON)
- ``css/style.css``
- ``js/*.js``     — split into core/charts/pills/media/settings/main

They are served as static files at ``/static/*``; the index template has three
string placeholders (``__DATA_JSON__``, ``__PROJECT_JSON__``, ``__PROJECTS_JSON__``)
replaced at render time.
"""

import functools
import hmac
import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..compare import find_all_tags  # noqa: F401  (re-exported for tests)
from ..config import load_config, save_config
from ..reader import ExperimentReader, RunReader
from .base import BaseOutput

_WEB_DIR = Path(__file__).parent / "web"
_INDEX_PATH = _WEB_DIR / "index.html"

_SYSTEM_PREFIXES = ("system/", "gpu/")


@functools.lru_cache(maxsize=1)
def _load_template() -> str:
    return _INDEX_PATH.read_text(encoding="utf-8")


def _json_for_html(obj: Any) -> str:
    """Serialize *obj* to JSON safe for embedding inside ``<script>``."""
    return json.dumps(obj).replace("</", r"\u003c/")


def _render(
    data: list,
    project: Optional[str],
    projects: Sequence[str],
) -> str:
    """Render the index template with JSON-safe replacements."""
    return (
        _load_template()
        .replace("__DATA_JSON__", _json_for_html(data))
        .replace("__PROJECT_JSON__", _json_for_html(project))
        .replace("__PROJECTS_JSON__", _json_for_html(list(projects)))
    )


def _resolve_host(host: str) -> str:
    """Resolve ``0.0.0.0`` to the primary LAN IP; leave explicit addresses alone."""
    if host != "0.0.0.0":
        return host
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _resolve_media_path(log_dir: str, rel_path: str) -> str:
    """Resolve a relative media path to absolute using the experiment's log_dir."""
    if not rel_path:
        return rel_path
    if os.path.isabs(rel_path):
        return rel_path  # already absolute (legacy data)
    if log_dir:
        return str(Path(log_dir).resolve() / rel_path)
    return rel_path


def _scalar_series(exp: ExperimentReader, tag: str) -> Dict[str, list]:
    rows = exp.scalars(tag)
    return {
        "steps": [r["step"] for r in rows],
        "values": [r["value"] for r in rows],
        "wall_times": [r["wall_time"] for r in rows],
    }


def _serialize_experiment(exp: ExperimentReader) -> Dict[str, Any]:
    """Build the JSON-ready dict for a single experiment."""
    log_dir = exp.log_dir

    all_tags = exp.scalar_tags()
    regular_tags = [t for t in all_tags if not t.startswith(_SYSTEM_PREFIXES)]
    system_tags = [t for t in all_tags if t.startswith(_SYSTEM_PREFIXES)]
    scalars = {tag: _scalar_series(exp, tag) for tag in regular_tags}
    system_scalars = {tag: _scalar_series(exp, tag) for tag in system_tags}

    images = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in exp.images(tag)
        ]
        for tag in exp.image_tags()
    }
    audio_data = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in exp.audio(tag)
        ]
        for tag in exp.audio_tags()
    }
    video_data = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in exp.video(tag)
        ]
        for tag in exp.video_tags()
    }
    artifacts = {
        tag: [
            {
                "step": r["step"],
                "path": _resolve_media_path(log_dir, r["path"]),
                "metadata": r["metadata"],
            }
            for r in exp.artifacts(tag)
        ]
        for tag in exp.artifact_tags()
    }
    text_data = {
        tag: [{"step": r["step"], "value": r["value"]} for r in exp.texts(tag)]
        for tag in exp.text_tags()
    }
    histogram_data = {
        tag: [
            {"step": r["step"], "bins": r["bins"], "counts": r["counts"]}
            for r in exp.histograms(tag)
        ]
        for tag in exp.histogram_tags()
    }

    return {
        "name": exp.name,
        "project": exp.project,
        "log_dir": log_dir,
        "tags": regular_tags,
        "scalars": scalars,
        "system_tags": system_tags,
        "system_scalars": system_scalars,
        "image_tags": list(images),
        "images": images,
        "audio_tags": list(audio_data),
        "audio_data": audio_data,
        "video_tags": list(video_data),
        "video_data": video_data,
        "artifact_tags": list(artifacts),
        "artifacts": artifacts,
        "text_tags": list(text_data),
        "text_data": text_data,
        "histogram_tags": list(histogram_data),
        "histogram_data": histogram_data,
        "hparams": exp.hparams(),
    }


def _build_data(exps: Sequence[ExperimentReader]) -> list:
    return [_serialize_experiment(exp) for exp in exps]


def _experiments_for_project(
    db: Any,
    project: str,
    names: Optional[Sequence[str]] = None,
) -> List[ExperimentReader]:
    rows = db.list_experiments(project=project)
    if names is not None:
        allowed = set(names)
        rows = [r for r in rows if r["name"] in allowed]
    return [ExperimentReader(db, r["id"], r["name"], row=r) for r in rows]


def _get_media_roots(reader: RunReader, project: Optional[str] = None) -> List[str]:
    """Return normalized media root dirs from experiment log_dirs."""
    roots: set[str] = set()
    db = reader._db
    if db is None:
        return []
    for row in db.list_experiments(project=project):
        log_dir = row["log_dir"] if hasattr(row, "__getitem__") else ""
        if log_dir:
            media_dir = os.path.normpath(os.path.join(log_dir, "media"))
            roots.add(media_dir + os.sep)
    return list(roots)


def _cleanup_experiment_files(log_dir: str) -> None:
    """Remove media and the run directory if they are now empty."""
    if not log_dir:
        return
    shutil.rmtree(os.path.join(log_dir, "media"), ignore_errors=True)
    with suppress(OSError):
        os.rmdir(log_dir)


# ── Route handlers (extracted from closures) ────────────────────────────


def _handle_rename(
    db: Any,
    old_name: str,
    new_name: str,
    project: Optional[str],
) -> Tuple[dict, int]:
    """Rename an experiment; returns ``(body, status)``."""
    new_name = new_name.strip()
    if not new_name:
        return {"error": "name cannot be empty"}, 400
    if db is None:
        return {"error": "no database"}, 500
    for exp_row in db.list_experiments(project=project):
        if exp_row["name"] == old_name:
            if not db.rename_experiment(exp_row["id"], new_name):
                return {"error": "name already exists"}, 409
            return {"ok": True, "new_name": new_name}, 200
    return {"error": "experiment not found"}, 404


def _handle_delete_experiment(
    db: Any,
    name: str,
    project: Optional[str],
) -> Tuple[dict, int]:
    if not name:
        return {"error": "name required"}, 400
    if db is None:
        return {"error": "no database"}, 500
    row = db.get_experiment_by_name(name, project=project)
    if row is None:
        return {"error": "experiment not found"}, 404
    log_dir = db.delete_experiment(int(row["id"]))
    _cleanup_experiment_files(log_dir or "")
    return {"ok": True}, 200


def _handle_move_logdir(db: Any, old_dir: str, new_dir: str) -> Tuple[dict, int]:
    if not old_dir or not new_dir:
        return {"error": "old and new required"}, 400
    old_path = Path(old_dir).expanduser().resolve()
    new_path = Path(new_dir).expanduser().resolve()
    if old_path == new_path:
        return {"ok": True}, 200
    if db is None:
        return {"error": "no database"}, 500

    # Source must be EXACTLY a known experiment log_dir.
    known = [
        Path(r["log_dir"]).expanduser().resolve()
        for r in db.list_experiments()
        if r["log_dir"]
    ]
    if old_path not in known:
        return {"error": "not a known log directory"}, 403
    if not old_path.is_dir():
        return {"error": "source directory not found"}, 404
    if new_path.exists():
        return {"error": "destination already exists"}, 409

    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
    except OSError:
        return {"error": "move failed"}, 500

    # Update all experiments whose log_dir is under the moved directory.
    for row in db.list_experiments():
        ld = row["log_dir"]
        if not ld:
            continue
        ld_path = Path(ld).expanduser().resolve()
        try:
            rel = ld_path.relative_to(old_path)
        except ValueError:
            continue
        db.update_log_dir(int(row["id"]), str((new_path / rel).resolve()))
    return {"ok": True}, 200


class WebOutput(BaseOutput):
    """Serve an interactive web dashboard."""

    def show(self, **kwargs: Any) -> Any:
        """Launch uvicorn server."""
        host: str = kwargs.get("host", "127.0.0.1")
        port: int = kwargs.get("port", 6006)
        token: Optional[str] = kwargs.get("token")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        return self._serve_uvicorn(experiments, host, port, token=token)

    def _serve_uvicorn(
        self,
        experiments: Optional[Sequence[str]],
        host: str,
        port: int,
        token: Optional[str] = None,
    ) -> None:
        import uvicorn

        app = self._build_app(experiments, host, token=token)
        print(f"vibetrack web UI: http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")

    def start_in_thread(
        self,
        experiments: Optional[Sequence[str]] = None,
        host: str = "0.0.0.0",
        port: int = 6006,
        token: Optional[str] = None,
    ) -> "threading.Thread":
        """Start the web UI in a daemon thread and return immediately."""
        import threading
        import uvicorn

        app = self._build_app(experiments, host, token=token)
        print(f"vibetrack web UI: http://{_resolve_host(host)}:{port}")
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        return thread

    def _build_app(
        self,
        experiments: Optional[Sequence[str]],
        host: str,
        token: Optional[str] = None,
    ) -> Any:
        """Build and return the FastAPI app (without running it)."""
        from contextlib import asynccontextmanager

        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            HTTPException,
            Request,
            UploadFile,
        )
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles

        from ..cli import _ALLOWED_SUFFIXES, _write_upload_to_tempfile
        from ..writer import SummaryWriter

        logging.getLogger("uvicorn.error").addFilter(
            type(
                "_F",
                (logging.Filter,),
                {
                    "filter": lambda self, r: "Invalid HTTP request received"
                    not in r.getMessage()
                },
            )()
        )

        host = _resolve_host(host)

        # Build MCP sub-app early so we can wire its lifespan.
        _mcp_asgi = None
        _mcp_session_manager = None
        try:
            from .mcp import MCPOutput, _suppress_streamable_http_startup_log

            _mcp_output = MCPOutput(self.project_folder)
            _mcp_output._build_mcp()
            _mcp_asgi = _mcp_output._mcp.streamable_http_app()
            _mcp_session_manager = _mcp_output._mcp._session_manager
        except ImportError:
            pass

        @asynccontextmanager
        async def _lifespan(app):
            if _mcp_session_manager is not None:
                with _suppress_streamable_http_startup_log():
                    async with _mcp_session_manager.run():
                        yield
            else:
                yield

        app = FastAPI(title="vibetrack", lifespan=_lifespan)
        app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

        config_project = self.config_project()
        is_central = self.project_folder is None

        def _all_projects() -> List[str]:
            db = self._reader._db
            return [p for p in (db.list_projects() if db else []) if p]

        # ── Index / data ──────────────────────────────────────

        @app.get("/")
        def index() -> HTMLResponse:
            if is_central and self.project is None:
                projects = _all_projects()
                if not projects:
                    return HTMLResponse(
                        "<!DOCTYPE html><html><head>"
                        '<meta http-equiv="refresh" content="5">'
                        "</head><body>"
                        "<h1>vibetrack</h1><p>No projects yet.</p>"
                        "</body></html>"
                    )
                # Projects are returned most-recent-first; make that choice
                # browser-visible so stale localStorage cannot steal the
                # landing page from the current latest project.
                projects_json = _json_for_html(projects)
                return HTMLResponse(
                    "<!DOCTYPE html><html><head>"
                    "<title>vibetrack</title>"
                    "</head><body>"
                    "<script>"
                    f"const projects = {projects_json};"
                    "const targetProject = projects[0];"
                    "if (targetProject) {"
                    "try {"
                    "localStorage.setItem('vt_last_project', targetProject);"
                    "} catch (e) {}"
                    "window.location.replace('/' + encodeURIComponent(targetProject));"
                    "}"
                    "</script>"
                    "</body></html>"
                )
            data = _build_data(self._resolve_experiments(experiments))
            return HTMLResponse(_render(data, None, []))

        @app.get("/api/data")
        def api_data() -> list:
            return _build_data(self._resolve_experiments(experiments))

        @app.get("/api/data/{project}")
        def api_data_project(project: str) -> Any:
            db = self._reader._db
            if db is None or not db.list_experiments(project=project):
                return JSONResponse({"error": "not found"}, status_code=404)
            return _build_data(_experiments_for_project(db, project))

        # ── Config ────────────────────────────────────────────

        @app.get("/api/config")
        def get_config() -> dict:
            return load_config(project=config_project)

        @app.get("/api/config/{project}")
        def get_config_project(project: str) -> dict:
            return load_config(project=project)

        @app.post("/api/config")
        async def set_config(request: Request) -> dict:
            data = await request.json()
            save_config(data, project=config_project)
            return load_config(project=config_project)

        @app.post("/api/config/{project}")
        async def set_config_project(request: Request, project: str) -> dict:
            data = await request.json()
            save_config(data, project=project)
            return load_config(project=project)

        # ── Rename / delete / move ────────────────────────────

        @app.post("/api/rename")
        async def rename_exp(request: Request) -> JSONResponse:
            body = await request.json()
            project = body.get("project") or self._reader.project
            data, status = _handle_rename(
                self._reader._db,
                body.get("old_name", ""),
                body.get("new_name", ""),
                project,
            )
            return JSONResponse(data, status_code=status)

        @app.post("/api/rename/{project}")
        async def rename_exp_project(request: Request, project: str) -> JSONResponse:
            body = await request.json()
            data, status = _handle_rename(
                self._reader._db,
                body.get("old_name", ""),
                body.get("new_name", ""),
                body.get("project") or project,
            )
            return JSONResponse(data, status_code=status)

        @app.delete("/api/experiment")
        async def delete_exp(request: Request) -> JSONResponse:
            body = await request.json()
            project = body.get("project") or self._reader.project
            data, status = _handle_delete_experiment(
                self._reader._db,
                body.get("name", ""),
                project,
            )
            return JSONResponse(data, status_code=status)

        @app.delete("/api/project/{project}")
        def delete_project(project: str) -> JSONResponse:
            db = self._reader._db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)
            rows = db.list_experiments(project=project)
            if not rows:
                return JSONResponse({"error": "not found"}, status_code=404)
            db.delete_experiments_by_project(project)
            for row in rows:
                log_dir = row["log_dir"] if hasattr(row, "__getitem__") else ""
                if log_dir:
                    _cleanup_experiment_files(str(log_dir))
            return JSONResponse({"ok": True})

        @app.post("/api/move-logdir")
        async def move_logdir(request: Request) -> JSONResponse:
            body = await request.json()
            data, status = _handle_move_logdir(
                self._reader._db,
                body.get("old", ""),
                body.get("new", ""),
            )
            return JSONResponse(data, status_code=status)

        # ── Media ─────────────────────────────────────────────

        @app.get("/media")
        def serve_media(path: str = ""):  # type: ignore[no-untyped-def]
            """Serve media files by absolute path."""
            if not path:
                return JSONResponse({"error": "not found"}, status_code=404)
            candidate = os.path.realpath(path)
            allowed_dirs = _get_media_roots(self._reader)
            if not any(candidate.startswith(d) for d in allowed_dirs):
                return JSONResponse({"error": "not found"}, status_code=404)
            if os.path.isfile(candidate):
                return FileResponse(candidate)
            return JSONResponse({"error": "not found"}, status_code=404)

        # ── Ingest (listen) routes ────────────────────────────

        normalized_pf = (
            str(Path(self.project_folder).resolve())
            if self.project_folder is not None
            else None
        )

        def _new_writer(name: str, project_name: Optional[str] = None) -> SummaryWriter:
            if normalized_pf is not None:
                log_dir = Path(normalized_pf) / name
                return SummaryWriter(
                    str(log_dir),
                    name=name,
                    project_folder=normalized_pf,
                    system_metrics_interval=0,
                )
            return SummaryWriter(
                str(Path.cwd() / name),
                name=name,
                project=project_name,
                system_metrics_interval=0,
            )

        async def _check_listen_auth(request: Request):
            if token:
                auth = request.headers.get("Authorization", "")
                if not hmac.compare_digest(auth, f"Bearer {token}"):
                    raise HTTPException(status_code=401, detail="unauthorized")

        @app.post("/{project}/listen/log")
        async def listen_log(
            project: str, request: Request, _=Depends(_check_listen_auth)
        ) -> dict:
            data = await request.json()
            experiment = data.get("experiment", "default")
            step = data.get("step", 0)
            writer = _new_writer(experiment, project)
            try:
                for tag, value in data.get("scalars", {}).items():
                    writer.add_scalar(tag, value, step)
                for tag, value in data.get("texts", {}).items():
                    writer.add_text(tag, value, step)
            finally:
                writer.close()
            return {"status": "ok"}

        @app.post("/{project}/listen/media")
        async def listen_media(
            project: str,
            _=Depends(_check_listen_auth),
            experiment: str = Form("default"),
            tag: str = Form("upload"),
            step: int = Form(0),
            type: str = Form("artifact"),
            file: UploadFile = File(...),
        ) -> dict:
            writer: Optional[SummaryWriter] = None
            tmp_path: Optional[str] = None
            try:
                raw_suffix = os.path.splitext(file.filename or "")[1].lower()
                if type not in {"image", "audio", "video", "artifact"}:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported upload type {type!r}"
                    )
                allowed = _ALLOWED_SUFFIXES.get(type)
                if allowed is not None and raw_suffix not in allowed:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported file type for {type!r}"
                    )
                try:
                    tmp_path = await _write_upload_to_tempfile(file, suffix=raw_suffix)
                except ValueError:
                    raise HTTPException(
                        status_code=413, detail="File too large (max 1 GB)"
                    )
                writer = _new_writer(experiment, project)
                if type == "image":
                    writer.add_image(tag, tmp_path, step)
                elif type == "audio":
                    writer.add_audio(tag, tmp_path, step)
                elif type == "video":
                    writer.add_video(tag, tmp_path, step)
                else:
                    writer.add_artifact(tag, tmp_path, step)
                return {"status": "ok"}
            finally:
                if writer is not None:
                    writer.close()
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                await file.close()

        # ── MCP sub-app mount ─────────────────────────────────

        if _mcp_asgi is not None:
            app.mount("/vibetrack_mcp", _mcp_asgi)

        # ── Project-scoped index (must be registered last so fixed
        #     paths above take precedence over ``/{project}``) ──

        @app.get("/{project}")
        def project_index(project: str) -> HTMLResponse:
            db = self._reader._db
            if db is None:
                return HTMLResponse("no database", status_code=500)
            data = _build_data(_experiments_for_project(db, project))
            return HTMLResponse(_render(data, project, _all_projects()))

        return app
