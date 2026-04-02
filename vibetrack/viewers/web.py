"""Standalone web UI with interactive Chart.js graphs and media viewers.

Requires: ``pip install vibetrack[web]``  (fastapi, uvicorn)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..compare import find_all_tags
from ..config import load_config, save_config
from ..reader import RunReader
from .base import BaseOutput

_TEMPLATE_PATH = Path(__file__).parent / "web.html"


def _load_template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _json_for_html(obj: Any) -> str:
    """Serialize *obj* to JSON safe for embedding inside ``<script>``."""
    return json.dumps(obj).replace("</", r"\u003c/")


_SYSTEM_PREFIXES = ("system/", "gpu/")


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


def _build_data(
    output: "WebOutput",
    experiments: Optional[Sequence[str]],
) -> list:
    exps = output._resolve_experiments(experiments)
    data = []
    for exp in exps:
        log_dir = exp.log_dir
        # Scalars — split into regular and system
        all_tags = exp.scalar_tags()
        regular_tags = [t for t in all_tags if not t.startswith(_SYSTEM_PREFIXES)]
        system_tags = [t for t in all_tags if t.startswith(_SYSTEM_PREFIXES)]

        scalars: Dict[str, Any] = {}
        for tag in regular_tags:
            rows = exp.scalars(tag)
            scalars[tag] = {
                "steps": [r["step"] for r in rows],
                "values": [r["value"] for r in rows],
            }

        system_scalars: Dict[str, Any] = {}
        for tag in system_tags:
            rows = exp.scalars(tag)
            system_scalars[tag] = {
                "steps": [r["step"] for r in rows],
                "values": [r["value"] for r in rows],
            }

        # Images
        img_tags = exp.image_tags()
        images: Dict[str, list] = {}
        for tag in img_tags:
            rows = exp.images(tag)
            images[tag] = [{"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])} for r in rows]

        # Audio
        aud_tags = exp.audio_tags()
        audio_data: Dict[str, list] = {}
        for tag in aud_tags:
            rows = exp.audio(tag)
            audio_data[tag] = [{"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])} for r in rows]

        # Video
        vid_tags = exp.video_tags()
        video_data: Dict[str, list] = {}
        for tag in vid_tags:
            rows = exp.video(tag)
            video_data[tag] = [{"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])} for r in rows]

        # Artifacts
        art_tags = exp.artifact_tags()
        artifacts: Dict[str, list] = {}
        for tag in art_tags:
            rows = exp.artifacts(tag)
            artifacts[tag] = [
                {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"]), "metadata": r["metadata"]}
                for r in rows
            ]

        # Texts
        txt_tags = exp.text_tags()
        text_data: Dict[str, list] = {}
        for tag in txt_tags:
            rows = exp.texts(tag)
            text_data[tag] = [{"step": r["step"], "value": r["value"]} for r in rows]

        # Histograms
        hist_tags = exp.histogram_tags()
        histogram_data: Dict[str, list] = {}
        for tag in hist_tags:
            rows = exp.histograms(tag)
            histogram_data[tag] = [
                {"step": r["step"], "bins": r["bins"], "counts": r["counts"]}
                for r in rows
            ]

        # Hparams
        hparams = exp.hparams()

        data.append({
            "name": exp.name,
            "log_dir": log_dir,
            "tags": regular_tags,
            "scalars": scalars,
            "system_tags": system_tags,
            "system_scalars": system_scalars,
            "image_tags": img_tags,
            "images": images,
            "audio_tags": aud_tags,
            "audio_data": audio_data,
            "video_tags": vid_tags,
            "video_data": video_data,
            "artifact_tags": art_tags,
            "artifacts": artifacts,
            "text_tags": txt_tags,
            "text_data": text_data,
            "histogram_tags": hist_tags,
            "histogram_data": histogram_data,
            "hparams": hparams,
        })
    return data


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


def _build_project_data(
    db: Any,
    project: str,
    experiments: Optional[Sequence[str]] = None,
) -> list:
    """Build data for a specific project from the DB directly."""
    from ..reader import ExperimentReader
    rows = db.list_experiments(project=project)
    exps = [ExperimentReader(db, r["id"], r["name"], row=r) for r in rows]
    if experiments is not None:
        name_set = set(experiments)
        exps = [e for e in exps if e.name in name_set]
    data = []
    for exp in exps:
        ld = exp.log_dir
        all_tags = exp.scalar_tags()
        regular_tags = [t for t in all_tags if not t.startswith(_SYSTEM_PREFIXES)]
        system_tags = [t for t in all_tags if t.startswith(_SYSTEM_PREFIXES)]
        scalars: Dict[str, Any] = {}
        for tag in regular_tags:
            rows_s = exp.scalars(tag)
            scalars[tag] = {"steps": [r["step"] for r in rows_s], "values": [r["value"] for r in rows_s]}
        system_scalars: Dict[str, Any] = {}
        for tag in system_tags:
            rows_s = exp.scalars(tag)
            system_scalars[tag] = {"steps": [r["step"] for r in rows_s], "values": [r["value"] for r in rows_s]}
        images: Dict[str, list] = {tag: [{"step": r["step"], "path": _resolve_media_path(ld, r["path"])} for r in exp.images(tag)] for tag in exp.image_tags()}
        audio_d: Dict[str, list] = {tag: [{"step": r["step"], "path": _resolve_media_path(ld, r["path"])} for r in exp.audio(tag)] for tag in exp.audio_tags()}
        video_d: Dict[str, list] = {tag: [{"step": r["step"], "path": _resolve_media_path(ld, r["path"])} for r in exp.video(tag)] for tag in exp.video_tags()}
        artifacts: Dict[str, list] = {tag: [{"step": r["step"], "path": _resolve_media_path(ld, r["path"]), "metadata": r["metadata"]} for r in exp.artifacts(tag)] for tag in exp.artifact_tags()}
        text_d: Dict[str, list] = {tag: [{"step": r["step"], "value": r["value"]} for r in exp.texts(tag)] for tag in exp.text_tags()}
        hist_d: Dict[str, list] = {tag: [{"step": r["step"], "bins": r["bins"], "counts": r["counts"]} for r in exp.histograms(tag)] for tag in exp.histogram_tags()}
        data.append({
            "name": exp.name, "log_dir": ld, "tags": regular_tags, "scalars": scalars,
            "system_tags": system_tags, "system_scalars": system_scalars,
            "image_tags": list(images), "images": images,
            "audio_tags": list(audio_d), "audio_data": audio_d,
            "video_tags": list(video_d), "video_data": video_d,
            "artifact_tags": list(artifacts), "artifacts": artifacts,
            "text_tags": list(text_d), "text_data": text_d,
            "histogram_tags": list(hist_d), "histogram_data": hist_d,
            "hparams": exp.hparams(),
        })
    return data


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
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
        import logging

        logging.getLogger("uvicorn.error").addFilter(
            type("_F", (logging.Filter,), {"filter": lambda self, r: "Invalid HTTP request received" not in r.getMessage()})()
        )

        import shutil

        host = _resolve_host(host)

        # Build MCP sub-app early so we can wire its lifespan
        _mcp_asgi = None
        try:
            from .mcp import MCPOutput, _suppress_streamable_http_startup_log
            _mcp_output = MCPOutput(self.project_folder)
            _mcp_output._build_mcp()
            _mcp_asgi = _mcp_output._mcp.streamable_http_app()
            _mcp_session_manager = _mcp_output._mcp._session_manager
        except ImportError:
            _mcp_output = None
            _mcp_session_manager = None

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _lifespan(app):
            if _mcp_session_manager is not None:
                with _suppress_streamable_http_startup_log():
                    async with _mcp_session_manager.run():
                        yield
            else:
                yield

        app = FastAPI(title="vibetrack", lifespan=_lifespan)
        config_project = self.config_project()
        is_central = self.project_folder is None

        @app.get("/")
        def index() -> HTMLResponse:
            if is_central and self.project is None:
                # Central mode: redirect to last active project
                db = self._reader._db
                projects = [p for p in (db.list_projects() if db else []) if p]
                if not projects:
                    return HTMLResponse(
                        "<!DOCTYPE html><html><head>"
                        '<meta http-equiv="refresh" content="5">'
                        "</head><body>"
                        "<h1>vibetrack</h1><p>No projects yet.</p>"
                        "</body></html>"
                    )
                return HTMLResponse(
                    "<!DOCTYPE html><html><head><script>"
                    f"const projects={json.dumps(projects)};"
                    "const last=localStorage.getItem('vt_last_project');"
                    "if(last&&projects.includes(last))window.location.replace('/'+last);"
                    "else window.location.replace('/'+projects[0]);"
                    "</script></head><body></body></html>"
                )
            data = _build_data(self, experiments)
            html = _load_template().replace("__DATA_JSON__", _json_for_html(data))
            return HTMLResponse(content=html)

        @app.get("/api/data")
        def api_data() -> list:
            return _build_data(self, experiments)

        @app.get("/api/data/{project}")
        def api_data_project(project: str) -> Any:
            db = self._reader._db
            if db is None or not db.list_experiments(project=project):
                return JSONResponse({"error": "not found"}, status_code=404)
            return _build_project_data(db, project)

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

        @app.post("/api/rename")
        async def rename_exp(request: Request) -> JSONResponse:
            return await _do_rename(request, self._reader.project)

        @app.post("/api/rename/{project}")
        async def rename_exp_project(request: Request, project: str) -> JSONResponse:
            return await _do_rename(request, project)

        async def _do_rename(request: Request, project: Optional[str]) -> JSONResponse:
            body = await request.json()
            old_name = body.get("old_name", "")
            new_name = body.get("new_name", "").strip()
            if not new_name:
                return JSONResponse({"error": "name cannot be empty"}, status_code=400)
            db = self._reader._db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)
            for exp_row in db.list_experiments(project=project):
                if exp_row["name"] == old_name:
                    if not db.rename_experiment(exp_row["id"], new_name):
                        return JSONResponse(
                            {"error": "name already exists"},
                            status_code=409,
                        )
                    return JSONResponse({"ok": True, "new_name": new_name})
            return JSONResponse({"error": "experiment not found"}, status_code=404)

        @app.delete("/api/experiment")
        async def delete_exp(request: Request) -> JSONResponse:
            body = await request.json()
            name = body.get("name", "")
            if not name:
                return JSONResponse({"error": "name required"}, status_code=400)
            db = self._reader._db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)
            project = self._reader.project
            row = db.get_experiment_by_name(name, project=project)
            if row is None:
                return JSONResponse({"error": "experiment not found"}, status_code=404)
            log_dir = db.delete_experiment(int(row["id"]))
            # Clean up media directory if empty after delete
            if log_dir:
                md = os.path.join(log_dir, "media")
                if os.path.isdir(md):
                    shutil.rmtree(md, ignore_errors=True)
            return JSONResponse({"ok": True})

        @app.delete("/api/project/{project}")
        def delete_project(project: str) -> JSONResponse:
            db = self._reader._db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)
            rows = db.list_experiments(project=project)
            if not rows:
                return JSONResponse({"error": "not found"}, status_code=404)
            # Collect media dirs to clean up
            media_dirs: set[str] = set()
            for row in rows:
                log_dir = row["log_dir"] if hasattr(row, "__getitem__") else ""
                if log_dir:
                    md = os.path.join(log_dir, "media")
                    if os.path.isdir(md):
                        media_dirs.add(md)
            # Delete experiment rows and all associated data
            db.delete_experiments_by_project(project)
            # Remove media directories
            for md in media_dirs:
                shutil.rmtree(md, ignore_errors=True)
            return JSONResponse({"ok": True})

        @app.post("/api/move-logdir")
        async def move_logdir(request: Request) -> JSONResponse:
            body = await request.json()
            old_dir = body.get("old", "")
            new_dir = body.get("new", "")
            if not old_dir or not new_dir:
                return JSONResponse({"error": "old and new required"}, status_code=400)
            old_path = Path(old_dir).expanduser().resolve()
            new_path = Path(new_dir).expanduser().resolve()
            if old_path == new_path:
                return JSONResponse({"ok": True})
            db = self._reader._db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)

            def _row_log_dir(row: Any) -> str:
                if isinstance(row, dict):
                    return str(row.get("log_dir", ""))
                try:
                    return str(row["log_dir"])
                except Exception:
                    return ""

            def _normalize_path(value: str) -> Path:
                return Path(value).expanduser().resolve()

            def _is_under_or_equal(parent: Path, child: Path) -> bool:
                try:
                    child.relative_to(parent)
                    return True
                except ValueError:
                    return False

            # Validate source is EXACTLY a known experiment log_dir.
            known = [
                _normalize_path(_row_log_dir(r))
                for r in db.list_experiments() if _row_log_dir(r)
            ]
            if old_path not in known:
                return JSONResponse({"error": "not a known log directory"}, status_code=403)
            if not old_path.is_dir():
                return JSONResponse({"error": "source directory not found"}, status_code=404)
            if new_path.exists():
                return JSONResponse({"error": "destination already exists"}, status_code=409)
            # Move the directory
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
            except OSError:
                return JSONResponse({"error": "move failed"}, status_code=500)
            # Update all experiments whose log_dir is under the moved directory
            for row in db.list_experiments():
                ld_raw = _row_log_dir(row)
                if not ld_raw:
                    continue
                ld_path = _normalize_path(ld_raw)
                if _is_under_or_equal(old_path, ld_path):
                    rel = ld_path.relative_to(old_path)
                    db.update_log_dir(int(row["id"]), str((new_path / rel).resolve()))
            return JSONResponse({"ok": True})

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
        from fastapi import Depends, File, Form, HTTPException, UploadFile
        from ..cli import _write_upload_to_tempfile, _MAX_UPLOAD_BYTES, _ALLOWED_SUFFIXES
        from ..writer import SummaryWriter
        import hmac

        normalized_pf = (
            str(Path(self.project_folder).resolve())
            if self.project_folder is not None else None
        )

        def _new_writer(name: str, project_name: Optional[str] = None) -> SummaryWriter:
            if normalized_pf is not None:
                log_dir = Path(normalized_pf) / name
                return SummaryWriter(
                    str(log_dir), name=name,
                    project_folder=normalized_pf,
                    system_metrics_interval=0,
                )
            return SummaryWriter(
                str(Path.cwd() / name), name=name,
                project=project_name,
                system_metrics_interval=0,
            )

        async def _check_listen_auth(request: Request):
            if token:
                auth = request.headers.get("Authorization", "")
                if not hmac.compare_digest(auth, f"Bearer {token}"):
                    raise HTTPException(status_code=401, detail="unauthorized")

        @app.post("/{project}/listen/log")
        async def listen_log(project: str, request: Request, _=Depends(_check_listen_auth)) -> dict:
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
                    raise HTTPException(status_code=400, detail=f"Unsupported upload type {type!r}")
                allowed = _ALLOWED_SUFFIXES.get(type)
                if allowed is not None and raw_suffix not in allowed:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type for {type!r}")
                try:
                    tmp_path = await _write_upload_to_tempfile(file, suffix=raw_suffix)
                except ValueError:
                    raise HTTPException(status_code=413, detail="File too large (max 1 GB)")
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

        @app.get("/{project}")
        def project_index(project: str) -> HTMLResponse:
            """Serve project-scoped dashboard."""
            db = self._reader._db
            if db is None:
                return HTMLResponse("no database", status_code=500)
            data = _build_project_data(db, project)
            all_projects = [p for p in db.list_projects() if p]
            html = _load_template().replace("__DATA_JSON__", _json_for_html(data))
            # Inject project context for frontend
            html = html.replace(
                "</head>",
                f"<script>"
                f"window.VT_PROJECT={json.dumps(project)};"
                f"window.VT_PROJECTS={json.dumps(all_projects)};"
                f"</script></head>",
            )
            # Rewrite API paths to project-scoped versions
            html = html.replace("/api/data", f"/api/data/{project}")
            html = html.replace("/api/rename", f"/api/rename/{project}")
            html = html.replace("/api/config", f"/api/config/{project}")
            return HTMLResponse(content=html)

        return app
