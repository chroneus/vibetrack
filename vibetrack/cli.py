"""CLI entry point for central-DB and project-local vibetrack workflows."""

import argparse
import hmac
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from .db import Database

_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB
_ALLOWED_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
    "audio": {".wav", ".mp3", ".ogg", ".flac", ".m4a"},
    "video": {".mp4", ".webm", ".avi", ".mov", ".mkv"},
}


async def _write_upload_to_tempfile(
    upload: object,
    suffix: str,
    max_bytes: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Stream an uploaded file to disk while enforcing a hard size limit."""
    if max_bytes is None:
        max_bytes = _MAX_UPLOAD_BYTES
    total = 0
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await upload.read(chunk_size)  # type: ignore[attr-defined]
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("File too large")
                tmp.write(chunk)
        return tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _parse_listen(s: str) -> Tuple[str, int]:
    """Parse ``host:port`` or bare ``port`` string."""
    if ":" in s:
        host, port_str = s.rsplit(":", 1)
        return host, int(port_str)
    return "127.0.0.1", int(s)


def _normalize_project_folder(project_folder: Optional[str]) -> Optional[str]:
    if project_folder is None:
        return None
    path = Path(project_folder).expanduser().resolve()
    if path.name == "vibetrack.db":
        return str(path.parent)
    return str(path)


def _create_listen_app(project_folder: Optional[str], token: Optional[str] = None):
    """Build FastAPI app for the HTTP ingest server."""
    from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

    from .writer import SummaryWriter

    app = FastAPI(title="vibetrack-ingest")
    normalized_project_folder = _normalize_project_folder(project_folder)
    project_root = (
        Path(normalized_project_folder).resolve()
        if normalized_project_folder is not None else None
    )

    def _new_writer(name: str) -> SummaryWriter:
        if project_root is not None:
            log_dir = project_root / name
            return SummaryWriter(
                str(log_dir),
                name=name,
                project_folder=str(project_root),
                system_metrics_interval=0,
            )
        log_dir = Path.cwd() / name
        return SummaryWriter(str(log_dir), name=name, system_metrics_interval=0)

    async def check_auth(request: Request):  # type: ignore[no-untyped-def]
        if token:
            auth = request.headers.get("Authorization", "")
            if not hmac.compare_digest(auth, f"Bearer {token}"):
                raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/log")
    async def log_data(request: Request, _=Depends(check_auth)) -> dict:
        data = await request.json()
        experiment = data.get("experiment", "default")
        if "/" in experiment or "\\" in experiment or ".." in experiment:
            raise HTTPException(status_code=400, detail="Invalid experiment name")
        step = data.get("step", 0)
        writer = _new_writer(experiment)
        try:
            for tag, value in data.get("scalars", {}).items():
                writer.add_scalar(tag, value, step)
            for tag, value in data.get("texts", {}).items():
                writer.add_text(tag, value, step)
        finally:
            writer.close()
        return {"status": "ok"}

    @app.post("/media")
    async def upload_media(
        _=Depends(check_auth),
        experiment: str = Form("default"),
        tag: str = Form("upload"),
        step: int = Form(0),
        type: str = Form("artifact"),
        file: UploadFile = File(...),
    ) -> dict:
        if "/" in experiment or "\\" in experiment or ".." in experiment:
            raise HTTPException(status_code=400, detail="Invalid experiment name")
        writer: Optional[SummaryWriter] = None
        tmp_path: Optional[str] = None
        try:
            raw_suffix = os.path.splitext(file.filename or "")[1].lower()
            if type not in {"image", "audio", "video", "artifact"}:
                raise HTTPException(status_code=400, detail=f"Unsupported upload type {type!r}")

            if type == "artifact":
                if raw_suffix in {".html", ".htm", ".js", ".svg", ".php", ".sh", ".exe", ".cgi", ".pl"}:
                    raise HTTPException(status_code=400, detail="Refusing to serve potentially dangerous file extension")
            else:
                allowed = _ALLOWED_SUFFIXES.get(type)
                if allowed is not None and raw_suffix not in allowed:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type for {type!r}")

            try:
                tmp_path = await _write_upload_to_tempfile(file, suffix=raw_suffix)
            except ValueError:
                raise HTTPException(status_code=413, detail="File too large (max 1 GB)")

            writer = _new_writer(experiment)
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

    return app


def _start_listen_server(
    project_folder: Optional[str],
    host: str,
    port: int,
    token: Optional[str],
) -> threading.Thread:
    """Start the ingest server in a daemon thread."""
    import uvicorn

    app = _create_listen_app(project_folder, token)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    print(f"vibetrack listen server: http://{host}:{port}")
    return thread


def _experiment_has_data(db: Database, experiment_id: int) -> bool:
    return any([
        db.get_scalar_tags(experiment_id),
        db.get_text_tags(experiment_id),
        db.get_image_tags(experiment_id),
        db.get_audio_tags(experiment_id),
        db.get_video_tags(experiment_id),
        db.get_artifact_tags(experiment_id),
        db.get_histogram_tags(experiment_id),
        db.get_hparams(experiment_id),
    ])


def _migrate_experiment(
    source_db: Database,
    source_row: object,
    target_db: Database,
    target_project: str,
    fallback_log_dir: str,
) -> bool:
    row = dict(source_row)
    name = row["name"]
    source_id = row["id"]
    config = json.loads(row["config"]) if row.get("config") else None
    log_dir = row.get("log_dir") or fallback_log_dir

    existing = target_db.get_experiment_by_name(name, project=target_project)
    if existing is not None and _experiment_has_data(target_db, existing["id"]):
        return False

    target_id = target_db.create_experiment(
        name,
        config=config,
        project=target_project,
        log_dir=log_dir,
    )

    scalar_rows = []
    for tag in source_db.get_scalar_tags(source_id):
        scalar_rows.extend(
            (target_id, tag, row["step"], row["value"], row["wall_time"])
            for row in source_db.get_scalars(source_id, tag)
        )
    if scalar_rows:
        target_db.add_scalars_bulk(scalar_rows)

    for hparams in [source_db.get_hparams(source_id)]:
        if hparams:
            target_db.add_hparams(target_id, hparams)

    for tag in source_db.get_text_tags(source_id):
        for entry in source_db.get_texts(source_id, tag):
            target_db.add_text(
                target_id,
                tag,
                entry["value"],
                entry["step"],
                entry["wall_time"],
            )

    for tag in source_db.get_image_tags(source_id):
        for entry in source_db.get_images(source_id, tag):
            target_db.add_image(
                target_id,
                tag,
                entry["path"],
                entry["step"],
                entry["wall_time"],
            )

    for tag in source_db.get_audio_tags(source_id):
        for entry in source_db.get_audio(source_id, tag):
            target_db.add_audio(
                target_id,
                tag,
                entry["path"],
                entry["step"],
                entry["sample_rate"],
                entry["wall_time"],
            )

    for tag in source_db.get_video_tags(source_id):
        for entry in source_db.get_video(source_id, tag):
            target_db.add_video(
                target_id,
                tag,
                entry["path"],
                entry["step"],
                entry["wall_time"],
            )

    for tag in source_db.get_artifact_tags(source_id):
        for entry in source_db.get_artifacts(source_id, tag):
            target_db.add_artifact(
                target_id,
                tag,
                entry["path"],
                entry["metadata"],
                entry["step"],
                entry["wall_time"],
            )

    for tag in source_db.get_histogram_tags(source_id):
        for entry in source_db.get_histograms(source_id, tag):
            target_db.add_histogram(
                target_id,
                tag,
                entry["bins"],
                entry["counts"],
                entry["step"],
                entry["wall_time"],
            )

    return True


def migrate_project(project_folder: str) -> int:
    """Merge legacy per-run DBs into the project-level DB."""
    project_root = Path(project_folder).resolve()
    if not project_root.exists():
        print(f"Error: project folder {str(project_root)!r} does not exist.", file=sys.stderr)
        return 1

    target_db_path = project_root / "vibetrack.db"
    legacy_dbs = sorted(
        path for path in project_root.rglob("vibetrack.db")
        if path != target_db_path
    )
    if not legacy_dbs:
        print(f"No legacy run databases found under {project_root}.")
        return 0

    target_project = project_root.name
    target_db = Database(target_db_path)
    migrated = 0
    skipped = 0
    try:
        for db_path in legacy_dbs:
            source_db = Database(db_path)
            try:
                fallback_log_dir = str(db_path.parent.resolve())
                for row in source_db.list_experiments():
                    if _migrate_experiment(
                        source_db,
                        row,
                        target_db,
                        target_project,
                        fallback_log_dir,
                    ):
                        migrated += 1
                    else:
                        skipped += 1
            finally:
                source_db.close()
    finally:
        target_db.close()

    print(
        f"Migrated {migrated} experiment(s) into {target_db_path} "
        f"from {len(legacy_dbs)} legacy DB(s); skipped {skipped} existing experiment(s)."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vibetrack",
        description="Lightweight experiment tracking.",
    )
    parser.add_argument(
        "target_pos",
        nargs="?",
        default=None,
        metavar="PROJECT_FOLDER",
        help="Project folder, or the literal command 'migrate'.",
    )
    parser.add_argument(
        "project_folder_pos",
        nargs="?",
        default=None,
        metavar="PROJECT_FOLDER",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--project-folder",
        default=None,
        help="Project folder containing a local vibetrack.db",
    )
    parser.add_argument(
        "--logdir",
        dest="legacy_project_folder",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--viewer",
        default="web",
        help="Viewer backend (default: web). Web includes MCP + ingest. Use 'mcp', 'console', etc. for standalone.",
    )
    parser.add_argument(
        "--listen",
        metavar="HOST:PORT",
        help="Start HTTP ingest server on host:port to receive remote logs",
    )
    parser.add_argument(
        "--token",
        help="Bearer token for listen server authentication",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viewer host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Viewer port (default: 6006)",
    )
    parser.add_argument(
        "--mcp-transport",
        default="streamable-http",
        choices=["streamable-http", "sse"],
        help="MCP transport type (default: streamable-http). Only used with --viewer=mcp.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.legacy_project_folder is not None and args.target_pos is None:
        args.project_folder = args.legacy_project_folder

    if args.target_pos == "migrate":
        project_folder = args.project_folder_pos or args.project_folder or "."
        sys.exit(migrate_project(project_folder))

    if args.target_pos is not None:
        args.project_folder = args.target_pos

    project_folder = _normalize_project_folder(args.project_folder)

    if args.listen:
        try:
            listen_host, listen_port = _parse_listen(args.listen)
        except ValueError:
            print(f"Invalid --listen value: {args.listen!r}. Use HOST:PORT or PORT.", file=sys.stderr)
            sys.exit(1)
        try:
            _start_listen_server(project_folder, listen_host, listen_port, args.token)
        except ImportError:
            print("FastAPI/uvicorn required for --listen. Install with: pip install vibetrack[web]", file=sys.stderr)
            sys.exit(1)

    from .viewers import load_viewer

    viewer_cls = load_viewer(args.viewer)
    viewer = viewer_cls(project_folder)
    viewer.show(host=args.host, port=args.port, token=args.token, mcp_transport=args.mcp_transport)


if __name__ == "__main__":
    main()
