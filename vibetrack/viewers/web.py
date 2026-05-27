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

import collections
import functools
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from ..compare import find_all_tags  # noqa: F401  (re-exported for tests)
from ..config import load_config, save_config
from ..reader import ExperimentReader, RunReader
from .base import BaseOutput

_WEB_DIR = Path(__file__).parent / "web"
_INDEX_PATH = _WEB_DIR / "index.html"

_SYSTEM_PREFIXES = ("system/", "gpu/")

# In-process rate limiter for state-mutating endpoints. 10 requests per
# second per source IP is generous for a human clicking buttons but
# stops a script from wiping a project in a tight loop.
_MUTATION_RATE_WINDOW_SEC = 1.0
_MUTATION_RATE_LIMIT = 10
_mutation_rate_lock = threading.Lock()
_mutation_rate_state: Dict[str, Deque[float]] = collections.defaultdict(
    collections.deque
)


def _rate_limit_mutation(client_ip: str) -> bool:
    """Return ``True`` if *client_ip* is allowed another mutation.

    Sliding-window: any caller exceeding ``_MUTATION_RATE_LIMIT`` requests
    within ``_MUTATION_RATE_WINDOW_SEC`` is throttled.
    """
    now = time.monotonic()
    cutoff = now - _MUTATION_RATE_WINDOW_SEC
    with _mutation_rate_lock:
        bucket = _mutation_rate_state[client_ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _MUTATION_RATE_LIMIT:
            return False
        bucket.append(now)
        return True


def _client_ip(request: Any) -> str:
    """Best-effort client IP for rate limiting (no spoofing protection)."""
    client = getattr(request, "client", None)
    if client is not None and getattr(client, "host", None):
        return str(client.host)
    return "unknown"


def _is_same_origin(request: Any, allowed_host: str) -> bool:
    """Check that Origin/Referer (if present) match the bound host.

    Browsers send ``Origin`` on cross-origin POST/DELETE; a missing
    header is treated as same-origin (curl, server-side scripts). When
    present, the host portion must match the server's bound host —
    that's enough to block CSRF from another page in the same browser.
    """
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True
    try:
        parsed_host = urlparse(origin).hostname or ""
    except ValueError:
        return False
    if not parsed_host:
        return False
    # Treat any loopback variant as equivalent — uvicorn binds 127.0.0.1
    # but the browser may navigate via "localhost".
    loopbacks = {"localhost", "127.0.0.1", "::1"}
    if parsed_host in loopbacks and allowed_host in loopbacks:
        return True
    return parsed_host == allowed_host


@functools.lru_cache(maxsize=1)
def _load_template() -> str:
    return _INDEX_PATH.read_text(encoding="utf-8")


# Per-process cache-busting stamp. Static asset URLs in index.html get
# rewritten to include `?v=<stamp>` so a server restart forces browsers to
# refetch JS/CSS even if their HTTP cache holds the previous version.
_STATIC_VERSION = str(int(time.time()))


def _json_for_html(obj: Any) -> str:
    """Serialize *obj* to JSON safe for embedding inside ``<script>``."""
    return json.dumps(obj).replace("</", r"\u003c/")


def _render(
    data: list,
    project: Optional[str],
    projects: Sequence[str],
) -> str:
    """Render the index template with JSON-safe replacements."""
    html = (
        _load_template()
        .replace("__DATA_JSON__", _json_for_html(data))
        .replace("__PROJECT_JSON__", _json_for_html(project))
        .replace("__PROJECTS_JSON__", _json_for_html(list(projects)))
    )
    # Cache-bust local static URLs: src="/static/foo.js" → src="/static/foo.js?v=…"
    html = re.sub(
        r'(src|href)="(/static/[^"?#]+)"',
        rf'\1="\2?v={_STATIC_VERSION}"',
        html,
    )
    return html


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
    if not log_dir:
        # No log_dir to anchor containment against — preserve legacy behaviour.
        return rel_path
    log_dir_resolved = str(Path(log_dir).resolve())
    if os.path.isabs(rel_path):
        candidate = str(Path(rel_path).resolve())
    else:
        candidate = str((Path(log_dir) / rel_path).resolve())
    # Defense in depth: even with a poisoned absolute path, refuse to return
    # anything outside the experiment's log_dir. The /media endpoint enforces
    # this again, but catching it here keeps poisoned paths off the wire.
    if (
        not candidate.startswith(log_dir_resolved + os.sep)
        and candidate != log_dir_resolved
    ):
        return ""
    return candidate


def _scalar_series(exp: ExperimentReader, tag: str) -> Dict[str, list]:
    rows = exp.scalars(tag)
    return _scalar_series_from_rows(rows)


def _scalar_series_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, list]:
    return {
        "steps": [r["step"] for r in rows],
        "values": [r["value"] for r in rows],
        "wall_times": [r["wall_time"] for r in rows],
    }


# Cap how many points we PCA-reduce + push to the browser per (tag, step).
# 5k points renders smoothly under a custom THREE.Points shader and keeps the
# embedded JSON payload reasonable. Larger tags are random-sampled with a
# tag-derived seed so the projection is stable across reloads.
_EMBEDDING_MAX_POINTS = 5000


def _project_3d(vectors: Any) -> Any:
    """Project an (N, D) matrix to (N, 3) via mean-centered SVD (PCA)."""
    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1) if arr.size else arr
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    n, d = arr.shape
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centered = arr - arr.mean(axis=0, keepdims=True)
    if d == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if d < 3:
        # Pad to 3-D so the viewer always gets a renderable cloud.
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, :d] = centered
        return out
    # Truncated SVD via numpy — fast for the sizes we cap to (N ≤ 5000).
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        comps = (u[:, :3] * s[:3]).astype(np.float32)
        if comps.shape[1] < 3:
            padded = np.zeros((n, 3), dtype=np.float32)
            padded[:, : comps.shape[1]] = comps
            comps = padded
        return comps
    except np.linalg.LinAlgError:
        return np.zeros((n, 3), dtype=np.float32)


def _embedding_payload(
    log_dir: str,
    entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the per-step JSON the embeddings tab consumes, or ``None``.

    Drops empty payloads (no vectors) so the server doesn't ship dead cards.
    """
    import numpy as np  # type: ignore[import-untyped]

    vectors = entry.get("vectors")
    if not vectors:
        return None
    try:
        mat = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if mat.ndim != 2 or mat.shape[0] == 0:
        return None

    rows = entry.get("metadata_rows")
    indices = None
    if mat.shape[0] > _EMBEDDING_MAX_POINTS:
        # Deterministic sub-sample so reloads don't reshuffle the cloud.
        seed = abs(hash((entry.get("step"), mat.shape[0], mat.shape[1]))) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(mat.shape[0], size=_EMBEDDING_MAX_POINTS, replace=False)
        )
        mat = mat[indices]
        if isinstance(rows, list):
            rows = [rows[i] for i in indices.tolist()]

    points3d = _project_3d(mat)

    sprite_abs = entry.get("sprite_abs")
    sprite_url: Optional[str] = None
    if sprite_abs and os.path.exists(sprite_abs):
        sprite_url = "/media?path=" + sprite_abs

    out: Dict[str, Any] = {
        "step": entry.get("step"),
        "path": _resolve_media_path(log_dir, entry.get("path", "")),
        "shape": list(map(int, entry.get("shape") or []))
        or [int(mat.shape[0]), int(mat.shape[1])],
        "n_points": int(points3d.shape[0]),
        "n_total": (
            int(entry.get("shape", [mat.shape[0]])[0])
            if entry.get("shape")
            else int(mat.shape[0])
        ),
        "points3d": points3d.tolist(),
        "metadata_rows": rows if isinstance(rows, list) else None,
        "metadata_header": (
            entry.get("metadata_header")
            if isinstance(entry.get("metadata_header"), list)
            else None
        ),
        "sprite_url": sprite_url,
        "sprite": entry.get("sprite"),
        "sampled_indices": indices.tolist() if indices is not None else None,
    }
    return out


def _serialize_experiment(exp: ExperimentReader) -> Dict[str, Any]:
    """Build the JSON-ready dict for a single experiment.

    All per-media data is fetched with a single SQL query per kind via
    the ``ExperimentReader.all_*`` bulk loaders, instead of one query
    per tag — eliminating the N+1 pattern that scaled badly with the
    number of tags.
    """
    log_dir = exp.log_dir

    all_scalar_data = exp.all_scalars()
    all_tags = sorted(all_scalar_data.keys())
    regular_tags = [t for t in all_tags if not t.startswith(_SYSTEM_PREFIXES)]
    system_tags = [t for t in all_tags if t.startswith(_SYSTEM_PREFIXES)]
    scalars = {
        tag: _scalar_series_from_rows(all_scalar_data[tag]) for tag in regular_tags
    }
    system_scalars = {
        tag: _scalar_series_from_rows(all_scalar_data[tag]) for tag in system_tags
    }

    images = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in rows
        ]
        for tag, rows in exp.all_images().items()
    }
    audio_data = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in rows
        ]
        for tag, rows in exp.all_audio().items()
    }
    video_data = {
        tag: [
            {"step": r["step"], "path": _resolve_media_path(log_dir, r["path"])}
            for r in rows
        ]
        for tag, rows in exp.all_video().items()
    }
    # all_artifacts() returns parsed metadata dicts; _serialize_experiment
    # historically downstreamed raw JSON strings, so keep both forms in
    # the dict (``metadata`` for the parsed object, used by _is_kind).
    raw_artifacts_parsed = exp.all_artifacts()
    raw_artifacts = {
        tag: [
            {
                "step": r["step"],
                "path": _resolve_media_path(log_dir, r["path"]),
                "metadata": r["metadata"],
            }
            for r in rows
        ]
        for tag, rows in raw_artifacts_parsed.items()
    }

    def _is_kind(item: Dict[str, Any], *kinds: str) -> bool:
        meta = item.get("metadata") or {}
        return meta.get("kind") in kinds

    # Pull each special "kind" (graphs/pr_curves/figures/meshes) into its
    # own bucket and *remove* them from the generic artifacts feed so they
    # don't double up in the Artifacts tab. Each gets its own UI surface.
    models: Dict[str, list] = {}
    pr_curves: Dict[str, list] = {}
    figures: Dict[str, list] = {}
    meshes: Dict[str, list] = {}
    embeddings: Dict[str, list] = {}
    artifacts: Dict[str, list] = {}
    for tag, rows in raw_artifacts.items():
        model_rows = [
            r
            for r in rows
            if _is_kind(r, "graph") or (r.get("metadata") or {}).get("format") == "dot"
        ]
        pr_rows = [r for r in rows if _is_kind(r, "pr_curve")]
        figure_rows = [r for r in rows if _is_kind(r, "figure")]
        mesh_rows = [r for r in rows if _is_kind(r, "mesh")]
        embedding_rows = [r for r in rows if _is_kind(r, "embedding")]
        leftover = [
            r
            for r in rows
            if r not in model_rows
            and r not in pr_rows
            and r not in figure_rows
            and r not in mesh_rows
            and r not in embedding_rows
        ]
        if model_rows:
            for r in model_rows:
                # Resolve embedded rendered PNG path to absolute so the JS
                # can serve it via /media without further work.
                meta = r.get("metadata") or {}
                rel = meta.get("rendered_png_path")
                if rel:
                    meta["rendered_png_abs"] = _resolve_media_path(log_dir, rel)
            models[tag] = model_rows
        if pr_rows:
            pr_curves[tag] = pr_rows
        if figure_rows:
            figures[tag] = figure_rows
        if mesh_rows:
            # Inline the mesh JSON payload (vertices/colors/faces) so the
            # client renders without a follow-up fetch. Strip the .shape
            # wrapper — the JS only needs flat .values arrays.
            mesh_entries = exp.meshes(tag)
            by_step = {e["step"]: e for e in mesh_entries}
            for r in mesh_rows:
                entry = by_step.get(r["step"], {})
                r["vertices"] = (entry.get("vertices") or {}).get("values")
                r["colors"] = (entry.get("colors") or {}).get("values")
                r["faces"] = (entry.get("faces") or {}).get("values")
            meshes[tag] = mesh_rows
        if embedding_rows:
            # Server-side PCA-reduce + sample so the client gets a render-
            # ready 3-D point cloud + sprite URL without parsing megabytes
            # of high-D vectors. Drop entries whose JSON is empty/corrupt.
            embed_entries = exp.embeddings(tag)
            by_step = {e["step"]: e for e in embed_entries}
            collected: List[Dict[str, Any]] = []
            for r in embedding_rows:
                entry = by_step.get(r["step"])
                if entry is None:
                    continue
                payload = _embedding_payload(log_dir, entry)
                if payload is not None:
                    collected.append(payload)
            if collected:
                embeddings[tag] = collected
        if leftover:
            artifacts[tag] = leftover
    text_data = {
        tag: [{"step": r["step"], "value": r["value"]} for r in rows]
        for tag, rows in exp.all_texts().items()
    }
    histogram_data = {
        tag: [
            {"step": r["step"], "bins": r["bins"], "counts": r["counts"]} for r in rows
        ]
        for tag, rows in exp.all_histograms().items()
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
        "model_tags": list(models),
        "models": models,
        # Back-compat aliases for older clients/tests that may still query
        # the "graphs" key. New code should use "models".
        "graph_tags": list(models),
        "graphs": models,
        "pr_curve_tags": list(pr_curves),
        "pr_curves": pr_curves,
        "figure_tags": list(figures),
        "figures": figures,
        "mesh_tags": list(meshes),
        "meshes": meshes,
        "embedding_tags": list(embeddings),
        "embeddings": embeddings,
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
            media_dir = os.path.realpath(os.path.join(log_dir, "media"))
            roots.add(media_dir)
    return list(roots)


def _path_is_under(candidate: str, roots: Sequence[str]) -> bool:
    """Return True when candidate is contained by one of the allowed roots."""
    for root in roots:
        try:
            if os.path.commonpath([candidate, root]) == root:
                return True
        except ValueError:
            continue
    return False


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
        port: int = kwargs.get("port", 6116)
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
        host: str = "127.0.0.1",
        port: int = 6116,
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

        from ..cli import (
            _ALLOWED_SUFFIXES,
            _DANGEROUS_ARTIFACT_SUFFIXES,
            _coerce_remote_step,
            _validate_remote_experiment_name,
            _write_upload_to_tempfile,
        )
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

        # ── Auth / CSRF / rate-limit dependencies ─────────────
        # Defined here (before any route decorator) so they're in scope
        # when ``@app.get(..., dependencies=[Depends(_check_token)])``
        # decorators are evaluated at module import time.

        async def _check_token(request: Request) -> None:
            """Bearer-token auth used on every route when a token is set.

            When ``token`` is ``None`` (the default), no auth is enforced
            — preserving the localhost-only convenience for dev use. When
            ``token`` is supplied (typically with a non-loopback host),
            *all* routes require it, not just ingest. Constant-time
            comparison prevents a timing oracle.
            """
            if not token:
                return
            auth = request.headers.get("Authorization", "")
            if not hmac.compare_digest(auth, f"Bearer {token}"):
                raise HTTPException(status_code=401, detail="unauthorized")

        # Back-compat alias for ingest routes that previously used a
        # separate dependency name.
        _check_listen_auth = _check_token

        async def _check_mutation(request: Request) -> None:
            """Auth + CSRF + rate-limit gate for state-mutating endpoints.

            Layered defenses:
              1. Bearer token (when configured) — primary auth.
              2. Origin/Referer match — blocks CSRF from a same-browser
                 page on a different origin even when no token is set.
              3. Per-IP sliding-window rate limit — caps damage from a
                 runaway loop or buggy client.
            """
            await _check_token(request)
            if not _is_same_origin(request, host):
                raise HTTPException(status_code=403, detail="cross-origin denied")
            if not _rate_limit_mutation(_client_ip(request)):
                raise HTTPException(status_code=429, detail="rate limit exceeded")

        # ── Index / data ──────────────────────────────────────

        @app.get("/", dependencies=[Depends(_check_token)])
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

        @app.get("/api/data", dependencies=[Depends(_check_token)])
        def api_data() -> list:
            return _build_data(self._resolve_experiments(experiments))

        @app.get("/api/data/{project}", dependencies=[Depends(_check_token)])
        def api_data_project(project: str) -> Any:
            db = self._reader._db
            if db is None or not db.list_experiments(project=project):
                return JSONResponse({"error": "not found"}, status_code=404)
            return _build_data(_experiments_for_project(db, project))

        @app.get("/api/projects", dependencies=[Depends(_check_token)])
        def api_projects() -> List[str]:
            return _all_projects()

        # ── Config ────────────────────────────────────────────

        @app.get("/api/config", dependencies=[Depends(_check_token)])
        def get_config() -> dict:
            return load_config(project=config_project)

        @app.get("/api/config/{project}", dependencies=[Depends(_check_token)])
        def get_config_project(project: str) -> dict:
            return load_config(project=project)

        @app.post("/api/config", dependencies=[Depends(_check_mutation)])
        async def set_config(request: Request) -> dict:
            data = await request.json()
            save_config(data, project=config_project)
            return load_config(project=config_project)

        @app.post("/api/config/{project}", dependencies=[Depends(_check_mutation)])
        async def set_config_project(request: Request, project: str) -> dict:
            data = await request.json()
            save_config(data, project=project)
            return load_config(project=project)

        # ── Rename / delete / move ────────────────────────────

        @app.post("/api/rename", dependencies=[Depends(_check_mutation)])
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

        @app.post("/api/rename/{project}", dependencies=[Depends(_check_mutation)])
        async def rename_exp_project(request: Request, project: str) -> JSONResponse:
            body = await request.json()
            data, status = _handle_rename(
                self._reader._db,
                body.get("old_name", ""),
                body.get("new_name", ""),
                body.get("project") or project,
            )
            return JSONResponse(data, status_code=status)

        @app.delete("/api/experiment", dependencies=[Depends(_check_mutation)])
        async def delete_exp(request: Request) -> JSONResponse:
            body = await request.json()
            project = body.get("project") or self._reader.project
            data, status = _handle_delete_experiment(
                self._reader._db,
                body.get("name", ""),
                project,
            )
            return JSONResponse(data, status_code=status)

        @app.delete("/api/project/{project}", dependencies=[Depends(_check_mutation)])
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

        @app.post("/api/move-logdir", dependencies=[Depends(_check_mutation)])
        async def move_logdir(request: Request) -> JSONResponse:
            body = await request.json()
            data, status = _handle_move_logdir(
                self._reader._db,
                body.get("old", ""),
                body.get("new", ""),
            )
            return JSONResponse(data, status_code=status)

        # ── Media ─────────────────────────────────────────────

        @app.get("/media", dependencies=[Depends(_check_token)])
        def serve_media(path: str = ""):  # type: ignore[no-untyped-def]
            """Serve media files by absolute path."""
            if not path:
                return JSONResponse({"error": "not found"}, status_code=404)
            candidate = os.path.realpath(path)
            allowed_dirs = _get_media_roots(self._reader)
            if not _path_is_under(candidate, allowed_dirs):
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
                    resume=True,
                )
            return SummaryWriter(
                str(Path.cwd() / name),
                name=name,
                project=project_name,
                system_metrics_interval=0,
                resume=True,
            )

        @app.post("/{project}/listen/log")
        async def listen_log(
            project: str, request: Request, _=Depends(_check_listen_auth)
        ) -> dict:
            data = await request.json()
            try:
                experiment = _validate_remote_experiment_name(
                    data.get("experiment", "default")
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid experiment name")
            try:
                step = _coerce_remote_step(data.get("step", 0))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid step")
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
                try:
                    experiment = _validate_remote_experiment_name(experiment)
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="Invalid experiment name"
                    )
                raw_suffix = os.path.splitext(file.filename or "")[1].lower()
                if type not in {"image", "audio", "video", "artifact"}:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported upload type {type!r}"
                    )
                if type == "artifact":
                    if raw_suffix in _DANGEROUS_ARTIFACT_SUFFIXES:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Refusing to serve potentially dangerous file "
                                "extension"
                            ),
                        )
                else:
                    allowed = _ALLOWED_SUFFIXES.get(type)
                    if allowed is not None and raw_suffix not in allowed:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported file type for {type!r}",
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

        @app.get("/{project}", dependencies=[Depends(_check_token)])
        def project_index(project: str) -> HTMLResponse:
            db = self._reader._db
            if db is None:
                return HTMLResponse("no database", status_code=500)
            data = _build_data(_experiments_for_project(db, project))
            return HTMLResponse(_render(data, project, _all_projects()))

        return app
