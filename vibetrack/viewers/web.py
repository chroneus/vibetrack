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
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Collection, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse

from ..cli import _bearer_matches, _secret_equals
from ..compare import find_all_tags  # noqa: F401  (re-exported for tests)
from ..config import VIBETRACK_CONFIG_DIR, load_public_config, save_public_config
from ..db import log_dir_in_use, remove_experiment_files
from ..insights import run_card
from ..reader import ExperimentReader, RunReader
from ..storage import display_uri
from .base import BaseOutput

_WEB_DIR = Path(__file__).parent / "web"
_INDEX_PATH = _WEB_DIR / "index.html"

_SYSTEM_PREFIXES = ("system/", "gpu/")

# In-process rate limiter for state-mutating endpoints. 10 requests per
# second per source IP is generous for a human clicking buttons but
# stops a script from wiping a project in a tight loop.
_MUTATION_RATE_WINDOW_SEC = 1.0
_MUTATION_RATE_LIMIT = 10
# Past this many tracked IPs, idle ones are swept so the table can't grow
# without bound (e.g. one entry per IPv6 address).
_MUTATION_RATE_MAX_TRACKED = 1024
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
        if len(_mutation_rate_state) >= _MUTATION_RATE_MAX_TRACKED:
            idle = [ip for ip, b in _mutation_rate_state.items() if not b or b[-1] < cutoff]
            for ip in idle:
                del _mutation_rate_state[ip]
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


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_same_origin(request: Any, trusted_hosts: Collection[str] = ()) -> bool:
    """False if Origin/Referer names a different site than this server.

    The browser's own address for this server is the ``Host`` header, so
    the Origin's ``host[:port]`` must equal it — ports included, since a
    page on another localhost port is a different site. A missing header is
    treated as same-origin (curl, server-side scripts); browsers always send
    Origin on cross-origin POST/DELETE. ``trusted_hosts`` adds names a
    reverse proxy serves this app under (``--allow-host``).
    """
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        hostname = parsed.hostname
    except ValueError:
        return False
    if not parsed.netloc or not hostname:  # includes "Origin: null"
        return False
    if parsed.netloc.lower() == request.headers.get("host", "").lower():
        return True
    return hostname in trusted_hosts


def _scope_header(scope: Dict[str, Any], name: bytes) -> Optional[str]:
    for key, value in scope.get("headers", ()):
        if key == name:
            return value.decode("latin-1")
    return None


async def _send_json(send: Any, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


def _host_header_name(value: str) -> Optional[str]:
    try:
        return urlparse("//" + value).hostname
    except ValueError:
        return None


class _HostGuard:
    """ASGI middleware rejecting requests whose ``Host`` isn't an allowed name.

    Blocks DNS rebinding: a page on attacker.example can re-point its DNS at
    127.0.0.1 and then read the dashboard as "same-origin", but the browser
    still sends ``Host: attacker.example``. Installed for loopback binds (or
    when ``--allow-host`` names are given); on a plain LAN bind the valid
    names are unknown and ``--token`` is the defense.
    """

    def __init__(self, app: Any, allowed: Collection[str] = _LOOPBACK_HOSTS) -> None:
        self.app = app
        self.allowed = frozenset(allowed)

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            host_header = _scope_header(scope, b"host")
            if host_header is not None and _host_header_name(host_header) not in self.allowed:
                await _send_json(send, 403, {
                    "error": "invalid host header; serving behind a proxy or "
                             "under another name? start with --allow-host NAME",
                })
                return
        await self.app(scope, receive, send)


# Browsers can't send an Authorization header for page loads or <img>/<video>
# requests, so ``/?token=…`` trades the token for an HttpOnly cookie. The
# cookie is named after a hash of the token so two servers on one host
# (different ports share cookies) don't clobber each other.
def _auth_cookie(token: str) -> Tuple[str, str]:
    digest = hashlib.sha256(f"vibetrack-web\0{token}".encode()).hexdigest()
    return f"vibetrack_auth_{digest[:12]}", digest


def _require_bearer(app: Any, token: str) -> Any:
    """Wrap a mounted ASGI sub-app so it enforces the server's bearer token.

    Mounts bypass FastAPI route dependencies, so without this the MCP
    endpoint would stay open when ``--token`` protects everything else.
    """

    async def guarded(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not _bearer_matches(
            _scope_header(scope, b"authorization") or "", token
        ):
            await _send_json(send, 401, {"detail": "unauthorized"})
            return
        await app(scope, receive, send)

    return guarded


# Per-process cache-busting stamp. Static asset URLs in index.html get
# rewritten to include `?v=<stamp>` so a server restart forces browsers to
# refetch JS/CSS even if their HTTP cache holds the previous version.
_STATIC_VERSION = str(int(time.time()))

_TEMPLATE_PLACEHOLDER = re.compile(r"__(DATA|PROJECT|PROJECTS)_JSON__")


@functools.lru_cache(maxsize=1)
def _load_template() -> str:
    # Cache-bust local static URLs: src="/static/foo.js" → src="/static/foo.js?v=…"
    return re.sub(
        r'(src|href)="(/static/[^"?#]+)"',
        rf'\1="\2?v={_STATIC_VERSION}"',
        _INDEX_PATH.read_text(encoding="utf-8"),
    )


@functools.lru_cache(maxsize=32)
def _video_timeline(path: str, mtime_ns: int, size: int) -> dict:
    """Probe display-order timestamps on demand; stat keys invalidate replacements."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=best_effort_timestamp_time:stream=width,height,duration",
         "-of", "json", path], capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)
    times = [float(f["best_effort_timestamp_time"]) for f in data.get("frames", [])
             if "best_effort_timestamp_time" in f]
    if not times or any(not math.isfinite(t) for t in times):
        raise ValueError("No valid frame timestamps")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Frame timestamps are not strictly increasing")
    start = times[0]
    return {"times": [t - start for t in times], "frames": len(times)}


# Grid tiles show downscaled previews: decoding dozens of full-resolution PNGs,
# or opening one video decoder per tile, is what stalls the browser on
# media-heavy projects. Widths are bucketed so each file has few cached variants.
_THUMB_DIR = VIBETRACK_CONFIG_DIR / "thumbs"
_THUMB_WIDTHS = (320, 640, 1280, 2560)
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".ogv"}
# Animated / vector formats would lose what makes them useful; serve as-is.
_THUMB_PASSTHROUGH_SUFFIXES = {".gif", ".svg", ".apng"}
_THUMB_SAFE_MODES = {"1", "L", "LA", "P", "PA", "RGB", "RGBA"}
_thumb_slots = threading.BoundedSemaphore(max(1, min(4, os.cpu_count() or 1)))


def _thumb_bucket(width: int) -> int:
    return next((b for b in _THUMB_WIDTHS if b >= width), _THUMB_WIDTHS[-1])


def _is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_SUFFIXES


def _media_thumbnail(path: str, width: int) -> Optional[Path]:
    """Return a cached preview of *path* at most *width* px wide.

    ``None`` means the original should be served (already small, or a format
    we don't re-encode). Videos raise when no poster frame can be extracted.
    """
    if Path(path).suffix.lower() in _THUMB_PASSTHROUGH_SUFFIXES:
        return None
    stat = os.stat(path)
    key = hashlib.sha256(
        f"{path}\0{stat.st_mtime_ns}\0{stat.st_size}\0{width}".encode()
    ).hexdigest()
    cache_dir = _THUMB_DIR / key[:2]
    for cached in (cache_dir / f"{key}.jpg", cache_dir / f"{key}.png"):
        if cached.is_file():
            return cached
    with _thumb_slots:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if _is_video_path(path):
            return _video_poster(path, width, cache_dir / f"{key}.jpg")
        return _image_thumbnail(path, width, cache_dir, key)


def _video_poster(path: str, width: int, out: Path) -> Path:
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-frames:v", "1",
             "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", "-y", str(tmp)],
            capture_output=True, check=True, timeout=30)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise ValueError("ffmpeg produced no poster frame")
        os.replace(tmp, out)
    finally:
        with suppress(OSError):
            tmp.unlink()
    return out


def _remote_video_poster(remote: Any, path: str, width: int) -> Optional[Path]:
    """Poster frame of a not-yet-downloaded synced video, read via a presigned URL."""
    url = remote.direct_url(path, force=True)
    if not url:
        return None
    key = hashlib.sha256(f"{path}\0remote\0{width}".encode()).hexdigest()
    cache_dir = _THUMB_DIR / key[:2]
    out = cache_dir / f"{key}.jpg"
    if out.is_file():
        return out
    with _thumb_slots:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return _video_poster(url, width, out)


def _image_thumbnail(path: str, width: int, cache_dir: Path, key: str) -> Optional[Path]:
    try:
        from PIL import Image, ImageOps  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            # 16-bit / float images need tone mapping we can't guess; let the browser do it.
            if img.mode not in _THUMB_SAFE_MODES:
                return None
            if img.getexif().get(0x0112, 1) != 1:  # orientation: size the box as displayed
                img = ImageOps.exif_transpose(img)
            if img.width <= width:
                return None
            img.thumbnail((width, img.height))
            has_alpha = img.mode in ("LA", "PA", "RGBA") or "transparency" in img.info
            img = img.convert("RGBA" if has_alpha else "RGB")
            ext, fmt = ("png", "PNG") if has_alpha else ("jpg", "JPEG")
            out = cache_dir / f"{key}.{ext}"
            tmp = cache_dir / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                img.save(tmp, format=fmt, quality=88)
                os.replace(tmp, out)
            finally:
                with suppress(OSError):
                    tmp.unlink()
            return out
    except Exception:  # any decode failure falls back to the original
        logging.getLogger(__name__).debug("thumbnail failed for %s", path, exc_info=True)
        return None


def _json_for_html(obj: Any) -> str:
    """Serialize *obj* to JSON safe for embedding inside ``<script>``.

    Escaping every ``<`` (it only occurs inside JSON strings) rules out both
    ``</script>`` and ``<!--`` / ``<script``, which would otherwise flip the
    HTML parser into a state that swallows the rest of the page.
    """
    return json.dumps(obj).replace("<", r"\u003c")


def _render(
    data: list,
    project: Optional[str],
    projects: Sequence[str],
) -> str:
    """Render the index template with JSON-safe replacements.

    All placeholders are substituted in one pass: chained ``str.replace``
    calls would re-scan already-inserted JSON, so a logged value containing
    ``__PROJECT_JSON__`` could splice an unquoted project name into the
    script and execute it.
    """
    values = {
        "DATA": _json_for_html(data),
        "PROJECT": _json_for_html(project),
        "PROJECTS": _json_for_html(list(projects)),
    }
    return _TEMPLATE_PLACEHOLDER.sub(lambda m: values[m.group(1)], _load_template())


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
        "run": _run_card_safe(exp),
    }


def _run_card_safe(exp: ExperimentReader) -> Dict[str, Any]:
    """Runs-tab payload; a broken record must not take the dashboard down."""
    try:
        return run_card(exp)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "vibetrack web: run summary failed for %s: %s", exp.name, exc
        )
        return {"status": "unknown"}


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
    db = reader.db
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


# Shared with the writer (discarded runs) and `vibetrack push --remove-local`.
_log_dir_in_use = log_dir_in_use
_cleanup_experiment_files = remove_experiment_files


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
    _cleanup_experiment_files(log_dir or "", db)
    return {"ok": True}, 200


# Files vibetrack itself (or the OS) leaves next to run directories.
_MANAGED_SIBLING_FILES = frozenset({
    "vibetrack.db", "vibetrack.db-wal", "vibetrack.db-shm", "vibetrack.db-journal",
    ".DS_Store", "Thumbs.db", "desktop.ini",
})


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _handle_move_logdir(
    db: Any,
    old_dir: str,
    new_dir: str,
    db_path: Optional[str] = None,
) -> Tuple[dict, int]:
    """Move a run directory, or a folder holding only run directories.

    The UI offers each run's *parent* folder, so that is accepted too — but
    only when vibetrack owns everything in it, so a request can never move,
    say, a home directory that happens to contain a run. The destination
    must be a new absolute path inside an existing folder.
    """
    if not old_dir or not new_dir:
        return {"error": "old and new required"}, 400
    if not Path(new_dir).expanduser().is_absolute():
        return {"error": "destination must be an absolute path"}, 400
    old_path = Path(old_dir).expanduser().resolve()
    new_path = Path(new_dir).expanduser().resolve()
    if old_path == new_path:
        return {"ok": True}, 200
    if db is None:
        return {"error": "no database"}, 500

    known = {
        Path(r["log_dir"]).expanduser().resolve()
        for r in db.list_experiments()
        if r["log_dir"]
    }
    if old_path not in known:
        if not old_path.is_dir():
            return {"error": "not a known log directory"}, 403
        children = list(old_path.iterdir())
        runs = [c for c in children if c.resolve() in known]
        unmanaged = [
            c for c in children
            if c.resolve() not in known and c.name not in _MANAGED_SIBLING_FILES
        ]
        if not runs or unmanaged:
            return {"error": "not a known log directory"}, 403
    if not old_path.is_dir():
        return {"error": "source directory not found"}, 404
    if db_path and _is_within(Path(db_path).resolve(), old_path):
        return {"error": "directory holds the open database"}, 409
    if new_path.exists():
        return {"error": "destination already exists"}, 409
    if _is_within(new_path, old_path):
        return {"error": "cannot move a directory into itself"}, 400
    if not new_path.parent.is_dir():
        return {"error": "destination folder does not exist"}, 400

    try:
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


def _handle_rename_project(
    db: Any,
    old_project: str,
    new_name: str,
) -> Tuple[dict, int]:
    """Rename a project or migrate experiments via slash syntax."""
    new_name = new_name.strip()
    if not new_name:
        return {"error": "name cannot be empty"}, 400
    if db is None:
        return {"error": "no database"}, 500

    experiments = db.list_experiments(project=old_project)
    if not experiments:
        return {"error": "project not found"}, 404

    if "/" not in new_name:
        if new_name == old_project:
            return {"ok": True, "new_project": new_name}, 200
        success, conflicts = db.rename_project(old_project, new_name)
        if not success:
            if conflicts:
                return {
                    "error": "name conflicts in target project: " + ", ".join(conflicts)
                }, 409
            return {"error": "project not found"}, 404
        return {"ok": True, "new_project": new_name}, 200

    # Slash syntax: "target_project/experiment_base_name"
    parts = new_name.split("/", 1)
    target_project = parts[0].strip()
    base_exp_name = parts[1].strip()
    if not target_project or not base_exp_name:
        return {"error": "invalid format — use 'project/name'"}, 400

    moved = []
    for i, exp in enumerate(experiments):
        if len(experiments) == 1:
            exp_name = base_exp_name
        else:
            exp_name = base_exp_name if i == 0 else f"{base_exp_name} ({i + 1})"

        existing = db.get_experiment_by_name(exp_name, project=target_project)
        if existing is not None:
            exp_name = db.find_next_suffix_name(
                base_exp_name if i == 0 else exp_name, target_project
            )

        if not db.move_experiment_to_project(int(exp["id"]), target_project, exp_name):
            return {"error": f"failed to move experiment '{exp['name']}'"}, 500
        moved.append({"old": exp["name"], "new": exp_name})

    return {"ok": True, "target_project": target_project, "moved": moved}, 200


def _ui_url(host: str, port: int, token: Optional[str]) -> str:
    url = f"http://{host}:{port}"
    return f"{url}/?{urlencode({'token': token})}" if token else url


class WebOutput(BaseOutput):
    """Serve an interactive web dashboard."""

    def show(self, **kwargs: Any) -> Any:
        """Launch uvicorn server."""
        host: str = kwargs.get("host", "127.0.0.1")
        port: int = kwargs.get("port", 6116)
        token: Optional[str] = kwargs.get("token")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        allowed_hosts: Sequence[str] = kwargs.get("allowed_hosts") or ()
        return self._serve_uvicorn(
            experiments, host, port, token=token, allowed_hosts=allowed_hosts
        )

    def _serve_uvicorn(
        self,
        experiments: Optional[Sequence[str]],
        host: str,
        port: int,
        token: Optional[str] = None,
        allowed_hosts: Sequence[str] = (),
    ) -> None:
        import uvicorn

        app = self._build_app(experiments, host, token=token, allowed_hosts=allowed_hosts)
        print(f"vibetrack web UI: {_ui_url(host, port, token)}")
        uvicorn.run(app, host=host, port=port, log_level="warning")

    def start_in_thread(
        self,
        experiments: Optional[Sequence[str]] = None,
        host: str = "127.0.0.1",
        port: int = 6116,
        token: Optional[str] = None,
        allowed_hosts: Sequence[str] = (),
    ) -> "threading.Thread":
        """Start the web UI in a daemon thread and return immediately."""
        import threading
        import uvicorn

        app = self._build_app(experiments, host, token=token, allowed_hosts=allowed_hosts)
        print(f"vibetrack web UI: {_ui_url(_resolve_host(host), port, token)}")
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
        allowed_hosts: Sequence[str] = (),
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
        from fastapi.responses import (
            FileResponse,
            HTMLResponse,
            JSONResponse,
            RedirectResponse,
        )
        from fastapi.staticfiles import StaticFiles

        from ..cli import (
            _ALLOWED_SUFFIXES,
            _DANGEROUS_ARTIFACT_SUFFIXES,
            _central_ingest_log_dir,
            _coerce_remote_step,
            _parse_log_payload,
            _read_json_object,
            _validate_remote_experiment_name,
            _validate_remote_project,
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

        trusted_hosts = frozenset(h.strip().lower() for h in allowed_hosts if h.strip())
        app = FastAPI(title="vibetrack", lifespan=_lifespan)
        if host in _LOOPBACK_HOSTS or trusted_hosts:
            app.add_middleware(_HostGuard, allowed=_LOOPBACK_HOSTS | trusted_hosts | {host})
        app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

        config_project = self.config_project()
        # Synced storage may hold many projects, like the central DB.
        is_central = self.project_folder is None or self.is_remote
        remote = self._reader.remote

        def _all_projects() -> List[str]:
            db = self._reader.db
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
            comparison prevents a timing oracle. Browsers authenticate via
            the cookie that ``/?token=…`` sets (see ``_auth_cookie``).
            """
            if not token:
                return
            if (
                _bearer_matches(request.headers.get("Authorization", ""), token)
                or _secret_equals(request.cookies.get(cookie_name, ""), cookie_value)
                or _secret_equals(request.query_params.get("token", ""), token)
            ):
                return
            raise HTTPException(
                status_code=401,
                detail="unauthorized: send 'Authorization: Bearer <token>', "
                "or open /?token=<token> once in the browser",
            )

        def _login_redirect(request: Optional[Request]) -> Optional[RedirectResponse]:
            """Swap a valid ``?token=`` for the auth cookie and drop it from the URL."""
            if request is None:
                return None
            given = request.query_params.get("token")
            if not token or given is None or not _secret_equals(given, token):
                return None
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
            target = request.url.path + ("?" + urlencode(rest) if rest else "")
            response = RedirectResponse(target, status_code=303)
            response.set_cookie(
                cookie_name,
                cookie_value,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
                path="/",
            )
            return response

        cookie_name, cookie_value = _auth_cookie(token) if token else ("", "")

        async def _check_ingest(request: Request) -> None:
            """Auth + CSRF gate for the remote-logging routes.

            Remote loggers (urllib/requests) send no Origin and pass; a
            browser always sends one cross-origin, so a web page can't forge
            log entries. No rate limit: training loops post in bursts.
            """
            await _check_token(request)
            if not _is_same_origin(request, trusted_hosts):
                raise HTTPException(status_code=403, detail="cross-origin denied")

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
            if not _is_same_origin(request, trusted_hosts):
                raise HTTPException(status_code=403, detail="cross-origin denied")
            if not _rate_limit_mutation(_client_ip(request)):
                raise HTTPException(status_code=429, detail="rate limit exceeded")

        # ── Index / data ──────────────────────────────────────

        def _body_str(body: Dict[str, Any], key: str, required: bool = True) -> Any:
            value = body.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"'{key}' must be a string")
            return value

        def _write_back_edit(data: Dict[str, Any], status: int, project: Optional[str]) -> None:
            """After a dashboard rename of a synced run, record it in storage."""
            if remote is None or status != 200:
                return
            db = self._reader.db
            row = db.get_experiment_by_name(data.get("new_name", ""), project=project) if db else None
            if row is None:
                return
            try:
                remote.push_edits(db, [int(row["id"])])
            except Exception as exc:
                data.update(ok=False, error=f"renamed locally, but not in storage: {exc}")

        def _local_media(candidate: str) -> bool:
            """The file is on disk, fetching it from synced storage if needed."""
            if os.path.isfile(candidate):
                return True
            return remote is not None and remote.ensure_local(candidate)

        @app.get("/", dependencies=[Depends(_check_token)])
        def index(request: Request = None) -> Any:  # type: ignore[assignment]
            login = _login_redirect(request)
            if login is not None:
                return login
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
            db = self._reader.db
            if db is None or not db.list_experiments(project=project):
                return JSONResponse({"error": "not found"}, status_code=404)
            return _build_data(_experiments_for_project(db, project))

        @app.get("/api/projects", dependencies=[Depends(_check_token)])
        def api_projects() -> List[str]:
            return _all_projects()

        @app.get("/api/info", dependencies=[Depends(_check_token)])
        def api_info() -> dict:
            """Identify this server; ``vibetrack open`` reuses a matching one."""
            from .. import __version__

            return {
                "app": "vibetrack",
                "version": __version__,
                "db_path": str(self._reader.db_path),
                "project_folder": self.project_folder,
                "storage": display_uri(self.project_folder) if self.is_remote else None,
            }

        # ── Config ────────────────────────────────────────────

        # Config over HTTP never includes notification credentials: they are
        # neither returned nor overwritten (see ``load_public_config``).

        @app.get("/api/config", dependencies=[Depends(_check_token)])
        def get_config() -> dict:
            return load_public_config(project=config_project)

        @app.get("/api/config/{project}", dependencies=[Depends(_check_token)])
        def get_config_project(project: str) -> dict:
            return load_public_config(project=project)

        @app.post("/api/config", dependencies=[Depends(_check_mutation)])
        async def set_config(request: Request) -> dict:
            save_public_config(await _read_json_object(request), project=config_project)
            return load_public_config(project=config_project)

        @app.post("/api/config/{project}", dependencies=[Depends(_check_mutation)])
        async def set_config_project(request: Request, project: str) -> dict:
            save_public_config(await _read_json_object(request), project=project)
            return load_public_config(project=project)

        # ── Rename / delete / move ────────────────────────────

        @app.post("/api/rename", dependencies=[Depends(_check_mutation)])
        async def rename_exp(request: Request) -> JSONResponse:
            body = await _read_json_object(request)
            project = _body_str(body, "project", required=False) or self._reader.project
            data, status = _handle_rename(
                self._reader.db,
                _body_str(body, "old_name"),
                _body_str(body, "new_name"),
                project,
            )
            _write_back_edit(data, status, project)
            return JSONResponse(data, status_code=status)

        @app.post("/api/rename/{project}", dependencies=[Depends(_check_mutation)])
        async def rename_exp_project(request: Request, project: str) -> JSONResponse:
            body = await _read_json_object(request)
            project = _body_str(body, "project", required=False) or project
            data, status = _handle_rename(
                self._reader.db,
                _body_str(body, "old_name"),
                _body_str(body, "new_name"),
                project,
            )
            _write_back_edit(data, status, project)
            return JSONResponse(data, status_code=status)

        @app.delete("/api/experiment", dependencies=[Depends(_check_mutation)])
        async def delete_exp(request: Request) -> JSONResponse:
            body = await _read_json_object(request)
            project = _body_str(body, "project", required=False) or self._reader.project
            name = _body_str(body, "name")
            db = self._reader.db
            row = db.get_experiment_by_name(name, project=project) if (remote and db and name) else None
            data, status = _handle_delete_experiment(db, name, project)
            if remote is not None and status == 200 and row is not None:
                try:
                    remote.delete_runs([int(row["id"])])
                except Exception as exc:
                    return JSONResponse(
                        {"error": f"deleted locally, but not from storage: {exc}"}, status_code=502
                    )
            return JSONResponse(data, status_code=status)

        @app.delete("/api/project/{project}", dependencies=[Depends(_check_mutation)])
        def delete_project(project: str) -> JSONResponse:
            db = self._reader.db
            if db is None:
                return JSONResponse({"error": "no database"}, status_code=500)
            rows = db.list_experiments(project=project)
            if not rows:
                return JSONResponse({"error": "not found"}, status_code=404)
            db.delete_experiments_by_project(project)
            if remote is not None:
                try:
                    remote.delete_runs([int(r["id"]) for r in rows])
                except Exception as exc:
                    return JSONResponse(
                        {"error": f"deleted locally, but not from storage: {exc}"}, status_code=502
                    )
            for row in rows:
                log_dir = row["log_dir"] if hasattr(row, "__getitem__") else ""
                if log_dir:
                    _cleanup_experiment_files(str(log_dir), db)
            return JSONResponse({"ok": True})

        @app.post("/api/move-logdir", dependencies=[Depends(_check_mutation)])
        async def move_logdir(request: Request) -> JSONResponse:
            body = await _read_json_object(request)
            if remote is not None:
                return JSONResponse(
                    {"error": "log directories of synced storage cannot be moved"},
                    status_code=400,
                )
            data, status = _handle_move_logdir(
                self._reader.db,
                _body_str(body, "old"),
                _body_str(body, "new"),
                db_path=str(self._reader.db_path),
            )
            return JSONResponse(data, status_code=status)

        @app.post(
            "/api/rename-project/{project}",
            dependencies=[Depends(_check_mutation)],
        )
        async def rename_project_route(request: Request, project: str) -> JSONResponse:
            body = await _read_json_object(request)
            db = self._reader.db
            ids = [int(r["id"]) for r in db.list_experiments(project=project)] if (remote and db) else []
            data, status = _handle_rename_project(db, project, _body_str(body, "new_name"))
            if remote is not None and status == 200 and ids:
                try:
                    remote.push_edits(db, ids)
                except Exception as exc:
                    return JSONResponse(
                        {"error": f"renamed locally, but not in storage: {exc}"}, status_code=502
                    )
            return JSONResponse(data, status_code=status)

        # ── Media ─────────────────────────────────────────────

        @app.get("/media", dependencies=[Depends(_check_token)])
        def serve_media(path: str = "", proxy: int = 0):  # type: ignore[no-untyped-def]
            """Serve media files by absolute path.

            For synced storage, videos and large files redirect to a
            short-lived presigned URL so the browser streams them directly
            (with seeking); everything else is fetched once into the local
            cache. ``proxy=1`` forces the cache (same-origin, e.g. for
            drawing video frames on a canvas).
            """
            if not path:
                return JSONResponse({"error": "not found"}, status_code=404)
            candidate = os.path.realpath(path)
            allowed_dirs = _get_media_roots(self._reader)
            if not _path_is_under(candidate, allowed_dirs):
                return JSONResponse({"error": "not found"}, status_code=404)
            if os.path.isfile(candidate):
                return FileResponse(candidate)
            if remote is not None and not proxy:
                url = remote.direct_url(candidate)
                if url:
                    return RedirectResponse(url, status_code=307)
            if _local_media(candidate):
                return FileResponse(candidate)
            return JSONResponse({"error": "not found"}, status_code=404)

        @app.get("/media/thumb", dependencies=[Depends(_check_token)])
        def serve_media_thumb(path: str = "", w: int = 640):  # type: ignore[no-untyped-def]
            """Downscaled image, or a video's first frame, for grid previews."""
            candidate = os.path.realpath(path)
            if not path or not _path_is_under(candidate, _get_media_roots(self._reader)):
                return JSONResponse({"error": "not found"}, status_code=404)
            if (
                remote is not None
                and not os.path.isfile(candidate)
                and _is_video_path(candidate)
            ):
                # Poster straight from storage: ffmpeg reads only the first
                # frame over HTTP instead of downloading the whole video.
                try:
                    thumb = _remote_video_poster(remote, candidate, _thumb_bucket(w))
                except (OSError, ValueError, subprocess.SubprocessError):
                    thumb = None
                if thumb is not None:
                    return FileResponse(thumb)
            if not _local_media(candidate):
                return JSONResponse({"error": "not found"}, status_code=404)
            try:
                thumb = _media_thumbnail(candidate, _thumb_bucket(w))
            except (OSError, ValueError, subprocess.SubprocessError):
                thumb = None
            if thumb is not None:
                return FileResponse(thumb)
            if _is_video_path(candidate):
                # The client falls back to decoding the first frame itself.
                return JSONResponse({"error": "preview unavailable"}, status_code=422)
            return FileResponse(candidate)

        @app.get("/api/video-timeline", dependencies=[Depends(_check_token)])
        def video_timeline(path: str = ""):
            candidate = os.path.realpath(path)
            if not path or not _path_is_under(candidate, _get_media_roots(self._reader)) or not _local_media(candidate):
                return JSONResponse({"error": "not found"}, status_code=404)
            try:
                stat = os.stat(candidate)
                return JSONResponse(_video_timeline(candidate, stat.st_mtime_ns, stat.st_size))
            except (OSError, ValueError, subprocess.SubprocessError):
                return JSONResponse({"error": "Frame timestamps unavailable for this video"}, status_code=422)

        # ── Ingest (listen) routes ────────────────────────────

        normalized_pf = (
            str(Path(self.project_folder).resolve())
            if self.project_folder is not None and not self.is_remote
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
                    report=False,
                    track_status=False,
                    # One writer per request: syncing would re-upload every time.
                    sync="",
                )
            return SummaryWriter(
                _central_ingest_log_dir(name, project_name),
                name=name,
                project=project_name,
                system_metrics_interval=0,
                resume=True,
                report=False,
                track_status=False,
                sync="",
            )

        def _ingest_project(project: str) -> str:
            if self.is_remote:
                raise HTTPException(
                    status_code=400,
                    detail="this dashboard shows synced storage; log runs with "
                    "VIBETRACK_SYNC on the training machine instead of remote ingest",
                )
            # Central mode turns the project into a directory name.
            if normalized_pf is not None:
                return project
            try:
                return _validate_remote_project(project) or project
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid project name")

        @app.post("/{project}/listen/log")
        async def listen_log(
            project: str, request: Request, _=Depends(_check_ingest)
        ) -> dict:
            data = await _read_json_object(request)
            project = _ingest_project(project)
            try:
                experiment = _validate_remote_experiment_name(
                    data.get("experiment", "default")
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid experiment name")
            try:
                step = _coerce_remote_step(data.get("step", 0))
                scalars, texts, _ = _parse_log_payload(data)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            writer = _new_writer(experiment, project)
            try:
                for tag, value in scalars.items():
                    writer.add_scalar(tag, value, step)
                for tag, value in texts.items():
                    writer.add_text(tag, value, step)
            finally:
                writer.close()
            return {"status": "ok"}

        @app.post("/{project}/listen/media")
        async def listen_media(
            project: str,
            _=Depends(_check_ingest),
            experiment: str = Form("default"),
            tag: str = Form("upload"),
            step: int = Form(0),
            type: str = Form("artifact"),
            file: UploadFile = File(...),
        ) -> dict:
            writer: Optional[SummaryWriter] = None
            tmp_path: Optional[str] = None
            try:
                project = _ingest_project(project)
                try:
                    experiment = _validate_remote_experiment_name(experiment)
                    step = _coerce_remote_step(step)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
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
            app.mount(
                "/vibetrack_mcp",
                _require_bearer(_mcp_asgi, token) if token else _mcp_asgi,
            )

        # ── Project-scoped index (must be registered last so fixed
        #     paths above take precedence over ``/{project}``) ──

        @app.get("/{project}", dependencies=[Depends(_check_token)])
        def project_index(project: str, request: Request = None) -> Any:  # type: ignore[assignment]
            login = _login_redirect(request)
            if login is not None:
                return login
            db = self._reader.db
            if db is None:
                return HTMLResponse("no database", status_code=500)
            data = _build_data(_experiments_for_project(db, project))
            return HTMLResponse(_render(data, project, _all_projects()))

        return app
