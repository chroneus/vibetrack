# vibetrack

Modern, lightweight experiment tracker for ML/AI. Drop-in replacement for TensorBoard and W&B with multiple output backends.

## Project overview

SQLite-backed (WAL mode, stdlib `sqlite3`) experiment tracker that logs scalars, images, audio, video, text, histograms, and arbitrary artifacts. Supports comparing results (including image outputs) across runs side-by-side.

### Output backends (auto-discovered)

| Backend  | Module                        | Extra                          |
|----------|-------------------------------|--------------------------------|
| Web UI   | `vibetrack/viewers/web.py`    | included in base deps          |
| Gradio   | `vibetrack/viewers/gradio_ui.py` | `pip install vibetrack[gradio]` |
| Console  | `vibetrack/viewers/console.py`| (no extra deps)                |
| Telegram | `vibetrack/viewers/telegram.py`| `pip install vibetrack[telegram]` |
| Slack    | `vibetrack/viewers/slack.py`  | (no extra deps; webhook-based) |
| MCP      | `vibetrack/viewers/mcp.py`    | `pip install vibetrack[all]` on Python >= 3.10 |

Viewers are auto-discovered from `vibetrack/viewers/` — any `.py` file except `__init__.py`, `base.py`, and `event.py` becomes available via `--viewer=name` if it exposes a `BaseOutput` subclass. Filenames ending in `_ui.py` drop the suffix, so `gradio_ui.py` becomes `--viewer=gradio`.

## Architecture

```
vibetrack/
  __init__.py    — Public API, W&B-style module-level init/log/finish
  config.py      — User config (~/.vibetrack/config.json; migrates legacy ~/.cache/vibetrack/config.json)
  default_config.py — Central default values used by config/writer/viewers
  db.py          — Database class (SQLite WAL, thread-local conns, bulk insert, precache)
  writer.py      — SummaryWriter (TB-compatible + W&B-style .log())
  reader.py      — ExperimentReader + RunReader (central DB or explicit project DB)
  smoother.py    — EMA (TB-style debiased), moving average, gaussian
  compare.py     — Cross-experiment comparison (scalars, hparams, summary tables)
  types.py       — Media wrapper types: Image, Audio, Video, Artifact
  media.py       — File saving for media (lazy imports, zero-dep at import time)
  sysmetrics.py  — Optional OS/GPU system metrics collection
  cli.py         — CLI: project-folder workflow, viewer auto-discovery, HTTP ingest, migration
  viewers/
    __init__.py  — Auto-discovery: discover_viewers(), load_viewer()
    base.py      — BaseOutput abstract class (show(**kwargs))
    event.py     — LogEvent/EventHandle dispatch helpers for writer.to() and handle.to()
    web.py       — FastAPI + Chart.js web dashboard, unified ingest, optional MCP mount, static assets under viewers/web/
    gradio_ui.py — Gradio dashboard
    console.py   — Terminal output
    telegram.py  — Telegram bot notifications
    slack.py     — Slack incoming-webhook notifications
    mcp.py       — MCP server (FastMCP tools + resources; streamable HTTP by default)
    web/
      index.html — Web UI shell
      css/       — Web UI styles
      js/        — Web UI modules (core/charts/pills/media/settings/main)
```

## Key design decisions

- **Lightweight** — output backends use lazy imports; heavier integrations stay behind optional extras where practical
- Default writes go to the central DB at `~/.vibetrack/vibetrack.db`; pass `project_folder=...` to write a project-local `vibetrack.db`
- Experiments are unique by `(project, name)` and keep `log_dir` for resolving relative media paths
- Scalar writes are buffered (default 1000) then bulk-flushed via `executemany`
- Existing run names are resolved lazily on first write: later steps resume the run, earlier/overlapping steps create a suffixed restart such as `exp (2)` with an isolated log directory
- Only distributed rank 0 logs by default, based on `RANK` / `LOCAL_RANK`; pass `rank="all"` to force all ranks to write
- Web UI serves packaged static assets from `vibetrack/viewers/web/`; there is no frontend build step
- **Precache mode** (`precache_secs`): holds all data in memory, defers DB creation; daemon timer materializes after timeout; if process dies before timeout, no stale files
- Media files stored under `<log_dir>/media/<tag>/<step>.<ext>`, paths in DB are relative for portability
- **Config system**: `~/.vibetrack/config.json` stores smoothing, system metric cadence, theme, auto-refresh, image playback FPS, raw scalar opacity, x-axis mode, and viewer-specific settings. It supports both legacy flat config and `{default, projects}` project-scoped config. Legacy `~/.cache/vibetrack/config.json` is copied forward if present. Viewers read config on startup; web UI Settings tab writes back.
- **Viewer auto-discovery**: `viewers/__init__.py` scans viewer modules, excluding infrastructure files, and maps filename to viewer name (stripping `_ui` suffix)
- **Event dispatch**: every `add_*` and W&B-style `log()` call returns an event handle; use `writer.to("telegram" | "slack" | ...)` for registered dispatch or `handle.to(...)` for one-shot forwarding
- **Best-effort side-effects never crash the caller**: telemetry, notifications, and any network/external export (Telegram, Slack, HTTP ingest, logging hooks, etc.) must catch exceptions, log a warning, and return gracefully. Missing credentials (e.g. `VIBETRACK_TELEGRAM_TOKEN`/`VIBETRACK_TELEGRAM_CHAT_ID`) should log and no-op rather than raise. Delivery failures for one item must not abort the rest. The user's training loop is the priority — a failed Telegram send or down webhook must never bring down their Python process. The `SummaryWriter._best_effort` / `_best_effort_emit` pattern in `writer.py` is the reference; viewers and external exporters follow the same rule.

## Two API styles

```python
# TensorBoard style
from vibetrack import SummaryWriter
writer = SummaryWriter("runs/exp1", project_folder="my_project")
writer.add_scalar("loss", 0.5, step=0)
writer.add_image("sample", img_array, step=0)
writer.close()

# W&B style
import vibetrack
vibetrack.init(project="cifar10", name="resnet18", config={"lr": 1e-3})
vibetrack.log({"loss": 0.5, "sample": vibetrack.Image(img_array)})
vibetrack.finish()
```

## CLI

Flat argparse CLI with one special `migrate` literal:

```
vibetrack                                    # default: central DB, --viewer=web, port 6006
vibetrack my_project/ --viewer=web           # project-local web dashboard
vibetrack --project-folder=my_project/       # explicit project-local dashboard
vibetrack --viewer=console                   # terminal summary
vibetrack --viewer=gradio                    # Gradio dashboard
vibetrack --listen=0.0.0.0:9009              # start HTTP ingest server
vibetrack --listen=0.0.0.0:9009 --token=SECRET  # with bearer auth
vibetrack --host=127.0.0.1 --port=8080      # restrict to localhost only
vibetrack --viewer=mcp --mcp-transport=sse  # standalone MCP with SSE instead of streamable HTTP
vibetrack migrate PROJECT_FOLDER             # merge legacy per-run DBs into project DB
```

### HTTP ingest server (`--listen`)

Receives log data from remote systems via FastAPI HTTP API:
- `POST /log` — JSON `{experiment, step, scalars: {}, texts: {}}`
- `POST /media` — multipart form `{experiment, tag, step, type, file}`
- Web viewer also exposes project-scoped ingest routes on the same server: `POST /{project}/listen/log` and `POST /{project}/listen/media`
- Bearer token auth via `--token` flag
- Runs alongside the viewer in a daemon thread

### Web server routes

- `/` redirects central-DB users to the most recently active project when projects exist
- `/{project}` renders a project-scoped dashboard
- `/api/data` and `/api/data/{project}` return serialized experiment data
- `/api/config` and `/api/config/{project}` read/write config
- `/api/rename`, `/api/rename/{project}`, `/api/experiment`, `/api/project/{project}`, and `/api/move-logdir` support dashboard management actions
- `/media?path=...` serves only files under known experiment `media/` roots
- `/vibetrack_mcp` mounts the MCP sub-app when the optional MCP dependency is installed

## Config

Stored at `~/.vibetrack/config.json`:

```json
{
  "smoothing": "ema",
  "smooth_weight": 0.6,
  "system_metrics_interval": 3600,
  "web": {
    "theme": "light",
    "auto_refresh": 5,
    "image_play_fps": 4,
    "raw_scalar_opacity": 0.17,
    "x_axis_mode": "step"
  },
  "gradio": { "share": false },
  "console": {},
  "telegram": {},
  "mcp": {}
}
```

Project-scoped config is also supported:

```json
{
  "default": { "web": { "theme": "light" } },
  "projects": {
    "cifar10": { "web": { "theme": "orange" } }
  }
}
```

## Web UI tabs (dynamic)

Tabs are data-driven; Settings is always shown:
- **Scalars** — Chart.js line charts with smoothing, experiment pills
- **Images** — lazy-loaded grid, step playback, side-by-side/lens/blend comparison
- **Audio** — HTML5 audio players
- **Video** — HTML5 video players and comparison overlay
- **Artifacts** — download table with file size/MIME
- **Text** — monospace formatted text entries with copy controls
- **Histograms** — Chart.js bar charts
- **System** — system/GPU metrics (tags starting with `system/` or `gpu/`)
- **Settings** — smoothing, theme, auto-refresh, image playback FPS, raw scalar opacity, and x-axis mode (persisted to config)
- **Run/project management** — experiment pills support visibility, color, rename, and delete; Settings includes log-dir move and project deletion controls

## Tests

- Current full suite on macOS/Python 3.11: 234 passed, 2 skipped (`pytest tests/`)
- Use `tmp_path` fixture for all DB/file tests
- `multi_run_dir` fixture creates realistic multi-experiment setups
- Performance test: 10k bulk inserts must complete < 1s

```
pytest tests/
```

## Development

```
pip install -e .           # install base package
pip install -e ".[all]"    # install optional backends plus test deps
```

Python >= 3.8.
