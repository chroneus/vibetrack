# vibetrack

Modern, lightweight experiment tracker for ML/AI. Drop-in replacement for TensorBoard and W&B with multiple output backends.

## Project overview

SQLite-backed (WAL mode, stdlib `sqlite3`) experiment tracker that logs scalars, images, audio, video, text, histograms, and arbitrary artifacts. Supports comparing results (including image outputs) across runs side-by-side.

### Output backends (all optional extras, auto-discovered)

| Backend  | Module                        | Extra                          |
|----------|-------------------------------|--------------------------------|
| Web UI   | `vibetrack/viewers/web.py`    | `pip install vibetrack[web]`   |
| Gradio   | `vibetrack/viewers/gradio_ui.py` | `pip install vibetrack[gradio]` |
| Console  | `vibetrack/viewers/console.py`| (no extra deps)                |
| Telegram | `vibetrack/viewers/telegram.py`| `pip install vibetrack[telegram]` |
| MCP      | `vibetrack/viewers/mcp.py`    | `pip install vibetrack[mcp]`   |

Viewers are auto-discovered from `vibetrack/viewers/` — any `.py` file with a `BaseOutput` subclass becomes available via `--viewer=name`.

## Architecture

```
vibetrack/
  __init__.py    — Public API, W&B-style module-level init/log/finish
  config.py      — User config (~/.cache/vibetrack/config.json)
  db.py          — Database class (SQLite WAL, thread-local conns, bulk insert, precache)
  writer.py      — SummaryWriter (TB-compatible + W&B-style .log())
  reader.py      — ExperimentReader + RunReader (discovers DBs in logdir tree)
  smoother.py    — EMA (TB-style debiased), moving average, gaussian
  compare.py     — Cross-experiment comparison (scalars, hparams, summary tables)
  types.py       — Media wrapper types: Image, Audio, Video, Artifact
  media.py       — File saving for media (lazy imports, zero-dep at import time)
  sysmetrics.py  — Optional OS/GPU system metrics collection
  cli.py         — CLI: flat options, viewer auto-discovery, HTTP ingest server
  viewers/
    __init__.py  — Auto-discovery: discover_viewers(), load_viewer()
    base.py      — BaseOutput abstract class (show(**kwargs))
    web.py       — FastAPI + inline Chart.js (no build step, no static files)
    gradio_ui.py — Gradio dashboard
    console.py   — Terminal output
    telegram.py  — Telegram bot notifications
    mcp.py       — MCP server (tools + resources)
```

## Key design decisions

- **Lightweight** — all output backends use lazy imports and optional extras
- Each logdir gets a `vibetrack.db` file; multiple experiments coexist as rows in the same DB
- Scalar writes are buffered (default 1000) then bulk-flushed via `executemany`
- Web UI uses inline Chart.js — no build step, no static file serving
- **Precache mode** (`precache_secs`): holds all data in memory, defers DB creation; daemon timer materializes after timeout; if process dies before timeout, no stale files
- Media files stored under `<log_dir>/media/<tag>/<step>.<ext>`, paths in DB are relative for portability
- **Config system**: `~/.cache/vibetrack/config.json` stores smoothing, theme, viewer-specific settings. Viewers read config on startup; web UI Settings tab writes back.
- **Viewer auto-discovery**: `viewers/__init__.py` scans `*.py` files, finds `BaseOutput` subclasses, maps filename to viewer name (stripping `_ui` suffix)

## Two API styles

```python
# TensorBoard style
from vibetrack import SummaryWriter
writer = SummaryWriter("runs/exp1")
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

No subcommands — flat options only:

```
vibetrack                                    # default: --logdir=. --viewer=web
vibetrack --logdir=runs/ --viewer=web        # web dashboard on port 6006
vibetrack --viewer=console                   # terminal summary
vibetrack --viewer=gradio                    # Gradio dashboard
vibetrack --listen=0.0.0.0:9009              # start HTTP ingest server
vibetrack --listen=0.0.0.0:9009 --token=SECRET  # with bearer auth
vibetrack --host=127.0.0.1 --port=8080      # restrict to localhost only
```

### HTTP ingest server (`--listen`)

Receives log data from remote systems via FastAPI HTTP API:
- `POST /log` — JSON `{experiment, step, scalars: {}, texts: {}}`
- `POST /media` — multipart form `{experiment, tag, step, type, file}`
- Bearer token auth via `--token` flag
- Runs alongside the viewer in a daemon thread

## Config

Stored at `~/.cache/vibetrack/config.json`:

```json
{
  "smoothing": "ema",
  "smooth_weight": 0.6,
  "web": { "theme": "dark", "auto_refresh": 0 },
  "gradio": { "share": false }
}
```

## Web UI tabs (dynamic)

Only tabs with data are shown:
- **Scalars** — Chart.js line charts with smoothing, experiment pills
- **Images** — lazy-loaded grid
- **Audio** — HTML5 audio players
- **Video** — HTML5 video players
- **Artifacts** — download table with file size/MIME
- **Text** — monospace formatted text entries
- **Histograms** — Chart.js bar charts
- **System** — system/GPU metrics (tags starting with `system/` or `gpu/`)
- **Settings** — smoothing, theme, auto-refresh (persisted to config)

## Tests

- 158 tests, all passing
- Use `tmp_path` fixture for all DB/file tests
- `multi_run_dir` fixture creates realistic multi-experiment setups
- Performance test: 10k bulk inserts must complete < 1s

```
pytest tests/
```

## Development

```
pip install -e ".[dev]"    # install with test deps
pip install -e ".[all]"    # install all optional backends
```

Python >= 3.8.
