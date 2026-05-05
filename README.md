# vibetrack

Lightweight experiment tracking.

Key features:
    Run locally but could receive experiment data over network via REST API. 
    Open formats: store experiment data in sqlite and local files. 
    Image2image comparison. 
    Rich UI.
    Send experiment results to Gradio/Telegram/Slack
    TensorBoard or W&B drop-in API replacement 
    MCP server with results



## Install

```bash
pip install vibetrack          # default with web
pip install vibetrack[all]     # all optional backends+dev; MCP on Python >=3.10
```

## Quick start

### TensorBoard-style API

```python
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/exp1", project_folder="my_project")
for step in range(100):
    writer.add_scalar("loss", 1.0 / (step + 1), step)
    writer.add_scalar("acc", step / 100, step)
writer.close()
```

### W&B-style API

```python
import vibetrack

vibetrack.init(project="my_project", name="run_1", config={"lr": 0.01, "epochs": 50})
for step in range(100):
    vibetrack.log({"loss": 1.0 / (step + 1), "acc": step / 100})
vibetrack.finish()
```

### Send events to Telegram, Slack, console, gradio

Every `add_*` and `log()` call returns a handle with `.to(name, **creds)` for
ad-hoc dispatch, and `writer.to(name, every=...)` registers a persistent
destination. Fire-and-forget: adapter exceptions never bubble up to the
training loop.

```python
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/exp1", project_folder="my_project")

# Register destinations; chainable, one per call.
writer = (
    writer
    .to("console")                          # every event -> stdout
    .to("telegram", every=100)              # every 100 events
    .to("slack", every="15m")               # time-based digest
)

for step in range(1000):
    writer.add_scalar("loss", 1.0 / (step + 1), step)

# Per-event one-shot: send just this event, independent of registration.
writer.add_image("samples", "out.png", step=999).to("telegram")

writer.close()
```

`every=` accepts `None` (every event), an `int` (N events), or a duration
string (`"5s"`, `"15m"`, `"1h"`, `"2d"`).

**Adapters** (under `vibetrack/viewers/`):
- `"telegram"` — needs `VIBETRACK_TELEGRAM_TOKEN` + `VIBETRACK_TELEGRAM_CHAT_ID` (or `token=` / `chat_id=`)
- `"slack"` — needs `SLACK_WEBHOOK_URL` (or `webhook=`)
- `"console"` — prints one line per event to stdout
- `"gradio"` — buffers events for a running Gradio dashboard
- `"remote"` — forwards events to another vibetrack server (see *Track to a remote server* below)
- The built-in web dashboard / SQLite store is always active; `.to(...)` adds *additional* destinations.

W&B-style equivalent: `vibetrack.init(..., to=["console", "telegram"])`.

### Track to a remote server

Run `vibetrack --listen HOST:PORT` on a peer machine to expose `/log`, `/media`,
and `/hparams` ingest endpoints. Then point any training run at it with
`writer.to("remote", url=..., token=...)`:

```bash
# Server side
vibetrack --listen 0.0.0.0:8080 --token devtoken --project-folder /srv/runs
```

```python
# Client side
import vibetrack

writer = vibetrack.init(project="cifar10", name="resnet18")
writer.to("remote",
          url="http://server:8080",
          token="devtoken",
          every="10m")          # batch dispatches every 10 minutes

vibetrack.log({"loss": 0.5})
```

W&B-style at init time:

```python
vibetrack.init(
    project="cifar10", name="resnet18",
    to=[{"name": "remote", "url": "http://server:8080", "token": "devtoken", "every": "10m"}],
)
```

All event kinds round-trip — scalars, texts, histograms, images, audio, video,
artifacts, and hparams. Local SQLite stays the source of truth; if the remote
server is unreachable the adapter logs one warning and keeps training going.

### Launch the dashboard

```bash
vibetrack 
# -> Web UI + ingest endpoint on http://0.0.0.0:6006
# -> MCP is also mounted when installed with vibetrack[all] on Python 3.10+
```

## API reference

### SummaryWriter

```python
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/exp1", name="experiment_name", project_folder="project/")

# Scalars
writer.add_scalar("loss", 0.5, step=0)
writer.add_scalars("metrics", {"train_loss": 0.5, "val_loss": 0.6}, step=0)

# Images — accepts file paths, numpy arrays, or PIL Images
writer.add_image("samples", "path/to/image.png", step=0)

# Audio — accepts file paths or numpy waveforms
writer.add_audio("speech", waveform_array, step=0, sample_rate=16000)

# Video
writer.add_video("rollout", "path/to/video.mp4", step=0)

# Artifacts — any file with optional metadata
writer.add_artifact("checkpoint", "model.pt", step=0, metadata={"val_acc": 0.95})

# Text
writer.add_text("notes", "Training started with lr=0.01", step=0)

# Histograms
writer.add_histogram("weights", weight_tensor, step=0)

# Hyperparameters
writer.add_hparams({"lr": 0.01, "batch_size": 32}, {"best_acc": 0.95})

writer.close()
```

### W&B-compatible module API

```python
import vibetrack
from vibetrack import Image, Audio, Video, Artifact

vibetrack.init(project="nlp", name="bert-finetune", config={"lr": 3e-5})

# Log scalars
vibetrack.log({"loss": 0.3, "acc": 0.92})

# Log media with wrapper types
vibetrack.log({"sample": Image("output.png")})
vibetrack.log({"audio": Audio("clip.wav", sample_rate=22050)})
vibetrack.log({"demo": Video("result.mp4")})
vibetrack.log({"model": Artifact("best_model.pt", metadata={"epoch": 10})})

# Access config
vibetrack.config["lr"]  # 3e-5

vibetrack.finish()
```

### Comparison and analysis

```python
from vibetrack import RunReader
from vibetrack.compare import compare_scalars, compare_hparams, summary_table

reader = RunReader("my_project/")
experiments = reader.experiments()

# Summary table — last value of each tag per experiment
summary_table(experiments, tags=["loss", "acc"])

# Compare scalars with smoothing
compare_scalars(experiments, "loss", smoothing="ema", weight=0.6)

# Side-by-side hyperparameter comparison
compare_hparams(experiments)
```

## Distributed training (torchrun)

vibetrack automatically detects `RANK` / `LOCAL_RANK` environment variables. Only rank 0 logs data — all other ranks get a silent no-op writer.

```bash
torchrun --nproc_per_node=4 --nnodes=2 train.py
```

```python
# train.py — no code changes needed
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/distributed", project_folder="project/")
# Only rank 0 writes to the database. Other ranks silently skip.
writer.add_scalar("loss", loss.item(), step)
writer.close()
```

Force all ranks to log:

```python
writer = SummaryWriter("runs/distributed", rank="all")
```

## Remote logging over HTTP

vibetrack's built-in ingest endpoints accept metrics from remote machines:

```bash
# Server (included in default web server)
vibetrack my_project/ --token mysecret
# -> Ingest at http://host:6006/{project}/listen/log
```

```python
# Remote client
import requests

requests.post("http://server:6006/my_project/listen/log", json={
    "experiment": "remote_run",
    "step": 42,
    "scalars": {"loss": 0.3, "acc": 0.91},
    "texts": {"note": "checkpoint saved"},
}, headers={"Authorization": "Bearer mysecret"})
```

Upload media:

```python
requests.post("http://server:6006/my_project/listen/media",
    data={"experiment": "remote_run", "tag": "sample", "step": "0", "type": "image"},
    files={"file": open("output.png", "rb")},
    headers={"Authorization": "Bearer mysecret"},
)
```

## System metrics

Built-in collection of CPU, GPU, memory, and disk metrics. Runs in a background thread.

```python
writer = SummaryWriter("runs/exp1", system_metrics_interval=3600)  # every hour (default)
```

Collected metrics: `system/cpu_percent`, `system/mem_used_gb`, `system/disk_free_gb`, `gpu/utilization`, `gpu/memory_used_gb`, `gpu/temperature`, and automatic alerts when resources are critically low.

## MCP server (AI agent integration)

When installed with `vibetrack[all]` on Python 3.10+, the web dashboard includes an MCP (Model Context Protocol) server at `/vibetrack_mcp`, enabling AI agents like Claude to query your experiment data directly.

**Available MCP tools:** `list_experiments`, `get_experiment_tags`, `get_scalars`, `get_texts`, `get_images`, `get_audio`, `get_hparams`, `get_histograms`, `summary`, `compare_hparams_tool`

**MCP resources:** `vibetrack://experiments`, `vibetrack://experiments/{name}`, `vibetrack://experiments/{name}/scalars/{tag}`, etc.

Standalone MCP server:

```bash
pip install vibetrack[all]
vibetrack --viewer mcp --project-folder my_project/
```



## CLI

```bash
vibetrack                           # default  
vibetrack [PROJECT_FOLDER]          # Launch dashboard (web + ingest; MCP with vibetrack[all] on Python 3.10+)
vibetrack --port 8080               # Custom port
vibetrack --host 127.0.0.1          # Bind to localhost only (by default it is open on LAN IP)
vibetrack --token SECRET            # Protect ingest endpoints
vibetrack --listen 0.0.0.0:9009     # Standalone  server on separate port
vibetrack migrate PROJECT_FOLDER    # Merge legacy per-run DBs into project DB
```

## Configuration

Settings are stored in `~/.vibetrack/config.json` (global) or per-project via the API:

```json
{
  "smoothing": "ema",
  "smooth_weight": 0.6,
  "system_metrics_interval": 3600,
  "web": {
    "theme": "light",
    "auto_refresh": 5,
    "image_play_fps": 2,
    "original_values_opacity": 0.17
  }
}
```

## License

Apache 2.0
