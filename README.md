# vibetrack

Lightweight experiment tracking — drop-in compatible with TensorBoard and W&B APIs.

No cloud. No account. No signup. Just a single SQLite file.

## Why vibetrack?

| | TensorBoard | W&B | vibetrack |
|---|---|---|---|
| Storage | tfevents files | Cloud (paid tiers) | Single SQLite DB |
| Account required | No | Yes | No |
| AI agent integration (MCP) | No | No | Built-in |
| Distributed torchrun support | Manual | Manual | Automatic (rank-0 only) |
| Single-port server (web + API + ingest) | No | N/A | Yes |
| Viewer backends | Web only | Web only | Web, Gradio, Telegram, Console, MCP |
| System metrics | Via plugin | Built-in | Built-in (CPU, GPU, memory, disk) |
| Remote HTTP logging | No | Yes (cloud) | Yes (self-hosted, token auth) |
| Zero mandatory dependencies | No | No | Yes |

## Install

```bash
pip install vibetrack          # default
pip install vibetrack[all]     # all optional backends
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

### Launch the dashboard

```bash
vibetrack 
# -> Web UI + MCP server + ingest endpoint on http://0.0.0.0:6006
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

### Smoothing

```python
from vibetrack import smooth, ema, moving_average, gaussian

smoothed = ema(values, weight=0.6)           # Exponential moving average
smoothed = moving_average(values, window=10)  # Uniform window
smoothed = gaussian(values, sigma=2.0)        # Gaussian kernel
smoothed = smooth(values, method="ema", weight=0.6)  # Unified dispatcher
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

The web dashboard includes an MCP (Model Context Protocol) server at `/vibetrack_mcp`, enabling AI agents like Claude to query your experiment data directly.

**Available MCP tools:** `list_experiments`, `get_experiment_tags`, `get_scalars`, `get_texts`, `get_images`, `get_audio`, `get_hparams`, `get_histograms`, `summary`, `compare_hparams_tool`

**MCP resources:** `vibetrack://experiments`, `vibetrack://experiments/{name}`, `vibetrack://experiments/{name}/scalars/{tag}`, etc.

Standalone MCP server:

```bash
vibetrack --viewer mcp --project-folder my_project/
```



## CLI

```bash
vibetrack                           # default  
vibetrack [PROJECT_FOLDER]          # Launch dashboard (web + MCP + ingest)
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
    "image_play_fps": 2
  }
}
```

## License

Apache 2.0
