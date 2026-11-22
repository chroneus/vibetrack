# vibetrack API

## SummaryWriter

```python
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/exp1", name="experiment_name", project_folder="project/")

# Scalars
writer.add_scalar("loss", 0.5, step=0)
writer.add_scalars("metrics", {"train_loss": 0.5, "val_loss": 0.6}, step=0)

# Images - accepts file paths, numpy arrays, or PIL Images
writer.add_image("samples", "path/to/image.png", step=0)

# Audio - accepts file paths or numpy waveforms
writer.add_audio("speech", waveform_array, step=0, sample_rate=16000)

# Video
writer.add_video("rollout", "path/to/video.mp4", step=0)

# Artifacts - any file with optional metadata
writer.add_artifact("checkpoint", "model.pt", step=0, metadata={"val_acc": 0.95})

# Text
writer.add_text("notes", "Training started with lr=0.01", step=0)

# Histograms
writer.add_histogram("weights", weight_tensor, step=0)

# Hyperparameters
writer.add_hparams({"lr": 0.01, "batch_size": 32}, {"best_acc": 0.95})

writer.close()
```

## Module-level logging API

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

## Run status, provenance and the end-of-run report

Every run records its lifecycle in the database:

| status | meaning |
| --- | --- |
| `running` | the writer is open; a heartbeat is written while it logs |
| `finished` | closed normally |
| `failed` | closed while an exception propagated, or the script died with an uncaught exception; the error and traceback tail are stored |
| `interrupted` | `KeyboardInterrupt` |
| `crashed` | still `running` but its process is gone (killed, OOM); detected when read |

The command line (credential-looking arguments are masked), cwd, host, pid,
Python version, and git commit / branch / dirty state are stored as well.
Set `VIBETRACK_GIT=0` to skip git. Close runs in `finally:` or use the writer
as a context manager so failures are recorded:

```python
with SummaryWriter("runs/exp1", project="cv", config=cfg) as writer:
    train(writer)            # an exception here marks the run `failed`
```

Declare the metric you optimize, so rankings don't have to guess it from the
tag name:

```python
writer.define_metric("val/acc", goal="max", primary=True)
vibetrack.define_metric("eval/return", goal="max", primary=True)  # module API
```

On close, a compact report is printed to stderr: status, config, headline
metrics, media paths, rank among the project's runs, the dashboard link and the
next commands. Disable it with `report=False` or `VIBETRACK_REPORT=0`.

A run that fails, is interrupted or logs nothing within its first 60 seconds
(a bug at start-up, a wrong argument, Ctrl-C) is not kept: on close its rows,
media files and any synced copy are deleted, and a one-line notice replaces
the report. Runs that log data and finish, and resumed runs, are always kept. Change the limit with
`discard_failed_secs=` or `VIBETRACK_DISCARD_FAILED_SECS`; `0` keeps every
run.

## Syncing to S3-compatible storage

```python
writer = SummaryWriter("runs/exp1", sync="s3://ml-runs/team?endpoint=https://minio.lab:9000")
vibetrack.init(project="cv", name="r1", sync="file:///mnt/shared/runs")  # same for the module API
```

`sync=None` (the default) follows `VIBETRACK_SYNC`, and `sync=""` turns it off.
New rows are uploaded every `VIBETRACK_SYNC_INTERVAL` seconds (default 60),
and a final snapshot on close, failure or SIGTERM. Read synced runs with
`RunReader("s3://…")` or `vibetrack s3://…`. See [STORAGE.md](STORAGE.md).

## Reading runs programmatically

`vibetrack.insights` returns the same JSON-ready payloads as the CLI:

```python
from vibetrack import RunReader, insights

with RunReader("my_project") as reader:
    exps = reader.experiments(project=None)
    rows = insights.list_runs(exps)               # status + headline metrics
    ov = insights.run_overview(exps[0])           # one run in detail
    cmp = insights.compare_runs(exps, metric="val/acc", goal="max")
    print(insights.format_compare(cmp))           # Markdown
```

## Video from frames

`add_video` / `Video(...)` accept a file path, bytes, a `T,H,W,C` / `T,C,H,W` /
`N,T,C,H,W` array or tensor, or a list of `H,W,C` frames (e.g. from
`env.render()`):

```python
vibetrack.log({"eval/rollout": vibetrack.Video(frames, fps=30)}, step=step)
```

Frames are encoded to browser-playable H.264 MP4 with ffmpeg: the system
binary, or the one from `pip install imageio-ffmpeg`. If neither is available,
moviepy is used. Without any encoder, the frames are saved as an animated GIF.
