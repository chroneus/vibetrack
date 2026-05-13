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
