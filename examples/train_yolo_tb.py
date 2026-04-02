#!/usr/bin/env python3
"""Minimal real YOLO training with TensorBoard logging.

Same training as train_yolo.py but logs to TensorBoard instead of vibetrack.
Useful for side-by-side comparison of the two dashboards.

Install:  pip install ultralytics torch tensorboard
Run:      python examples/train_yolo_tb.py
View:     tensorboard --logdir runs/yolo_tb
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────────
SEED = 42
EPOCHS = 25
MODEL = "yolov8n.pt"
DATA = "coco8.yaml"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

writer = SummaryWriter("runs/yolo_tb")

# ── Callbacks ────────────────────────────────────────────────────

def on_fit_epoch_end(trainer):
    """Log scalars at the end of each train+val epoch."""
    epoch = trainer.epoch

    # Losses (box, cls, dfl)
    if trainer.label_loss_items is not None:
        for key, val in trainer.label_loss_items(trainer.tloss).items():
            writer.add_scalar(f"train/{key}", val, epoch)

    # Learning rate
    if hasattr(trainer, "lr"):
        for key, val in trainer.lr.items():
            writer.add_scalar(f"lr/{key}", val, epoch)

    # Val metrics (populated after validation runs)
    for key, val in trainer.metrics.items():
        writer.add_scalar(f"val/{key}", val, epoch)


def on_train_end(trainer):
    """Log prediction images after training finishes."""
    # Find val images from the downloaded dataset
    datasets_dir = Path.home() / "datasets" / "coco8" / "images" / "val"
    if not datasets_dir.exists():
        from ultralytics.utils import DATASETS_DIR
        datasets_dir = Path(DATASETS_DIR) / "coco8" / "images" / "val"

    if not datasets_dir.exists():
        return

    img_files = sorted(
        f for f in datasets_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:8]

    # Load best weights through the YOLO API (trainer.model is raw nn.Module)
    best = Path(trainer.best)
    best_model = YOLO(str(best))
    results = best_model.predict(
        source=[str(f) for f in img_files],
        save=True,
        conf=0.25,
        verbose=False,
    )

    if results and hasattr(results[0], "save_dir"):
        save_dir = str(results[0].save_dir)
        for i, fname in enumerate(sorted(os.listdir(save_dir))):
            fpath = os.path.join(save_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                from PIL import Image

                img = Image.open(fpath)
                img_array = np.array(img)
                # TensorBoard expects (C, H, W) for add_image
                writer.add_image(
                    f"predictions/{fname}",
                    img_array,
                    global_step=0,
                    dataformats="HWC",
                )


# ── Train ────────────────────────────────────────────────────────
model = YOLO(MODEL)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
model.add_callback("on_train_end", on_train_end)

model.train(data=DATA, epochs=EPOCHS, imgsz=640, batch=16, seed=SEED, verbose=False)

writer.close()

print("\nDone! View results:")
print("  tensorboard --logdir runs/yolo_tb")
