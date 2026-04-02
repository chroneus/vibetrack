#!/usr/bin/env python3
"""Compare multiple training runs side by side.

Creates 3 runs with different learning rates, then shows how to
compare them using vibetrack's comparison utilities.

Run:  python examples/compare_runs.py
View: vibetrack --project-folder runs/lr_search
"""

import math
import random

from vibetrack import SummaryWriter, compare_scalars, summary_table

CONFIGS = [
    {"name": "lr_0.1", "lr": 0.1},
    {"name": "lr_0.01", "lr": 0.01},
    {"name": "lr_0.001", "lr": 0.001},
]

STEPS = 200


def simulate_training(lr: float, steps: int):
    """Generate synthetic loss/acc curves for a given learning rate."""
    random.seed(hash(str(lr)))
    results = []
    for step in range(steps):
        # Higher LR → faster initial drop, but noisier + higher floor
        decay = math.exp(-step * lr * 0.5)
        floor = 0.05 + lr * 0.3  # higher lr = higher floor (overshoot)
        noise = random.gauss(0, lr * 0.1)
        loss = decay * 2.0 + floor + noise
        acc = 1 - loss * 0.4 + random.gauss(0, 0.01)
        results.append((loss, max(0, min(1, acc))))
    return results


def main():
    print("Running 3 experiments with different learning rates...\n")
    project_folder = "runs/lr_search"

    for cfg in CONFIGS:
        log_dir = f"{project_folder}/{cfg['name']}"
        with SummaryWriter(log_dir, config=cfg, project_folder=project_folder) as writer:
            data = simulate_training(cfg["lr"], STEPS)
            for step, (loss, acc) in enumerate(data):
                writer.add_scalar("loss", loss, step)
                writer.add_scalar("acc", acc, step)
            writer.add_hparams(cfg, {"loss": data[-1][0], "acc": data[-1][1]})

        print(f"  {cfg['name']:>10s}: final loss={data[-1][0]:.4f}, acc={data[-1][1]:.4f}")

    print()

    # ── Programmatic comparison ──────────────────────────────────

    from vibetrack.reader import RunReader

    reader = RunReader(project_folder)
    experiments = reader.experiments()

    print("Experiments found:", [e.name for e in experiments])
    print()

    # Summary table
    table = summary_table(experiments, tags=["loss", "acc"])
    print("Summary table:")
    for row in table:
        print(f"  {row}")
    print()

    # Compare specific tag
    comparison = compare_scalars(experiments, "loss")
    print("Loss series:")
    for entry in comparison:
        values = entry["values"]
        print(
            f"  {entry['name']}: min={min(values):.4f} "
            f"max={max(values):.4f} last={values[-1]:.4f}"
        )

    reader.close()

    print()
    print("View results:")
    print(f"  vibetrack --project-folder {project_folder}")
    print(f"  vibetrack --project-folder {project_folder} --viewer=console")


if __name__ == "__main__":
    main()
