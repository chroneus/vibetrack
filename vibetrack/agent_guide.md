# vibetrack: guide for AI coding agents

Use this when the user asks you to track, log, monitor or compare experiments,
or says "use vibetrack". The goal: every run records **all parameters**,
**namespaced metrics** and **qualitative results (videos/images/samples)**, so
that you can compare runs from the command line and the user can see results
in the dashboard without asking you.

## 1. Install

```bash
pip install vibetrack            # or: uv add vibetrack / poetry add vibetrack
vibetrack --help                 # sanity check
```

Logging videos from frame arrays needs an ffmpeg binary (`ffmpeg` on PATH or
`pip install imageio-ffmpeg`); without one vibetrack falls back to an animated
GIF. MCP tools need `pip install "vibetrack[all]"` (Python 3.10+).

## 2. Instrument the code (do every item)

1. **Start one run per training process** with `vibetrack.init(...)`:
   - `project=`: a stable name for the task (same for all runs you will compare).
   - `name=`: short and descriptive, encode what differs, e.g. `ppo-lr3e-4-s1`.
     Re-running the same name later *resumes* it; overlapping steps create `name (2)`.
   - `config=`: **every** hyperparameter and setting: lr, batch size, model
     size/arch, optimizer, schedule, dataset/env, seed, number of steps,
     augmentation flags. Use `vars(args)` (argparse), `dataclasses.asdict(cfg)`,
     or `OmegaConf.to_container(cfg, resolve=True)` (Hydra). Nested dicts are fine.
   - vibetrack records the command line, git commit + dirty state, host, pid,
     start/end time and the exit status (finished / failed with traceback /
     interrupted / crashed) automatically.
2. **Declare the metric you optimize**: `vibetrack.define_metric("val/acc", goal="max", primary=True)`.
   Rankings otherwise guess from the name (loss/error -> min, acc/return/reward -> max).
3. **Log scalars with namespaced tags on one global step**:
   `train/loss`, `train/lr`, `val/loss`, `val/acc`, `eval/return`, `test/acc`.
   Pass `step=global_step` so all metrics share an x-axis. Values may be Python
   numbers, numpy scalars or 0-d tensors.
4. **Log qualitative results at every eval and at the end.** This is what the
   user looks at first in the dashboard's Runs tab:
   - RL / control / robotics / simulation: a rollout video,
     `vibetrack.Video(frames, fps=30)` where `frames` is a list or array of
     `H x W x 3` uint8 frames (`env = gym.make(..., render_mode="rgb_array")`, `env.render()`).
   - Vision: `vibetrack.Image(array_or_path)` for predictions, samples or
     reconstructions of a **fixed** eval batch (comparable across steps and runs).
   - Generative / LLM: text samples as strings (`{"samples/text": "..."}`),
     audio as `vibetrack.Audio(wave, sample_rate=16000)`.
   - Plots: `run.add_figure("eval/confusion", fig, step)` (matplotlib).
   - Keep media small: short clips, downscale to <= 512 px, log every few evals.
5. **Log final metrics** (e.g. `test/acc`) at the last step.
6. Optional: checkpoints as `vibetrack.Artifact("ckpt.pt")`.
7. **Always finish the run** in `finally:` (or use the writer as a context
   manager), so a crash is recorded as `failed` with its traceback:
   `try: train() finally: vibetrack.finish()`.
8. Don't: start a new run per epoch, log from DataLoader workers (only rank 0
   logs by default anyway), or put secrets in `config`.

### Template: training loop

```python
import vibetrack

def main(cfg):
    run = vibetrack.init(project="cifar10", name=f"resnet18-lr{cfg['lr']}-s{cfg['seed']}", config=cfg)
    vibetrack.define_metric("val/acc", goal="max", primary=True)
    try:
        step = 0
        for epoch in range(cfg["epochs"]):
            for batch in train_loader:
                loss = train_step(batch)
                step += 1
                if step % 50 == 0:
                    vibetrack.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]}, step=step)
            val_loss, val_acc, preview = evaluate()          # preview: HxWx3 uint8 grid
            vibetrack.log({"val/loss": val_loss, "val/acc": val_acc,
                           "val/predictions": vibetrack.Image(preview)}, step=step)
        vibetrack.log({"test/acc": test()}, step=step)
    finally:
        vibetrack.finish()   # prints the end-of-run report
```

### Template: RL rollout video

```python
import gymnasium as gym, vibetrack

def record_rollout(policy, step):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    frames, total, done = [], 0.0, False
    while not done:
        frames.append(env.render())
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        done = terminated or truncated
    env.close()
    vibetrack.log({"eval/return": total, "eval/rollout": vibetrack.Video(frames, fps=30)}, step=step)
```

### TensorBoard-style code

Replace `from torch.utils.tensorboard import SummaryWriter` with
`from vibetrack import SummaryWriter`. Then pass the config:
`SummaryWriter("runs/exp1", project="myproj", config=cfg)`. Call `writer.close()`
in `finally:`. All `add_*` calls keep working. Use `writer.define_metric(...)`
the same way as above.

## 3. Run it

Run the script as usual. When the run closes, vibetrack prints a short report to
**stderr**: status, config, headline metrics (last and best @step), media
paths, rank among runs of the same project, the dashboard link and the next
commands. Read it; it is meant for you. For long runs, start them in the
background and check progress with `vibetrack ls`. A run whose process died
shows as `crashed`.

## 4. Inspect and compare (compact Markdown; add `--json` for structured data)

```bash
vibetrack ls [-p PROJECT]                     # runs: status, duration, step, headline metrics, media
vibetrack show [RUN] [-p PROJECT]             # one run: config, metric table + trends, media files, errors
vibetrack compare [RUN|GLOB ...] [-p PROJECT] [--metric val/acc --goal max]
```

- Runs logged with `project_folder=DIR` live in `DIR/vibetrack.db`: add
  `--project-folder DIR` to every command. Otherwise data is in the central
  `~/.vibetrack/vibetrack.db`.
- `compare` shows:
  - the ranking by the primary metric's **best** value, with the % gap to the winner;
  - params that **differ** (only these are columns) and params that are the same everywhere;
  - a per-metric last/best table;
  - latest media files;
  - git commit per run.
- "fell back N% of range": the last value is far from the best one (overfitting,
  divergence). "Param effects" are heuristics on few runs; say so when you cite them.
- Media paths are absolute: you can open an image to look at it yourself.

## 5. Show the user the results

After a run or a sweep finishes:

```bash
vibetrack open RUN [RUN ...] [-p PROJECT]     # starts the dashboard in the background if needed
```

This opens the **Runs** tab: one card per run with status, key metrics, the
params that differ and the latest video/image playing. Then reply to the user with:
- what ran: runs and the params that changed;
- outcome per run: status, primary metric best/last, rank vs. previous best;
- notable issues: failures with their error, divergence, overfitting;
- the dashboard link, and what you suggest trying next.

Don't paste raw logs.

## 6. MCP (optional)

With `vibetrack[all]`, `vibetrack` (the dashboard) also serves MCP at
`http://127.0.0.1:6116/vibetrack_mcp`. Its tools include `list_runs`,
`run_overview`, `compare_runs` (same data as the CLI) and lower-level
`get_scalars`, `analyze_scalar` and `compare_image_lpips`.

## Reference

- Disable the end-of-run report: `VIBETRACK_REPORT=0` or `init(..., report=False)`.
- Skip git metadata: `VIBETRACK_GIT=0`.
- Media files: `<log_dir>/media/<tag>/<step>.<ext>`; `log_dir` defaults to `./<project>/<name>`.
- Full API: https://github.com/chroneus/vibetrack/blob/main/docs/API.md
