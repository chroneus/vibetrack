"""vibetrack — lightweight experiment tracking.

Drop-in compatible with TensorBoard and W&B APIs::

    # TensorBoard style
    from vibetrack import SummaryWriter
    writer = SummaryWriter("runs/exp1")
    writer.add_scalar("loss", 0.5, step)

    # W&B style
    import vibetrack
    vibetrack.init(project="my_project", name="run_1", config={"lr": 0.01})
    vibetrack.log({"loss": 0.5, "acc": 0.9})
    vibetrack.finish()

    # Also works as a namespace alias
    import vibetrack as tb
    writer = tb.SummaryWriter("runs/exp1")
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Union

from .writer import SummaryWriter
from .reader import ExperimentReader, RunReader
from .smoother import smooth, ema, moving_average, gaussian
from .compare import compare_scalars, compare_hparams, summary_table
from .types import Image, Audio, Video, Artifact
from .defaults import SYSTEM_METRICS_INTERVAL

__version__ = "0.1a0"
__all__ = [
    # Core
    "SummaryWriter",
    "ExperimentReader",
    "RunReader",
    # W&B-style module API
    "init",
    "log",
    "finish",
    "config",
    # Smoothing
    "smooth",
    "ema",
    "moving_average",
    "gaussian",
    # Compare
    "compare_scalars",
    "compare_hparams",
    "summary_table",
    # Media types
    "Image",
    "Audio",
    "Video",
    "Artifact",
]

# ── W&B-style module-level API ──────────────────────────────────

_active_writer: Optional[SummaryWriter] = None
_step: int = 0

config: Dict[str, Any] = {}


def _warn(msg: str) -> None:
    print(f"vibetrack warning: {msg}", file=sys.stderr)


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
    project_folder: Optional[str] = None,
    precache_secs: float = 0,
    system_metrics_interval: float = SYSTEM_METRICS_INTERVAL,
    rank: Optional[Union[int, str]] = None,
    **kwargs: Any,
) -> SummaryWriter:
    """Initialize a new run (W&B-style).

    Only rank 0 logs by default.  Other ranks get a no-op writer.
    Set ``rank="all"`` to force every rank to log.

    ::

        import vibetrack
        vibetrack.init(project="cifar10", name="resnet18", config={"lr": 1e-3})
        vibetrack.init(..., system_metrics_interval=10)  # collect OS/GPU stats
    """
    global _active_writer, _step
    import vibetrack as _mod

    if _active_writer is not None:
        try:
            _active_writer.close()
        except Exception as exc:
            _warn(f"failed to close active writer during init: {exc}")

    _active_writer = SummaryWriter(
        log_dir=log_dir, project=project, name=name, config=config,
        project_folder=project_folder,
        precache_secs=precache_secs,
        system_metrics_interval=system_metrics_interval,
        rank=rank, **kwargs
    )
    _step = 0
    if config:
        _mod.config = dict(config)
    return _active_writer


def log(data: Dict[str, Any], step: Optional[int] = None, **kwargs: Any) -> None:
    """Log metrics for the current step (W&B-style).

    ::

        vibetrack.log({"loss": 0.5, "acc": 0.9})
    """
    global _step
    if _active_writer is None:
        _warn("log() called before init(); dropping data")
        return
    if step is not None:
        _step = step
    try:
        _active_writer.log(data, step=_step, **kwargs)
    except Exception as exc:
        _warn(f"failed to log data: {exc}")
    _step += 1


def finish() -> None:
    """Flush and close the active writer (W&B-style)."""
    global _active_writer
    if _active_writer is not None:
        try:
            _active_writer.close()
        except Exception as exc:
            _warn(f"failed to close writer during finish: {exc}")
        _active_writer = None
