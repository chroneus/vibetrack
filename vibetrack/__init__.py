"""vibetrack — lightweight experiment tracking.

Example usage:
    # TensorBoard style
    from vibetrack import SummaryWriter
    writer = SummaryWriter("my_project/run_1")
    writer.add_scalar("loss", 0.5, step)

    # Module-level API
    import vibetrack
    vibetrack.init(project="my_project", name="run_1", config={"lr": 0.01})
    vibetrack.log({"loss": 0.5, "acc": 0.9})
    vibetrack.finish()

"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .writer import SummaryWriter
from .reader import ExperimentReader, RunReader
from .smoother import smooth, ema, moving_average, gaussian
from .compare import compare_scalars, compare_hparams, summary_table
from .types import Image, Audio, Video, Artifact
from .default_config import SYSTEM_METRICS_INTERVAL


def _read_version_from_pyproject() -> Optional[str]:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None

    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    return None


__version__ = _read_version_from_pyproject()
if __version__ is None:
    try:
        __version__ = version("vibetrack")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    # Core
    "SummaryWriter",
    "ExperimentReader",
    "RunReader",
    # Module-level logging API
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

# ── Module-level logging API ────────────────────────────────────

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
    to: Optional[Union[str, list, tuple]] = None,
    **kwargs: Any,
) -> SummaryWriter:
    """Initialize a new run.

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
        log_dir=log_dir,
        project=project,
        name=name,
        config=config,
        project_folder=project_folder,
        precache_secs=precache_secs,
        system_metrics_interval=system_metrics_interval,
        rank=rank,
        **kwargs,
    )
    _step = 0
    if config:
        _mod.config = dict(config)
    if to is not None:
        names = [to] if isinstance(to, str) else list(to)
        for entry in names:
            if isinstance(entry, str):
                _active_writer.to(entry)
            elif isinstance(entry, dict):
                _active_writer.to(**entry)
            else:
                _warn(f"ignoring unknown to= entry: {entry!r}")
    return _active_writer


def log(data: Dict[str, Any], step: Optional[int] = None, **kwargs: Any) -> None:
    """Log metrics for the current step.

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
    """Flush and close the active writer."""
    global _active_writer
    if _active_writer is not None:
        try:
            _active_writer.close()
        except Exception as exc:
            _warn(f"failed to close writer during finish: {exc}")
        _active_writer = None
