"""SummaryWriter — drop-in replacement for tensorboard.SummaryWriter.

Also supports W&B-style log() calls via the module-level API.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .db import Database, central_db_path
from .defaults import SYSTEM_METRICS_INTERVAL

_DEFAULT_LOGDIR = "runs"


def _detect_rank() -> int:
    """Return the process rank from torchrun env vars, defaulting to 0."""
    rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank_str is not None:
        try:
            return int(rank_str)
        except ValueError:
            pass
    return 0


def _make_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SummaryWriter:
    """TensorBoard-compatible experiment writer backed by SQLite.

    Usage::

        from vibetrack import SummaryWriter
        writer = SummaryWriter("runs/exp1")
        writer.add_scalar("loss", 0.5, 0)
        writer.add_scalar("loss", 0.3, 1)
        writer.close()

    By default all experiments are stored in a single system-wide database
    at ``~/.vibetrack/vibetrack.db``.  Pass ``project_folder="."`` to use a
    local ``vibetrack.db`` in the given directory instead.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 1000,
        flush_secs: int = 120,
        filename_suffix: str = "",
        # W&B-style extras
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        precache_secs: float = 0,
        system_metrics_interval: float = SYSTEM_METRICS_INTERVAL,
        project_folder: Optional[str] = None,
        rank: Optional[Union[int, str]] = None,
    ) -> None:
        # Resolve log_dir — mimic TensorBoard's default behaviour
        if log_dir is None:
            base = project or _DEFAULT_LOGDIR
            run = name or _make_run_name()
            if comment:
                run = f"{run}_{comment}"
            log_dir = os.path.join(base, run)

        self.log_dir = str(Path(log_dir).resolve())
        # Infer project from parent dir name, run name from leaf dir name
        self._project = project or Path(self.log_dir).parent.name
        self._run_name = name or Path(self.log_dir).name
        self._config = config
        self._closed = False
        self._warned_errors: set[str] = set()
        self._db: Optional[Database] = None
        self._exp_id = -1
        self.project_folder = project_folder
        self._max_queue = max_queue
        self._buffer_lock = threading.Lock()
        self._scalar_buffer: List[Tuple[int, str, int, float, float]] = []
        self._step_counters: Dict[str, int] = {}
        self._sysmetrics: Any = None

        # Rank gating — only rank 0 logs, others become no-op
        if rank is None:
            self._rank = _detect_rank()
        elif rank == "all":
            self._rank = 0
        else:
            self._rank = int(rank)

        if self._rank != 0:
            self._enabled = False
            return

        self._enabled = True
        self._exp_id = 0

        try:
            # DB location: project_folder overrides, else central DB
            if project_folder is not None:
                pf = Path(project_folder).resolve()
                pf.mkdir(parents=True, exist_ok=True)
                db_path = pf / "vibetrack.db"
            else:
                db_path = central_db_path()
            # Ensure media dir exists
            media_dir = Path(self.log_dir) / "media"
            if precache_secs <= 0:
                media_dir.parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(db_path, precache_secs=precache_secs)

            # Create (or reuse) experiment row
            existing = self._db.get_experiment_by_name(
                self._run_name, project=self._project,
            )
            if existing is not None:
                self._exp_id = existing["id"]
            else:
                self._exp_id = self._db.create_experiment(
                    self._run_name, config=config,
                    project=self._project, log_dir=self.log_dir,
                )

            # Register remap callback for when precache materializes
            if precache_secs > 0:
                self._db.register_remap_callback(self._on_precache_flush)
        except Exception as exc:
            self._enabled = False
            try:
                if self._sysmetrics is not None:
                    self._sysmetrics.stop()
            except Exception:
                pass
            try:
                if self._db is not None:
                    self._db.close()
            except Exception:
                pass
            self._db = None
            self._handle_runtime_error("initialize writer", exc)
            return

        if config and self._db is not None:
            try:
                self._db.add_hparams(self._exp_id, config)
            except Exception as exc:
                self._handle_runtime_error("store initial hparams", exc)

        if system_metrics_interval > 0:
            try:
                from .sysmetrics import SystemMetricsCollector

                self._sysmetrics = SystemMetricsCollector(
                    writer=self,
                    interval=system_metrics_interval,
                    disk_path=self.log_dir,
                )
                self._sysmetrics.start()
            except Exception as exc:
                self._sysmetrics = None
                self._handle_runtime_error("start system metrics", exc)

    def _handle_runtime_error(self, action: str, exc: Exception) -> None:
        msg = f"{action}:{type(exc).__name__}:{exc}"
        if msg in self._warned_errors:
            return
        self._warned_errors.add(msg)
        print(
            f"vibetrack warning: failed to {action}: {exc}",
            file=sys.stderr,
        )

    def _best_effort(self, action: str, fn: Any, default: Any = None) -> Any:
        if not self._enabled or self._closed:
            return default
        try:
            return fn()
        except Exception as exc:
            self._handle_runtime_error(action, exc)
            return default

    # ── TensorBoard-compatible API ──────────────────────────────

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        new_style: bool = False,
    ) -> None:
        def _op() -> None:
            val = float(scalar_value)
            if not math.isfinite(val):
                return  # skip NaN / Inf
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            with self._buffer_lock:
                self._scalar_buffer.append((self._exp_id, tag, step, val, wt))
                if len(self._scalar_buffer) >= self._max_queue:
                    self._flush_locked()

        self._best_effort("add scalar", _op)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        def _op() -> None:
            for sub_tag, value in tag_scalar_dict.items():
                self.add_scalar(f"{main_tag}/{sub_tag}", value, global_step, walltime)

        self._best_effort("add scalars", _op)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            self._db.add_text(self._exp_id, tag, text_string, step, walltime)

        self._best_effort("add text", _op)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW",
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            from .media import save_image

            rel_path = save_image(img_tensor, self.log_dir, tag, step)
            self._db.add_image(self._exp_id, tag, rel_path, step, walltime)

        self._best_effort("add image", _op)

    def add_audio(
        self,
        tag: str,
        snd_tensor: Any,
        global_step: Optional[int] = None,
        sample_rate: int = 44100,
        walltime: Optional[float] = None,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            from .media import save_audio

            rel_path = save_audio(snd_tensor, self.log_dir, tag, step, sample_rate)
            self._db.add_audio(self._exp_id, tag, rel_path, step, sample_rate, walltime)

        self._best_effort("add audio", _op)

    def add_video(
        self,
        tag: str,
        vid_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        fps: int = 4,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            from .media import save_video

            rel_path = save_video(vid_tensor, self.log_dir, tag, step)
            self._db.add_video(self._exp_id, tag, rel_path, step, walltime)

        self._best_effort("add video", _op)

    def add_artifact(
        self,
        tag: str,
        file_path: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            from .media import save_artifact

            rel_path, auto_meta = save_artifact(file_path, self.log_dir, tag, step)
            if metadata:
                auto_meta.update(metadata)
            self._db.add_artifact(
                self._exp_id, tag, rel_path, json.dumps(auto_meta), step, walltime
            )

        self._best_effort("add artifact", _op)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: str = "tensorflow",
        walltime: Optional[float] = None,
        max_bins: Optional[int] = None,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            step = self._resolve_step(tag, global_step)
            # Simple histogram: convert values to list, compute bins
            try:
                vals = list(values)
            except TypeError:
                vals = [float(values)]

            n_bins = max_bins or 30
            if len(vals) == 0:
                return
            lo, hi = min(vals), max(vals)
            if lo == hi:
                bin_edges = [lo - 0.5, lo + 0.5]
                counts = [float(len(vals))]
            else:
                width = (hi - lo) / n_bins
                bin_edges = [lo + i * width for i in range(n_bins + 1)]
                counts = [0.0] * n_bins
                for v in vals:
                    idx = min(int((v - lo) / width), n_bins - 1)
                    counts[idx] += 1.0

            self._db.add_histogram(self._exp_id, tag, bin_edges, counts, step, walltime)

        self._best_effort("add histogram", _op)

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Optional[Dict[str, Any]] = None,
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
    ) -> None:
        def _op() -> None:
            if self._db is None:
                return
            self._db.add_hparams(self._exp_id, hparam_dict)
            if metric_dict:
                for tag, val in metric_dict.items():
                    self.add_scalar(tag, val)

        self._best_effort("add hparams", _op)

    # ── W&B-style log() ─────────────────────────────────────────

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """W&B-compatible log call: ``writer.log({"loss": 0.5, "acc": 0.9})``."""
        def _op() -> None:
            from .types import Image, Audio, Video, Artifact

            for tag, value in data.items():
                if isinstance(value, (int, float)):
                    self.add_scalar(tag, value, global_step=step)
                elif isinstance(value, str):
                    self.add_text(tag, value, global_step=step)
                elif isinstance(value, Image):
                    self.add_image(tag, value.data, global_step=step)
                elif isinstance(value, Audio):
                    self.add_audio(
                        tag, value.data, global_step=step,
                        sample_rate=value.sample_rate,
                    )
                elif isinstance(value, Video):
                    self.add_video(tag, value.data, global_step=step)
                elif isinstance(value, Artifact):
                    self.add_artifact(
                        tag, value.path, global_step=step,
                        metadata=value.metadata,
                    )
            if commit:
                with self._buffer_lock:
                    self._flush_locked()

        self._best_effort("log data", _op)

    # ── Lifecycle ───────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Flush buffer. Caller must hold _buffer_lock."""
        if self._db is not None and self._scalar_buffer:
            self._db.add_scalars_bulk(self._scalar_buffer)
            self._scalar_buffer.clear()

    def flush(self) -> None:
        """Compatibility no-op for TensorBoard-style APIs."""
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enabled = False
        try:
            if self._sysmetrics is not None:
                self._sysmetrics.stop()
                self._sysmetrics = None
        except Exception as exc:
            self._handle_runtime_error("stop system metrics", exc)
        try:
            with self._buffer_lock:
                self._flush_locked()
        except Exception as exc:
            self._handle_runtime_error("flush during close", exc)
        try:
            if self._db is not None:
                self._db.close()
                self._db = None
        except Exception as exc:
            self._handle_runtime_error("close database", exc)

    def __enter__(self) -> "SummaryWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── Properties ──────────────────────────────────────────────

    @property
    def experiment_id(self) -> int:
        return self._exp_id

    @property
    def run_name(self) -> str:
        return self._run_name

    # ── Internal ────────────────────────────────────────────────

    def _on_precache_flush(self, id_remap: Dict[int, int]) -> None:
        """Called by Database after precache materializes to remap IDs."""
        old_id = self._exp_id
        if old_id in id_remap:
            new_id = id_remap[old_id]
            self._exp_id = new_id
            # Remap any buffered scalars that haven't been flushed yet
            with self._buffer_lock:
                self._scalar_buffer = [
                    (new_id if r[0] == old_id else r[0], r[1], r[2], r[3], r[4])
                    for r in self._scalar_buffer
                ]

    def _resolve_step(self, tag: str, global_step: Optional[int]) -> int:
        with self._buffer_lock:
            if global_step is not None:
                self._step_counters[tag] = global_step
                return global_step
            step = self._step_counters.get(tag, -1) + 1
            self._step_counters[tag] = step
            return step
