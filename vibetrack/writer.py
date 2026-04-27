"""SummaryWriter — drop-in replacement for tensorboard.SummaryWriter.

Also supports W&B-style log() calls via the module-level API.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .db import Database, central_db_path
from .default_config import SYSTEM_METRICS_INTERVAL
from .viewers.event import (
    EventHandle,
    LogEvent,
    MultiEventHandle,
    NullEventHandle,
    null_handle,
)

_DEFAULT_LOGDIR = "runs"


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])?\s*$", re.IGNORECASE)
_DURATION_FACTORS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _parse_every(
    every: Optional[Union[int, float, str]],
) -> Tuple[str, Optional[float]]:
    """Translate an ``every=`` argument to ``(mode, value)``.

    - ``None`` → ("event", None)                 — dispatch each event immediately
    - ``int``  → ("steps", N)                    — flush after every N events
    - ``"15m"`` / ``"1h"`` / ``"5s"`` → ("time", seconds)
    """
    if every is None:
        return ("event", None)
    if isinstance(every, bool):
        raise TypeError("every= must be int, float, or str (got bool)")
    if isinstance(every, int):
        if every <= 0:
            raise ValueError(f"every= must be positive (got {every})")
        return ("steps", float(every))
    if isinstance(every, float):
        if every <= 0:
            raise ValueError(f"every= must be positive (got {every})")
        return ("time", float(every))
    if isinstance(every, str):
        match = _DURATION_RE.match(every)
        if not match:
            raise ValueError(
                f"every={every!r} unparseable; use 'Ns', 'Nm', 'Nh', 'Nd' or an int"
            )
        n = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        return ("time", n * _DURATION_FACTORS[unit])
    raise TypeError(
        f"every= must be None, int, float, or str; got {type(every).__name__}"
    )


@dataclass
class _Dispatcher:
    name: str
    adapter: Any  # BaseOutput subclass
    mode: str  # "event" | "steps" | "time"
    every: Optional[float]
    buffer: List[LogEvent] = field(default_factory=list)
    step_counter: int = 0
    last_flush_at: float = field(default_factory=time.time)


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
        flush_secs: int = 5,
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
        self._flush_secs = flush_secs
        self._flush_timer: Optional[threading.Timer] = None
        self._buffer_lock = threading.RLock()
        self._scalar_buffer: List[Tuple[int, str, int, float, float]] = []
        self._step_counters: Dict[str, int] = {}
        self._sysmetrics: Any = None

        # Adapter dispatch state (populated by .to(...))
        self._dispatchers: List[_Dispatcher] = []
        self._dispatch_lock = threading.Lock()
        self._dispatch_executor: Optional[ThreadPoolExecutor] = None
        self._one_shot_adapters: Dict[Tuple[str, frozenset], Any] = {}

        # Rank gating — only rank 0 logs, others become no-op
        if rank is None:
            self._rank = _detect_rank()
        elif rank == "all":
            self._rank = 0
        else:
            self._rank = int(rank)

        self._pending = False
        self._pending_existing_id: Optional[int] = None
        self._pending_max_step: Optional[int] = None
        self._pending_config: Optional[Dict[str, Any]] = None

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

            # Create or defer experiment row resolution
            existing = self._db.get_experiment_by_name(
                self._run_name,
                project=self._project,
            )
            if existing is not None:
                self._pending = True
                self._pending_existing_id = existing["id"]
                self._pending_max_step = self._db.get_max_step(existing["id"])
                self._pending_config = config
                self._exp_id = existing["id"]  # optimistic default
            else:
                self._pending = False
                self._exp_id = self._db.create_experiment(
                    self._run_name,
                    config=config,
                    project=self._project,
                    log_dir=self.log_dir,
                )

            # Register remap callback for when precache materializes
            if precache_secs > 0:
                self._db.register_remap_callback(self._on_precache_flush)
            if self._flush_secs > 0:
                self._start_flush_timer()
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

    def _best_effort_emit(
        self,
        action: str,
        fn: Callable[[], Optional[LogEvent]],
    ) -> Any:
        """Like :meth:`_best_effort` but emits the returned :class:`LogEvent`
        to registered dispatchers and returns an :class:`EventHandle`.

        ``fn`` may return ``None`` to signal "no event" (e.g. NaN dropped); in
        that case a :class:`NullEventHandle` is returned so the caller's
        ``.to(...)`` is a safe no-op.
        """
        if not self._enabled or self._closed:
            return null_handle()
        try:
            event = fn()
        except Exception as exc:
            self._handle_runtime_error(action, exc)
            return null_handle()
        if event is None:
            return null_handle()
        self._emit(event)
        return EventHandle(self, event)

    # ── Adapter dispatch (`.to(...)`) ───────────────────────────

    def to(
        self,
        name: str,
        every: Optional[Union[int, float, str]] = None,
        **creds: Any,
    ) -> "SummaryWriter":
        """Register *name* as a persistent adapter destination.

        ``every`` controls throttling: ``None`` dispatches every event;
        an ``int`` sends every N events; a string like ``"15m"``/``"1h"``/
        ``"5s"`` groups events into time-based digests.

        Returns ``self`` so calls chain: ``writer.to("slack", every="15m").to("telegram", every=100)``.
        """
        if not self._enabled or self._closed:
            return self
        mode, value = _parse_every(every)
        try:
            from .viewers import load_viewer

            adapter_cls = load_viewer(name)
            adapter = adapter_cls(
                project_folder=self.project_folder,
                project=self._project,
                **creds,
            )
        except TypeError:
            # Fall back for adapters that don't accept project/project_folder kwargs
            try:
                adapter = adapter_cls(**creds)  # type: ignore[assignment]
            except Exception as exc:
                self._handle_runtime_error(f"load adapter {name!r}", exc)
                return self
        except Exception as exc:
            self._handle_runtime_error(f"load adapter {name!r}", exc)
            return self
        disp = _Dispatcher(name=name, adapter=adapter, mode=mode, every=value)
        with self._dispatch_lock:
            self._dispatchers.append(disp)
        return self

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._dispatch_executor is None:
            self._dispatch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"vibetrack-dispatch-{self._run_name}",
            )
        return self._dispatch_executor

    def _emit(self, event: LogEvent) -> None:
        """Route *event* through each registered dispatcher, honouring throttling."""
        if not self._dispatchers:
            return
        with self._dispatch_lock:
            dispatchers = list(self._dispatchers)
        now = event.walltime
        for disp in dispatchers:
            disp.buffer.append(event)
            disp.step_counter += 1
            flush = False
            if disp.mode == "event":
                flush = True
            elif disp.mode == "steps" and disp.every is not None:
                if disp.step_counter >= int(disp.every):
                    flush = True
            elif disp.mode == "time" and disp.every is not None:
                if now - disp.last_flush_at >= disp.every:
                    flush = True
            if flush:
                self._flush_dispatcher(disp, now=now)

    def _flush_dispatcher(self, disp: _Dispatcher, now: Optional[float] = None) -> None:
        if not disp.buffer:
            return
        events = disp.buffer
        disp.buffer = []
        disp.step_counter = 0
        disp.last_flush_at = now if now is not None else time.time()
        try:
            executor = self._get_executor()
            executor.submit(self._safe_send, disp.adapter, events, disp.name)
        except RuntimeError as exc:
            # Executor shut down (writer closing); do a best-effort sync send.
            self._safe_send(disp.adapter, events, disp.name)
            self._handle_runtime_error(f"submit to {disp.name}", exc)

    def _safe_send(self, adapter: Any, events: List[LogEvent], name: str) -> None:
        try:
            adapter.send(events)
        except Exception as exc:
            self._handle_runtime_error(f"{name}.send", exc)

    def _flush_dispatchers(self, final: bool = False) -> None:
        with self._dispatch_lock:
            dispatchers = list(self._dispatchers)
        now = time.time()
        for disp in dispatchers:
            if disp.buffer and (final or disp.mode == "time"):
                if disp.mode == "time" and disp.every is not None:
                    if not final and now - disp.last_flush_at < disp.every:
                        continue
                self._flush_dispatcher(disp, now=now)

    def _one_shot_send(self, event: LogEvent, name: str, **creds: Any) -> None:
        """Dispatch a single event to *name* without registering it permanently."""
        if not self._enabled or self._closed:
            return
        key = (name, frozenset(creds.items()))
        adapter = self._one_shot_adapters.get(key)
        if adapter is None:
            try:
                from .viewers import load_viewer

                adapter_cls = load_viewer(name)
                try:
                    adapter = adapter_cls(
                        project_folder=self.project_folder,
                        project=self._project,
                        **creds,
                    )
                except TypeError:
                    adapter = adapter_cls(**creds)
            except Exception as exc:
                self._handle_runtime_error(f"one-shot load {name!r}", exc)
                return
            self._one_shot_adapters[key] = adapter
        try:
            executor = self._get_executor()
            executor.submit(self._safe_send, adapter, [event], name)
        except RuntimeError:
            self._safe_send(adapter, [event], name)

    # ── TensorBoard-compatible API ──────────────────────────────

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        new_style: bool = False,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            val = float(scalar_value)
            if not math.isfinite(val):
                return None  # skip NaN / Inf
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            with self._buffer_lock:
                self._resolve_experiment(step)
                self._scalar_buffer.append((self._exp_id, tag, step, val, wt))
                if len(self._scalar_buffer) >= self._max_queue:
                    self._flush_locked()
            return LogEvent(
                kind="scalar",
                tag=tag,
                step=step,
                value=val,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
            )

        return self._best_effort_emit("add scalar", _op)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        if not self._enabled or self._closed:
            return null_handle()
        handles: List[Any] = []
        try:
            for sub_tag, value in tag_scalar_dict.items():
                handles.append(
                    self.add_scalar(
                        f"{main_tag}/{sub_tag}", value, global_step, walltime
                    )
                )
        except Exception as exc:
            self._handle_runtime_error("add scalars", exc)
            return null_handle()
        return MultiEventHandle(handles)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            self._db.add_text(self._exp_id, tag, text_string, step, wt)
            return LogEvent(
                kind="text",
                tag=tag,
                step=step,
                value=text_string,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
            )

        return self._best_effort_emit("add text", _op)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW",
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_image

            rel_path = save_image(img_tensor, self.log_dir, tag, step)
            self._db.add_image(self._exp_id, tag, rel_path, step, wt)
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="image",
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"rel_path": rel_path},
            )

        return self._best_effort_emit("add image", _op)

    def add_audio(
        self,
        tag: str,
        snd_tensor: Any,
        global_step: Optional[int] = None,
        sample_rate: int = 44100,
        walltime: Optional[float] = None,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_audio

            rel_path = save_audio(snd_tensor, self.log_dir, tag, step, sample_rate)
            self._db.add_audio(self._exp_id, tag, rel_path, step, sample_rate, wt)
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="audio",
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"sample_rate": sample_rate, "rel_path": rel_path},
            )

        return self._best_effort_emit("add audio", _op)

    def add_video(
        self,
        tag: str,
        vid_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        fps: int = 4,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_video

            rel_path = save_video(vid_tensor, self.log_dir, tag, step)
            self._db.add_video(self._exp_id, tag, rel_path, step, wt)
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="video",
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"fps": fps, "rel_path": rel_path},
            )

        return self._best_effort_emit("add video", _op)

    def add_artifact(
        self,
        tag: str,
        file_path: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_artifact

            rel_path, auto_meta = save_artifact(file_path, self.log_dir, tag, step)
            if metadata:
                auto_meta.update(metadata)
            self._db.add_artifact(
                self._exp_id, tag, rel_path, json.dumps(auto_meta), step, wt
            )
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="artifact",
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"metadata": auto_meta, "rel_path": rel_path},
            )

        return self._best_effort_emit("add artifact", _op)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: str = "tensorflow",
        walltime: Optional[float] = None,
        max_bins: Optional[int] = None,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            # Simple histogram: convert values to list, compute bins
            try:
                vals = list(values)
            except TypeError:
                vals = [float(values)]

            n_bins = max_bins or 30
            if len(vals) == 0:
                return None
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

            self._db.add_histogram(self._exp_id, tag, bin_edges, counts, step, wt)
            return LogEvent(
                kind="histogram",
                tag=tag,
                step=step,
                value={"bin_edges": bin_edges, "counts": counts},
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
            )

        return self._best_effort_emit("add histogram", _op)

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Optional[Dict[str, Any]] = None,
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
    ) -> Any:
        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            # hparams have no step — if still pending, force resume
            if self._pending:
                self._resolve_experiment(
                    self._pending_max_step + 1
                    if self._pending_max_step is not None
                    else 0
                )
            self._db.add_hparams(self._exp_id, hparam_dict)
            if metric_dict:
                for tag, val in metric_dict.items():
                    self.add_scalar(tag, val)
            return LogEvent(
                kind="hparams",
                tag="hparams",
                step=None,
                value=dict(hparam_dict),
                walltime=time.time(),
                run_name=self._run_name,
                project=self._project,
                extra={"metrics": dict(metric_dict) if metric_dict else {}},
            )

        return self._best_effort_emit("add hparams", _op)

    # ── W&B-style log() ─────────────────────────────────────────

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> Any:
        """W&B-compatible log call: ``writer.log({"loss": 0.5, "acc": 0.9})``."""
        if not self._enabled or self._closed:
            return null_handle()
        from .types import Image, Audio, Video, Artifact

        handles: List[Any] = []
        try:
            for tag, value in data.items():
                if isinstance(value, (int, float)):
                    handles.append(self.add_scalar(tag, value, global_step=step))
                elif isinstance(value, str):
                    handles.append(self.add_text(tag, value, global_step=step))
                elif isinstance(value, Image):
                    handles.append(self.add_image(tag, value.data, global_step=step))
                elif isinstance(value, Audio):
                    handles.append(
                        self.add_audio(
                            tag,
                            value.data,
                            global_step=step,
                            sample_rate=value.sample_rate,
                        )
                    )
                elif isinstance(value, Video):
                    handles.append(self.add_video(tag, value.data, global_step=step))
                elif isinstance(value, Artifact):
                    handles.append(
                        self.add_artifact(
                            tag,
                            value.path,
                            global_step=step,
                            metadata=value.metadata,
                        )
                    )
            if commit:
                with self._buffer_lock:
                    self._flush_locked()
        except Exception as exc:
            self._handle_runtime_error("log data", exc)
            return null_handle()
        return MultiEventHandle(handles)

    # ── Lifecycle ───────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Flush buffer. Caller must hold _buffer_lock."""
        if self._db is not None and self._scalar_buffer:
            self._db.add_scalars_bulk(self._scalar_buffer)
            self._scalar_buffer.clear()

    def _start_flush_timer(self) -> None:
        if self._flush_secs <= 0 or self._closed or not self._enabled:
            return
        timer = threading.Timer(self._flush_secs, self._on_flush_timer)
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _on_flush_timer(self) -> None:
        if self._closed or not self._enabled:
            return
        try:
            with self._buffer_lock:
                self._flush_locked()
        except Exception as exc:
            self._handle_runtime_error("flush timer", exc)
        try:
            self._flush_dispatchers(final=False)
        except Exception as exc:
            self._handle_runtime_error("dispatch flush timer", exc)
        finally:
            self._flush_timer = None
            if not self._closed and self._enabled:
                self._start_flush_timer()

    def flush(self) -> None:
        """Compatibility no-op for TensorBoard-style APIs."""
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enabled = False
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
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
            self._flush_dispatchers(final=True)
        except Exception as exc:
            self._handle_runtime_error("final dispatch flush", exc)
        if self._dispatch_executor is not None:
            try:
                self._dispatch_executor.shutdown(wait=True)
            except Exception as exc:
                self._handle_runtime_error("shutdown dispatcher", exc)
            self._dispatch_executor = None
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

    def _resolve_experiment(self, first_step: int) -> None:
        """Resolve resume vs restart on first write. Thread-safe."""
        if not self._pending:
            return
        with self._buffer_lock:
            if not self._pending:
                return  # double-check after lock
            if self._pending_max_step is None or first_step > self._pending_max_step:
                # Resume — keep existing experiment
                self._exp_id = self._pending_existing_id  # type: ignore[assignment]
            else:
                # Restart — create new experiment with suffix. Mirror the
                # suffix onto log_dir too so sibling experiments don't share
                # a media folder (deleting one would otherwise rmtree the
                # other's files, and same-step writes would collide).
                new_name = self._db.find_next_suffix_name(self._run_name, self._project)  # type: ignore[union-attr]
                suffix = new_name[len(self._run_name) :]  # " (2)", " (3)", …
                new_log_dir = self.log_dir + suffix
                self.log_dir = new_log_dir
                Path(new_log_dir).mkdir(parents=True, exist_ok=True)
                self._exp_id = self._db.create_experiment(  # type: ignore[union-attr]
                    new_name,
                    config=self._pending_config,
                    project=self._project,
                    log_dir=new_log_dir,
                )
                self._run_name = new_name
            self._pending = False
            self._pending_existing_id = None
            self._pending_max_step = None
            self._pending_config = None

    def _on_precache_flush(self, id_remap: Dict[int, int]) -> None:
        """Called by Database after precache materializes to remap IDs."""
        # Remap pending ID if resolution hasn't happened yet
        if self._pending and self._pending_existing_id in id_remap:
            self._pending_existing_id = id_remap[self._pending_existing_id]
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
