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
    summary: bool = False
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


def _is_torch_tensor(obj: Any) -> bool:
    return type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor"


def _is_numpy_array(obj: Any) -> bool:
    return type(obj).__module__ == "numpy" and type(obj).__name__ == "ndarray"


def _to_numpy_array(obj: Any) -> Optional[Any]:
    if _is_torch_tensor(obj):
        return obj.detach().cpu().numpy()
    if _is_numpy_array(obj):
        return obj
    return None


def _shape_of(obj: Any) -> Optional[List[int]]:
    arr = _to_numpy_array(obj)
    if arr is not None:
        return [int(v) for v in arr.shape]
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return [int(v) for v in shape]
        except Exception:
            return None
    return None


def _jsonable(obj: Any, include_array_values: bool = True) -> Any:
    arr = _to_numpy_array(obj)
    if arr is not None:
        payload: Dict[str, Any] = {"shape": [int(v) for v in arr.shape]}
        if include_array_values:
            payload["values"] = arr.tolist()
        return payload
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, include_array_values) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, include_array_values) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return repr(obj)


def _flatten_numeric(values: Any) -> List[float]:
    arr = _to_numpy_array(values)
    if arr is not None:
        return [float(v) for v in arr.reshape(-1).tolist()]
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("string/blob histogram inputs are not supported")
    if isinstance(values, dict):
        iterable = values.values()
    else:
        try:
            iterable = iter(values)
        except TypeError:
            iterable = (values,)
    out: List[float] = []
    for value in iterable:
        if isinstance(value, (list, tuple, dict)) or _to_numpy_array(value) is not None:
            out.extend(_flatten_numeric(value))
            continue
        if hasattr(value, "item"):
            value = value.item()
        out.append(float(value))
    return out


def _render_figures_to_image(figure: Any, close: bool) -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("add_figure requires matplotlib and numpy")

    figures = figure if isinstance(figure, list) else [figure]
    images = []
    for fig in figures:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        images.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())

    if close:
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]

            for fig in figures:
                plt.close(fig)
        except Exception:
            pass

    if len(images) == 1:
        return images[0]

    max_h = max(img.shape[0] for img in images)
    total_w = sum(img.shape[1] for img in images)
    grid = np.full((max_h, total_w, 3), 255, dtype=images[0].dtype)
    x = 0
    for img in images:
        grid[: img.shape[0], x : x + img.shape[1], :] = img
        x += img.shape[1]
    return grid


def _hwc_to_png_bytes(arr: Any) -> bytes:
    """Encode an HWC uint8 numpy array as PNG bytes via PIL."""
    import io as _io

    from PIL import Image as _PILImage  # type: ignore[import-untyped]

    buf = _io.BytesIO()
    _PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _render_label_img_atlas(label_img: Any) -> Tuple[bytes, Dict[str, Any]]:
    """Pack a batch of thumbnails into a single sprite-atlas PNG.

    Accepts NCHW (torch convention) or NHWC (TF/numpy convention).
    Returns ``(png_bytes, {"tile_w","tile_h","cols","rows","count"})`` —
    the PNG laid out left-to-right, top-to-bottom in a near-square grid so
    sprite index ``i`` lives at ``(col=i%cols, row=i//cols)`` for the JS UV
    math in the embedding viewer.
    """
    import io as _io
    import math as _math

    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(
        _to_numpy_array(label_img)
        if _to_numpy_array(label_img) is not None
        else label_img
    )
    if arr.ndim != 4:
        raise ValueError(
            f"label_img must be a 4-D batch (NCHW or NHWC), got shape {arr.shape}"
        )
    # NCHW → NHWC if the second axis looks like a channel count.
    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"label_img channel axis must be 1/3/4, got shape {arr.shape}")

    # Coerce to uint8 with TB-style float-in-[0,1] convention.
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    n, h, w, c = arr.shape
    if c == 1:
        arr = np.repeat(arr, 3, axis=-1)
        c = 3
    cols = max(1, int(_math.ceil(_math.sqrt(n))))
    rows = max(1, int(_math.ceil(n / cols)))
    atlas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        col = i % cols
        atlas[r * h : r * h + h, col * w : col * w + w, :] = arr[i]

    from PIL import Image as _PILImage  # type: ignore[import-untyped]

    buf = _io.BytesIO()
    mode = "RGBA" if c == 4 else "RGB"
    _PILImage.fromarray(atlas, mode=mode).save(buf, format="PNG")
    meta = {
        "tile_w": int(w),
        "tile_h": int(h),
        "cols": int(cols),
        "rows": int(rows),
        "count": int(n),
    }
    return buf.getvalue(), meta


def _pr_curve_payload(
    labels: Any,
    predictions: Any,
    num_thresholds: int,
    weights: Optional[Any] = None,
) -> Dict[str, Any]:
    label_values = [bool(v) for v in _flatten_numeric(labels)]
    pred_values = _flatten_numeric(predictions)
    weight_values = (
        _flatten_numeric(weights)
        if weights is not None
        else [1.0 for _ in range(len(pred_values))]
    )
    n = min(len(label_values), len(pred_values), len(weight_values))
    if n == 0:
        raise ValueError("labels and predictions must be non-empty")
    label_values = label_values[:n]
    pred_values = pred_values[:n]
    weight_values = weight_values[:n]
    thresholds = (
        [0.0]
        if num_thresholds <= 1
        else [i / float(num_thresholds - 1) for i in range(num_thresholds)]
    )
    points = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0.0
        for label, pred, weight in zip(label_values, pred_values, weight_values):
            positive = pred >= threshold
            if positive and label:
                tp += weight
            elif positive and not label:
                fp += weight
            elif not positive and label:
                fn += weight
            else:
                tn += weight
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        points.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return {"num_examples": n, "points": points}


def _pr_curve_raw_payload(
    true_positive_counts: Any,
    false_positive_counts: Any,
    true_negative_counts: Any,
    false_negative_counts: Any,
    precision: Any,
    recall: Any,
    num_thresholds: int,
    weights: Optional[Any] = None,
) -> Dict[str, Any]:
    tp_values = _flatten_numeric(true_positive_counts)
    fp_values = _flatten_numeric(false_positive_counts)
    tn_values = _flatten_numeric(true_negative_counts)
    fn_values = _flatten_numeric(false_negative_counts)
    precision_values = _flatten_numeric(precision)
    recall_values = _flatten_numeric(recall)
    n = min(
        len(tp_values),
        len(fp_values),
        len(tn_values),
        len(fn_values),
        len(precision_values),
        len(recall_values),
    )
    if n == 0:
        raise ValueError("raw PR curve arrays must be non-empty")
    thresholds = [0.0] if n <= 1 else [i / float(n - 1) for i in range(n)]
    points = [
        {
            "threshold": thresholds[i],
            "precision": precision_values[i],
            "recall": recall_values[i],
            "tp": tp_values[i],
            "fp": fp_values[i],
            "fn": fn_values[i],
            "tn": tn_values[i],
        }
        for i in range(n)
    ]
    payload: Dict[str, Any] = {
        "num_thresholds": int(num_thresholds),
        "points": points,
    }
    if weights is not None:
        payload["weights"] = _jsonable(weights, include_array_values=True)
    return payload


def _draw_image_with_boxes(
    img_tensor: Any,
    box_tensor: Any,
    rescale: float = 1,
    dataformats: str = "CHW",
    labels: Optional[Any] = None,
) -> Any:
    import numpy as np  # type: ignore[import-untyped]
    from PIL import Image as _PILImage  # type: ignore[import-untyped]
    from PIL import ImageDraw, ImageFont  # type: ignore[import-untyped]

    from .media import _as_uint8_image, _is_pil_image, _normalize_image_array

    if isinstance(img_tensor, (str, Path)):
        image = _PILImage.open(img_tensor).convert("RGB")
    elif _is_pil_image(img_tensor):
        image = img_tensor.copy()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
    else:
        arr = _to_numpy_array(img_tensor)
        if arr is None:
            arr = img_tensor
        arr = _normalize_image_array(arr, dataformats)
        arr = _as_uint8_image(arr)
        if arr.ndim == 2:
            image = _PILImage.fromarray(arr, mode="L").convert("RGB")
        else:
            image = _PILImage.fromarray(arr)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

    boxes = np.asarray(
        (
            _to_numpy_array(box_tensor)
            if _to_numpy_array(box_tensor) is not None
            else box_tensor
        ),
        dtype=float,
    ).reshape((-1, 4))

    label_values: Optional[List[str]] = None
    if labels is not None:
        label_values = [labels] if isinstance(labels, str) else list(labels)
        label_values = [str(label) for label in label_values]
        if len(label_values) != boxes.shape[0]:
            label_values = None

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=2,
            fill="red",
        )
        if label_values:
            text = label_values[i]
            left, top, right, bottom = font.getbbox(text)
            text_w = right - left
            text_h = bottom - top
            margin = int(np.ceil(0.05 * text_h))
            text_bottom = ymax
            draw.rectangle(
                [
                    (xmin, text_bottom - text_h - 2 * margin),
                    (xmin + text_w, text_bottom),
                ],
                fill="red",
            )
            draw.text(
                (xmin + margin, text_bottom - text_h - margin),
                text,
                fill="black",
                font=font,
            )

    if rescale != 1:
        width = max(1, int(image.width * float(rescale)))
        height = max(1, int(image.height * float(rescale)))
        try:
            resample = _PILImage.Resampling.LANCZOS
        except AttributeError:
            resample = _PILImage.LANCZOS
        image = image.resize((width, height), resample)
    return image


def _looks_like_dot(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("digraph ") or stripped.startswith("graph ")


def _default_dot_graph(model: Any, input_to_model: Any = None) -> str:
    model_label = (
        str(model).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )
    input_shape = _shape_of(input_to_model)
    input_label = (
        "input"
        if input_shape is None
        else "input\\nshape=" + "x".join(str(v) for v in input_shape)
    )
    return (
        "digraph vibetrack_graph {\n"
        "  rankdir=LR;\n"
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.6"];\n'
        '  node [shape=box, style="rounded,filled", color="#4b5563", '
        'fillcolor="#dbeafe", fontname="Inter,Arial", fontsize=11];\n'
        '  edge [color="#64748b", arrowsize="0.8"];\n'
        f'  input [label="{input_label}", fillcolor="#dcfce7"];\n'
        f'  model [label="{model_label}"];\n'
        '  output [label="output", fillcolor="#fae8ff"];\n'
        "  input -> model -> output;\n"
        "}\n"
    )


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

    Errors during initialization or any ``add_*`` call are caught and
    logged once to stderr; subsequent calls become no-ops returning a
    null event handle. Non-rank-0 processes (detected via ``RANK`` /
    ``LOCAL_RANK``) are silently disabled unless ``rank="all"``.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
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
        """Open (or resume) an experiment row in the central / project DB.

        Args:
            log_dir: Directory for media files. Defaults to
                ``runs/<timestamp>``; if ``project`` and ``name`` are given,
                defaults to ``<project>/<name>``.
            comment: Suffix appended to the auto-generated run name when
                ``log_dir`` is not supplied.
            purge_step: When resuming an existing run, delete all rows with
                ``step >= purge_step`` before writing.
            max_queue: Number of buffered scalars before an automatic flush.
            flush_secs: Periodic flush interval, in seconds. ``0`` disables
                the timer (only ``max_queue`` and ``flush()`` trigger writes).
            filename_suffix: Reserved for TensorBoard parity; currently unused.
            project: Logical project name. Becomes the row's ``project``
                column and is used by readers to filter runs.
            name: Run name. Defaults to a timestamp; reusing a name resumes
                the existing experiment unless ``purge_step`` is set.
            config: Hparam-style dict persisted as the run's initial hparams
                and stored on the experiment row for later inspection.
            precache_secs: Buffer all writes in memory for this many seconds
                before materializing to SQLite. Useful for short-lived runs.
            system_metrics_interval: Sample interval for the background CPU /
                GPU / disk collector. ``0`` disables collection.
            project_folder: When set, store data in a local
                ``vibetrack.db`` here instead of the central DB.
            rank: Process rank for distributed training. ``None`` reads
                ``RANK`` / ``LOCAL_RANK`` from the environment; ``"all"``
                forces every rank to log; an int gates on ``rank == 0``.
        """
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
        self._purge_step = purge_step
        self._filename_suffix = filename_suffix
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
            if existing is not None and purge_step is not None:
                self._pending = False
                self._exp_id = existing["id"]
                self._db.purge_experiment_from_step(self._exp_id, purge_step)
            elif existing is not None:
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
        summary: bool = False,
        **creds: Any,
    ) -> "SummaryWriter":
        """Register *name* as a persistent adapter destination.

        ``every`` controls throttling: ``None`` dispatches every event;
        an ``int`` sends every N events; a string like ``"15m"``/``"1h"``/
        ``"5s"`` groups events into time-based digests.

        ``summary=True`` opts the adapter into an end-of-run digest sent
        from :meth:`close` (one post containing scalar charts, latest
        media, stitched image series, and concatenated text). Adapters
        without a ``send_summary`` implementation silently ignore it.

        Returns ``self`` so calls chain: ``writer.to("slack", every="15m", summary=True).to("telegram", every=100)``.
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
        disp = _Dispatcher(
            name=name, adapter=adapter, mode=mode, every=value, summary=summary
        )
        with self._dispatch_lock:
            self._dispatchers.append(disp)
        self._start_adapter_if_supported(adapter, name)
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
            # summary=True dispatchers receive only the close-time digest;
            # skip per-event buffering entirely so we don't double-post.
            if disp.summary:
                continue
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

    def _start_adapter_if_supported(self, adapter: Any, name: str) -> None:
        start = getattr(adapter, "start", None)
        if not callable(start):
            return
        try:
            start()
        except Exception as exc:
            self._handle_runtime_error(f"{name}.start", exc)

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

    def _send_summaries(self) -> None:
        """Invoke ``send_summary`` on every dispatcher that opted in.

        Called from :meth:`close` after the executor has drained, so this
        runs synchronously and after the per-event flush has completed.
        """
        with self._dispatch_lock:
            dispatchers = list(self._dispatchers)
        for disp in dispatchers:
            if not disp.summary:
                continue
            try:
                disp.adapter.send_summary(self._run_name, self._project)
            except Exception as exc:
                self._handle_runtime_error(f"{disp.name}.send_summary", exc)

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
            self._start_adapter_if_supported(adapter, name)
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
        double_precision: bool = False,
    ) -> Any:
        """Record a single scalar value.

        Args:
            tag: Series name (e.g. ``"loss"`` or ``"train/acc"``).
            scalar_value: Numeric value. Non-finite values (NaN / Inf) are
                silently dropped — the writer keeps going rather than poison
                the chart.
            global_step: Step number. ``None`` auto-increments per tag.
            walltime: Override timestamp; defaults to ``time.time()``.
            new_style: Reserved for TensorBoard parity; ignored.
            double_precision: Reserved for TensorBoard parity; ignored.

        Returns:
            An :class:`EventHandle` that can be chained with ``.to(...)`` to
            forward this single event to a one-shot adapter.
        """

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
        """Record several scalars under the same group.

        Each entry of ``tag_scalar_dict`` is logged as
        ``"{main_tag}/{sub_tag}"`` so they share a chart group in the UI.

        Args:
            main_tag: Group prefix (e.g. ``"loss"``).
            tag_scalar_dict: Mapping of sub-tag → value.
            global_step: Step number applied to every scalar.
            walltime: Override timestamp.

        Returns:
            A :class:`MultiEventHandle` aggregating one handle per scalar.
        """
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
        """Append a text snippet (Markdown supported in the web UI).

        Args:
            tag: Channel name (e.g. ``"notes"``).
            text_string: Body of the entry. Stored verbatim.
            global_step: Step number; ``None`` auto-increments per tag.
            walltime: Override timestamp.
        """

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
        """Save a single image (or a batch / grid) under ``tag``.

        Args:
            tag: Image series name.
            img_tensor: A torch / numpy tensor, a PIL image, a path on disk,
                or raw bytes. Floats in ``[0, 1]`` are scaled to ``uint8``.
            global_step: Step number; ``None`` auto-increments per tag.
            walltime: Override timestamp.
            dataformats: Axis ordering string — ``"CHW"`` (default),
                ``"HWC"``, ``"HW"``, or ``"NCHW"`` / ``"NHWC"`` for batches
                (which are tiled into a grid).
        """

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_image

            rel_path = save_image(
                img_tensor,
                self.log_dir,
                tag,
                step,
                dataformats=dataformats,
            )
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

    def add_images(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "NCHW",
    ) -> Any:
        """Save a batch of images. Alias for :meth:`add_image` defaulting
        to the ``"NCHW"`` dataformat — kept for TensorBoard parity.
        """
        return self.add_image(
            tag,
            img_tensor,
            global_step=global_step,
            walltime=walltime,
            dataformats=dataformats,
        )

    def add_image_with_boxes(
        self,
        tag: str,
        img_tensor: Any,
        box_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        rescale: float = 1,
        dataformats: str = "CHW",
        labels: Optional[Any] = None,
    ) -> Any:
        """Draw bounding boxes on an image and save the result.

        Args:
            tag: Image series name.
            img_tensor: Input image (see :meth:`add_image`).
            box_tensor: ``(N, 4)`` array of ``(xmin, ymin, xmax, ymax)``
                coordinates in pixels.
            global_step: Step number; ``None`` auto-increments per tag.
            walltime: Override timestamp.
            rescale: Multiplicative resize applied after drawing.
            dataformats: Axis ordering — see :meth:`add_image`.
            labels: Optional list of strings, one per box. Length must match
                the box count or labels are silently dropped.
        """
        if not self._enabled or self._closed:
            return null_handle()
        try:
            boxed = _draw_image_with_boxes(
                img_tensor,
                box_tensor,
                rescale=rescale,
                dataformats=dataformats,
                labels=labels,
            )
        except Exception as exc:
            self._handle_runtime_error("add image with boxes", exc)
            return null_handle()
        return self.add_image(
            tag,
            boxed,
            global_step=global_step,
            walltime=walltime,
            dataformats="HWC",
        )

    def add_figure(
        self,
        tag: str,
        figure: Any,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> Any:
        """Render a matplotlib ``Figure`` (or list of figures) and store it.

        Figures are user-rendered *charts*, not media — they live alongside
        scalar charts in the web UI's Scalars tab rather than in the Images
        gallery. Stored as an artifact with ``kind="figure"`` so the
        serializer can route them to the right bucket.
        """
        if not self._enabled or self._closed:
            return null_handle()
        try:
            image = _render_figures_to_image(figure, close=close)
            png_bytes = _hwc_to_png_bytes(image)
        except Exception as exc:
            self._handle_runtime_error("add figure", exc)
            return null_handle()
        shape = list(image.shape) if hasattr(image, "shape") else None
        return self._add_binary_artifact_event(
            "add figure",
            "figure",
            tag,
            png_bytes,
            ext=".png",
            extra_meta={"format": "png", "shape": shape},
            global_step=global_step,
            walltime=walltime,
        )

    def add_audio(
        self,
        tag: str,
        snd_tensor: Any,
        global_step: Optional[int] = None,
        sample_rate: int = 44100,
        walltime: Optional[float] = None,
    ) -> Any:
        """Save an audio clip as a WAV file.

        Args:
            tag: Audio series name.
            snd_tensor: 1-D / 2-D tensor of samples in ``[-1, 1]`` or
                ``int16`` PCM.
            global_step: Step number; ``None`` auto-increments per tag.
            sample_rate: Samples per second. Defaults to ``44100``.
            walltime: Override timestamp.
        """

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
        fps: int = 4,
        walltime: Optional[float] = None,
    ) -> Any:
        """Save a clip as MP4 (or GIF when ffmpeg is unavailable).

        Args:
            tag: Video series name.
            vid_tensor: ``(T, H, W, C)`` or ``(T, C, H, W)`` frame stack,
                a path to an existing video, or raw bytes.
            global_step: Step number; ``None`` auto-increments per tag.
            fps: Playback rate in frames per second.
            walltime: Override timestamp.
        """

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_video

            rel_path = save_video(vid_tensor, self.log_dir, tag, step, fps=fps)
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
        """Copy an arbitrary file into the run's media folder and index it.

        Args:
            tag: Artifact channel name.
            file_path: Path to the source file. Copied (not symlinked) into
                ``{log_dir}/media/{tag}/``.
            global_step: Step number; ``None`` auto-increments per tag.
            walltime: Override timestamp.
            metadata: User dict merged on top of auto-detected file
                metadata (size, mime type, etc.).
        """

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
        """Bin ``values`` and store the resulting histogram.

        Args:
            tag: Histogram series name.
            values: 1-D iterable / numpy array / torch tensor of numbers.
                Empty inputs are silently skipped.
            global_step: Step number; ``None`` auto-increments per tag.
            bins: Reserved for TensorBoard parity (only equal-width binning
                is implemented).
            walltime: Override timestamp.
            max_bins: Number of bins. Defaults to ``30``.
        """

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            vals = _flatten_numeric(values)

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

    def add_histogram_raw(
        self,
        tag: str,
        min: float,
        max: float,
        num: float,
        sum: float,
        sum_squares: float,
        bucket_limits: Any,
        bucket_counts: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        """Store a pre-binned histogram.

        TensorBoard-compatible: skip the binning pass and pass the bucket
        layout directly. Useful when the caller already maintains running
        statistics (e.g. across distributed workers).

        Args:
            tag: Histogram series name.
            min: Lower bound of all observations.
            max: Upper bound of all observations.
            num: Total observation count.
            sum: Sum of observations (for mean reconstruction).
            sum_squares: Sum of squares (for variance reconstruction).
            bucket_limits: Right-edge of each bucket. Length ``B`` or
                ``B + 1`` (in which case it is treated as the bin edges).
            bucket_counts: Count per bucket; length ``B``.
            global_step: Step number; ``None`` auto-increments per tag.
            walltime: Override timestamp.
        """

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            limits = _flatten_numeric(bucket_limits)
            counts = _flatten_numeric(bucket_counts)
            if not limits or not counts:
                return None
            if len(limits) == len(counts) + 1:
                bins = limits
            else:
                usable = len(limits) if len(limits) < len(counts) else len(counts)
                limits = limits[:usable]
                counts = counts[:usable]
                lower = float(min)
                if not math.isfinite(lower):
                    lower = limits[0]
                bins = [lower] + limits
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            self._db.add_histogram(self._exp_id, tag, bins, counts, step, wt)
            return LogEvent(
                kind="histogram",
                tag=tag,
                step=step,
                value={
                    "bin_edges": bins,
                    "counts": counts,
                    "min": float(min),
                    "max": float(max),
                    "num": float(num),
                    "sum": float(sum),
                    "sum_squares": float(sum_squares),
                },
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
            )

        return self._best_effort_emit("add histogram raw", _op)

    def _add_json_artifact_event(
        self,
        action: str,
        kind: str,
        tag: str,
        payload: Dict[str, Any],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        data = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        return self._add_binary_artifact_event(
            action,
            kind,
            tag,
            data,
            ext=".json",
            global_step=global_step,
            walltime=walltime,
        )

    def _add_binary_artifact_event(
        self,
        action: str,
        kind: str,
        tag: str,
        payload: bytes,
        ext: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        """Persist a raw binary blob as an artifact tagged with *kind*.

        Used by writers (``add_figure``) whose canonical on-disk form is the
        binary itself (a PNG) rather than a JSON sidecar. Mirrors the DB +
        LogEvent emission of :meth:`_add_json_artifact_event` so adapter
        dispatch stays uniform.
        """

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = walltime if walltime is not None else time.time()
            self._resolve_experiment(step)
            from .media import save_artifact

            filename = f"{step}{ext}" if ext.startswith(".") else f"{step}.{ext}"
            rel_path, metadata = save_artifact(
                payload,
                self.log_dir,
                tag,
                step,
                filename=filename,
            )
            metadata.update({"kind": kind})
            if extra_meta:
                metadata.update(extra_meta)
            self._db.add_artifact(
                self._exp_id,
                tag,
                rel_path,
                json.dumps(metadata),
                step,
                wt,
            )
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind=kind,
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"metadata": metadata, "rel_path": rel_path},
            )

        return self._best_effort_emit(action, _op)

    def add_graph(
        self,
        model: Any,
        input_to_model: Any = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> Any:
        """Capture a model's computation graph as a DOT artifact.

        If ``input_to_model`` is provided, a forward pass is traced to
        record per-layer shapes and a sibling PNG diagram is rendered next
        to the DOT file for inline display in the Models tab. Tracing
        failures fall back to a static walk over ``model.named_modules()``.

        Args:
            model: A ``torch.nn.Module``, raw DOT string, or list of DOT
                strings to concatenate.
            input_to_model: Example input(s) to trace. ``None`` skips the
                trace and renders a placeholder diagram.
            verbose: Stored on the artifact metadata; reserved for future
                use.
            use_strict_trace: Stored on the artifact metadata; reserved.
        """
        # Render the diagram OUTSIDE the DB op so a render failure can't
        # block the DOT artifact from being written. The PNG, when produced,
        # is stored as a sibling file in the same artifact folder so the
        # "Models" tab can show it inline without polluting the Images tab.
        rendered_png: Optional[bytes] = None
        if input_to_model is not None and not isinstance(model, str):
            try:
                from ._graph import (
                    capture_graph,
                    human_params,
                    render_graph_png,
                    static_graph,
                )

                try:
                    layers = capture_graph(model, input_to_model)
                except Exception:
                    # Forward pass failed (shape mismatch, missing kwargs,
                    # …). Fall back to a static walk so the structure is
                    # still rendered, just without shapes.
                    layers = static_graph(model)

                total = 0
                try:
                    total = sum(int(p.numel()) for p in model.parameters())
                except Exception:
                    pass
                in_shape = _shape_of(input_to_model)
                header = f"{type(model).__name__}  •  " f"{human_params(total)} params"
                if in_shape is not None:
                    header += f"  •  input {in_shape}"
                arr = render_graph_png(layers, header)
                rendered_png = _hwc_to_png_bytes(arr)
            except Exception as exc:
                self._handle_runtime_error("render graph diagram", exc)

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step("graph", None)
            wt = time.time()
            self._resolve_experiment(step)
            from .media import save_artifact

            dot = model if isinstance(model, str) and _looks_like_dot(model) else None
            if dot is None and isinstance(model, (list, tuple)):
                dot = "\n\n".join(
                    str(item) for item in model if _looks_like_dot(str(item))
                )
            if dot is None:
                dot = _default_dot_graph(model, input_to_model)

            rel_path, metadata = save_artifact(
                dot.encode("utf-8"),
                self.log_dir,
                "graph",
                step,
                filename=f"{step}.dot",
            )
            # Don't stuff the entire DOT source into "model" — that gets
            # rendered as the model class label in the UI. For raw DOT or
            # list-of-DOT inputs there is no meaningful class name.
            if isinstance(model, str) or (
                isinstance(model, (list, tuple))
                and all(isinstance(it, str) for it in model)
            ):
                model_label: Optional[str] = None
            else:
                model_label = type(model).__name__
            metadata.update(
                {
                    "kind": "graph",
                    "format": "dot",
                    "model": model_label,
                    "input_shape": _shape_of(input_to_model),
                    "verbose": bool(verbose),
                    "use_strict_trace": bool(use_strict_trace),
                }
            )

            # Store the rendered diagram as a sibling PNG inside the same
            # artifact folder; embed its relative path in the metadata so
            # the Models tab and chat adapters can find it without going
            # through the images table.
            if rendered_png is not None:
                png_rel, png_meta = save_artifact(
                    rendered_png,
                    self.log_dir,
                    "graph",
                    step,
                    filename=f"{step}.png",
                )
                metadata["rendered_png_path"] = png_rel
                metadata["rendered_png_size"] = png_meta.get("file_size")
                metadata["rendered_png_mime"] = png_meta.get("mime_type", "image/png")

            self._db.add_artifact(
                self._exp_id,
                "graph",
                rel_path,
                json.dumps(metadata),
                step,
                wt,
            )
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="graph",
                tag="graph",
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"metadata": metadata, "rel_path": rel_path},
            )

        return self._best_effort_emit("add graph", _op)

    def add_onnx_graph(self, prototxt: Any) -> Any:
        """Persist an ONNX prototxt blob as an artifact under
        ``"onnx_graph"``. ``prototxt`` may be ``bytes`` or a string.
        """
        if isinstance(prototxt, bytes):
            payload = prototxt
        else:
            payload = str(prototxt).encode("utf-8")
        return self._add_binary_artifact_event(
            "add onnx graph",
            "onnx_graph",
            "onnx_graph",
            payload,
            ext=".pbtxt",
            extra_meta={"format": "onnx"},
            global_step=0,
        )

    def add_embedding(
        self,
        mat: Any,
        metadata: Optional[List[Any]] = None,
        label_img: Any = None,
        global_step: Optional[int] = None,
        tag: str = "default",
        metadata_header: Optional[List[str]] = None,
    ) -> Any:
        """Store a high-dimensional embedding for the embedding viewer.

        Args:
            mat: ``(N, D)`` matrix of embedding vectors.
            metadata: Optional ``N``-length list of per-point labels (str
                or list of column values when ``metadata_header`` is set).
            label_img: Optional ``(N, C, H, W)`` / ``(N, H, W, C)`` batch
                of thumbnails. Tiled into a sprite atlas PNG so the JS
                viewer can render them as Three.js textures.
            global_step: Step number; ``None`` auto-increments per tag.
            tag: Embedding bucket name. Defaults to ``"default"``.
            metadata_header: Column names when ``metadata`` rows are lists.
        """
        # Render the sprite atlas (if any) OUTSIDE the DB op so a render
        # failure can't block the JSON artifact from being written.
        sprite_png: Optional[bytes] = None
        sprite_meta: Optional[Dict[str, Any]] = None
        if label_img is not None:
            try:
                sprite_png, sprite_meta = _render_label_img_atlas(label_img)
            except Exception as exc:
                self._handle_runtime_error("render embedding sprite atlas", exc)

        payload: Dict[str, Any] = {
            "mat": _jsonable(mat, include_array_values=True),
            "metadata": _jsonable(metadata, include_array_values=True),
            "label_img": _jsonable(label_img, include_array_values=False),
            "metadata_header": _jsonable(metadata_header, include_array_values=True),
        }
        if sprite_meta is not None:
            payload["sprite"] = sprite_meta

        def _op() -> Optional[LogEvent]:
            if self._db is None:
                return None
            step = self._resolve_step(tag, global_step)
            wt = time.time()
            self._resolve_experiment(step)
            from .media import save_artifact

            data = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
            rel_path, metadata_dict = save_artifact(
                data, self.log_dir, tag, step, filename=f"{step}.json"
            )
            metadata_dict.update({"kind": "embedding"})

            # Companion sprite atlas — written under the same tag folder so the
            # web viewer can locate it from the JSON path. We embed both the
            # relative path and the grid metadata so the JS can build the UV
            # math without re-parsing the JSON file.
            if sprite_png is not None and sprite_meta is not None:
                sprite_rel, sprite_file_meta = save_artifact(
                    sprite_png,
                    self.log_dir,
                    tag,
                    step,
                    filename=f"{step}.sprite.png",
                )
                metadata_dict["sprite_path"] = sprite_rel
                metadata_dict["sprite_size"] = sprite_file_meta.get("file_size")
                metadata_dict["sprite_mime"] = sprite_file_meta.get(
                    "mime_type", "image/png"
                )
                metadata_dict["sprite"] = sprite_meta

            self._db.add_artifact(
                self._exp_id,
                tag,
                rel_path,
                json.dumps(metadata_dict),
                step,
                wt,
            )
            abs_path = (
                str(Path(self.log_dir) / rel_path)
                if not os.path.isabs(rel_path)
                else rel_path
            )
            return LogEvent(
                kind="embedding",
                tag=tag,
                step=step,
                value=abs_path,
                walltime=wt,
                run_name=self._run_name,
                project=self._project,
                extra={"metadata": metadata_dict, "rel_path": rel_path},
            )

        return self._best_effort_emit("add embedding", _op)

    def add_pr_curve(
        self,
        tag: str,
        labels: Any,
        predictions: Any,
        global_step: Optional[int] = None,
        num_thresholds: int = 127,
        weights: Optional[Any] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        """Compute and persist a precision-recall curve.

        Args:
            tag: Curve series name.
            labels: Ground-truth booleans (or 0/1).
            predictions: Predicted scores in ``[0, 1]``.
            global_step: Step number; ``None`` auto-increments per tag.
            num_thresholds: Number of evenly-spaced thresholds in ``[0, 1]``.
            weights: Optional per-sample weights.
            walltime: Override timestamp.
        """
        payload = _pr_curve_payload(labels, predictions, num_thresholds, weights)
        return self._add_json_artifact_event(
            "add pr curve",
            "pr_curve",
            tag,
            payload,
            global_step=global_step,
            walltime=walltime,
        )

    def add_pr_curve_raw(
        self,
        tag: str,
        true_positive_counts: Any,
        false_positive_counts: Any,
        true_negative_counts: Any,
        false_negative_counts: Any,
        precision: Any,
        recall: Any,
        global_step: Optional[int] = None,
        num_thresholds: int = 127,
        weights: Optional[Any] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        """Persist a pre-computed PR curve.

        TensorBoard-compatible variant of :meth:`add_pr_curve` for callers
        that already maintain per-threshold counters (e.g. across workers).

        Args:
            tag: Curve series name.
            true_positive_counts: Per-threshold TP counts.
            false_positive_counts: Per-threshold FP counts.
            true_negative_counts: Per-threshold TN counts.
            false_negative_counts: Per-threshold FN counts.
            precision: Per-threshold precision values.
            recall: Per-threshold recall values.
            global_step: Step number; ``None`` auto-increments per tag.
            num_thresholds: Number of thresholds (stored on metadata).
            weights: Optional per-threshold weights stored alongside.
            walltime: Override timestamp.
        """
        payload = _pr_curve_raw_payload(
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
            num_thresholds,
            weights,
        )
        return self._add_json_artifact_event(
            "add pr curve raw",
            "pr_curve",
            tag,
            payload,
            global_step=global_step,
            walltime=walltime,
        )

    def add_custom_scalars(self, layout: Dict[str, Any]) -> Any:
        payload = {"layout": _jsonable(layout, include_array_values=True)}
        return self._add_json_artifact_event(
            "add custom scalars",
            "custom_scalars",
            "custom_scalars",
            payload,
            global_step=0,
        )

    def add_custom_scalars_marginchart(
        self,
        tags: Sequence[str],
        category: str = "default",
        title: str = "untitled",
    ) -> Any:
        if len(tags) != 3:
            self._handle_runtime_error(
                "add custom scalars marginchart",
                AssertionError(f"Expected 3 tags, got {len(tags)}."),
            )
            return null_handle()
        return self.add_custom_scalars({category: {title: ["Margin", list(tags)]}})

    def add_custom_scalars_multilinechart(
        self,
        tags: Sequence[str],
        category: str = "default",
        title: str = "untitled",
    ) -> Any:
        return self.add_custom_scalars({category: {title: ["Multiline", list(tags)]}})

    def add_mesh(
        self,
        tag: str,
        vertices: Any,
        colors: Any = None,
        faces: Any = None,
        config_dict: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        payload = {
            "vertices": _jsonable(vertices, include_array_values=True),
            "colors": _jsonable(colors, include_array_values=True),
            "faces": _jsonable(faces, include_array_values=True),
            "config": _jsonable(config_dict, include_array_values=True),
        }
        return self._add_json_artifact_event(
            "add mesh",
            "mesh",
            tag,
            payload,
            global_step=global_step,
            walltime=walltime,
        )

    def add_tensor(
        self,
        tag: str,
        tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> Any:
        payload = {"tensor": _jsonable(tensor, include_array_values=True)}
        return self._add_json_artifact_event(
            "add tensor",
            "tensor",
            tag,
            payload,
            global_step=global_step,
            walltime=walltime,
        )

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, Any],
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
        global_step: Optional[int] = None,
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
                    self.add_scalar(tag, val, global_step=global_step)
            return LogEvent(
                kind="hparams",
                tag="hparams",
                step=global_step,
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
        """Flush pending scalar and dispatcher buffers."""
        if not self._enabled or self._closed:
            return None
        try:
            with self._buffer_lock:
                self._flush_locked()
        except Exception as exc:
            self._handle_runtime_error("flush", exc)
        try:
            self._flush_dispatchers(final=True)
        except Exception as exc:
            self._handle_runtime_error("flush dispatchers", exc)
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
            self._send_summaries()
        except Exception as exc:
            self._handle_runtime_error("dispatch summary", exc)
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

    def get_logdir(self) -> str:
        """Return the resolved log directory, matching TensorBoard's helper."""
        return self.log_dir

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
