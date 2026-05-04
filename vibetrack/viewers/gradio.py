"""Gradio-based live dashboard.

Requires: ``pip install vibetrack[gradio]``
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from bisect import bisect_right
from collections import OrderedDict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import load_config
from .base import BaseOutput
from .web import _serialize_experiment

_log = logging.getLogger(__name__)

_SYSTEM_PREFIXES = ("system/", "gpu/")
_DEFAULT_IMAGE_MAX_PX = 1024


def _system_unit(tag: str) -> Tuple[str, int]:
    """Return ``(y_axis_label, decimals)`` for a system metric tag.

    Mirrors the formatting used by the web viewer's ``formatVal`` so plots
    like ``system/memory_total_gb`` show ``24.0 GB`` rather than ``24.0000``.
    """
    name = tag.lower().rsplit("/", 1)[-1]
    if name.endswith("_gb"):
        return "GB", 2
    if name.endswith("_mb"):
        return "MB", 1
    if name.endswith("_percent"):
        return "%", 1
    if name.endswith("_c") or name.endswith("_celsius"):
        return "°C", 1
    if name.endswith("_w") or name.endswith("_watts"):
        return "W", 1
    if "load" in name:
        return "load", 2
    return "value", 3


class GradioOutput(BaseOutput):
    """Interactive Gradio dashboard for live and persisted run data."""

    _live_buffer: List[Any] = []
    _live_buffer_max: int = 2048
    _live_lock = threading.RLock()
    _launch_lock = threading.RLock()
    _thumb_lock = threading.RLock()
    _live_keys: set = set()
    _launch_threads: Dict[Tuple[Optional[str], Optional[str]], threading.Thread] = {}
    _thumb_cache: "OrderedDict[Tuple[str, int, int, int], str]" = OrderedDict()
    _thumb_cache_max: int = 256

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        **launch_kwargs: Any,
    ) -> None:
        super().__init__(project_folder=project_folder, project=project)
        self._launch_kwargs = dict(launch_kwargs)

    def _live_key(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.project_folder, self.project)

    def send(self, events: Sequence[Any]) -> None:
        """Buffer live events and ensure a dashboard is running."""
        with GradioOutput._live_lock:
            GradioOutput._live_buffer.extend(events)
            overflow = len(GradioOutput._live_buffer) - GradioOutput._live_buffer_max
            if overflow > 0:
                del GradioOutput._live_buffer[:overflow]
        self.start()

    def start(self, **kwargs: Any) -> None:
        """Launch the Gradio dashboard without blocking the training process."""
        key = self._live_key()
        with GradioOutput._launch_lock:
            if key in GradioOutput._live_keys:
                return
            thread = GradioOutput._launch_threads.get(key)
            if thread is not None and thread.is_alive():
                return
            launch_kwargs = dict(self._launch_kwargs)
            launch_kwargs.update(kwargs)
            launch_kwargs.setdefault("prevent_thread_lock", True)
            launch_kwargs.setdefault("quiet", False)
            thread = threading.Thread(
                target=self._start_worker,
                args=(key, launch_kwargs),
                name="vibetrack-gradio",
                daemon=True,
            )
            GradioOutput._launch_threads[key] = thread
            thread.start()

    def _start_worker(
        self,
        key: Tuple[Optional[str], Optional[str]],
        launch_kwargs: Dict[str, Any],
    ) -> None:
        try:
            self.show(**launch_kwargs)
        except Exception as exc:  # pragma: no cover - defensive background path
            msg = f"vibetrack gradio: dashboard failed to start: {exc}"
            _log.error(msg)
            print(msg, file=sys.stderr, flush=True)
            with GradioOutput._launch_lock:
                GradioOutput._live_keys.discard(key)
                GradioOutput._launch_threads.pop(key, None)

    def _resolve_experiments(
        self,
        names: Optional[Sequence[str]] = None,
        project: Optional[str] = None,
    ):
        # Pass `project` directly to the reader instead of mutating
        # ``self._reader.project`` — concurrent dropdown changes / timer
        # ticks must not see each other's transient state.
        if project is not None:
            # Explicit dropdown selection; "" means "all projects".
            return self._filter_by_names(
                self._reader.experiments(project=project or None), names
            )
        exps = self._filter_by_names(self._reader.experiments(), names)
        if exps:
            return exps
        # Fallback: cwd-derived project filter found nothing. Show all
        # projects so the dashboard is useful regardless of where it was
        # launched from.
        if self.project is None and self._reader.project is not None:
            return self._filter_by_names(self._reader.experiments(project=None), names)
        return exps

    @staticmethod
    def _filter_by_names(exps, names):
        if names is None:
            return list(exps)
        name_set = set(names)
        return [e for e in exps if e.name in name_set]

    def _list_projects(self) -> List[str]:
        self._reader._discover()
        db = self._reader._db
        if db is None:
            return []
        try:
            return [p for p in db.list_projects() if p]
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("failed to list Gradio projects: %s", exc)
            return []

    def _resolve_live_events(
        self,
        experiments: Optional[Sequence[str]] = None,
        project: Optional[str] = None,
    ) -> List[Any]:
        names = set(experiments) if experiments is not None else None
        # Dropdown wins over the constructor's project; "" means "all".
        if project is not None:
            project_filter = project or None
        else:
            project_filter = self.project
        with GradioOutput._live_lock:
            events = list(GradioOutput._live_buffer)
        out = []
        for event in events:
            if (
                project_filter is not None
                and getattr(event, "project", None) != project_filter
            ):
                continue
            if names is not None and getattr(event, "run_name", None) not in names:
                continue
            out.append(event)
        return out

    @staticmethod
    def _empty_run(event: Any) -> Dict[str, Any]:
        return {
            "name": getattr(event, "run_name", "run"),
            "project": getattr(event, "project", None) or "",
            "log_dir": "",
            "tags": [],
            "scalars": {},
            "system_tags": [],
            "system_scalars": {},
            "image_tags": [],
            "images": {},
            "audio_tags": [],
            "audio_data": {},
            "video_tags": [],
            "video_data": {},
            "artifact_tags": [],
            "artifacts": {},
            "model_tags": [],
            "models": {},
            "graph_tags": [],
            "graphs": {},
            "pr_curve_tags": [],
            "pr_curves": {},
            "figure_tags": [],
            "figures": {},
            "mesh_tags": [],
            "meshes": {},
            "embedding_tags": [],
            "embeddings": {},
            "text_tags": [],
            "text_data": {},
            "histogram_tags": [],
            "histogram_data": {},
            "hparams": {},
        }

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        if value not in items:
            items.append(value)

    @staticmethod
    def _append_row_once(
        rows: List[Dict[str, Any]],
        row: Dict[str, Any],
        keys: Sequence[str],
    ) -> None:
        identity = tuple(row.get(key) for key in keys)
        for existing in rows:
            if tuple(existing.get(key) for key in keys) == identity:
                return
        rows.append(row)

    def _merge_live_events(
        self, data: List[Dict[str, Any]], events: Sequence[Any]
    ) -> None:
        by_name = {run["name"]: run for run in data}
        for event in events:
            run = by_name.get(event.run_name)
            if run is None:
                run = self._empty_run(event)
                by_name[event.run_name] = run
                data.append(run)

            tag = str(event.tag)
            step = event.step
            if event.kind == "scalar":
                scalar_key = (
                    "system_scalars" if tag.startswith(_SYSTEM_PREFIXES) else "scalars"
                )
                tag_key = "system_tags" if tag.startswith(_SYSTEM_PREFIXES) else "tags"
                self._append_unique(run[tag_key], tag)
                series = run[scalar_key].setdefault(
                    tag,
                    {"steps": [], "values": [], "wall_times": []},
                )
                if step in series["steps"]:
                    idx = series["steps"].index(step)
                    series["values"][idx] = event.value
                    series["wall_times"][idx] = event.walltime
                    continue
                series["steps"].append(step)
                series["values"].append(event.value)
                series["wall_times"].append(event.walltime)
            elif event.kind == "image":
                self._append_unique(run["image_tags"], tag)
                self._append_row_once(
                    run["images"].setdefault(tag, []),
                    {"step": step, "path": event.value},
                    ("step", "path"),
                )
            elif event.kind == "audio":
                self._append_unique(run["audio_tags"], tag)
                self._append_row_once(
                    run["audio_data"].setdefault(tag, []),
                    {"step": step, "path": event.value},
                    ("step", "path"),
                )
            elif event.kind == "video":
                self._append_unique(run["video_tags"], tag)
                self._append_row_once(
                    run["video_data"].setdefault(tag, []),
                    {"step": step, "path": event.value},
                    ("step", "path"),
                )
            elif event.kind == "text":
                self._append_unique(run["text_tags"], tag)
                self._append_row_once(
                    run["text_data"].setdefault(tag, []),
                    {"step": step, "value": event.value},
                    ("step", "value"),
                )
            elif event.kind == "histogram":
                self._append_unique(run["histogram_tags"], tag)
                value = event.value if isinstance(event.value, dict) else {}
                self._append_row_once(
                    run["histogram_data"].setdefault(tag, []),
                    {
                        "step": step,
                        "bins": value.get("bins") or value.get("bin_edges") or [],
                        "counts": value.get("counts") or [],
                    },
                    ("step",),
                )
            elif event.kind in {
                "artifact",
                "figure",
                "graph",
                "pr_curve",
                "mesh",
                "embedding",
            }:
                meta = dict((event.extra or {}).get("metadata") or {})
                row = {"step": step, "path": event.value, "metadata": meta}
                if event.kind == "figure":
                    self._append_unique(run["figure_tags"], tag)
                    self._append_row_once(
                        run["figures"].setdefault(tag, []), row, ("step", "path")
                    )
                elif event.kind == "graph":
                    self._append_unique(run["model_tags"], tag)
                    self._append_unique(run["graph_tags"], tag)
                    self._append_row_once(
                        run["models"].setdefault(tag, []), row, ("step", "path")
                    )
                    self._append_row_once(
                        run["graphs"].setdefault(tag, []), row, ("step", "path")
                    )
                elif event.kind == "pr_curve":
                    self._append_unique(run["pr_curve_tags"], tag)
                    self._append_row_once(
                        run["pr_curves"].setdefault(tag, []), row, ("step", "path")
                    )
                elif event.kind == "mesh":
                    self._append_unique(run["mesh_tags"], tag)
                    self._append_row_once(
                        run["meshes"].setdefault(tag, []), row, ("step", "path")
                    )
                elif event.kind == "embedding":
                    self._append_unique(run["embedding_tags"], tag)
                    self._append_row_once(
                        run["embeddings"].setdefault(tag, []), row, ("step", "path")
                    )
                else:
                    self._append_unique(run["artifact_tags"], tag)
                    self._append_row_once(
                        run["artifacts"].setdefault(tag, []), row, ("step", "path")
                    )
            elif event.kind == "hparams" and isinstance(event.value, dict):
                run["hparams"].update(event.value)

    def _snapshot(
        self,
        experiments: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            exps = self._resolve_experiments(experiments, project=project)
            data = [_serialize_experiment(exp) for exp in exps]
        except Exception as exc:
            _log.warning("failed to read Gradio dashboard data: %s", exc)
            data = []
        self._merge_live_events(
            data, self._resolve_live_events(experiments, project=project)
        )
        if tags is not None:
            keep = set(tags)
            for run in data:
                run["tags"] = [tag for tag in run.get("tags", []) if tag in keep]
                run["scalars"] = {
                    tag: series
                    for tag, series in run.get("scalars", {}).items()
                    if tag in keep
                }
        return data

    @staticmethod
    def _channels(data: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
        return {
            "scalars": any(
                run.get("tags") or run.get("figure_tags") or run.get("pr_curve_tags")
                for run in data
            ),
            "images": any(run.get("image_tags") for run in data),
            "audio": any(run.get("audio_tags") for run in data),
            "video": any(run.get("video_tags") for run in data),
            "text": any(run.get("text_tags") for run in data),
            "histograms": any(run.get("histogram_tags") for run in data),
            "system": any(run.get("system_tags") for run in data),
            "hparams": any(run.get("hparams") for run in data),
        }

    @staticmethod
    def _snapshot_signature(
        selected_project: str,
        data: Sequence[Dict[str, Any]],
    ) -> str:
        parts: List[Any] = [selected_project or ""]
        for run in sorted(
            data,
            key=lambda item: (
                str(item.get("project") or ""),
                str(item.get("name") or ""),
            ),
        ):
            parts.append(("run", run.get("project"), run.get("name")))
            for tag in sorted(run.get("tags") or []):
                series = (run.get("scalars") or {}).get(tag) or {}
                steps = series.get("steps") or []
                values = series.get("values") or []
                parts.append(
                    (
                        "scalar",
                        tag,
                        len(steps),
                        steps[-1] if steps else None,
                        values[-1] if values else None,
                    )
                )
            for tag in sorted(run.get("system_tags") or []):
                series = (run.get("system_scalars") or {}).get(tag) or {}
                steps = series.get("steps") or []
                values = series.get("values") or []
                parts.append(
                    (
                        "system",
                        tag,
                        len(steps),
                        steps[-1] if steps else None,
                        values[-1] if values else None,
                    )
                )
            for data_key, tag_key in [
                ("images", "image_tags"),
                ("audio_data", "audio_tags"),
                ("video_data", "video_tags"),
                ("artifacts", "artifact_tags"),
                ("figures", "figure_tags"),
                ("models", "model_tags"),
                ("pr_curves", "pr_curve_tags"),
                ("meshes", "mesh_tags"),
                ("embeddings", "embedding_tags"),
                ("text_data", "text_tags"),
                ("histogram_data", "histogram_tags"),
            ]:
                for tag in sorted(run.get(tag_key) or []):
                    rows = (run.get(data_key) or {}).get(tag) or []
                    last = rows[-1] if rows else {}
                    parts.append(
                        (
                            data_key,
                            tag,
                            len(rows),
                            last.get("step"),
                            last.get("path") or last.get("value"),
                        )
                    )
            parts.append(
                (
                    "hparams",
                    sorted(
                        (str(key), repr(value))
                        for key, value in (run.get("hparams") or {}).items()
                    ),
                )
            )
        return repr(parts)

    @staticmethod
    def _all_tags(data: Sequence[Dict[str, Any]], key: str) -> List[str]:
        tags = set()
        for run in data:
            tags.update(run.get(key, []) or [])
        return sorted(tags)

    @staticmethod
    def _dedupe_series(
        steps: Sequence[Any], values: Sequence[Any]
    ) -> Tuple[List[int], List[float]]:
        merged: Dict[int, float] = {}
        for step, value in zip(steps, values):
            if step is None:
                continue
            try:
                merged[int(step)] = float(value)
            except (TypeError, ValueError):
                continue
        points = sorted(merged.items())
        return [p[0] for p in points], [p[1] for p in points]

    @staticmethod
    def _empty_frame(columns: Sequence[str]) -> Any:
        import pandas as pd  # type: ignore[import-untyped]

        return pd.DataFrame(columns=list(columns))

    def _scalar_frame(
        self,
        data: Sequence[Dict[str, Any]],
        tag: str,
        data_key: str = "scalars",
        round_decimals: Optional[int] = None,
    ) -> Any:
        import pandas as pd  # type: ignore[import-untyped]

        rows: List[Dict[str, Any]] = []
        for run in data:
            series = (run.get(data_key) or {}).get(tag)
            if not series:
                continue
            steps, values = self._dedupe_series(
                series.get("steps", []),
                series.get("values", []),
            )
            if not steps:
                continue
            if round_decimals is not None:
                values = [round(v, round_decimals) for v in values]
            rows.extend(
                {"step": step, "value": value, "series": run["name"]}
                for step, value in zip(steps, values)
            )
        if not rows:
            return self._empty_frame(["step", "value", "series"])
        return pd.DataFrame(rows)

    @staticmethod
    def _histogram_frame(data: Sequence[Dict[str, Any]], tag: str) -> Any:
        import pandas as pd  # type: ignore[import-untyped]

        plot_rows: List[Dict[str, Any]] = []
        for run in data:
            rows = (run.get("histogram_data") or {}).get(tag) or []
            if not rows:
                continue
            latest = sorted(rows, key=lambda row: row.get("step") or 0)[-1]
            bins = latest.get("bins") or []
            counts = latest.get("counts") or []
            x = bins[:-1] if len(bins) == len(counts) + 1 else bins
            plot_rows.extend(
                {
                    "bin": bin_value,
                    "count": count,
                    "series": f"{run['name']} step {latest.get('step')}",
                }
                for bin_value, count in zip(x, counts)
            )
        if not plot_rows:
            return GradioOutput._empty_frame(["bin", "count", "series"])
        return pd.DataFrame(plot_rows)

    @staticmethod
    def _gallery_items(
        data: Sequence[Dict[str, Any]],
        data_key: str,
        tag: str,
        path_key: str = "path",
    ) -> List[Any]:
        items = []
        for run in data:
            for row in (run.get(data_key) or {}).get(tag, []) or []:
                path = row.get(path_key) or row.get("path")
                if path:
                    items.append((path, f"{run['name']} step {row.get('step')}"))
        return items[-120:]

    @staticmethod
    def _table_rows(
        data: Sequence[Dict[str, Any]],
        data_key: str,
        tag_key: str,
    ) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for run in data:
            for tag in run.get(tag_key, []) or []:
                for row in (run.get(data_key) or {}).get(tag, []) or []:
                    rows.append(
                        [
                            run.get("name"),
                            tag,
                            row.get("step"),
                            row.get("path"),
                            row.get("metadata") or {},
                        ]
                    )
        return rows

    @staticmethod
    def _hparam_rows(data: Sequence[Dict[str, Any]]) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for run in data:
            for key, value in (run.get("hparams") or {}).items():
                rows.append([run.get("name"), key, value])
        return rows

    @staticmethod
    def _hparam_tree(data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for run in data:
            hparams = run.get("hparams") or {}
            if not hparams:
                continue
            tree: Dict[str, Any] = {}
            for key, value in hparams.items():
                parts = str(key).replace(".", "/").strip("/").split("/")
                node = tree
                for part in parts[:-1]:
                    sub = node.get(part)
                    if not isinstance(sub, dict):
                        sub = {}
                        node[part] = sub
                    node = sub
                node[parts[-1]] = value
            out[str(run.get("name") or "run")] = tree
        return out

    @staticmethod
    def _text_blocks(data: Sequence[Dict[str, Any]], tag: str) -> str:
        rows: List[Tuple[Any, str]] = []
        for run in data:
            for row in (run.get("text_data") or {}).get(tag, []) or []:
                rows.append(
                    (
                        row.get("step"),
                        f"[{run['name']} step {row.get('step')}]\n{row.get('value', '')}",
                    )
                )
        rows.sort(key=lambda item: -1 if item[0] is None else int(item[0]))
        return "\n\n".join(text for _step, text in rows[-40:])

    @staticmethod
    def _latest_media_rows(
        data: Sequence[Dict[str, Any]],
        data_key: str,
        tag: str,
        limit: int = 8,
    ) -> List[Tuple[str, Any, str]]:
        rows: List[Tuple[str, Any, str]] = []
        for run in data:
            for row in (run.get(data_key) or {}).get(tag, []) or []:
                path = row.get("path")
                if path:
                    rows.append((run["name"], row.get("step"), path))
        rows.sort(key=lambda item: -1 if item[1] is None else int(item[1]))
        return rows[-limit:]

    @classmethod
    def _display_image_path(cls, path: str, max_px: Optional[int]) -> str:
        if not max_px or max_px <= 0:
            return path

        try:
            source = Path(path)
            stat = source.stat()
            resolved = str(source.resolve())
        except OSError:
            return path

        cache_key = (resolved, int(stat.st_mtime_ns), int(stat.st_size), int(max_px))
        with cls._thumb_lock:
            cached = cls._thumb_cache.get(cache_key)
            if cached and os.path.exists(cached):
                cls._thumb_cache.move_to_end(cache_key)
                return cached

        try:
            from PIL import Image as PILImage
            from PIL import ImageOps

            with PILImage.open(source) as img:
                if getattr(img, "is_animated", False):
                    return path
                img = ImageOps.exif_transpose(img)
                if max(img.size) <= max_px:
                    return path
                img.thumbnail((max_px, max_px))
                if img.mode not in {"RGB", "RGBA"}:
                    img = img.convert("RGB")

                digest = sha1(
                    f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}:{max_px}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
                target = (
                    Path(tempfile.gettempdir()) / "vibetrack-gradio" / f"{digest}.webp"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    img.save(target, "WEBP", quality=85, method=4)
        except Exception as exc:  # pragma: no cover - defensive cache path
            _log.debug("failed to create Gradio image thumbnail for %s: %s", path, exc)
            return path

        out = str(target)
        with cls._thumb_lock:
            cls._thumb_cache[cache_key] = out
            cls._thumb_cache.move_to_end(cache_key)
            while len(cls._thumb_cache) > cls._thumb_cache_max:
                _, evicted = cls._thumb_cache.popitem(last=False)
                try:
                    os.unlink(evicted)
                except OSError:
                    pass
        return out

    @staticmethod
    def _image_index(
        entries: Sequence[Tuple[str, int, str]],
    ) -> Dict[str, Tuple[List[int], List[str]]]:
        per_run: Dict[str, Dict[int, str]] = {}
        for run, step, path in entries:
            per_run.setdefault(run, {})[step] = path
        index = {}
        for run, step_map in per_run.items():
            rows = sorted(step_map.items())
            index[run] = (
                [step for step, _path in rows],
                [path for _step, path in rows],
            )
        return index

    @classmethod
    def _pick_indexed_image(
        cls,
        index: Dict[str, Tuple[List[int], List[str]]],
        run: str,
        target: int,
        max_px: Optional[int],
    ) -> Optional[str]:
        row = index.get(run)
        if row is None:
            return None
        steps, paths = row
        if not steps:
            return None
        pos = bisect_right(steps, target) - 1
        if pos < 0:
            pos = 0
        return cls._display_image_path(paths[pos], max_px)

    def _render_scalar_tab(
        self,
        gr: Any,
        data: Sequence[Dict[str, Any]],
    ) -> None:
        scalar_tags = self._all_tags(data, "tags")
        for tag in scalar_tags:
            gr.LinePlot(
                value=self._scalar_frame(data, tag, "scalars"),
                x="step",
                y="value",
                color="series",
                title=tag,
                x_title="step",
                y_title="value",
                label=tag,
                key=("scalar", tag),
            )

        figure_tags = self._all_tags(data, "figure_tags")
        for tag in figure_tags:
            items = self._gallery_items(data, "figures", tag)
            if items:
                gr.Gallery(
                    value=items,
                    label=f"Figure: {tag}",
                    columns=3,
                    key=("figure", tag),
                )

        pr_tags = self._all_tags(data, "pr_curve_tags")
        if pr_tags:
            gr.Dataframe(
                value=self._table_rows(data, "pr_curves", "pr_curve_tags"),
                headers=["run", "tag", "step", "path", "metadata"],
                label="PR curves",
                interactive=False,
            )

    def _render_image_tab(
        self,
        gr: Any,
        data: Sequence[Dict[str, Any]],
        image_max_px: Optional[int],
    ) -> None:
        for tag in self._all_tags(data, "image_tags"):
            entries: List[Tuple[str, int, str]] = []
            for run in data:
                for row in (run.get("images") or {}).get(tag, []) or []:
                    path = row.get("path")
                    step = row.get("step")
                    if path is None or step is None:
                        continue
                    try:
                        entries.append((run["name"], int(step), path))
                    except (TypeError, ValueError):
                        continue
            if not entries:
                continue
            steps = sorted({s for _, s, _ in entries})
            by_run = self._image_index(entries)
            runs = sorted(by_run)
            latest = steps[-1]
            with gr.Accordion(tag, open=True):
                slider = gr.Slider(
                    minimum=steps[0],
                    maximum=steps[-1],
                    value=latest,
                    step=1,
                    label="step",
                    key=("image-step", tag),
                )
                outputs: List[Any] = []
                for run in runs:
                    outputs.append(
                        gr.Image(
                            value=self._pick_indexed_image(
                                by_run,
                                run,
                                latest,
                                image_max_px,
                            ),
                            label=run,
                            type="filepath",
                            interactive=False,
                            height=320,
                            key=("image", tag, run),
                        )
                    )

                def _on_slide(
                    step_v: float,
                    _runs: List[str] = runs,
                    _by_run: Dict[str, Tuple[List[int], List[str]]] = by_run,
                    _image_max_px: Optional[int] = image_max_px,
                ) -> List[Any]:
                    target = int(step_v)
                    return [
                        self._pick_indexed_image(
                            _by_run,
                            r,
                            target,
                            _image_max_px,
                        )
                        for r in _runs
                    ]

                step_event = (
                    slider.release if hasattr(slider, "release") else slider.change
                )
                step_event(
                    _on_slide,
                    inputs=[slider],
                    outputs=outputs,
                    show_progress="hidden",
                    queue=False,
                    trigger_mode="always_last",
                )

    @staticmethod
    def _pick_image(
        by_key: Dict[Tuple[str, int], str],
        run: str,
        steps: Sequence[int],
        target: int,
    ) -> Optional[str]:
        best: Optional[str] = None
        for s in steps:
            if s <= target and (run, s) in by_key:
                best = by_key[(run, s)]
        if best is None:
            for s in steps:
                if (run, s) in by_key:
                    return by_key[(run, s)]
        return best

    def _render_media_tab(
        self,
        gr: Any,
        data: Sequence[Dict[str, Any]],
        kind: str,
        image_max_px: Optional[int] = None,
    ) -> None:
        if kind == "images":
            self._render_image_tab(gr, data, image_max_px)
            return

        data_key = "audio_data" if kind == "audio" else "video_data"
        tag_key = "audio_tags" if kind == "audio" else "video_tags"
        component = gr.Audio if kind == "audio" else gr.Video
        for tag in self._all_tags(data, tag_key):
            with gr.Accordion(tag, open=True, key=(kind, tag)):
                for run_name, step, path in self._latest_media_rows(
                    data, data_key, tag
                ):
                    component(value=path, label=f"{run_name} step {step}")

    def _render_gradio_body(
        self,
        gr: Any,
        data: Sequence[Dict[str, Any]],
        image_max_px: Optional[int] = None,
    ) -> None:
        channels = self._channels(data)
        if not any(channels.values()):
            gr.Markdown("No run data yet.")
            return

        with gr.Tabs():
            if channels["scalars"]:
                with gr.Tab("Scalars", id="scalars"):
                    self._render_scalar_tab(gr, data)
            if channels["images"]:
                with gr.Tab("Images", id="images"):
                    self._render_media_tab(
                        gr,
                        data,
                        "images",
                        image_max_px=image_max_px,
                    )
            if channels["audio"]:
                with gr.Tab("Audio", id="audio"):
                    self._render_media_tab(gr, data, "audio")
            if channels["video"]:
                with gr.Tab("Video", id="video"):
                    self._render_media_tab(gr, data, "video")
            if channels["text"]:
                with gr.Tab("Text", id="text"):
                    for tag in self._all_tags(data, "text_tags"):
                        gr.Textbox(
                            value=self._text_blocks(data, tag),
                            label=tag,
                            lines=12,
                            interactive=False,
                            key=("text", tag),
                        )
            if channels["histograms"]:
                with gr.Tab("Histograms", id="histograms"):
                    for tag in self._all_tags(data, "histogram_tags"):
                        gr.BarPlot(
                            value=self._histogram_frame(data, tag),
                            x="bin",
                            y="count",
                            color="series",
                            title=tag,
                            x_title="bin",
                            y_title="count",
                            label=tag,
                        )
            if channels["system"]:
                with gr.Tab("System", id="system"):
                    for tag in self._all_tags(data, "system_tags"):
                        unit, decimals = _system_unit(tag)
                        gr.LinePlot(
                            value=self._scalar_frame(
                                data,
                                tag,
                                "system_scalars",
                                round_decimals=decimals,
                            ),
                            x="step",
                            y="value",
                            color="series",
                            title=tag,
                            x_title="step",
                            y_title=unit,
                            label=tag,
                            key=("system", tag),
                        )
            if channels["hparams"]:
                with gr.Tab("HParams", id="hparams"):
                    gr.JSON(
                        value=self._hparam_tree(data),
                        label="HParams",
                        open=True,
                        key="hparams-tree",
                    )

    def show(self, **kwargs: Any) -> Any:
        launch_kwargs = dict(self._launch_kwargs)
        launch_kwargs.update(kwargs)
        tags: Optional[Sequence[str]] = launch_kwargs.pop("tags", None)
        experiments: Optional[Sequence[str]] = launch_kwargs.pop("experiments", None)

        cfg = load_config(self.config_project())
        gradio_cfg = cfg.get("gradio", {})
        share = bool(launch_kwargs.pop("share", gradio_cfg.get("share", True)))
        host = launch_kwargs.pop(
            "host",
            launch_kwargs.pop("server_name", gradio_cfg.get("host", "127.0.0.1")),
        )
        port = launch_kwargs.pop(
            "port",
            launch_kwargs.pop("server_port", gradio_cfg.get("port")),
        )
        refresh_raw = launch_kwargs.pop(
            "refresh_interval",
            gradio_cfg.get(
                "refresh_interval", cfg.get("web", {}).get("auto_refresh", 5)
            ),
        )
        try:
            refresh_interval = float(refresh_raw)
        except (TypeError, ValueError):
            refresh_interval = 0.0
        image_max_px_raw = launch_kwargs.pop(
            "image_max_px",
            gradio_cfg.get("image_max_px", _DEFAULT_IMAGE_MAX_PX),
        )
        try:
            image_max_px = int(image_max_px_raw)
        except (TypeError, ValueError):
            image_max_px = _DEFAULT_IMAGE_MAX_PX
        prevent_thread_lock = bool(launch_kwargs.pop("prevent_thread_lock", False))
        quiet = bool(launch_kwargs.pop("quiet", False))
        inbrowser = bool(
            launch_kwargs.pop("inbrowser", gradio_cfg.get("inbrowser", False))
        )
        launch_kwargs.pop("token", None)
        launch_kwargs.pop("mcp_transport", None)

        import gradio as gr

        projects = self._list_projects()
        # Default selection: explicit project arg → cwd-derived project (if it
        # has data) → first project in the list → "All projects" sentinel ("").
        if self.project and self.project in projects:
            default_project = self.project
        elif self._reader.project and self._reader.project in projects:
            default_project = self._reader.project
        elif projects:
            default_project = projects[0]
        else:
            default_project = ""
        choices: List[Tuple[str, str]] = [("All projects", "")] + [
            (p, p) for p in projects
        ]

        def _snap(selected: str) -> List[Dict[str, Any]]:
            return self._snapshot(experiments, tags, project=selected)

        def _snap_if_changed(
            selected: str,
            previous_signature: str,
        ) -> Tuple[Any, str]:
            data = _snap(selected)
            signature = self._snapshot_signature(selected, data)
            if signature == previous_signature:
                return gr.skip(), previous_signature
            return data, signature

        initial_data = _snap(default_project)
        initial_signature = self._snapshot_signature(default_project, initial_data)

        with gr.Blocks(title="vibetrack") as demo:
            gr.Markdown("## vibetrack")
            project_dd = gr.Dropdown(
                choices=choices,
                value=default_project,
                label="Project",
                interactive=True,
                visible=bool(projects),
            )
            state = gr.State(value=initial_data)
            signature_state = gr.State(value=initial_signature)
            timer = gr.Timer(
                value=max(refresh_interval, 0.1), active=refresh_interval > 0
            )

            demo.load(
                fn=_snap_if_changed,
                inputs=[project_dd, signature_state],
                outputs=[state, signature_state],
                show_progress="hidden",
                queue=False,
            )
            timer.tick(
                fn=_snap_if_changed,
                inputs=[project_dd, signature_state],
                outputs=[state, signature_state],
                show_progress="hidden",
                queue=False,
                trigger_mode="always_last",
            )
            project_dd.change(
                fn=_snap_if_changed,
                inputs=[project_dd, signature_state],
                outputs=[state, signature_state],
                show_progress="hidden",
                queue=False,
                trigger_mode="always_last",
            )

            @gr.render(inputs=[state], show_progress="hidden")
            def _render(data: List[Dict[str, Any]]) -> None:
                self._render_gradio_body(
                    gr,
                    data or [],
                    image_max_px=image_max_px,
                )

        launch_args: Dict[str, Any] = {
            "share": share,
            "theme": gr.themes.Soft(primary_hue="blue"),
            "prevent_thread_lock": prevent_thread_lock,
            "quiet": quiet,
            "inbrowser": inbrowser,
        }
        if host:
            launch_args["server_name"] = host
        if port is not None:
            launch_args["server_port"] = int(port)
        launch_args.update(launch_kwargs)

        with GradioOutput._launch_lock:
            GradioOutput._live_keys.add(self._live_key())
        try:
            demo.launch(**launch_args)
        except Exception:
            with GradioOutput._launch_lock:
                GradioOutput._live_keys.discard(self._live_key())
            raise
        return demo
