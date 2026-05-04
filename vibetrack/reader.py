"""Read experiment data from a project's central vibetrack database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import Database, central_db_path


class ExperimentReader:
    """Read a single experiment's data."""

    def __init__(
        self,
        db: Database,
        experiment_id: int,
        name: str,
        row: Optional[Any] = None,
    ) -> None:
        self._db = db
        self.experiment_id = experiment_id
        self.name = name
        self._row: Optional[Dict[str, Any]] = dict(row) if row is not None else None

    def _experiment_row(self) -> Dict[str, Any]:
        if self._row is None:
            row = self._db.get_experiment(self.experiment_id)
            self._row = dict(row) if row is not None else {}
        return self._row

    @property
    def project(self) -> str:
        return str(self._experiment_row().get("project", ""))

    @property
    def log_dir(self) -> str:
        return str(self._experiment_row().get("log_dir", ""))

    def resolve_media_path(self, rel_path: str) -> str:
        """Resolve a stored media path to an absolute file path."""
        if not rel_path:
            return rel_path
        path = Path(rel_path)
        log_dir = self.log_dir
        if not log_dir:
            # No log dir to anchor against — preserve legacy behaviour.
            return str(path)
        log_dir_resolved = str(Path(log_dir).resolve())
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (Path(log_dir) / path).resolve()
        # Reject path traversal — even for stored absolute paths, since a
        # poisoned row could otherwise leak files outside the experiment dir.
        if (
            not str(resolved).startswith(log_dir_resolved + os.sep)
            and str(resolved) != log_dir_resolved
        ):
            return ""
        return str(resolved)

    def scalar_tags(self) -> List[str]:
        return self._db.get_scalar_tags(self.experiment_id)

    def scalars(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_scalars(self.experiment_id, tag)
        return [
            {"step": r["step"], "value": r["value"], "wall_time": r["wall_time"]}
            for r in rows
        ]

    def texts(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_texts(self.experiment_id, tag)
        return [
            {"step": r["step"], "value": r["value"], "wall_time": r["wall_time"]}
            for r in rows
        ]

    def text_tags(self) -> List[str]:
        return self._db.get_text_tags(self.experiment_id)

    def images(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_images(self.experiment_id, tag)
        return [
            {
                "step": r["step"],
                "path": r["path"],
                "abs_path": self.resolve_media_path(r["path"]),
                "wall_time": r["wall_time"],
            }
            for r in rows
        ]

    def image_tags(self) -> List[str]:
        return self._db.get_image_tags(self.experiment_id)

    def audio(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_audio(self.experiment_id, tag)
        return [
            {
                "step": r["step"],
                "path": r["path"],
                "abs_path": self.resolve_media_path(r["path"]),
                "sample_rate": r["sample_rate"],
                "wall_time": r["wall_time"],
            }
            for r in rows
        ]

    def audio_tags(self) -> List[str]:
        return self._db.get_audio_tags(self.experiment_id)

    def video(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_video(self.experiment_id, tag)
        return [
            {
                "step": r["step"],
                "path": r["path"],
                "abs_path": self.resolve_media_path(r["path"]),
                "wall_time": r["wall_time"],
            }
            for r in rows
        ]

    def video_tags(self) -> List[str]:
        return self._db.get_video_tags(self.experiment_id)

    def artifacts(self, tag: str) -> List[Dict[str, Any]]:
        rows = self._db.get_artifacts(self.experiment_id, tag)
        return [
            {
                "step": r["step"],
                "path": r["path"],
                "abs_path": self.resolve_media_path(r["path"]),
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "wall_time": r["wall_time"],
            }
            for r in rows
        ]

    def artifact_tags(self) -> List[str]:
        """All tags in the artifacts table — including graphs and PR curves.

        For a "user artifacts only" view (excluding graphs and PR curves,
        which now live under their own kinds), use :meth:`user_artifact_tags`.
        """
        return self._db.get_artifact_tags(self.experiment_id)

    def _filter_artifact_tags(self, *kinds: str) -> List[str]:
        """Tags whose first-row metadata.kind matches any of *kinds*."""
        out: List[str] = []
        for tag in self._db.get_artifact_tags(self.experiment_id):
            rows = self._db.get_artifacts(self.experiment_id, tag)
            if not rows:
                continue
            meta_raw = rows[0]["metadata"]
            meta = json.loads(meta_raw) if meta_raw else {}
            if meta.get("kind") in kinds or (
                "dot" in kinds and meta.get("format") == "dot"
            ):
                out.append(tag)
        return out

    def user_artifact_tags(self) -> List[str]:
        """Artifact tags excluding kinds that have their own UI bucket.

        Models (``add_graph``), PR curves (``add_pr_curve``), figures
        (``add_figure``), meshes (``add_mesh``) and embeddings
        (``add_embedding``) live under their own readers and are surfaced
        in dedicated tabs — they should not appear in the generic
        Artifacts feed.
        """
        special = (
            set(self.model_tags())
            | set(self.pr_curve_tags())
            | set(self.figure_tags())
            | set(self.mesh_tags())
            | set(self.embedding_tags())
        )
        return [t for t in self.artifact_tags() if t not in special]

    def model_tags(self) -> List[str]:
        """Tags stored under the ``add_graph`` (model diagram) channel."""
        return self._filter_artifact_tags("graph", "dot")

    def models(self, tag: str) -> List[Dict[str, Any]]:
        """Per-step model entries: DOT path, rendered PNG path, metadata."""
        rows = self.artifacts(tag)
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = r.get("metadata") or {}
            png_rel = meta.get("rendered_png_path")
            out.append(
                {
                    "step": r["step"],
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "rendered_png_path": png_rel,
                    "rendered_png_abs": (
                        self.resolve_media_path(png_rel) if png_rel else None
                    ),
                    "metadata": meta,
                    "wall_time": r["wall_time"],
                }
            )
        return out

    def pr_curve_tags(self) -> List[str]:
        """Tags stored under the ``add_pr_curve`` channel."""
        return self._filter_artifact_tags("pr_curve")

    def pr_curves(self, tag: str) -> List[Dict[str, Any]]:
        """Per-step PR-curve payloads with parsed precision/recall points."""
        out: List[Dict[str, Any]] = []
        for r in self.artifacts(tag):
            payload: Dict[str, Any] = {}
            try:
                if r["abs_path"]:
                    with open(r["abs_path"], "rb") as fh:
                        payload = json.loads(fh.read())
            except (OSError, json.JSONDecodeError):
                payload = {}
            out.append(
                {
                    "step": r["step"],
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "num_examples": payload.get("num_examples"),
                    "points": payload.get("points", []),
                    "metadata": r["metadata"],
                    "wall_time": r["wall_time"],
                }
            )
        return out

    def figure_tags(self) -> List[str]:
        """Tags stored under the ``add_figure`` (matplotlib chart) channel."""
        return self._filter_artifact_tags("figure")

    def figures(self, tag: str) -> List[Dict[str, Any]]:
        """Per-step figure entries: PNG path + metadata.

        The artifact file *is* the rendered PNG (no JSON sidecar).
        """
        out: List[Dict[str, Any]] = []
        for r in self.artifacts(tag):
            meta = r.get("metadata") or {}
            out.append(
                {
                    "step": r["step"],
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "shape": meta.get("shape"),
                    "format": meta.get("format", "png"),
                    "metadata": meta,
                    "wall_time": r["wall_time"],
                }
            )
        return out

    def mesh_tags(self) -> List[str]:
        """Tags stored under the ``add_mesh`` channel."""
        return self._filter_artifact_tags("mesh")

    def meshes(self, tag: str) -> List[Dict[str, Any]]:
        """Per-step mesh payloads with parsed vertices / colors / faces."""
        out: List[Dict[str, Any]] = []
        for r in self.artifacts(tag):
            payload: Dict[str, Any] = {}
            try:
                if r["abs_path"]:
                    with open(r["abs_path"], "rb") as fh:
                        payload = json.loads(fh.read())
            except (OSError, json.JSONDecodeError):
                payload = {}
            out.append(
                {
                    "step": r["step"],
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "vertices": payload.get("vertices"),
                    "colors": payload.get("colors"),
                    "faces": payload.get("faces"),
                    "config": payload.get("config"),
                    "metadata": r["metadata"],
                    "wall_time": r["wall_time"],
                }
            )
        return out

    def embedding_tags(self) -> List[str]:
        """Tags stored under the ``add_embedding`` channel."""
        return self._filter_artifact_tags("embedding")

    def embeddings(self, tag: str) -> List[Dict[str, Any]]:
        """Per-step embedding payloads with parsed ``mat`` / ``metadata``.

        Each entry exposes the high-D vectors (``vectors`` / ``shape``),
        per-row metadata, and — when the writer rendered a sprite atlas
        from ``label_img`` — the absolute path to the companion PNG so the
        web viewer can fetch it as a Three.js texture without re-parsing
        the JSON.
        """
        out: List[Dict[str, Any]] = []
        for r in self.artifacts(tag):
            payload: Dict[str, Any] = {}
            try:
                if r["abs_path"]:
                    with open(r["abs_path"], "rb") as fh:
                        payload = json.loads(fh.read())
            except (OSError, json.JSONDecodeError):
                payload = {}
            meta = r.get("metadata") or {}
            sprite_rel = meta.get("sprite_path")
            sprite_abs = self.resolve_media_path(sprite_rel) if sprite_rel else None
            mat = payload.get("mat") or {}
            label_img_meta = payload.get("label_img") or {}
            out.append(
                {
                    "step": r["step"],
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "vectors": mat.get("values"),
                    "shape": mat.get("shape"),
                    "metadata_rows": payload.get("metadata"),
                    "metadata_header": payload.get("metadata_header"),
                    "label_img_shape": label_img_meta.get("shape"),
                    "sprite": payload.get("sprite") or meta.get("sprite"),
                    "sprite_path": sprite_rel,
                    "sprite_abs": sprite_abs,
                    "metadata": meta,
                    "wall_time": r["wall_time"],
                }
            )
        return out

    def histograms(self, tag: str) -> List[Dict[str, Any]]:
        return self._db.get_histograms(self.experiment_id, tag)

    def histogram_tags(self) -> List[str]:
        return self._db.get_histogram_tags(self.experiment_id)

    def hparams(self) -> Dict[str, Any]:
        return self._db.get_hparams(self.experiment_id)

    def config(self) -> Optional[Dict[str, Any]]:
        row = self._experiment_row()
        if row and row.get("config"):
            return json.loads(row["config"])
        return None


class RunReader:
    """Read experiments from the central DB or an explicit local project DB."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        self.project_folder = (
            self._resolve_project_folder(project_folder)
            if project_folder is not None
            else None
        )
        self.project = self._resolve_project(project_folder, project)
        self.db_path = self._resolve_db_path(project_folder)
        self._db: Optional[Database] = None
        self._discover()

    @staticmethod
    def _resolve_project_folder(project_folder: str) -> Path:
        path = Path(project_folder).expanduser().resolve()
        if path.name == "vibetrack.db":
            return path.parent
        return path

    @classmethod
    def _resolve_db_path(cls, project_folder: Optional[str]) -> Path:
        if project_folder is None:
            return central_db_path()
        path = Path(project_folder).expanduser().resolve()
        if path.name == "vibetrack.db":
            return path
        return cls._resolve_project_folder(project_folder) / "vibetrack.db"

    @staticmethod
    def _resolve_project(
        project_folder: Optional[str],
        project: Optional[str],
    ) -> Optional[str]:
        if project is not None:
            return project
        if project_folder is None:
            # Path("/").resolve().name is "" — treat as "no project" rather
            # than a literal empty-string filter that matches legacy rows.
            name = Path.cwd().resolve().name
            return name or None
        return None

    def _discover(self) -> None:
        """Open or close the project DB as it appears or disappears."""
        if self.db_path.exists():
            if self._db is None:
                self._db = Database(self.db_path)
            return
        if self._db is not None:
            self._db.close()
            self._db = None

    _UNSET_PROJECT: Any = object()

    def experiments(self, project: Any = _UNSET_PROJECT) -> List[ExperimentReader]:
        """List experiments stored in the project database.

        ``project`` overrides the reader's default filter for this call only.
        Pass ``None`` to list all projects, or omit the argument to use
        ``self.project``. Lets callers (e.g. the Gradio dashboard) ask for a
        specific project without mutating shared reader state.
        """
        self._discover()
        if self._db is None:
            return []
        effective = self.project if project is RunReader._UNSET_PROJECT else project
        rows = self._db.list_experiments(project=effective)
        return [
            ExperimentReader(self._db, row["id"], row["name"], row=row) for row in rows
        ]

    def experiment(self, name: str) -> Optional[ExperimentReader]:
        """Get a specific experiment by name."""
        self._discover()
        if self._db is None:
            return None
        row = self._db.get_experiment_by_name(name, project=self.project)
        if row is None:
            return None
        return ExperimentReader(self._db, row["id"], row["name"], row=row)

    def projects(self) -> List[str]:
        """List all non-empty project names in the active database."""
        self._discover()
        if self._db is None:
            return []
        return [name for name in self._db.list_projects() if name]

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "RunReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
