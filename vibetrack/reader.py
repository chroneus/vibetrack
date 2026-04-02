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
        if path.is_absolute():
            return str(path)
        log_dir = self.log_dir
        if not log_dir:
            return str(path)
        resolved = (Path(log_dir) / path).resolve()
        # Reject path traversal attempts
        log_dir_resolved = str(Path(log_dir).resolve())
        if not str(resolved).startswith(log_dir_resolved + os.sep) and str(resolved) != log_dir_resolved:
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
        return self._db.get_artifact_tags(self.experiment_id)

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
            if project_folder is not None else None
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
            return Path.cwd().resolve().name
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

    def experiments(self) -> List[ExperimentReader]:
        """List all experiments stored in the project database."""
        self._discover()
        if self._db is None:
            return []
        rows = self._db.list_experiments(project=self.project)
        return [
            ExperimentReader(self._db, row["id"], row["name"], row=row)
            for row in rows
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
