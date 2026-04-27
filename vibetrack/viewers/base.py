"""Base interface for all output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..reader import ExperimentReader, RunReader
from .event import LogEvent


class BaseOutput(ABC):
    """Every output backend implements this interface.

    Two roles are supported and need not both be implemented:

    * ``show(**kwargs)`` — pull-based dashboard/summary (required).
    * ``send(events)``  — push-based per-event dispatch (optional, default no-op).
    """

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        if project_folder is None:
            self.project_folder = None
        else:
            resolved = Path(project_folder).expanduser().resolve()
            if resolved.name == "vibetrack.db":
                resolved = resolved.parent
            self.project_folder = str(resolved)
        self.project = project
        self._reader = RunReader(project_folder, project=project)

    def config_project(self) -> Optional[str]:
        if self.project is not None:
            return self.project
        if self.project_folder is not None:
            return Path(self.project_folder).name
        return Path.cwd().resolve().name

    @abstractmethod
    def show(self, **kwargs: Any) -> Any:
        """Render the output. What 'show' means depends on the backend."""
        ...

    def send(self, events: List[LogEvent]) -> None:
        """Receive a batch of log events for push-based dispatch.

        Default: no-op.  Override in adapters that forward per-event data
        (telegram, slack, console, gradio).  Pure dashboards (web, mcp) leave
        the default so they are silently ignored by ``SummaryWriter.to(...)``.
        """
        return None

    def close(self) -> None:
        self._reader.close()

    def _resolve_experiments(
        self,
        names: Optional[Sequence[str]] = None,
    ) -> List[ExperimentReader]:
        all_exps = self._reader.experiments()
        if names is None:
            return all_exps
        name_set = set(names)
        return [e for e in all_exps if e.name in name_set]
