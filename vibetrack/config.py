"""User configuration stored at ~/.vibetrack/config.json."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


from .defaults import DEFAULT_CONFIG  # noqa: F401 — re-exported

_OLD_CONFIG_DIR = Path.home() / ".cache" / "vibetrack"


def config_dir() -> Path:
    """Return ``~/.vibetrack``."""
    from .db import central_db_dir
    return central_db_dir()


def config_path() -> Path:
    return config_dir() / "config.json"


def _read_config_file() -> Optional[Dict[str, Any]]:
    path = config_path()
    if not path.exists():
        old = _OLD_CONFIG_DIR / "config.json"
        if old.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(old), str(path))
            except OSError:
                pass
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _is_project_store(data: Dict[str, Any]) -> bool:
    projects = data.get("projects")
    default = data.get("default")
    return isinstance(projects, dict) or isinstance(default, dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(project: Optional[str] = None) -> Dict[str, Any]:
    """Load config from disk, merged with defaults for missing keys."""
    data = _read_config_file()
    if data is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not _is_project_store(data):
        return _deep_merge(DEFAULT_CONFIG, data)

    default_cfg = data.get("default", {})
    result = _deep_merge(DEFAULT_CONFIG, default_cfg if isinstance(default_cfg, dict) else {})
    if project:
        projects = data.get("projects", {})
        project_cfg = projects.get(project, {}) if isinstance(projects, dict) else {}
        if isinstance(project_cfg, dict):
            result = _deep_merge(result, project_cfg)
    return result


def save_config(data: Dict[str, Any], project: Optional[str] = None) -> None:
    """Write config to disk, optionally scoped to a project."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if project:
        existing = _read_config_file()
        if existing is None:
            payload: Dict[str, Any] = {"projects": {project: data}}
        elif _is_project_store(existing):
            payload = copy.deepcopy(existing)
            projects = payload.setdefault("projects", {})
            if not isinstance(projects, dict):
                projects = {}
                payload["projects"] = projects
            projects[project] = data
        else:
            payload = {"default": existing, "projects": {project: data}}
    else:
        existing = _read_config_file()
        if existing is not None and _is_project_store(existing):
            payload = copy.deepcopy(existing)
            payload["default"] = data
        else:
            payload = data
    # Atomic write: temp file + rename to prevent corruption on crash
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
