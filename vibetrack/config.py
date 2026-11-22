"""User configuration stored at ~/.vibetrack/config.json."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


from .default_config import DEFAULT_CONFIG  # noqa: F401 — re-exported

VIBETRACK_CONFIG_DIR = Path.home() / ".cache" / "vibetrack"


def config_dir() -> Path:
    """Return ``~/.vibetrack``."""
    from .db import central_db_dir

    return central_db_dir()


def config_path() -> Path:
    return config_dir() / "config.json"


def _read_config_file() -> Optional[Dict[str, Any]]:
    path = config_path()
    if not path.exists():
        old = VIBETRACK_CONFIG_DIR / "config.json"
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
    result = _deep_merge(
        DEFAULT_CONFIG, default_cfg if isinstance(default_cfg, dict) else {}
    )
    if project:
        projects = data.get("projects", {})
        project_cfg = projects.get(project, {}) if isinstance(projects, dict) else {}
        if isinstance(project_cfg, dict):
            result = _deep_merge(result, project_cfg)
    return result


# Sections holding credentials and destinations (bot tokens, webhooks,
# chat/channel ids, S3 keys and endpoints). The web API neither returns nor overwrites
# them: anyone who can reach it could otherwise read the tokens, or point
# notifications (with logged media) at their own chat.
CREDENTIAL_SECTIONS = ("telegram", "slack", "s3")


def _stored_scope(project: Optional[str]) -> Dict[str, Any]:
    """Return the raw on-disk dict that ``save_config(..., project)`` replaces."""
    data = _read_config_file()
    if data is None:
        return {}
    if not _is_project_store(data):
        # A flat file is the default scope; saving a project scope moves it
        # under "default" untouched, so that project starts out empty.
        return {} if project else data
    if project:
        projects = data.get("projects")
        scope = projects.get(project) if isinstance(projects, dict) else None
    else:
        scope = data.get("default")
    return scope if isinstance(scope, dict) else {}


def load_public_config(project: Optional[str] = None) -> Dict[str, Any]:
    """``load_config`` with credential sections blanked, for HTTP clients."""
    cfg = load_config(project)
    for key in CREDENTIAL_SECTIONS:
        if key in cfg:
            cfg[key] = {}
    return cfg


def save_public_config(data: Dict[str, Any], project: Optional[str] = None) -> None:
    """``save_config`` for HTTP clients: credential sections keep stored values.

    Whatever the client sends for those sections is ignored, so the web UI
    (which posts back the blanked config it loaded) doesn't erase them.
    """
    stored = _stored_scope(project)
    merged = {k: v for k, v in data.items() if k not in CREDENTIAL_SECTIONS}
    for key in CREDENTIAL_SECTIONS:
        if key in stored:
            merged[key] = stored[key]
    save_config(merged, project)


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
