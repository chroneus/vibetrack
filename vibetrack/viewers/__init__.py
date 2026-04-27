"""Output backends for vibetrack.

Each backend lives in its own module with optional heavy dependencies.
Import only what you need::

    from vibetrack.viewers.console import ConsoleOutput
    from vibetrack.viewers.web import WebOutput

Auto-discovery::

    from vibetrack.viewers import discover_viewers, load_viewer
    viewers = discover_viewers()    # {"web": "vibetrack.viewers.web", ...}
    cls = load_viewer("web")        # WebOutput class (lazy import)
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Dict, Type

from .base import BaseOutput

__all__ = ["BaseOutput", "discover_viewers", "load_viewer"]

_EXCLUDED = {"__init__.py", "base.py", "event.py"}


def _viewer_name(filename: str) -> str:
    """Derive viewer name from filename: strip ``.py`` and ``_ui`` suffix."""
    name = filename.removesuffix(".py")
    if name.endswith("_ui"):
        name = name[:-3]
    return name


def discover_viewers() -> Dict[str, str]:
    """Return ``{name: module_path}`` for all viewer modules.

    Scans ``vibetrack/viewers/*.py``, excluding ``__init__.py`` and ``base.py``.
    """
    pkg_dir = Path(__file__).parent
    result: Dict[str, str] = {}
    for child in sorted(pkg_dir.iterdir()):
        if child.suffix == ".py" and child.name not in _EXCLUDED:
            name = _viewer_name(child.name)
            result[name] = f"vibetrack.viewers.{child.stem}"
    return result


def load_viewer(name: str) -> Type[BaseOutput]:
    """Lazily import and return the :class:`BaseOutput` subclass for *name*.

    Raises :class:`ValueError` if *name* is not a known viewer.
    """
    viewers = discover_viewers()
    if name not in viewers:
        available = ", ".join(sorted(viewers))
        raise ValueError(f"Unknown viewer {name!r}. Available: {available}")
    module = importlib.import_module(viewers[name])
    for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseOutput) and obj is not BaseOutput:
            return obj
    raise ImportError(f"No BaseOutput subclass found in {viewers[name]}")
