"""Run provenance and lifecycle helpers (process, git, exit status).

Everything here is best-effort: a missing ``git`` binary, a slow repository
or an unusual platform must never break the user's training script, so each
helper swallows its own errors and returns partial information.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

_SECRET_ARG = re.compile(r"(token|secret|password|passwd|api[-_]?key|auth)", re.I)
_ENV_MARKERS = ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "CUDA_VISIBLE_DEVICES")

_process_info_cache: Optional[Dict[str, Any]] = None


def _redact_argv(argv: List[str]) -> List[str]:
    """Mask values of arguments whose name looks like a credential."""
    out: List[str] = []
    mask_next = False
    for arg in argv:
        if mask_next:
            out.append("***")
            mask_next = False
            continue
        if arg.startswith("-") and _SECRET_ARG.search(arg):
            if "=" in arg:
                out.append(arg.split("=", 1)[0] + "=***")
            else:
                out.append(arg)
                mask_next = True
            continue
        out.append(arg)
    return out


def _git(args: List[str], cwd: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_info(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return ``{commit, branch, dirty, root}`` for the repo containing *path*."""
    if os.environ.get("VIBETRACK_GIT", "1").lower() in ("0", "false", "no", "off"):
        return None
    cwd = path or os.getcwd()
    if not os.path.isdir(cwd):
        return None
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return None
    commit = _git(["rev-parse", "HEAD"], cwd)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    info: Dict[str, Any] = {"root": root}
    if commit:
        info["commit"] = commit
    if branch and branch != "HEAD":
        info["branch"] = branch
    if status is not None:
        changed = [line for line in status.splitlines() if line.strip()]
        info["dirty"] = bool(changed)
        if changed:
            info["changed_files"] = len(changed)
    return info


def _script_dir() -> Optional[str]:
    main = sys.argv[0] if sys.argv else ""
    if main and os.path.isfile(main):
        return os.path.dirname(os.path.abspath(main))
    return None


def process_info() -> Dict[str, Any]:
    """Describe the current process; cached because git calls are not free."""
    global _process_info_cache
    pid = os.getpid()
    if _process_info_cache is not None and _process_info_cache.get("pid") == pid:
        return dict(_process_info_cache)
    info: Dict[str, Any] = {"pid": pid}
    try:
        info["host"] = socket.gethostname()
    except OSError:
        pass
    try:
        info["argv"] = _redact_argv([str(a) for a in sys.argv])
    except Exception:
        pass
    try:
        info["cwd"] = os.getcwd()
    except OSError:
        pass
    info["python"] = platform.python_version()
    try:
        from . import __version__

        info["vibetrack"] = __version__
    except Exception:
        pass
    env = {k: os.environ[k] for k in _ENV_MARKERS if os.environ.get(k)}
    if env:
        info["env"] = env
    try:
        git = git_info(_script_dir() or info.get("cwd"))
    except Exception:
        git = None
    if git:
        info["git"] = git
    _process_info_cache = info
    return dict(info)


def command_line(info: Dict[str, Any]) -> str:
    """Render the recorded argv as a shell-like command string."""
    argv = info.get("argv") or []
    if not argv:
        return ""
    import shlex

    parts = list(argv)
    if parts and parts[0].endswith(".py"):
        parts = ["python", *parts]
    try:
        return " ".join(shlex.quote(p) for p in parts)
    except Exception:
        return " ".join(parts)


def exit_status(exc: Optional[BaseException]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Map the exception a run ended with to ``(status, error_info)``."""
    if exc is None:
        return "finished", None
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", {"error": "KeyboardInterrupt"}
    if isinstance(exc, SystemExit):
        if exc.code in (0, None):
            return "finished", None
        return "failed", {"error": f"SystemExit({exc.code!r})"}
    if isinstance(exc, GeneratorExit):
        return "finished", None
    message = f"{type(exc).__name__}: {exc}"
    error: Dict[str, Any] = {"error": message[:2000]}
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error["traceback"] = tb[-4000:]
    except Exception:
        pass
    return "failed", error


def is_interactive() -> bool:
    """True in a REPL / notebook, where ``sys.last_value`` can be stale."""
    return bool(
        hasattr(sys, "ps1")
        or sys.flags.interactive
        or "ipykernel" in sys.modules
        or "IPython" in sys.modules
    )


def uncaught_exception() -> Optional[BaseException]:
    """The exception that is terminating the interpreter, if any.

    Valid inside ``atexit`` handlers of scripts: CPython stores the unhandled
    exception in ``sys.last_value`` before running them.
    """
    if is_interactive():
        return None
    exc = getattr(sys, "last_exc", None) or getattr(sys, "last_value", None)
    return exc if isinstance(exc, BaseException) else None


def pid_alive(pid: Any, started_at: Optional[float] = None) -> Optional[bool]:
    """Whether *pid* still runs the process that started at *started_at*.

    Returns ``None`` when this cannot be determined.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        import psutil  # base dependency, but keep the import lazy
    except ImportError:
        return None
    try:
        if not psutil.pid_exists(pid):
            return False
        if started_at is not None:
            created = psutil.Process(pid).create_time()
            # A process created after the run started is a recycled pid.
            if created > float(started_at) + 1.0:
                return False
        return True
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def this_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""
