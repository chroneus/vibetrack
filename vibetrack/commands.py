"""Agent-friendly CLI commands: ``ls``, ``show``, ``compare``, ``open``, ``guide``,
plus ``push`` / ``pull`` for object storage (``s3://``, ``file://``).

Dispatched from :func:`vibetrack.cli.main` before the viewer parser, so
``vibetrack``, ``vibetrack PROJECT_FOLDER`` and friends behave as before.
All read commands print compact Markdown by default and ``--json`` for
machine use; both come from :mod:`vibetrack.insights`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from . import insights
from .reader import ExperimentReader, RunReader
from .storage import is_remote_uri

COMMANDS = ("ls", "show", "compare", "open", "guide", "push", "pull")

_GUIDE_FILE = Path(__file__).with_name("agent_guide.md")
_AGENTS_START = "<!-- vibetrack:start -->"
_AGENTS_END = "<!-- vibetrack:end -->"


class CommandError(Exception):
    """User-facing error: printed without a traceback, exit code 1."""


# ── Output helpers ───────────────────────────────────────────────


def _emit(text: str, stream: Any = None) -> None:
    stream = stream or sys.stdout
    try:
        print(text, file=stream, flush=True)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        stream.buffer.write(text.encode(enc, "replace") + b"\n")
        stream.flush()


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _emit_json(payload: Any) -> None:
    _emit(json.dumps(_jsonable(payload), indent=2, default=str))


# ── Database / run selection ─────────────────────────────────────


def _open_reader(args: argparse.Namespace) -> Tuple[RunReader, Optional[str]]:
    folder = args.project_folder or os.environ.get("VIBETRACK_PROJECT_FOLDER") or None
    if folder is not None and not is_remote_uri(folder):
        folder = str(Path(folder).expanduser().resolve())
        if Path(folder).name == "vibetrack.db":
            folder = str(Path(folder).parent)
    reader = RunReader(folder)
    if reader.remote is not None and reader.remote.last_error:
        raise CommandError(f"Cannot read {folder}: {reader.remote.last_error}")
    if reader.db is None:
        where = f"{folder}/vibetrack.db" if folder else str(reader.db_path)
        hint = _local_db_hint() if folder is None else ""
        raise CommandError(f"No vibetrack database at {where}.{hint}")
    return reader, folder


def _local_db_hint() -> str:
    """Point at project-local databases near the cwd (one or two levels down)."""
    found: List[str] = []
    cwd = Path.cwd()
    for pattern in ("vibetrack.db", "*/vibetrack.db", "*/*/vibetrack.db"):
        for path in sorted(cwd.glob(pattern))[:5]:
            rel = os.path.relpath(path.parent, cwd)
            if rel not in found:
                found.append(rel)
    if not found:
        return ""
    return " Project-local databases here: " + ", ".join(
        f"--project-folder {insights._q(f)}" for f in found[:5]
    )


def _experiments(reader: RunReader, project: Optional[str]) -> List[ExperimentReader]:
    return reader.experiments(project=project)


def _select(
    reader: RunReader,
    patterns: Sequence[str],
    project: Optional[str],
) -> List[ExperimentReader]:
    """Resolve run names or glob patterns; error out on unknown or ambiguous names."""
    pool = _experiments(reader, project)
    chosen: List[ExperimentReader] = []
    seen = set()
    for pattern in patterns:
        exact = [e for e in pool if e.name == pattern]
        matches = exact or [e for e in pool if fnmatch.fnmatchcase(e.name, pattern)]
        if not matches:
            recent = ", ".join(e.name for e in pool[:10]) or "none"
            where = f" in project {project!r}" if project else ""
            raise CommandError(f"No run matches {pattern!r}{where}. Recent runs: {recent}")
        if exact and len({e.project for e in exact}) > 1:
            projects = ", ".join(sorted({e.project for e in exact}))
            raise CommandError(
                f"Run {pattern!r} exists in several projects ({projects}); pass -p PROJECT"
            )
        for exp in matches:
            if exp.experiment_id not in seen:
                seen.add(exp.experiment_id)
                chosen.append(exp)
    return chosen


def _latest_project(reader: RunReader) -> Optional[str]:
    exps = _experiments(reader, None)
    return exps[0].project if exps else None


# ── Commands ─────────────────────────────────────────────────────


def cmd_ls(args: argparse.Namespace) -> int:
    reader, folder = _open_reader(args)
    exps = _experiments(reader, args.project)[: max(1, args.limit)]
    rows = insights.list_runs(exps)
    if args.json:
        _emit_json(rows)
    else:
        text = insights.format_runs(rows, project_folder=folder)
        if not rows and args.project:
            projects = reader.projects()
            if projects:
                text += "\nProjects: " + ", ".join(projects[:20])
        _emit(text)
    return 0


_FETCH_KINDS = ("image", "figure", "audio")


def _fetch_latest_media(exp: ExperimentReader, ov: Any) -> None:
    """Download a synced run's latest images so the listed paths can be opened.

    Videos and artifacts can be large, so they stay in storage until opened
    in the dashboard (which streams videos directly).
    """
    for kind, tags in (ov.get("media") or {}).items():
        if kind not in _FETCH_KINDS:
            continue
        for entry in tags.values():
            path = entry.get("latest")
            if isinstance(path, str):
                exp.ensure_local(path)


def cmd_show(args: argparse.Namespace) -> int:
    reader, folder = _open_reader(args)
    if args.run:
        exps = _select(reader, [args.run], args.project)
        if len(exps) > 1:
            names = ", ".join(e.name for e in exps[:10])
            raise CommandError(f"{args.run!r} matches several runs ({names}); use compare")
        exp = exps[0]
    else:
        pool = _experiments(reader, args.project)
        if not pool:
            raise CommandError("No runs found.")
        exp = pool[0]
    ov = insights.run_overview(exp, include_system=args.system)
    if reader.remote is not None:
        _fetch_latest_media(exp, ov)
    if args.json:
        _emit_json(ov)
    else:
        _emit(insights.format_overview(ov, project_folder=folder))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reader, folder = _open_reader(args)
    if args.runs:
        exps = _select(reader, args.runs, args.project)
    else:
        project = args.project if args.project is not None else _latest_project(reader)
        exps = _experiments(reader, project)[: max(1, args.last)]
    if not exps:
        raise CommandError("No runs to compare.")
    cmp = insights.compare_runs(
        exps,
        metric=args.metric,
        goal=args.goal,
        include_system=args.system,
        focus=bool(args.runs),
    )
    if args.json:
        _emit_json(cmp)
    else:
        _emit(insights.format_compare(cmp, project_folder=folder))
    return 0


def _probe(port: int, host: str = "127.0.0.1") -> Optional[dict]:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/info", timeout=0.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("app") == "vibetrack" else None


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _same_path(a: Any, b: Any) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except Exception:
        return False


def ensure_dashboard(
    project_folder: Optional[str],
    db_path: Path,
    port: int = 6116,
    tries: int = 10,
    timeout: float = 25.0,
) -> Tuple[str, Optional[int], Optional[Path]]:
    """Reuse or start a local dashboard for *db_path*.

    Returns ``(base_url, started_pid, log_path)``; the pid and log are
    ``None`` when an existing dashboard was reused.
    """
    free: Optional[int] = None
    for p in range(port, port + tries):
        info = _probe(p)
        if info is not None and _same_path(info.get("db_path"), db_path):
            return f"http://127.0.0.1:{p}", None, None
        if info is None and free is None and _port_free(p):
            free = p
    if free is None:
        raise CommandError(f"No free port in {port}..{port + tries - 1} for the dashboard.")

    log_dir = Path.home() / ".vibetrack"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dashboard-{free}.log"
    cmd = [sys.executable, "-m", "vibetrack.cli", "--host", "127.0.0.1", "--port", str(free)]
    if project_folder:
        cmd += ["--project-folder", project_folder]
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs
        )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-15:]
            raise CommandError(
                "Dashboard exited during startup:\n" + "\n".join(tail)
            )
        info = _probe(free)
        if info is not None:
            return f"http://127.0.0.1:{free}", proc.pid, log_path
        time.sleep(0.25)
    raise CommandError(f"Dashboard did not answer within {timeout:.0f}s; see {log_path}")


def cmd_open(args: argparse.Namespace) -> int:
    reader, folder = _open_reader(args)
    runs: List[str] = []
    project = args.project
    if args.runs:
        exps = _select(reader, args.runs, project)
        runs = [e.name for e in exps]
        projects = {e.project for e in exps}
        if project is None:
            project = exps[0].project
        if len(projects) > 1:
            _emit(f"note: runs span projects {sorted(projects)}; showing {project!r}", sys.stderr)
    elif project is None:
        project = _latest_project(reader)
    base, pid, log_path = ensure_dashboard(folder, reader.db_path, port=args.port)
    url = insights.dashboard_url(project, runs, tab=args.tab, base=base)
    if pid is not None:
        _emit(
            f"Started vibetrack dashboard in the background (pid {pid}, log {log_path}); "
            f"stop it with: kill {pid}"
            if os.name != "nt"
            else f"Started vibetrack dashboard (pid {pid}, log {log_path})"
        )
    else:
        _emit(f"Using the vibetrack dashboard already running at {base}")
    _emit(f"Dashboard: {url}")
    if not args.no_browser:
        import webbrowser

        try:
            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            _emit("(no browser available here; open the URL above)")
    return 0


def guide_text() -> str:
    return _GUIDE_FILE.read_text(encoding="utf-8")


_SKILL_FRONTMATTER = """---
name: vibetrack
description: Track ML experiments with vibetrack - instrument training code (config, metrics, videos/images), run it, then list, inspect and compare runs and open the dashboard on the results. Use when the user asks to track, log, monitor or compare experiments, or says "use vibetrack".
---

"""

_AGENTS_BLOCK = f"""{_AGENTS_START}
## Experiment tracking (vibetrack)

This project tracks experiments with [vibetrack](https://github.com/chroneus/vibetrack).
Run `vibetrack guide` for the full instrumentation checklist. In short:
log the full config, namespaced metrics (`train/`, `val/`, `eval/`) and
qualitative media (videos/images) with `vibetrack.init(...)` / `vibetrack.log(...)`,
then use `vibetrack ls`, `vibetrack show RUN`, `vibetrack compare` to report
results and `vibetrack open RUN` to show the user the dashboard.
{_AGENTS_END}
"""


def install_guide(root: Path, agents_md: bool = True) -> List[str]:
    """Write the Claude Code skill and an AGENTS.md block under *root*."""
    written: List[str] = []
    skill = root / ".claude" / "skills" / "vibetrack" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(_SKILL_FRONTMATTER + guide_text(), encoding="utf-8")
    written.append(str(skill))
    if agents_md:
        target = root / "AGENTS.md"
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if _AGENTS_START in current and _AGENTS_END in current:
            head, rest = current.split(_AGENTS_START, 1)
            tail = rest.split(_AGENTS_END, 1)[1]
            updated = head + _AGENTS_BLOCK.rstrip("\n") + tail
        else:
            sep = "" if not current or current.endswith("\n\n") else ("\n" if current.endswith("\n") else "\n\n")
            updated = current + sep + _AGENTS_BLOCK
        if updated != current:
            target.write_text(updated, encoding="utf-8")
            written.append(str(target))
    return written


def cmd_guide(args: argparse.Namespace) -> int:
    if args.install:
        root = Path(args.dir).expanduser().resolve()
        written = install_guide(root, agents_md=not args.no_agents_md)
        _emit("Installed vibetrack agent instructions:\n" + "\n".join(f"- {p}" for p in written))
        return 0
    _emit(guide_text())
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from .db import Database
    from .reader import RunReader as _Reader
    from .sync import push_runs

    targets = list(args.targets)
    dest_idx = next((i for i, t in enumerate(targets) if is_remote_uri(t)), None)
    if dest_idx is None:
        raise CommandError("push needs a destination such as s3://bucket/prefix or file:///path")
    before, uri, runs = targets[:dest_idx], targets[dest_idx], targets[dest_idx + 1:]
    if len(before) > 1:
        raise CommandError("push takes at most one PROJECT_FOLDER before the destination")
    folder = before[0] if before else args.project_folder
    db_path = _Reader._resolve_db_path(folder)
    if not db_path.exists():
        raise CommandError(f"No vibetrack database at {db_path}.")
    db = Database(db_path)
    try:
        return push_runs(db, uri, project=args.project, names=runs or None,
                         remove_local=args.remove_local, out=_emit)
    finally:
        db.close()


def cmd_pull(args: argparse.Namespace) -> int:
    from .sync import pull

    if not is_remote_uri(args.source):
        raise CommandError("pull needs a source such as s3://bucket/prefix or file:///path")
    dest = str(Path(args.dest).expanduser().resolve())
    return pull(args.source, dest, out=_emit)


# ── Parser ───────────────────────────────────────────────────────


def _add_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-folder",
        default=None,
        metavar="DIR",
        help="Folder holding a project-local vibetrack.db, or synced storage such as s3://bucket/prefix (default: central ~/.vibetrack/vibetrack.db; env VIBETRACK_PROJECT_FOLDER)",
    )
    parser.add_argument("-p", "--project", default=None, help="Only runs of this project")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibetrack")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ls", help="List runs with status, timing and headline metrics")
    _add_scope(p)
    p.add_argument("--limit", type=int, default=30, help="Most recent N runs (default 30)")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("show", help="Details of one run (default: the latest)")
    p.add_argument("run", nargs="?", help="Run name (default: most recent run)")
    _add_scope(p)
    p.add_argument("--system", action="store_true", help="Include system/GPU metrics")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("compare", help="Rank runs and diff their params and metrics")
    p.add_argument("runs", nargs="*", help="Run names or glob patterns (default: latest runs of the latest project)")
    _add_scope(p)
    p.add_argument("--metric", help="Metric to rank by (default: declared or auto-selected)")
    p.add_argument("--goal", choices=["max", "min"], help="Whether higher or lower is better")
    p.add_argument("--last", type=int, default=10, help="Without run names: compare the N most recent runs (default 10)")
    p.add_argument("--system", action="store_true", help="Include system/GPU metrics")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("open", help="Open the dashboard on runs, starting it in the background if needed")
    p.add_argument("runs", nargs="*", help="Run names or globs to focus")
    _add_scope(p)
    p.add_argument("--port", type=int, default=6116, help="First port to try (default 6116)")
    p.add_argument("--tab", default="runs", help="Dashboard tab to open (default: runs)")
    p.add_argument("--no-browser", action="store_true", help="Only print the URL")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("guide", help="Print instructions for AI coding agents")
    p.add_argument("--install", action="store_true", help="Write .claude/skills/vibetrack/SKILL.md and an AGENTS.md section")
    p.add_argument("--dir", default=".", help="Project root for --install (default: .)")
    p.add_argument("--no-agents-md", action="store_true", help="With --install: skip AGENTS.md")
    p.set_defaults(func=cmd_guide)

    p = sub.add_parser(
        "push",
        help="Upload runs to object storage: push [PROJECT_FOLDER] s3://bucket/prefix [RUN ...]",
    )
    p.add_argument("targets", nargs="+", metavar="ARG",
                   help="[PROJECT_FOLDER] DESTINATION [RUN ...]; DESTINATION is s3://… or file://…")
    _add_scope(p)
    p.add_argument("--remove-local", action="store_true",
                   help="Delete each run's local rows and media once its upload is verified")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("pull", help="Copy all runs (with media) from object storage into a folder")
    p.add_argument("source", help="s3://bucket/prefix or file:///path")
    p.add_argument("dest", help="Folder to create or update; open it later with `vibetrack DEST`")
    p.set_defaults(func=cmd_pull)
    return parser


def run(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(list(argv))
    try:
        return int(args.func(args) or 0)
    except CommandError as exc:
        _emit(f"vibetrack: {exc}", sys.stderr)
        return 1
