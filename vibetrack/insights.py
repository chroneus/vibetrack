"""Run summaries and cross-run comparisons for coding agents and people.

This module is the single source for ``vibetrack ls / show / compare``, the
end-of-run report printed on close, the web dashboard's Runs tab and the
MCP ``list_runs`` / ``run_overview`` / ``compare_runs`` tools. Every
function returns plain JSON-serialisable dicts; the ``format_*`` helpers
render them as compact Markdown that an LLM can read in one pass.

Output is ASCII-only so it prints on any console encoding.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urlencode

from ._runmeta import command_line, pid_alive, this_host

SYSTEM_PREFIXES = ("system/", "gpu/")
DEFAULT_DASHBOARD = "http://127.0.0.1:6116"
#: A ``running`` run without any write for this long is flagged ``stale``.
STALE_SECS = 3600.0

# ── Metric goals ─────────────────────────────────────────────────

_MIN_WORDS = frozenset(
    """loss losses err error errors mse mae rmse mape nll bce perplexity ppl
    wer cer ter fid kid lpips regret cost costs latency bpd bpc distance
    penalty violations collisions crashes failures misses""".split()
)
_MAX_WORDS = frozenset(
    """acc accuracy f1 auc auroc auprc ap map precision recall return returns
    ret reward rewards rew score scores bleu rouge meteor chrf psnr ssim
    msssim iou miou dice success successes win wins winrate top1 top5 r2
    ndcg mrr hits coverage elo sps throughput""".split()
)
_EVAL_WORDS = frozenset(
    "val valid validation eval evaluation test dev holdout".split()
)
_BORING_WORDS = frozenset(
    """lr learning_rate epoch epochs step steps iter iteration iterations
    time timestamp global_step""".split()
)


def is_system_tag(tag: str) -> bool:
    return tag.startswith(SYSTEM_PREFIXES)


def _tokens(tag: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", tag.lower()) if t]


def infer_goal(tag: str) -> Optional[str]:
    """Guess whether *tag* should be minimised or maximised from its name.

    The right-most recognised word wins, so ``train/reward_loss`` is ``min``
    and ``eval/return`` is ``max``. Returns ``None`` when unknown.
    """
    for tok in reversed(_tokens(tag)):
        bare = tok.rstrip("0123456789") or tok
        if tok in _MIN_WORDS or bare in _MIN_WORDS:
            return "min"
        if tok.startswith("loss") or tok.endswith("loss"):
            return "min"
        if tok in _MAX_WORDS or bare in _MAX_WORDS:
            return "max"
    return None


def declared_metrics(run_info: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Metric declarations from ``define_metric`` stored on a run."""
    if not run_info:
        return {}
    metrics = (run_info.get("info") or {}).get("metrics") or {}
    return {str(k): dict(v) for k, v in metrics.items() if isinstance(v, dict)}


def goal_for(tag: str, declared: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    goal = ((declared or {}).get(tag) or {}).get("goal")
    return goal if goal in ("min", "max") else infer_goal(tag)


def _tag_score(tag: str, declared: Dict[str, Dict[str, Any]]) -> float:
    entry = declared.get(tag)
    if entry and entry.get("primary"):
        return 100.0
    toks = _tokens(tag)
    goal = goal_for(tag, declared)
    score = 0.0
    if entry is not None:
        score += 10
    if goal is not None:
        score += 2 + (1 if goal == "max" else 0)
    else:
        score -= 1
    if set(toks) & _EVAL_WORDS:
        score += 4
    if toks and (toks[-1] in _BORING_WORDS or "_".join(toks[-2:]) in _BORING_WORDS):
        score -= 5
    return score


def rank_tags(tags: Iterable[str], declared: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Order tags from most to least interesting; system tags are dropped."""
    declared = declared or {}
    items = [t for t in dict.fromkeys(tags) if not is_system_tag(t)]
    order = {t: i for i, t in enumerate(items)}
    return sorted(items, key=lambda t: (-_tag_score(t, declared), order[t]))


def headline_tags(
    tags: Iterable[str],
    declared: Optional[Dict[str, Dict[str, Any]]] = None,
    limit: int = 3,
) -> List[str]:
    """Most informative tags; schedule-like tags (lr, epoch) only as filler."""
    declared = declared or {}
    ranked = rank_tags(tags, declared)
    good = [t for t in ranked if _tag_score(t, declared) > -4]
    return (good or ranked)[:limit]


def pick_primary(
    tag_sets: Sequence[Set[str]],
    declared: Optional[Dict[str, Dict[str, Any]]] = None,
    metric: Optional[str] = None,
    goal: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Choose the metric runs are ranked by: ``(tag, goal, reason)``."""
    declared = declared or {}
    if metric:
        g = goal or goal_for(metric, declared)
        if g is None:
            return metric, "max", "given; goal unknown, assumed max (pass --goal)"
        return metric, g, "given"
    for tag, entry in declared.items():
        if entry.get("primary"):
            return tag, goal or goal_for(tag, declared) or "max", "declared with define_metric"
    sets = [s for s in tag_sets if s]
    if not sets:
        return None, None, "no scalar metrics logged"
    common = set.intersection(*sets) if len(sets) > 1 else set(sets[0])
    pool = common or set.union(*sets)
    candidates = [t for t in rank_tags(sorted(pool), declared) if goal_for(t, declared)]
    if not candidates:
        return None, None, "no metric with a known goal; pass --metric TAG --goal max|min"
    tag = candidates[0]
    return tag, goal or goal_for(tag, declared), "auto-selected (override with --metric/--goal)"


# ── Formatting ───────────────────────────────────────────────────


def fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == int(value) and abs(value) < 1e9:
            return str(int(value))
        a = abs(value)
        if a >= 1e6 or a < 1e-3:
            return f"{value:.3e}"
        if a >= 100:
            return f"{value:.1f}"
        return f"{value:.4g}"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(str(text).split())
    return text if len(text) <= 48 else text[:45] + "..."


def fmt_duration(secs: Optional[float]) -> str:
    if secs is None or secs < 0:
        return "-"
    s = int(round(secs))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def fmt_age(ts: Optional[float], now: Optional[float] = None) -> str:
    if ts is None:
        return "-"
    delta = (now or time.time()) - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _cell(value: Any) -> str:
    """Make *value* safe inside a Markdown table cell."""
    text = value if isinstance(value, str) else fmt_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _q(text: str) -> str:
    return shlex.quote(text)


def scope_args(project: Optional[str], project_folder: Optional[str] = None) -> str:
    """CLI flags that select the same database and project, e.g. `` -p cv``."""
    parts = []
    if project_folder:
        parts.append(f"--project-folder {_q(str(project_folder))}")
    if project:
        parts.append(f"-p {_q(project)}")
    return (" " + " ".join(parts)) if parts else ""


def dashboard_url(
    project: Optional[str],
    runs: Sequence[str] = (),
    tab: Optional[str] = "runs",
    base: str = DEFAULT_DASHBOARD,
) -> str:
    path = "/" + quote(project, safe="") if project else "/"
    params = [("run", r) for r in runs]
    if tab:
        params.append(("tab", tab))
    return base.rstrip("/") + path + ("?" + urlencode(params) if params else "")


def flatten_params(params: Optional[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dicts to dotted keys: ``{"opt": {"lr": 1}}`` → ``opt.lr``."""
    out: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten_params(value, name + "."))
        else:
            out[name] = value
    return out


def _value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


# ── Status ───────────────────────────────────────────────────────


def run_status(exp: Any, now: Optional[float] = None) -> Dict[str, Any]:
    """Effective lifecycle of a run, including dead-process detection."""
    now = now or time.time()
    rec = exp.run_info()
    activity = exp.last_activity() or {}
    last_wall = activity.get("wall_time")
    out: Dict[str, Any] = {
        "status": "unknown",
        "started_at": exp.created_at,
        "ended_at": None,
        "duration_s": None,
        "last_step": activity.get("step"),
        "last_activity": last_wall,
        "stale": False,
    }
    if rec is None:
        if exp.created_at is not None and last_wall is not None:
            out["duration_s"] = max(0.0, last_wall - exp.created_at)
        return out

    info = rec.get("info") or {}
    status = rec.get("status") or "unknown"
    started = rec.get("started_at") or exp.created_at
    ended = rec.get("ended_at")
    beats = [t for t in (rec.get("heartbeat"), last_wall) if t is not None]
    heartbeat = max(beats) if beats else None
    if status == "running":
        alive = None
        if info.get("pid") and info.get("host") and info.get("host") == this_host():
            alive = pid_alive(info.get("pid"), info.get("resumed_at") or started)
        if alive is False:
            status = "crashed"
            ended = heartbeat
            out["error"] = (
                "process exited without closing the run "
                "(killed, out of memory or crashed hard)"
            )
        elif heartbeat is not None and now - heartbeat > STALE_SECS and alive is None:
            out["stale"] = True
    end = ended if ended is not None else (now if status == "running" else heartbeat)
    out.update(
        status=status,
        started_at=started,
        ended_at=ended,
        heartbeat=heartbeat,
        duration_s=(end - started) if (end is not None and started is not None) else None,
    )
    for key in ("error", "traceback", "host", "pid", "cwd", "git", "env", "resumed_at"):
        if info.get(key) is not None and key not in out:
            out[key] = info[key]
    cmd = command_line(info)
    if cmd:
        out["command"] = cmd
    return out


def _status_text(st: Dict[str, Any], width: int = 60) -> str:
    status = st.get("status", "unknown")
    if status in ("failed", "crashed") and st.get("error"):
        text = f"{status}: {st['error'].splitlines()[0]}"
    elif status == "running" and st.get("stale"):
        seen = st.get("heartbeat") or st.get("last_activity")
        text = "running? (no writes for " + fmt_duration(
            time.time() - seen if seen else None
        ) + ")"
    else:
        text = status
    return text if len(text) <= width else text[: width - 3] + "..."


# ── Series summaries ─────────────────────────────────────────────


def classify_trend(values: Sequence[float]) -> Dict[str, Any]:
    """Direction and consistency of a series (also used by the MCP tools)."""
    if len(values) < 2:
        return {
            "direction": "insufficient_data",
            "positive_steps": 0,
            "negative_steps": 0,
            "flat_steps": 0,
            "consistency": None,
        }
    diffs = [b - a for a, b in zip(values, values[1:])]
    value_range = max(values) - min(values)
    eps = max(value_range * 1e-6, 1e-12)
    positive = sum(1 for d in diffs if d > eps)
    negative = sum(1 for d in diffs if d < -eps)
    flat = len(diffs) - positive - negative
    net = values[-1] - values[0]
    if abs(net) <= eps:
        direction = "flat"
        consistency = flat / len(diffs) if diffs else None
    elif net < 0:
        consistency = negative / len(diffs)
        direction = "decreasing" if consistency >= 0.8 else "volatile_decreasing"
    else:
        consistency = positive / len(diffs)
        direction = "increasing" if consistency >= 0.8 else "volatile_increasing"
    return {
        "direction": direction,
        "positive_steps": positive,
        "negative_steps": negative,
        "flat_steps": flat,
        "consistency": consistency,
    }


def _best_of(stats: Dict[str, Any], goal: Optional[str]) -> Tuple[Any, Any]:
    if goal == "min":
        return stats.get("min"), stats.get("min_step")
    if goal == "max":
        return stats.get("max"), stats.get("max_step")
    return None, None


def _regression(stats: Dict[str, Any], goal: Optional[str]) -> Optional[float]:
    """How far the last value fell back from the best, as a share of the range."""
    best, _ = _best_of(stats, goal)
    lo, hi, last = stats.get("min"), stats.get("max"), stats.get("last")
    if best is None or lo is None or hi is None or last is None or hi <= lo:
        return None
    return abs(last - best) / (hi - lo)


def summarize_stats(stats: Dict[str, Any], goal: Optional[str]) -> Dict[str, Any]:
    """Flat metric summary from :meth:`Database.get_scalar_stats` aggregates."""
    best, best_step = _best_of(stats, goal)
    out = {
        "goal": goal,
        "n": stats.get("n"),
        "last": stats.get("last"),
        "last_step": stats.get("last_step"),
        "min": stats.get("min"),
        "min_step": stats.get("min_step"),
        "max": stats.get("max"),
        "max_step": stats.get("max_step"),
        "best": best,
        "best_step": best_step,
    }
    reg = _regression(stats, goal)
    if reg is not None:
        out["regression"] = round(reg, 4)
    return out


def summarize_series(rows: Sequence[Dict[str, Any]], goal: Optional[str]) -> Dict[str, Any]:
    """Summarise one scalar series in O(n): extremes, best, trend, samples."""
    clean: List[Tuple[int, float]] = []
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append((row["step"], value))
    if not clean:
        return {"goal": goal, "n": 0}
    values = [v for _, v in clean]
    n = len(clean)
    i_min = min(range(n), key=values.__getitem__)
    i_max = max(range(n), key=values.__getitem__)
    stats = {
        "n": n,
        "first_step": clean[0][0],
        "last_step": clean[-1][0],
        "last": values[-1],
        "min": values[i_min],
        "min_step": clean[i_min][0],
        "max": values[i_max],
        "max_step": clean[i_max][0],
    }
    out = summarize_stats(stats, goal)
    out["first"] = values[0]
    out["first_step"] = clean[0][0]
    out["trend"] = classify_trend(values)["direction"]
    if n >= 6:
        w = max(1, n // 10)
        recent = sum(values[-w:]) / w
        before = sum(values[-2 * w : -w]) / w
        span = stats["max"] - stats["min"]
        rel = (recent - before) / span if span > 0 else 0.0
        if abs(rel) < 0.02:
            label = "flat"
        elif goal is None:
            label = "rising" if rel > 0 else "falling"
        else:
            label = "improving" if (rel > 0) == (goal == "max") else "worsening"
        out["recent"] = label
    picks = sorted({round(i * (n - 1) / 5) for i in range(6)}) if n > 1 else [0]
    out["samples"] = [[clean[i][0], values[i]] for i in picks]
    return out


def _metric_brief(tag: str, m: Dict[str, Any]) -> str:
    """``val/acc 0.912 (best 0.915 @900)`` — best shown only when it differs."""
    text = f"{tag} {fmt_value(m.get('last'))}"
    best = m.get("best")
    if best is not None and best != m.get("last"):
        text += f" (best {fmt_value(best)} @{m.get('best_step')})"
    return text


# ── Media ────────────────────────────────────────────────────────


def _latest(rows: Sequence[Dict[str, Any]], path_key: str = "abs_path") -> Dict[str, Any]:
    latest = max(rows, key=lambda r: (r.get("step") or 0))
    return {
        "count": len(rows),
        "first_step": min(r.get("step") or 0 for r in rows),
        "last_step": latest.get("step"),
        "latest": latest.get(path_key) or latest.get("path"),
    }


def latest_media(exp: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{kind: {tag: {count, first_step, last_step, latest}}}`` per media kind."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    loaders = (
        ("video", exp.all_video),
        ("image", exp.all_images),
        ("audio", exp.all_audio),
    )
    for kind, loader in loaders:
        for tag, rows in sorted(loader().items()):
            if rows:
                out.setdefault(kind, {})[tag] = _latest(rows)
    try:
        for tag in exp.figure_tags():
            rows = exp.figures(tag)
            if rows:
                out.setdefault("figure", {})[tag] = _latest(rows)
        for tag in exp.user_artifact_tags():
            rows = exp.artifacts(tag)
            if rows:
                out.setdefault("artifact", {})[tag] = _latest(rows)
    except Exception:
        pass
    return out


def _media_counts(media: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, int]:
    return {kind: sum(t["count"] for t in tags.values()) for kind, tags in media.items()}


# ── Runs listing ─────────────────────────────────────────────────


def run_row(exp: Any, now: Optional[float] = None, headline_limit: int = 3) -> Dict[str, Any]:
    """One compact row per run: status, timing, headline metrics, media counts."""
    st = run_status(exp, now)
    declared = declared_metrics(exp.run_info())
    tags = [t for t in exp.scalar_tags() if not is_system_tag(t)]
    head = headline_tags(tags, declared, headline_limit)
    stats = exp.scalar_stats(tags=head) if head else {}
    headline = []
    for tag in head:
        if tag in stats:
            headline.append({"tag": tag, **summarize_stats(stats[tag], goal_for(tag, declared))})
    media = latest_media(exp)
    row = {
        "name": exp.name,
        "project": exp.project,
        "experiment_id": exp.experiment_id,
        "status": st["status"],
        "stale": st.get("stale", False),
        "heartbeat": st.get("heartbeat"),
        "started_at": st.get("started_at"),
        "ended_at": st.get("ended_at"),
        "duration_s": st.get("duration_s"),
        "last_step": st.get("last_step"),
        "headline": headline,
        "n_metrics": len(tags),
        "media": _media_counts(media),
        "latest_media": {
            kind: {tag: info["latest"] for tag, info in tags_.items()}
            for kind, tags_ in media.items()
            if kind in ("video", "image")
        },
    }
    if st.get("error"):
        row["error"] = st["error"]
    if st.get("git"):
        row["git"] = st["git"]
    return row


def list_runs(exps: Sequence[Any], now: Optional[float] = None, headline_limit: int = 3) -> List[Dict[str, Any]]:
    return [run_row(exp, now, headline_limit) for exp in exps]


def run_card(exp: Any) -> Dict[str, Any]:
    """Payload for the web dashboard's Runs tab."""
    row = run_row(exp, headline_limit=4)
    st = run_status(exp)
    if st.get("command"):
        row["command"] = st["command"]
    return row


def _git_short(git: Optional[Dict[str, Any]]) -> str:
    if not git or not git.get("commit"):
        return ""
    text = git["commit"][:8]
    if git.get("dirty"):
        text += "+dirty"
    return text


def format_runs(
    rows: Sequence[Dict[str, Any]],
    project_folder: Optional[str] = None,
    now: Optional[float] = None,
) -> str:
    if not rows:
        return "No runs found." + _empty_hint(project_folder)
    projects = list(dict.fromkeys(r["project"] for r in rows))
    single = len(projects) == 1
    head = f"# vibetrack runs ({len(rows)})"
    if single:
        head += f" - project `{projects[0]}`"
    lines = [head, ""]
    cols = ([] if single else ["project"]) + ["run", "status", "started", "duration", "step", "media", "metrics (last, best @step)"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        media = ", ".join(f"{k} x{v}" for k, v in r.get("media", {}).items() if k in ("video", "image", "audio"))
        metrics = "; ".join(_metric_brief(m["tag"], m) for m in r.get("headline", []))
        if r.get("n_metrics", 0) > len(r.get("headline", [])):
            metrics += f" (+{r['n_metrics'] - len(r['headline'])})"
        cells = ([] if single else [r["project"]]) + [
            r["name"],
            _status_text(r),
            fmt_age(r.get("started_at"), now),
            fmt_duration(r.get("duration_s")),
            fmt_value(r.get("last_step")),
            media or "-",
            metrics or "-",
        ]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    p = projects[0] if single else None
    lines += ["", _next_hint(
        ["vibetrack show RUN", "vibetrack compare", "vibetrack open RUN"],
        scope_args(p, project_folder) if single else scope_args(None, project_folder) + " -p PROJECT",
    )]
    return "\n".join(lines)


def _next_hint(commands: Sequence[str], scope: str) -> str:
    """``Next: `a` | `b``` plus the shared flags, stated once."""
    text = "Next: " + " | ".join(f"`{c}`" for c in commands)
    if scope.strip():
        text += f"  (add `{scope.strip()}`)"
    return text


def _empty_hint(project_folder: Optional[str]) -> str:
    if project_folder:
        return f" (database: {project_folder}/vibetrack.db)"
    return " (central database ~/.vibetrack/vibetrack.db; pass --project-folder DIR for a project-local one)"


# ── Single run ───────────────────────────────────────────────────


def run_overview(exp: Any, include_system: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
    """Everything an agent needs to judge one run, without raw series."""
    st = run_status(exp, now)
    declared = declared_metrics(exp.run_info())
    all_scalars = exp.all_scalars()
    tags = rank_tags(all_scalars.keys(), declared)
    metrics = {tag: summarize_series(all_scalars[tag], goal_for(tag, declared)) for tag in tags}
    system = sorted(t for t in all_scalars if is_system_tag(t))
    if include_system:
        for tag in system:
            metrics[tag] = summarize_series(all_scalars[tag], None)
    texts: Dict[str, Any] = {}
    for tag, rows in sorted(exp.all_texts().items()):
        if not rows:
            continue
        latest = max(rows, key=lambda r: r.get("step") or 0)
        body = str(latest.get("value", ""))
        texts[tag] = {
            "count": len(rows),
            "last_step": latest.get("step"),
            "latest": body if len(body) <= 600 else body[:600] + "...",
        }
    return {
        "name": exp.name,
        "project": exp.project,
        "experiment_id": exp.experiment_id,
        "log_dir": exp.log_dir,
        "run": st,
        "config": flatten_params(exp.hparams()),
        "declared_metrics": declared,
        "metrics": metrics,
        "system_metrics": system,
        "media": latest_media(exp),
        "texts": texts,
        "dashboard": dashboard_url(exp.project, [exp.name]),
    }


def format_overview(ov: Dict[str, Any], project_folder: Optional[str] = None, now: Optional[float] = None) -> str:
    st = ov["run"]
    scope = scope_args(ov["project"], project_folder)
    lines = [f"# Run `{ov['name']}` - project `{ov['project']}` - {_status_text(st, 200)}", ""]
    timing = [f"started {fmt_time(st.get('started_at'))} ({fmt_age(st.get('started_at'), now)})"]
    timing.append(f"duration {fmt_duration(st.get('duration_s'))}")
    if st.get("last_step") is not None:
        timing.append(f"last step {st['last_step']}")
    lines.append("- " + ", ".join(timing))
    if st.get("command"):
        lines.append(f"- command: `{st['command']}`" + (f" (cwd {st['cwd']})" if st.get("cwd") else ""))
    git = st.get("git") or {}
    if git.get("commit"):
        g = f"- git: {git['commit'][:10]}"
        if git.get("branch"):
            g += f" on {git['branch']}"
        if git.get("dirty"):
            g += f", uncommitted changes in {git.get('changed_files', '?')} file(s)"
        lines.append(g)
    if st.get("host"):
        lines.append(f"- host {st['host']}, pid {st.get('pid', '-')}")
    lines.append(f"- log_dir: {ov.get('log_dir') or '-'}")
    if st.get("traceback"):
        tail = "\n".join(st["traceback"].rstrip().splitlines()[-12:])
        lines += ["", "Traceback (tail):", "```", tail, "```"]

    cfg = ov.get("config") or {}
    lines += ["", f"## Config ({len(cfg)})"]
    if cfg:
        lines.append(", ".join(f"{k}={fmt_value(v)}" for k, v in sorted(cfg.items())))
    else:
        lines.append("(none logged - pass config={...} to vibetrack.init or SummaryWriter)")

    metrics = ov.get("metrics") or {}
    lines += ["", f"## Metrics ({len(metrics)})"]
    if metrics:
        lines.append("| metric | goal | n | last @step | best @step | min | max | trend | recent |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for tag, m in metrics.items():
            if not m.get("n"):
                continue
            best = f"{fmt_value(m.get('best'))} @{m.get('best_step')}" if m.get("best") is not None else "-"
            if m.get("regression", 0) and m["regression"] >= 0.2:
                best += f" (last is {int(m['regression'] * 100)}% of range off best)"
            lines.append(
                "| " + " | ".join(
                    _cell(c)
                    for c in (
                        tag,
                        m.get("goal") or "?",
                        m.get("n"),
                        f"{fmt_value(m.get('last'))} @{m.get('last_step')}",
                        best,
                        m.get("min"),
                        m.get("max"),
                        m.get("trend", "-"),
                        m.get("recent", "-"),
                    )
                ) + " |"
            )
        lines.append("")
        lines.append("Curves (step:value samples):")
        for tag in list(metrics)[:6]:
            samples = metrics[tag].get("samples") or []
            if samples:
                lines.append(f"- {tag}: " + " -> ".join(f"{s}:{fmt_value(v)}" for s, v in samples))
    else:
        lines.append("(no scalar metrics)")
    if ov.get("system_metrics"):
        lines.append(f"System metrics: {len(ov['system_metrics'])} tags (use --system to include)")

    media = ov.get("media") or {}
    if media:
        lines += ["", "## Media (latest file per tag)", "| kind | tag | count | steps | latest file |", "|---|---|---|---|---|"]
        for kind, tags in media.items():
            for tag, info in tags.items():
                lines.append(
                    "| " + " | ".join(
                        _cell(c)
                        for c in (kind, tag, info["count"], f"{info['first_step']}..{info['last_step']}", info.get("latest") or "-")
                    ) + " |"
                )
    texts = ov.get("texts") or {}
    if texts:
        lines += ["", "## Text (latest)"]
        for tag, info in texts.items():
            lines.append(f"- `{tag}` @{info['last_step']} ({info['count']} entries): {info['latest']}")
    lines += [
        "",
        f"Dashboard: {ov.get('dashboard')}",
        _next_hint([f"vibetrack open {_q(ov['name'])}", "vibetrack compare"], scope),
    ]
    return "\n".join(lines)


# ── Comparison ───────────────────────────────────────────────────


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def hparam_effects(
    varying: Dict[str, Dict[str, Any]],
    scores: Dict[str, float],
    metric: str,
    goal: str,
) -> List[Dict[str, Any]]:
    """Rough per-parameter association with the primary metric (heuristic)."""
    effects: List[Dict[str, Any]] = []
    for key, by_run in varying.items():
        pairs = [(by_run[r], s) for r, s in scores.items() if r in by_run and by_run[r] is not None]
        if len(pairs) < 3:
            continue
        values = [p[0] for p in pairs]
        if all(_is_number(v) for v in values) and len(set(values)) >= 3 and len(pairs) >= 4:
            rho = spearman([float(v) for v in values], [p[1] for p in pairs])
            if rho is None or abs(rho) < 0.3:
                continue
            better = (rho > 0) == (goal == "max")
            effects.append(
                {
                    "param": key,
                    "kind": "numeric",
                    "spearman": round(rho, 3),
                    "n": len(pairs),
                    "summary": f"higher {key} -> {'better' if better else 'worse'} {metric} (Spearman {rho:+.2f}, n={len(pairs)})",
                }
            )
            continue
        groups: Dict[str, List[float]] = {}
        labels: Dict[str, Any] = {}
        for v, s in pairs:
            k = _value_key(v)
            groups.setdefault(k, []).append(s)
            labels[k] = v
        if not 2 <= len(groups) <= 8 or len(groups) == len(pairs):
            continue
        means = sorted(
            ((sum(g) / len(g), k, len(g)) for k, g in groups.items()),
            reverse=(goal == "max"),
        )
        spread = abs(means[0][0] - means[-1][0])
        effects.append(
            {
                "param": key,
                "kind": "categorical",
                "groups": [{"value": labels[k], "mean": m, "n": c} for m, k, c in means],
                "spread": spread,
                "summary": f"{key}: "
                + " > ".join(f"{fmt_value(labels[k])} {fmt_value(m)} (n={c})" for m, k, c in means),
            }
        )
    effects.sort(key=lambda e: -abs(e.get("spearman", 0.0)) if e["kind"] == "numeric" else 0.0)
    return effects


def _labels(exps: Sequence[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    for exp in exps:
        counts[exp.name] = counts.get(exp.name, 0) + 1
    return [f"{exp.project}/{exp.name}" if counts[exp.name] > 1 else exp.name for exp in exps]


def compare_runs(
    exps: Sequence[Any],
    metric: Optional[str] = None,
    goal: Optional[str] = None,
    include_system: bool = False,
    now: Optional[float] = None,
    focus: bool = True,
) -> Dict[str, Any]:
    """Rank runs, split params into varying/constant and tabulate metrics.

    *focus* puts the compared runs in the dashboard link (``?run=``); pass
    ``False`` when comparing a whole project.
    """
    now = now or time.time()
    labels = _labels(exps)
    runs: List[Dict[str, Any]] = []
    declared: Dict[str, Dict[str, Any]] = {}
    for label, exp in zip(labels, exps):
        st = run_status(exp, now)
        decl = declared_metrics(exp.run_info())
        for tag, entry in decl.items():
            declared.setdefault(tag, entry)
        stats = {
            t: s for t, s in exp.scalar_stats().items() if include_system or not is_system_tag(t)
        }
        media = latest_media(exp)
        runs.append(
            {
                "label": label,
                "name": exp.name,
                "project": exp.project,
                "experiment_id": exp.experiment_id,
                "status": st["status"],
                "stale": st.get("stale", False),
                "heartbeat": st.get("heartbeat"),
                "error": st.get("error"),
                "started_at": st.get("started_at"),
                "duration_s": st.get("duration_s"),
                "last_step": st.get("last_step"),
                "command": st.get("command"),
                "git": _git_short(st.get("git")),
                "params": flatten_params(exp.hparams()),
                "_stats": stats,
                "media": {
                    kind: {tag: {"step": i["last_step"], "path": i["latest"], "count": i["count"]} for tag, i in tags.items()}
                    for kind, tags in media.items()
                    if kind in ("video", "image", "audio")
                },
            }
        )

    primary, pgoal, reason = pick_primary([set(r["_stats"]) for r in runs], declared, metric, goal)

    ranking: List[Dict[str, Any]] = []
    if primary:
        for r in runs:
            st = r["_stats"].get(primary)
            if not st:
                continue
            best, best_step = _best_of(st, pgoal)
            ranking.append({"run": r["label"], "best": best, "best_step": best_step, "last": st.get("last"), "last_step": st.get("last_step")})
        ranking.sort(key=lambda x: (x["best"] if pgoal == "min" else -x["best"]))
        if ranking:
            top = ranking[0]["best"]
            for i, item in enumerate(ranking):
                item["rank"] = i + 1
                item["delta"] = item["best"] - top
                if top > 0 and item["best"] > 0:
                    item["delta_pct"] = round((item["best"] - top) / top * 100.0, 2)

    keys = sorted({k for r in runs for k in r["params"]})
    varying: Dict[str, Dict[str, Any]] = {}
    constant: Dict[str, Any] = {}
    for key in keys:
        present = {r["label"]: r["params"][key] for r in runs if key in r["params"]}
        distinct = {_value_key(v) for v in present.values()}
        if len(present) == len(runs) and len(distinct) == 1:
            constant[key] = next(iter(present.values()))
        else:
            varying[key] = present

    tag_order = rank_tags([t for r in runs for t in r["_stats"]], declared)
    if primary in tag_order:
        tag_order.remove(primary)
        tag_order.insert(0, primary)
    metrics: Dict[str, Any] = {}
    for tag in tag_order:
        g = pgoal if tag == primary else goal_for(tag, declared)
        per_run = {r["label"]: summarize_stats(r["_stats"][tag], g) for r in runs if tag in r["_stats"]}
        metrics[tag] = {"goal": g, "runs": per_run}

    effects: List[Dict[str, Any]] = []
    if primary and pgoal and len(ranking) >= 3:
        scores = {item["run"]: item["best"] for item in ranking}
        effects = hparam_effects(varying, scores, primary, pgoal)

    for r in runs:
        r.pop("_stats", None)
        r.pop("params", None)
    project_names = list(dict.fromkeys(r["project"] for r in runs))
    return {
        "project": project_names[0] if len(project_names) == 1 else None,
        "projects": project_names,
        "runs": runs,
        "primary": {"metric": primary, "goal": pgoal, "reason": reason},
        "ranking": ranking,
        "params": {"varying": varying, "constant": constant},
        "metrics": metrics,
        "effects": effects,
        "dashboard": dashboard_url(
            project_names[0] if len(project_names) == 1 else None,
            [r["name"] for r in runs][:12] if focus else [],
        ),
    }


def format_compare(cmp: Dict[str, Any], project_folder: Optional[str] = None, max_rows: int = 20) -> str:
    runs = cmp["runs"]
    if not runs:
        return "No runs to compare."
    by_label = {r["label"]: r for r in runs}
    prim = cmp["primary"]
    metric, goal = prim["metric"], prim["goal"]
    proj = cmp.get("project")
    head = f"# Compare {len(runs)} runs" + (f" - project `{proj}`" if proj else "")
    lines = [head]
    if metric:
        lines.append(f"Ranked by **{metric}** ({goal}, best value; {prim['reason']})")
    else:
        lines.append(f"Not ranked: {prim['reason']}")
    lines.append("")

    varying = cmp["params"]["varying"]
    vkeys = list(varying)[:8]
    others = [t for t in cmp["metrics"] if t != metric and _tag_score(t, {}) > -4][:2]
    cols = ["#", "run", "status"] + vkeys
    if metric:
        cols += [f"{metric} best @step", "last"]
    cols += [f"{t} last" for t in others] + ["time"]
    lines.append("| " + " | ".join(_cell(c) for c in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    rank_of = {item["run"]: item for item in cmp["ranking"]}
    ordered = [item["run"] for item in cmp["ranking"]] + [r["label"] for r in runs if r["label"] not in rank_of]
    for label in ordered[:max_rows]:
        r = by_label[label]
        item = rank_of.get(label)
        cells: List[str] = [str(item["rank"]) if item else "-", label, _status_text(r, 40)]
        cells += [fmt_value(varying[k].get(label)) if label in varying[k] else "-" for k in vkeys]
        if metric:
            if item:
                best = f"{fmt_value(item['best'])} @{item['best_step']}"
                if item["rank"] > 1 and item.get("delta_pct") is not None:
                    best += f" ({item['delta_pct']:+.1f}%)"
                elif item["rank"] > 1 and item.get("delta"):
                    sign = "+" if item["delta"] > 0 else "-"
                    best += f" ({sign}{fmt_value(abs(item['delta']))})"
                last = fmt_value(item["last"])
                summ = cmp["metrics"].get(metric, {}).get("runs", {}).get(label, {})
                if summ.get("regression", 0) >= 0.2:
                    last += f" (fell back {int(summ['regression'] * 100)}% of range)"
                cells += [best, last]
            else:
                cells += ["-", "-"]
        for t in others:
            m = cmp["metrics"][t]["runs"].get(label)
            cells.append(fmt_value(m["last"]) if m else "-")
        cells.append(fmt_duration(r.get("duration_s")))
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    if len(ordered) > max_rows:
        lines.append(f"(+{len(ordered) - max_rows} more runs; use --json for all)")
    if cmp["ranking"]:
        leader = by_label[cmp["ranking"][0]["run"]]
        if leader["status"] not in ("finished", "running"):
            done = next(
                (i for i in cmp["ranking"] if by_label[i["run"]]["status"] == "finished"), None
            )
            note = f"Note: leader {leader['label']} {leader['status']} (its best value came before that)"
            if done:
                note += f"; best finished run: {done['run']} ({fmt_value(done['best'])})"
            lines.append(note)
    if len(varying) > len(vkeys):
        lines.append("More varying params: " + ", ".join(list(varying)[len(vkeys):]))
    constant = cmp["params"]["constant"]
    if constant:
        items = [f"{k}={fmt_value(v)}" for k, v in constant.items()]
        shown = items[:20]
        lines.append("")
        lines.append("Same in all runs: " + ", ".join(shown) + (f" (+{len(items) - 20} more)" if len(items) > 20 else ""))
    if not varying and not constant:
        lines.append("")
        lines.append("No params logged - pass config={...} to vibetrack.init / SummaryWriter so runs can be compared.")

    if cmp.get("effects"):
        lines += ["", f"## Param effects on {metric} (heuristic, small samples - verify before trusting)"]
        lines += [f"- {e['summary']}" for e in cmp["effects"][:8]]

    labels = ordered[:8]
    tags = list(cmp["metrics"])[:12]
    if tags and len(labels) > 1:
        lines += ["", "## Metrics (last; best @step when different)"]
        lines.append("| metric | goal | " + " | ".join(_cell(lbl) for lbl in labels) + " |")
        lines.append("|" + "|".join("---" for _ in range(len(labels) + 2)) + "|")
        for tag in tags:
            info = cmp["metrics"][tag]
            cells = [tag, info["goal"] or "?"]
            for label in labels:
                m = info["runs"].get(label)
                if not m:
                    cells.append("-")
                    continue
                text = fmt_value(m.get("last"))
                if m.get("best") is not None and m["best"] != m.get("last"):
                    text += f" [{fmt_value(m['best'])} @{m['best_step']}]"
                cells.append(text)
            lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
        if len(cmp["metrics"]) > len(tags):
            lines.append(f"(+{len(cmp['metrics']) - len(tags)} more metrics; use --json)")

    media_lines = []
    for label in ordered[:max_rows]:
        r = by_label[label]
        for kind in ("video", "image", "audio"):
            for tag, info in (r.get("media") or {}).get(kind, {}).items():
                media_lines.append(f"- {label}: {kind} `{tag}` x{info['count']}, latest @{info['step']}: {info['path']}")
    if media_lines:
        lines += ["", "## Media (latest file per tag)"] + media_lines[:30]

    code = [f"- {r['label']}: {r['git'] or 'no git'}" + (f", `{r['command']}`" if r.get("command") else "") for r in runs if r.get("git") or r.get("command")]
    if code:
        lines += ["", "## Code"] + code[:max_rows]
    scope = scope_args(proj, project_folder)
    lines += [
        "",
        f"Dashboard: {cmp['dashboard']}  (start/open it with `vibetrack open{scope}`)",
    ]
    return "\n".join(lines)


# ── End-of-run report ────────────────────────────────────────────


def _error_location(tb: str) -> str:
    """``path.py:12 in fn: <source line>`` for the innermost traceback frame."""
    lines = tb.rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r'\s*File "([^"]+)", line (\d+), in (.+)', lines[i])
        if m:
            where = f"{m.group(1)}:{m.group(2)} in {m.group(3)}"
            if i + 1 < len(lines) and not lines[i + 1].lstrip().startswith(("File ", "^", "~")):
                code = lines[i + 1].strip()
                if code and not re.match(r"^\w+(Error|Exception|Interrupt)\b", code):
                    where += f": {code}"
            return where
    return ""


def end_of_run_report(db: Any, experiment_id: int, project_folder: Optional[str] = None) -> str:
    """Compact digest printed to stderr when a run closes."""
    from .reader import ExperimentReader

    row = db.get_experiment(experiment_id)
    if row is None:
        return ""
    exp = ExperimentReader(db, experiment_id, row["name"], row=row)
    st = run_status(exp)
    declared = declared_metrics(exp.run_info())
    tags = [t for t in exp.scalar_tags() if not is_system_tag(t)]
    media = latest_media(exp)
    texts = exp.text_tags()
    if not tags and not media and not texts:
        return ""
    name, project = exp.name, exp.project
    scope = scope_args(project, project_folder)

    status = st["status"]
    dur = fmt_duration(st.get("duration_s"))
    if status == "finished":
        what = f"finished in {dur}"
    elif status in ("failed", "crashed"):
        what = f"FAILED after {dur}: {(st.get('error') or '').splitlines()[0] if st.get('error') else ''}".rstrip(": ")
    elif status == "interrupted":
        what = f"interrupted after {dur}"
    else:
        what = f"closed after {dur}"
    first = f"== vibetrack: run {name!r} {what} | project {project!r}"
    if st.get("last_step") is not None:
        first += f" | step {st['last_step']}"
    lines = [first]

    cfg = flatten_params(exp.hparams())
    if cfg:
        parts, used = [], 0
        for k, v in cfg.items():
            item = f"{k}={fmt_value(v)}"
            if used + len(item) > 110 and parts:
                break
            parts.append(item)
            used += len(item) + 1
        extra = len(cfg) - len(parts)
        lines.append("config:  " + " ".join(parts) + (f" (+{extra})" if extra else ""))
    else:
        lines.append("config:  (none logged; pass config={...} so runs can be compared)")

    head = headline_tags(tags, declared, 3)
    stats = exp.scalar_stats(tags=head) if head else {}
    if head:
        briefs = [_metric_brief(t, summarize_stats(stats[t], goal_for(t, declared))) for t in head if t in stats]
        more = len(tags) - len(head)
        lines.append("metrics: " + "; ".join(briefs) + (f" (+{more} more)" if more > 0 else ""))
    if media:
        bits = []
        for kind in ("video", "image", "audio", "figure", "artifact"):
            for tag, info in (media.get(kind) or {}).items():
                text = f"{kind} {tag} x{info['count']}"
                if kind == "video" or (kind == "image" and "video" not in media):
                    text += f" (latest: {info['latest']})"
                bits.append(text)
        lines.append("media:   " + "; ".join(bits[:4]) + (f" (+{len(bits) - 4})" if len(bits) > 4 else ""))

    primary, pgoal, _ = pick_primary([set(tags)], declared)
    if primary and pgoal:
        ranked = []
        for other in db.list_experiments(project=project)[:50]:
            ost = db.get_scalar_stats(other["id"], tags=[primary]).get(primary)
            if ost:
                best, _step = _best_of(ost, pgoal)
                ranked.append((best, other["name"], other["id"]))
        if len(ranked) >= 2:
            ranked.sort(key=lambda x: x[0] if pgoal == "min" else -x[0])
            pos = next((i for i, x in enumerate(ranked) if x[2] == experiment_id), None)
            if pos is not None:
                text = f"rank:    #{pos + 1} of {len(ranked)} runs by {primary} ({pgoal}, best value)"
                if pos == 0:
                    text += f"; next: {ranked[1][1]} {fmt_value(ranked[1][0])}"
                else:
                    text += f"; leader: {ranked[0][1]} {fmt_value(ranked[0][0])}"
                lines.append(text)

    if st.get("traceback") and status == "failed":
        where = _error_location(st["traceback"])
        if where:
            lines.append(f"error:   at {where}")
    lines.append(f"view:    vibetrack open {_q(name)}{scope}")
    lines.append(f"         (dashboard: {dashboard_url(project, [name])})")
    lines.append(
        f"inspect: vibetrack show {_q(name)}  |  vibetrack compare"
        + (f"  (same flags: {scope.strip()})" if scope else "")
    )
    return "\n".join(lines)
