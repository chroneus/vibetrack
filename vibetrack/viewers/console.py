"""Console/terminal output with Unicode sparkline charts."""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Sequence

from ..compare import compare_scalars, find_all_tags, summary_table
from ..config import load_config
from ..smoother import smooth
from .base import BaseOutput

# Unicode block elements for sparklines
_BARS = " ▁▂▃▄▅▆▇█"

_SYSTEM_TAG_PREFIXES = ("system/", "gpu/")


def _is_system_tag(tag: str) -> bool:
    return any(tag.startswith(p) for p in _SYSTEM_TAG_PREFIXES)


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    # Down-sample to `width` points if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = list(values)
    chars = []
    for v in sampled:
        idx = int((v - mn) / rng * (len(_BARS) - 1))
        chars.append(_BARS[idx])
    return "".join(chars)


class ConsoleOutput(BaseOutput):
    """Print experiment summaries and sparkline charts to the terminal."""

    def send(self, events: Sequence[Any]) -> None:
        """Print a one-line summary per event — used by ``writer.to("console")``."""
        for ev in events:
            step_s = f"step={ev.step}" if ev.step is not None else "—"
            if ev.kind == "scalar":
                line = f"[{step_s}] {ev.tag}={ev.value:.6g}"
            elif ev.kind == "text":
                body = str(ev.value)
                if len(body) > 160:
                    body = body[:160] + "…"
                line = f"[{step_s}] {ev.tag}: {body}"
            elif ev.kind in ("image", "audio", "video", "artifact"):
                line = f"[{step_s}] {ev.kind} {ev.tag} -> {ev.value}"
            elif ev.kind == "histogram":
                line = f"[{step_s}] histogram {ev.tag}"
            elif ev.kind == "hparams":
                hp = ev.value if isinstance(ev.value, dict) else {}
                line = f"hparams: {hp}"
            else:
                line = f"[{step_s}] {ev.kind} {ev.tag}"
            print(line, file=sys.stdout, flush=True)

    def show(self, **kwargs: Any) -> str:
        tags: Optional[Sequence[str]] = kwargs.get("tags")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        cfg = load_config(self.config_project())
        smoothing: str = kwargs.get("smoothing", cfg.get("smoothing", "ema"))
        smooth_weight: float = kwargs.get(
            "smooth_weight", cfg.get("smooth_weight", 0.6)
        )
        exps = self._resolve_experiments(experiments)
        if not exps:
            return "No experiments found."

        if tags is None:
            tags = find_all_tags(exps)

        lines: List[str] = []
        lines.append(
            f"{'Experiment':<25} {'Tag':<25} {'Last':>10} {'Min':>10} {'Max':>10}  Trend"
        )
        lines.append("─" * 120)

        for exp in exps:
            for tag in tags:
                data = exp.scalars(tag)
                if not data:
                    continue
                values = [d["value"] for d in data]
                if smoothing != "none":
                    # The web UI already treats the slider as EMA-based smoothing.
                    # Keep the console aligned with that behavior so configs using
                    # non-EMA labels do not break rendering.
                    display = smooth(values, method="ema", weight=smooth_weight)
                else:
                    display = values
                spark = _sparkline(display)
                last = values[-1]
                mn, mx = min(values), max(values)
                lines.append(
                    f"{exp.name:<25} {tag:<25} {last:>10.4f} {mn:>10.4f} {mx:>10.4f}  {spark}"
                )

        # ── Media summary ─────────────────────────────────────────
        media_rows = []
        for exp in exps:
            n_img = sum(len(exp.images(t)) for t in exp.image_tags())
            n_aud = sum(len(exp.audio(t)) for t in exp.audio_tags())
            n_vid = sum(len(exp.video(t)) for t in exp.video_tags())
            # Models and PR curves share the artifacts table but have their
            # own conceptual home — count separately so they don't get
            # silently lumped into "Artifacts".
            n_mod = len(exp.model_tags())
            n_pr = len(exp.pr_curve_tags())
            n_art = sum(len(exp.artifacts(t)) for t in exp.user_artifact_tags())
            if n_img + n_aud + n_vid + n_art + n_mod + n_pr > 0:
                media_rows.append((exp.name, n_img, n_aud, n_vid, n_art, n_mod, n_pr))
        if media_rows:
            lines.append("")
            lines.append(
                f"{'Experiment':<25} {'Images':>8} {'Audio':>8} {'Video':>8} "
                f"{'Artifacts':>10} {'Models':>8} {'PR':>5}"
            )
            lines.append("─" * 84)
            for name, ni, na, nv, nf, nm, np_ in media_rows:
                lines.append(
                    f"{name:<25} {ni:>8} {na:>8} {nv:>8} {nf:>10} {nm:>8} {np_:>5}"
                )

        output = "\n".join(lines)
        print(output, file=sys.stdout)
        return output

    def _build_run_report(self, exp: Any, project: Optional[str] = None) -> str:
        """Compose the text digest for one experiment. Pure — no I/O."""
        header_name = f"{project}/{exp.name}" if project else exp.name
        lines: List[str] = [f"=== vibetrack run summary: {header_name} ==="]

        hp = exp.hparams() or {}
        if hp:
            lines.append("")
            lines.append("Hyperparameters:")
            for k, v in hp.items():
                lines.append(f"  {k} = {v}")

        user_tags = [t for t in exp.scalar_tags() if not _is_system_tag(t)]
        if user_tags:
            lines.append("")
            lines.append("Scalars:")
            for tag in user_tags:
                data = exp.scalars(tag)
                if not data:
                    continue
                values = [d["value"] for d in data]
                last, mn, mx = values[-1], min(values), max(values)
                mean = sum(values) / len(values)
                spark = _sparkline(values)
                lines.append(
                    f"  {tag:<28} last={last:.4g}  min={mn:.4g}  "
                    f"max={mx:.4g}  mean={mean:.4g}  {spark}"
                )

        text_tags = exp.text_tags()
        if text_tags:
            lines.append("")
            lines.append("Text entries:")
            for tag in text_tags:
                entries = exp.texts(tag)
                if not entries:
                    continue
                lines.append(f"  [{tag}]")
                for e in sorted(entries, key=lambda x: x["step"] or 0):
                    body = str(e["value"])
                    if len(body) > 500:
                        body = body[:500] + "…"
                    step = e["step"] if e["step"] is not None else "—"
                    lines.append(f"    step={step}: {body}")

        n_img = sum(len(exp.images(t)) for t in exp.image_tags())
        n_aud = sum(len(exp.audio(t)) for t in exp.audio_tags())
        n_vid = sum(len(exp.video(t)) for t in exp.video_tags())
        user_art_tags = exp.user_artifact_tags()
        n_art = sum(len(exp.artifacts(t)) for t in user_art_tags)
        model_tags = exp.model_tags()
        n_mod = sum(len(exp.models(t)) for t in model_tags)
        pr_tags = exp.pr_curve_tags()
        n_pr = sum(len(exp.pr_curves(t)) for t in pr_tags)
        n_hist = sum(len(exp.histograms(t)) for t in exp.histogram_tags())
        if any((n_img, n_aud, n_vid, n_art, n_mod, n_pr, n_hist)):
            lines.append("")
            lines.append("Media:")
            if n_img:
                lines.append(f"  images:     {n_img:>4}  tags={exp.image_tags()}")
            if n_aud:
                lines.append(f"  audio:      {n_aud:>4}  tags={exp.audio_tags()}")
            if n_vid:
                lines.append(f"  video:      {n_vid:>4}  tags={exp.video_tags()}")
            if n_art:
                lines.append(f"  artifacts:  {n_art:>4}  tags={user_art_tags}")
            if n_mod:
                lines.append(f"  models:     {n_mod:>4}  tags={model_tags}")
            if n_pr:
                lines.append(f"  pr_curves:  {n_pr:>4}  tags={pr_tags}")
            if n_hist:
                lines.append(f"  histograms: {n_hist:>4}  tags={exp.histogram_tags()}")

        return "\n".join(lines)

    def send_summary(self, run_name: str, project: Optional[str] = None) -> None:
        """Print a text-only end-of-run digest for one experiment."""
        exp = next(
            (e for e in self._reader.experiments() if e.name == run_name),
            None,
        )
        if exp is None:
            print(
                f"vibetrack console: run {run_name!r} not found in DB.",
                file=sys.stderr,
            )
            return
        report = self._build_run_report(exp, project)
        print(report, file=sys.stdout, flush=True)

    def summary(
        self,
        experiments: Optional[Sequence[str]] = None,
    ) -> str:
        exps = self._resolve_experiments(experiments)
        if not exps:
            return "No experiments found."

        table = summary_table(exps)
        if not table:
            return "No data."

        # Collect all keys
        keys = [k for k in table[0] if k not in ("name", "experiment_id")]
        header = f"{'Experiment':<25}" + "".join(f" {k:>12}" for k in keys)
        lines = [header, "─" * len(header)]
        for row in table:
            vals = "".join(
                f" {row[k]:>12.4f}" if row[k] is not None else f" {'N/A':>12}"
                for k in keys
            )
            lines.append(f"{row['name']:<25}{vals}")

        output = "\n".join(lines)
        print(output, file=sys.stdout)
        return output
