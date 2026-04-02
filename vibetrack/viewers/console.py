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

    def show(self, **kwargs: Any) -> str:
        tags: Optional[Sequence[str]] = kwargs.get("tags")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        cfg = load_config(self.config_project())
        smoothing: str = kwargs.get("smoothing", cfg.get("smoothing", "ema"))
        smooth_weight: float = kwargs.get("smooth_weight", cfg.get("smooth_weight", 0.6))
        exps = self._resolve_experiments(experiments)
        if not exps:
            return "No experiments found."

        if tags is None:
            tags = find_all_tags(exps)

        lines: List[str] = []
        lines.append(f"{'Experiment':<25} {'Tag':<25} {'Last':>10} {'Min':>10} {'Max':>10}  Trend")
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
            n_art = sum(len(exp.artifacts(t)) for t in exp.artifact_tags())
            if n_img + n_aud + n_vid + n_art > 0:
                media_rows.append((exp.name, n_img, n_aud, n_vid, n_art))
        if media_rows:
            lines.append("")
            lines.append(
                f"{'Experiment':<25} {'Images':>8} {'Audio':>8} {'Video':>8} {'Artifacts':>10}"
            )
            lines.append("─" * 70)
            for name, ni, na, nv, nf in media_rows:
                lines.append(f"{name:<25} {ni:>8} {na:>8} {nv:>8} {nf:>10}")

        output = "\n".join(lines)
        print(output, file=sys.stdout)
        return output

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
