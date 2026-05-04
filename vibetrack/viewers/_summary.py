"""Shared helpers for end-of-run summary rendering (Slack, Telegram, …).

All optional deps (matplotlib, imageio) are imported lazily; missing deps
return empty bytes so callers can degrade gracefully.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

_log = logging.getLogger(__name__)


# Suffix → (display unit, precision-style) for system/gpu metric formatting.
# Order matters: longest match wins ("_used_percent" before "_percent").
_UNIT_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("_gb", "Gb"),
    ("_percent", "%"),
    ("_c", "°C"),
    ("_count", "_count"),  # sentinel: integer, no unit
)


def format_metric(tag: str, value: float) -> Tuple[str, str]:
    """Pretty-print a (tag, value) pair for chat summaries.

    Strips known unit suffixes from *tag* and appends a human-readable unit to
    *value*, with precision tuned per unit:

    >>> format_metric("gpu/0/memory_total_gb", 24.0)
    ('gpu/0/memory_total', '24 Gb')
    >>> format_metric("gpu/0/memory_used_percent", 4.0080)
    ('gpu/0/memory_used', '4.0 %')
    >>> format_metric("system/cpu_count", 32.0)
    ('system/cpu_count', '32')
    >>> format_metric("train/loss", 0.123456)
    ('train/loss', '0.1235')
    """
    for suffix, unit in _UNIT_SUFFIXES:
        if not tag.endswith(suffix):
            continue
        display_tag = tag[: -len(suffix)] if unit != "_count" else tag
        if unit == "_count":
            return display_tag, f"{int(round(value))}"
        if unit == "%":
            return display_tag, f"{value:.1f} %"
        if unit == "°C":
            return display_tag, f"{int(round(value))} °C"
        if unit == "Gb":
            # Render as integer only when the value is genuinely whole
            # (24.0 → "24 Gb"), not merely near-whole (0.96 → "0.96 Gb").
            if abs(value - round(value)) < 0.005:
                return display_tag, f"{int(round(value))} Gb"
            return display_tag, f"{value:.2f} Gb"
    # No unit suffix matched → integer if exact, else 4 sig figs
    if isinstance(value, (int, float)) and abs(value - round(value)) < 1e-9:
        return tag, f"{int(round(value))}"
    return tag, f"{value:.4g}"


def render_scalar_chart_png(comparison_data: list, tag: str) -> bytes:
    """Render a scalar comparison as a PNG. Returns b'' if matplotlib missing.

    *comparison_data* is a list of dicts shaped like
    ``{"name": str, "steps": [...], "values": [...], "smoothed": [...]}``
    — same shape ``compare_scalars`` produces.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not installed — skipping chart for %s", tag)
        return b""

    fig, ax = plt.subplots(figsize=(8, 4))
    for entry in comparison_data:
        values = entry.get("smoothed", entry["values"])
        ax.plot(entry["steps"], values, label=entry["name"])
    ax.set_title(tag)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_single_series_png(steps: List[int], values: List[float], tag: str) -> bytes:
    """Render one scalar series (no comparison overlay) as a PNG."""
    return render_scalar_chart_png(
        [{"name": tag, "steps": steps, "values": values}], tag
    )


def stitch_images_to_mp4(image_paths: List[str], fps: int = 4) -> bytes:
    """Concatenate images into an MP4. Returns b'' on failure or <2 frames."""
    paths = [p for p in image_paths if p and Path(p).is_file()]
    if len(paths) < 2:
        return b""
    try:
        import imageio.v2 as imageio
    except ImportError:
        _log.warning("imageio not installed — cannot stitch images to mp4")
        return b""

    try:
        frames: List[Any] = [imageio.imread(p) for p in paths]
    except Exception as exc:
        _log.warning("failed to read frames for stitching: %s", exc)
        return b""

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        imageio.mimwrite(tmp_path, frames, fps=fps, codec="libx264")
        data = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        return data
    except Exception as exc:
        _log.warning("mp4 stitch failed: %s", exc)
        return b""
