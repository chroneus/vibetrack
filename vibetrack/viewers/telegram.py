"""Telegram bot output — send experiment charts and summaries to a Telegram chat.

Requires: ``pip install vibetrack[telegram]``

Usage::

    from vibetrack.viewers.telegram import TelegramOutput
    out = TelegramOutput("project/", token="BOT_TOKEN", chat_id="CHAT_ID")
    out.show()  # sends summary + chart images to the chat
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional, Sequence

from ..compare import compare_scalars, find_all_tags, summary_table
from ..config import load_config
from ..smoother import smooth
from .base import BaseOutput


def _render_chart_png(
    comparison_data: list,
    tag: str,
) -> bytes:
    """Render a chart to PNG bytes using matplotlib (stdlib fallback: skip)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
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


class TelegramOutput(BaseOutput):
    """Send experiment results to a Telegram chat."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        super().__init__(project_folder)
        self.token = token or os.environ.get("VIBETRACK_TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("VIBETRACK_TELEGRAM_CHAT_ID", "")

    def show(self, **kwargs: Any) -> Any:
        tags: Optional[Sequence[str]] = kwargs.get("tags")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        cfg = load_config(self.config_project())
        smoothing: str = kwargs.get("smoothing", cfg.get("smoothing", "ema"))
        smooth_weight: float = kwargs.get("smooth_weight", cfg.get("smooth_weight", 0.6))
        import asyncio

        from telegram import Bot

        bot = Bot(token=self.token)

        exps = self._resolve_experiments(experiments)
        if tags is None:
            tags = find_all_tags(exps)

        # Build text summary
        table = summary_table(exps)
        lines = ["*vibetrack summary*\n"]
        for row in table:
            parts = [f"`{row['name']}`"]
            for k, v in row.items():
                if k in ("name", "experiment_id"):
                    continue
                if v is not None:
                    parts.append(f"  {k}: {v:.4f}")
            lines.append("\n".join(parts))
        text = "\n\n".join(lines)

        async def _send() -> None:
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            )
            for tag in tags:  # type: ignore[union-attr]
                comparison = compare_scalars(
                    exps, tag, smoothing=smoothing, weight=smooth_weight
                )
                png = _render_chart_png(comparison, tag)
                if png:
                    await bot.send_photo(
                        chat_id=self.chat_id,
                        photo=io.BytesIO(png),
                        caption=tag,
                    )

        asyncio.run(_send())
        return text
