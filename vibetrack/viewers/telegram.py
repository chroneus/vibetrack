"""Telegram bot output — send experiment charts and summaries to a Telegram chat.

Requires: ``pip install vibetrack[telegram]``

Usage::

    from vibetrack.viewers.telegram import TelegramOutput
    out = TelegramOutput("project/", token="BOT_TOKEN", chat_id="CHAT_ID")
    out.show()  # sends summary + chart images to the chat
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, List, Optional, Sequence

from ..compare import compare_scalars, find_all_tags, summary_table
from ..config import load_config
from ..smoother import smooth
from .base import BaseOutput

_log = logging.getLogger(__name__)


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
        project: Optional[str] = None,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        super().__init__(project_folder, project=project)
        self.token = token or os.environ.get("VIBETRACK_TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("VIBETRACK_TELEGRAM_CHAT_ID", "")

    # ── Push: per-event dispatch via ``writer.to("telegram")`` ──
    def send(self, events: Sequence[Any]) -> None:
        """Forward a batch of :class:`LogEvent` objects to the chat.

        Scalars/text are merged into one text message.  Images/audio/video/
        artifacts are uploaded as individual files.  Histograms/hparams are
        summarised textually.  Credentials must be configured; if missing, the
        batch is silently dropped with a warning.
        """
        if not events:
            return
        if not self.token or not self.chat_id:
            _log.warning(
                "TelegramOutput.send skipped: VIBETRACK_TELEGRAM_TOKEN "
                "and VIBETRACK_TELEGRAM_CHAT_ID must both be set."
            )
            return
        import asyncio

        from telegram import Bot

        bot = Bot(token=self.token)

        text_lines: List[str] = []
        media_events: List[Any] = []
        for ev in events:
            kind = ev.kind
            step_s = f" step={ev.step}" if ev.step is not None else ""
            if kind == "scalar":
                text_lines.append(f"`{ev.tag}`={ev.value:.6g}{step_s}")
            elif kind == "text":
                body = str(ev.value)
                if len(body) > 500:
                    body = body[:500] + "…"
                text_lines.append(f"*{ev.tag}*{step_s}\n{body}")
            elif kind in ("image", "audio", "video", "artifact"):
                media_events.append(ev)
            elif kind == "histogram":
                text_lines.append(f"`{ev.tag}` histogram{step_s}")
            elif kind == "hparams":
                hp = ev.value if isinstance(ev.value, dict) else {}
                pairs = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:8])
                text_lines.append(f"*hparams* {pairs}")
            else:
                text_lines.append(f"`{ev.tag}` ({kind}){step_s}")

        run_header = events[0].run_name
        if events[0].project:
            run_header = f"{events[0].project}/{run_header}"

        TELEGRAM_MEDIA_LIMIT = 50 * 1024 * 1024

        async def _send_file(ev: Any) -> None:
            path = ev.value
            if not isinstance(path, str):
                return
            try:
                size = os.path.getsize(path)
                fh = open(path, "rb")
            except OSError:
                return
            caption = f"{run_header} / {ev.tag} — step {ev.step}"
            try:
                if size > TELEGRAM_MEDIA_LIMIT or ev.kind == "artifact":
                    await bot.send_document(
                        chat_id=self.chat_id, document=fh, caption=caption
                    )
                elif ev.kind == "image":
                    await bot.send_photo(
                        chat_id=self.chat_id, photo=fh, caption=caption
                    )
                elif ev.kind == "audio":
                    await bot.send_audio(
                        chat_id=self.chat_id, audio=fh, caption=caption
                    )
                elif ev.kind == "video":
                    await bot.send_video(
                        chat_id=self.chat_id, video=fh, caption=caption
                    )
            except Exception as exc:
                _log.warning("Telegram %s failed for %s: %s", ev.kind, path, exc)
            finally:
                fh.close()

        async def _run() -> None:
            if text_lines:
                msg = f"*{run_header}*\n" + "\n".join(text_lines)
                # Telegram messages max 4096 chars
                if len(msg) > 4000:
                    msg = msg[:4000] + "…"
                try:
                    await bot.send_message(
                        chat_id=self.chat_id, text=msg, parse_mode="Markdown"
                    )
                except Exception as exc:
                    _log.warning("Telegram text send failed: %s", exc)
            for ev in media_events:
                await _send_file(ev)

        try:
            asyncio.run(_run())
        except Exception as exc:
            _log.warning("Telegram delivery failed: %s", exc)

    def show(self, **kwargs: Any) -> Any:
        if not self.token or not self.chat_id:
            _log.warning(
                "TelegramOutput skipped: VIBETRACK_TELEGRAM_TOKEN and "
                "VIBETRACK_TELEGRAM_CHAT_ID must both be set."
            )
            return None
        tags: Optional[Sequence[str]] = kwargs.get("tags")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        cfg = load_config(self.config_project())
        smoothing: str = kwargs.get("smoothing", cfg.get("smoothing", "ema"))
        smooth_weight: float = kwargs.get(
            "smooth_weight", cfg.get("smooth_weight", 0.6)
        )
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

        TELEGRAM_MEDIA_LIMIT = 50 * 1024 * 1024

        async def _send_file(path: str, kind: str, caption: str) -> None:
            try:
                size = os.path.getsize(path)
            except OSError:
                return
            try:
                fh = open(path, "rb")
            except OSError:
                return
            try:
                if size > TELEGRAM_MEDIA_LIMIT:
                    await bot.send_document(
                        chat_id=self.chat_id, document=fh, caption=caption
                    )
                elif kind == "photo":
                    await bot.send_photo(
                        chat_id=self.chat_id, photo=fh, caption=caption
                    )
                elif kind == "audio":
                    await bot.send_audio(
                        chat_id=self.chat_id, audio=fh, caption=caption
                    )
                elif kind == "video":
                    await bot.send_video(
                        chat_id=self.chat_id, video=fh, caption=caption
                    )
            except Exception as exc:
                _log.warning("Telegram %s failed for %s: %s", kind, path, exc)
            finally:
                fh.close()

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
            for exp in exps:
                for kind, list_tags, get_entries in (
                    ("photo", exp.image_tags(), exp.images),
                    ("audio", exp.audio_tags(), exp.audio),
                    ("video", exp.video_tags(), exp.video),
                ):
                    for media_tag in list_tags:
                        entries = get_entries(media_tag)
                        if not entries:
                            continue
                        latest = max(entries, key=lambda e: e["step"])
                        await _send_file(
                            latest["abs_path"],
                            kind,
                            f"{exp.name} / {media_tag} — step {latest['step']}",
                        )
                for txt_tag in exp.text_tags():
                    entries = exp.texts(txt_tag)
                    if not entries:
                        continue
                    latest = max(entries, key=lambda e: e["step"])
                    body = str(latest["value"])
                    if len(body) > 3500:
                        body = body[:3500] + "…"
                    try:
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=f"*{exp.name} / {txt_tag}* — step {latest['step']}\n\n{body}",
                            parse_mode="Markdown",
                        )
                    except Exception as exc:
                        _log.warning(
                            "Telegram text failed for %s/%s: %s",
                            exp.name,
                            txt_tag,
                            exc,
                        )

        try:
            asyncio.run(_send())
        except Exception as exc:
            _log.warning("Telegram delivery failed: %s", exc)
            return None
        return text
