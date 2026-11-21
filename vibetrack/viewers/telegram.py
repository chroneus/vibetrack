"""Telegram bot output — send experiment charts and summaries to a Telegram chat.

Requires: ``pip install vibetrack[telegram]``

Usage::

    from vibetrack.viewers.telegram import TelegramOutput
    out = TelegramOutput("project/", token="BOT_TOKEN", chat_id="CHAT_ID")
    out.show()  # sends summary + chart images to the chat
"""

from __future__ import annotations

import html
import io
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from ..compare import compare_scalars, find_all_tags, summary_table
from ..config import load_config
from ..default_config import SMOOTH_WEIGHT
from ._summary import format_metric as _format_metric
from ._summary import render_scalar_chart_png as _render_chart_png
from .base import BaseOutput


_TEXT_LIMIT = 4000  # Telegram caps messages at 4096 characters


def _h(value: Any) -> str:
    """Escape a value for Telegram's HTML parse mode.

    Logged tags, run names and text routinely contain ``_`` or ``*``; the
    legacy Markdown mode rejects the whole message on unbalanced markers.
    """
    return html.escape(str(value), quote=False)


def _html_clip(text: str, limit: int) -> str:
    """Escape *text*, cut so the result fits *limit* without splitting ``&…;``."""
    escaped = _h(text)
    if len(escaped) <= limit:
        return escaped
    cut = escaped[:limit]
    amp = cut.rfind("&")
    if amp != -1 and ";" not in cut[amp:]:
        cut = cut[:amp]
    return cut + "…"


def _join_within(lines: Sequence[str], sep: str = "\n", limit: int = _TEXT_LIMIT) -> str:
    """Join whole lines up to *limit* chars, so no HTML tag is cut in half."""
    out: List[str] = []
    size = 0
    for line in lines:
        extra = len(line) + (len(sep) if out else 0)
        if size + extra > limit:
            out.append("…")
            break
        out.append(line)
        size += extra
    return sep.join(out)


def _import_telegram_bot() -> Any:
    """Import ``telegram.Bot`` with a helpful error if the dep is missing."""
    try:
        from telegram import Bot
    except ImportError as exc:
        raise ImportError(
            "The Telegram viewer requires the optional `python-telegram-bot` "
            "package. Install it with: pip install 'vibetrack[telegram]'"
        ) from exc
    return Bot


_log = logging.getLogger(__name__)


class TelegramOutput(BaseOutput):
    """Send experiment results to a Telegram chat."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        use_config_credentials: bool = False,
    ) -> None:
        super().__init__(project_folder, project=project)
        cfg: Dict[str, Any] = {}
        if use_config_credentials:
            raw = load_config(self.config_project()).get("telegram", {})
            cfg = raw if isinstance(raw, dict) else {}
        self.token = (
            token
            or os.environ.get("VIBETRACK_TELEGRAM_TOKEN", "")
            or str(cfg.get("token", "") or "")
        )
        self.chat_id = (
            chat_id
            or os.environ.get("VIBETRACK_TELEGRAM_CHAT_ID", "")
            or str(cfg.get("chat_id", "") or "")
        )

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

        Bot = _import_telegram_bot()

        bot = Bot(token=self.token)

        text_lines: List[str] = []
        media_events: List[Any] = []
        for ev in events:
            kind = ev.kind
            step_s = f" step={ev.step}" if ev.step is not None else ""
            if kind == "scalar":
                text_lines.append(f"<code>{_h(ev.tag)}</code>={ev.value:.6g}{step_s}")
            elif kind == "text":
                text_lines.append(
                    f"<b>{_h(ev.tag)}</b>{step_s}\n{_html_clip(str(ev.value), 500)}"
                )
            elif kind in ("image", "audio", "video", "artifact"):
                media_events.append(ev)
            elif kind == "histogram":
                text_lines.append(f"<code>{_h(ev.tag)}</code> histogram{step_s}")
            elif kind == "hparams":
                hp = ev.value if isinstance(ev.value, dict) else {}
                pairs = ", ".join(f"{_h(k)}={_h(v)}" for k, v in list(hp.items())[:8])
                text_lines.append(f"<b>hparams</b> {pairs}")
            else:
                text_lines.append(f"<code>{_h(ev.tag)}</code> ({_h(kind)}){step_s}")

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
                msg = _join_within([f"<b>{_h(run_header)}</b>", *text_lines])
                try:
                    await bot.send_message(
                        chat_id=self.chat_id, text=msg, parse_mode="HTML"
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
            "smooth_weight", cfg.get("smooth_weight", SMOOTH_WEIGHT)
        )
        import asyncio

        Bot = _import_telegram_bot()

        bot = Bot(token=self.token)

        exps = self._resolve_experiments(experiments)
        if tags is None:
            tags = find_all_tags(exps)

        # Build text summary — split user metrics from auto-collected
        # system/gpu metrics so the message stays scannable. System metrics
        # use unit-aware formatting (24 Gb, 4.0 %, 58 °C, …).
        table = summary_table(exps)
        lines = ["<b>vibetrack summary</b>"]
        for row in table:
            parts = [f"<code>{_h(row['name'])}</code>"]
            user_items: list[tuple[str, str]] = []
            sys_items: list[tuple[str, str]] = []
            for k, v in row.items():
                if k in ("name", "experiment_id") or v is None:
                    continue
                disp_tag, disp_val = _format_metric(k, v)
                if k.startswith("system/") or k.startswith("gpu/"):
                    sys_items.append((disp_tag, disp_val))
                else:
                    user_items.append((disp_tag, disp_val))
            for tag, val in user_items:
                parts.append(f"  <code>{_h(tag)}</code>: {_h(val)}")
            if sys_items:
                parts.append("  <i>system</i>:")
                for tag, val in sys_items:
                    parts.append(f"    <code>{_h(tag)}</code>: {_h(val)}")
            lines.append("\n".join(parts))
        text = _join_within(lines, sep="\n\n")

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
                parse_mode="HTML",
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
                # Models: upload the rendered PNG diagram (when present)
                # alongside chart media so it appears in the digest.
                for model_tag in exp.model_tags():
                    entries = exp.models(model_tag)
                    if not entries:
                        continue
                    latest = max(entries, key=lambda e: e["step"])
                    png_path = latest.get("rendered_png_abs")
                    if png_path:
                        await _send_file(
                            png_path,
                            "photo",
                            f"{exp.name} / model {model_tag} — step {latest['step']}",
                        )
                for txt_tag in exp.text_tags():
                    entries = exp.texts(txt_tag)
                    if not entries:
                        continue
                    latest = max(entries, key=lambda e: e["step"])
                    body = _html_clip(str(latest["value"]), 3500)
                    try:
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=(
                                f"<b>{_h(exp.name)} / {_h(txt_tag)}</b>"
                                f" — step {latest['step']}\n\n{body}"
                            ),
                            parse_mode="HTML",
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

    def send_summary(self, run_name: str, project: Optional[str] = None) -> None:
        """Send a single-run digest scoped to *run_name*.

        Reuses :meth:`show` (text + scalar charts + latest media + text
        entries) but limits the experiment set to one run and filters out
        auto-collected ``system/*`` and ``gpu/*`` scalars from the chart set.
        """
        if not self.token or not self.chat_id:
            print(
                "vibetrack telegram: NO post sent — set "
                "VIBETRACK_TELEGRAM_TOKEN and VIBETRACK_TELEGRAM_CHAT_ID.",
                file=sys.stderr,
            )
            return
        exp = next((e for e in self._reader.experiments() if e.name == run_name), None)
        if exp is None:
            print(
                f"vibetrack telegram: run {run_name!r} not found in DB.",
                file=sys.stderr,
            )
            return
        user_tags = [
            t
            for t in exp.scalar_tags()
            if not (t.startswith("system/") or t.startswith("gpu/"))
        ]
        self.show(experiments=[run_name], tags=user_tags)
        print(
            f"vibetrack telegram: posted summary for {run_name!r} "
            f"({len(user_tags)} chart(s)).",
            file=sys.stderr,
        )
