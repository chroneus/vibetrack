"""Slack output — push experiment events to a Slack channel.

Two modes:

* **Webhook (text only).** Set ``SLACK_WEBHOOK_URL`` (or ``webhook=``).
  Slack incoming webhooks cannot upload files, so media events are
  reported as paths only.

* **Bot token (text + media).** Set ``SLACK_BOT_TOKEN`` (xoxb-…) and
  ``SLACK_CHANNEL`` (channel ID like ``C0123ABCD``, or a name like
  ``#general`` which is resolved via ``conversations.list``). Required
  bot scopes: ``chat:write``, ``files:write``, and ``channels:read``
  (only if you pass a channel name instead of an ID). Media files
  attach inline via ``files.upload_v2`` — one Slack post per ``send()``
  batch with the summary as the message body.

Usage::

    from vibetrack import SummaryWriter
    writer = SummaryWriter("runs/exp1").to("slack", every=10000)
    writer.add_scalar("loss", 0.3, step=5)
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import parse as _urlparse
from urllib import request as _urlreq

from .base import BaseOutput

_log = logging.getLogger(__name__)

_SLACK_API = "https://slack.com/api"
_MEDIA_KINDS = ("image", "audio", "video", "artifact")


class SlackOutput(BaseOutput):
    """Forward log events to Slack via webhook or bot-token API."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        webhook: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> None:
        super().__init__(project_folder, project=project)
        self.webhook = webhook or os.environ.get("SLACK_WEBHOOK_URL", "")
        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self.channel = channel or os.environ.get("SLACK_CHANNEL", "")
        self._channel_id: Optional[str] = None

    def show(self, **kwargs: Any) -> Any:
        """Pull-mode: summarize the most recent experiment in this project."""
        exps = self._reader.experiments()
        if not exps:
            _log.info("SlackOutput.show: no experiments to summarize.")
            return None
        latest = exps[-1]
        self.send_summary(latest.name, latest.project or None)
        return None

    def send_summary(self, run_name: str, project: Optional[str] = None) -> None:
        """Push a one-post summary for *run_name*: text body + chart PNGs +
        stitched image series + latest video/audio."""
        exp = next(
            (e for e in self._reader.experiments() if e.name == run_name),
            None,
        )
        if exp is None:
            print(
                f"vibetrack slack: run {run_name!r} not found in DB — nothing to summarize.",
                file=sys.stderr,
            )
            return

        text_body = _build_summary_text(exp, project=project or exp.project)
        attachments = _collect_summary_attachments(exp)

        if self.bot_token:
            try:
                self._post_summary_via_api(text_body, attachments)
                print(
                    f"vibetrack slack: posted summary for {run_name!r} "
                    f"({len(attachments)} attachments) via bot token.",
                    file=sys.stderr,
                )
                return
            except Exception as exc:
                print(
                    f"vibetrack slack: bot-token post failed ({exc}); "
                    f"falling back to webhook.",
                    file=sys.stderr,
                )
        if self.webhook:
            self._send_via_webhook(text_body)
            print(
                f"vibetrack slack: posted text summary for {run_name!r} via webhook "
                f"(media is text-only — set SLACK_BOT_TOKEN + channel ID for inline files).",
                file=sys.stderr,
            )
            return
        print(
            "vibetrack slack: NO post sent — set SLACK_BOT_TOKEN + SLACK_CHANNEL "
            "(channel ID like C0123ABCD) or SLACK_WEBHOOK_URL.",
            file=sys.stderr,
        )

    def _post_summary_via_api(
        self,
        text_body: str,
        attachments: List[Tuple[str, bytes, str]],
    ) -> None:
        """Upload all summary attachments + post under one Slack message."""
        channel_id = self._resolve_channel()
        if not channel_id:
            raise RuntimeError("SLACK_CHANNEL not set or could not be resolved")

        if not attachments:
            self._api_post(
                "chat.postMessage", {"channel": channel_id, "text": text_body}
            )
            return

        file_ids: List[Dict[str, str]] = []
        for filename, blob, title in attachments:
            file_id = self._upload_bytes(filename, blob)
            if file_id:
                file_ids.append({"id": file_id, "title": title})

        if not file_ids:
            self._api_post(
                "chat.postMessage", {"channel": channel_id, "text": text_body}
            )
            return

        self._api_post_json(
            "files.completeUploadExternal",
            {
                "files": file_ids,
                "channel_id": channel_id,
                "initial_comment": text_body,
            },
        )

    # ── public dispatch ─────────────────────────────────────────────

    def send(self, events: Sequence[Any]) -> None:
        if not events:
            return
        summary = self._build_summary(events)
        media = [
            ev
            for ev in events
            if ev.kind in _MEDIA_KINDS
            and isinstance(ev.value, str)
            and Path(ev.value).is_file()
        ]
        if self.bot_token:
            try:
                self._send_via_api(summary, media)
                return
            except Exception as exc:
                _log.warning("Slack API send failed (%s); falling back to webhook", exc)
        if self.webhook:
            self._send_via_webhook(summary)
            return
        _log.warning(
            "SlackOutput.send skipped: set SLACK_BOT_TOKEN+SLACK_CHANNEL or SLACK_WEBHOOK_URL."
        )

    # ── summary builder ─────────────────────────────────────────────

    def _build_summary(self, events: Sequence[Any]) -> str:
        run_header = events[0].run_name
        if events[0].project:
            run_header = f"{events[0].project}/{run_header}"

        lines: List[str] = [f"*{run_header}*"]
        for ev in events:
            step_s = f" _step={ev.step}_" if ev.step is not None else ""
            if ev.kind == "scalar":
                lines.append(f"• `{ev.tag}` = {ev.value:.6g}{step_s}")
            elif ev.kind == "text":
                body = str(ev.value)
                if len(body) > 400:
                    body = body[:400] + "…"
                lines.append(f"• *{ev.tag}*{step_s}\n> {body}")
            elif ev.kind in _MEDIA_KINDS:
                lines.append(f"• {ev.kind} `{ev.tag}`{step_s}")
            elif ev.kind == "histogram":
                lines.append(f"• histogram `{ev.tag}`{step_s}")
            elif ev.kind == "hparams":
                hp = ev.value if isinstance(ev.value, dict) else {}
                pairs = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:8])
                lines.append(f"• hparams: {pairs}")
            else:
                lines.append(f"• {ev.kind} `{ev.tag}`{step_s}")
        return "\n".join(lines)

    # ── webhook path ────────────────────────────────────────────────

    def _send_via_webhook(self, text: str) -> None:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = _urlreq.Request(
            self.webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _urlreq.urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    _log.warning("Slack webhook returned %s", resp.status)
        except Exception as exc:
            _log.warning("Slack webhook failed: %s", exc)

    # ── bot-token path ──────────────────────────────────────────────

    def _send_via_api(self, summary: str, media: List[Any]) -> None:
        channel_id = self._resolve_channel()
        if not channel_id:
            raise RuntimeError("SLACK_CHANNEL not set or could not be resolved")

        if not media:
            self._api_post("chat.postMessage", {"channel": channel_id, "text": summary})
            return

        # Upload all media files, then finalize them in one batch
        # with the summary as the initial_comment so everything appears
        # under a single Slack post.
        file_ids: List[Dict[str, str]] = []
        for ev in media:
            file_id = self._upload_file(ev.value, title=f"{ev.tag} (step={ev.step})")
            if file_id:
                file_ids.append({"id": file_id, "title": f"{ev.tag} (step={ev.step})"})

        if not file_ids:
            self._api_post("chat.postMessage", {"channel": channel_id, "text": summary})
            return

        self._api_post_json(
            "files.completeUploadExternal",
            {
                "files": file_ids,
                "channel_id": channel_id,
                "initial_comment": summary,
            },
        )

    def _resolve_channel(self) -> Optional[str]:
        if self._channel_id:
            return self._channel_id
        ch = self.channel.strip()
        if not ch:
            _log.warning(
                "SlackOutput: SLACK_CHANNEL not set — cannot use bot-token path."
            )
            return None
        # Looks like an ID already (C…, G…, D… plus alphanumeric tail).
        if ch[:1] in ("C", "G", "D") and ch[1:].isalnum() and len(ch) >= 9:
            self._channel_id = ch
            return ch
        # Treat as name; query conversations.list (needs channels:read scope).
        name = ch.lstrip("#")
        cursor = ""
        for _ in range(10):
            params: Dict[str, Any] = {
                "limit": 1000,
                "types": "public_channel,private_channel",
            }
            if cursor:
                params["cursor"] = cursor
            try:
                resp = self._api_post("conversations.list", params)
            except RuntimeError as exc:
                if "missing_scope" in str(exc):
                    _log.warning(
                        "SlackOutput: bot lacks channels:read scope to look up "
                        "%r by name. Pass the channel ID instead (e.g. C0123ABCD), "
                        "or add channels:read + groups:read scopes and reinstall.",
                        self.channel,
                    )
                else:
                    _log.warning("SlackOutput: conversations.list failed: %s", exc)
                return None
            for c in resp.get("channels", []):
                if c.get("name") == name:
                    self._channel_id = c.get("id")
                    return self._channel_id
            cursor = resp.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
        _log.warning("Slack channel %r not found", self.channel)
        return None

    def _upload_file(self, path: str, title: str) -> Optional[str]:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            _log.warning("Slack upload skipped — cannot read %s: %s", path, exc)
            return None
        return self._upload_bytes(Path(path).name, data)

    def _upload_bytes(self, filename: str, data: bytes) -> Optional[str]:
        """Two-step files.upload_v2: get URL, POST bytes; return ``file_id``."""
        step1 = self._api_post(
            "files.getUploadURLExternal",
            {"filename": filename, "length": len(data)},
        )
        upload_url = step1.get("upload_url")
        file_id = step1.get("file_id")
        if not upload_url or not file_id:
            _log.warning("Slack getUploadURLExternal failed: %s", step1)
            return None
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body, content_type = _multipart_encode(filename, data, ctype)
        req = _urlreq.Request(
            upload_url,
            data=body,
            headers={"Content-Type": content_type},
        )
        try:
            with _urlreq.urlopen(req, timeout=60) as resp:
                if resp.status >= 400:
                    _log.warning("Slack upload PUT returned %s", resp.status)
                    return None
        except Exception as exc:
            _log.warning("Slack upload PUT failed for %s: %s", filename, exc)
            return None
        return file_id

    # ── HTTP helpers ────────────────────────────────────────────────

    def _api_post(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        data = _urlparse.urlencode(
            {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in params.items()
            }
        ).encode("utf-8")
        req = _urlreq.Request(
            f"{_SLACK_API}/{method}",
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "Authorization": f"Bearer {self.bot_token}",
            },
        )
        return self._read_json(req, method)

    def _api_post_json(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = _urlreq.Request(
            f"{_SLACK_API}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.bot_token}",
            },
        )
        return self._read_json(req, method)

    @staticmethod
    def _read_json(req: _urlreq.Request, method: str) -> Dict[str, Any]:
        with _urlreq.urlopen(req, timeout=15) as resp:
            body = resp.read()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError(f"Slack {method} returned non-JSON: {body[:200]!r}")
        if not data.get("ok", False):
            raise RuntimeError(f"Slack {method} error: {data.get('error', 'unknown')}")
        return data


# ── module-level helpers ────────────────────────────────────────────


_SUMMARY_TEXT_LIMIT = 2800  # Slack initial_comment soft cap (3000 hard)
_SYSTEM_TAG_PREFIXES = ("system/", "gpu/")  # auto-collected; usually noise in summaries


def _is_system_tag(tag: str) -> bool:
    return any(tag.startswith(p) for p in _SYSTEM_TAG_PREFIXES)


def _sanitize_filename(name: str, fallback: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    safe = safe.strip("._")
    return safe or fallback


def _build_summary_text(exp: Any, project: Optional[str]) -> str:
    """Compose the `initial_comment` body: scalars (final), hparams, text entries."""
    header = exp.name
    if project:
        header = f"{project}/{header}"
    lines: List[str] = [f"*{header}*"]

    hp = {}
    try:
        hp = exp.hparams() or {}
    except Exception:
        hp = {}
    if hp:
        flat = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:12])
        lines.append(f"*hparams* {flat}")

    user_scalars = [t for t in exp.scalar_tags() if not _is_system_tag(t)]
    for tag in user_scalars:
        rows = exp.scalars(tag)
        if not rows:
            continue
        last = max(rows, key=lambda r: r["step"])
        lines.append(f"• `{tag}` final = {last['value']:.6g} _step={last['step']}_")

    for tag in exp.text_tags():
        rows = exp.texts(tag)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["step"])
        lines.append(f"*{tag}*")
        for r in rows:
            body = str(r["value"])
            if len(body) > 400:
                body = body[:400] + "…"
            lines.append(f"_step={r['step']}_  {body}")

    text = "\n".join(lines)
    if len(text) > _SUMMARY_TEXT_LIMIT:
        text = text[:_SUMMARY_TEXT_LIMIT] + "…"
    return text


def _collect_summary_attachments(exp: Any) -> List[Tuple[str, bytes, str]]:
    """Return ``[(filename, blob, title), …]`` for the summary upload batch."""
    from ..compare import compare_scalars
    from ._summary import render_scalar_chart_png, stitch_images_to_mp4

    out: List[Tuple[str, bytes, str]] = []

    # Scalar charts — one PNG per user-logged tag (skip auto system metrics)
    user_scalars = [t for t in exp.scalar_tags() if not _is_system_tag(t)]
    for tag in user_scalars:
        try:
            comparison = compare_scalars([exp], tag)
            png = render_scalar_chart_png(comparison, tag)
        except Exception as exc:
            _log.warning("chart render failed for %s: %s", tag, exc)
            continue
        if png:
            out.append((_sanitize_filename(f"{tag}.png", "chart.png"), png, tag))

    # Image series — stitched mp4 if multi-frame, otherwise a single PNG
    for tag in exp.image_tags():
        try:
            entries = sorted(exp.images(tag), key=lambda r: r["step"])
        except Exception as exc:
            _log.warning("read images failed for %s: %s", tag, exc)
            continue
        if not entries:
            continue
        if len(entries) == 1:
            path = entries[0]["abs_path"]
            try:
                blob = Path(path).read_bytes()
                out.append(
                    (Path(path).name, blob, f"{tag} (step={entries[0]['step']})")
                )
            except OSError:
                pass
            continue
        mp4 = stitch_images_to_mp4([e["abs_path"] for e in entries])
        if mp4:
            out.append(
                (_sanitize_filename(f"{tag}.mp4", "series.mp4"), mp4, f"{tag} (series)")
            )
        else:  # fallback: latest frame as PNG
            latest = entries[-1]
            try:
                blob = Path(latest["abs_path"]).read_bytes()
                out.append(
                    (
                        Path(latest["abs_path"]).name,
                        blob,
                        f"{tag} (step={latest['step']})",
                    )
                )
            except OSError:
                pass

    # Latest video / audio per tag
    for kind, tags, getter in (
        ("video", exp.video_tags(), exp.video),
        ("audio", exp.audio_tags(), exp.audio),
    ):
        for tag in tags:
            try:
                entries = getter(tag)
            except Exception as exc:
                _log.warning("read %s failed for %s: %s", kind, tag, exc)
                continue
            if not entries:
                continue
            latest = max(entries, key=lambda r: r["step"])
            try:
                blob = Path(latest["abs_path"]).read_bytes()
            except OSError:
                continue
            out.append(
                (Path(latest["abs_path"]).name, blob, f"{tag} (step={latest['step']})")
            )

    # Models: latest rendered PNG diagram per tag
    for tag in exp.model_tags():
        try:
            entries = exp.models(tag)
        except Exception as exc:
            _log.warning("read models failed for %s: %s", tag, exc)
            continue
        if not entries:
            continue
        latest = max(entries, key=lambda r: r["step"])
        png_path = latest.get("rendered_png_abs")
        if not png_path:
            continue
        try:
            blob = Path(png_path).read_bytes()
        except OSError:
            continue
        out.append((Path(png_path).name, blob, f"model {tag} (step={latest['step']})"))

    return out


def _multipart_encode(
    filename: str, file_bytes: bytes, content_type: str
) -> Tuple[bytes, str]:
    """Encode a single-file multipart/form-data body. Returns (body, ctype)."""
    boundary = "----vibetrack-" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts: List[bytes] = []
    parts.append(f"--{boundary}".encode())
    parts.append(f'Content-Disposition: form-data; name="filename"'.encode())
    parts.append(b"")
    parts.append(filename.encode())
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode()
    )
    parts.append(f"Content-Type: {content_type}".encode())
    parts.append(b"")
    parts.append(file_bytes)
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"
