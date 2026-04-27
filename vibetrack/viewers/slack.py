"""Slack webhook output — push experiment events to a Slack channel.

Requires: ``pip install vibetrack[slack]``

Usage::

    from vibetrack import SummaryWriter
    writer = SummaryWriter("runs/exp1")
    writer = writer.to("slack", webhook="https://hooks.slack.com/services/…")
    writer.add_scalar("loss", 0.3, step=5)

Slack incoming webhooks can only POST JSON — file upload is not available.
Media events are reported by tag/path; attach them manually or run the web
dashboard to serve them.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, Sequence
from urllib import request as _urlreq

from .base import BaseOutput

_log = logging.getLogger(__name__)


class SlackOutput(BaseOutput):
    """Forward log events to Slack via an incoming webhook."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        webhook: Optional[str] = None,
    ) -> None:
        super().__init__(project_folder, project=project)
        self.webhook = webhook or os.environ.get("SLACK_WEBHOOK_URL", "")

    def show(self, **kwargs: Any) -> Any:
        """Slack has no pull-dashboard form; see :meth:`send`."""
        _log.info(
            "SlackOutput.show is a no-op; use writer.to('slack') for per-event dispatch."
        )
        return None

    def send(self, events: Sequence[Any]) -> None:
        if not events:
            return
        if not self.webhook:
            _log.warning(
                "SlackOutput.send skipped: SLACK_WEBHOOK_URL or webhook= must be set."
            )
            return
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
            elif ev.kind in ("image", "audio", "video", "artifact"):
                lines.append(f"• {ev.kind} `{ev.tag}`{step_s} — `{ev.value}`")
            elif ev.kind == "histogram":
                lines.append(f"• histogram `{ev.tag}`{step_s}")
            elif ev.kind == "hparams":
                hp = ev.value if isinstance(ev.value, dict) else {}
                pairs = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:8])
                lines.append(f"• hparams: {pairs}")
            else:
                lines.append(f"• {ev.kind} `{ev.tag}`{step_s}")

        payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
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
