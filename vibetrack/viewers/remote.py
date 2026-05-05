"""Remote output — push log events to another vibetrack server.

Forwards every event kind to a peer running ``vibetrack --listen HOST:PORT``:

* scalars / texts / histograms → ``POST /log`` (JSON, one request per step)
* images / audio / video / artifacts → ``POST /media`` (multipart per file)
* hparams → ``POST /hparams`` (JSON)

Usage::

    import vibetrack
    writer = vibetrack.init(project="cifar10", name="resnet18")
    writer.to("remote", url="http://localhost:8080", token="devtoken", every="10m")

Local SQLite stays the source of truth — network failures log a single warning
and silently drop subsequent batches without ever blocking training.

Stdlib only (``urllib``); no extra runtime dependencies.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error as _urlerr
from urllib import request as _urlreq

from .base import BaseOutput
from .event import LogEvent

_log = logging.getLogger(__name__)

_MEDIA_KINDS = ("image", "audio", "video", "artifact")
_LOG_KINDS = ("scalar", "text", "histogram")


class RemoteOutput(BaseOutput):
    """Forward log events to a remote ``vibetrack --listen`` ingest server."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 10.0,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        super().__init__(project_folder=project_folder, project=project)
        self.url = (url or os.environ.get("VIBETRACK_REMOTE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("VIBETRACK_REMOTE_TOKEN") or None
        self.timeout = timeout
        self._degraded = False

    def show(self, **_kwargs: Any) -> Any:
        return None

    def send(self, events: Sequence[LogEvent]) -> None:
        if not events or self._degraded:
            return
        if not self.url:
            self._fail("RemoteOutput: no url set; pass url=… to .to() or set VIBETRACK_REMOTE_URL")
            return

        log_groups: Dict[Tuple[str, Optional[int]], Dict[str, Dict[str, Any]]] = {}
        for ev in events:
            if ev.kind not in _LOG_KINDS:
                continue
            key = (ev.run_name, ev.step)
            slot = log_groups.setdefault(
                key, {"scalars": {}, "texts": {}, "histograms": {}}
            )
            if ev.kind == "scalar":
                slot["scalars"][ev.tag] = float(ev.value)
            elif ev.kind == "text":
                slot["texts"][ev.tag] = str(ev.value)
            elif ev.kind == "histogram":
                payload = ev.value if isinstance(ev.value, dict) else {}
                bin_edges = list(payload.get("bin_edges", []))
                counts = list(payload.get("counts", []))
                if bin_edges and counts:
                    slot["histograms"][ev.tag] = {
                        "bin_edges": bin_edges,
                        "counts": counts,
                    }

        for (run_name, step), payload in log_groups.items():
            if not any(payload.values()):
                continue
            body = {"experiment": run_name, "step": step or 0, **payload}
            if not self._post_json("/log", body):
                return

        for ev in events:
            if ev.kind not in _MEDIA_KINDS:
                continue
            path = ev.value if isinstance(ev.value, str) else None
            if not path or not Path(path).is_file():
                continue
            if not self._post_media(ev, path):
                return

        for ev in events:
            if ev.kind != "hparams":
                continue
            metrics = (ev.extra or {}).get("metrics") or {}
            body = {
                "experiment": ev.run_name,
                "hparams": ev.value if isinstance(ev.value, dict) else {},
                "metrics": metrics,
            }
            if not self._post_json("/hparams", body):
                return

    def _post_json(self, path: str, body: Dict[str, Any]) -> bool:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = _urlreq.Request(self.url + path, data=data, headers=headers)
        try:
            with _urlreq.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    self._fail(f"POST {path} returned HTTP {resp.status}")
                    return False
        except (_urlerr.URLError, OSError) as exc:
            self._fail(f"POST {path} failed: {exc}")
            return False
        return True

    def _post_media(self, ev: LogEvent, path: str) -> bool:
        try:
            blob = Path(path).read_bytes()
        except OSError as exc:
            _log.warning("RemoteOutput: cannot read %s: %s", path, exc)
            return True
        filename = Path(path).name
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body, content_type = _multipart_encode(
            fields={
                "experiment": ev.run_name,
                "tag": ev.tag,
                "step": str(ev.step or 0),
                "type": ev.kind,
            },
            file_field="file",
            filename=filename,
            file_bytes=blob,
            file_ctype=ctype,
        )
        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = _urlreq.Request(self.url + "/media", data=body, headers=headers)
        try:
            with _urlreq.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    self._fail(f"POST /media returned HTTP {resp.status}")
                    return False
        except (_urlerr.URLError, OSError) as exc:
            self._fail(f"POST /media failed: {exc}")
            return False
        return True

    def _fail(self, msg: str) -> None:
        if not self._degraded:
            _log.warning("%s — disabling remote dispatch for this run.", msg)
            self._degraded = True


def _multipart_encode(
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_ctype: str,
) -> Tuple[bytes, str]:
    boundary = "----vibetrack-" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode()
    )
    parts.append(f"Content-Type: {file_ctype}".encode())
    parts.append(b"")
    parts.append(file_bytes)
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = crlf.join(parts)
    return body, f"multipart/form-data; boundary={boundary}"
