"""Jupyter notebook viewer — embeds the web dashboard as a live iframe.

Usage::

    writer = SummaryWriter("runs/my_run").to("jupyter")
    writer = SummaryWriter("runs/my_run").to("jupyter", height=900, port=6117)
    writer = SummaryWriter("runs/my_run").to("jupyter", display_mode="link")

Outside a Jupyter kernel (plain terminal), falls back silently to the
blocking web viewer.
"""

from __future__ import annotations

import socket
import time
import threading
from typing import Any, List, Optional

from .base import BaseOutput
from .event import LogEvent


class JupyterOutput(BaseOutput):
    """Display vibetrack as a live iframe inside a Jupyter notebook."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        port: int = 0,
        height: int = 700,
        display_mode: str = "iframe",
        **kwargs: Any,
    ) -> None:
        super().__init__(project_folder=project_folder, project=project)
        self._host = "127.0.0.1"
        self._port = port
        self._height = height
        self._display_mode = display_mode
        self._server_thread: Optional[threading.Thread] = None
        self._server_url: Optional[str] = None
        self._display_handle: Any = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Called by SummaryWriter right after .to() registration."""
        self._ensure_server()
        if self._is_notebook():
            self._render()

    def show(self, **kwargs: Any) -> None:
        """Start server (if needed) and render in the calling cell."""
        host = kwargs.get("host", self._host)
        port = kwargs.get("port", self._port)
        height = kwargs.get("height", self._height)
        display_mode = kwargs.get("display_mode", self._display_mode)

        if not self._is_notebook():
            from .web import WebOutput

            web = WebOutput(self.project_folder, project=self.project)
            effective_port = port or 6116
            print(
                f"vibetrack jupyter: not in a notebook — starting web server at http://{host}:{effective_port}"
            )
            web._serve_uvicorn(None, host, effective_port, token=None)
            return

        self._ensure_server(host=host, port=port)
        self._render(height=height, display_mode=display_mode)

    def send(self, events: List[LogEvent]) -> None:
        # Data flows to DB via SummaryWriter; the iframe polls automatically.
        pass

    def refresh(self) -> None:
        """Force-reload the iframe in-place (replaces the existing output)."""
        if self._is_notebook() and self._display_handle and self._server_url:
            self._render()

    # ── public helpers ───────────────────────────────────────────────────────

    @property
    def url(self) -> Optional[str]:
        return self._server_url

    def start_in_thread(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> threading.Thread:
        """Start the server without rendering — returns the daemon thread."""
        self._ensure_server(host=host, port=port)
        assert self._server_thread is not None
        return self._server_thread

    # ── internals ────────────────────────────────────────────────────────────

    def _is_notebook(self) -> bool:
        try:
            ip = get_ipython()  # type: ignore[name-defined]  # built-in in IPython
            return ip is not None
        except NameError:
            return False

    def _find_free_port(self, host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    def _ensure_server(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return
        if port == 0:
            port = self._find_free_port(host)
        self._host = host
        self._port = port
        self._server_url = f"http://{host}:{port}"

        from .web import WebOutput

        web = WebOutput(self.project_folder, project=self.project)
        self._server_thread = web.start_in_thread(host=host, port=port)

        # Poll until the server accepts connections (max 3 s)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)

    def _render(
        self, height: Optional[int] = None, display_mode: Optional[str] = None
    ) -> None:
        if height is None:
            height = self._height
        if display_mode is None:
            display_mode = self._display_mode

        try:
            from IPython.display import HTML, IFrame, display
        except ImportError:
            print(f"vibetrack: {self._server_url}")
            return

        # Cache-bust so refresh() forces the browser to reload the page
        url = f"{self._server_url}?_t={int(time.time())}"

        if display_mode == "link":
            widget = HTML(
                f'<a href="{self._server_url}" target="_blank">'
                f"Open vibetrack dashboard &rarr; {self._server_url}</a>"
            )
        else:
            widget = IFrame(src=url, width="100%", height=height)

        if self._display_handle is None:
            self._display_handle = display(widget, display_id=True)
        else:
            self._display_handle.update(widget)
