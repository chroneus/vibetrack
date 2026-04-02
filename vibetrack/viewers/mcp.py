"""MCP server — expose experiment data as MCP tools and resources.

"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
from typing import Any, List, Optional, Sequence

from ..compare import compare_hparams, summary_table
from .base import BaseOutput

_STREAMABLE_HTTP_LOGGER = "mcp.server.streamable_http_manager"


@contextlib.contextmanager
def _suppress_streamable_http_startup_log() -> Any:
    """Mute the noisy startup INFO emitted by MCP's streamable HTTP manager."""
    logger = logging.getLogger(_STREAMABLE_HTTP_LOGGER)
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        yield
    finally:
        logger.setLevel(previous_level)


class MCPOutput(BaseOutput):
    """Expose experiment tracking data via an MCP server."""

    def __init__(self, project_folder: Optional[str] = None) -> None:
        super().__init__(project_folder)
        self._mcp_kwargs: dict = {}
        self._transport = "streamable-http"

    def _build_mcp(self) -> Any:
        """Build the FastMCP instance (deferred so host/port can be set first)."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("vibetrack", **self._mcp_kwargs)
        self._mcp = mcp
        self._register_tools()
        self._register_resources()
        return mcp

    # ── Tools ────────────────────────────────────────────────

    def _register_tools(self) -> None:
        reader = self._reader
        resolve = self._resolve_experiments
        mcp = self._mcp

        @mcp.tool()
        def list_experiments() -> str:
            """List all experiments in the project database."""
            exps = reader.experiments()
            return json.dumps(
                [{"name": e.name, "experiment_id": e.experiment_id} for e in exps]
            )

        @mcp.tool()
        def get_experiment_tags(experiment: str) -> str:
            """Get all available tags for an experiment, grouped by type."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            exp = exps[0]
            return json.dumps({
                "name": exp.name,
                "scalars": exp.scalar_tags(),
                "texts": exp.text_tags(),
                "images": exp.image_tags(),
                "audio": exp.audio_tags(),
                "video": exp.video_tags(),
                "artifacts": exp.artifact_tags(),
                "histograms": exp.histogram_tags(),
            })

        @mcp.tool()
        def get_scalars(experiment: str, tag: str) -> str:
            """Get scalar time-series data for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].scalars(tag))

        @mcp.tool()
        def get_texts(experiment: str, tag: str) -> str:
            """Get text entries for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].texts(tag))

        @mcp.tool()
        def get_images(experiment: str, tag: str) -> str:
            """Get image entries (step, path) for a specific tag."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].images(tag))

        @mcp.tool()
        def get_audio(experiment: str, tag: str) -> str:
            """Get audio entries (step, path, sample_rate) for a specific tag."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].audio(tag))

        @mcp.tool()
        def get_hparams(experiment: str) -> str:
            """Get hyperparameters for an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].hparams())

        @mcp.tool()
        def get_histograms(experiment: str, tag: str) -> str:
            """Get histogram data for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].histograms(tag))

        @mcp.tool()
        def summary(
            experiments: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
        ) -> str:
            """Get a summary table: last value of each tag per experiment."""
            exps = resolve(experiments)
            if not exps:
                return json.dumps({"error": "No experiments found"})
            return json.dumps(summary_table(exps, tags))

        @mcp.tool()
        def compare_hparams_tool(
            experiments: Optional[List[str]] = None,
        ) -> str:
            """Compare hyperparameters across experiments side-by-side."""
            exps = resolve(experiments)
            if not exps:
                return json.dumps({"error": "No experiments found"})
            return json.dumps(compare_hparams(exps))

    # ── Resources ────────────────────────────────────────────

    def _register_resources(self) -> None:
        reader = self._reader
        resolve = self._resolve_experiments
        mcp = self._mcp

        @mcp.resource("vibetrack://experiments")
        def experiments_list() -> str:
            """List all experiments."""
            exps = reader.experiments()
            return json.dumps(
                [{"name": e.name, "experiment_id": e.experiment_id} for e in exps]
            )

        @mcp.resource("vibetrack://experiments/{name}")
        def experiment_detail(name: str) -> str:
            """Experiment overview: tags by type, hparams, config."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            exp = exps[0]
            return json.dumps({
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "scalars": exp.scalar_tags(),
                "texts": exp.text_tags(),
                "images": exp.image_tags(),
                "audio": exp.audio_tags(),
                "video": exp.video_tags(),
                "artifacts": exp.artifact_tags(),
                "histograms": exp.histogram_tags(),
                "hparams": exp.hparams(),
                "config": exp.config(),
            })

        @mcp.resource("vibetrack://experiments/{name}/scalars/{tag}")
        def experiment_scalars(name: str, tag: str) -> str:
            """Scalar time-series for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].scalars(tag))

        @mcp.resource("vibetrack://experiments/{name}/texts/{tag}")
        def experiment_texts(name: str, tag: str) -> str:
            """Text entries for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].texts(tag))

        @mcp.resource("vibetrack://experiments/{name}/images/{tag}")
        def experiment_images(name: str, tag: str) -> str:
            """Image entries for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].images(tag))

        @mcp.resource("vibetrack://experiments/{name}/hparams")
        def experiment_hparams(name: str) -> str:
            """Hyperparameters for an experiment."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].hparams())

    def asgi_app(self) -> Any:
        """Build MCP and return the Starlette ASGI app for mounting."""
        self._build_mcp()
        return self._mcp.streamable_http_app()

    # ── Viewer interface ─────────────────────────────────────

    def show(self, **kwargs: Any) -> None:
        host = kwargs.get("host", "127.0.0.1")
        port = kwargs.get("port", 6006)
        transport = kwargs.get("mcp_transport", self._transport)

        self._mcp_kwargs.update(host=host, port=port)
        self._build_mcp()

        path = "/mcp" if transport == "streamable-http" else "/sse"
        print(f"vibetrack MCP server: http://{host}:{port}{path}")
        runner = self._mcp.run
        if transport == "streamable-http":
            with _suppress_streamable_http_startup_log():
                runner(transport=transport)
        else:
            runner(transport=transport)

    def start_in_thread(
        self,
        host: str = "127.0.0.1",
        port: int = 6007,
        transport: str = "streamable-http",
    ) -> "Any":
        """Start the MCP server in a daemon thread and return immediately."""
        import threading

        self._mcp_kwargs.update(host=host, port=port)
        self._build_mcp()

        path = "/mcp" if transport == "streamable-http" else "/sse"
        print(f"vibetrack MCP server: http://{host}:{port}{path}")
        if transport == "streamable-http":
            def _run() -> None:
                with _suppress_streamable_http_startup_log():
                    self._mcp.run(transport=transport)
        else:
            def _run() -> None:
                self._mcp.run(transport=transport)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vibetrack MCP server")
    parser.add_argument("--project-folder", default=None, help="Project folder")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6006, help="Port (default: 6006)")
    parser.add_argument(
        "--transport", default="streamable-http",
        choices=["streamable-http", "sse"],
        help="MCP transport (default: streamable-http)",
    )
    args = parser.parse_args()

    viewer = MCPOutput(args.project_folder)
    viewer.show(host=args.host, port=args.port, mcp_transport=args.transport)
