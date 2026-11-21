"""Agent-facing CLI commands, dashboard reuse, web/MCP payloads and video logging."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from vibetrack import cli, commands
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "proj"
    for name, lr in [("base", 0.01), ("fast", 0.1), ("fast-2", 0.2)]:
        with SummaryWriter(
            str(tmp_path / "logs" / name),
            name=name,
            project="exp",
            project_folder=str(folder),
            config={"lr": lr, "batch": 32},
            system_metrics_interval=0,
            report=False,
        ) as w:
            for step in range(5):
                w.add_scalar("val/acc", step * lr, step)
                w.add_scalar("train/loss", 1 / (step + 1), step)
            w.add_text("notes", f"lr={lr}", 0)
    return str(folder)


def _run(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    out = capsys.readouterr()
    return exc.value.code, out.out, out.err


class TestReadCommands:
    def test_ls_markdown_and_json(self, project, capsys):
        code, out, _ = _run(["ls", "--project-folder", project], capsys)
        assert code == 0
        assert "# vibetrack runs (3) - project `exp`" in out
        assert "| fast-2 | finished |" in out
        code, out, _ = _run(["ls", "--project-folder", project, "--json"], capsys)
        rows = json.loads(out)
        assert {r["name"] for r in rows} == {"base", "fast", "fast-2"}

    def test_show_named_latest_and_unknown(self, project, capsys):
        code, out, _ = _run(["show", "base", "--project-folder", project], capsys)
        assert code == 0 and "# Run `base`" in out and "lr=0.01" in out
        assert "`notes` @0" in out
        code, out, _ = _run(["show", "--project-folder", project, "--json"], capsys)
        assert json.loads(out)["name"] == "fast-2"  # most recent
        code, _, err = _run(["show", "nope", "--project-folder", project], capsys)
        assert code == 1 and "No run matches 'nope'" in err and "Recent runs:" in err

    def test_compare_glob_and_metric(self, project, capsys):
        code, out, _ = _run(["compare", "fast*", "--project-folder", project, "--json"], capsys)
        data = json.loads(out)
        assert [r["name"] for r in data["runs"]] == ["fast-2", "fast"]
        assert data["params"]["varying"] == {"lr": {"fast-2": 0.2, "fast": 0.1}}
        code, out, _ = _run(
            ["compare", "--project-folder", project, "--metric", "train/loss", "--goal", "min"],
            capsys,
        )
        assert code == 0 and "Ranked by **train/loss** (min" in out

    def test_missing_db_points_to_local_dbs(self, project, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: tmp_path / "none" / "vibetrack.db")
        code, _, err = _run(["ls"], capsys)
        assert code == 1
        assert "No vibetrack database" in err and "--project-folder proj" in err

    def test_help_lists_agent_commands(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        assert "vibetrack compare" in out and "vibetrack guide" in out

    def test_existing_viewer_cli_untouched(self, project, monkeypatch):
        seen = {}

        class FakeViewer:
            def __init__(self, folder, project=None):
                seen["folder"] = folder

            def show(self, **kwargs):
                seen["port"] = kwargs["port"]

        monkeypatch.setattr("vibetrack.viewers.load_viewer", lambda name: FakeViewer)
        cli.main([project, "--port", "7000"])
        assert seen == {"folder": str(Path(project).resolve()), "port": 7000}


class TestGuide:
    def test_guide_prints_checklist(self, capsys):
        code, out, _ = _run(["guide"], capsys)
        assert code == 0
        assert "Instrument the code" in out and "vibetrack open" in out

    def test_install_is_idempotent(self, tmp_path, capsys):
        (tmp_path / "AGENTS.md").write_text("# Project\n\nExisting notes.\n")
        for _ in range(2):
            code, _, _ = _run(["guide", "--install", "--dir", str(tmp_path)], capsys)
            assert code == 0
        skill = (tmp_path / ".claude/skills/vibetrack/SKILL.md").read_text()
        assert skill.startswith("---\nname: vibetrack\n")
        agents = (tmp_path / "AGENTS.md").read_text()
        assert agents.startswith("# Project\n\nExisting notes.\n")
        assert agents.count(commands._AGENTS_START) == 1


class TestOpen:
    def test_open_uses_dashboard_url(self, project, capsys, monkeypatch):
        calls = {}

        def fake_ensure(folder, db_path, port=6116):
            calls["args"] = (folder, Path(db_path), port)
            return "http://127.0.0.1:6200", None, None

        monkeypatch.setattr(commands, "ensure_dashboard", fake_ensure)
        code, out, _ = _run(["open", "fast", "--project-folder", project, "--no-browser"], capsys)
        assert code == 0
        assert "Dashboard: http://127.0.0.1:6200/exp?run=fast&tab=runs" in out
        assert calls["args"] == (project, Path(project) / "vibetrack.db", 6116)

    def test_ensure_dashboard_starts_and_reuses_server(self, project, tmp_path, monkeypatch):
        pytest.importorskip("uvicorn")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        db_path = Path(project) / "vibetrack.db"
        base, pid, log = commands.ensure_dashboard(project, db_path, port=port, tries=3)
        try:
            assert pid is not None and log.parent == tmp_path / ".vibetrack"
            info = commands._probe(int(base.rsplit(":", 1)[1]))
            assert info["app"] == "vibetrack" and Path(info["db_path"]) == db_path
            again, pid2, _ = commands.ensure_dashboard(project, db_path, port=port, tries=3)
            assert again == base and pid2 is None  # reused
        finally:
            os.kill(pid, 15)


class TestWebPayloads:
    def test_api_info_and_run_card(self, project):
        httpx = pytest.importorskip("httpx")
        import asyncio

        from vibetrack.viewers.web import WebOutput

        app = WebOutput(project)._build_app(None, "127.0.0.1")

        async def get(url):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:6116") as c:
                return await c.get(url)

        info = asyncio.run(get("/api/info")).json()
        assert info["app"] == "vibetrack"
        assert Path(info["db_path"]) == Path(project) / "vibetrack.db"
        data = asyncio.run(get("/api/data/exp")).json()
        card = next(d["run"] for d in data if d["name"] == "fast")
        assert card["status"] == "finished"
        assert card["headline"][0]["tag"] == "val/acc"
        page = asyncio.run(get("/exp")).text
        assert 'id="tab-runs"' in page and "/static/js/runs.js" in page


def test_mcp_run_tools(project):
    pytest.importorskip("mcp")
    import asyncio

    from vibetrack.viewers.mcp import MCPOutput

    out = MCPOutput(project)
    out._build_mcp()

    async def call(name, args):
        result = await out._mcp.call_tool(name, args)
        content = result[0] if isinstance(result, tuple) else result
        return content[0].text

    rows = json.loads(asyncio.run(call("list_runs", {"project": "exp"})))
    assert len(rows) == 3
    md = asyncio.run(call("compare_runs", {"as_markdown": True}))
    assert "Ranked by **val/acc**" in md
    ov = json.loads(asyncio.run(call("run_overview", {"experiment": "base"})))
    assert ov["config"]["lr"] == 0.01


class TestVideoLogging:
    def test_log_passes_video_fps(self, tmp_path, monkeypatch):
        import vibetrack
        import vibetrack.media as media

        seen = {}

        def fake_save(data, log_dir, tag, step, fps=4):
            seen["fps"] = fps
            return "media/clip/0.mp4"

        monkeypatch.setattr(media, "save_video", fake_save)
        with SummaryWriter(str(tmp_path / "r"), project_folder=str(tmp_path), system_metrics_interval=0, report=False) as w:
            w.log({"clip": vibetrack.Video(b"x", fps=30)}, step=0)
        assert seen["fps"] == 30

    def test_log_accepts_numpy_scalars(self, tmp_path):
        np = pytest.importorskip("numpy")
        with SummaryWriter(str(tmp_path / "r"), name="r", project_folder=str(tmp_path), system_metrics_interval=0, report=False) as w:
            w.log({"a": np.float32(0.5), "b": np.array(2.0), "c": np.int64(3)}, step=0)
        with RunReader(str(tmp_path)) as reader:
            exp = reader.experiments(project=None)[0]
            assert sorted(exp.scalar_tags()) == ["a", "b", "c"]

    def test_frame_list_encodes_mp4(self, tmp_path):
        np = pytest.importorskip("numpy")
        from vibetrack.media import _ffmpeg_exe, save_video

        if _ffmpeg_exe() is None:
            pytest.skip("no ffmpeg")
        frames = [np.full((31, 41, 3), i * 20, np.uint8) for i in range(8)]
        log_dir = str(tmp_path.resolve())
        rel = save_video(frames, log_dir, "eval/rollout", 7, fps=15)
        assert rel == os.path.join("media", "eval_rollout", "7.mp4")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(tmp_path / rel)],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            assert probe.stdout.strip() == "h264,yuv420p"

    def test_gif_fallback_without_encoders(self, tmp_path, monkeypatch):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        import vibetrack.media as media

        monkeypatch.setattr(media, "_ffmpeg_exe", lambda: None)
        monkeypatch.setattr(media, "_encode_mp4_moviepy", lambda *a, **k: False)
        frames = np.zeros((4, 16, 16, 3), np.uint8)
        rel = media.save_video(frames, str(tmp_path.resolve()), "clip", 0, fps=10)
        assert rel.endswith("0.gif") and (tmp_path / rel).stat().st_size > 0


def test_entrypoint_runs_as_module(project):
    proc = subprocess.run(
        [sys.executable, "-m", "vibetrack.cli", "ls", "--project-folder", project],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "vibetrack runs (3)" in proc.stdout
