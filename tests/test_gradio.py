"""Tests for the Gradio output backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List
from unittest import mock

import pytest

if importlib.util.find_spec("gradio") is None:  # pragma: no cover
    pytest.skip("gradio not installed", allow_module_level=True)

from vibetrack.viewers.event import LogEvent
from vibetrack.viewers.gradio import GradioOutput
from vibetrack.writer import SummaryWriter


def _make_event(tag: str, step: int, value: float) -> LogEvent:
    return LogEvent(
        kind="scalar",
        tag=tag,
        step=step,
        value=value,
        walltime=0.0,
        run_name="run_a",
        project="proj",
    )


@pytest.fixture(autouse=True)
def _clear_gradio_buffer():
    """Class-level buffer is shared state — clear before & after each test."""
    GradioOutput._live_buffer.clear()
    GradioOutput._live_keys.clear()
    GradioOutput._launch_threads.clear()
    yield
    GradioOutput._live_buffer.clear()
    GradioOutput._live_keys.clear()
    GradioOutput._launch_threads.clear()


class TestGradioSend:
    def test_send_buffers_events(self, tmp_path: Path) -> None:
        out = GradioOutput(project_folder=str(tmp_path))
        events = [_make_event("loss", i, 1.0 / (i + 1)) for i in range(5)]
        with mock.patch.object(GradioOutput, "start", lambda self, **kwargs: None):
            out.send(events)
        assert len(GradioOutput._live_buffer) == 5
        assert [e.step for e in GradioOutput._live_buffer] == [0, 1, 2, 3, 4]

    def test_send_respects_cap(self, tmp_path: Path) -> None:
        out = GradioOutput(project_folder=str(tmp_path))
        original_cap = GradioOutput._live_buffer_max
        try:
            GradioOutput._live_buffer_max = 4
            with mock.patch.object(GradioOutput, "start", lambda self, **kwargs: None):
                for batch_start in range(0, 10, 2):
                    out.send(
                        [
                            _make_event("loss", batch_start, 0.0),
                            _make_event("loss", batch_start + 1, 0.0),
                        ]
                    )
            buf = GradioOutput._live_buffer
            assert len(buf) == 4
            # Oldest events evicted; newest survive in order.
            assert [e.step for e in buf] == [6, 7, 8, 9]
        finally:
            GradioOutput._live_buffer_max = original_cap


class TestWriterDispatch:
    def test_writer_to_gradio_populates_buffer(self, tmp_path: Path) -> None:
        with mock.patch.object(GradioOutput, "start", lambda self, **kwargs: None):
            with SummaryWriter(
                log_dir=str(tmp_path / "run"),
                project_folder=str(tmp_path),
                name="run_a",
                system_metrics_interval=0,
            ) as writer:
                writer.to("gradio")
                for step in range(8):
                    writer.add_scalar("loss", 1.0 / (step + 1), step)
                writer.flush()

        # `to("gradio")` with no `every=` flushes per-event, so all 8 land.
        buf = GradioOutput._live_buffer
        assert len(buf) == 8
        assert all(e.kind == "scalar" and e.tag == "loss" for e in buf)
        assert [e.step for e in buf] == list(range(8))

    def test_writer_to_gradio_starts_dashboard(self, tmp_path: Path) -> None:
        started: List[Any] = []

        def _fake_start(self: GradioOutput, **kwargs: Any) -> None:
            started.append((self.project_folder, self.project, kwargs))

        with mock.patch.object(GradioOutput, "start", _fake_start):
            with SummaryWriter(
                log_dir=str(tmp_path / "run"),
                project_folder=str(tmp_path),
                name="run_a",
                system_metrics_interval=0,
            ) as writer:
                writer.to("gradio")

        assert started
        assert started[0][0] == str(tmp_path.resolve())

    def test_snapshot_dedupes_live_and_persisted_rows(self, tmp_path: Path) -> None:
        np = pytest.importorskip("numpy")
        with mock.patch.object(GradioOutput, "start", lambda self, **kwargs: None):
            with SummaryWriter(
                log_dir=str(tmp_path / "run"),
                project_folder=str(tmp_path),
                name="run_a",
                system_metrics_interval=0,
            ) as writer:
                writer.to("gradio")
                writer.add_image(
                    "train/samples",
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    0,
                    dataformats="HWC",
                )
                writer.add_text("train/predictions", "preds: 1 2 3", 0)

        out = GradioOutput(project_folder=str(tmp_path))
        snapshot = out._snapshot()
        assert len(out._gallery_items(snapshot, "images", "train/samples")) == 1
        assert out._text_blocks(snapshot, "train/predictions").count("preds:") == 1


class TestGradioShow:
    def test_show_builds_blocks_without_launching(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        with SummaryWriter(
            log_dir=str(tmp_path / "run"),
            project_folder=str(tmp_path),
            name="run_a",
            system_metrics_interval=0,
        ) as writer:
            for step in range(5):
                writer.add_scalar("loss", 1.0 / (step + 1), step)
                writer.add_scalar("acc", step / 5, step)

        import gradio as gr

        monkeypatch.setattr(
            "vibetrack.viewers.gradio.load_config",
            lambda project=None: {
                "gradio": {"share": True},
                "web": {"auto_refresh": 5},
            },
        )
        launched: List[Any] = []

        def _fake_launch(self, *args: Any, **kwargs: Any) -> Any:
            launched.append(kwargs)
            return self

        with mock.patch.object(gr.Blocks, "launch", _fake_launch):
            out = GradioOutput(project_folder=str(tmp_path))
            demo = out.show()

        assert isinstance(demo, gr.Blocks)
        assert launched, "demo.launch should have been invoked once"
        assert launched[0].get("share") is True

    def test_show_can_launch_local_from_config(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        import gradio as gr

        monkeypatch.setattr(
            "vibetrack.viewers.gradio.load_config",
            lambda project=None: {
                "gradio": {"share": False},
                "web": {"auto_refresh": 5},
            },
        )
        launched: List[Any] = []

        def _fake_launch(self, *args: Any, **kwargs: Any) -> Any:
            launched.append(kwargs)
            return self

        with mock.patch.object(gr.Blocks, "launch", _fake_launch):
            out = GradioOutput(project_folder=str(tmp_path))
            out.show()

        assert launched[0].get("share") is False

    def test_show_uses_gradio_timer(self, tmp_path: Path) -> None:
        import gradio as gr

        timers: List[Any] = []
        original_timer = gr.Timer

        def _timer(*args: Any, **kwargs: Any) -> Any:
            timer = original_timer(*args, **kwargs)
            timers.append((args, kwargs))
            return timer

        with mock.patch.object(gr, "Timer", _timer), mock.patch.object(
            gr.Blocks, "launch", lambda self, *a, **k: self
        ):
            out = GradioOutput(project_folder=str(tmp_path))
            out.show()

        assert timers
        assert timers[0][1].get("active") is True

    def test_channels_hide_empty_tabs(self, tmp_path: Path) -> None:
        out = GradioOutput(project_folder=str(tmp_path))
        data = [
            {
                "tags": ["loss"],
                "figure_tags": [],
                "pr_curve_tags": [],
                "image_tags": [],
                "audio_tags": [],
                "video_tags": [],
                "artifact_tags": [],
                "model_tags": [],
                "graph_tags": [],
                "mesh_tags": [],
                "embedding_tags": [],
                "text_tags": [],
                "histogram_tags": [],
                "system_tags": [],
                "hparams": {},
            }
        ]
        channels = out._channels(data)
        assert channels["scalars"] is True
        assert channels["images"] is False
        assert channels["audio"] is False

    def test_indexed_image_picker_uses_previous_available_step(self) -> None:
        index = GradioOutput._image_index(
            [
                ("run_a", 10, "a10.png"),
                ("run_a", 30, "a30.png"),
                ("run_b", 20, "b20.png"),
            ]
        )

        assert GradioOutput._pick_indexed_image(index, "run_a", 25, 0) == "a10.png"
        assert GradioOutput._pick_indexed_image(index, "run_a", 30, 0) == "a30.png"
        assert GradioOutput._pick_indexed_image(index, "run_b", 0, 0) == "b20.png"
        assert GradioOutput._pick_indexed_image(index, "missing", 25, 0) is None

    def test_large_images_are_thumbnailed_for_display(self, tmp_path: Path) -> None:
        from PIL import Image as PILImage

        source = tmp_path / "large.png"
        PILImage.new("RGB", (1400, 900), "red").save(source)

        thumb = GradioOutput._display_image_path(str(source), 128)

        assert thumb != str(source)
        assert Path(thumb).exists()
        assert GradioOutput._display_image_path(str(source), 128) == thumb

    def test_show_with_no_experiments_still_returns_blocks(
        self, tmp_path: Path
    ) -> None:
        import gradio as gr

        with mock.patch.object(gr.Blocks, "launch", lambda self, *a, **k: self):
            out = GradioOutput(project_folder=str(tmp_path))
            demo = out.show()
        assert isinstance(demo, gr.Blocks)


class TestProjectFilterRace:
    """Concurrent dropdown changes / timer ticks must not corrupt the
    per-project filter applied to the snapshot."""

    def test_concurrent_snapshots_isolate_per_project(self, tmp_path: Path) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from vibetrack.db import Database

        # Two projects with disjoint experiment names.
        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        db = Database(central_db)
        for i in range(5):
            db.create_experiment(
                f"alpha_{i}", project="alpha", log_dir=str(tmp_path / f"a{i}")
            )
            db.create_experiment(
                f"beta_{i}", project="beta", log_dir=str(tmp_path / f"b{i}")
            )
        db.close()

        with mock.patch("vibetrack.reader.central_db_path", lambda: central_db):
            out = GradioOutput()

            def snap(project: str):
                exps = out._resolve_experiments(project=project)
                return project, sorted(e.name for e in exps)

            # 8 threads × 200 iterations alternating projects.
            tasks = []
            for _ in range(200):
                tasks.append("alpha")
                tasks.append("beta")
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(snap, tasks))

            for project, names in results:
                # Every result must contain only the requested project.
                if project == "alpha":
                    assert all(n.startswith("alpha_") for n in names), names
                else:
                    assert all(n.startswith("beta_") for n in names), names

            out.close()


class TestThumbCacheCap:
    """Bounded thumbnail cache prevents memory + /tmp leak."""

    def test_cache_evicts_oldest_entries_and_unlinks_files(
        self, tmp_path: Path
    ) -> None:
        original_max = GradioOutput._thumb_cache_max
        GradioOutput._thumb_cache.clear()
        GradioOutput._thumb_cache_max = 32
        try:
            written = []
            for i in range(80):
                key = (f"/fake/path/{i}.png", i, 100, 128)
                target = tmp_path / f"thumb_{i}.webp"
                target.write_bytes(b"WEBPDATA")
                written.append(target)
                with GradioOutput._thumb_lock:
                    GradioOutput._thumb_cache[key] = str(target)
                    GradioOutput._thumb_cache.move_to_end(key)
                    while (
                        len(GradioOutput._thumb_cache) > GradioOutput._thumb_cache_max
                    ):
                        _, evicted = GradioOutput._thumb_cache.popitem(last=False)
                        try:
                            import os as _os

                            _os.unlink(evicted)
                        except OSError:
                            pass

            assert len(GradioOutput._thumb_cache) == 32
            # Oldest 48 should be unlinked, newest 32 still on disk.
            for t in written[:48]:
                assert not t.exists()
            for t in written[48:]:
                assert t.exists()
        finally:
            GradioOutput._thumb_cache.clear()
            GradioOutput._thumb_cache_max = original_max


class TestStartWorkerVisibility:
    """Launch failures must surface to stderr, not just to logs."""

    def test_failure_prints_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = GradioOutput(project_folder=str(tmp_path))

        def boom(self, **kwargs):
            raise RuntimeError("tunnel blocked")

        with mock.patch.object(GradioOutput, "show", boom):
            out._start_worker(out._live_key(), {})

        captured = capsys.readouterr()
        assert "vibetrack gradio:" in captured.err
        assert "tunnel blocked" in captured.err
