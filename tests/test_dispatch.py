"""Tests for per-event adapter dispatch via ``writer.to(...)``."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest

from vibetrack.viewers import base as viewers_base
from vibetrack.viewers import event as _event_mod
from vibetrack.viewers.base import BaseOutput
from vibetrack.viewers.event import LogEvent
from vibetrack.writer import SummaryWriter


# ── Fake adapter plumbing ──────────────────────────────────────

_FAKE_REGISTRY: dict = {}


class FakeAdapter(BaseOutput):
    """Records every batch of events it receives, thread-safely."""

    def __init__(
        self,
        project_folder: Optional[str] = None,
        project: Optional[str] = None,
        name: str = "fake",
        fail: bool = False,
    ) -> None:
        super().__init__(project_folder, project=project)
        self.name = name
        self.fail = fail
        self._lock = threading.Lock()
        self.batches: List[List[LogEvent]] = []
        _FAKE_REGISTRY[name] = self

    def show(self, **kwargs: Any) -> Any:
        return None

    def send(self, events: Sequence[LogEvent]) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} intentionally failing")
        with self._lock:
            self.batches.append(list(events))

    @property
    def flat(self) -> List[LogEvent]:
        with self._lock:
            return [e for batch in self.batches for e in batch]


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    _FAKE_REGISTRY.clear()

    def fake_load_viewer(name: str):
        if name == "fake":

            class _Factory(FakeAdapter):
                def __init__(self, **kwargs):
                    super().__init__(name="fake", **kwargs)

            return _Factory
        if name == "fake2":

            class _Factory2(FakeAdapter):
                def __init__(self, **kwargs):
                    super().__init__(name="fake2", **kwargs)

            return _Factory2
        if name == "fake_fail":

            class _FactoryFail(FakeAdapter):
                def __init__(self, **kwargs):
                    super().__init__(name="fake_fail", fail=True, **kwargs)

            return _FactoryFail
        raise ValueError(f"unknown test adapter {name!r}")

    from vibetrack import viewers as viewers_pkg

    monkeypatch.setattr(viewers_pkg, "load_viewer", fake_load_viewer)
    yield
    _FAKE_REGISTRY.clear()


@pytest.fixture
def log_dir(tmp_path):
    return str(tmp_path / "runs" / "test_run")


# ── Tests ──────────────────────────────────────────────────────


class TestWriterLevelTo:
    def test_every_event_dispatch(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.to("fake")
            w.add_scalar("loss", 0.5, 0)
            w.add_scalar("loss", 0.4, 1)
            w.add_scalar("loss", 0.3, 2)

        adapter = _FAKE_REGISTRY["fake"]
        events = adapter.flat
        assert [e.kind for e in events] == ["scalar"] * 3
        assert [e.value for e in events] == [0.5, 0.4, 0.3]
        assert [e.tag for e in events] == ["loss"] * 3

    def test_every_n_steps_batches(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.to("fake", every=3)
            for i in range(7):
                w.add_scalar("loss", float(i), i)
        adapter = _FAKE_REGISTRY["fake"]
        # 7 events, every=3 → two batches of 3 flushed mid-run plus a final
        # flush of the remaining 1 during close().
        assert len(adapter.batches) == 3
        assert [len(b) for b in adapter.batches] == [3, 3, 1]

    def test_chainable_multiple_adapters(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.to("fake").to("fake2")
            w.add_scalar("loss", 0.1, 0)
        assert len(_FAKE_REGISTRY["fake"].flat) == 1
        assert len(_FAKE_REGISTRY["fake2"].flat) == 1

    def test_error_isolation(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.to("fake_fail")
            w.to("fake")
            for i in range(3):
                w.add_scalar("loss", float(i), i)
        # Failing adapter must not prevent the other from receiving events.
        assert len(_FAKE_REGISTRY["fake"].flat) == 3

    def test_parse_every_time_units(self):
        from vibetrack.writer import _parse_every

        assert _parse_every(None) == ("event", None)
        assert _parse_every(100) == ("steps", 100.0)
        assert _parse_every("15m") == ("time", 900.0)
        assert _parse_every("1h") == ("time", 3600.0)
        assert _parse_every("5s") == ("time", 5.0)
        with pytest.raises(ValueError):
            _parse_every("nope")
        with pytest.raises(ValueError):
            _parse_every(0)


class TestEventLevelTo:
    def test_per_event_chain(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 0.9, 0).to("fake").to("fake2")
            w.add_scalar("loss", 0.8, 1)  # no .to() — no dispatch
        # Only the first event should have been forwarded.
        assert len(_FAKE_REGISTRY["fake"].flat) == 1
        assert _FAKE_REGISTRY["fake"].flat[0].value == 0.9
        assert len(_FAKE_REGISTRY["fake2"].flat) == 1

    def test_per_event_independent_of_registration(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 0.5, 0).to("fake")
        # No writer-level .to("fake2"), and no .to("fake2") on the event →
        # fake2 must not be instantiated or receive anything.
        assert "fake2" not in _FAKE_REGISTRY
        assert len(_FAKE_REGISTRY["fake"].flat) == 1

    def test_log_multi_handle_fans_out(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.log({"loss": 0.1, "acc": 0.9}).to("fake")
        events = _FAKE_REGISTRY["fake"].flat
        assert len(events) == 2
        assert {e.tag for e in events} == {"loss", "acc"}


class TestImageEvent:
    def test_image_event_carries_abs_path(self, log_dir, tmp_path):
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("Pillow not installed")
        img_path = tmp_path / "in.png"
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img_path)

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.to("fake")
            w.add_image("samples", str(img_path), 0)

        events = _FAKE_REGISTRY["fake"].flat
        assert len(events) == 1
        assert events[0].kind == "image"
        assert Path(events[0].value).exists()
