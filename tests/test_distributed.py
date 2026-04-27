"""Tests for distributed training (torchrun/DDP) rank gating."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vibetrack.writer import SummaryWriter, _detect_rank


class TestDetectRank:
    def test_defaults_to_zero(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert _detect_rank() == 0

    def test_reads_rank_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert _detect_rank() == 3

    def test_falls_back_to_local_rank(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert _detect_rank() == 2

    def test_rank_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("RANK", "5")
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert _detect_rank() == 5

    def test_invalid_rank_defaults_to_zero(self, monkeypatch):
        monkeypatch.setenv("RANK", "not_a_number")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert _detect_rank() == 0


class TestNoOpOnNonZeroRank:
    def test_noop_writer_attributes(self, tmp_path):
        w = SummaryWriter(str(tmp_path / "noop_run"), rank=1)
        assert not w._enabled
        assert w._db is None
        assert w.experiment_id == -1
        assert w._sysmetrics is None
        w.close()

    def test_noop_writer_methods_succeed(self, tmp_path):
        w = SummaryWriter(str(tmp_path / "noop_run"), rank=1)
        w.add_scalar("loss", 0.5, 0)
        w.add_scalars("metrics", {"a": 1, "b": 2}, 0)
        w.add_text("note", "hello", 0)
        w.log({"x": 1, "y": 2})
        w.flush()
        w.close()

    def test_no_directory_created(self, tmp_path):
        log_dir = tmp_path / "should_not_exist" / "run1"
        w = SummaryWriter(str(log_dir), rank=1)
        w.add_scalar("x", 1, 0)
        w.close()
        assert not log_dir.exists()

    def test_context_manager(self, tmp_path):
        with SummaryWriter(str(tmp_path / "noop_run"), rank=1) as w:
            w.add_scalar("loss", 0.5, 0)
        assert w._closed

    def test_env_auto_detection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        w = SummaryWriter(str(tmp_path / "auto_noop"))
        assert not w._enabled
        assert w._rank == 2
        w.close()


class TestRankZeroLogs:
    def test_rank_zero_writes_to_db(self, tmp_path):
        pf = tmp_path / "project"
        w = SummaryWriter(
            str(tmp_path / "run1"),
            project="test",
            rank=0,
            project_folder=str(pf),
        )
        w.add_scalar("loss", 0.5, 0)
        w.close()
        assert w._exp_id >= 0
        assert (pf / "vibetrack.db").exists()

    def test_rank_all_forces_logging(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        pf = tmp_path / "project"
        w = SummaryWriter(
            str(tmp_path / "run1"),
            project="test",
            rank="all",
            project_folder=str(pf),
        )
        assert w._enabled
        assert w._rank == 0
        w.add_scalar("loss", 0.5, 0)
        w.close()
        assert (pf / "vibetrack.db").exists()


class TestInitModuleAPI:
    def test_init_with_rank(self, tmp_path):
        import vibetrack

        w = vibetrack.init(
            project="test",
            name="noop_run",
            log_dir=str(tmp_path / "run1"),
            rank=1,
        )
        assert not w._enabled
        vibetrack.log({"loss": 0.5})  # should not crash
        vibetrack.finish()
