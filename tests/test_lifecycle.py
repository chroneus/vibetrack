"""Run lifecycle: status, provenance, crash detection and the end-of-run report."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import vibetrack
from vibetrack import insights, writer as writer_mod
from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter

REPO = Path(__file__).resolve().parent.parent


def _writer(tmp_path, name="run", **kwargs):
    folder = tmp_path / "proj"
    kwargs.setdefault("system_metrics_interval", 0)
    return SummaryWriter(
        str(tmp_path / "logs" / name),
        name=name,
        project="demo",
        project_folder=str(folder),
        **kwargs,
    )


def _record(tmp_path, name="run"):
    with RunReader(str(tmp_path / "proj")) as reader:
        exp = reader.experiment(name) or next(
            e for e in reader.experiments(project=None) if e.name == name
        )
        return exp.run_info(), insights.run_status(exp)


def _script_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["VIBETRACK_GIT"] = "0"
    return env


class TestStatus:
    def test_finished_records_provenance(self, tmp_path):
        w = _writer(tmp_path, config={"lr": 0.1})
        w.add_scalar("loss", 1.0, 0)
        w.close()
        rec, st = _record(tmp_path)
        assert rec["status"] == "finished"
        assert rec["ended_at"] >= rec["started_at"]
        info = rec["info"]
        assert info["pid"] == os.getpid()
        assert info["host"]
        assert isinstance(info["argv"], list)
        assert st["status"] == "finished"
        assert st["duration_s"] is not None

    def test_exception_in_with_block_marks_failed(self, tmp_path):
        with pytest.raises(ValueError):
            with _writer(tmp_path) as w:
                w.add_scalar("loss", 1.0, 0)
                raise ValueError("boom")
        rec, _ = _record(tmp_path)
        assert rec["status"] == "failed"
        assert "ValueError: boom" in rec["info"]["error"]
        assert "Traceback" in rec["info"]["traceback"]

    def test_close_in_finally_during_exception_marks_failed(self, tmp_path):
        w = _writer(tmp_path)
        with pytest.raises(RuntimeError):
            try:
                w.add_scalar("loss", 1.0, 0)
                raise RuntimeError("diverged")
            finally:
                w.close()
        rec, _ = _record(tmp_path)
        assert rec["status"] == "failed"
        assert "diverged" in rec["info"]["error"]

    def test_keyboard_interrupt_marks_interrupted(self, tmp_path):
        with pytest.raises(KeyboardInterrupt):
            with _writer(tmp_path) as w:
                w.add_scalar("loss", 1.0, 0)
                raise KeyboardInterrupt
        rec, _ = _record(tmp_path)
        assert rec["status"] == "interrupted"

    def test_resume_keeps_start_and_clears_error(self, tmp_path):
        with pytest.raises(ValueError):
            with _writer(tmp_path) as w:
                w.add_scalar("loss", 1.0, 0)
                raise ValueError("first try")
        first, _ = _record(tmp_path)
        time.sleep(0.01)
        w = _writer(tmp_path)
        w.add_scalar("loss", 0.5, 5)  # later step -> resumes the same run
        w.close()
        rec, _ = _record(tmp_path)
        assert rec["status"] == "finished"
        assert rec["started_at"] == first["started_at"]
        assert "resumed_at" in rec["info"]
        assert "error" not in rec["info"]

    def test_precache_replays_run_info(self, tmp_path):
        w = _writer(tmp_path, precache_secs=60)
        w.add_scalar("loss", 1.0, 0)
        w.define_metric("loss", goal="min", primary=True)
        w._db._materialize()
        w.close()
        rec, _ = _record(tmp_path)
        assert rec["status"] == "finished"
        assert rec["info"]["metrics"] == {"loss": {"goal": "min", "primary": True}}

    def test_define_metric_stored_and_primary_unique(self, tmp_path):
        w = _writer(tmp_path)
        w.add_scalar("val/acc", 0.5, 0)
        w.define_metric("val/acc", goal="max", primary=True)
        w.define_metric("val/f1", goal="max", primary=True)
        w.define_metric("bad", goal="sideways")  # warns, ignored
        w.close()
        rec, _ = _record(tmp_path)
        assert rec["info"]["metrics"] == {
            "val/acc": {"goal": "max"},
            "val/f1": {"goal": "max", "primary": True},
        }

    def test_track_status_false_writes_nothing(self, tmp_path):
        w = _writer(tmp_path, track_status=False)
        w.add_scalar("loss", 1.0, 0)
        w.close()
        rec, st = _record(tmp_path)
        assert rec is None
        assert st["status"] == "unknown"

    def test_foreign_process_does_not_end_run(self, tmp_path):
        w = _writer(tmp_path)
        w.add_scalar("loss", 1.0, 0)
        w._owner_pid = -1  # pretend we are a forked child
        w.close()
        rec, _ = _record(tmp_path)
        assert rec["status"] == "running"

    def test_heartbeat_updates_while_logging(self, tmp_path, monkeypatch):
        monkeypatch.setattr(writer_mod, "_HEARTBEAT_SECS", 0.0)
        w = _writer(tmp_path)
        w.add_scalar("loss", 1.0, 0)
        w.flush()
        first = w._db.get_run_info(w.experiment_id)["heartbeat"]
        time.sleep(0.01)
        w.add_scalar("loss", 0.5, 1)
        w.flush()
        assert w._db.get_run_info(w.experiment_id)["heartbeat"] > first
        w.close()


class TestProcessExit:
    def test_uncaught_exception_marks_failed_at_exit(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import vibetrack
                vibetrack.init(project="demo", name="run", project_folder={str(tmp_path / 'proj')!r},
                               log_dir={str(tmp_path / 'logs' / 'run')!r}, system_metrics_interval=0)
                vibetrack.log({{"loss": 1.0}}, step=0)
                raise RuntimeError("CUDA out of memory")
                """
            )
        )
        proc = subprocess.run(
            [sys.executable, str(script)], env=_script_env(), capture_output=True, text=True
        )
        assert proc.returncode != 0
        rec, _ = _record(tmp_path)
        assert rec["status"] == "failed"
        assert "CUDA out of memory" in rec["info"]["error"]
        # The report on stderr names the failure.
        assert "FAILED" in proc.stderr and "CUDA out of memory" in proc.stderr

    @pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX only")
    def test_killed_process_reads_as_crashed(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys, time, vibetrack
                w = vibetrack.init(project="demo", name="run", project_folder={str(tmp_path / 'proj')!r},
                                   log_dir={str(tmp_path / 'logs' / 'run')!r}, system_metrics_interval=0)
                vibetrack.log({{"loss": 1.0}}, step=0)
                print("ready", flush=True)
                time.sleep(60)
                """
            )
        )
        proc = subprocess.Popen(
            [sys.executable, str(script)], env=_script_env(), stdout=subprocess.PIPE, text=True
        )
        try:
            assert proc.stdout.readline().strip() == "ready"
            _, st = _record(tmp_path)
            assert st["status"] == "running"
        finally:
            proc.kill()
            proc.wait()
        rec, st = _record(tmp_path)
        assert rec["status"] == "running"  # never closed...
        assert st["status"] == "crashed"  # ...but the process is gone
        assert "process exited" in st["error"]


class TestReport:
    def test_report_printed_on_close(self, tmp_path, capsys):
        for name, acc in [("a", 0.5), ("b", 0.9)]:
            w = _writer(tmp_path, name=name, config={"lr": acc})
            for step in range(3):
                w.add_scalar("val/acc", acc * step / 2, step)
            w.close()
        err = capsys.readouterr().err
        assert "run 'b' finished" in err
        assert "val/acc 0.9" in err
        assert "rank:    #1 of 2 runs by val/acc" in err
        assert "vibetrack open b --project-folder" in err
        assert "tab=runs" in err

    def test_report_opt_out(self, tmp_path, capsys, monkeypatch):
        w = _writer(tmp_path, name="quiet", report=False)
        w.add_scalar("loss", 1.0, 0)
        w.close()
        monkeypatch.setenv("VIBETRACK_REPORT", "0")
        w = _writer(tmp_path, name="quiet2")
        w.add_scalar("loss", 1.0, 0)
        w.close()
        assert "vibetrack: run" not in capsys.readouterr().err

    def test_no_report_without_data(self, tmp_path, capsys):
        _writer(tmp_path, name="empty").close()
        assert "vibetrack: run" not in capsys.readouterr().err

    def test_module_api_define_metric(self, tmp_path, capsys):
        vibetrack.init(
            project="demo",
            name="mod",
            project_folder=str(tmp_path / "proj"),
            log_dir=str(tmp_path / "logs" / "mod"),
            system_metrics_interval=0,
        )
        vibetrack.define_metric("score", goal="max", primary=True)
        vibetrack.log({"score": 3.0}, step=0)
        vibetrack.finish()
        rec, _ = _record(tmp_path, "mod")
        assert rec["info"]["metrics"]["score"]["primary"] is True


def test_run_info_rows_deleted_with_experiment(tmp_path):
    w = _writer(tmp_path)
    w.add_scalar("loss", 1.0, 0)
    exp_id = w.experiment_id
    w.close()
    db = Database(tmp_path / "proj" / "vibetrack.db")
    try:
        assert db.get_run_info(exp_id) is not None
        db.delete_experiment(exp_id)
        assert db.get_run_info(exp_id) is None
    finally:
        db.close()
