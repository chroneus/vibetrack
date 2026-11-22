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


@pytest.fixture(autouse=True)
def keep_early_failures(monkeypatch):
    """These tests inspect how failed runs are recorded, so keep them.

    Discarding runs that fail within the first minute is tested in
    ``TestDiscardEarlyFailures``.
    """
    monkeypatch.setenv("VIBETRACK_DISCARD_FAILED_SECS", "0")


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


def _names(tmp_path):
    db = Database(tmp_path / "proj" / "vibetrack.db")
    try:
        return sorted(r["name"] for r in db.list_experiments())
    finally:
        db.close()


def _image():
    np = pytest.importorskip("numpy")
    return np.full((4, 4, 3), 100, dtype=np.uint8)


class TestDiscardEarlyFailures:
    """Runs that fail or are interrupted in their first minute are not kept."""

    @pytest.fixture(autouse=True)
    def default_limit(self, monkeypatch):
        monkeypatch.delenv("VIBETRACK_DISCARD_FAILED_SECS", raising=False)

    def test_early_exception_deletes_run_and_media(self, tmp_path, capsys):
        with pytest.raises(ValueError):
            with _writer(tmp_path, name="keep", report=False) as w:
                w.add_scalar("loss", 1.0, 0)
            with _writer(tmp_path, name="buggy") as w:
                w.add_scalar("loss", 1.0, 0)
                w.add_image("img", _image(), 0, dataformats="HWC")
                log_dir = Path(w.log_dir)
                assert (log_dir / "media").is_dir()
                raise ValueError("bad shape")
        assert _names(tmp_path) == ["keep"]
        assert not log_dir.exists()
        err = capsys.readouterr().err
        assert "run 'buggy' FAILED after 0s: ValueError: bad shape" in err
        assert "discarded" in err and "VIBETRACK_DISCARD_FAILED_SECS=0" in err

    def test_early_ctrl_c_is_discarded(self, tmp_path):
        with pytest.raises(KeyboardInterrupt):
            with _writer(tmp_path) as w:
                w.add_scalar("loss", 1.0, 0)
                raise KeyboardInterrupt
        assert _names(tmp_path) == []

    def test_short_finished_run_is_kept(self, tmp_path):
        with _writer(tmp_path) as w:
            w.add_scalar("loss", 1.0, 0)
        assert _names(tmp_path) == ["run"]

    def test_failure_after_the_limit_is_kept(self, tmp_path):
        with pytest.raises(ValueError):
            with _writer(tmp_path, discard_failed_secs=0.2) as w:
                w.add_scalar("loss", 1.0, 0)
                time.sleep(0.3)
                raise ValueError("late")
        rec, _ = _record(tmp_path)
        assert rec["status"] == "failed"

    @pytest.mark.parametrize("how", ["kwarg", "env"])
    def test_zero_keeps_every_run(self, tmp_path, monkeypatch, how):
        kwargs = {"discard_failed_secs": 0} if how == "kwarg" else {}
        if how == "env":
            monkeypatch.setenv("VIBETRACK_DISCARD_FAILED_SECS", "0")
        with pytest.raises(ValueError):
            with _writer(tmp_path, **kwargs) as w:
                w.add_scalar("loss", 1.0, 0)
                raise ValueError("boom")
        assert _names(tmp_path) == ["run"]

    def test_resumed_run_failing_early_is_kept(self, tmp_path):
        with _writer(tmp_path) as w:
            w.add_scalar("loss", 1.0, 0)
        with pytest.raises(ValueError):
            with _writer(tmp_path, resume=True) as w:
                w.add_scalar("loss", 2.0, 1)
                raise ValueError("boom")
        rec, _ = _record(tmp_path)
        assert rec["status"] == "failed"
        with RunReader(str(tmp_path / "proj")) as reader:
            assert len(reader.experiment("run").scalars("loss")) == 2

    def test_restarted_run_failing_early_keeps_the_original(self, tmp_path):
        with _writer(tmp_path) as w:
            w.add_scalar("loss", 1.0, 5)
        with pytest.raises(ValueError):
            with _writer(tmp_path) as w:
                w.add_scalar("loss", 2.0, 0)  # step <= 5: a new "run (2)"
                raise ValueError("boom")
        assert _names(tmp_path) == ["run"]

    def test_media_of_a_shared_log_dir_is_kept(self, tmp_path):
        shared = str(tmp_path / "logs" / "shared")
        with SummaryWriter(shared, name="a", project="demo", project_folder=str(tmp_path / "proj"),
                           system_metrics_interval=0) as w:
            w.add_image("img", _image(), 0, dataformats="HWC")
        with pytest.raises(ValueError):
            with SummaryWriter(shared, name="b", project="demo",
                               project_folder=str(tmp_path / "proj"),
                               system_metrics_interval=0) as w:
                w.add_image("img", _image(), 1, dataformats="HWC")
                raise ValueError("boom")
        assert _names(tmp_path) == ["a"]
        assert (Path(shared) / "media" / "img" / "0.png").is_file()

    def test_run_that_logged_nothing_is_discarded(self, tmp_path, capsys):
        with _writer(tmp_path, config={"lr": 0.1}):
            pass  # config only: the script stopped before its first step
        assert _names(tmp_path) == []
        assert "ended after 0s without logging anything" in capsys.readouterr().err

    def test_run_with_only_system_metrics_is_discarded(self, tmp_path):
        w = _writer(tmp_path)
        w._db.add_scalar(w.experiment_id, "system/cpu_percent", 5.0, 0)
        w.close()
        assert _names(tmp_path) == []

    @pytest.mark.parametrize("log", ["text", "hparams-metric"])
    def test_any_logged_data_keeps_a_finished_run(self, tmp_path, log):
        with _writer(tmp_path) as w:
            if log == "text":
                w.add_text("notes", "hello", 0)
            else:
                w.add_hparams({"lr": 0.1}, {"hparam/acc": 0.5})
        assert _names(tmp_path) == ["run"]

    def test_exit_on_bad_argument_is_discarded(self, tmp_path):
        """``sys.exit(2)`` during setup (argparse) is not even recorded as failed."""
        script = tmp_path / "train.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys, vibetrack
                vibetrack.init(project="demo", name="run", project_folder={str(tmp_path / 'proj')!r},
                               log_dir={str(tmp_path / 'logs' / 'run')!r}, system_metrics_interval=0)
                sys.exit(2)
                """
            )
        )
        env = _script_env()
        env.pop("VIBETRACK_DISCARD_FAILED_SECS", None)
        proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
        assert proc.returncode == 2
        assert _names(tmp_path) == []

    @pytest.mark.skipif(not hasattr(signal, "SIGINT") or os.name == "nt", reason="POSIX only")
    def test_ctrl_c_in_a_real_process_is_discarded(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import time, vibetrack
                vibetrack.init(project="demo", name="run", project_folder={str(tmp_path / 'proj')!r},
                               log_dir={str(tmp_path / 'logs' / 'run')!r}, system_metrics_interval=0)
                vibetrack.log({{"loss": 1.0}}, step=0)
                print("ready", flush=True)
                time.sleep(60)
                """
            )
        )
        env = _script_env()
        env.pop("VIBETRACK_DISCARD_FAILED_SECS", None)
        proc = subprocess.Popen(
            [sys.executable, str(script)], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=30)
        assert "interrupted after" in err and "discarded" in err
        assert _names(tmp_path) == []

    def test_uncaught_exception_at_exit_is_discarded(self, tmp_path):
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
        env = _script_env()
        env.pop("VIBETRACK_DISCARD_FAILED_SECS", None)
        proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "RuntimeError: CUDA out of memory" in proc.stderr  # Python's traceback
        assert "discarded" in proc.stderr
        assert _names(tmp_path) == []
        assert not (tmp_path / "logs" / "run").exists()


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
