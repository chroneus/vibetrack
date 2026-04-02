"""Tests for SummaryWriter — TensorBoard & W&B API compatibility."""

import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


@pytest.fixture
def project_folder(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def log_dir(project_folder):
    return str(project_folder / "test_run")


def _project_db_path(log_dir: str) -> str:
    return str(Path(log_dir).parent / "vibetrack.db")


def _open_project_db(log_dir: str) -> Database:
    return Database(_project_db_path(log_dir))


def _cleanup_active_writer(vibetrack_module) -> None:
    writer = getattr(vibetrack_module, "_active_writer", None)
    if writer is not None:
        writer.close()
    vibetrack_module._active_writer = None


class TestSummaryWriter:
    def test_add_scalar(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 0.9, 0)
            w.add_scalar("loss", 0.7, 1)
            w.add_scalar("loss", 0.5, 2)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 3
        assert [r["value"] for r in rows] == [0.9, 0.7, 0.5]
        db.close()

    def test_add_scalars(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalars("metrics", {"loss": 0.5, "acc": 0.8}, 0)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        tags = db.get_scalar_tags(exp["id"])
        assert "metrics/loss" in tags
        assert "metrics/acc" in tags
        db.close()

    def test_add_text(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_text("notes", "started training", 0)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_texts(exp["id"], "notes")
        assert rows[0]["value"] == "started training"
        db.close()

    def test_add_histogram(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_histogram("weights", [0.1, 0.2, 0.3, 0.4, 0.5], 0)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_histograms(exp["id"], "weights")
        assert len(rows) == 1
        assert len(rows[0]["bins"]) > 0
        db.close()

    def test_add_hparams(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_hparams({"lr": 0.01, "bs": 32}, {"loss": 0.1})

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        hp = db.get_hparams(exp["id"])
        assert hp["lr"] == 0.01
        assert hp["bs"] == 32
        db.close()

    def test_auto_step(self, log_dir):
        """Steps auto-increment when not provided."""
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 1.0)
            w.add_scalar("loss", 0.5)
            w.add_scalar("loss", 0.3)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert [r["step"] for r in rows] == [0, 1, 2]
        db.close()

    def test_buffering(self, log_dir):
        """Scalars are buffered and flushed in bulk."""
        w = SummaryWriter(log_dir, max_queue=5, project_folder=str(Path(log_dir).parent))
        for i in range(4):
            w.add_scalar("x", float(i), i)
        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "x")) == 0

        w.add_scalar("x", 4.0, 4)
        rows = db.get_scalars(exp["id"], "x")
        assert len(rows) == 5
        w.close()
        db.close()

    def test_context_manager(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("v", 1.0, 0)
        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "v")) == 1
        db.close()

    def test_config_stored(self, project_folder):
        run_dir = str(project_folder / "configured")
        cfg = {"lr": 0.001, "epochs": 10}
        with SummaryWriter(run_dir, config=cfg, project_folder=str(project_folder)):
            pass
        db = Database(project_folder / "vibetrack.db")
        exp = db.get_experiment_by_name("configured")
        hp = db.get_hparams(exp["id"])
        assert hp["lr"] == 0.001
        db.close()

    def test_wb_style_log(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.log({"loss": 0.5, "acc": 0.9}, step=0)
            w.log({"loss": 0.3, "acc": 0.95}, step=1)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        loss = db.get_scalars(exp["id"], "loss")
        assert len(loss) == 2
        assert loss[1]["value"] == 0.3
        db.close()

    def test_flush_is_compatibility_noop(self, log_dir):
        """flush() should not force buffered scalars into the DB."""
        w = SummaryWriter(log_dir, max_queue=1000, project_folder=str(Path(log_dir).parent))
        for i in range(5):
            w.add_scalar("loss", float(i), i)
        w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "loss")) == 0
        w.close()
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 5
        db.close()

    def test_global_step_per_tag_independent(self, log_dir):
        """Auto-step counters for different tags must not interfere with each other."""
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 1.0)
            w.add_scalar("acc", 0.5)
            w.add_scalar("loss", 0.8)
            w.add_scalar("acc", 0.6)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        loss_steps = [r["step"] for r in db.get_scalars(exp["id"], "loss")]
        acc_steps = [r["step"] for r in db.get_scalars(exp["id"], "acc")]
        assert loss_steps == [0, 1]
        assert acc_steps == [0, 1]
        db.close()

    def test_runtime_errors_do_not_raise_into_caller(self, log_dir):
        with SummaryWriter(
            log_dir,
            system_metrics_interval=0,
            project_folder=str(Path(log_dir).parent),
        ) as w:
            assert w.experiment_id > 0
            with mock.patch.object(w._db, "add_text", side_effect=RuntimeError("boom")):
                w.add_text("notes", "should not raise", 0)
            w.add_scalar("loss", 1.0, 0)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 1
        db.close()

    def test_concurrent_startup_same_run_does_not_error(self, project_folder):
        log_dir = str(project_folder / "shared_run")
        errors = []

        def make_writer():
            try:
                w = SummaryWriter(
                    log_dir,
                    name="shared_run",
                    system_metrics_interval=0,
                    project_folder=str(project_folder),
                )
                w.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=make_writer) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"Concurrent SummaryWriter errors: {errors}"
        db = Database(project_folder / "vibetrack.db")
        rows = [row for row in db.list_experiments() if row["name"] == "shared_run"]
        assert len(rows) == 1
        db.close()


class TestPrecacheWriter:
    def test_precache_no_file_until_close(self, project_folder):
        """SummaryWriter with precache should not create files until close()."""
        run_dir = str(project_folder / "precache_run")
        db_file = project_folder / "vibetrack.db"

        w = SummaryWriter(
            run_dir,
            name="precache_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        w.add_scalar("loss", 0.9, 0)
        w.add_scalar("loss", 0.5, 1)
        w.flush()

        assert not db_file.exists()

        w.close()

        assert db_file.exists()
        db = Database(db_file)
        exp = db.get_experiment_by_name("precache_run")
        assert exp is not None
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 2
        assert [r["value"] for r in rows] == [0.9, 0.5]
        db.close()

    def test_precache_writer_reads_during_precache(self, project_folder):
        """flush() should not populate precache reads before close()."""
        run_dir = str(project_folder / "read_run")
        db_file = project_folder / "vibetrack.db"
        w = SummaryWriter(
            run_dir,
            name="read_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        w.add_scalar("loss", 0.9, 0)
        w.add_scalar("acc", 0.8, 0)
        w.flush()

        tags = w._db.get_scalar_tags(w.experiment_id)
        assert "loss" not in tags
        assert "acc" not in tags

        w.close()
        db = Database(db_file)
        exp = db.get_experiment_by_name("read_run")
        scalars = db.get_scalars(exp["id"], "loss")
        assert len(scalars) == 1
        assert scalars[0]["value"] == 0.9
        db.close()

    def test_precache_id_remap(self, project_folder):
        """After materialization, writer's exp_id should be a valid SQLite ID."""
        run_dir = str(project_folder / "remap_run")
        db_file = project_folder / "vibetrack.db"

        w = SummaryWriter(
            run_dir,
            name="remap_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        assert w.experiment_id < 0

        w.add_scalar("loss", 0.5, 0)
        w.close()

        db = Database(db_file)
        exp = db.get_experiment_by_name("remap_run")
        assert exp is not None
        assert exp["id"] > 0
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 1
        db.close()


class TestModuleAPI:
    def test_init_log_finish_uses_central_db_for_current_project(self, tmp_path, monkeypatch):
        import vibetrack

        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)

        project_dir = tmp_path / "demo_project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        writer = vibetrack.init(
            project=project_dir.name,
            name="wandb_run",
            log_dir=str(project_dir / "runs" / "wandb_run"),
            config={"lr": 0.01},
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})
        vibetrack.finish()

        reader = RunReader()
        exp = reader.experiment("wandb_run")
        assert exp is not None
        assert exp.log_dir == str(project_dir / "runs" / "wandb_run")
        assert [row["value"] for row in exp.scalars("loss")] == [0.9, 0.5]
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_init_log_finish(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="test_run",
            config={"lr": 0.01},
            log_dir=str(project_folder / "test_run"),
            project_folder=str(project_folder),
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})
        vibetrack.finish()

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        assert len(exps) >= 1
        data = exps[0].scalars("loss")
        assert len(data) == 2
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_finish_closes_active_writer(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="compat_run",
            log_dir=str(project_folder / "compat_run"),
            project_folder=str(project_folder),
            system_metrics_interval=0,
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.finish()
        assert writer._closed is True
        assert vibetrack._active_writer is None
        vibetrack.log({"loss": 0.5})

        reader = RunReader(str(project_folder))
        exp = reader.experiment("compat_run")
        assert exp is not None
        assert [row["value"] for row in exp.scalars("loss")] == [0.9]
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_init_with_precache(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="precache_api",
            config={"lr": 0.01},
            log_dir=str(project_folder / "precache_api"),
            project_folder=str(project_folder),
            precache_secs=60,
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})

        db_file = project_folder / "vibetrack.db"
        assert not db_file.exists()

        vibetrack.finish()
        assert db_file.exists()
        assert vibetrack._active_writer is None

        writer.close()
        vibetrack._active_writer = None
        assert db_file.exists()

    def test_log_without_init_does_not_raise(self):
        import vibetrack

        _cleanup_active_writer(vibetrack)
        vibetrack.finish()
        vibetrack.log({"loss": 0.5})
