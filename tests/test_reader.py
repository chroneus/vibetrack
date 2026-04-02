"""Tests for reader access via the project-level central DB."""

from pathlib import Path

import pytest

from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


@pytest.fixture
def multi_run_dir(tmp_path):
    """Create a project folder with multiple experiment runs."""
    project_folder = tmp_path
    for name in ["exp_a", "exp_b", "exp_c"]:
        run_dir = project_folder / name
        with SummaryWriter(str(run_dir), name=name, project_folder=str(project_folder)) as w:
            for i in range(20):
                w.add_scalar("loss", 1.0 / (i + 1), i)
                w.add_scalar("acc", i / 20.0, i)
            w.add_hparams({"lr": 0.01 if name == "exp_a" else 0.001}, {})
            w.add_text("notes", f"run {name}", 0)
            w.add_histogram("weights", [0.1, 0.2, 0.3, 0.4, 0.5], global_step=0)
    return str(project_folder)


class TestRunReader:
    def test_central_db_defaults_to_current_project(self, tmp_path, monkeypatch):
        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        db = Database(central_db)
        db.create_experiment("run_a", project="project_a", log_dir=str(tmp_path / "a" / "run_a"))
        db.create_experiment("run_b", project="project_b", log_dir=str(tmp_path / "b" / "run_b"))
        db.close()

        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)
        project_dir = tmp_path / "project_a"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        reader = RunReader()
        names = {e.name for e in reader.experiments()}
        assert names == {"run_a"}
        reader.close()

    def test_discover_experiments(self, multi_run_dir):
        reader = RunReader(multi_run_dir)
        exps = reader.experiments()
        names = {e.name for e in exps}
        assert names == {"exp_a", "exp_b", "exp_c"}
        reader.close()

    def test_get_experiment_by_name(self, multi_run_dir):
        reader = RunReader(multi_run_dir)
        exp = reader.experiment("exp_a")
        assert exp is not None
        assert exp.name == "exp_a"
        reader.close()

    def test_rediscover_picks_up_new_experiments(self, tmp_path):
        """Experiments added after initial open are visible after _discover()."""
        project_folder = tmp_path
        with SummaryWriter(
            str(project_folder / "exp_a"),
            name="exp_a",
            project_folder=str(project_folder),
        ) as w:
            w.add_scalar("loss", 0.5, 0)

        reader = RunReader(str(project_folder))
        assert {e.name for e in reader.experiments()} == {"exp_a"}

        with SummaryWriter(
            str(project_folder / "exp_b"),
            name="exp_b",
            project_folder=str(project_folder),
        ) as w:
            w.add_scalar("loss", 0.9, 0)

        reader._discover()
        assert {e.name for e in reader.experiments()} == {"exp_a", "exp_b"}
        reader.close()

    def test_resolves_absolute_media_path(self, tmp_path):
        project_folder = tmp_path
        image_path = project_folder / "source.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        run_dir = project_folder / "exp_media"

        with SummaryWriter(
            str(run_dir),
            name="exp_media",
            project_folder=str(project_folder),
        ) as writer:
            writer.add_image("samples", str(image_path), 0)

        reader = RunReader(str(project_folder))
        exp = reader.experiment("exp_media")
        image = exp.images("samples")[0]
        assert Path(image["abs_path"]).is_absolute()
        assert image["abs_path"].endswith("exp_media/media/samples/0.png")
        reader.close()


class TestExperimentReader:
    def test_scalar_tags(self, multi_run_dir):
        with RunReader(multi_run_dir) as reader:
            exp = reader.experiment("exp_a")
            tags = exp.scalar_tags()
            assert "loss" in tags
            assert "acc" in tags

    def test_scalars_ordered_by_step(self, multi_run_dir):
        """Scalars must always be returned in ascending step order."""
        with RunReader(multi_run_dir) as reader:
            exp = reader.experiment("exp_a")
            data = exp.scalars("loss")
            assert len(data) == 20
            steps = [r["step"] for r in data]
            assert steps == sorted(steps)
            assert data[0]["value"] == 1.0

    def test_hparams(self, multi_run_dir):
        with RunReader(multi_run_dir) as reader:
            exp = reader.experiment("exp_a")
            hp = exp.hparams()
            assert hp["lr"] == 0.01

    def test_hparam_values_differ_across_experiments(self, multi_run_dir):
        """Each experiment must expose its own distinct hparam values."""
        with RunReader(multi_run_dir) as reader:
            lr_a = reader.experiment("exp_a").hparams()["lr"]
            lr_b = reader.experiment("exp_b").hparams()["lr"]
            assert lr_a != lr_b

    def test_missing_tag_returns_empty(self, multi_run_dir):
        """Querying a tag that doesn't exist must return an empty list, not crash."""
        with RunReader(multi_run_dir) as reader:
            exp = reader.experiment("exp_a")
            assert exp.scalars("nonexistent_tag") == []
