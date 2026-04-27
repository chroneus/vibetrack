"""Tests for experiment comparison."""

import pytest

from vibetrack.compare import (
    compare_hparams,
    compare_scalars,
    find_all_tags,
    find_common_tags,
    summary_table,
)
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


@pytest.fixture
def experiments(tmp_path):
    """Create three experiments with different characteristics."""
    project_folder = tmp_path
    configs = [
        ("fast_lr", {"lr": 0.1}),
        ("slow_lr", {"lr": 0.001}),
        ("medium_lr", {"lr": 0.01}),
    ]
    for name, cfg in configs:
        run_dir = project_folder / name
        with SummaryWriter(
            str(run_dir), name=name, config=cfg, project_folder=str(project_folder)
        ) as w:
            for i in range(50):
                loss = cfg["lr"] * 10 / (i + 1)
                w.add_scalar("loss", loss, i)
                w.add_scalar("acc", 1.0 - loss, i)
            if name == "fast_lr":
                w.add_scalar("special_metric", 42.0, 0)

    reader = RunReader(str(project_folder))
    yield reader.experiments()
    reader.close()


class TestCompareScalars:
    def test_basic(self, experiments):
        result = compare_scalars(experiments, "loss")
        assert len(result) == 3
        for entry in result:
            assert "name" in entry
            assert "steps" in entry
            assert "values" in entry
            assert len(entry["steps"]) == 50

    def test_with_smoothing(self, experiments):
        result = compare_scalars(experiments, "loss", smoothing="ema", weight=0.6)
        for entry in result:
            assert "smoothed" in entry
            assert len(entry["smoothed"]) == 50

    def test_missing_tag_excludes_experiment(self, experiments):
        """compare_scalars must silently exclude experiments lacking the tag."""
        result = compare_scalars(experiments, "special_metric")
        assert len(result) == 1
        assert result[0]["name"] == "fast_lr"

    def test_nonexistent_tag_returns_empty(self, experiments):
        result = compare_scalars(experiments, "nonexistent")
        assert result == []

    def test_step_ranges_differ(self, tmp_path):
        """Experiments with different step counts must each return their own steps."""
        project_folder = tmp_path
        for name, n_steps in [("short", 10), ("long", 40)]:
            run_dir = project_folder / name
            with SummaryWriter(
                str(run_dir), name=name, project_folder=str(project_folder)
            ) as w:
                for i in range(n_steps):
                    w.add_scalar("loss", 1.0 / (i + 1), i)

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        result = compare_scalars(exps, "loss")
        lengths = {e["name"]: len(e["steps"]) for e in result}
        assert lengths["short"] == 10
        assert lengths["long"] == 40
        reader.close()


class TestFindTags:
    def test_common_tags_excludes_partial(self, experiments):
        common = find_common_tags(experiments)
        assert "loss" in common
        assert "acc" in common
        assert "special_metric" not in common

    def test_all_tags_includes_partial(self, experiments):
        all_t = find_all_tags(experiments)
        assert "loss" in all_t
        assert "acc" in all_t
        assert "special_metric" in all_t


class TestSummaryTable:
    def test_last_value_used(self, tmp_path):
        """summary_table should report the *last* logged value for each tag."""
        project_folder = tmp_path
        with SummaryWriter(
            str(project_folder / "run1"),
            name="run1",
            project_folder=str(project_folder),
        ) as w:
            for i in range(10):
                w.add_scalar("loss", float(10 - i), i)

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        table = summary_table(exps, tags=["loss"])
        assert len(table) == 1
        assert abs(table[0]["loss"] - 1.0) < 1e-9
        reader.close()

    def test_all_experiments_present(self, experiments):
        table = summary_table(experiments)
        names = {row["name"] for row in table}
        assert names == {"fast_lr", "slow_lr", "medium_lr"}

    def test_specific_tags(self, experiments):
        table = summary_table(experiments, tags=["loss"])
        for row in table:
            assert "loss" in row
            assert "acc" not in row


class TestCompareHparams:
    def test_basic(self, experiments):
        result = compare_hparams(experiments)
        assert {entry["name"] for entry in result} == {
            "fast_lr",
            "slow_lr",
            "medium_lr",
        }
