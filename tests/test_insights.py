"""Run summaries and comparisons in vibetrack.insights."""

import random
import subprocess
import sys

import pytest

from vibetrack import insights
from vibetrack.db import Database
from vibetrack.insights import (
    compare_runs,
    dashboard_url,
    end_of_run_report,
    format_compare,
    format_overview,
    format_runs,
    infer_goal,
    list_runs,
    pick_primary,
    run_overview,
    spearman,
    summarize_series,
)
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


def _sweep(tmp_path, runs):
    folder = tmp_path / "proj"
    for name, cfg, fn in runs:
        w = SummaryWriter(
            str(tmp_path / "logs" / name),
            name=name,
            project="sweep",
            project_folder=str(folder),
            config=cfg,
            system_metrics_interval=0,
            report=False,
        )
        for step in range(20):
            for tag, value in fn(step).items():
                w.add_scalar(tag, value, step)
        w.close()
    return str(folder)


@pytest.fixture
def sweep(tmp_path):
    def acc_curve(lr):
        return lambda s: {
            "val/acc": min(1.0, s * lr) if lr < 0.08 else (0.9 if s < 10 else 0.4),
            "train/loss": 1.0 / (s + 1) + lr,
            "system/cpu_percent": 50.0,
        }

    runs = [
        (f"lr{lr}", {"lr": lr, "opt": opt, "seed": 0, "model": {"depth": 4}}, acc_curve(lr))
        for lr, opt in [(0.01, "sgd"), (0.02, "sgd"), (0.04, "adam"), (0.1, "adam")]
    ]
    folder = _sweep(tmp_path, runs)
    reader = RunReader(folder)
    yield folder, reader.experiments(project=None)
    reader.close()


class TestGoals:
    @pytest.mark.parametrize(
        "tag,goal",
        [
            ("train/loss", "min"),
            ("val_loss", "min"),
            ("lossD", "min"),
            ("train/reward_loss", "min"),
            ("eval/return", "max"),
            ("val/acc_top1", "max"),
            ("acc1", "max"),
            ("success_rate", "max"),
            ("fid", "min"),
            ("rollout/ep_rew_mean", "max"),
            ("lr", None),
            ("grad_norm", None),
        ],
    )
    def test_infer_goal(self, tag, goal):
        assert infer_goal(tag) == goal

    def test_pick_primary_prefers_declared_then_eval_max(self):
        tags = [{"train/loss", "val/loss", "val/acc", "lr"}]
        assert pick_primary(tags)[0] == "val/acc"
        declared = {"val/loss": {"goal": "min", "primary": True}}
        assert pick_primary(tags, declared)[:2] == ("val/loss", "min")
        assert pick_primary(tags, metric="lr")[:2] == ("lr", "max")  # assumed, reason says so
        assert pick_primary([{"lr", "epoch"}])[0] is None


class TestScalarStats:
    @pytest.mark.parametrize("precache", [0, 60])
    def test_matches_python_reference(self, tmp_path, precache):
        rng = random.Random(0)
        db = Database(tmp_path / "x.db", precache_secs=precache)
        eid = db.create_experiment("e", project="p")
        rows, ref = [], {}
        for tag in ("a", "b", "system/cpu"):
            for step in range(50):
                v = rng.uniform(-5, 5)
                rows.append((eid, tag, step, v, 1000.0 + step))
                ref.setdefault(tag, []).append((step, v))
        db.add_scalars_bulk(rows)
        stats = db.get_scalar_stats(eid)
        for tag, pts in ref.items():
            st = stats[tag]
            lo = min(pts, key=lambda p: p[1])
            hi = max(pts, key=lambda p: p[1])
            assert st["n"] == 50
            assert (st["min"], st["min_step"]) == (lo[1], lo[0])
            assert (st["max"], st["max_step"]) == (hi[1], hi[0])
            assert (st["last"], st["last_step"]) == (pts[-1][1], 49)
        assert set(db.get_scalar_stats(eid, tags=["a"])) == {"a"}
        assert db.get_scalar_stats(eid, tags=[]) == {}
        assert db.get_last_activity(eid)["step"] == 49
        db.close()


class TestSeries:
    def test_summary_best_regression_and_samples(self):
        rows = [{"step": s, "value": v} for s, v in enumerate([0.1, 0.5, 0.9, 0.8, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])]
        m = summarize_series(rows, "max")
        assert (m["best"], m["best_step"]) == (0.9, 2)
        assert m["last"] == 0.3 and m["regression"] == pytest.approx(0.75)
        assert m["samples"][0] == [0, 0.1] and m["samples"][-1] == [9, 0.3]
        assert m["recent"] == "flat"

    def test_ignores_non_finite(self):
        rows = [{"step": 0, "value": float("nan")}, {"step": 1, "value": 2.0}]
        assert summarize_series(rows, "min")["n"] == 1

    def test_spearman(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        assert spearman([1, 1, 1], [1, 2, 3]) is None


class TestCompare:
    def test_ranking_params_and_effects(self, sweep):
        _, exps = sweep
        cmp = compare_runs(exps)
        assert cmp["primary"]["metric"] == "val/acc"
        assert cmp["primary"]["goal"] == "max"
        ranking = [r["run"] for r in cmp["ranking"]]
        assert ranking[0] == "lr0.1"  # best value 0.9 at step 0..9
        assert cmp["ranking"][0]["delta_pct"] == 0
        assert set(cmp["params"]["varying"]) == {"lr", "opt"}
        assert cmp["params"]["constant"] == {"seed": 0, "model.depth": 4}
        assert "system/cpu_percent" not in cmp["metrics"]
        lr_effect = next(e for e in cmp["effects"] if e["param"] == "lr")
        assert lr_effect["kind"] == "numeric" and lr_effect["spearman"] > 0
        opt_effect = next(e for e in cmp["effects"] if e["param"] == "opt")
        assert opt_effect["kind"] == "categorical"
        # The diverging run ended far below its best.
        assert cmp["metrics"]["val/acc"]["runs"]["lr0.1"]["regression"] > 0.5

    def test_markdown_is_compact_and_ascii(self, sweep):
        folder, exps = sweep
        text = format_compare(compare_runs(exps), project_folder=folder)
        assert text.isascii()
        assert "Ranked by **val/acc** (max" in text
        assert "| 1 | lr0.1 | finished |" in text
        assert "Same in all runs: model.depth=4, seed=0" in text
        assert "fell back" in text
        assert "Param effects on val/acc" in text
        assert "vibetrack open --project-folder" in text

    def test_explicit_metric_and_goal(self, sweep):
        _, exps = sweep
        cmp = compare_runs(exps, metric="train/loss", goal="min")
        assert cmp["ranking"][0]["run"] == "lr0.01"


class TestOverviewAndList:
    def test_overview_and_markdown(self, sweep):
        folder, exps = sweep
        exp = next(e for e in exps if e.name == "lr0.1")
        ov = run_overview(exp)
        assert ov["run"]["status"] == "finished"
        assert ov["config"]["model.depth"] == 4
        assert list(ov["metrics"])[0] == "val/acc"
        assert ov["system_metrics"] == ["system/cpu_percent"]
        text = format_overview(ov, project_folder=folder)
        assert text.isascii()
        assert "# Run `lr0.1`" in text and "| val/acc | max | 20 |" in text
        assert "Curves (step:value samples):" in text

    def test_list_rows(self, sweep):
        folder, exps = sweep
        rows = list_runs(exps)
        assert [r["status"] for r in rows] == ["finished"] * 4
        assert rows[0]["headline"][0]["tag"] == "val/acc"
        text = format_runs(rows, project_folder=folder)
        assert "# vibetrack runs (4) - project `sweep`" in text
        assert "val/acc" in text

    def test_report_mentions_rank(self, sweep):
        folder, exps = sweep
        exp = next(e for e in exps if e.name == "lr0.1")
        text = end_of_run_report(exp._db, exp.experiment_id, project_folder=folder)
        assert text.startswith("== vibetrack: run 'lr0.1' finished")
        assert "rank:    #1 of 4 runs by val/acc" in text


def test_dashboard_url_encodes_names():
    url = dashboard_url("my proj", ["a b", "c&d"])
    assert url == "http://127.0.0.1:6116/my%20proj?run=a+b&run=c%26d&tab=runs"


def test_dead_pid_is_crashed(tmp_path):
    proc = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(proc.stdout)
    db = Database(tmp_path / "x.db")
    eid = db.create_experiment("e", project="p")
    import time

    db.update_run_info(eid, info={"pid": dead_pid, "host": insights.this_host()}, status="running", started_at=time.time() - 5)
    from vibetrack.reader import ExperimentReader

    st = insights.run_status(ExperimentReader(db, eid, "e"))
    assert st["status"] == "crashed"
    db.close()
