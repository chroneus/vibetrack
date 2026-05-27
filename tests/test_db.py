"""Tests for the SQLite backend."""

import json
import sqlite3
import threading
import time

import pytest

from vibetrack.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


class TestExperiments:
    def test_create_and_get(self, db):
        exp_id = db.create_experiment("run_1", config={"lr": 0.01})
        assert exp_id == 1
        row = db.get_experiment(exp_id)
        assert row["name"] == "run_1"
        assert json.loads(row["config"]) == {"lr": 0.01}
        assert row["created_at"] > 0


class TestScalars:
    def test_add_and_get(self, db):
        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "loss", 0.9, 0)
        db.add_scalar(exp_id, "loss", 0.7, 1)
        db.add_scalar(exp_id, "loss", 0.5, 2)
        rows = db.get_scalars(exp_id, "loss")
        assert len(rows) == 3
        assert [r["value"] for r in rows] == [0.9, 0.7, 0.5]
        assert [r["step"] for r in rows] == [0, 1, 2]

    def test_bulk_insert(self, db):
        exp_id = db.create_experiment("bulk")
        t = time.time()
        rows = [(exp_id, "loss", i, 1.0 / (i + 1), t) for i in range(10000)]
        db.add_scalars_bulk(rows)
        result = db.get_scalars(exp_id, "loss")
        assert len(result) == 10000

    def test_bulk_insert_performance(self, db):
        """10k inserts should complete in under 1 second."""
        exp_id = db.create_experiment("perf")
        t = time.time()
        rows = [(exp_id, "metric", i, float(i), t) for i in range(10000)]
        start = time.time()
        db.add_scalars_bulk(rows)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Bulk insert took {elapsed:.2f}s"

    def test_scalar_tags(self, db):
        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "loss", 0.5, 0)
        db.add_scalar(exp_id, "acc", 0.8, 0)
        db.add_scalar(exp_id, "lr", 0.01, 0)
        tags = db.get_scalar_tags(exp_id)
        assert tags == ["acc", "loss", "lr"]

    def test_ordering_by_step_not_insertion(self, db):
        """Scalars inserted out of step order must be returned sorted by step."""
        exp_id = db.create_experiment("run")
        t = time.time()
        # Insert in reverse step order
        for step in [9, 3, 6, 0]:
            db.add_scalar(exp_id, "metric", float(step), step)
        rows = db.get_scalars(exp_id, "metric")
        assert [r["step"] for r in rows] == [0, 3, 6, 9]

    def test_experiment_isolation(self, db):
        """Two experiments sharing a tag must not bleed data into each other."""
        e1 = db.create_experiment("exp1")
        e2 = db.create_experiment("exp2")
        t = time.time()
        bulk1 = [(e1, "loss", i, float(i), t) for i in range(500)]
        bulk2 = [(e2, "loss", i, float(i) * 10, t) for i in range(500)]
        db.add_scalars_bulk(bulk1 + bulk2)
        vals1 = {r["step"]: r["value"] for r in db.get_scalars(e1, "loss")}
        vals2 = {r["step"]: r["value"] for r in db.get_scalars(e2, "loss")}
        assert vals1[0] == 0.0
        assert vals2[0] == 0.0
        assert vals1[499] == 499.0
        assert vals2[499] == 4990.0


class TestTexts:
    def test_add_and_get(self, db):
        exp_id = db.create_experiment("run")
        db.add_text(exp_id, "log", "epoch 1 done", 0)
        db.add_text(exp_id, "log", "epoch 2 done", 1)
        rows = db.get_texts(exp_id, "log")
        assert len(rows) == 2
        assert rows[0]["value"] == "epoch 1 done"

    def test_text_tags_after_insert(self, db):
        exp_id = db.create_experiment("run")
        db.add_text(exp_id, "notes", "hello", 0)
        db.add_text(exp_id, "log", "step done", 0)
        db.add_text(exp_id, "notes", "world", 1)
        tags = db.get_text_tags(exp_id)
        assert tags == ["log", "notes"]


class TestImages:
    def test_add_and_get(self, db):
        exp_id = db.create_experiment("run")
        db.add_image(exp_id, "sample", "/path/img.png", 0)
        rows = db.get_images(exp_id, "sample")
        assert len(rows) == 1
        assert rows[0]["path"] == "/path/img.png"


class TestHistograms:
    def test_add_and_get(self, db):
        exp_id = db.create_experiment("run")
        db.add_histogram(exp_id, "weights", [0.0, 0.5, 1.0], [10.0, 20.0], 0)
        rows = db.get_histograms(exp_id, "weights")
        assert len(rows) == 1
        assert rows[0]["bins"] == [0.0, 0.5, 1.0]
        assert rows[0]["counts"] == [10.0, 20.0]

    def test_histogram_tags_after_insert(self, db):
        exp_id = db.create_experiment("run")
        db.add_histogram(exp_id, "weights", [0.0, 1.0], [5.0], 0)
        db.add_histogram(exp_id, "biases", [0.0, 1.0], [3.0], 0)
        tags = db.get_histogram_tags(exp_id)
        assert tags == ["biases", "weights"]


class TestHparams:
    def test_add_and_get(self, db):
        exp_id = db.create_experiment("run")
        db.add_hparams(exp_id, {"lr": 0.01, "batch_size": 32, "model": "resnet"})
        hp = db.get_hparams(exp_id)
        assert hp == {"lr": 0.01, "batch_size": 32, "model": "resnet"}

    def test_upsert(self, db):
        exp_id = db.create_experiment("run")
        db.add_hparams(exp_id, {"lr": 0.01})
        db.add_hparams(exp_id, {"lr": 0.001})
        hp = db.get_hparams(exp_id)
        assert hp["lr"] == 0.001


class TestWALMode:
    def test_wal_journal_mode(self, tmp_path):
        """DB must be opened in WAL mode for concurrent reader + writer support."""
        db = Database(tmp_path / "wal.db")
        db.create_experiment("probe")
        # Inspect via a raw sqlite3 connection
        conn = sqlite3.connect(str(tmp_path / "wal.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        db.close()
        assert mode == "wal"


class TestConcurrency:
    def test_concurrent_reads(self, db):
        exp_id = db.create_experiment("run")
        for i in range(100):
            db.add_scalar(exp_id, "loss", float(i), i)

        errors = []

        def reader():
            try:
                rows = db.get_scalars(exp_id, "loss")
                assert len(rows) == 100
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent read errors: {errors}"

    def test_concurrent_writes_no_corruption(self, tmp_path):
        """Multiple threads writing distinct scalars must not lose or corrupt rows."""
        db = Database(tmp_path / "concurrent.db")
        exp_id = db.create_experiment("run")
        errors = []
        n_threads = 8
        steps_per_thread = 200

        def writer(thread_idx):
            try:
                t = time.time()
                rows = [
                    (exp_id, f"metric_{thread_idx}", step, float(step), t)
                    for step in range(steps_per_thread)
                ]
                db.add_scalars_bulk(rows)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        for i in range(n_threads):
            rows = db.get_scalars(exp_id, f"metric_{i}")
            assert (
                len(rows) == steps_per_thread
            ), f"metric_{i}: expected {steps_per_thread} rows, got {len(rows)}"
        db.close()

    def test_concurrent_create_same_name_reuses_existing_row(self, tmp_path):
        db_path = tmp_path / "same_name.db"
        Database(db_path).close()

        created_ids = []
        errors = []
        barrier = threading.Barrier(2)

        def creator():
            try:
                db = Database(db_path)
                barrier.wait()
                created_ids.append(db.create_experiment("shared"))
                db.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=creator) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent create errors: {errors}"

        db = Database(db_path)
        rows = [r for r in db.list_experiments() if r["name"] == "shared"]
        assert len(rows) == 1
        assert len(set(created_ids)) == 1
        db.close()


class TestPrecache:
    def test_no_file_until_close(self, tmp_path):
        """DB file should not exist during precache, only after close()."""
        db_path = tmp_path / "precache.db"
        db = Database(db_path, precache_secs=60)

        exp_id = db.create_experiment("run_1", config={"lr": 0.01})
        db.add_scalar(exp_id, "loss", 0.9, 0)
        db.add_scalar(exp_id, "loss", 0.5, 1)
        db.add_text(exp_id, "log", "hello", 0)
        db.add_image(exp_id, "img", "/path/a.png", 0)
        db.add_histogram(exp_id, "w", [0.0, 0.5, 1.0], [5.0, 10.0], 0)
        db.add_hparams(exp_id, {"lr": 0.01})

        assert not db_path.exists()
        db.close()
        assert db_path.exists()

        db2 = Database(db_path)
        exp = db2.get_experiment_by_name("run_1")
        assert exp is not None
        assert len(db2.get_scalars(exp["id"], "loss")) == 2
        assert len(db2.get_texts(exp["id"], "log")) == 1
        assert len(db2.get_images(exp["id"], "img")) == 1
        assert len(db2.get_histograms(exp["id"], "w")) == 1
        assert db2.get_hparams(exp["id"])["lr"] == 0.01
        db2.close()

    def test_reads_from_cache(self, tmp_path):
        """All read methods should return data from in-memory cache."""
        db_path = tmp_path / "cache_read.db"
        db = Database(db_path, precache_secs=60)

        exp_id = db.create_experiment("run_1", config={"lr": 0.01})
        db.add_scalar(exp_id, "loss", 0.9, 0)
        db.add_scalar(exp_id, "acc", 0.8, 0)
        db.add_text(exp_id, "log", "started", 0)
        db.add_image(exp_id, "img", "/a.png", 0)
        db.add_histogram(exp_id, "w", [0.0, 1.0], [5.0], 0)
        db.add_hparams(exp_id, {"lr": 0.01, "bs": 32})

        assert not db_path.exists()

        assert db.get_experiment(exp_id) is not None
        assert db.get_experiment(exp_id)["name"] == "run_1"
        assert db.get_experiment_by_name("run_1") is not None
        assert db.get_experiment_by_name("nope") is None
        assert len(db.list_experiments()) == 1

        rows = db.get_scalars(exp_id, "loss")
        assert len(rows) == 1 and rows[0]["value"] == 0.9
        assert db.get_scalar_tags(exp_id) == ["acc", "loss"]

        texts = db.get_texts(exp_id, "log")
        assert len(texts) == 1 and texts[0]["value"] == "started"

        imgs = db.get_images(exp_id, "img")
        assert len(imgs) == 1 and imgs[0]["path"] == "/a.png"

        hists = db.get_histograms(exp_id, "w")
        assert len(hists) == 1 and hists[0]["bins"] == [0.0, 1.0]

        assert db.get_hparams(exp_id) == {"lr": 0.01, "bs": 32}
        db.close()

    def test_timeout_triggers_flush(self, tmp_path):
        """After precache_secs expires, timer should materialize the DB.

        Uses ``wait_for_materialize`` (an internal event) rather than a
        fixed ``time.sleep`` — eliminates the race window on loaded CI
        machines without slowing the happy path.
        """
        db_path = tmp_path / "timeout.db"
        db = Database(db_path, precache_secs=0.3)

        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "loss", 0.5, 0)

        assert not db_path.exists()
        assert db.wait_for_materialize(timeout=5.0), "materialization timed out"
        assert db_path.exists()

        db2 = Database(db_path)
        exp = db2.get_experiment_by_name("run")
        assert exp is not None
        rows = db2.get_scalars(exp["id"], "loss")
        assert len(rows) == 1 and rows[0]["value"] == 0.5
        db2.close()
        db.close()

    def test_zero_means_normal(self, tmp_path):
        """precache_secs=0 should behave identically to the default."""
        db_path = tmp_path / "normal.db"
        db = Database(db_path, precache_secs=0)
        assert db_path.exists()
        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "x", 1.0, 0)
        assert len(db.get_scalars(exp_id, "x")) == 1
        db.close()

    def test_bulk_insert_during_precache(self, tmp_path):
        """Bulk insert should work during precache and survive materialization."""
        db_path = tmp_path / "bulk.db"
        db = Database(db_path, precache_secs=60)

        exp_id = db.create_experiment("run")
        t = time.time()
        rows = [(exp_id, "loss", i, 1.0 / (i + 1), t) for i in range(1000)]
        db.add_scalars_bulk(rows)

        assert len(db.get_scalars(exp_id, "loss")) == 1000

        db.close()
        assert db_path.exists()
        db2 = Database(db_path)
        exp = db2.get_experiment_by_name("run")
        assert len(db2.get_scalars(exp["id"], "loss")) == 1000
        db2.close()

    def test_hparam_upsert_during_precache(self, tmp_path):
        """Hparam upsert should work correctly in precache mode."""
        db_path = tmp_path / "hparam_upsert.db"
        db = Database(db_path, precache_secs=60)

        exp_id = db.create_experiment("run")
        db.add_hparams(exp_id, {"lr": 0.01})
        db.add_hparams(exp_id, {"lr": 0.001})

        assert db.get_hparams(exp_id)["lr"] == 0.001
        db.close()

    def test_timer_and_close_race_no_duplication(self, tmp_path):
        """Concurrent timer flush and close() must not duplicate rows in the DB."""
        db_path = tmp_path / "race.db"
        db = Database(db_path, precache_secs=0.15)

        exp_id = db.create_experiment("run")
        t = time.time()
        rows = [(exp_id, "loss", i, float(i), t) for i in range(200)]
        db.add_scalars_bulk(rows)

        # Wait until the timer has materialized, then immediately close.
        # Deterministic wait instead of fixed-duration sleep.
        assert db.wait_for_materialize(timeout=5.0), "materialization timed out"
        db.close()

        db2 = Database(db_path)
        exp = db2.get_experiment_by_name("run")
        result = db2.get_scalars(exp["id"], "loss")
        assert (
            len(result) == 200
        ), f"Expected 200 rows, got {len(result)} (possible duplication)"
        db2.close()


class TestBulkLoaders:
    """``get_all_*`` should return ``{tag: rows}`` in a single query."""

    def test_get_all_scalars(self, tmp_path):
        db_path = tmp_path / "bulk_scalars.db"
        db = Database(db_path)
        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "loss", 1.0, 0)
        db.add_scalar(exp_id, "loss", 0.5, 1)
        db.add_scalar(exp_id, "acc", 0.9, 0)
        result = db.get_all_scalars(exp_id)
        assert set(result) == {"loss", "acc"}
        assert [r["step"] for r in result["loss"]] == [0, 1]
        assert [r["value"] for r in result["loss"]] == [1.0, 0.5]
        assert [r["value"] for r in result["acc"]] == [0.9]
        db.close()

    def test_get_all_images(self, tmp_path):
        db_path = tmp_path / "bulk_images.db"
        db = Database(db_path)
        exp_id = db.create_experiment("run")
        db.add_image(exp_id, "sample", "media/img_0.png", 0)
        db.add_image(exp_id, "sample", "media/img_1.png", 1)
        db.add_image(exp_id, "grid", "media/grid_0.png", 0)
        result = db.get_all_images(exp_id)
        assert set(result) == {"sample", "grid"}
        assert len(result["sample"]) == 2
        assert len(result["grid"]) == 1
        db.close()

    def test_get_all_scalars_precache(self, tmp_path):
        """Bulk loader must also work when data is still in-memory."""
        db_path = tmp_path / "bulk_precache.db"
        db = Database(db_path, precache_secs=60)
        exp_id = db.create_experiment("run")
        for i in range(5):
            db.add_scalar(exp_id, "loss", float(i), i)
        result = db.get_all_scalars(exp_id)
        assert set(result) == {"loss"}
        assert len(result["loss"]) == 5
        assert [r["step"] for r in result["loss"]] == [0, 1, 2, 3, 4]
        db.close()


class TestPrecacheFailureRecovery:
    """If precache materialization raises, caches must be preserved."""

    def test_materialize_failure_preserves_caches(self, tmp_path, monkeypatch):
        db_path = tmp_path / "fail.db"
        db = Database(db_path, precache_secs=60)
        exp_id = db.create_experiment("run")
        db.add_scalar(exp_id, "loss", 1.0, 0)
        db.add_scalar(exp_id, "loss", 2.0, 1)

        # Force ``add_scalars_bulk`` to fail during the replay; this happens
        # *after* the precache flag has flipped, so the failure path must
        # restore it and keep the in-memory rows intact.
        orig = db.add_scalars_bulk

        call_count = {"n": 0}

        def explode(rows):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated DB failure")
            return orig(rows)

        monkeypatch.setattr(db, "add_scalars_bulk", explode)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            db._materialize()

        # Cache should still hold the rows, ready for retry.
        assert db._precache_active
        assert len(db._cache_scalars) == 2
        # And the read API still returns them via the cache path.
        rows = db.get_scalars(exp_id, "loss")
        assert len(rows) == 2
        db.close()

    def test_wait_for_materialize_immediate_non_precache(self, tmp_path):
        db = Database(tmp_path / "imm.db", precache_secs=0)
        # Non-precache DBs are "materialized" from the start.
        assert db.wait_for_materialize(timeout=0.0)
        db.close()
