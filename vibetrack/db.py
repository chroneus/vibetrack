"""SQLite backend with WAL mode for fast concurrent reads and writes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_SCHEMA_VERSION = 3

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    project    TEXT NOT NULL DEFAULT '',
    log_dir    TEXT NOT NULL DEFAULT '',
    config     TEXT,  -- JSON
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_project_name ON experiments(project, name);

CREATE TABLE IF NOT EXISTS scalars (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    value         REAL    NOT NULL,
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_scalars_exp_tag ON scalars(experiment_id, tag);
CREATE INDEX IF NOT EXISTS idx_scalars_exp_tag_step ON scalars(experiment_id, tag, step);

CREATE TABLE IF NOT EXISTS texts (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    value         TEXT    NOT NULL,
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_texts_exp_tag ON texts(experiment_id, tag);

CREATE TABLE IF NOT EXISTS images (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    path          TEXT    NOT NULL,
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_images_exp_tag ON images(experiment_id, tag);

CREATE TABLE IF NOT EXISTS histograms (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    bins          TEXT    NOT NULL,  -- JSON array
    counts        TEXT    NOT NULL,  -- JSON array
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_histograms_exp_tag ON histograms(experiment_id, tag);

CREATE TABLE IF NOT EXISTS hparams (
    experiment_id INTEGER NOT NULL,
    key           TEXT    NOT NULL,
    value         TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    PRIMARY KEY (experiment_id, key)
);

CREATE TABLE IF NOT EXISTS audio (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    path          TEXT    NOT NULL,
    sample_rate   INTEGER,
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_audio_exp_tag ON audio(experiment_id, tag);

CREATE TABLE IF NOT EXISTS video (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    path          TEXT    NOT NULL,
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_video_exp_tag ON video(experiment_id, tag);

CREATE TABLE IF NOT EXISTS artifacts (
    experiment_id INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    path          TEXT    NOT NULL,
    metadata      TEXT,   -- JSON: {file_size, mime_type, original_filename, ...}
    wall_time     REAL    NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_exp_tag ON artifacts(experiment_id, tag);
"""


def central_db_dir() -> Path:
    """Return ``~/.vibetrack/``."""
    return Path.home() / ".vibetrack"


def central_db_path() -> Path:
    """Return ``~/.vibetrack/vibetrack.db``."""
    return central_db_dir() / "vibetrack.db"


def open_central_db() -> "Database":
    """Open (or create) the system-wide central database."""
    return Database(central_db_path())


class Database:
    """Thread-safe SQLite database with WAL mode and bulk insert support.

    Parameters
    ----------
    path : str or Path
        Path to the SQLite database file.
    precache_secs : float
        When > 0, hold all data in memory for this many seconds before
        creating the DB file.  If the process dies before the timeout
        (or before ``close()``), no file is left on disk.
    """

    def __init__(self, path: str | Path, precache_secs: float = 0) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.Lock()

        # ── Precache state ──────────────────────────────────────
        self._precache_secs = precache_secs
        self._precache_active = precache_secs > 0
        self._precache_deadline = (
            time.time() + precache_secs if precache_secs > 0 else 0.0
        )
        self._precache_timer: Optional[threading.Timer] = None
        self._materializing = False
        self._remap_callbacks: List[Callable[[Dict[int, int]], None]] = []

        # In-memory caches (used only during precache)
        self._cache_experiments: List[Dict[str, Any]] = []
        self._cache_scalars: List[Tuple[int, str, int, float, float]] = []
        self._cache_texts: List[Tuple[int, str, int, str, float]] = []
        self._cache_images: List[Tuple[int, str, int, str, float]] = []
        # histograms: (exp_id, tag, step, bins_json, counts_json, wall_time)
        self._cache_histograms: List[Tuple[int, str, int, str, str, float]] = []
        # hparams: (exp_id, key, json_value)
        self._cache_hparams: List[Tuple[int, str, str]] = []
        # audio: (exp_id, tag, step, path, sample_rate, wall_time)
        self._cache_audio: List[Tuple[int, str, int, str, Optional[int], float]] = []
        # video: (exp_id, tag, step, path, wall_time)
        self._cache_video: List[Tuple[int, str, int, str, float]] = []
        # artifacts: (exp_id, tag, step, path, metadata_json, wall_time)
        self._cache_artifacts: List[Tuple[int, str, int, str, Optional[str], float]] = (
            []
        )
        self._next_synthetic_id = -1

        if self._precache_active:
            # Start daemon timer — won't prevent process exit
            self._precache_timer = threading.Timer(precache_secs, self._materialize)
            self._precache_timer.daemon = True
            self._precache_timer.start()
        else:
            # Normal path: create DB immediately
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._init_schema(conn)

    # ── Precache internals ──────────────────────────────────────

    def _alloc_synthetic_id(self) -> int:
        eid = self._next_synthetic_id
        self._next_synthetic_id -= 1
        return eid

    def register_remap_callback(self, cb: Callable[[Dict[int, int]], None]) -> None:
        """Register a callback invoked with {synthetic_id: real_id} after materialize."""
        self._remap_callbacks.append(cb)

    def _materialize(self) -> None:
        """Flush in-memory caches to SQLite and switch to direct mode."""
        with self._lock:
            if not self._precache_active:
                return  # already materialized

            # 1. Create directory + DB + schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._init_schema(conn)

            # 2. Flip flag so replayed writes go to SQLite
            self._precache_active = False

            # 3. Replay experiments, build ID remap
            id_remap: Dict[int, int] = {}
            for exp in self._cache_experiments:
                config = json.loads(exp["config"]) if exp["config"] else None
                real_id = self.create_experiment(
                    exp["name"],
                    config,
                    project=exp.get("project", ""),
                    log_dir=exp.get("log_dir", ""),
                )
                id_remap[exp["id"]] = real_id

            # 4. Replay scalars (bulk)
            if self._cache_scalars:
                remapped = [
                    (id_remap.get(r[0], r[0]), r[1], r[2], r[3], r[4])
                    for r in self._cache_scalars
                ]
                self.add_scalars_bulk(remapped)

            # 5. Replay texts
            for row in self._cache_texts:
                self.add_text(
                    id_remap.get(row[0], row[0]),
                    row[1],
                    row[3],
                    row[2],
                    row[4],
                )

            # 6. Replay images
            for row in self._cache_images:
                self.add_image(
                    id_remap.get(row[0], row[0]),
                    row[1],
                    row[3],
                    row[2],
                    row[4],
                )

            # 7. Replay histograms (bins/counts already JSON strings)
            for row in self._cache_histograms:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO histograms"
                        "(experiment_id, tag, step, bins, counts, wall_time) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            id_remap.get(row[0], row[0]),
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                        ),
                    )

            # 8. Replay hparams (value already JSON string)
            for row in self._cache_hparams:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO hparams"
                        "(experiment_id, key, value) VALUES (?, ?, ?)",
                        (id_remap.get(row[0], row[0]), row[1], row[2]),
                    )

            # 9. Replay audio
            for row in self._cache_audio:
                self.add_audio(
                    id_remap.get(row[0], row[0]),
                    row[1],
                    row[3],
                    row[2],
                    row[4],
                    row[5],
                )

            # 10. Replay video
            for row in self._cache_video:
                self.add_video(
                    id_remap.get(row[0], row[0]),
                    row[1],
                    row[3],
                    row[2],
                    row[4],
                )

            # 11. Replay artifacts
            for row in self._cache_artifacts:
                self.add_artifact(
                    id_remap.get(row[0], row[0]),
                    row[1],
                    row[3],
                    row[4],
                    row[2],
                    row[5],
                )

            # 12. Fire callbacks
            for cb in self._remap_callbacks:
                cb(id_remap)

            # 13. Clear caches
            self._cache_experiments.clear()
            self._cache_scalars.clear()
            self._cache_texts.clear()
            self._cache_images.clear()
            self._cache_histograms.clear()
            self._cache_hparams.clear()
            self._cache_audio.clear()
            self._cache_video.clear()
            self._cache_artifacts.clear()

    def _check_and_maybe_materialize(self) -> bool:
        """If precache deadline has passed, materialize. Must hold self._lock.

        Returns True if we just materialized (caller should use SQLite path).
        """
        if self._materializing:
            return not self._precache_active
        if time.time() >= self._precache_deadline:
            self._materializing = True
            self._lock.release()
            try:
                self._materialize()
            finally:
                self._materializing = False
                self._lock.acquire()
            return True
        return False

    # ── Connection helpers ──────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path),
                timeout=10,
                check_same_thread=False,
            )
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        if not mode or str(mode[0]).lower() != "wal":
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_SQL)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
        else:
            stored = int(row["value"])
            if stored < 3:
                self._migrate_v2_to_v3(conn)
        conn.commit()

    @staticmethod
    def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
        """Add project and log_dir columns to experiments table."""
        cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
        if "project" not in cols:
            conn.execute(
                "ALTER TABLE experiments ADD COLUMN project TEXT NOT NULL DEFAULT ''"
            )
        if "log_dir" not in cols:
            conn.execute(
                "ALTER TABLE experiments ADD COLUMN log_dir TEXT NOT NULL DEFAULT ''"
            )
        # Replace old name-only index with project+name unique index
        conn.execute("DROP INDEX IF EXISTS idx_experiments_name")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "idx_experiments_project_name ON experiments(project, name)"
        )
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (str(_SCHEMA_VERSION),),
        )

    # ── Experiments ──────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        project: str = "",
        log_dir: str = "",
    ) -> int:
        config_json = json.dumps(config) if config else None
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through to SQLite
                    else:
                        eid = self._alloc_synthetic_id()
                        self._cache_experiments.append(
                            {
                                "id": eid,
                                "name": name,
                                "project": project,
                                "log_dir": log_dir,
                                "config": config_json,
                                "created_at": time.time(),
                            }
                        )
                        return eid
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM experiments WHERE project=? AND name=? "
                "ORDER BY created_at DESC LIMIT 1",
                (project, name),
            ).fetchone()
            if existing is not None:
                if config_json is not None:
                    conn.execute(
                        "UPDATE experiments SET config=COALESCE(config, ?) WHERE id=?",
                        (config_json, existing["id"]),
                    )
                return int(existing["id"])
            cur = conn.execute(
                "INSERT INTO experiments(name, project, log_dir, config, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, project, log_dir, config_json, time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_experiment(self, experiment_id: int) -> Optional[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        for exp in self._cache_experiments:
                            if exp["id"] == experiment_id:
                                return exp
                        return None
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()

    def get_experiment_by_name(
        self,
        name: str,
        project: Optional[str] = None,
    ) -> Optional[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        for exp in self._cache_experiments:
                            if exp["name"] == name and (
                                project is None or exp.get("project", "") == project
                            ):
                                return exp
                        return None
        with self._connect() as conn:
            if project is not None:
                return conn.execute(
                    "SELECT * FROM experiments WHERE project=? AND name=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (project, name),
                ).fetchone()
            return conn.execute(
                "SELECT * FROM experiments WHERE name=? "
                "ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()

    def list_experiments(self, project: Optional[str] = None) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        exps = self._cache_experiments
                        if project is not None:
                            exps = [e for e in exps if e.get("project", "") == project]
                        return sorted(
                            exps,
                            key=lambda e: e["created_at"],
                            reverse=True,
                        )
        with self._connect() as conn:
            if project is not None:
                return conn.execute(
                    "SELECT * FROM experiments WHERE project=? "
                    "ORDER BY created_at DESC",
                    (project,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()

    def list_projects(self) -> List[str]:
        """Return distinct project names ordered by most recent activity first."""
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        latest: Dict[str, float] = {}
                        for e in self._cache_experiments:
                            p = e.get("project", "")
                            ts = e.get("created_at", 0) or 0
                            if p not in latest or ts > latest[p]:
                                latest[p] = ts
                        return [
                            p for p, _ in sorted(
                                latest.items(), key=lambda kv: kv[1], reverse=True
                            )
                        ]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT project, MAX(created_at) AS last_active "
                "FROM experiments GROUP BY project ORDER BY last_active DESC"
            ).fetchall()
            return [r["project"] for r in rows]

    def get_max_step(self, experiment_id: int) -> Optional[int]:
        """Return the maximum step across all data tables, or None if no data."""
        _DATA_TABLES = (
            "scalars",
            "texts",
            "images",
            "audio",
            "video",
            "artifacts",
            "histograms",
        )
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through to SQLite
                    else:
                        max_step: Optional[int] = None
                        for cache, step_idx in [
                            (self._cache_scalars, 2),
                            (self._cache_texts, 2),
                            (self._cache_images, 2),
                            (self._cache_audio, 2),
                            (self._cache_video, 2),
                            (self._cache_artifacts, 2),
                            (self._cache_histograms, 2),
                        ]:
                            for row in cache:
                                if row[0] == experiment_id:
                                    s = row[step_idx]
                                    if max_step is None or s > max_step:
                                        max_step = s
                        return max_step
        with self._connect() as conn:
            parts = [
                f"SELECT MAX(step) AS m FROM {t} WHERE experiment_id=?"
                for t in _DATA_TABLES
            ]
            sql = f"SELECT MAX(m) AS max_step FROM ({' UNION ALL '.join(parts)})"
            row = conn.execute(sql, (experiment_id,) * len(_DATA_TABLES)).fetchone()
            return row["max_step"] if row and row["max_step"] is not None else None

    def find_next_suffix_name(self, base_name: str, project: str = "") -> str:
        """Given 'exp', return 'exp (2)' or 'exp (3)' — next available suffix."""
        import re

        pattern = re.compile(r"^" + re.escape(base_name) + r" \((\d+)\)$")
        max_n = 1  # base_name itself counts as 1

        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        for exp in self._cache_experiments:
                            if exp.get("project", "") != project:
                                continue
                            m = pattern.match(exp["name"])
                            if m:
                                max_n = max(max_n, int(m.group(1)))
                        return f"{base_name} ({max_n + 1})"

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM experiments WHERE project=? AND "
                "(name=? OR name LIKE ?)",
                (project, base_name, f"{base_name} (%"),
            ).fetchall()
            for row in rows:
                m = pattern.match(row["name"])
                if m:
                    max_n = max(max_n, int(m.group(1)))
        return f"{base_name} ({max_n + 1})"

    def delete_project(self, project: str) -> Optional[List[str]]:
        """Delete a project's experiments and all associated rows.

        Returns the deleted experiments' ``log_dir`` values, or ``None`` if the
        project does not exist.
        """
        if self._precache_active:
            self._materialize()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, log_dir FROM experiments WHERE project=?",
                (project,),
            ).fetchall()
            if not rows:
                return None

            exp_ids = [int(row["id"]) for row in rows]
            log_dirs = sorted({str(row["log_dir"]) for row in rows if row["log_dir"]})
            placeholders = ",".join("?" for _ in exp_ids)
            params = tuple(exp_ids)

            for table in [
                "scalars",
                "texts",
                "images",
                "histograms",
                "hparams",
                "audio",
                "video",
                "artifacts",
            ]:
                conn.execute(
                    f"DELETE FROM {table} WHERE experiment_id IN ({placeholders})",
                    params,
                )
            conn.execute(
                "DELETE FROM experiments WHERE project=?",
                (project,),
            )
            return log_dirs

    def delete_experiment(self, experiment_id: int) -> Optional[str]:
        """Delete an experiment and all its data.

        Returns the experiment's ``log_dir``, or ``None`` if not found.
        """
        if self._precache_active:
            self._materialize()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT log_dir FROM experiments WHERE id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                return None
            log_dir = str(row["log_dir"]) if row["log_dir"] else ""
            for table in [
                "scalars",
                "texts",
                "images",
                "histograms",
                "hparams",
                "audio",
                "video",
                "artifacts",
            ]:
                conn.execute(
                    f"DELETE FROM {table} WHERE experiment_id=?",
                    (experiment_id,),
                )
            conn.execute(
                "DELETE FROM experiments WHERE id=?",
                (experiment_id,),
            )
            return log_dir

    def rename_experiment(self, experiment_id: int, new_name: str) -> bool:
        """Rename an experiment. Returns True if successful."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Get the project of the experiment being renamed
            exp = conn.execute(
                "SELECT project FROM experiments WHERE id=?",
                (experiment_id,),
            ).fetchone()
            if exp is None:
                return False
            existing = conn.execute(
                "SELECT id FROM experiments WHERE project=? AND name=? "
                "ORDER BY created_at DESC LIMIT 1",
                (exp["project"], new_name),
            ).fetchone()
            if existing is not None and existing["id"] != experiment_id:
                return False
            cur = conn.execute(
                "UPDATE experiments SET name=? WHERE id=?",
                (new_name, experiment_id),
            )
            return cur.rowcount > 0

    def update_log_dir(self, experiment_id: int, new_log_dir: str) -> bool:
        """Update the log_dir of an experiment. Returns True if successful."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE experiments SET log_dir=? WHERE id=?",
                (new_log_dir, experiment_id),
            )
            return cur.rowcount > 0

    # ── Scalars ─────────────────────────────────────────────────

    def add_scalar(
        self,
        experiment_id: int,
        tag: str,
        value: float,
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_scalars.append(
                            (experiment_id, tag, step, float(value), wt)
                        )
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scalars(experiment_id, tag, step, value, wall_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (experiment_id, tag, step, value, wt),
            )

    def add_scalars_bulk(
        self,
        rows: Sequence[Tuple[int, str, int, float, float]],
    ) -> None:
        """Bulk insert scalars. Each row: (experiment_id, tag, step, value, wall_time)."""
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_scalars.extend(rows)
                        return
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO scalars(experiment_id, tag, step, value, wall_time) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def get_scalars(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {"step": r[2], "value": r[3], "wall_time": r[4]}
                            for r in self._cache_scalars
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, value, wall_time FROM scalars "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_scalar_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        tags = sorted(
                            {r[1] for r in self._cache_scalars if r[0] == experiment_id}
                        )
                        return tags
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM scalars WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Texts ───────────────────────────────────────────────────

    def add_text(
        self,
        experiment_id: int,
        tag: str,
        value: str,
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_texts.append((experiment_id, tag, step, value, wt))
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO texts(experiment_id, tag, step, value, wall_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (experiment_id, tag, step, value, wt),
            )

    def get_texts(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {"step": r[2], "value": r[3], "wall_time": r[4]}
                            for r in self._cache_texts
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, value, wall_time FROM texts "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_text_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {r[1] for r in self._cache_texts if r[0] == experiment_id}
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM texts WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Images ──────────────────────────────────────────────────

    def add_image(
        self,
        experiment_id: int,
        tag: str,
        path: str,
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_images.append((experiment_id, tag, step, path, wt))
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO images(experiment_id, tag, step, path, wall_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (experiment_id, tag, step, path, wt),
            )

    def get_images(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {"step": r[2], "path": r[3], "wall_time": r[4]}
                            for r in self._cache_images
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, path, wall_time FROM images "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_image_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {r[1] for r in self._cache_images if r[0] == experiment_id}
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM images WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Audio ────────────────────────────────────────────────────

    def add_audio(
        self,
        experiment_id: int,
        tag: str,
        path: str,
        step: int,
        sample_rate: Optional[int] = None,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_audio.append(
                            (experiment_id, tag, step, path, sample_rate, wt)
                        )
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audio(experiment_id, tag, step, path, sample_rate, wall_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, tag, step, path, sample_rate, wt),
            )

    def get_audio(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {
                                "step": r[2],
                                "path": r[3],
                                "sample_rate": r[4],
                                "wall_time": r[5],
                            }
                            for r in self._cache_audio
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, path, sample_rate, wall_time FROM audio "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_audio_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {r[1] for r in self._cache_audio if r[0] == experiment_id}
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM audio WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Video ────────────────────────────────────────────────────

    def add_video(
        self,
        experiment_id: int,
        tag: str,
        path: str,
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_video.append((experiment_id, tag, step, path, wt))
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO video(experiment_id, tag, step, path, wall_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (experiment_id, tag, step, path, wt),
            )

    def get_video(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {"step": r[2], "path": r[3], "wall_time": r[4]}
                            for r in self._cache_video
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, path, wall_time FROM video "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_video_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {r[1] for r in self._cache_video if r[0] == experiment_id}
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM video WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Artifacts ────────────────────────────────────────────────

    def add_artifact(
        self,
        experiment_id: int,
        tag: str,
        path: str,
        metadata_json: Optional[str],
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_artifacts.append(
                            (experiment_id, tag, step, path, metadata_json, wt)
                        )
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts"
                "(experiment_id, tag, step, path, metadata, wall_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, tag, step, path, metadata_json, wt),
            )

    def get_artifacts(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {
                                "step": r[2],
                                "path": r[3],
                                "metadata": r[4],
                                "wall_time": r[5],
                            }
                            for r in self._cache_artifacts
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            return conn.execute(
                "SELECT step, path, metadata, wall_time FROM artifacts "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()

    def get_artifact_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {
                                r[1]
                                for r in self._cache_artifacts
                                if r[0] == experiment_id
                            }
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM artifacts WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Histograms ──────────────────────────────────────────────

    def add_histogram(
        self,
        experiment_id: int,
        tag: str,
        bins: List[float],
        counts: List[float],
        step: int,
        wall_time: Optional[float] = None,
    ) -> None:
        wt = wall_time if wall_time is not None else time.time()
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        self._cache_histograms.append(
                            (
                                experiment_id,
                                tag,
                                step,
                                json.dumps(bins),
                                json.dumps(counts),
                                wt,
                            )
                        )
                        return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO histograms(experiment_id, tag, step, bins, counts, wall_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, tag, step, json.dumps(bins), json.dumps(counts), wt),
            )

    def get_histograms(
        self,
        experiment_id: int,
        tag: str,
    ) -> List[Dict[str, Any]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        results = [
                            {
                                "step": r[2],
                                "bins": json.loads(r[3]),
                                "counts": json.loads(r[4]),
                                "wall_time": r[5],
                            }
                            for r in self._cache_histograms
                            if r[0] == experiment_id and r[1] == tag
                        ]
                        results.sort(key=lambda r: r["step"])
                        return results
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step, bins, counts, wall_time FROM histograms "
                "WHERE experiment_id=? AND tag=? ORDER BY step",
                (experiment_id, tag),
            ).fetchall()
            return [
                {
                    "step": r["step"],
                    "bins": json.loads(r["bins"]),
                    "counts": json.loads(r["counts"]),
                    "wall_time": r["wall_time"],
                }
                for r in rows
            ]

    def get_histogram_tags(self, experiment_id: int) -> List[str]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return sorted(
                            {
                                r[1]
                                for r in self._cache_histograms
                                if r[0] == experiment_id
                            }
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM histograms WHERE experiment_id=? ORDER BY tag",
                (experiment_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    # ── Hyperparameters ─────────────────────────────────────────

    def add_hparams(
        self,
        experiment_id: int,
        hparams: Dict[str, Any],
    ) -> None:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        for k, v in hparams.items():
                            # Upsert: remove old entry if exists
                            self._cache_hparams = [
                                r
                                for r in self._cache_hparams
                                if not (r[0] == experiment_id and r[1] == k)
                            ]
                            self._cache_hparams.append(
                                (experiment_id, k, json.dumps(v))
                            )
                        return
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO hparams(experiment_id, key, value) "
                "VALUES (?, ?, ?)",
                [(experiment_id, k, json.dumps(v)) for k, v in hparams.items()],
            )

    def get_hparams(self, experiment_id: int) -> Dict[str, Any]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass  # fall through
                    else:
                        return {
                            r[1]: json.loads(r[2])
                            for r in self._cache_hparams
                            if r[0] == experiment_id
                        }
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM hparams WHERE experiment_id=?",
                (experiment_id,),
            ).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    # ── Bulk operations ──────────────────────────────────────────

    def delete_experiments_by_project(self, project: str) -> int:
        """Delete all experiments and their data for a project. Returns count."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM experiments WHERE project=?", (project,)
                ).fetchall()
            ]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            for table in (
                "scalars",
                "texts",
                "images",
                "audio",
                "video",
                "artifacts",
                "histograms",
                "hparams",
            ):
                conn.execute(
                    f"DELETE FROM {table} WHERE experiment_id IN ({placeholders})",
                    ids,
                )
            conn.execute(
                f"DELETE FROM experiments WHERE id IN ({placeholders})",
                ids,
            )
            return len(ids)

    # ── Cleanup ─────────────────────────────────────────────────

    def close(self) -> None:
        # Cancel timer and materialize if precache is still active
        if self._precache_timer is not None:
            self._precache_timer.cancel()
            self._precache_timer = None
        if self._precache_active:
            self._materialize()

        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
