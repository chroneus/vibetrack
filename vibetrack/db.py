"""SQLite backend with WAL mode for fast concurrent reads and writes."""

from __future__ import annotations

import json
import os
import logging
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

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

-- Run lifecycle (status, timing, provenance). No FOREIGN KEY on purpose: an
-- older vibetrack that does not know this table must still be able to delete
-- experiments, and AUTOINCREMENT ids are never reused, so orphans are inert.
CREATE TABLE IF NOT EXISTS run_info (
    experiment_id INTEGER PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'running',
    started_at    REAL,
    ended_at      REAL,
    heartbeat     REAL,
    info          TEXT   -- JSON: pid, host, argv, cwd, git, error, ...
);
"""


#: Per-experiment tables a synced run carries, with their data columns
#: (everything except ``experiment_id``). ``run_info`` travels separately.
RUN_TABLES: Dict[str, Tuple[str, ...]] = {
    "scalars": ("tag", "step", "value", "wall_time"),
    "texts": ("tag", "step", "value", "wall_time"),
    "images": ("tag", "step", "path", "wall_time"),
    "histograms": ("tag", "step", "bins", "counts", "wall_time"),
    "hparams": ("key", "value"),
    "audio": ("tag", "step", "path", "sample_rate", "wall_time"),
    "video": ("tag", "step", "path", "wall_time"),
    "artifacts": ("tag", "step", "path", "metadata", "wall_time"),
}


def _relative_media_path(path: Any, log_dir: str) -> Any:
    """*path* relative to *log_dir* when it is an absolute path inside it."""
    if not isinstance(path, str) or not log_dir or not os.path.isabs(path):
        return path
    prefix = log_dir.rstrip("/\\") + os.sep
    return path[len(prefix):] if path.startswith(prefix) else path


def _create_plain_db(path: Path) -> None:
    """Create a schema-only database in rollback-journal mode.

    Snapshots are shipped as one file, so they must not keep rows in a WAL
    side file.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        conn.commit()
    finally:
        conn.close()


def central_db_dir() -> Path:
    """Return ``~/.vibetrack/``."""
    return Path.home() / ".vibetrack"


def central_db_path() -> Path:
    """Return ``~/.vibetrack/vibetrack.db``."""
    return central_db_dir() / "vibetrack.db"


def open_central_db() -> "Database":
    """Open (or create) the system-wide central database."""
    return Database(central_db_path())


def log_dir_in_use(db: Any, log_dir: str) -> bool:
    """True if any experiment still in *db* (any project) uses *log_dir*."""
    target = os.path.realpath(log_dir)
    return any(
        row["log_dir"] and os.path.realpath(row["log_dir"]) == target
        for row in db.list_experiments()
    )


def remove_experiment_files(log_dir: str, db: Any = None) -> None:
    """Remove a deleted experiment's media, and its run directory if now empty.

    Call after the experiment's rows are deleted. Skipped while another
    experiment in *db* still uses the same directory: older remote-ingest
    servers gave same-named runs in different projects one shared
    ``log_dir``, so its media belongs to both.
    """
    if not log_dir:
        return
    if db is not None and log_dir_in_use(db, log_dir):
        _log.info("keeping %s: still used by another experiment", log_dir)
        return
    shutil.rmtree(os.path.join(log_dir, "media"), ignore_errors=True)
    try:
        os.rmdir(log_dir)
    except OSError:
        pass


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
        # ``RLock`` so ``_materialize`` (which acquires the lock) can be
        # called re-entrantly from a code path that already holds it
        # (e.g. ``_check_and_maybe_materialize``). Avoids the previous
        # manual release/re-acquire pattern, which leaked the lock if
        # the re-acquire ever failed.
        self._lock = threading.RLock()
        #: Set by readers of a synced (s3:// / file://) project: called with
        #: an absolute media path, fetches the file if needed and returns
        #: whether it now exists locally. ``None`` for ordinary databases.
        self.fetch_media: Optional[Callable[[str], bool]] = None

        # ── Precache state ──────────────────────────────────────
        self._precache_secs = precache_secs
        self._precache_active = precache_secs > 0
        self._precache_deadline = (
            time.time() + precache_secs if precache_secs > 0 else 0.0
        )
        self._precache_timer: Optional[threading.Timer] = None
        self._materializing = False
        self._remap_callbacks: List[Callable[[Dict[int, int]], None]] = []
        # Set the moment materialization finishes (successfully or not).
        # Tests can wait on this rather than polling with ``time.sleep``.
        self._materialized_event = threading.Event()
        if not self._precache_active:
            # Non-precache databases are "materialized" immediately.
            self._materialized_event.set()

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
        # run_info: exp_id -> merged record (see update_run_info)
        self._cache_run_info: Dict[int, Dict[str, Any]] = {}
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
        """Flush in-memory caches to SQLite and switch to direct mode.

        If any replay step raises, the precache flag is restored so the
        in-memory caches remain authoritative and the caller can retry,
        rather than silently losing data. Remap callbacks fire only on
        full success so writers don't switch over to a half-populated DB.
        """
        with self._lock:
            if not self._precache_active:
                return  # already materialized

            # 1. Create directory + DB + schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._init_schema(conn)

            try:
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
                # 12. Replay run lifecycle records
                for eid, rec in self._cache_run_info.items():
                    with self._connect() as conn:
                        self._write_run_info(conn, id_remap.get(eid, eid), rec)
            except Exception:
                # Restore precache flag so caches remain authoritative;
                # the caller can retry materialization on the next write.
                # NB: partial writes that did commit will remain in the DB
                # (SQLite auto-commits per add_* call); a clean rollback
                # would require unifying these calls under one explicit
                # transaction, which is out of scope here.
                self._precache_active = True
                _log.exception(
                    "vibetrack: precache materialization failed at %s; "
                    "keeping in-memory caches for retry",
                    self.path,
                )
                raise

            # 13. Fire callbacks (only after successful materialize)
            for cb in self._remap_callbacks:
                try:
                    cb(id_remap)
                except Exception:
                    _log.exception("vibetrack: precache remap callback %r failed", cb)

            # 14. Clear caches
            self._cache_run_info.clear()
            self._cache_experiments.clear()
            self._cache_scalars.clear()
            self._cache_texts.clear()
            self._cache_images.clear()
            self._cache_histograms.clear()
            self._cache_hparams.clear()
            self._cache_audio.clear()
            self._cache_video.clear()
            self._cache_artifacts.clear()

            # 15. Signal any waiters that materialization is complete.
            self._materialized_event.set()

    def wait_for_materialize(self, timeout: Optional[float] = None) -> bool:
        """Block until precache materialization completes.

        Returns ``True`` if the database is materialized (or was never in
        precache mode), ``False`` if the timeout expired first. Intended
        for tests so they don't have to poll with ``time.sleep``.
        """
        return self._materialized_event.wait(timeout)

    def _check_and_maybe_materialize(self) -> bool:
        """If precache deadline has passed, materialize. Must hold self._lock.

        Returns True if we just materialized (caller should use SQLite path).

        ``_lock`` is an ``RLock``, so we can hold it across the call to
        ``_materialize`` (which re-acquires it internally) without
        manually releasing/re-acquiring.
        """
        if self._materializing:
            return not self._precache_active
        if time.time() >= self._precache_deadline:
            self._materializing = True
            try:
                self._materialize()
            finally:
                self._materializing = False
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
                            p
                            for p, _ in sorted(
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
                "run_info",
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
                "run_info",
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

    def purge_experiment_from_step(self, experiment_id: int, step: int) -> None:
        """Delete step-indexed data for an experiment at ``step`` and beyond."""
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if self._check_and_maybe_materialize():
                        pass
                    else:
                        self._cache_scalars = [
                            r
                            for r in self._cache_scalars
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_texts = [
                            r
                            for r in self._cache_texts
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_images = [
                            r
                            for r in self._cache_images
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_audio = [
                            r
                            for r in self._cache_audio
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_video = [
                            r
                            for r in self._cache_video
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_artifacts = [
                            r
                            for r in self._cache_artifacts
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        self._cache_histograms = [
                            r
                            for r in self._cache_histograms
                            if not (r[0] == experiment_id and r[2] >= step)
                        ]
                        return
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in (
                "scalars",
                "texts",
                "images",
                "audio",
                "video",
                "artifacts",
                "histograms",
            ):
                conn.execute(
                    f"DELETE FROM {table} WHERE experiment_id=? AND step>=?",
                    (experiment_id, step),
                )

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

    def move_experiment_to_project(
        self,
        experiment_id: int,
        new_project: str,
        new_name: Optional[str] = None,
    ) -> bool:
        """Move an experiment to a different project, optionally renaming it.

        Returns True on success, False if (project, name) already taken.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            exp = conn.execute(
                "SELECT name FROM experiments WHERE id=?",
                (experiment_id,),
            ).fetchone()
            if exp is None:
                return False
            target_name = new_name if new_name else exp["name"]
            existing = conn.execute(
                "SELECT id FROM experiments WHERE project=? AND name=? LIMIT 1",
                (new_project, target_name),
            ).fetchone()
            if existing is not None and existing["id"] != experiment_id:
                return False
            conn.execute(
                "UPDATE experiments SET project=?, name=? WHERE id=?",
                (new_project, target_name, experiment_id),
            )
            return True

    def rename_project(
        self, old_project: str, new_project: str
    ) -> Tuple[bool, List[str]]:
        """Rename a project by updating all its experiments' project field.

        Returns ``(success, conflict_names)``.  When names in *new_project*
        would collide, returns ``(False, [conflicting_names])``.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            experiments = conn.execute(
                "SELECT id, name FROM experiments WHERE project=?",
                (old_project,),
            ).fetchall()
            if not experiments:
                return False, []
            conflicts: List[str] = []
            for exp in experiments:
                dup = conn.execute(
                    "SELECT id FROM experiments " "WHERE project=? AND name=? LIMIT 1",
                    (new_project, exp["name"]),
                ).fetchone()
                if dup is not None:
                    conflicts.append(exp["name"])
            if conflicts:
                return False, conflicts
            conn.execute(
                "UPDATE experiments SET project=? WHERE project=?",
                (new_project, old_project),
            )
            return True, []

    def update_log_dir(self, experiment_id: int, new_log_dir: str) -> bool:
        """Update the log_dir of an experiment. Returns True if successful."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE experiments SET log_dir=? WHERE id=?",
                (new_log_dir, experiment_id),
            )
            return cur.rowcount > 0

    # ── Single-run export / import (storage sync) ───────────────
    #
    # A synced run travels as a one-experiment snapshot DB plus deltas of
    # rows added after it. Copies use ATTACH + INSERT … SELECT, so they stay
    # fast for runs with millions of scalars.

    def run_marks(self, experiment_id: int) -> Dict[str, Tuple[int, int]]:
        """``{table: (max rowid, row count)}`` for one experiment's rows."""
        out: Dict[str, Tuple[int, int]] = {}
        with self._connect() as conn:
            for table in RUN_TABLES:
                row = conn.execute(
                    f"SELECT COALESCE(MAX(rowid), 0), COUNT(*) FROM {table} "
                    "WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                out[table] = (int(row[0]), int(row[1]))
        return out

    def run_delta(
        self,
        experiment_id: int,
        since: Dict[str, int],
        log_dir: str = "",
    ) -> Tuple[Dict[str, List[list]], Dict[str, int]]:
        """Rows added after *since* (``{table: rowid}``) and the new marks.

        Media paths under *log_dir* come back relative to it.
        """
        rows_out: Dict[str, List[list]] = {}
        marks: Dict[str, int] = {}
        with self._connect() as conn:
            for table, cols in RUN_TABLES.items():
                last = int(since.get(table, 0))
                rows = conn.execute(
                    f"SELECT rowid, {', '.join(cols)} FROM {table} "
                    "WHERE experiment_id=? AND rowid>? ORDER BY rowid",
                    (experiment_id, last),
                ).fetchall()
                marks[table] = int(rows[-1][0]) if rows else last
                if not rows:
                    continue
                path_idx = cols.index("path") if "path" in cols else -1
                out_rows = []
                for r in rows:
                    vals = list(r[1:])
                    if path_idx >= 0:
                        vals[path_idx] = _relative_media_path(vals[path_idx], log_dir)
                    out_rows.append(vals)
                rows_out[table] = out_rows
        return rows_out, marks

    def export_experiment(
        self,
        experiment_id: int,
        dest: str | Path,
        meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """Write one experiment into a new database file at *dest*.

        The copy gets experiment id 1, media paths relative to its log_dir,
        and *meta* in its ``meta`` table. Returns the ``{table: rowid}``
        marks the copy is consistent with (for later :meth:`run_delta`).
        """
        dest = Path(dest)
        if dest.exists():
            dest.unlink()
        _create_plain_db(dest)
        conn = self._get_conn()
        conn.commit()
        conn.execute("ATTACH DATABASE ? AS snap", (str(dest),))
        try:
            marks: Dict[str, int] = {}
            conn.execute("BEGIN")
            exp = conn.execute(
                "SELECT log_dir FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if exp is None:
                raise KeyError(f"no experiment {experiment_id}")
            log_dir = str(exp[0] or "")
            conn.execute(
                "INSERT INTO snap.experiments(id, name, project, log_dir, config, created_at) "
                "SELECT 1, name, project, log_dir, config, created_at "
                "FROM main.experiments WHERE id=?",
                (experiment_id,),
            )
            for table, cols in RUN_TABLES.items():
                row = conn.execute(
                    f"SELECT COALESCE(MAX(rowid), 0) FROM main.{table} WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                marks[table] = int(row[0])
                col_sql = ", ".join(cols)
                conn.execute(
                    f"INSERT INTO snap.{table}(experiment_id, {col_sql}) "
                    f"SELECT 1, {col_sql} FROM main.{table} "
                    "WHERE experiment_id=? AND rowid<=? ORDER BY rowid",
                    (experiment_id, marks[table]),
                )
                if "path" in cols and log_dir:
                    prefix = log_dir.rstrip("/\\") + os.sep
                    conn.execute(
                        f"UPDATE snap.{table} SET path=substr(path, ?) "
                        "WHERE substr(path, 1, ?)=?",
                        (len(prefix) + 1, len(prefix), prefix),
                    )
            conn.execute(
                "INSERT INTO snap.run_info"
                "(experiment_id, status, started_at, ended_at, heartbeat, info) "
                "SELECT 1, status, started_at, ended_at, heartbeat, info "
                "FROM main.run_info WHERE experiment_id=?",
                (experiment_id,),
            )
            for key, value in (meta or {}).items():
                conn.execute(
                    "INSERT OR REPLACE INTO snap.meta(key, value) VALUES (?, ?)",
                    (key, str(value)),
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.execute("DETACH DATABASE snap")
        return marks

    @contextmanager
    def transaction(
        self, attach: Optional[Dict[str, str | Path]] = None
    ) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` write transaction on this thread's connection.

        Commits when the block ends, rolls back if it raises. *attach* maps
        schema names to database files attached for the duration (SQLite
        cannot attach inside a transaction). Holding the write lock from the
        first read makes read-check-write sequences safe across processes.
        """
        if self._precache_active:
            self._materialize()
        conn = self._get_conn()
        conn.commit()
        attached: List[str] = []
        try:
            for alias, path in (attach or {}).items():
                if not alias.isidentifier():
                    raise ValueError(f"invalid schema name {alias!r}")
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))
                attached.append(alias)
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            conn.commit()
        finally:
            for alias in attached:
                conn.execute(f"DETACH DATABASE {alias}")

    def import_snapshot(
        self,
        src: str | Path,
        *,
        replace_id: Optional[int] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        log_dir: str = "",
    ) -> int:
        """Import the single experiment in snapshot *src*; return its new id.

        *replace_id* is deleted in the same transaction, so readers never see
        the run missing or half-imported. A ``(project, name)`` already used
        by another experiment gets the next ``" (N)"`` suffix.
        """
        with self.transaction(attach={"src": src}) as conn:
            return self.import_attached_snapshot(
                conn, replace_id=replace_id, name=name, project=project, log_dir=log_dir
            )

    def import_attached_snapshot(
        self,
        conn: sqlite3.Connection,
        *,
        replace_id: Optional[int] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        log_dir: str = "",
    ) -> int:
        """:meth:`import_snapshot` inside ``transaction(attach={"src": ...})``."""
        exp = conn.execute(
            "SELECT id, name, project, config, created_at FROM src.experiments "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        if exp is None:
            raise ValueError("snapshot holds no experiment")
        src_id = int(exp["id"])
        if replace_id is not None:
            self.delete_rows_in(conn, replace_id)
        target_project = exp["project"] if project is None else project
        target_name = self._free_name(
            conn, exp["name"] if name is None else name, target_project
        )
        cur = conn.execute(
            "INSERT INTO main.experiments(name, project, log_dir, config, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (target_name, target_project, log_dir, exp["config"], exp["created_at"]),
        )
        new_id = int(cur.lastrowid)  # type: ignore[arg-type]
        for table, cols in RUN_TABLES.items():
            col_sql = ", ".join(cols)
            conn.execute(
                f"INSERT INTO main.{table}(experiment_id, {col_sql}) "
                f"SELECT ?, {col_sql} FROM src.{table} WHERE experiment_id=? "
                "ORDER BY rowid",
                (new_id, src_id),
            )
        conn.execute(
            "INSERT OR REPLACE INTO main.run_info"
            "(experiment_id, status, started_at, ended_at, heartbeat, info) "
            "SELECT ?, status, started_at, ended_at, heartbeat, info "
            "FROM src.run_info WHERE experiment_id=?",
            (new_id, src_id),
        )
        return new_id

    def apply_delta(
        self,
        experiment_id: int,
        rows: Dict[str, List[list]],
        run_info: Optional[Dict[str, Any]] = None,
        config: Any = None,
    ) -> None:
        """Append rows produced by :meth:`run_delta`, in one transaction."""
        with self.transaction() as conn:
            self.apply_delta_in(conn, experiment_id, rows, run_info=run_info, config=config)

    def apply_delta_in(
        self,
        conn: sqlite3.Connection,
        experiment_id: int,
        rows: Dict[str, List[list]],
        run_info: Optional[Dict[str, Any]] = None,
        config: Any = None,
    ) -> None:
        """:meth:`apply_delta` inside a caller's :meth:`transaction`."""
        for table, table_rows in rows.items():
            cols = RUN_TABLES.get(table)
            if cols is None or not table_rows:
                continue
            verb = "INSERT OR REPLACE" if table == "hparams" else "INSERT"
            marks = ", ".join("?" for _ in range(len(cols) + 1))
            conn.executemany(
                f"{verb} INTO {table}(experiment_id, {', '.join(cols)}) VALUES ({marks})",
                [(experiment_id, *r[: len(cols)]) for r in table_rows if len(r) >= len(cols)],
            )
        if run_info is not None:
            self._write_run_info(conn, experiment_id, run_info)
        if config is not None:
            conn.execute(
                "UPDATE experiments SET config=? WHERE id=?",
                (json.dumps(config) if not isinstance(config, str) else config, experiment_id),
            )

    def set_name_and_project(self, experiment_id: int, name: str, project: str) -> str:
        """Set an experiment's name/project, suffixing on collision; return the name."""
        with self.transaction() as conn:
            return self.set_name_and_project_in(conn, experiment_id, name, project)

    def set_name_and_project_in(
        self, conn: sqlite3.Connection, experiment_id: int, name: str, project: str
    ) -> str:
        """:meth:`set_name_and_project` inside a caller's :meth:`transaction`."""
        row = conn.execute(
            "SELECT name, project FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if row is None:
            return name
        if row["name"] == name and row["project"] == project:
            return name
        final = self._free_name(conn, name, project, exclude_id=experiment_id)
        conn.execute(
            "UPDATE experiments SET name=?, project=? WHERE id=?",
            (final, project, experiment_id),
        )
        return final

    @staticmethod
    def delete_rows_in(conn: sqlite3.Connection, experiment_id: int) -> None:
        """Delete one experiment and its rows inside a caller's :meth:`transaction`."""
        for table in list(RUN_TABLES) + ["run_info"]:
            conn.execute(f"DELETE FROM main.{table} WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM main.experiments WHERE id=?", (experiment_id,))

    @staticmethod
    def _free_name(
        conn: sqlite3.Connection, name: str, project: str, exclude_id: Optional[int] = None
    ) -> str:
        taken = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM main.experiments WHERE project=? AND id IS NOT ? "
                "AND (name=? OR name LIKE ?)",
                (project, exclude_id, name, f"{name} (%"),
            )
        }
        if name not in taken:
            return name
        n = 2
        while f"{name} ({n})" in taken:
            n += 1
        return f"{name} ({n})"

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

    # ── Run lifecycle ────────────────────────────────────────────

    _RUN_FIELDS = ("status", "started_at", "ended_at", "heartbeat")

    @staticmethod
    def _run_row_to_dict(row: Any) -> Dict[str, Any]:
        try:
            info = json.loads(row["info"]) if row["info"] else {}
        except (TypeError, ValueError):
            info = {}
        return {
            "status": row["status"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "heartbeat": row["heartbeat"],
            "info": info if isinstance(info, dict) else {},
        }

    @staticmethod
    def _write_run_info(
        conn: sqlite3.Connection, experiment_id: int, rec: Dict[str, Any]
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO run_info"
            "(experiment_id, status, started_at, ended_at, heartbeat, info) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                rec.get("status") or "running",
                rec.get("started_at"),
                rec.get("ended_at"),
                rec.get("heartbeat"),
                json.dumps(rec.get("info") or {}, default=str),
            ),
        )

    def update_run_info(
        self,
        experiment_id: int,
        info: Optional[Dict[str, Any]] = None,
        **fields: Any,
    ) -> None:
        """Merge lifecycle fields into an experiment's ``run_info`` record.

        *fields* may set ``status``, ``started_at``, ``ended_at`` and
        ``heartbeat``. ``started_at`` keeps its first value so a resumed run
        reports when it originally started. *info* is shallow-merged into the
        stored JSON (``None`` values delete keys).
        """
        unknown = set(fields) - set(self._RUN_FIELDS)
        if unknown:
            raise ValueError(f"unknown run_info fields: {sorted(unknown)}")

        def merge(rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            out: Dict[str, Any] = dict(rec) if rec else {"status": "running", "info": {}}
            for key, value in fields.items():
                if key == "started_at" and out.get("started_at") is not None:
                    continue
                out[key] = value
            if info:
                merged = dict(out.get("info") or {})
                for key, value in info.items():
                    if value is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = value
                out["info"] = merged
            return out

        if self._precache_active:
            with self._lock:
                if self._precache_active and not self._check_and_maybe_materialize():
                    self._cache_run_info[experiment_id] = merge(
                        self._cache_run_info.get(experiment_id)
                    )
                    return
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM run_info WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            current = self._run_row_to_dict(row) if row is not None else None
            self._write_run_info(conn, experiment_id, merge(current))

    def get_run_info(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Return the lifecycle record for one experiment, or ``None``."""
        if self._precache_active:
            with self._lock:
                if self._precache_active and not self._check_and_maybe_materialize():
                    rec = self._cache_run_info.get(experiment_id)
                    return dict(rec) if rec is not None else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM run_info WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        return self._run_row_to_dict(row) if row is not None else None

    def get_all_run_info(self) -> Dict[int, Dict[str, Any]]:
        """Return every lifecycle record keyed by experiment id."""
        if self._precache_active:
            with self._lock:
                if self._precache_active and not self._check_and_maybe_materialize():
                    return {k: dict(v) for k, v in self._cache_run_info.items()}
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM run_info").fetchall()
        return {int(r["experiment_id"]): self._run_row_to_dict(r) for r in rows}

    def get_last_activity(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Return ``{"step", "wall_time"}`` of the newest scalar row, or ``None``."""
        if self._precache_active:
            with self._lock:
                if self._precache_active and not self._check_and_maybe_materialize():
                    rows = [r for r in self._cache_scalars if r[0] == experiment_id]
                    if not rows:
                        return None
                    steps = [
                        r[2] for r in rows if not r[1].startswith(("system/", "gpu/"))
                    ]
                    return {
                        "step": max(steps) if steps else None,
                        "wall_time": max(r[4] for r in rows),
                    }
        with self._connect() as conn:
            row = conn.execute(
                # The step ignores system/GPU samples, which count their own.
                "SELECT MAX(CASE WHEN tag NOT LIKE 'system/%' "
                "AND tag NOT LIKE 'gpu/%' THEN step END) AS step, "
                "MAX(wall_time) AS wall_time "
                "FROM scalars WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        if row is None or row["wall_time"] is None:
            return None
        return {"step": row["step"], "wall_time": row["wall_time"]}

    def get_scalar_stats(
        self,
        experiment_id: int,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-tag scalar aggregates without loading whole series.

        Returns ``{tag: {n, first_step, last_step, last, min, min_step, max,
        max_step}}``, optionally limited to *tags*. Used by run listings and
        comparisons, where only the last and extreme values matter.
        """
        wanted = set(tags) if tags is not None else None
        if wanted is not None and not wanted:
            return {}
        if self._precache_active:
            with self._lock:
                if self._precache_active and not self._check_and_maybe_materialize():
                    stats: Dict[str, Dict[str, Any]] = {}
                    for r in self._cache_scalars:
                        if r[0] != experiment_id:
                            continue
                        if wanted is not None and r[1] not in wanted:
                            continue
                        tag, step, value = r[1], r[2], r[3]
                        st = stats.get(tag)
                        if st is None:
                            stats[tag] = {
                                "n": 1,
                                "first_step": step,
                                "last_step": step,
                                "last": value,
                                "min": value,
                                "min_step": step,
                                "max": value,
                                "max_step": step,
                            }
                            continue
                        st["n"] += 1
                        if step < st["first_step"]:
                            st["first_step"] = step
                        if step >= st["last_step"]:
                            st["last_step"], st["last"] = step, value
                        if value < st["min"]:
                            st["min"], st["min_step"] = value, step
                        if value > st["max"]:
                            st["max"], st["max_step"] = value, step
                    return stats
        out: Dict[str, Dict[str, Any]] = {}
        tag_sql = ""
        tag_params: Tuple[Any, ...] = ()
        if wanted is not None:
            tag_sql = f" AND tag IN ({','.join('?' for _ in wanted)})"
            tag_params = tuple(sorted(wanted))
        with self._connect() as conn:
            for r in conn.execute(
                "SELECT tag, COUNT(*) AS n, MIN(step) AS first_step, "
                "MAX(step) AS last_step FROM scalars WHERE experiment_id=?"
                f"{tag_sql} GROUP BY tag",
                (experiment_id, *tag_params),
            ):
                out[r["tag"]] = {
                    "n": r["n"],
                    "first_step": r["first_step"],
                    "last_step": r["last_step"],
                }
            # SQLite returns the other ("bare") columns from the row that
            # holds the MIN / MAX, which gives the step of each extreme.
            for r in conn.execute(
                "SELECT tag, MIN(value) AS v, step FROM scalars "
                f"WHERE experiment_id=?{tag_sql} GROUP BY tag",
                (experiment_id, *tag_params),
            ):
                if r["tag"] in out:
                    out[r["tag"]].update(min=r["v"], min_step=r["step"])
            for r in conn.execute(
                "SELECT tag, MAX(value) AS v, step FROM scalars "
                f"WHERE experiment_id=?{tag_sql} GROUP BY tag",
                (experiment_id, *tag_params),
            ):
                if r["tag"] in out:
                    out[r["tag"]].update(max=r["v"], max_step=r["step"])
            for r in conn.execute(
                "SELECT s.tag, s.value FROM scalars s JOIN ("
                "  SELECT tag, MAX(step) AS ms FROM scalars"
                f"  WHERE experiment_id=?{tag_sql} GROUP BY tag"
                ") m ON s.tag = m.tag AND s.step = m.ms "
                "WHERE s.experiment_id=? ORDER BY s.rowid",
                (experiment_id, *tag_params, experiment_id),
            ):
                if r["tag"] in out:
                    out[r["tag"]]["last"] = r["value"]
        return out

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
                "run_info",
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

    # ── Bulk get-all-by-experiment helpers ──────────────────────
    #
    # Each ``get_all_*`` returns ``Dict[tag, List[row]]`` in a *single*
    # SQL query, replacing the N+1 pattern of ``for tag in get_*_tags():
    # rows = get_*(tag)`` used by the web/console viewers. Precache mode
    # is supported by grouping the in-memory cache instead of the DB.

    def _group_cache_by_tag(
        self,
        cache: List[Tuple[Any, ...]],
        experiment_id: int,
        row_builder: Callable[[Tuple[Any, ...]], Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in cache:
            if r[0] != experiment_id:
                continue
            out.setdefault(r[1], []).append(row_builder(r))
        for tag in out:
            out[tag].sort(key=lambda d: d["step"])
        return out

    def _group_rows_by_tag(self, rows: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            d = dict(r)
            tag = d.pop("tag")
            out.setdefault(tag, []).append(d)
        return out

    def get_all_scalars(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_scalars,
                            experiment_id,
                            lambda r: {"step": r[2], "value": r[3], "wall_time": r[4]},
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, value, wall_time FROM scalars "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_texts(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_texts,
                            experiment_id,
                            lambda r: {"step": r[2], "value": r[3], "wall_time": r[4]},
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, value, wall_time FROM texts "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_images(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_images,
                            experiment_id,
                            lambda r: {"step": r[2], "path": r[3], "wall_time": r[4]},
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, path, wall_time FROM images "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_audio(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_audio,
                            experiment_id,
                            lambda r: {
                                "step": r[2],
                                "path": r[3],
                                "sample_rate": r[4],
                                "wall_time": r[5],
                            },
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, path, sample_rate, wall_time FROM audio "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_video(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_video,
                            experiment_id,
                            lambda r: {"step": r[2], "path": r[3], "wall_time": r[4]},
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, path, wall_time FROM video "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_artifacts(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_artifacts,
                            experiment_id,
                            lambda r: {
                                "step": r[2],
                                "path": r[3],
                                "metadata": r[4],
                                "wall_time": r[5],
                            },
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, path, metadata, wall_time FROM artifacts "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

    def get_all_histograms(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        if self._precache_active:
            with self._lock:
                if self._precache_active:
                    if not self._check_and_maybe_materialize():
                        return self._group_cache_by_tag(
                            self._cache_histograms,
                            experiment_id,
                            lambda r: {
                                "step": r[2],
                                "bins": r[3],
                                "counts": r[4],
                                "wall_time": r[5],
                            },
                        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag, step, bins, counts, wall_time FROM histograms "
                "WHERE experiment_id=? ORDER BY tag, step",
                (experiment_id,),
            ).fetchall()
            return self._group_rows_by_tag(rows)

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
