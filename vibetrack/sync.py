"""Sync runs to object storage and read them back from anywhere.

Bucket layout under the storage prefix (see ``docs/STORAGE.md``)::

    _runs/<uid>/snap.db.gz              one-run SQLite snapshot
    _runs/<uid>/d/<gen>-<seq>.json.gz   rows added after snapshot <gen>
    _runs/<uid>/edit.json               rename/move made in a dashboard
    _runs/<uid>/deleted                 delete marker
    <project>/<run>-<uid>/media/...    media files

Only the run's own writer (or ``vibetrack push``) writes ``snap`` and ``d/``;
only dashboards write ``edit.json`` and ``deleted``. No object is ever
read-modify-written by two machines, so any number of hosts can share one
bucket.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import mimetypes
import os
import re
import shlex
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import weakref
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .db import Database, remove_experiment_files
from .storage import ObjectInfo, Storage, StorageError, display_uri, is_remote_uri, open_storage

_log = logging.getLogger(__name__)

RUNS = "_runs"
SNAPSHOT = "snap.db.gz"
EDIT = "edit.json"
DELETED = "deleted"

DEFAULT_INTERVAL = 60.0
DEFAULT_FINAL_TIMEOUT = 120.0
#: Chunks uploaded after a snapshot before the writer compacts them into a new one.
COMPACT_EVERY = 50
#: Media at least this big (or any video) streams from a presigned URL in "auto" mode.
PRESIGN_MIN_BYTES = 20 * 1024 * 1024

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi", ".ogv", ".gif")
#: Also streamed directly in "presign" mode. Text/JSON never is: the UI
#: fetch()es those, which a cross-origin redirect without CORS would break.
_PLAIN_MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".wav", ".mp3", ".ogg", ".flac", ".m4a")
_CHUNK_RE = re.compile(r"^d/(\d+)-(\d+)\.json\.gz$")

__all__ = [
    "RemoteProject",
    "RunSync",
    "RunUploader",
    "is_remote_uri",
    "pull",
    "push_runs",
]


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def sync_interval() -> float:
    return _env_float("VIBETRACK_SYNC_INTERVAL", DEFAULT_INTERVAL)


def _safe_segment(value: str, default: str) -> str:
    seg = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("._")
    return (seg or default)[:80]


def run_uid(db_path: str, experiment_id: int, created_at: float, host: str) -> str:
    """Stable id of a run: same across resumes, unique across machines."""
    raw = f"{host}|{os.path.realpath(db_path)}|{experiment_id}|{created_at!r}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


_UID = re.compile(r"[0-9a-f]{16}")


def _check_uid(uid: str) -> None:
    if not _UID.fullmatch(uid):
        raise ValueError(f"invalid run id: {uid!r}")


def _media_prefix(uid: str, prefix: str) -> str:
    _check_uid(uid)
    # Older snapshots use the first eight UID digits. They remain readable,
    # but a short suffix cannot establish exclusive ownership for deletion.
    parts = prefix.split("/")
    if (len(parts) != 2 or parts[0] == RUNS
            or any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) or part in (".", "..")
                   for part in parts)
            or not parts[1].endswith(("-" + uid, "-" + uid[:8]))):
        raise ValueError(f"invalid media prefix for run {uid}")
    return prefix


def _run_key(uid: str, name: str) -> str:
    _check_uid(uid)
    return f"{RUNS}/{uid}/{name}"


def _gzip_file(src: str, dest: str) -> None:
    with open(src, "rb") as fin, gzip.open(dest, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, 1024 * 1024)


def _gunzip_bytes_to(data: bytes, dest: str) -> None:
    with open(dest, "wb") as fh:
        fh.write(gzip.decompress(data))


def _safe_key_parts(parts: Sequence[str]) -> bool:
    return all(p not in ("", ".", "..") and "\\" not in p and ":" not in p for p in parts)


def _purge_run(storage: Storage, uid: str, media_prefix: Optional[str]) -> None:
    """Delete a run's objects, keeping its delete marker (written first)."""
    root = _run_key(uid, "")
    keys = [o.key for o in storage.list(root) if o.key.startswith(root)
            and o.key != root + DELETED and _safe_key_parts(o.key[len(root):].split("/"))]
    if media_prefix and _media_prefix(uid, media_prefix).endswith("-" + uid):
        media = media_prefix + "/media/"
        keys += [o.key for o in storage.list(media) if o.key.startswith(media)
                 and _safe_key_parts(o.key[len(media):].split("/"))]
    storage.delete_many(keys)


class _Warner:
    """Print each distinct sync problem once, never raise."""

    def __init__(self, sink: Optional[Callable[[str], None]] = None) -> None:
        self._seen: set = set()
        self._sink = sink

    def __call__(self, msg: str) -> None:
        if msg in self._seen:
            return
        self._seen.add(msg)
        if self._sink is not None:
            self._sink(msg)
        else:
            print(f"vibetrack warning: {msg}", file=sys.stderr)


# ── Upload side ─────────────────────────────────────────────────────


class RunUploader:
    """Upload one experiment of a local database: snapshot, deltas, media."""

    def __init__(
        self,
        storage: Storage,
        db: Database,
        experiment_id: int,
        host: Optional[str] = None,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        from ._runmeta import this_host

        self.storage = storage
        self.db = db
        self.experiment_id = experiment_id
        self.host = host if host is not None else this_host()
        self.warn = warn or _Warner()
        self.uid: Optional[str] = None
        self.media_prefix: Optional[str] = None
        self.gen: Optional[int] = None
        self.seq = 0
        self.marks: Dict[str, int] = {}
        self.expected: Dict[str, int] = {}
        self.chunks: List[str] = []
        self.deleted = False
        self._last_head: Optional[Tuple[Any, ...]] = None
        self._media_sent: Dict[str, Tuple[int, int]] = {}
        self._lock = threading.Lock()

    # Identity is fixed on the first sync: the writer may rename or suffix
    # the run before that.
    def _ensure_identity(self, row: Any) -> None:
        if self.uid is not None:
            return
        self.uid = run_uid(
            str(self.db.path), int(row["id"]), float(row["created_at"] or 0), self.host
        )
        self.media_prefix = (
            f"{_safe_segment(row['project'], '_')}/"
            f"{_safe_segment(row['name'], 'run')}-{self.uid}"
        )

    def is_deleted(self) -> bool:
        if self.uid is None:
            return False
        try:
            return self.storage.exists(_run_key(self.uid, DELETED))
        except (StorageError, OSError):
            return False

    def sync(self, final: bool = False, deadline: Optional[float] = None) -> bool:
        """Upload what changed. Returns ``False`` if the run must stop syncing.

        With *deadline*, waiting for a pass still running in another thread
        stops at the deadline too.
        """
        wait = -1.0 if deadline is None else max(0.0, deadline - time.time())
        if not self._lock.acquire(timeout=wait):
            raise TimeoutError("an earlier upload was still running at the deadline")
        try:
            return self._sync_locked(final, deadline)
        finally:
            self._lock.release()

    def _sync_locked(self, final: bool, deadline: Optional[float]) -> bool:
        if self.deleted:
            return False
        row = self.db.get_experiment(self.experiment_id)
        if row is None:
            return True
        row = dict(row)
        self._ensure_identity(row)
        if self.is_deleted():
            self.deleted = True
            self.warn(
                f"run {row['name']!r} was deleted from {display_uri(self.storage.uri)}; "
                "no longer syncing it"
            )
            return False
        log_dir = str(row.get("log_dir") or "")
        counts = self.db.run_marks(self.experiment_id)
        if final:
            self._upload_rows(row, counts, log_dir, snapshot=True)
            self._upload_media(log_dir, deadline)
        else:
            # Media first, so rows never point at files not yet uploaded.
            self._upload_media(log_dir, deadline)
            self._upload_rows(row, counts, log_dir, snapshot=self._needs_snapshot(counts))
        return True

    def _needs_snapshot(self, counts: Dict[str, Tuple[int, int]]) -> bool:
        if self.gen is None or len(self.chunks) >= COMPACT_EVERY:
            return True
        # Rows were deleted (a purge on resume): deltas cannot express that.
        # hparams are replaced in place, which readers apply as an upsert.
        return any(
            counts[t][1] < self.expected.get(t, 0) for t in counts if t != "hparams"
        )

    def _snapshot_meta(self, gen: int) -> Dict[str, str]:
        return {
            "sync_uid": str(self.uid),
            "sync_gen": str(gen),
            "sync_media_prefix": str(self.media_prefix),
            "sync_host": self.host,
        }

    def _upload_rows(
        self,
        row: Dict[str, Any],
        counts: Dict[str, Tuple[int, int]],
        log_dir: str,
        snapshot: bool,
    ) -> None:
        assert self.uid is not None
        run_info = self.db.get_run_info(self.experiment_id)
        head = (row["name"], row["project"], json.dumps(run_info, sort_keys=True, default=str))
        if snapshot:
            old_chunks = list(self.chunks)
            gen = int(time.time() * 1000)
            if self.gen is not None and gen <= self.gen:
                gen = self.gen + 1
            with tempfile.TemporaryDirectory(prefix="vibetrack-sync-") as tmp:
                raw = os.path.join(tmp, "snap.db")
                packed = os.path.join(tmp, SNAPSHOT)
                marks = self.db.export_experiment(
                    self.experiment_id, raw, meta=self._snapshot_meta(gen)
                )
                _gzip_file(raw, packed)
                self.storage.put_file(
                    _run_key(self.uid, SNAPSHOT), packed, "application/gzip"
                )
            self.gen = gen
            self.marks = marks
            self.expected = {t: counts[t][1] for t in counts}
            self.seq = 0
            self.chunks = []
            self._last_head = head
            for key in old_chunks:
                try:
                    self.storage.delete(key)
                except (StorageError, OSError):
                    pass  # readers ignore chunks of an older snapshot
            return
        rows, marks = self.db.run_delta(self.experiment_id, self.marks, log_dir=log_dir)
        if not rows and head == self._last_head:
            return
        payload = {
            "gen": self.gen,
            "seq": self.seq + 1,
            "name": row["name"],
            "project": row["project"],
            "media_prefix": self.media_prefix,
            "run_info": run_info,
            "rows": rows,
        }
        key = _run_key(self.uid, f"d/{self.gen}-{self.seq + 1:06d}.json.gz")
        data = gzip.compress(json.dumps(payload, default=str).encode("utf-8"), 6)
        self.storage.put_bytes(key, data, "application/gzip")
        self.seq += 1
        self.chunks.append(key)
        self.marks = marks
        for table, table_rows in rows.items():
            self.expected[table] = self.expected.get(table, 0) + len(table_rows)
        self._last_head = head

    def _upload_media(self, log_dir: str, deadline: Optional[float]) -> None:
        if not log_dir:
            return
        media_root = os.path.join(log_dir, "media")
        if not os.path.isdir(media_root):
            return
        for dirpath, _dirs, files in os.walk(media_root):
            for name in sorted(files):
                if name.startswith(".") or name.endswith((".tmp", ".part")):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, log_dir).replace(os.sep, "/")
                sig = (st.st_size, st.st_mtime_ns)
                if self._media_sent.get(rel) == sig:
                    continue
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError("media upload did not finish before the deadline")
                ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                self.storage.put_file(f"{self.media_prefix}/{rel}", full, ctype)
                self._media_sent[rel] = sig

    def archive(self) -> bool:
        """Upload, then delete locally only if the source stayed unchanged."""
        # Observe from a separate connection so commits from every writer,
        # including this thread, advance the observed main-DB version.
        with closing(sqlite3.connect(str(self.db.path))) as observer:
            version = observer.execute("PRAGMA main.data_version").fetchone()[0]
            if not self.sync(final=True):
                return False
            if not self.verify_uploaded():
                raise StorageError("upload verification failed; local copy kept")
            with self.db.transaction() as conn:
                if observer.execute("PRAGMA main.data_version").fetchone()[0] != version:
                    raise StorageError("database changed during upload; local copy kept")
                row = conn.execute(
                    "SELECT e.log_dir, r.status FROM experiments e LEFT JOIN run_info r "
                    "ON r.experiment_id=e.id WHERE e.id=?", (self.experiment_id,),
                ).fetchone()
                if row is None or row["status"] == "running":
                    raise StorageError("run is missing or still running; local copy kept")
                self.db.delete_rows_in(conn, self.experiment_id)
            remove_experiment_files(row["log_dir"] or "", self.db)
        return True

    def verify_uploaded(self) -> bool:
        """True when the snapshot and every local media file are in storage."""
        if self.uid is None or self.gen is None:
            return False
        try:
            if not self.storage.exists(_run_key(self.uid, SNAPSHOT)):
                return False
            remote = {
                o.key: o.size for o in self.storage.list(f"{self.media_prefix}/")
            }
        except (StorageError, OSError):
            return False
        row = self.db.get_experiment(self.experiment_id)
        log_dir = str(row["log_dir"] or "") if row is not None else ""
        media_root = os.path.join(log_dir, "media")
        for dirpath, _dirs, files in os.walk(media_root):
            for name in files:
                if name.startswith(".") or name.endswith((".tmp", ".part")):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, log_dir).replace(os.sep, "/")
                st = os.stat(full)
                if (self._media_sent.get(rel) != (st.st_size, st.st_mtime_ns)
                        or remote.get(f"{self.media_prefix}/{rel}") != st.st_size):
                    return False
        return True


class RunSync:
    """Background uploader attached to a :class:`~vibetrack.SummaryWriter`.

    Every ``VIBETRACK_SYNC_INTERVAL`` seconds (default 60) it uploads the rows
    added since the last pass, plus new media. Close, failure and SIGTERM
    trigger a final snapshot. Errors are reported once and retried later;
    training is never blocked or interrupted.
    """

    def __init__(self, writer: Any, uri: str, interval: Optional[float] = None) -> None:
        self.uri = uri
        self.storage = open_storage(uri)
        self.interval = interval if interval is not None else sync_interval()
        self._writer = weakref.ref(writer)
        self._warn = _Warner()
        self._uploader: Optional[RunUploader] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failures = 0
        self._next_try = 0.0
        self.last_ok: Optional[float] = None
        self.stopped = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="vibetrack-sync", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if time.time() < self._next_try:
                continue
            if not self.tick():
                return

    def _current_uploader(self, writer: Any) -> Optional[RunUploader]:
        db = writer._db
        exp_id = writer._exp_id
        if db is None or db._precache_active or exp_id is None or exp_id <= 0:
            return None
        if self._uploader is None or self._uploader.db is not db:
            self._uploader = RunUploader(self.storage, db, exp_id, warn=self._warn)
        elif self._uploader.experiment_id != exp_id:
            # The writer switched to a suffixed run before its first step.
            self._uploader = RunUploader(self.storage, db, exp_id, warn=self._warn)
        return self._uploader

    def tick(self, final: bool = False, deadline: Optional[float] = None) -> bool:
        """One sync pass. Returns ``False`` once syncing stopped for good."""
        if self.stopped:
            return False
        writer = self._writer()
        if writer is None:
            return False
        if writer._pending:
            return True  # run identity not resolved yet (first step pending)
        try:
            with writer._buffer_lock:
                writer._flush_locked()
        except Exception:
            pass  # the writer reports its own flush errors
        uploader = self._current_uploader(writer)
        if uploader is None:
            return True
        try:
            if not uploader.sync(final=final, deadline=deadline):
                self.stopped = True
                return False
        except Exception as exc:
            self._failures += 1
            self._next_try = time.time() + min(600.0, self.interval * (2 ** min(self._failures, 4)))
            self._warn(
                f"could not sync to {display_uri(self.uri)} ({exc}); "
                "will retry — training is not affected"
            )
            return True
        self._failures = 0
        self._next_try = 0.0
        self.last_ok = time.time()
        return True

    def close(self, timeout: Optional[float] = None) -> bool:
        """Stop the thread and upload a final snapshot. Returns success.

        Everything, including an upload the background thread is still
        doing, ends within *timeout* seconds (``VIBETRACK_SYNC_TIMEOUT``,
        default 120).
        """
        if timeout is None:
            timeout = _env_float("VIBETRACK_SYNC_TIMEOUT", DEFAULT_FINAL_TIMEOUT)
        deadline = time.time() + timeout
        storage = self.storage
        # Bounds every request, retry and backoff, in this thread and the
        # background one alike.
        storage.deadline = deadline
        if hasattr(storage, "retries"):
            storage.retries = min(int(storage.retries), 2)  # type: ignore[attr-defined]
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, min(5.0, deadline - time.time())))
        if self.stopped:
            return False
        before = self.last_ok
        self._failures = 0
        self.tick(final=True, deadline=deadline)
        ok = self.last_ok is not None and self.last_ok != before
        if not ok and not self.stopped:
            self._warn(
                f"final sync to {display_uri(self.uri)} did not complete; "
                f"upload later with: {self.push_command()}"
            )
        return ok

    def discard(self, timeout: Optional[float] = None) -> None:
        """Stop syncing a run that is being discarded; remove what it uploaded.

        Writes the delete marker first, so readers drop the run even if an
        upload still in flight lands afterwards. Bounded like :meth:`close`.
        """
        if timeout is None:
            timeout = _env_float("VIBETRACK_SYNC_TIMEOUT", DEFAULT_FINAL_TIMEOUT)
        deadline = time.time() + timeout
        self.storage.deadline = deadline
        if hasattr(self.storage, "retries"):
            self.storage.retries = min(int(self.storage.retries), 2)  # type: ignore[attr-defined]
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, min(5.0, deadline - time.time())))
        self.stopped = True
        uploader = self._uploader
        if uploader is None:
            return
        got_lock = uploader._lock.acquire(timeout=max(0.0, deadline - time.time()))
        try:
            uploader.deleted = True  # a background pass still running stops here
            if uploader.uid is None:
                return  # nothing was uploaded
            self.storage.put_bytes(_run_key(uploader.uid, DELETED), b"", "text/plain")
            _purge_run(self.storage, uploader.uid, uploader.media_prefix)
        except (StorageError, OSError) as exc:
            self._warn(f"could not remove the discarded run from {display_uri(self.uri)}: {exc}")
        finally:
            if got_lock:
                uploader._lock.release()

    def push_command(self) -> str:
        """Shell command that uploads this run later with ``vibetrack push``."""
        writer = self._writer()
        args = ["vibetrack", "push"]
        folder = getattr(writer, "project_folder", None) if writer is not None else None
        if folder:
            args.append(str(Path(folder).expanduser().resolve()))
        args.append(self.uri)
        row = None
        uploader = self._uploader
        if uploader is not None:
            try:
                row = uploader.db.get_experiment(uploader.experiment_id)
            except Exception:
                row = None
        if row is not None:
            args.append(str(row["name"]))
            if row["project"]:
                args += ["-p", str(row["project"])]
        return " ".join(shlex.quote(a) for a in args)


# ── Read side: local mirror of a bucket ─────────────────────────────


def default_cache_root() -> Path:
    env = os.environ.get("VIBETRACK_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    from .db import central_db_dir

    return central_db_dir() / "cache"


#: Where each mirrored run stands. Lives in the mirror database and changes
#: in the same transaction as the run's rows.
_MIRROR_SQL = """
CREATE TABLE IF NOT EXISTS sync_mirror (
    source       TEXT NOT NULL,
    uid          TEXT NOT NULL,
    exp_id       INTEGER NOT NULL,
    snap         TEXT NOT NULL,
    gen          INTEGER NOT NULL DEFAULT 0,
    seq          INTEGER NOT NULL DEFAULT 0,
    edit         TEXT,
    media_prefix TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source, uid)
)
"""


class RemoteProject:
    """A local, read-mostly mirror of the runs stored at *uri*.

    The mirror is an ordinary project folder: ``vibetrack.db`` plus
    ``runs/<uid>/media/...`` (fetched lazily). Rows arrive incrementally:
    a snapshot is downloaded once, then only chunks not applied yet.

    Which snapshot and chunk each run is at is kept in the mirror database
    (table ``sync_mirror``) and updated in the same transaction as the rows.
    Any number of processes can therefore share one mirror (a dashboard, its
    MCP server, CLI commands), and an interrupted refresh never applies rows
    twice. A mirrored run's experiment has ``log_dir`` ``runs/<uid>``; a new
    snapshot replaces every experiment with that ``log_dir``.
    """

    def __init__(
        self,
        uri: str,
        cache_dir: Optional[str] = None,
        refresh_secs: Optional[float] = None,
    ) -> None:
        self.uri = uri
        self.storage = open_storage(uri)
        #: Several locations may be mirrored into one folder (``pull``).
        self.source = self.storage.cache_id()
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            root = default_cache_root().resolve()
            self.cache_dir = (root / self.source).resolve()
            if root not in self.cache_dir.parents:
                raise ValueError("storage cache path escapes cache root")
        self.db_path = self.cache_dir / "vibetrack.db"
        self.runs_dir = self.cache_dir / "runs"
        self.refresh_secs = (
            refresh_secs if refresh_secs is not None else _env_float("VIBETRACK_SYNC_REFRESH", 10.0)
        )
        self._lock = threading.RLock()
        self._db_guard = threading.Lock()
        self._fetch_locks: Dict[str, threading.Lock] = {}
        self._fetch_guard = threading.Lock()
        self._bg: Optional[threading.Thread] = None
        self._last_refresh = 0.0
        self._db: Optional[Database] = None
        self._warn = _Warner()
        self.last_error: Optional[str] = None
        self.run_errors: List[str] = []

    # ── mirror bookkeeping ──────────────────────────────────────

    def _database(self) -> Database:
        with self._db_guard:
            if self._db is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                db = Database(self.db_path)
                with db.transaction() as conn:
                    conn.execute(_MIRROR_SQL)
                    conn.execute("CREATE TABLE IF NOT EXISTS sync_media ("
                                 "source TEXT, uid TEXT, key TEXT, etag TEXT, "
                                 "PRIMARY KEY(source, uid, key))")
                self._db = db
            return self._db

    def _run_dir(self, uid: str) -> str:
        _check_uid(uid)
        path = self.runs_dir / uid
        if path.resolve() != path:
            raise ValueError("mirror run directory contains a symbolic link")
        return str(path)

    def _row(self, conn: Any, uid: str) -> Optional[Any]:
        """The mirror row of *uid*, if its experiment is still in the mirror."""
        return conn.execute(
            "SELECT m.exp_id, m.snap, m.gen, m.seq, m.edit, m.media_prefix "
            "FROM sync_mirror m JOIN experiments e ON e.id = m.exp_id AND e.log_dir = ? "
            "WHERE m.source = ? AND m.uid = ?",
            (self._run_dir(uid), self.source, uid),
        ).fetchone()

    def _conn(self) -> Any:
        return self._database()._get_conn()

    def mirrored(self) -> Dict[str, Dict[str, Any]]:
        """``{uid: mirror row}`` of the runs mirrored from this location."""
        rows = self._conn().execute(
            "SELECT m.uid, m.exp_id, m.snap, m.gen, m.seq, m.edit, m.media_prefix, "
            "e.log_dir FROM sync_mirror m JOIN experiments e ON e.id = m.exp_id "
            "WHERE m.source = ?",
            (self.source,),
        ).fetchall()
        return {
            r["uid"]: dict(r) for r in rows
            if _UID.fullmatch(r["uid"]) and r["log_dir"] == str(self.runs_dir / r["uid"])
        }

    def _remove_in(self, conn: Any, uid: str) -> None:
        """Delete every experiment mirroring *uid* and its bookkeeping."""
        ids = conn.execute(
            "SELECT id FROM experiments WHERE log_dir = ?", (self._run_dir(uid),)
        ).fetchall()
        for (exp_id,) in ids:
            Database.delete_rows_in(conn, int(exp_id))
        conn.execute("DELETE FROM sync_media WHERE source=? AND uid=?", (self.source, uid))
        conn.execute(
            "DELETE FROM sync_mirror WHERE source = ? AND uid = ?", (self.source, uid)
        )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # ── refresh ─────────────────────────────────────────────────

    def maybe_refresh(self) -> None:
        """Refresh now if never done, else in the background when stale."""
        if self._last_refresh == 0.0:
            self.refresh()
            return
        if time.time() - self._last_refresh < self.refresh_secs:
            return
        if self._bg is not None and self._bg.is_alive():
            return
        self._bg = threading.Thread(target=self.refresh, name="vibetrack-mirror", daemon=True)
        self._bg.start()

    def refresh(self) -> bool:
        """Bring the mirror up to date. Returns whether anything changed."""
        with self._lock:
            self.run_errors.clear()
            self._last_refresh = time.time()
            try:
                changed = self._refresh_locked()
            except Exception as exc:
                self.last_error = str(exc)
                self._warn(f"could not read {display_uri(self.uri)}: {exc}")
                return False
            self.last_error = None
            return changed

    def _refresh_locked(self) -> bool:
        listing: Dict[str, Dict[str, Any]] = {}
        for obj in self.storage.list(f"{RUNS}/"):
            parts = obj.key.split("/", 2)
            if len(parts) < 3:
                continue
            uid, rest = parts[1], parts[2]
            if not _UID.fullmatch(uid):
                self.run_errors.append(f"invalid run id {uid!r}")
                continue
            entry = listing.setdefault(uid, {"chunks": []})
            if rest == SNAPSHOT:
                entry["snap"] = obj
            elif rest == EDIT:
                entry["edit"] = obj
            elif rest == DELETED:
                entry["deleted"] = True
            else:
                m = _CHUNK_RE.match(rest)
                if m:
                    entry["chunks"].append((int(m.group(1)), int(m.group(2)), obj))
        db = self._database()
        changed = False
        known = [
            r[0]
            for r in self._conn().execute(
                "SELECT uid FROM sync_mirror WHERE source = ?", (self.source,)
            ).fetchall()
        ]
        for uid in known:
            if not _UID.fullmatch(uid):
                with db.transaction() as conn:
                    row = conn.execute("SELECT exp_id FROM sync_mirror WHERE source=? AND uid=?",
                                       (self.source, uid)).fetchone()
                    if row is not None:
                        exp = conn.execute("SELECT log_dir FROM experiments WHERE id=?", (row[0],)).fetchone()
                        if exp is not None and exp[0] == str(self.runs_dir / uid):
                            db.delete_rows_in(conn, int(row[0]))
                        conn.execute("DELETE FROM sync_mirror WHERE source=? AND uid=?", (self.source, uid))
                changed = True
                continue
            entry = listing.get(uid)
            if entry is None or entry.get("deleted") or "snap" not in entry:
                self._drop_local(uid)
                changed = True
        order = sorted(
            (u for u, e in listing.items() if "snap" in e and not e.get("deleted")),
            key=lambda u: listing[u]["snap"].mtime,
        )
        for uid in order:
            try:
                changed |= self._refresh_run(db, uid, listing[uid])
            except Exception as exc:
                # Isolate malformed or unreadable runs so a pull can still
                # finish the healthy ones and report the complete error list.
                self.run_errors.append(f"{uid}: {exc}")
                self._warn(f"skipping run {uid} from {display_uri(self.uri)}: {exc}")
        return changed

    def _read_edit(self, uid: str) -> Optional[Dict[str, str]]:
        data = self.storage.get_bytes(_run_key(uid, EDIT))
        if not data:
            return None
        try:
            edit = json.loads(data.decode("utf-8"))
        except ValueError:
            return None
        return edit if isinstance(edit, dict) else None

    def _refresh_run(self, db: Database, uid: str, entry: Dict[str, Any]) -> bool:
        snap = entry["snap"]
        edit_obj = entry.get("edit")
        edit_tag = edit_obj.etag if edit_obj is not None else None
        changed = False
        row = self._row(self._conn(), uid)
        if row is None or row["snap"] != snap.etag:
            changed = self._import_run(db, uid, snap, edit_obj)
        elif row["edit"] != edit_tag:
            edit = self._read_edit(uid) if edit_obj is not None else None
            with db.transaction() as conn:
                cur = self._row(conn, uid)
                if cur is not None and cur["edit"] != edit_tag:
                    if edit:
                        db.set_name_and_project_in(
                            conn, int(cur["exp_id"]),
                            str(edit.get("name", "")), str(edit.get("project", "")),
                        )
                    conn.execute(
                        "UPDATE sync_mirror SET edit = ? WHERE source = ? AND uid = ?",
                        (edit_tag, self.source, uid),
                    )
                    changed = True
        row = self._row(self._conn(), uid)
        if row is not None:
            changed |= self._apply_chunks(db, uid, row, entry["chunks"])
        return changed

    def _import_run(self, db: Database, uid: str, snap: Any, edit_obj: Any) -> bool:
        """Download and import *uid*'s snapshot, replacing its previous import."""
        edit = self._read_edit(uid) if edit_obj is not None else None
        data = self.storage.get_bytes(_run_key(uid, SNAPSHOT))
        if data is None:
            raise StorageError(f"snapshot disappeared for {uid}")
        fd, tmp = tempfile.mkstemp(prefix="vt-snap-", suffix=".db", dir=str(self.cache_dir))
        os.close(fd)
        try:
            _gunzip_bytes_to(data, tmp)
            meta = _read_meta(tmp)
            with db.transaction(attach={"src": tmp}) as conn:
                cur = self._row(conn, uid)
                if cur is not None and cur["snap"] == snap.etag:
                    return False  # another process imported it meanwhile
                # Drops the previous import, plus strays left by older mirrors.
                self._remove_in(conn, uid)
                new_id = db.import_attached_snapshot(
                    conn,
                    name=(edit or {}).get("name"),
                    project=(edit or {}).get("project"),
                    log_dir=self._run_dir(uid),
                )
                conn.execute(
                    "INSERT INTO sync_mirror"
                    "(source, uid, exp_id, snap, gen, seq, edit, media_prefix) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        self.source, uid, new_id, snap.etag,
                        int(meta.get("sync_gen") or 0),
                        edit_obj.etag if edit_obj is not None else None,
                        _media_prefix(uid, meta.get("sync_media_prefix", "")),
                    ),
                )
            return True
        finally:
            for suffix in ("", "-journal", "-wal", "-shm"):
                try:
                    os.unlink(tmp + suffix)
                except OSError:
                    pass

    def _apply_chunks(
        self, db: Database, uid: str, row: Any, chunks: Sequence[Tuple[int, int, Any]]
    ) -> bool:
        """Apply chunks of the current snapshot not applied yet, in order."""
        gen = int(row["gen"])
        applied = int(row["seq"])
        changed = False
        for seq, obj in sorted((s, o) for g, s, o in chunks if g == gen and s > applied):
            if seq <= applied:
                continue
            if seq != applied + 1:
                raise StorageError(f"missing delta {applied + 1} for {uid}")
            raw = self.storage.get_bytes(obj.key)
            if raw is None:
                raise StorageError(f"delta {seq} disappeared for {uid}")
            chunk = json.loads(gzip.decompress(raw).decode("utf-8"))
            if not isinstance(chunk, dict) or not isinstance(chunk.get("rows"), dict):
                raise ValueError(f"invalid delta {seq}")
            with db.transaction() as conn:
                cur = self._row(conn, uid)
                if cur is None or int(cur["gen"]) != gen:
                    return changed  # re-imported meanwhile
                if int(cur["seq"]) != seq - 1:
                    # Another process got here first.
                    applied = int(cur["seq"])
                    continue
                exp_id = int(cur["exp_id"])
                db.apply_delta_in(
                    conn, exp_id, chunk.get("rows") or {}, run_info=chunk.get("run_info")
                )
                if cur["edit"] is None and chunk.get("name"):
                    db.set_name_and_project_in(
                        conn, exp_id, str(chunk["name"]), str(chunk.get("project", ""))
                    )
                conn.execute(
                    "UPDATE sync_mirror SET seq = ? WHERE source = ? AND uid = ?",
                    (seq, self.source, uid),
                )
            applied = seq
            changed = True
        return changed

    def _drop_local(self, uid: str) -> None:
        with self._database().transaction() as conn:
            self._remove_in(conn, uid)
        shutil.rmtree(self._run_dir(uid), ignore_errors=True)

    # ── media ───────────────────────────────────────────────────

    def _locate(self, path: str) -> Optional[Tuple[str, str]]:
        """``(uid, media key)`` for an absolute path inside the mirror."""
        try:
            rel = Path(os.path.realpath(path)).relative_to(os.path.realpath(self.runs_dir))
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) < 3 or not _UID.fullmatch(parts[0]) or parts[1] != "media":
            return None
        row = self._row(self._conn(), parts[0])
        if row is None or not row["media_prefix"]:
            return None
        prefix = _media_prefix(parts[0], row["media_prefix"])
        return parts[0], f"{prefix}/{'/'.join(parts[1:])}"

    def ensure_local(self, path: str) -> bool:
        """Download a mirrored media file on first use; True if it is local."""
        if os.path.isfile(path):
            return True
        loc = self._locate(path)
        if loc is None:
            return False
        with self._fetch_guard:
            lock = self._fetch_locks.setdefault(path, threading.Lock())
        with lock:
            if os.path.isfile(path):
                return True
            try:
                return self.storage.get_file(loc[1], path)
            except (StorageError, OSError) as exc:
                self._warn(f"could not fetch {loc[1]} from {display_uri(self.uri)}: {exc}")
                return False

    def direct_url(self, path: str, force: bool = False) -> Optional[str]:
        """Presigned URL for a mirrored file the browser should stream directly.

        Videos and files of at least :data:`PRESIGN_MIN_BYTES` qualify;
        ``presign`` mode adds images and audio, ``proxy`` mode never presigns.
        *force* presigns any mirrored file (for server-side readers).
        """
        mode = self.storage.links
        if mode == "proxy" and not force:
            return None
        loc = self._locate(path)
        if loc is None:
            return None
        lower = path.lower()
        direct = lower.endswith(_VIDEO_EXTS) or (
            mode == "presign" and lower.endswith(_PLAIN_MEDIA_EXTS)
        )
        if not force and not direct:
            try:
                info = self.storage.head(loc[1])
            except (StorageError, OSError):
                return None
            if info is None or info.size < PRESIGN_MIN_BYTES:
                return None
        try:
            return self.storage.presign(loc[1])
        except Exception:
            return None

    def _download_media(self, uid: str, prefix: str, obj: ObjectInfo) -> bool:
        parts = obj.key[len(prefix):].split("/")
        if not obj.key.startswith(prefix) or not _safe_key_parts(parts):
            raise ValueError("invalid media key")
        media = Path(self._run_dir(uid)) / "media"
        dest = media.joinpath(*parts)
        if dest.resolve() != dest or media not in dest.parents:
            raise ValueError("unsafe media destination")
        identity = (self.source, uid, obj.key)
        conn = self._conn()
        version = conn.execute(
            "SELECT etag FROM sync_media WHERE source=? AND uid=? AND key=?", identity,
        ).fetchone()
        if (obj.etag and version and version[0] == obj.etag
                and dest.is_file() and dest.stat().st_size == obj.size):
            return False
        # Download outside the SQLite write transaction so a slow object store
        # doesn't block concurrent mirror readers that need to refresh.
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".vt-", suffix=".part")
        os.close(fd)
        try:
            if not self.storage.get_file(obj.key, tmp):
                raise StorageError("object disappeared during pull")
            current = self.storage.head(obj.key)
            if current is None or current.etag != obj.etag or os.path.getsize(tmp) != obj.size:
                raise StorageError("object changed during pull; retry")
            with self._database().transaction() as conn:
                os.replace(tmp, dest)
                conn.execute("INSERT OR REPLACE INTO sync_media VALUES (?, ?, ?, ?)",
                             (*identity, obj.etag))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True

    def download_all_media(self, progress: Optional[Callable[[str], None]] = None) -> int:
        count = 0
        for uid, st in self.mirrored().items():
            try:
                prefix = _media_prefix(uid, st["media_prefix"]) + "/media/"
                for obj in self.storage.list(prefix):
                    try:
                        if self._download_media(uid, prefix, obj):
                            count += 1
                            if progress is not None:
                                progress(obj.key)
                    except (StorageError, OSError, ValueError) as exc:
                        self.run_errors.append(f"{obj.key}: {exc}")
            except (StorageError, OSError, ValueError) as exc:
                self.run_errors.append(f"{uid}: {exc}")
        return count

    # ── write-back from dashboards ──────────────────────────────

    def _by_experiment(self, experiment_id: int) -> Optional[Any]:
        return self._conn().execute(
            "SELECT uid, media_prefix FROM sync_mirror WHERE source = ? AND exp_id = ?",
            (self.source, int(experiment_id)),
        ).fetchone()

    def uid_for(self, experiment_id: int) -> Optional[str]:
        row = self._by_experiment(experiment_id)
        return str(row["uid"]) if row is not None else None

    def push_edits(self, db: Any, experiment_ids: Iterable[int]) -> None:
        """Record the current name/project of mirrored runs in the bucket."""
        with self._lock:
            for exp_id in experiment_ids:
                uid = self.uid_for(exp_id)
                row = db.get_experiment(exp_id) if uid else None
                if uid is None or row is None:
                    continue
                body = json.dumps({"name": row["name"], "project": row["project"]}).encode()
                self.storage.put_bytes(_run_key(uid, EDIT), body, "application/json")
                head = self.storage.head(_run_key(uid, EDIT))
                with self._database().transaction() as conn:
                    conn.execute(
                        "UPDATE sync_mirror SET edit = ? WHERE source = ? AND uid = ?",
                        (head.etag if head else None, self.source, uid),
                    )

    def delete_runs(self, experiment_ids: Iterable[int], wait: bool = False) -> List[str]:
        """Delete mirrored runs from the bucket (marker first, then objects)."""
        removed: List[Tuple[str, str]] = []
        with self._lock:
            for exp_id in experiment_ids:
                row = self._by_experiment(exp_id)
                if row is None:
                    continue
                uid, prefix = str(row["uid"]), str(row["media_prefix"] or "")
                self._run_dir(uid)
                _media_prefix(uid, prefix)
                self.storage.put_bytes(_run_key(uid, DELETED), b"", "text/plain")
                # Remove only the validated mirror directory.
                with self._database().transaction() as conn:
                    self._remove_in(conn, uid)
                shutil.rmtree(self._run_dir(uid), ignore_errors=True)
                removed.append((uid, prefix))

        def purge() -> None:
            for uid, prefix in removed:
                try:
                    _purge_run(self.storage, uid, prefix)
                except (StorageError, OSError) as exc:
                    self._warn(f"could not remove all objects of run {uid}: {exc}")

        if removed:
            if wait:
                purge()
            else:
                threading.Thread(target=purge, name="vibetrack-purge", daemon=True).start()
        return [uid for uid, _ in removed]


def _read_meta(path: str) -> Dict[str, str]:
    conn = sqlite3.connect(path)
    try:
        return {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM meta")}
    finally:
        conn.close()


# ── push / pull ─────────────────────────────────────────────────────


def push_runs(
    db: Database,
    uri: str,
    project: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    remove_local: bool = False,
    out: Callable[[str], None] = print,
) -> int:
    """Upload experiments of *db* to *uri*. Returns a process exit code."""
    storage = open_storage(uri)
    rows = db.list_experiments(project=project)
    if names:
        wanted = set(names)
        rows = [r for r in rows if r["name"] in wanted]
    if not rows:
        out("No runs to push.")
        return 0
    failed = 0
    for row in rows:
        label = f"{row['project']}/{row['name']}" if row["project"] else row["name"]
        problems: List[str] = []
        up = RunUploader(storage, db, int(row["id"]), warn=problems.append)
        try:
            ok = up.archive() if remove_local else up.sync(final=True)
        except Exception as exc:
            out(f"  failed  {label}: {exc}")
            failed += 1
            continue
        if not ok:
            out(f"  skipped {label}: deleted from {display_uri(uri)}")
            continue
        action = "moved " if remove_local else "pushed"
        out(f"  {action}  {label}")
    out(f"Runs are at {display_uri(uri)} — open with: vibetrack {shlex.quote(uri)}")
    return 1 if failed else 0


def pull(uri: str, dest: str, out: Callable[[str], None] = print) -> int:
    """Copy every run at *uri*, with all media, into project folder *dest*."""
    remote = RemoteProject(uri, cache_dir=dest)
    try:
        remote.refresh()
        if remote.last_error:
            out(f"Error: {remote.last_error}")
            return 1
        n = remote.download_all_media()
        runs = len(remote.mirrored())
        if remote.run_errors:
            for error in remote.run_errors:
                out(f"Error: {error}")
            out(f"Incomplete pull into {dest}")
            return 1
    finally:
        remote.close()
    out(f"Pulled {runs} run(s) and {n} media file(s) into {dest} — open with: vibetrack {shlex.quote(dest)}")
    return 0


