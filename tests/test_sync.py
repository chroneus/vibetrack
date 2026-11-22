"""Syncing runs to object storage (s3://, file://) and reading them back."""

import asyncio
import gzip
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from vibetrack import storage as storage_mod
from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.storage import Credentials, open_storage, presign_v4, sign_v4
from vibetrack.sync import RemoteProject, RunUploader, pull, push_runs
from vibetrack.writer import SummaryWriter

REPO = Path(__file__).resolve().parent.parent

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep caches, config and the central DB inside tmp_path."""
    for var in list(os.environ):
        if var.startswith(("VIBETRACK_S3_", "AWS_")) or var in ("VIBETRACK_SYNC",):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VIBETRACK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VIBETRACK_SYNC_INTERVAL", "3600")  # tests tick by hand
    monkeypatch.setenv("VIBETRACK_REPORT", "0")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-creds"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    import vibetrack.config  # noqa: F401  (the package attribute is the run config dict)

    cfg_mod = sys.modules["vibetrack.config"]
    cfg = tmp_path / "cfg"
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: cfg)
    monkeypatch.setattr(cfg_mod, "config_path", lambda: cfg / "config.json")
    central = tmp_path / "home" / "vibetrack.db"
    monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central)
    monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central)


def _uri(tmp_path, name="bucket"):
    return "file://" + str(tmp_path / name)


def _writer(tmp_path, name="run", project="cv", folder="host_a", **kwargs):
    kwargs.setdefault("system_metrics_interval", 0)
    return SummaryWriter(
        str(tmp_path / folder / "logs" / project / name),
        name=name,
        project=project,
        project_folder=str(tmp_path / folder),
        **kwargs,
    )


def _image():
    return np.full((8, 8, 3), 200, dtype=np.uint8)


def _bucket_keys(tmp_path, name="bucket"):
    root = tmp_path / name
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _mirror(uri, tmp_path, name="viewer"):
    remote = RemoteProject(uri, cache_dir=str(tmp_path / name))
    remote.refresh()
    return remote


def _experiments(remote):
    db = Database(remote.db_path)
    try:
        return {r["name"]: dict(r) for r in db.list_experiments()}, db
    except Exception:
        db.close()
        raise


# ── SigV4 ───────────────────────────────────────────────────────────

_AK = Credentials("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
_DATE = "20130524T000000Z"
_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_HOST = "examplebucket.s3.amazonaws.com"


class TestSigV4:
    """Examples published in the AWS S3 SigV4 documentation."""

    def _sig(self, *args):
        return sign_v4(*args).rsplit("Signature=", 1)[1]

    def test_get_object(self):
        headers = {"host": _HOST, "range": "bytes=0-9", "x-amz-content-sha256": _EMPTY, "x-amz-date": _DATE}
        sig = self._sig("GET", "/test.txt", [], headers, _EMPTY, _AK, "us-east-1", _DATE)
        assert sig == "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"

    def test_put_object(self):
        payload = "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        headers = {
            "date": "Fri, 24 May 2013 00:00:00 GMT",
            "host": _HOST,
            "x-amz-content-sha256": payload,
            "x-amz-date": _DATE,
            "x-amz-storage-class": "REDUCED_REDUNDANCY",
        }
        sig = self._sig("PUT", "/test%24file.text", [], headers, payload, _AK, "us-east-1", _DATE)
        assert sig == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"

    def test_list_and_subresource(self):
        headers = {"host": _HOST, "x-amz-content-sha256": _EMPTY, "x-amz-date": _DATE}
        assert self._sig(
            "GET", "/", [("max-keys", "2"), ("prefix", "J")], headers, _EMPTY, _AK, "us-east-1", _DATE
        ) == "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
        assert self._sig(
            "GET", "/", [("lifecycle", "")], headers, _EMPTY, _AK, "us-east-1", _DATE
        ) == "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"

    def test_presigned_url(self):
        params = dict(presign_v4(_HOST, "/test.txt", _AK, "us-east-1", _DATE, 86400))
        assert params["X-Amz-Signature"] == (
            "aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
        )


class TestS3Settings:
    def test_custom_endpoint_defaults_to_path_style_and_us_east_1(self):
        st = storage_mod.S3Storage("s3://runs/team/a?endpoint=minio.lab:9000")
        assert st.endpoint == "https://minio.lab:9000"
        assert st.addressing_style == "path"
        assert st.region == "us-east-1"
        assert st.prefix == "team/a"
        assert st._host_and_path("team/a/x y.png") == ("https", "minio.lab:9000", "/runs/team/a/x%20y.png")

    def test_aws_default_is_virtual_hosted(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        st = storage_mod.S3Storage("s3://runs")
        assert st.endpoint == "https://s3.eu-west-1.amazonaws.com"
        assert st._host_and_path("k")[1] == "runs.s3.eu-west-1.amazonaws.com"

    def test_precedence_uri_then_env_then_config(self, monkeypatch, tmp_path):
        from vibetrack.config import save_config

        save_config({"s3": {"endpoint_url": "http://cfg:1", "region": "cfg-region",
                            "access_key_id": "CFGKEY", "secret_access_key": "cfgsecret"}})
        monkeypatch.setenv("VIBETRACK_S3_REGION", "env-region")
        st = storage_mod.S3Storage("s3://b/p?region=uri-region")
        assert st.region == "uri-region"
        assert st.endpoint == "http://cfg:1"
        assert st.creds == Credentials("CFGKEY", "cfgsecret", None)
        assert storage_mod.S3Storage("s3://b/p").region == "env-region"

    def test_aws_profile_files(self, monkeypatch, tmp_path):
        creds = tmp_path / "creds"
        creds.write_text("[lab]\naws_access_key_id = LABKEY\naws_secret_access_key = labsecret\n")
        conf = tmp_path / "conf"
        conf.write_text("[profile lab]\nregion = ru-central1\nendpoint_url = https://storage.lab\n")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
        monkeypatch.setenv("AWS_CONFIG_FILE", str(conf))
        st = storage_mod.S3Storage("s3://b?profile=lab")
        assert st.creds.access_key == "LABKEY"
        assert st.region == "ru-central1"
        assert st.endpoint == "https://storage.lab"

    def test_s3_keys_never_reach_the_web_api(self):
        from vibetrack.config import load_public_config, save_config

        save_config({"s3": {"secret_access_key": "SECRET"}})
        assert load_public_config()["s3"] == {}


# ── Single-run export / import ──────────────────────────────────────


def test_export_import_roundtrip(tmp_path):
    w = _writer(tmp_path, config={"lr": 0.1})
    for i in range(4):
        w.add_scalar("loss", 1.0 / (i + 1), i)
    w.add_text("note", "hello", 0)
    w.add_image("img", _image(), 0, dataformats="HWC")
    w.add_histogram("h", np.arange(10.0), 0)
    w.add_video("clip", b"fake-mp4", 0)
    w.close()

    src = Database(tmp_path / "host_a" / "vibetrack.db")
    exp_id = src.list_experiments()[0]["id"]
    snap = tmp_path / "snap.db"
    src.export_experiment(exp_id, snap, meta={"sync_uid": "u"})
    assert not Path(str(snap) + "-wal").exists()

    dst = Database(tmp_path / "mirror.db")
    new_id = dst.import_snapshot(snap, log_dir="/somewhere/else")
    for table in ("scalars", "texts", "histograms", "hparams", "video", "images"):
        getter = getattr(src, f"get_all_{table}", None)
        if getter is not None:
            assert getter(exp_id).keys() == getattr(dst, f"get_all_{table}")(new_id).keys()
    assert [r["value"] for r in dst.get_scalars(new_id, "loss")] == [
        r["value"] for r in src.get_scalars(exp_id, "loss")
    ]
    assert dst.get_images(new_id, "img")[0]["path"] == "media/img/0.png"
    assert dst.get_hparams(new_id) == {"lr": 0.1}
    assert dst.get_run_info(new_id)["status"] == "finished"
    assert dst.get_experiment(new_id)["log_dir"] == "/somewhere/else"


# ── Live sync from a writer ─────────────────────────────────────────


def test_live_sync_deltas_media_and_final_snapshot(tmp_path):
    uri = _uri(tmp_path)
    w = _writer(tmp_path, sync=uri)
    for i in range(3):
        w.add_scalar("loss", 1.0 / (i + 1), i)
    w.add_image("img", _image(), 0, dataformats="HWC")
    assert w._sync.tick()

    remote = _mirror(uri, tmp_path)
    exps, db = _experiments(remote)
    run = exps["run"]
    assert run["project"] == "cv"
    assert db.get_run_info(run["id"])["status"] == "running"
    assert len(db.get_scalars(run["id"], "loss")) == 3

    # Second pass sends only the new rows, as one small chunk.
    for i in range(3, 6):
        w.add_scalar("loss", 1.0 / (i + 1), i)
    w._sync.tick()
    chunks = [k for k in _bucket_keys(tmp_path) if "/d/" in k]
    assert len(chunks) == 1
    rows = json.loads(gzip.decompress((tmp_path / "bucket" / chunks[0]).read_bytes()))["rows"]
    assert len(rows["scalars"]) == 3 and "images" not in rows
    remote.refresh()
    assert len(db.get_scalars(run["id"], "loss")) == 6  # appended, not re-imported

    # Media arrives lazily, on first access.
    reader = RunReader(uri)
    exp = reader.experiments()[0]
    img = exp.images("img")[0]["abs_path"]
    assert not os.path.exists(img)
    assert exp.ensure_local(img) and os.path.getsize(img) > 0

    w.close()
    remote.refresh()
    run = _experiments(remote)[0]["run"]  # the final snapshot replaced it
    assert db.get_run_info(run["id"])["status"] == "finished"
    assert len(db.get_scalars(run["id"], "loss")) == 6
    keys = _bucket_keys(tmp_path)
    assert not [k for k in keys if "/d/" in k], "close compacts chunks into the snapshot"
    assert any(k.endswith("/snap.db.gz") for k in keys)
    assert any(k.startswith("cv/run-") and k.endswith("media/img/0.png") for k in keys)
    db.close()
    reader.close()
    remote.close()


def test_nothing_uploaded_when_nothing_changed(tmp_path):
    uri = _uri(tmp_path)
    w = _writer(tmp_path, sync=uri)
    w.add_scalar("loss", 1.0, 0)
    w._sync.tick()
    w._sync.tick()
    w._sync.tick()
    assert not [k for k in _bucket_keys(tmp_path) if "/d/" in k]
    w.close()


def test_two_hosts_same_run_name_both_visible(tmp_path):
    uri = _uri(tmp_path)
    for host in ("host_a", "host_b"):
        w = _writer(tmp_path, folder=host, sync=uri)
        w.add_scalar("loss", 1.0, 0)
        w.close()
    remote = _mirror(uri, tmp_path)
    exps, db = _experiments(remote)
    assert sorted(exps) == ["run", "run (2)"]
    db.close()
    remote.close()


def test_purge_on_resume_uploads_a_fresh_snapshot(tmp_path):
    uri = _uri(tmp_path)
    w = _writer(tmp_path, sync=uri)
    for i in range(5):
        w.add_scalar("loss", float(i), i)
    w._sync.tick()
    w._db.purge_experiment_from_step(w.experiment_id, 2)
    w.add_scalar("loss", 9.0, 2)
    w._sync.tick()
    remote = _mirror(uri, tmp_path)
    exps, db = _experiments(remote)
    assert [r["value"] for r in db.get_scalars(exps["run"]["id"], "loss")] == [0.0, 1.0, 9.0]
    w.close()
    db.close()
    remote.close()


def test_unreachable_endpoint_never_blocks_training(tmp_path, monkeypatch, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again: connections are refused
    monkeypatch.setenv("VIBETRACK_SYNC_TIMEOUT", "3")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    w = _writer(tmp_path, sync=f"s3://b/p?endpoint=http://127.0.0.1:{port}")
    w._sync.storage.retries = 0
    for i in range(10):
        w.add_scalar("loss", float(i), i)
    w._sync.tick()
    started = time.time()
    w.close()
    assert time.time() - started < 15
    err = capsys.readouterr().err
    assert "could not sync" in err and "training is not affected" in err
    assert "vibetrack push" in err
    with RunReader(str(tmp_path / "host_a")) as reader:
        assert len(reader.experiments(project=None)[0].scalars("loss")) == 10


@pytest.fixture
def hung_endpoint():
    """An S3 endpoint that accepts connections and never answers."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(32)
    srv.settimeout(0.1)
    conns = []
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                conns.append(srv.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    stop.set()
    thread.join(2)
    for conn in conns:
        conn.close()
    srv.close()


@pytest.mark.parametrize("interval", ["3600", "0.1"], ids=["final-pass", "background-in-flight"])
def test_final_sync_ends_within_timeout_on_a_hung_endpoint(
    tmp_path, monkeypatch, capsys, hung_endpoint, interval
):
    monkeypatch.setenv("VIBETRACK_SYNC_TIMEOUT", "1.5")
    monkeypatch.setenv("VIBETRACK_SYNC_INTERVAL", interval)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    uri = f"s3://b/p?endpoint={hung_endpoint}"
    w = _writer(tmp_path, sync=uri)
    w.add_scalar("loss", 1.0, 0)
    if interval != "3600":
        time.sleep(0.5)  # the background thread is now stuck in a request
    started = time.time()
    w.close()
    # Before, every request (and each retry) got its own full timeout.
    assert time.time() - started < 4
    err = capsys.readouterr().err
    folder = str((tmp_path / "host_a").resolve())
    hint = f"vibetrack push {shlex.quote(folder)} {shlex.quote(uri)} run -p cv"
    assert hint in err


def test_push_hint_uploads_the_run(tmp_path, capsys):
    from vibetrack.commands import run

    uri = _uri(tmp_path, "my bucket")  # a space: the hint must quote it
    w = _writer(tmp_path, sync=uri)
    w.add_scalar("loss", 1.0, 0)
    w._sync.tick()
    hint = w._sync.push_command()
    w.close()
    shutil.rmtree(tmp_path / "my bucket")
    argv = shlex.split(hint)
    assert argv[:2] == ["vibetrack", "push"]
    assert run(argv[1:]) == 0
    assert "pushed  cv/run" in capsys.readouterr().out
    remote = _mirror(uri, tmp_path)
    assert list(_experiments(remote)[0]) == ["run"]
    remote.close()


def test_printed_locations_are_shell_quoted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VIBETRACK_REPORT", "1")
    uri = _uri(tmp_path, "my bucket")
    w = _writer(tmp_path, sync=uri)
    w.add_scalar("loss", 1.0, 0)
    w.close()
    assert f"view anywhere: vibetrack {shlex.quote(uri)}" in capsys.readouterr().err
    lines = []
    db = Database(tmp_path / "host_a" / "vibetrack.db")
    push_runs(db, uri, out=lines.append)
    db.close()
    assert f"open with: vibetrack {shlex.quote(uri)}" in lines[-1]
    dest = str(tmp_path / "offline copy")
    pull(uri, dest, out=lines.append)
    assert lines[-1].endswith(f"open with: vibetrack {shlex.quote(dest)}")


def test_run_failing_early_is_removed_from_storage(tmp_path):
    uri = _uri(tmp_path)
    with pytest.raises(ValueError):
        with _writer(tmp_path, sync=uri) as w:
            w.add_scalar("loss", 1.0, 0)
            w.add_image("img", _image(), 0, dataformats="HWC")
            w._sync.tick()  # uploaded before the crash (short sync interval)
            assert any(k.endswith("snap.db.gz") for k in _bucket_keys(tmp_path))
            raise ValueError("bad shape")
    # Only the delete marker is left, so readers skip the run.
    keys = _bucket_keys(tmp_path)
    assert len(keys) == 1 and keys[0].endswith("/deleted")
    remote = _mirror(uri, tmp_path)
    assert _experiments(remote)[0] == {}
    remote.close()


def test_run_failing_before_its_first_upload_sends_nothing(tmp_path):
    uri = _uri(tmp_path)
    with pytest.raises(ValueError):
        with _writer(tmp_path, sync=uri) as w:
            w.add_scalar("loss", 1.0, 0)
            raise ValueError("bad shape")
    assert _bucket_keys(tmp_path) == []


def test_push_remove_local_keeps_media_of_a_shared_log_dir(tmp_path):
    shared = tmp_path / "host_a" / "logs" / "shared"
    for name, step in (("a", 0), ("b", 1)):
        with SummaryWriter(str(shared), name=name, project="cv",
                           project_folder=str(tmp_path / "host_a"),
                           system_metrics_interval=0) as w:
            w.add_image("img", _image(), step, dataformats="HWC")
    db = Database(tmp_path / "host_a" / "vibetrack.db")
    assert push_runs(db, _uri(tmp_path), names=["b"], remove_local=True, out=lambda _: None) == 0
    assert [r["name"] for r in db.list_experiments()] == ["a"]
    db.close()
    assert (shared / "media" / "img" / "0.png").is_file(), "run a's media was deleted"


# ── One mirror shared by several readers (dashboard, its MCP, CLI) ──


def _scalar_counts(remote):
    db = Database(remote.db_path)
    try:
        return {
            r["name"]: len(db.get_scalars(r["id"], "loss")) for r in db.list_experiments()
        }
    finally:
        db.close()


def test_two_readers_of_one_mirror_apply_each_chunk_once(tmp_path):
    uri = _uri(tmp_path)
    w = _writer(tmp_path, sync=uri)
    for i in range(5):
        w.add_scalar("loss", float(i), i)
    w._sync.tick()
    dashboard = _mirror(uri, tmp_path)
    for i in range(5, 10):
        w.add_scalar("loss", float(i), i)
    w._sync.tick()
    cli = _mirror(uri, tmp_path)  # same cache folder, e.g. `vibetrack ls`
    cli.close()
    dashboard.refresh()
    assert _scalar_counts(dashboard) == {"run": 10}

    w.close()  # final snapshot
    cli = _mirror(uri, tmp_path)
    cli.close()
    dashboard.refresh()  # used to re-import it as "run (2)"
    assert _scalar_counts(dashboard) == {"run": 10}
    dashboard.close()


def test_interrupted_refresh_never_duplicates_runs(tmp_path):
    uri = _uri(tmp_path)
    for name in ("a", "b", "c"):
        w = _writer(tmp_path, name=name, sync=uri)
        w.add_scalar("loss", 1.0, 0)
        w.close()
    remote = RemoteProject(uri, cache_dir=str(tmp_path / "viewer"))
    get_bytes = remote.storage.get_bytes
    snapshots = []

    def interrupted(key):
        if key.endswith("snap.db.gz"):
            snapshots.append(key)
            if len(snapshots) == 3:
                raise KeyboardInterrupt  # Ctrl-C during the first `vibetrack ls`
        return get_bytes(key)

    remote.storage.get_bytes = interrupted
    with pytest.raises(KeyboardInterrupt):
        remote.refresh()
    remote.close()
    remote = _mirror(uri, tmp_path)
    assert _scalar_counts(remote) == {"a": 1, "b": 1, "c": 1}
    remote.close()


def test_new_snapshot_replaces_stray_copies_of_a_run(tmp_path):
    """Mirrors written by earlier versions could hold duplicates of a run."""
    uri = _uri(tmp_path)
    w = _writer(tmp_path, sync=uri)
    w.add_scalar("loss", 1.0, 0)
    w._sync.tick()
    remote = _mirror(uri, tmp_path)
    exps, db = _experiments(remote)
    stray = db.create_experiment("run (2)", project="cv", log_dir=exps["run"]["log_dir"])
    db.add_scalar(stray, "loss", 1.0, 0)
    db.close()
    w.add_scalar("loss", 2.0, 1)
    w.close()
    remote.refresh()
    assert _scalar_counts(remote) == {"run": 2}
    remote.close()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="POSIX only")
def test_sigterm_marks_interrupted_and_uploads(tmp_path):
    uri = _uri(tmp_path)
    script = tmp_path / "train.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import time, vibetrack
            vibetrack.init(project="cv", name="spot", project_folder={str(tmp_path / 'host_a')!r},
                           log_dir={str(tmp_path / 'logs' / 'spot')!r}, system_metrics_interval=0)
            for i in range(20):
                vibetrack.log({{"loss": 1.0 / (i + 1)}}, step=i)
            print("ready", flush=True)
            time.sleep(60)
            """
        )
    )
    # A preemption in the first minute would discard the run; test the upload.
    env = dict(os.environ, VIBETRACK_SYNC=uri, VIBETRACK_GIT="0", VIBETRACK_DISCARD_FAILED_SECS="0")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    remote = _mirror(uri, tmp_path)
    exps, db = _experiments(remote)
    info = db.get_run_info(exps["spot"]["id"])
    assert info["status"] == "interrupted"
    assert info["info"]["error"] == "SIGTERM"
    assert len(db.get_scalars(exps["spot"]["id"], "loss")) == 20
    db.close()
    remote.close()


# ── Dashboard on synced storage ─────────────────────────────────────


class _Client:
    def __init__(self, app):
        self._app = app

    def request(self, method, url, **kwargs):
        httpx = pytest.importorskip("httpx")

        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:6116") as c:
                return await c.request(method, url, **kwargs)

        return asyncio.run(_run())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def dashboard():
    pytest.importorskip("fastapi")
    from vibetrack.viewers import web
    from vibetrack.viewers.web import WebOutput

    web._mutation_rate_state.clear()
    made = []

    def _make(uri):
        out = WebOutput(uri)
        made.append(out)
        return out, _Client(out._build_app(None, "127.0.0.1"))

    yield _make
    for out in made:
        out.close()


def test_dashboard_reads_storage_and_writes_edits_back(tmp_path, dashboard):
    uri = _uri(tmp_path)
    live = _writer(tmp_path, name="live", sync=uri)
    live.add_scalar("loss", 1.0, 0)
    live.add_image("img", _image(), 0, dataformats="HWC")
    live._sync.tick()
    done = _writer(tmp_path, name="done", sync=uri)
    done.add_scalar("loss", 2.0, 0)
    done.close()

    out, client = dashboard(uri)
    assert client.get("/api/info").json()["storage"] == uri
    assert client.get("/api/projects").json() == ["cv"]
    data = client.get("/api/data/cv").json()
    assert {e["name"] for e in data} == {"live", "done"}

    img_path = next(e for e in data if e["name"] == "live")["images"]["img"][0]["path"]
    resp = client.get("/media", params={"path": img_path})
    assert resp.status_code == 200 and resp.content[:4] == b"\x89PNG"

    # Rename a running run: the edit wins over the writer's later uploads.
    resp = client.post("/api/rename/cv", json={"old_name": "live", "new_name": "best"})
    assert resp.status_code == 200, resp.text
    assert any(k.endswith("/edit.json") for k in _bucket_keys(tmp_path))
    live.add_scalar("loss", 0.5, 1)
    live._sync.tick()
    other = _mirror(uri, tmp_path, "laptop2")
    exps, db = _experiments(other)
    assert "best" in exps and "live" not in exps
    assert len(db.get_scalars(exps["best"]["id"], "loss")) == 2
    db.close()

    # Delete: marker first, then the run's objects; its writer stops syncing.
    resp = client.request("DELETE", "/api/experiment", json={"name": "best", "project": "cv"})
    assert resp.status_code == 200, resp.text
    out._reader.remote.delete_runs([], wait=True)
    deadline = time.time() + 5
    while time.time() < deadline and any("cv/live-" in k for k in _bucket_keys(tmp_path)):
        time.sleep(0.05)
    keys = _bucket_keys(tmp_path)
    assert any(k.endswith("/deleted") for k in keys)
    assert not any("cv/live-" in k for k in keys)
    live.add_scalar("loss", 0.1, 2)
    assert live._sync.tick() is False
    live.close()
    other.refresh()
    exps, db = _experiments(other)
    assert sorted(exps) == ["done"]
    db.close()
    other.close()

    resp = client.post("/api/move-logdir", json={"old": "/a", "new": "/b"})
    assert resp.status_code == 400


def test_dashboard_rejects_remote_ingest(tmp_path, dashboard):
    _, client = dashboard(_uri(tmp_path))
    resp = client.post("/cv/listen/log", json={"experiment": "x", "step": 0, "scalars": {"a": 1}})
    assert resp.status_code == 400


# ── push / pull ─────────────────────────────────────────────────────


def _local_runs(tmp_path, names=("a", "b")):
    for name in names:
        w = _writer(tmp_path, name=name)
        w.add_scalar("loss", 1.0, 0)
        w.add_image("img", _image(), 0, dataformats="HWC")
        w.close()
    return Database(tmp_path / "host_a" / "vibetrack.db")


def test_push_is_idempotent_and_skips_deleted_runs(tmp_path):
    uri = _uri(tmp_path)
    db = _local_runs(tmp_path)
    lines = []
    assert push_runs(db, uri, out=lines.append) == 0
    assert push_runs(db, uri, out=lines.append) == 0
    remote = _mirror(uri, tmp_path)
    exps, mdb = _experiments(remote)
    assert sorted(exps) == ["a", "b"]
    remote.delete_runs([exps["a"]["id"]], wait=True)
    lines.clear()
    push_runs(db, uri, out=lines.append)
    assert any(line.startswith("  skipped cv/a") for line in lines)
    remote.refresh()
    assert sorted(_experiments(remote)[0]) == ["b"]
    mdb.close()
    remote.close()
    db.close()


def test_push_remove_local_only_after_verified_upload(tmp_path, monkeypatch):
    uri = _uri(tmp_path)
    db = _local_runs(tmp_path)
    log_dir = db.get_experiment_by_name("a")["log_dir"]
    monkeypatch.setattr(RunUploader, "verify_uploaded", lambda self: False)
    assert push_runs(db, uri, names=["a"], remove_local=True, out=lambda _: None) == 1
    assert db.get_experiment_by_name("a") is not None
    assert os.path.isdir(os.path.join(log_dir, "media"))
    monkeypatch.undo()
    monkeypatch.setenv("VIBETRACK_CACHE_DIR", str(tmp_path / "cache"))
    assert push_runs(db, uri, names=["a"], remove_local=True, out=lambda _: None) == 0
    assert db.get_experiment_by_name("a") is None
    assert not os.path.exists(os.path.join(log_dir, "media"))
    db.close()


def test_pull_copies_runs_and_media(tmp_path):
    uri = _uri(tmp_path)
    db = _local_runs(tmp_path)
    push_runs(db, uri, out=lambda _: None)
    db.close()
    dest = tmp_path / "offline"
    assert pull(uri, str(dest), out=lambda _: None) == 0
    with RunReader(str(dest)) as reader:
        exps = reader.experiments(project=None)
        assert sorted(e.name for e in exps) == ["a", "b"]
        for exp in exps:
            assert os.path.isfile(exp.images("img")[0]["abs_path"])


def test_cli_push_pull_and_ls(tmp_path, capsys):
    from vibetrack.commands import run

    uri = _uri(tmp_path)
    _local_runs(tmp_path).close()
    assert run(["push", str(tmp_path / "host_a"), uri]) == 0
    assert run(["ls", "--project-folder", uri, "-p", "cv"]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "b" in out
    assert run(["pull", uri, str(tmp_path / "copy")]) == 0
    assert (tmp_path / "copy" / "vibetrack.db").is_file()
    assert run(["push", str(tmp_path / "host_a")]) == 1


# ── A real S3 API: moto's standalone server on localhost ────────────


@pytest.fixture
def s3_endpoint(monkeypatch):
    moto_server = pytest.importorskip("moto.server")
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    endpoint = f"http://{host}:{port}"
    bucket = storage_mod.S3Storage(f"s3://runs?endpoint={endpoint}")
    bucket._request("PUT", "", raw_key=True)  # create the bucket
    yield endpoint
    server.stop()


def test_s3_live_sync_and_mirror(tmp_path, s3_endpoint):
    uri = f"s3://runs/team?endpoint={s3_endpoint}"
    w = _writer(tmp_path, sync=uri)
    for i in range(3):
        w.add_scalar("loss", float(i), i)
    w.add_image("img", _image(), 0, dataformats="HWC")
    w._sync.tick()
    w.add_scalar("loss", 3.0, 3)
    w._sync.tick()
    remote = RemoteProject(uri)
    remote.refresh()
    exps, db = _experiments(remote)
    assert len(db.get_scalars(exps["run"]["id"], "loss")) == 4
    w.close()
    remote.refresh()
    exps = _experiments(remote)[0]
    assert db.get_run_info(exps["run"]["id"])["status"] == "finished"
    keys = [o.key for o in open_storage(uri).list("")]
    assert not [k for k in keys if "/d/" in k]
    with RunReader(uri) as reader:
        exp = reader.experiments()[0]
        assert exp.ensure_local(exp.images("img")[0]["abs_path"])
    db.close()
    remote.close()


def test_s3_multipart_presign_and_list_pages(tmp_path, s3_endpoint, monkeypatch):
    monkeypatch.setattr(storage_mod, "MULTIPART_THRESHOLD", 6 * 1024 * 1024)
    monkeypatch.setattr(storage_mod, "_MIN_PART", 5 * 1024 * 1024)
    st = open_storage(f"s3://runs/p?endpoint={s3_endpoint}")
    blob = os.urandom(12 * 1024 * 1024)
    src = tmp_path / "big.bin"
    src.write_bytes(blob)
    st.put_file("big.bin", str(src))
    assert st.head("big.bin").size == len(blob)
    assert st.get_file("big.bin", str(tmp_path / "back.bin"))
    assert (tmp_path / "back.bin").read_bytes() == blob

    req = urllib.request.Request(st.presign("big.bin"), headers={"Range": "bytes=100-199"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206 and resp.read() == blob[100:200]

    for i in range(1005):  # more than one ListObjectsV2 page
        st.put_bytes(f"many/{i}", b"")
    assert len([o for o in st.list("many/")]) == 1005
    st.delete("big.bin")
    assert st.head("big.bin") is None and st.get_bytes("big.bin") is None


def test_s3_dashboard_streams_video_from_presigned_url(tmp_path, s3_endpoint, dashboard):
    uri = f"s3://runs/team?endpoint={s3_endpoint}"
    video = os.urandom(4096)
    w = _writer(tmp_path, sync=uri)
    w.add_video("clip", video, 0)
    w.add_image("img", _image(), 0, dataformats="HWC")
    w.close()
    _, client = dashboard(uri)
    exp = client.get("/api/data/cv").json()[0]
    clip = exp["video_data"]["clip"][0]["path"]
    resp = client.get("/media", params={"path": clip})
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location.startswith(s3_endpoint) and "X-Amz-Signature=" in location
    req = urllib.request.Request(location, headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(req) as r:
        assert r.read() == video[10:20]  # seeking works against storage directly
    assert not os.path.exists(clip), "streamed, not downloaded"
    # proxy=1 keeps it same-origin (frame compare draws it on a canvas).
    proxied = client.get("/media", params={"path": clip, "proxy": 1})
    assert proxied.status_code == 200 and proxied.content == video
    # Small images still go through the local cache.
    img = exp["images"]["img"][0]["path"]
    assert client.get("/media", params={"path": img}).status_code == 200


def test_snapshot_retry_publishes_a_base_before_deltas(tmp_path, monkeypatch):
    db = _local_runs(tmp_path, names=("a",))
    store = open_storage(_uri(tmp_path))
    up = RunUploader(store, db, db.list_experiments()[0]["id"])
    put = store.put_file

    def fail_snapshot(key, *args):
        if key.endswith("snap.db.gz"):
            raise OSError("interrupted")
        put(key, *args)

    with monkeypatch.context() as patch:
        patch.setattr(store, "put_file", fail_snapshot)
        with pytest.raises(OSError):
            up.sync()
    assert up.gen is None
    assert up.sync()
    assert pull(store.uri, str(tmp_path / "offline")) == 0
    with RunReader(str(tmp_path / "offline")) as reader:
        assert reader.experiments(project=None)[0].scalars("loss")[0]["value"] == 1
    db.close()


def test_archive_keeps_run_changed_after_snapshot(tmp_path, monkeypatch):
    db = _local_runs(tmp_path, names=("a",))
    eid = db.list_experiments()[0]["id"]
    verify = RunUploader.verify_uploaded

    def changed(up):
        assert verify(up)
        db.add_scalar(eid, "loss", 2, 1)
        return True

    monkeypatch.setattr(RunUploader, "verify_uploaded", changed)
    assert push_runs(db, _uri(tmp_path), remove_local=True) == 1
    assert db.get_experiment(eid) is not None
    assert len(db.get_scalars(eid, "loss")) == 2
    db.close()


def test_pull_refreshes_media_and_reports_incomplete_backup(tmp_path):
    db = _local_runs(tmp_path, names=("a",))
    uri = _uri(tmp_path)
    assert push_runs(db, uri) == 0
    db.close()
    source = next((tmp_path / "bucket").rglob("*.png"))
    source.write_bytes(b"first")
    dest = tmp_path / "offline"
    assert pull(uri, str(dest)) == 0
    source.write_bytes(b"other")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    store = open_storage(uri)
    store.put_bytes("_runs/0123456789abcdef/snap.db.gz", b"incomplete")
    assert pull(uri, str(dest)) == 1
    assert next(dest.rglob("*.png")).read_bytes() == b"other"


def test_mirror_requires_canonical_run_ids_and_media_destinations(tmp_path):
    from vibetrack.storage import ObjectInfo

    remote = RemoteProject(_uri(tmp_path), cache_dir=str(tmp_path / "cache"))
    with pytest.raises(ValueError):
        remote._run_dir("old-format-id")
    uid = "0123456789abcdef"
    media = Path(remote._run_dir(uid)) / "media"
    media.parent.mkdir(parents=True)
    media.symlink_to(tmp_path, target_is_directory=True)
    prefix = f"cv/a-{uid}/media/"
    obj = ObjectInfo(prefix + "image.png", 0, "version", 0)
    with pytest.raises(ValueError, match="destination"):
        remote._download_media(uid, prefix, obj)


def test_purge_requires_full_uid_ownership(tmp_path):
    from vibetrack.sync import _media_prefix, _purge_run

    store = open_storage(_uri(tmp_path))
    uid = "0123456789abcdef"
    legacy = "cv/a-" + uid[:8]
    other = "cv/a-fedcba9876543210"
    for prefix in (legacy, other):
        store.put_bytes(prefix + "/media/sample", b"sample")
    assert _media_prefix(uid, legacy) == legacy
    _purge_run(store, uid, legacy)
    with pytest.raises(ValueError):
        _purge_run(store, uid, other)
    assert all(store.exists(p + "/media/sample") for p in (legacy, other))
