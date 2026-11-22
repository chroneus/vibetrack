"""Object storage for synced runs: any S3-compatible store, or a plain folder.

``open_storage(uri)`` accepts:

* ``s3://bucket/prefix`` — AWS S3 or any S3-compatible service (MinIO, Ceph
  RGW, Cloudflare R2, Backblaze B2, Wasabi, Yandex, GCS interop, ...).
  Non-secret settings may ride in the query string, so one string carries the
  whole location: ``s3://bucket/prefix?endpoint=https://minio.lab:9000``.
* ``file:///abs/path`` — a folder, e.g. on a shared drive.

Stdlib only: requests are signed with AWS Signature Version 4 and sent with
``urllib``. Only calls every S3-compatible store implements are used: Put/
Get/Head/Delete Object, ListObjectsV2 (falling back to ListObjects) and
multipart upload. Keys passed to a backend are relative to the URI prefix.

S3 settings, first match wins: URI query > ``VIBETRACK_S3_*`` environment >
``"s3"`` section of ``~/.vibetrack/config.json`` > the standard ``AWS_*``
environment and ``~/.aws/credentials`` / ``~/.aws/config`` profile files.
"""

from __future__ import annotations

import configparser
import datetime as _dt
import hashlib
import hmac
import logging
import os
import shutil
import ssl
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple
from urllib import error as _urlerr
from urllib import request as _urlreq
from urllib.parse import parse_qsl, quote, urlsplit

_log = logging.getLogger(__name__)

REMOTE_SCHEMES = ("s3://", "file://")

#: Files at least this big are uploaded in parts.
MULTIPART_THRESHOLD = 64 * 1024 * 1024
_MIN_PART = 16 * 1024 * 1024
_MAX_PARTS = 9000

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_LINK_MODES = ("auto", "proxy", "presign")


class StorageError(RuntimeError):
    """A storage request failed for a reason other than a missing object."""


class ObjectInfo(NamedTuple):
    key: str  # relative to the storage prefix
    size: int
    etag: str
    mtime: float


def is_remote_uri(value: Any) -> bool:
    """True for ``s3://`` and ``file://`` locations."""
    return isinstance(value, str) and value.startswith(REMOTE_SCHEMES)


def display_uri(uri: str) -> str:
    """The location without its query string, for messages and paths."""
    return uri.split("?", 1)[0]


def open_storage(uri: str) -> "Storage":
    if uri.startswith("file://"):
        return FileStorage(uri)
    if uri.startswith("s3://"):
        return S3Storage(uri)
    raise ValueError(f"unsupported storage location {uri!r}: use s3://… or file://…")


def _join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


class Storage:
    """Interface shared by the backends."""

    uri: str = ""
    #: "auto" | "proxy" | "presign": how the web UI links large media.
    links: str = "proxy"
    #: ``time.time()`` after which requests fail instead of starting, retrying
    #: or waiting longer (the final upload of a closing run sets it).
    deadline: Optional[float] = None

    def cache_id(self) -> str:
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        raise NotImplementedError

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def get_file(self, key: str, dest: str) -> bool:
        raise NotImplementedError

    def head(self, key: str) -> Optional[ObjectInfo]:
        raise NotImplementedError

    def list(self, prefix: str = "") -> Iterator[ObjectInfo]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def delete_many(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.delete(key)

    def presign(self, key: str, expires: int = 3600) -> Optional[str]:
        """A time-limited direct URL for *key*, or ``None`` if unsupported."""
        return None

    def exists(self, key: str) -> bool:
        return self.head(key) is not None


def _atomic_write_from(dest: str, write: Callable[[Any], None]) -> None:
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".vt-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── file:// ─────────────────────────────────────────────────────────


class FileStorage(Storage):
    """A folder used as an object store (shared drives, NFS, tests)."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        parts = urlsplit(uri)
        path = parts.netloc + parts.path if parts.netloc not in ("", "localhost") else parts.path
        if not path:
            raise ValueError(f"file storage needs a path: {uri!r}")
        self.root = Path(path).expanduser().resolve()
        self.links = "proxy"

    def cache_id(self) -> str:
        digest = hashlib.sha1(str(self.root).encode("utf-8")).hexdigest()[:12]
        return f"file/{self.root.name or 'root'}-{digest}"

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"key escapes the storage root: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        _atomic_write_from(str(self._path(key)), lambda fh: fh.write(data))

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        def write(fh: Any) -> None:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, fh, 1024 * 1024)

        _atomic_write_from(str(self._path(key)), write)

    def get_bytes(self, key: str) -> Optional[bytes]:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def get_file(self, key: str, dest: str) -> bool:
        src = self._path(key)
        if not src.is_file():
            return False

        def write(fh: Any) -> None:
            with open(src, "rb") as s:
                shutil.copyfileobj(s, fh, 1024 * 1024)

        _atomic_write_from(dest, write)
        return True

    def head(self, key: str) -> Optional[ObjectInfo]:
        try:
            st = self._path(key).stat()
        except FileNotFoundError:
            return None
        return ObjectInfo(key, st.st_size, f"{st.st_size}-{st.st_mtime_ns}", st.st_mtime)

    def list(self, prefix: str = "") -> Iterator[ObjectInfo]:
        base = self.root
        if not base.is_dir():
            return
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if name.startswith(".vt-"):
                    continue  # an upload in progress
                full = Path(dirpath) / name
                key = full.relative_to(base).as_posix()
                if not key.startswith(prefix):
                    continue
                try:
                    st = full.stat()
                except OSError:
                    continue
                yield ObjectInfo(key, st.st_size, f"{st.st_size}-{st.st_mtime_ns}", st.st_mtime)

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        # Drop now-empty parent folders, like keys vanishing from a bucket.
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


# ── s3:// ───────────────────────────────────────────────────────────


class Credentials(NamedTuple):
    access_key: str
    secret_key: str
    token: Optional[str] = None


def _uri_encode(value: str, safe: str = "-_.~") -> str:
    return quote(value, safe=safe)


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, datestamp: str, region: str, service: str) -> bytes:
    k = _hmac(("AWS4" + secret).encode("utf-8"), datestamp)
    k = _hmac(k, region)
    k = _hmac(k, service)
    return _hmac(k, "aws4_request")


def _canonical_query(params: Sequence[Tuple[str, str]]) -> str:
    encoded = sorted((_uri_encode(k), _uri_encode(v)) for k, v in params)
    return "&".join(f"{k}={v}" for k, v in encoded)


def sign_v4(
    method: str,
    path: str,
    params: Sequence[Tuple[str, str]],
    headers: Dict[str, str],
    payload_hash: str,
    creds: Credentials,
    region: str,
    amz_date: str,
    service: str = "s3",
) -> str:
    """Return the ``Authorization`` header for an S3 request (SigV4).

    *path* is the already URI-encoded request path. *headers* must include
    ``host``, ``x-amz-date`` and ``x-amz-content-sha256``; every header
    given is signed.
    """
    norm = {k.lower().strip(): " ".join(str(v).split()) for k, v in headers.items()}
    signed = ";".join(sorted(norm))
    canonical_headers = "".join(f"{k}:{norm[k]}\n" for k in sorted(norm))
    canonical = "\n".join(
        [method, path or "/", _canonical_query(params), canonical_headers, signed, payload_hash]
    )
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode("utf-8")).hexdigest()]
    )
    sig = hmac.new(
        _signing_key(creds.secret_key, datestamp, region, service),
        to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return (
        f"AWS4-HMAC-SHA256 Credential={creds.access_key}/{scope}, "
        f"SignedHeaders={signed}, Signature={sig}"
    )


def presign_v4(
    host: str,
    path: str,
    creds: Credentials,
    region: str,
    amz_date: str,
    expires: int,
    method: str = "GET",
    service: str = "s3",
) -> List[Tuple[str, str]]:
    """Query parameters (including the signature) for a presigned URL."""
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    params: List[Tuple[str, str]] = [
        ("X-Amz-Algorithm", "AWS4-HMAC-SHA256"),
        ("X-Amz-Credential", f"{creds.access_key}/{scope}"),
        ("X-Amz-Date", amz_date),
        ("X-Amz-Expires", str(int(expires))),
        ("X-Amz-SignedHeaders", "host"),
    ]
    if creds.token:
        params.append(("X-Amz-Security-Token", creds.token))
    canonical = "\n".join(
        [method, path or "/", _canonical_query(params), f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"]
    )
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode("utf-8")).hexdigest()]
    )
    sig = hmac.new(
        _signing_key(creds.secret_key, datestamp, region, service),
        to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return params + [("X-Amz-Signature", sig)]


def _aws_files_profile(profile: str) -> Dict[str, str]:
    """Merge ``~/.aws/config`` and ``~/.aws/credentials`` entries for *profile*."""
    out: Dict[str, str] = {}
    cfg_path = os.environ.get("AWS_CONFIG_FILE") or str(Path.home() / ".aws" / "config")
    cred_path = os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or str(
        Path.home() / ".aws" / "credentials"
    )
    for path, section in (
        (cfg_path, "default" if profile == "default" else f"profile {profile}"),
        (cred_path, profile),
    ):
        parser = configparser.RawConfigParser()
        try:
            if not parser.read(path):
                continue
        except configparser.Error:
            continue
        if parser.has_section(section):
            out.update({k.lower(): v.strip() for k, v in parser.items(section)})
    return out


def _vibetrack_s3_config() -> Dict[str, Any]:
    try:
        from .config import load_config

        section = load_config().get("s3")
    except Exception:
        return {}
    return section if isinstance(section, dict) else {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class S3Storage(Storage):
    """AWS S3 or any S3-compatible service, over ``urllib`` with SigV4."""

    def __init__(self, uri: str, timeout: float = 60.0, retries: int = 4) -> None:
        self.uri = uri
        parts = urlsplit(uri)
        if not parts.netloc:
            raise ValueError(f"s3 location needs a bucket: {uri!r}")
        self.bucket = parts.netloc
        self.prefix = parts.path.strip("/")
        query = {k.lower(): v for k, v in parse_qsl(parts.query)}
        cfg = _vibetrack_s3_config()
        self.timeout = timeout
        self.retries = retries

        def setting(name: str, env: str, cfg_key: str) -> Optional[str]:
            for value in (query.get(name), os.environ.get(env), cfg.get(cfg_key)):
                if value not in (None, ""):
                    return str(value)
            return None

        profile = (
            setting("profile", "VIBETRACK_S3_PROFILE", "profile")
            or os.environ.get("AWS_PROFILE")
            or "default"
        )
        aws_files = _aws_files_profile(profile)

        endpoint = (
            setting("endpoint", "VIBETRACK_S3_ENDPOINT_URL", "endpoint_url")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL")
            or aws_files.get("endpoint_url")
        )
        self.region = (
            setting("region", "VIBETRACK_S3_REGION", "region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or aws_files.get("region")
            or "us-east-1"
        )
        if endpoint:
            if "://" not in endpoint:
                endpoint = "https://" + endpoint
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"https://s3.{self.region}.amazonaws.com"
        style = (setting("addressing_style", "VIBETRACK_S3_ADDRESSING_STYLE", "addressing_style") or "").lower()
        if style not in ("path", "virtual"):
            # Self-hosted stores (MinIO, Ceph) generally need path-style;
            # AWS prefers virtual-hosted. Dotted bucket names break TLS
            # wildcards, so they stay path-style too.
            style = "path" if endpoint or "." in self.bucket else "virtual"
        self.addressing_style = style
        links = (setting("links", "VIBETRACK_S3_LINKS", "links") or "auto").lower()
        self.links = links if links in _LINK_MODES else "auto"

        verify = setting("verify", "VIBETRACK_S3_VERIFY", "verify")
        self._ssl: Optional[ssl.SSLContext] = None
        if verify is not None and verify.strip().lower() in ("0", "false", "no", "off"):
            self._ssl = ssl._create_unverified_context()
        elif verify is not None and not _truthy(verify):
            self._ssl = ssl.create_default_context(cafile=verify)

        key = (
            setting("", "VIBETRACK_S3_ACCESS_KEY_ID", "access_key_id")
            or os.environ.get("AWS_ACCESS_KEY_ID")
            or aws_files.get("aws_access_key_id")
        )
        secret = (
            setting("", "VIBETRACK_S3_SECRET_ACCESS_KEY", "secret_access_key")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
            or aws_files.get("aws_secret_access_key")
        )
        token = (
            setting("", "VIBETRACK_S3_SESSION_TOKEN", "session_token")
            or os.environ.get("AWS_SESSION_TOKEN")
            or aws_files.get("aws_session_token")
        )
        # No credentials → unsigned requests (public buckets).
        self.creds: Optional[Credentials] = (
            Credentials(key, secret, token) if key and secret else None
        )
        self._list_v1 = False

    # ── URL building ────────────────────────────────────────────

    def cache_id(self) -> str:
        host = urlsplit(self.endpoint).netloc.replace(":", "_") or "s3"
        return _join("s3", host, self.bucket, self.prefix)

    def _full_key(self, key: str) -> str:
        return _join(self.prefix, key) if key else (self.prefix + "/" if self.prefix else "")

    def _host_and_path(self, full_key: str) -> Tuple[str, str, str]:
        ep = urlsplit(self.endpoint)
        encoded_key = _uri_encode(full_key, safe="-_.~/")
        if self.addressing_style == "virtual":
            host = f"{self.bucket}.{ep.netloc}"
            path = (ep.path.rstrip("/") + "/" + encoded_key) if full_key else (ep.path.rstrip("/") + "/")
        else:
            host = ep.netloc
            base = ep.path.rstrip("/") + "/" + _uri_encode(self.bucket, safe="-_.~")
            path = base + ("/" + encoded_key if full_key else "/")
        return ep.scheme or "https", host, path

    # ── transport ───────────────────────────────────────────────

    def _request(
        self,
        method: str,
        key: str,
        params: Sequence[Tuple[str, str]] = (),
        body: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
        ok_404: bool = False,
        stream_to: Optional[str] = None,
        raw_key: bool = False,
    ) -> Tuple[int, Dict[str, str], bytes]:
        full_key = key if raw_key else self._full_key(key)
        scheme, host, path = self._host_and_path(full_key)
        payload_hash = hashlib.sha256(body).hexdigest() if body else _EMPTY_SHA256
        query = _canonical_query(params)
        url = f"{scheme}://{host}{path}" + (f"?{query}" if query else "")
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            deadline = self.deadline
            if attempt:
                pause = min(8.0, 0.5 * (2 ** (attempt - 1)))
                if deadline is not None and time.time() + pause >= deadline:
                    break
                time.sleep(pause)
            timeout = self.timeout
            if deadline is not None:
                left = deadline - time.time()
                if left <= 0:
                    last_exc = last_exc or TimeoutError("deadline passed")
                    break
                timeout = min(timeout, left)
            now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            hdrs = {"host": host, "x-amz-date": now, "x-amz-content-sha256": payload_hash}
            if self.creds and self.creds.token:
                hdrs["x-amz-security-token"] = self.creds.token
            send = dict(hdrs)
            if self.creds is not None:
                send["Authorization"] = sign_v4(
                    method, path, params, hdrs, payload_hash, self.creds, self.region, now
                )
            send.update(headers or {})
            if body or method in ("PUT", "POST"):
                send["Content-Length"] = str(len(body))
                # Without this urllib labels the body as a submitted form,
                # and some servers then parse (and swallow) it.
                if not any(k.lower() == "content-type" for k in send):
                    send["Content-Type"] = "application/octet-stream"
            req = _urlreq.Request(url, data=body if body or method in ("PUT", "POST") else None, method=method)
            for k, v in send.items():
                req.add_header(k, v)
            try:
                with _urlreq.urlopen(req, timeout=timeout, context=self._ssl) as resp:
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    if stream_to is not None:
                        _atomic_write_from(stream_to, lambda fh: shutil.copyfileobj(resp, fh, 1024 * 1024))
                        return resp.status, resp_headers, b""
                    return resp.status, resp_headers, resp.read()
            except _urlerr.HTTPError as exc:
                data = b""
                try:
                    data = exc.read()
                except Exception:
                    pass
                if exc.code == 404 and ok_404:
                    return 404, {}, data
                if exc.code in _RETRY_STATUS and attempt < self.retries:
                    last_exc = exc
                    continue
                raise StorageError(
                    f"{method} {display_uri(self.uri)}/{key}: HTTP {exc.code} {_s3_error_message(data)}"
                ) from None
            except (_urlerr.URLError, OSError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    continue
        raise StorageError(f"{method} {display_uri(self.uri)}/{key} failed: {last_exc}")

    # ── object API ──────────────────────────────────────────────

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        headers = {"Content-Type": content_type} if content_type else {}
        self._request("PUT", key, body=data, headers=headers)

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        size = os.path.getsize(path)
        if size < MULTIPART_THRESHOLD:
            with open(path, "rb") as fh:
                self.put_bytes(key, fh.read(), content_type)
            return
        self._put_multipart(key, path, size, content_type)

    def _put_multipart(self, key: str, path: str, size: int, content_type: Optional[str]) -> None:
        part_size = max(_MIN_PART, -(-size // _MAX_PARTS))
        headers = {"Content-Type": content_type} if content_type else {}
        _, _, body = self._request("POST", key, params=[("uploads", "")], headers=headers)
        upload_id = _xml_find_text(body, "UploadId")
        if not upload_id:
            raise StorageError(f"multipart upload of {key}: no UploadId in response")
        etags: List[Tuple[int, str]] = []
        try:
            with open(path, "rb") as fh:
                number = 1
                while True:
                    chunk = fh.read(part_size)
                    if not chunk:
                        break
                    _, resp_headers, _ = self._request(
                        "PUT",
                        key,
                        params=[("partNumber", str(number)), ("uploadId", upload_id)],
                        body=chunk,
                    )
                    etags.append((number, resp_headers.get("etag", "")))
                    number += 1
            parts_xml = "".join(
                f"<Part><PartNumber>{n}</PartNumber><ETag>{_xml_escape(e)}</ETag></Part>"
                for n, e in etags
            )
            done = f"<CompleteMultipartUpload>{parts_xml}</CompleteMultipartUpload>".encode()
            _, _, resp = self._request(
                "POST", key, params=[("uploadId", upload_id)], body=done,
                headers={"Content-Type": "application/xml"},
            )
            # S3 can answer 200 with an <Error> body for a failed completion.
            if b"<Error>" in resp:
                raise StorageError(f"multipart upload of {key}: {_s3_error_message(resp)}")
        except BaseException:
            try:
                self._request("DELETE", key, params=[("uploadId", upload_id)], ok_404=True)
            except Exception:
                pass
            raise

    def get_bytes(self, key: str) -> Optional[bytes]:
        status, _, body = self._request("GET", key, ok_404=True)
        return None if status == 404 else body

    def get_file(self, key: str, dest: str) -> bool:
        status, _, _ = self._request("GET", key, ok_404=True, stream_to=dest)
        return status != 404

    def head(self, key: str) -> Optional[ObjectInfo]:
        status, headers, _ = self._request("HEAD", key, ok_404=True)
        if status == 404:
            return None
        return ObjectInfo(
            key,
            int(headers.get("content-length") or 0),
            headers.get("etag", "").strip('"'),
            _parse_http_date(headers.get("last-modified")),
        )

    def list(self, prefix: str = "") -> Iterator[ObjectInfo]:
        root = self.prefix + "/" if self.prefix else ""
        full_prefix = root + prefix
        token: Optional[str] = None
        marker: Optional[str] = None
        while True:
            if self._list_v1:
                params = [("prefix", full_prefix)] + ([("marker", marker)] if marker else [])
            else:
                params = [("list-type", "2"), ("prefix", full_prefix)]
                if token:
                    params.append(("continuation-token", token))
            try:
                _, _, body = self._request("GET", "", params=params, raw_key=True)
            except StorageError as exc:
                if not self._list_v1 and ("HTTP 400" in str(exc) or "HTTP 501" in str(exc)):
                    self._list_v1 = True  # store without ListObjectsV2
                    continue
                raise
            entries, truncated, next_token, next_marker = _parse_list(body)
            for full_key, size, etag, mtime in entries:
                if not full_key.startswith(root):
                    continue
                yield ObjectInfo(full_key[len(root):], size, etag, mtime)
            if not truncated:
                return
            if self._list_v1:
                marker = next_marker or (entries[-1][0] if entries else None)
                if not marker:
                    return
            else:
                token = next_token
                if not token:
                    return

    def delete(self, key: str) -> None:
        self._request("DELETE", key, ok_404=True)

    def presign(self, key: str, expires: int = 3600) -> Optional[str]:
        if self.creds is None:
            scheme, host, path = self._host_and_path(self._full_key(key))
            return f"{scheme}://{host}{path}"
        scheme, host, path = self._host_and_path(self._full_key(key))
        now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        params = presign_v4(host, path, self.creds, self.region, now, expires)
        return f"{scheme}://{host}{path}?{_canonical_query(params)}"


# ── XML helpers ─────────────────────────────────────────────────────


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_find_text(body: bytes, name: str) -> Optional[str]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    for el in root.iter():
        if _local(el.tag) == name:
            return el.text
    return None


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _s3_error_message(body: bytes) -> str:
    code = _xml_find_text(body, "Code") if body else None
    msg = _xml_find_text(body, "Message") if body else None
    return " ".join(p for p in (code, msg) if p)


def _parse_iso(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return _dt.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=_dt.timezone.utc
        ).timestamp()
    except ValueError:
        return 0.0


def _parse_http_date(value: Optional[str]) -> float:
    if not value:
        return 0.0
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _parse_list(
    body: bytes,
) -> Tuple[List[Tuple[str, int, str, float]], bool, Optional[str], Optional[str]]:
    root = ET.fromstring(body)
    entries: List[Tuple[str, int, str, float]] = []
    truncated = False
    next_token = None
    next_marker = None
    for child in root:
        name = _local(child.tag)
        if name == "Contents":
            fields = {_local(c.tag): (c.text or "") for c in child}
            entries.append(
                (
                    fields.get("Key", ""),
                    int(fields.get("Size") or 0),
                    fields.get("ETag", "").strip('"'),
                    _parse_iso(fields.get("LastModified")),
                )
            )
        elif name == "IsTruncated":
            truncated = (child.text or "").strip().lower() == "true"
        elif name == "NextContinuationToken":
            next_token = child.text
        elif name == "NextMarker":
            next_marker = child.text
    return entries, truncated, next_token, next_marker
