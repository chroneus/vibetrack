# Storing runs on S3-compatible storage

Upload runs to a bucket while they train. Open them from any machine with
`vibetrack s3://bucket/prefix`.

Any S3-compatible store works: MinIO, Ceph RGW, Cloudflare R2, Backblaze B2,
Wasabi, Yandex Object Storage, GCS (interoperability mode) and AWS S3. A plain
folder (`file:///mnt/shared/runs`, e.g. on NFS) works the same way.

There is nothing extra to install: requests are signed (AWS Signature V4) and
sent with the Python standard library.

## Quick start

```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export VIBETRACK_SYNC="s3://ml-runs/team?endpoint=https://minio.lab:9000"
python train.py                     # unchanged code: local SQLite as always, plus uploads
```

On your laptop (same credentials):

```bash
vibetrack "s3://ml-runs/team?endpoint=https://minio.lab:9000"            # dashboard
vibetrack ls --project-folder "s3://ml-runs/team?endpoint=https://minio.lab:9000"
```

`ls`, `show`, `compare`, `open` and MCP all accept the same location.

In code, instead of the environment variable:

```python
writer = SummaryWriter("runs/exp1", sync="s3://ml-runs/team?endpoint=https://minio.lab:9000")
```

`sync=""` turns syncing off for one writer even when `VIBETRACK_SYNC` is set.

## What gets uploaded, and when

Training still writes locally first. A background thread then uploads:

- **Every 60 s** (`VIBETRACK_SYNC_INTERVAL`): only the rows added since the
  last pass, as one small gzip chunk, plus media files logged since then.
  A pass with nothing new sends nothing.
- **On close, failure or SIGTERM:** a final snapshot that includes the run's
  outcome (`finished`, `failed`, `interrupted`). On spot or preemptible
  machines, SIGTERM records the run as `interrupted` and uploads it before the
  process exits. A hard kill loses at most one interval. The final upload
  waits up to `VIBETRACK_SYNC_TIMEOUT` seconds (default 120).
- A run that fails, is interrupted or logs nothing within its first 60 s
  (`VIBETRACK_DISCARD_FAILED_SECS`) is not kept: nothing more is uploaded,
  and what was already uploaded is removed (behind a delete marker).
- Every 50 chunks, and at close, the chunks are merged into one snapshot.

Upload errors never stop training. They print one warning and are retried
with backoff.

## Bucket layout

```
<prefix>/_runs/<uid>/snap.db.gz          one-run SQLite snapshot (written only by that run)
<prefix>/_runs/<uid>/d/<gen>-<seq>.json.gz  rows added after the snapshot
<prefix>/_runs/<uid>/edit.json           rename/move made in a dashboard
<prefix>/_runs/<uid>/deleted             delete marker
<prefix>/<project>/<run>-<uid8>/media/<tag>/<step>.<ext>
```

Each run writes only its own objects, so any number of machines can share one
bucket. `uid` stays the same when a run is resumed on the same machine.
A snapshot is a normal vibetrack database: gunzip it to
`some_dir/vibetrack.db` and run `vibetrack some_dir`.

## Viewing

`vibetrack s3://…` keeps a local mirror in `~/.vibetrack/cache/`
(`VIBETRACK_CACHE_DIR` moves it):

- It lists `_runs/` once every 10 s (`VIBETRACK_SYNC_REFRESH`), downloads
  only new chunks and appends their rows.
- The dashboard, its MCP server and CLI commands can use the same mirror at
  the same time; each chunk is applied exactly once.
- Images and small files are downloaded the first time they are viewed, then
  served from the cache.
- Videos and files of 20 MB or more are not downloaded through the
  dashboard. It redirects the browser to a short-lived presigned URL, so
  playback starts at once and seeking reads only the needed bytes. Poster
  frames are read from that URL by ffmpeg.
- If the browser cannot reach the endpoint (for example MinIO on a private
  cluster network), set `links=proxy` and everything goes through the
  dashboard's cache. `links=presign` also streams images and audio directly.

`vibetrack show` downloads the latest images of the run it shows, so the
listed paths can be opened.

### Edits in the dashboard

- **Rename / move to another project:** writes `edit.json`. Other viewers pick
  it up, and it survives a still-running writer uploading under the old name.
- **Delete run / project:** writes the delete marker, then removes the run's
  snapshot, chunks and media. A writer still running for that run notices the
  marker and stops uploading.
- Moving log directories is not available for synced storage.
- Remote ingest (`.to("remote")`) cannot target a dashboard opened on synced
  storage. Set `VIBETRACK_SYNC` on the training machine instead.

## Existing runs, offline copies, archiving

```bash
vibetrack push s3://ml-runs/team                     # all runs of the central DB
vibetrack push ./my_project s3://ml-runs/team        # a project folder
vibetrack push ./my_project s3://ml-runs/team -p cv run1 run2
vibetrack push ./my_project s3://ml-runs/team --remove-local   # archive: verify, then delete locally
vibetrack pull s3://ml-runs/team ./offline-copy      # everything, with all media
vibetrack ./offline-copy
```

Pushing again updates the same objects. Runs deleted from the bucket are
skipped. `--remove-local` deletes a run locally only after its snapshot and
every media file are confirmed in the bucket (sizes compared).

## Settings

Location settings can go in the URI query, so one string holds them:

| Query key          | Environment                      | `config.json` `"s3"` key | Default |
|--------------------|----------------------------------|--------------------------|---------|
| `endpoint`         | `VIBETRACK_S3_ENDPOINT_URL`      | `endpoint_url`           | AWS for the region |
| `region`           | `VIBETRACK_S3_REGION`            | `region`                 | `us-east-1` |
| `addressing_style` | `VIBETRACK_S3_ADDRESSING_STYLE`  | `addressing_style`       | `path` with a custom endpoint, else `virtual` |
| `verify`           | `VIBETRACK_S3_VERIFY`            | `verify`                 | on; `false` or a CA bundle path |
| `links`            | `VIBETRACK_S3_LINKS`             | `links`                  | `auto` (`proxy`, `presign`) |
| `profile`          | `VIBETRACK_S3_PROFILE`           | `profile`                | `AWS_PROFILE` or `default` |

The first match wins: URI query, then environment, then
`~/.vibetrack/config.json`, then the standard `AWS_ENDPOINT_URL_S3` /
`AWS_ENDPOINT_URL`, `AWS_REGION` and the `~/.aws/config` profile.

**Credentials** never go in the URI. vibetrack looks for them in this order:

1. `VIBETRACK_S3_ACCESS_KEY_ID` / `VIBETRACK_S3_SECRET_ACCESS_KEY` /
   `VIBETRACK_S3_SESSION_TOKEN`
2. `access_key_id` / `secret_access_key` in the `"s3"` section of
   `config.json`. The web API never returns or overwrites that section.
3. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
4. a profile in `~/.aws/credentials`

Without credentials, requests are sent unsigned (public buckets). Instance
roles and SSO logins are not supported; export keys instead.

### Provider examples

Quote the location in the shell: `?` and `&` are special characters there.

```bash
# MinIO / Ceph RGW (self-hosted, path-style)
"s3://runs/team?endpoint=https://minio.lab:9000"
# Cloudflare R2
"s3://runs/team?endpoint=https://<account>.r2.cloudflarestorage.com&region=auto"
# Backblaze B2
"s3://runs/team?endpoint=https://s3.us-west-004.backblazeb2.com&region=us-west-004"
# Yandex Object Storage
"s3://runs/team?endpoint=https://storage.yandexcloud.net&region=ru-central1"
# GCS interoperability (HMAC keys)
"s3://runs/team?endpoint=https://storage.googleapis.com&region=auto"
# AWS S3
"s3://runs/team?region=eu-west-1"
# Self-signed certificate
"s3://runs/team?endpoint=https://minio.local:9000&verify=/etc/ssl/minio-ca.pem"
```

Only basic S3 calls are used: Put, Get (with `Range`), Head and Delete
Object; ListObjectsV2 (falling back to ListObjects); multipart upload for files
of 64 MB or more; and presigned GET URLs.
