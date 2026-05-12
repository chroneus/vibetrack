# Viewers and Destinations

vibetrack always writes experiment data to SQLite and local media files. Viewers
and destinations are additional outputs for displaying or forwarding events.

## Launch the dashboard

```bash
vibetrack
# -> Web UI on http://0.0.0.0:6116
# -> MCP is also mounted when installed with vibetrack[all] on Python 3.10+
```

```bash
vibetrack my_project/
vibetrack --viewer console
vibetrack --viewer gradio
vibetrack --viewer mcp
```

## Event Dispatch

Each `add_*` and `log()` call returns an event handle, so you can forward a
single event with `.to(name, **creds)`. Use `writer.to(name, every=...)` to
attach a persistent destination for future events. Dispatch is best-effort:
adapter failures are caught and never interrupt the training loop.

```python
from vibetrack import SummaryWriter

writer = SummaryWriter("runs/exp1", project_folder="my_project")

writer = (
    writer
    .to("console")                          # every event -> stdout
    .to("telegram", every=100)              # every 100 events
    .to("slack", every="15m")               # time-based digest
)

for step in range(1000):
    writer.add_scalar("loss", 1.0 / (step + 1), step)

# One-shot dispatch for just this event.
writer.add_image("samples", "out.png", step=999).to("telegram")

writer.close()
```

`every=` accepts `None` for every event, an `int` for every N events, or a
duration string such as `"5s"`, `"15m"`, `"1h"`, or `"2d"`.

Module-level equivalent:

```python
import vibetrack

vibetrack.init(project="cifar10", name="resnet18", to=["console", "telegram"])
```

For destinations that need arguments:

```python
vibetrack.init(
    project="cifar10",
    name="resnet18",
    to=[
        {
            "name": "remote",
            "url": "http://server:8080",
            "token": "devtoken",
            "every": "10m",
        }
    ],
)
```

## Built-In Viewers

Viewers are auto-discovered from `vibetrack/viewers/`. Any `.py` file except
`__init__.py`, `base.py`, and `event.py` becomes available via `--viewer=name`
when it exposes a `BaseOutput` subclass. Filename mapping strips a trailing
`_ui` suffix.

| Name | Purpose | Notes |
| --- | --- | --- |
| `web` | Browser dashboard | Default viewer; included in base install |
| `console` | Terminal summaries | No extra dependencies |
| `gradio` | Gradio dashboard | Optional Gradio dependency |
| `telegram` | Telegram notifications | Scalars, text, media, summaries |
| `slack` | Slack notifications | Webhook text or bot-token media uploads |
| `jupyter` | Notebook display | Available when the Jupyter viewer module is installed |
| `mcp` | LLM/agent integration | See [MCP.md](MCP.md) |
| `remote` | Forward events to another vibetrack server | Best-effort HTTP dispatch |
| custom | User-defined viewer | Add a viewer module and call `.to("name")` |

The built-in web dashboard and SQLite store are always active; `.to(...)` adds
additional destinations.

## Telegram and Slack Credentials

Telegram needs:

- `VIBETRACK_TELEGRAM_TOKEN` and `VIBETRACK_TELEGRAM_CHAT_ID`
- or explicit `token=` and `chat_id=`

Slack needs either:

- `SLACK_WEBHOOK_URL` or explicit `webhook=` for text-only webhook posts
- or `SLACK_BOT_TOKEN` and `SLACK_CHANNEL` for media uploads

Credential precedence is:

```text
explicit kwargs > environment variables > config credentials
```

Config-backed credentials are available, but they are not read by default
because `~/.vibetrack/config.json` is plain JSON. Opt in explicitly:

```python
writer.to("telegram", use_config_credentials=True)
writer.to("slack", use_config_credentials=True)
```

```json
{
  "telegram": {"token": "BOT_TOKEN", "chat_id": "CHAT_ID"},
  "slack": {
    "webhook": "https://hooks.slack.com/services/...",
    "bot_token": "xoxb-...",
    "channel": "C0123ABCD"
  }
}
```

## Remote Event Forwarding

Run `vibetrack --listen HOST:PORT` on a peer machine to expose `/log`,
`/media`, and `/hparams` ingest endpoints. Then point a training run at it with
`writer.to("remote", url=..., token=...)`.

```bash
# Server side
vibetrack --listen 0.0.0.0:8080 --token devtoken --project-folder /srv/runs
```

```python
# Client side
import vibetrack

writer = vibetrack.init(project="cifar10", name="resnet18")
writer.to(
    "remote",
    url="http://server:8080",
    token="devtoken",
    every="10m",
)

vibetrack.log({"loss": 0.5})
```

All event kinds round-trip: scalars, texts, histograms, images, audio, video,
artifacts, and hparams. Local SQLite stays the source of truth; if the remote
server is unreachable, the adapter logs one warning and keeps training going.

## Direct HTTP Ingest

The built-in web server also accepts metrics from remote machines:

```bash
vibetrack my_project/ --token mysecret
# -> Ingest at http://host:6116/{project}/listen/log
```

```python
import requests

requests.post(
    "http://server:6116/my_project/listen/log",
    json={
        "experiment": "remote_run",
        "step": 42,
        "scalars": {"loss": 0.3, "acc": 0.91},
        "texts": {"note": "checkpoint saved"},
    },
    headers={"Authorization": "Bearer mysecret"},
)
```

Upload media:

```python
requests.post(
    "http://server:6116/my_project/listen/media",
    data={
        "experiment": "remote_run",
        "tag": "sample",
        "step": "0",
        "type": "image",
    },
    files={"file": open("output.png", "rb")},
    headers={"Authorization": "Bearer mysecret"},
)
```
