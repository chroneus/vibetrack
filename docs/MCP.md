# MCP Server

vibetrack exposes experiment data through MCP (Model Context Protocol), so LLM
apps and agents can query runs, metrics, media metadata, hparams, and compact
analysis summaries.

MCP support is available when installed with `vibetrack[all]` on Python 3.10+.

```bash
pip install vibetrack[all]
```

## Web Dashboard Mount

When MCP dependencies are installed, the web dashboard mounts MCP at:

```text
/vibetrack_mcp
```

Start the dashboard:

```bash
vibetrack my_project/
```

## Standalone MCP Server

```bash
vibetrack --viewer mcp --project-folder my_project/
```

The default streamable HTTP endpoint is:

```text
http://127.0.0.1:6116/mcp
```

SSE transport is also available:

```bash
vibetrack --viewer mcp --mcp-transport=sse --project-folder my_project/
```

## Tools

Available MCP tools:

| Tool | Purpose |
| --- | --- |
| `list_experiments` | List runs in the active project database |
| `get_experiment_tags` | Return available scalar/text/media/artifact tags |
| `get_scalars` | Return raw scalar time series for one tag |
| `analyze_scalar` | Summarize extrema, trend, jumps, plateau, and best point |
| `compare_scalar` | Rank experiments by one scalar metric |
| `find_metric_events` | Return compact graph events such as min/max/jumps/plateau |
| `get_texts` | Return logged text entries |
| `get_images` | Return logged image entries and paths |
| `compare_image_lpips` | Compare logged images with LPIPS when available plus pixel metrics |
| `get_audio` | Return logged audio entries |
| `get_hparams` | Return run hyperparameters |
| `get_histograms` | Return histogram payloads |
| `summary` | Return final scalar values per experiment |
| `compare_hparams_tool` | Compare hyperparameters side by side |
| `run_report` | Return a human-readable end-of-run digest |

The analysis tools are designed for LLMs. They avoid forcing the model to fetch
and parse large raw arrays when a compact answer is enough.

Example `analyze_scalar` result:

```json
{
  "experiment": "small_balanced",
  "tag": "loss/val",
  "objective": "min",
  "count": 14,
  "first": {"step": 0, "value": 0.92},
  "last": {"step": 260, "value": 0.196},
  "min": {"step": 260, "value": 0.196},
  "max": {"step": 0, "value": 0.92},
  "best": {"step": 260, "value": 0.196},
  "trend": {"direction": "decreasing"},
  "plateau": {"start_step": 180},
  "events": [
    {"type": "minimum", "step": 260, "value": 0.196},
    {"type": "largest_drop", "from_step": 0, "to_step": 20, "delta": -0.31}
  ]
}
```

## Image Comparison

`compare_image_lpips` compares images logged under an image tag:

```text
compare_image_lpips(
  reference_experiment="baseline",
  candidate_experiment="candidate",
  tag="samples",
  step=10
)
```

It uses real LPIPS when `lpips` and `torch` are installed. It also returns
fallback pixel metrics (`mse`, `rmse`, `mae`, `psnr_db`) when Pillow and numpy
are available.

```bash
pip install lpips torch Pillow numpy
```

If LPIPS dependencies are missing, the tool returns `lpips.available=false`
with an install hint instead of failing the MCP call.

## Resources

MCP resources include:

```text
vibetrack://experiments
vibetrack://experiments/{name}
vibetrack://experiments/{name}/scalars/{tag}
vibetrack://experiments/{name}/texts/{tag}
vibetrack://experiments/{name}/images/{tag}
vibetrack://experiments/{name}/hparams
```

## LLM Demo

A complete LLM demo is included in `examples/arch_search_mcp.py`. It trains
several small architecture candidates, logs their metrics and hparams to
vibetrack, starts the MCP server, exposes the MCP tools to an
OpenAI-compatible tool-calling chat endpoint, and asks the LLM to choose the
best run from the recorded evidence.

```bash
pip install -e ".[all]" numpy httpx
ollama pull llama3.1

LLM_BASE_URL=http://127.0.0.1:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=llama3.1 \
python examples/arch_search_mcp.py
```
