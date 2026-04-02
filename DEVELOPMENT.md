# Vibetrack Development Roadmap

## A. Ensure All Artifacts Display Correctly in Web UI

**Status**: All 8 tabs implemented (Scalars, Images, Audio, Video, Text, Histograms, Artifacts, System).

| Tab | Rendering | Multi-experiment | Fullscreen | Notes |
|-----|-----------|-----------------|------------|-------|
| Scalars | Chart.js line charts | Experiment pills | Yes | EMA/MA/Gaussian smoothing |
| Images | Lazy-loaded grid | Per-experiment galleries | Yes | |
| Audio | HTML5 `<audio>` | Per-experiment list | No | |
| Video | HTML5 `<video>` | Per-experiment grid | Yes | |
| Text | Monospace `<pre>` | Per-experiment list | Yes | HTML-escaped |
| Histograms | Chart.js bar charts | Stacked bars | Yes | Latest step only |
| Artifacts | Download table | Table rows | N/A | Filename, size, MIME |
| System | Chart.js line charts | Experiment pills | Yes | `system/*`, `gpu/*` tags |

### Action items

- [ ] Write integration test that logs every artifact type and verifies the web UI serves them
- [ ] Verify lazy-loading of images with 100+ entries
- [ ] Test artifact download links across nested logdir structures
- [ ] Verify media path resolution handles edge cases (spaces in paths, unicode tags)
- [ ] Test HTML-escaped text entries with special characters
- [ ] Confirm histogram bar charts with varying bin counts

**Key files**: `vibetrack/viewers/web.py`, `vibetrack/media.py`, `vibetrack/reader.py`

---

## B. Image-to-Image Comparison

**Status**: Not implemented. `compare.py` only supports scalars and hparams. Web UI shows images per-experiment in separate galleries with no side-by-side view.

### Action items

- [ ] Add `compare_images(experiments, tag)` to `vibetrack/compare.py` — return images grouped by step across experiments
- [ ] Add comparison view in web UI Images tab: step-aligned grid with one column per experiment
- [ ] Add slider/step selector to scrub through steps while comparing
- [ ] Write tests in `tests/test_compare.py`

**Key files**: `vibetrack/compare.py`, `vibetrack/viewers/web.py`, `tests/test_compare.py`

---

## C. Test Other Viewer Outputs

**Current viewer feature matrix**:

| Viewer | Scalars | Images | Audio | Video | Artifacts | Histograms | Text | Hparams | System |
|--------|---------|--------|-------|-------|-----------|------------|------|---------|--------|
| Web | Full | Gallery | Players | Players | Table | Bar charts | Monospace | Display | Charts |
| Console | Sparkline | Count | Count | Count | Count | -- | -- | -- | -- |
| Gradio | Plotly charts | Gallery | Players | Players | -- | -- | -- | -- | -- |
| Telegram | Chart PNG | -- | -- | -- | -- | -- | -- | -- | -- |
| MCP | JSON | Metadata | Metadata | Metadata | Metadata | JSON | JSON | JSON | -- |

### Action items

- [ ] Add integration tests for each viewer's `show()` in `tests/test_outputs.py`
- [ ] **Gradio**: add Artifacts tab, Text tab, Histograms tab, System metrics tab
- [ ] **Console**: add text and histogram summary output
- [ ] **Telegram**: add image sending support (send logged images as photos)
- [ ] **MCP**: add resource URIs for artifacts and system metrics
- [ ] Document updated viewer feature matrix

**Key files**: `vibetrack/viewers/console.py`, `vibetrack/viewers/gradio_ui.py`, `vibetrack/viewers/telegram.py`, `vibetrack/viewers/mcp.py`, `tests/test_outputs.py`

---

## D. Check System Metrics

**Status**: Collection works — disk (always), CPU/memory (with fallbacks), GPU (via `nvidia-smi`). Displayed in web UI System tab. 21 tests passing.

### Collected metrics

| Category | Tags | Source |
|----------|------|--------|
| Disk | `system/disk_total_gb`, `system/disk_used_gb`, `system/disk_free_gb`, `system/disk_used_percent` | `shutil.disk_usage()` |
| Memory | `system/memory_total_gb`, `system/memory_used_gb`, `system/memory_available_gb`, `system/memory_used_percent` | psutil / `/proc/meminfo` / `vm_stat` |
| CPU | `system/cpu_percent`, `system/cpu_count` or `system/cpu_load_*` | psutil / `os.getloadavg()` |
| GPU | `gpu/{idx}/utilization_percent`, `gpu/{idx}/memory_*`, `gpu/{idx}/temperature_c` | `nvidia-smi` |

### Action items

- [ ] Verify System tab appears/hides correctly based on data presence
- [ ] Test GPU metrics display with multiple GPUs (multi-GPU mock)
- [ ] Add system metrics to Gradio viewer (separate tab)
- [ ] Add system metrics to MCP viewer (tools + resources)
- [ ] Test `system_metrics_interval` edge cases (very fast intervals, zero)
- [ ] Verify background thread cleanup on `writer.close()` and process exit

**Key files**: `vibetrack/sysmetrics.py`, `vibetrack/viewers/web.py`, `vibetrack/viewers/gradio_ui.py`, `vibetrack/viewers/mcp.py`, `tests/test_sysmetrics.py`

---

## E. Display Model Artifacts

**Status**: Artifacts tab exists in web UI with download table (filename, size, MIME type). No inline preview or model-specific display.

### Action items

- [ ] Add inline preview for common artifact types:
  - JSON/YAML: syntax-highlighted collapsible view
  - Text/CSV: tabular preview
  - Images stored as artifacts: thumbnail preview
  - PDF: embedded viewer or first-page preview
- [ ] Add model checkpoint metadata display (framework, param count, file size)
- [ ] Add artifact comparison across experiments (which experiments saved which artifacts, size diff)
- [ ] Improve `save_artifact()` to optionally capture additional metadata (e.g. model architecture summary)
- [ ] Add artifact support to Gradio viewer

**Key files**: `vibetrack/viewers/web.py`, `vibetrack/media.py`, `vibetrack/types.py`, `vibetrack/viewers/gradio_ui.py`, `tests/test_media.py`

---

## Verification

After each section:

```bash
pytest tests/                                          # 158 existing tests must stay green
vibetrack --logdir=<test_dir> --viewer=web              # manual web UI check
vibetrack --logdir=<test_dir> --viewer=console           # console output check
vibetrack --logdir=<test_dir> --viewer=gradio            # gradio dashboard check
```

Test with a populated logdir containing all artifact types (scalars, images, audio, video, text, histograms, artifacts, system metrics).
