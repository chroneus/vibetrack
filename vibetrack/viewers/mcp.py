"""MCP server — expose experiment data as MCP tools and resources."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..compare import compare_hparams, summary_table
from .base import BaseOutput
from .console import ConsoleOutput

_STREAMABLE_HTTP_LOGGER = "mcp.server.streamable_http_manager"
_MAX_IMAGE_PIXELS = 25_000_000


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _json_response(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=_json_default)


def _point(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "step": row.get("step"),
        "value": float(row.get("value", 0.0)),
    }
    if row.get("wall_time") is not None:
        out["wall_time"] = row.get("wall_time")
    return out


def _finite_scalar_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            item = dict(row)
            item["value"] = value
            clean.append(item)
    return clean


def _percent(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / abs(denominator) * 100.0


def _classify_trend(values: Sequence[float]) -> Dict[str, Any]:
    if len(values) < 2:
        return {
            "direction": "insufficient_data",
            "positive_steps": 0,
            "negative_steps": 0,
            "flat_steps": 0,
            "consistency": None,
        }
    diffs = [b - a for a, b in zip(values, values[1:])]
    value_range = max(values) - min(values)
    eps = max(value_range * 1e-6, 1e-12)
    positive = sum(1 for d in diffs if d > eps)
    negative = sum(1 for d in diffs if d < -eps)
    flat = len(diffs) - positive - negative
    net = values[-1] - values[0]
    if abs(net) <= eps:
        direction = "flat"
        consistency = flat / len(diffs) if diffs else None
    elif net < 0:
        consistency = negative / len(diffs)
        direction = "decreasing" if consistency >= 0.8 else "volatile_decreasing"
    else:
        consistency = positive / len(diffs)
        direction = "increasing" if consistency >= 0.8 else "volatile_increasing"
    return {
        "direction": direction,
        "positive_steps": positive,
        "negative_steps": negative,
        "flat_steps": flat,
        "consistency": consistency,
    }


def _plateau(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(rows) < 4:
        return None
    values = [float(r["value"]) for r in rows]
    value_range = max(values) - min(values)
    tolerance = max(value_range * 0.02, 1e-12)
    for idx in range(1, len(rows) - 2):
        tail = values[idx:]
        if max(tail) - min(tail) <= tolerance:
            return {
                "start_step": rows[idx]["step"],
                "tolerance": tolerance,
                "tail_points": len(rows) - idx,
            }
    return None


def _scalar_events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not rows:
        return events
    min_row = min(rows, key=lambda r: float(r["value"]))
    max_row = max(rows, key=lambda r: float(r["value"]))
    events.append({"type": "minimum", **_point(min_row)})
    events.append({"type": "maximum", **_point(max_row)})

    if len(rows) >= 2:
        changes: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = [
            (
                float(curr["value"]) - float(prev["value"]),
                prev,
                curr,
            )
            for prev, curr in zip(rows, rows[1:])
        ]
        largest_drop = min(changes, key=lambda item: item[0])
        largest_rise = max(changes, key=lambda item: item[0])
        events.append(
            {
                "type": "largest_drop",
                "from_step": largest_drop[1]["step"],
                "to_step": largest_drop[2]["step"],
                "delta": largest_drop[0],
            }
        )
        events.append(
            {
                "type": "largest_rise",
                "from_step": largest_rise[1]["step"],
                "to_step": largest_rise[2]["step"],
                "delta": largest_rise[0],
            }
        )
    plateau = _plateau(rows)
    if plateau is not None:
        events.append({"type": "plateau", **plateau})
    return events


def _analyze_scalar_payload(
    experiment_name: str,
    experiment_id: int,
    tag: str,
    rows: Sequence[Dict[str, Any]],
    objective: str = "min",
) -> Dict[str, Any]:
    objective = (objective or "min").lower()
    if objective not in {"min", "max"}:
        return {"error": "objective must be 'min' or 'max'", "objective": objective}

    clean = _finite_scalar_rows(rows)
    if not clean:
        return {
            "error": "No scalar data found",
            "experiment": experiment_name,
            "experiment_id": experiment_id,
            "tag": tag,
        }

    values = [float(row["value"]) for row in clean]
    steps = [row["step"] for row in clean]
    first = clean[0]
    last = clean[-1]
    min_row = min(clean, key=lambda r: float(r["value"]))
    max_row = max(clean, key=lambda r: float(r["value"]))
    best_row = min_row if objective == "min" else max_row
    worst_row = max_row if objective == "min" else min_row
    delta = float(last["value"]) - float(first["value"])
    best_gain = (
        float(first["value"]) - float(best_row["value"])
        if objective == "min"
        else float(best_row["value"]) - float(first["value"])
    )
    step_span = steps[-1] - steps[0] if len(steps) >= 2 else 0
    return {
        "experiment": experiment_name,
        "experiment_id": experiment_id,
        "tag": tag,
        "objective": objective,
        "count": len(clean),
        "first": _point(first),
        "last": _point(last),
        "min": _point(min_row),
        "max": _point(max_row),
        "best": _point(best_row),
        "worst": _point(worst_row),
        "change": {
            "absolute": delta,
            "percent": _percent(delta, float(first["value"])),
        },
        "improvement": {
            "absolute": best_gain,
            "percent": _percent(best_gain, float(first["value"])),
        },
        "trend": {
            **_classify_trend(values),
            "slope_per_step": delta / step_span if step_span else None,
        },
        "plateau": _plateau(clean),
        "events": _scalar_events(clean),
    }


def _compare_scalar_payload(
    experiments: Sequence[Any],
    tag: str,
    objective: str = "min",
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    objective = (objective or "min").lower()
    if objective not in {"min", "max"}:
        return {"error": "objective must be 'min' or 'max'", "objective": objective}

    analyses: List[Dict[str, Any]] = []
    missing: List[str] = []
    for exp in experiments:
        analysis = _analyze_scalar_payload(
            exp.name,
            exp.experiment_id,
            tag,
            exp.scalars(tag),
            objective,
        )
        if "error" in analysis:
            missing.append(exp.name)
            continue
        analyses.append(analysis)

    reverse = objective == "max"
    analyses.sort(key=lambda row: row["best"]["value"], reverse=reverse)
    if top_k is not None and top_k > 0:
        analyses = analyses[:top_k]

    ranking = [
        {
            "rank": idx + 1,
            "experiment": row["experiment"],
            "experiment_id": row["experiment_id"],
            "count": row["count"],
            "first": row["first"],
            "last": row["last"],
            "best": row["best"],
            "worst": row["worst"],
            "improvement": row["improvement"],
            "trend": row["trend"],
            "plateau": row["plateau"],
        }
        for idx, row in enumerate(analyses)
    ]
    return {
        "tag": tag,
        "objective": objective,
        "winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "missing": missing,
    }


def _image_entry_by_step(
    rows: Sequence[Dict[str, Any]],
    step: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if step is None:
        return max(rows, key=lambda row: row.get("step", -1))
    matches = [row for row in rows if row.get("step") == step]
    if not matches:
        return None
    return matches[-1]


def _choose_image_pair(
    reference_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    step: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[int], str]:
    if step is not None:
        return (
            _image_entry_by_step(reference_rows, step),
            _image_entry_by_step(candidate_rows, step),
            step,
            "requested_step",
        )

    reference_steps = {row.get("step") for row in reference_rows}
    candidate_steps = {row.get("step") for row in candidate_rows}
    common_steps = sorted(
        s for s in reference_steps & candidate_steps if isinstance(s, int)
    )
    if common_steps:
        chosen = common_steps[-1]
        return (
            _image_entry_by_step(reference_rows, chosen),
            _image_entry_by_step(candidate_rows, chosen),
            chosen,
            "latest_common_step",
        )
    return (
        _image_entry_by_step(reference_rows, None),
        _image_entry_by_step(candidate_rows, None),
        None,
        "latest_each_no_common_step",
    )


def _load_rgb_image(path: str) -> Tuple[Any, Optional[str]]:
    if not path:
        return None, "Image path is empty or outside the experiment log directory"
    image_path = Path(path)
    if not image_path.is_file():
        return None, f"Image file not found: {path}"
    try:
        from PIL import Image
    except ImportError:
        return None, "Pillow is required for image comparison"

    try:
        img = Image.open(image_path)
        width, height = img.size
        if width * height > _MAX_IMAGE_PIXELS:
            return None, (
                f"Image is too large for MCP comparison: {width}x{height}; "
                f"limit is {_MAX_IMAGE_PIXELS} pixels"
            )
        return img.convert("RGB"), None
    except Exception as exc:
        return None, f"Could not open image {path}: {exc}"


def _resize_to_match(reference_img: Any, candidate_img: Any) -> Tuple[Any, bool]:
    if reference_img.size == candidate_img.size:
        return candidate_img, False
    resample = getattr(getattr(reference_img, "Resampling", None), "BICUBIC", None)
    if resample is None:
        resample = 3
    return candidate_img.resize(reference_img.size, resample), True


def _downsample_pair(
    reference_img: Any,
    candidate_img: Any,
    max_size: int,
) -> Tuple[Any, Any, bool]:
    if max_size <= 0:
        return reference_img, candidate_img, False
    width, height = reference_img.size
    largest = max(width, height)
    if largest <= max_size:
        return reference_img, candidate_img, False
    scale = max_size / float(largest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resample = getattr(getattr(reference_img, "Resampling", None), "BICUBIC", None)
    if resample is None:
        resample = 3
    return (
        reference_img.resize(new_size, resample),
        candidate_img.resize(new_size, resample),
        True,
    )


def _pixel_metrics(
    reference_img: Any,
    candidate_img: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        import numpy as np
    except ImportError:
        return None, "numpy is required for pixel fallback metrics"
    ref = np.asarray(reference_img).astype("float32") / 255.0
    cand = np.asarray(candidate_img).astype("float32") / 255.0
    diff = ref - cand
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(mse)
    psnr = None if mse == 0 else 20.0 * math.log10(1.0 / rmse)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr_db": psnr,
    }, None


def _lpips_metric(
    reference_img: Any,
    candidate_img: Any,
) -> Tuple[Optional[float], Optional[str]]:
    try:
        import numpy as np
        import torch
        import lpips
    except ImportError as exc:
        return None, f"LPIPS unavailable: {exc}"

    try:
        ref = np.asarray(reference_img).astype("float32") / 255.0
        cand = np.asarray(candidate_img).astype("float32") / 255.0
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        cand_tensor = torch.from_numpy(cand).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        model = lpips.LPIPS(net="alex", verbose=False)
        model.eval()
        with torch.no_grad():
            return float(model(ref_tensor, cand_tensor).item()), None
    except Exception as exc:
        return None, f"LPIPS computation failed: {exc}"


def _compare_image_lpips_payload(
    reference: Any,
    candidate: Any,
    tag: str,
    candidate_tag: Optional[str] = None,
    step: Optional[int] = None,
    max_size: int = 512,
) -> Dict[str, Any]:
    candidate_tag = candidate_tag or tag
    if step is not None:
        try:
            step = int(step)
        except (TypeError, ValueError):
            return {"error": "step must be an integer", "step": step}
    try:
        max_size = int(max_size)
    except (TypeError, ValueError):
        return {"error": "max_size must be an integer", "max_size": max_size}

    reference_rows = reference.images(tag)
    candidate_rows = candidate.images(candidate_tag)
    if not reference_rows:
        return {
            "error": "No reference images found",
            "experiment": reference.name,
            "tag": tag,
        }
    if not candidate_rows:
        return {
            "error": "No candidate images found",
            "experiment": candidate.name,
            "tag": candidate_tag,
        }

    reference_entry, candidate_entry, chosen_step, selection = _choose_image_pair(
        reference_rows,
        candidate_rows,
        step,
    )
    if reference_entry is None or candidate_entry is None:
        return {
            "error": "Could not find image entries at the requested step",
            "step": step,
            "reference_experiment": reference.name,
            "candidate_experiment": candidate.name,
        }

    reference_img, reference_error = _load_rgb_image(
        reference_entry.get("abs_path", "")
    )
    if reference_error is not None:
        return {"error": reference_error}
    candidate_img, candidate_error = _load_rgb_image(
        candidate_entry.get("abs_path", "")
    )
    if candidate_error is not None:
        return {"error": candidate_error}

    original_reference_size = list(reference_img.size)
    original_candidate_size = list(candidate_img.size)
    candidate_img, resized_to_match = _resize_to_match(
        reference_img,
        candidate_img,
    )
    reference_img, candidate_img, downsampled = _downsample_pair(
        reference_img,
        candidate_img,
        max_size,
    )
    pixel_metrics, pixel_error = _pixel_metrics(reference_img, candidate_img)
    lpips_distance, lpips_error = _lpips_metric(reference_img, candidate_img)

    return {
        "reference": {
            "experiment": reference.name,
            "experiment_id": reference.experiment_id,
            "tag": tag,
            "step": reference_entry.get("step"),
            "path": reference_entry.get("path"),
        },
        "candidate": {
            "experiment": candidate.name,
            "experiment_id": candidate.experiment_id,
            "tag": candidate_tag,
            "step": candidate_entry.get("step"),
            "path": candidate_entry.get("path"),
        },
        "selection": selection,
        "requested_step": step,
        "compared_step": chosen_step,
        "original_reference_size": original_reference_size,
        "original_candidate_size": original_candidate_size,
        "comparison_size": list(reference_img.size),
        "resized_candidate_to_reference": resized_to_match,
        "downsampled_for_comparison": downsampled,
        "lpips": {
            "distance": lpips_distance,
            "available": lpips_error is None,
            "error": lpips_error,
            "install": (
                "pip install lpips torch Pillow numpy"
                if lpips_error is not None
                else None
            ),
        },
        "pixel": pixel_metrics,
        "pixel_error": pixel_error,
    }


@contextlib.contextmanager
def _suppress_streamable_http_startup_log() -> Any:
    """Mute the noisy startup INFO emitted by MCP's streamable HTTP manager."""
    logger = logging.getLogger(_STREAMABLE_HTTP_LOGGER)
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        yield
    finally:
        logger.setLevel(previous_level)


class MCPOutput(BaseOutput):
    """Expose experiment tracking data via an MCP server."""

    def __init__(self, project_folder: Optional[str] = None) -> None:
        super().__init__(project_folder)
        self._mcp_kwargs: dict = {}
        self._transport = "streamable-http"

    def _build_mcp(self) -> Any:
        """Build the FastMCP instance (deferred so host/port can be set first)."""
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise ImportError(
                "The MCP viewer requires the optional `mcp` package "
                "(Python 3.10+). Install it with: "
                "pip install 'vibetrack[mcp]'"
            ) from exc

        mcp = FastMCP("vibetrack", **self._mcp_kwargs)
        self._mcp = mcp
        self._register_tools()
        self._register_resources()
        return mcp

    # ── Tools ────────────────────────────────────────────────

    def _register_tools(self) -> None:
        reader = self._reader
        resolve = self._resolve_experiments
        mcp = self._mcp
        project_folder = self.project_folder

        @mcp.tool()
        def list_experiments() -> str:
            """List all experiments in the project database."""
            exps = reader.experiments()
            return json.dumps(
                [{"name": e.name, "experiment_id": e.experiment_id} for e in exps]
            )

        @mcp.tool()
        def get_experiment_tags(experiment: str) -> str:
            """Get all available tags for an experiment, grouped by type."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            exp = exps[0]
            return json.dumps(
                {
                    "name": exp.name,
                    "scalars": exp.scalar_tags(),
                    "texts": exp.text_tags(),
                    "images": exp.image_tags(),
                    "audio": exp.audio_tags(),
                    "video": exp.video_tags(),
                    "artifacts": exp.user_artifact_tags(),
                    "models": exp.model_tags(),
                    "pr_curves": exp.pr_curve_tags(),
                    "histograms": exp.histogram_tags(),
                }
            )

        @mcp.tool()
        def get_scalars(experiment: str, tag: str) -> str:
            """Get scalar time-series data for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].scalars(tag))

        @mcp.tool()
        def analyze_scalar(
            experiment: str,
            tag: str,
            objective: str = "min",
        ) -> str:
            """Summarize a scalar graph: min/max, best step, trend, plateau, events."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            exp = exps[0]
            return _json_response(
                _analyze_scalar_payload(
                    exp.name,
                    exp.experiment_id,
                    tag,
                    exp.scalars(tag),
                    objective=objective,
                )
            )

        @mcp.tool()
        def compare_scalar(
            tag: str,
            experiments: Optional[List[str]] = None,
            objective: str = "min",
            top_k: Optional[int] = None,
        ) -> str:
            """Rank experiments by best value for one scalar tag."""
            exps = resolve(experiments)
            if not exps:
                return json.dumps({"error": "No experiments found"})
            return _json_response(
                _compare_scalar_payload(
                    exps,
                    tag,
                    objective=objective,
                    top_k=top_k,
                )
            )

        @mcp.tool()
        def find_metric_events(experiment: str, tag: str) -> str:
            """Find notable scalar graph events such as extrema, jumps, and plateau."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            exp = exps[0]
            rows = _finite_scalar_rows(exp.scalars(tag))
            if not rows:
                return json.dumps(
                    {
                        "error": "No scalar data found",
                        "experiment": exp.name,
                        "experiment_id": exp.experiment_id,
                        "tag": tag,
                    }
                )
            return _json_response(
                {
                    "experiment": exp.name,
                    "experiment_id": exp.experiment_id,
                    "tag": tag,
                    "count": len(rows),
                    "events": _scalar_events(rows),
                }
            )

        @mcp.tool()
        def get_texts(experiment: str, tag: str) -> str:
            """Get text entries for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].texts(tag))

        @mcp.tool()
        def get_images(experiment: str, tag: str) -> str:
            """Get image entries (step, path) for a specific tag."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].images(tag))

        @mcp.tool()
        def compare_image_lpips(
            reference_experiment: str,
            candidate_experiment: str,
            tag: str,
            candidate_tag: Optional[str] = None,
            step: Optional[int] = None,
            max_size: int = 512,
        ) -> str:
            """Compare two logged images with LPIPS when installed plus pixel metrics.

            Uses the same image tag for both experiments unless ``candidate_tag``
            is provided. If ``step`` is omitted, the latest common step is used.
            """
            exp_list = resolve([reference_experiment, candidate_experiment])
            exp_by_name = {exp.name: exp for exp in exp_list}
            reference = exp_by_name.get(reference_experiment)
            candidate = exp_by_name.get(candidate_experiment)
            if reference is None:
                return json.dumps(
                    {"error": f"Experiment {reference_experiment!r} not found"}
                )
            if candidate is None:
                return json.dumps(
                    {"error": f"Experiment {candidate_experiment!r} not found"}
                )

            return _json_response(
                _compare_image_lpips_payload(
                    reference,
                    candidate,
                    tag,
                    candidate_tag=candidate_tag,
                    step=step,
                    max_size=max_size,
                )
            )

        @mcp.tool()
        def get_audio(experiment: str, tag: str) -> str:
            """Get audio entries (step, path, sample_rate) for a specific tag."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].audio(tag))

        @mcp.tool()
        def get_hparams(experiment: str) -> str:
            """Get hyperparameters for an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].hparams())

        @mcp.tool()
        def get_histograms(experiment: str, tag: str) -> str:
            """Get histogram data for a specific tag in an experiment."""
            exps = resolve([experiment])
            if not exps:
                return json.dumps({"error": f"Experiment {experiment!r} not found"})
            return json.dumps(exps[0].histograms(tag))

        @mcp.tool()
        def summary(
            experiments: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
        ) -> str:
            """Get a summary table: last value of each tag per experiment."""
            exps = resolve(experiments)
            if not exps:
                return json.dumps({"error": "No experiments found"})
            return json.dumps(summary_table(exps, tags))

        @mcp.tool()
        def compare_hparams_tool(
            experiments: Optional[List[str]] = None,
        ) -> str:
            """Compare hyperparameters across experiments side-by-side."""
            exps = resolve(experiments)
            if not exps:
                return json.dumps({"error": "No experiments found"})
            return json.dumps(compare_hparams(exps))

        @mcp.tool()
        def run_report(experiment: str) -> str:
            """Get a human-readable end-of-run digest for one experiment.

            Includes hparams, scalar stats (last/min/max/mean + sparkline),
            text entries, and media counts. Mirrors what
            ``writer.to("console", summary=True)`` prints at close time.
            """
            exps = resolve([experiment])
            if not exps:
                return f"Experiment {experiment!r} not found"
            console = ConsoleOutput(project_folder)
            return console._build_run_report(exps[0])

    # ── Resources ────────────────────────────────────────────

    def _register_resources(self) -> None:
        reader = self._reader
        resolve = self._resolve_experiments
        mcp = self._mcp

        @mcp.resource("vibetrack://experiments")
        def experiments_list() -> str:
            """List all experiments."""
            exps = reader.experiments()
            return json.dumps(
                [{"name": e.name, "experiment_id": e.experiment_id} for e in exps]
            )

        @mcp.resource("vibetrack://experiments/{name}")
        def experiment_detail(name: str) -> str:
            """Experiment overview: tags by type, hparams, config."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            exp = exps[0]
            return json.dumps(
                {
                    "name": exp.name,
                    "experiment_id": exp.experiment_id,
                    "scalars": exp.scalar_tags(),
                    "texts": exp.text_tags(),
                    "images": exp.image_tags(),
                    "audio": exp.audio_tags(),
                    "video": exp.video_tags(),
                    "artifacts": exp.user_artifact_tags(),
                    "models": exp.model_tags(),
                    "pr_curves": exp.pr_curve_tags(),
                    "histograms": exp.histogram_tags(),
                    "hparams": exp.hparams(),
                    "config": exp.config(),
                }
            )

        @mcp.resource("vibetrack://experiments/{name}/scalars/{tag}")
        def experiment_scalars(name: str, tag: str) -> str:
            """Scalar time-series for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].scalars(tag))

        @mcp.resource("vibetrack://experiments/{name}/texts/{tag}")
        def experiment_texts(name: str, tag: str) -> str:
            """Text entries for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].texts(tag))

        @mcp.resource("vibetrack://experiments/{name}/images/{tag}")
        def experiment_images(name: str, tag: str) -> str:
            """Image entries for a specific tag."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].images(tag))

        @mcp.resource("vibetrack://experiments/{name}/hparams")
        def experiment_hparams(name: str) -> str:
            """Hyperparameters for an experiment."""
            exps = resolve([name])
            if not exps:
                return json.dumps({"error": f"Experiment {name!r} not found"})
            return json.dumps(exps[0].hparams())

    def asgi_app(self) -> Any:
        """Build MCP and return the Starlette ASGI app for mounting."""
        self._build_mcp()
        return self._mcp.streamable_http_app()

    # ── Viewer interface ─────────────────────────────────────

    def show(self, **kwargs: Any) -> None:
        host = kwargs.get("host", "127.0.0.1")
        port = kwargs.get("port", 6006)
        transport = kwargs.get("mcp_transport", self._transport)

        self._mcp_kwargs.update(host=host, port=port)
        self._build_mcp()

        path = "/mcp" if transport == "streamable-http" else "/sse"
        print(f"vibetrack MCP server: http://{host}:{port}{path}")
        runner = self._mcp.run
        if transport == "streamable-http":
            with _suppress_streamable_http_startup_log():
                runner(transport=transport)
        else:
            runner(transport=transport)

    def start_in_thread(
        self,
        host: str = "127.0.0.1",
        port: int = 6007,
        transport: str = "streamable-http",
    ) -> "Any":
        """Start the MCP server in a daemon thread and return immediately."""
        import threading

        self._mcp_kwargs.update(host=host, port=port)
        self._build_mcp()

        path = "/mcp" if transport == "streamable-http" else "/sse"
        print(f"vibetrack MCP server: http://{host}:{port}{path}")
        if transport == "streamable-http":

            def _run() -> None:
                with _suppress_streamable_http_startup_log():
                    self._mcp.run(transport=transport)

        else:

            def _run() -> None:
                self._mcp.run(transport=transport)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vibetrack MCP server")
    parser.add_argument("--project-folder", default=None, help="Project folder")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6006, help="Port (default: 6006)")
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["streamable-http", "sse"],
        help="MCP transport (default: streamable-http)",
    )
    args = parser.parse_args()

    viewer = MCPOutput(args.project_folder)
    viewer.show(host=args.host, port=args.port, mcp_transport=args.transport)
