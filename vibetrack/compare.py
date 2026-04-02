"""Compare metrics across multiple experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .reader import ExperimentReader, RunReader
from .smoother import smooth


def compare_scalars(
    experiments: Sequence[ExperimentReader],
    tag: str,
    smoothing: str = "none",
    **smooth_kwargs: float,
) -> List[Dict[str, Any]]:
    """Get scalar data for a tag across multiple experiments.

    Returns a list of dicts, one per experiment::

        [
            {
                "name": "run_001",
                "steps": [0, 1, 2, ...],
                "values": [0.9, 0.7, 0.5, ...],
                "smoothed": [0.9, 0.8, 0.65, ...],  # if smoothing != "none"
            },
            ...
        ]
    """
    results: List[Dict[str, Any]] = []
    for exp in experiments:
        data = exp.scalars(tag)
        if not data:
            continue
        steps = [d["step"] for d in data]
        values = [d["value"] for d in data]
        entry: Dict[str, Any] = {
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "steps": steps,
            "values": values,
        }
        if smoothing != "none":
            entry["smoothed"] = smooth(values, method=smoothing, **smooth_kwargs)
        results.append(entry)
    return results


def compare_hparams(
    experiments: Sequence[ExperimentReader],
) -> List[Dict[str, Any]]:
    """Get hyperparameters for multiple experiments side-by-side.

    Returns::

        [
            {"name": "run_001", "hparams": {"lr": 0.01, "batch_size": 32}},
            ...
        ]
    """
    return [
        {"name": exp.name, "experiment_id": exp.experiment_id, "hparams": exp.hparams()}
        for exp in experiments
    ]


def find_common_tags(experiments: Sequence[ExperimentReader]) -> List[str]:
    """Return scalar tags that exist in all experiments."""
    if not experiments:
        return []
    tag_sets = [set(exp.scalar_tags()) for exp in experiments]
    common = tag_sets[0]
    for ts in tag_sets[1:]:
        common &= ts
    return sorted(common)


def find_all_tags(experiments: Sequence[ExperimentReader]) -> List[str]:
    """Return the union of all scalar tags across experiments."""
    tags: set[str] = set()
    for exp in experiments:
        tags.update(exp.scalar_tags())
    return sorted(tags)


def summary_table(
    experiments: Sequence[ExperimentReader],
    tags: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Build a summary table: last value of each tag per experiment.

    Returns::

        [
            {"name": "run_001", "loss": 0.12, "acc": 0.95, ...},
            ...
        ]
    """
    if tags is None:
        tags = find_all_tags(experiments)
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        row: Dict[str, Any] = {"name": exp.name, "experiment_id": exp.experiment_id}
        for tag in tags:
            data = exp.scalars(tag)
            row[tag] = data[-1]["value"] if data else None
        rows.append(row)
    return rows
