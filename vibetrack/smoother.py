"""Smoothing algorithms for experiment metrics — like TensorBoard's smoothing slider."""

from __future__ import annotations

import math
from typing import List, Sequence


def ema(
    values: Sequence[float],
    weight: float = 0.6,
) -> List[float]:
    """Exponential moving average (TensorBoard-style).

    ``weight`` in [0, 1) — 0 means no smoothing, close to 1 means heavy smoothing.
    This matches TensorBoard's smoothing slider exactly.
    """
    if not values:
        return []
    smoothed: List[float] = []
    last = values[0]
    for v in values:
        # De-bias like TensorBoard: keep a running numerator/denominator
        last = weight * last + (1 - weight) * v
        smoothed.append(last)

    # TensorBoard applies de-biasing
    debiased: List[float] = []
    num = 0.0
    den = 0.0
    for v in values:
        num = weight * num + (1 - weight) * v
        den = weight * den + (1 - weight)
        debiased.append(num / den)

    return debiased


def moving_average(
    values: Sequence[float],
    window: int = 10,
) -> List[float]:
    """Simple (uniform-weight) moving average.

    Each output is the arithmetic mean of the last *window* inputs.
    For the first few points where fewer than *window* values are
    available, the mean is taken over whatever is available.
    """
    if not values or window < 1:
        return list(values)
    result: List[float] = []
    buf: List[float] = []
    running_sum = 0.0
    for v in values:
        buf.append(v)
        running_sum += v
        if len(buf) > window:
            running_sum -= buf[-window - 1]
        n = min(len(buf), window)
        result.append(running_sum / n)
    return result


def gaussian(
    values: Sequence[float],
    sigma: float = 2.0,
) -> List[float]:
    """Gaussian kernel smoothing."""
    if not values or sigma <= 0:
        return list(values)
    n = len(values)
    radius = max(1, int(3 * sigma))
    # Pre-compute kernel weights
    kernel = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(radius + 1)]
    result: List[float] = []
    for i in range(n):
        total_w = 0.0
        total_v = 0.0
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            w = kernel[abs(i - j)]
            total_w += w
            total_v += w * values[j]
        result.append(total_v / total_w)
    return result


def smooth(
    values: Sequence[float],
    method: str = "ema",
    **kwargs: float,
) -> List[float]:
    """Unified smoothing entry point.

    Parameters
    ----------
    method : str
        One of ``"ema"``, ``"moving_average"``, ``"gaussian"``, ``"none"``.
    """
    dispatch = {
        "ema": ema,
        "moving_average": moving_average,
        "gaussian": gaussian,
        "none": lambda v, **kw: list(v),
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown smoothing method {method!r}. "
            f"Choose from {list(dispatch.keys())}"
        )
    return fn(values, **kwargs)
