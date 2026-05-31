"""Tests for smoothing algorithms.

Each smoother is verified against its mathematical definition:

- **EMA** (debiased):  s_t = β·s_{t-1} + (1-β)·v_t,  output = s_t / (1 - β^t)
- **SMA** (simple moving average):  output_t = mean(v[max(0,t-w+1) : t+1])
- **Gaussian**:  output_t = Σ_j G(t-j, σ) · v_j  /  Σ_j G(t-j, σ)
  where G(d, σ) = exp(-0.5·(d/σ)²)
"""

import math

import pytest

from vibetrack.smoother import ema, moving_average, gaussian, smooth

# ── EMA ──────────────────────────────────────────────────────────────


class TestEMA:
    def test_no_smoothing(self):
        """weight=0 → debiased EMA must reproduce the original values exactly."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ema(vals, weight=0.0)
        for r, v in zip(result, vals):
            assert abs(r - v) < 1e-12

    def test_single_value(self):
        """Debiased EMA of a single value must equal that value for any weight."""
        for w in [0.0, 0.5, 0.9, 0.99]:
            result = ema([7.0], weight=w)
            assert abs(result[0] - 7.0) < 1e-12

    def test_debiasing_prevents_warmup_pull_to_zero(self):
        """Without debiasing, ema([5.0], weight=0.9) ≈ 0.5.
        With debiasing it must equal 5.0."""
        result = ema([5.0], weight=0.9)
        assert abs(result[0] - 5.0) < 1e-9

    def test_convergence_on_constant_input(self):
        """EMA of a constant sequence must converge to that constant."""
        val = 3.7
        result = ema([val] * 100, weight=0.9)
        assert abs(result[-1] - val) < 1e-10

    def test_debiased_formula_step_by_step(self):
        """Verify every step against the closed-form debiased EMA formula.

        s_t = β·s_{t-1} + (1-β)·v_t   (s_0 = 0 before first input)
        output_t = s_t / (1 - β^t)     (t is 1-indexed)
        """
        beta = 0.6
        vals = [2.0, 4.0, 6.0, 1.0, 8.0]
        result = ema(vals, weight=beta)

        s = 0.0
        for t, v in enumerate(vals, start=1):
            s = beta * s + (1 - beta) * v
            expected = s / (1 - beta**t)
            assert abs(result[t - 1] - expected) < 1e-12, f"Mismatch at step {t}"

    def test_monotone_on_increasing_input(self):
        """If input is strictly increasing, debiased EMA must also be increasing."""
        vals = [float(i) for i in range(1, 20)]
        result = ema(vals, weight=0.7)
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]

    def test_heavy_smoothing_lags_behind(self):
        """High weight → output lags the input (last smoothed < last raw for increasing)."""
        vals = [float(i) for i in range(1, 50)]
        result = ema(vals, weight=0.95)
        assert result[-1] < vals[-1]

    def test_empty(self):
        assert ema([], weight=0.5) == []

    def test_linear_ramp_exact(self):
        """For v_t = t, verify the debiased EMA matches the formula.

        This tests numerical precision on a non-trivial input.
        """
        beta = 0.8
        vals = [float(t) for t in range(20)]
        result = ema(vals, weight=beta)

        s = 0.0
        for t, v in enumerate(vals, start=1):
            s = beta * s + (1 - beta) * v
            expected = s / (1 - beta**t)
            assert abs(result[t - 1] - expected) < 1e-10


# ── Simple Moving Average ────────────────────────────────────────────


class TestMovingAverage:
    def test_window_1_identity(self):
        """Window=1 → must reproduce original values exactly."""
        vals = [1.0, 2.0, 3.0]
        result = moving_average(vals, window=1)
        for r, v in zip(result, vals):
            assert abs(r - v) < 1e-12

    def test_exact_arithmetic_mean(self):
        """Each output must be the arithmetic mean of the last `window` inputs."""
        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = moving_average(vals, window=3)
        assert len(result) == 5
        # First: mean([1]) = 1
        assert abs(result[0] - 1.0) < 1e-12
        # Second: mean([1,3]) = 2
        assert abs(result[1] - 2.0) < 1e-12
        # Index 2: mean([1,3,5]) = 3.0
        assert abs(result[2] - 3.0) < 1e-12
        # Index 3: mean([3,5,7]) = 5.0
        assert abs(result[3] - 5.0) < 1e-12
        # Index 4: mean([5,7,9]) = 7.0
        assert abs(result[4] - 7.0) < 1e-12

    def test_full_window_formula(self):
        """Verify every output against the definition: mean(v[max(0,t-w+1):t+1])."""
        vals = [2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 9.0, 5.0, 0.0]
        window = 4
        result = moving_average(vals, window=window)
        for t in range(len(vals)):
            start = max(0, t - window + 1)
            expected = sum(vals[start : t + 1]) / (t - start + 1)
            assert abs(result[t] - expected) < 1e-12, f"Mismatch at t={t}"

    def test_constant_input(self):
        """SMA of a constant sequence must return that constant at every step."""
        val = 4.2
        result = moving_average([val] * 30, window=5)
        for r in result:
            assert abs(r - val) < 1e-12

    def test_window_larger_than_sequence(self):
        """Window > len(vals) — uses expanding window, must not crash."""
        vals = [2.0, 4.0, 6.0]
        result = moving_average(vals, window=100)
        assert len(result) == 3
        assert abs(result[0] - 2.0) < 1e-12
        assert abs(result[1] - 3.0) < 1e-12  # mean(2,4)
        assert abs(result[2] - 4.0) < 1e-12  # mean(2,4,6)

    def test_sum_preservation(self):
        """For window covering full sequence, the last output equals the global mean."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average(vals, window=len(vals))
        assert abs(result[-1] - 3.0) < 1e-12

    def test_empty(self):
        assert moving_average([], window=3) == []

    def test_single_value(self):
        result = moving_average([42.0], window=5)
        assert abs(result[0] - 42.0) < 1e-12


# ── Gaussian ─────────────────────────────────────────────────────────


class TestGaussian:
    def test_constant_unchanged(self):
        """Gaussian smoothing of a constant signal must return that constant."""
        vals = [5.0] * 30
        result = gaussian(vals, sigma=3.0)
        for r in result:
            assert abs(r - 5.0) < 1e-10

    def test_smooths_spike(self):
        """A single spike must be spread and reduced by gaussian smoothing."""
        vals = [0.0] * 10 + [10.0] + [0.0] * 10
        result = gaussian(vals, sigma=2.0)
        assert result[10] < 10.0  # peak reduced
        assert result[9] > 0.0  # neighbors raised
        assert result[11] > 0.0

    def test_symmetry_around_spike(self):
        """Gaussian kernel is symmetric — output around a centered spike must be symmetric."""
        vals = [0.0] * 20 + [10.0] + [0.0] * 20
        result = gaussian(vals, sigma=3.0)
        center = 20
        for d in range(1, 8):
            assert abs(result[center - d] - result[center + d]) < 1e-10

    def test_formula_against_manual_computation(self):
        """Verify output at every point against the exact Gaussian kernel formula."""
        vals = [1.0, 3.0, 0.0, 5.0, 2.0]
        sigma = 1.5
        result = gaussian(vals, sigma=sigma)
        n = len(vals)
        radius = max(1, int(3 * sigma))

        for i in range(n):
            total_w = 0.0
            total_v = 0.0
            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                w = math.exp(-0.5 * ((i - j) / sigma) ** 2)
                total_w += w
                total_v += w * vals[j]
            expected = total_v / total_w
            assert abs(result[i] - expected) < 1e-12, f"Mismatch at i={i}"

    def test_area_preservation(self):
        """Gaussian smoothing must approximately preserve the signal integral.

        For a kernel much narrower than the signal, the total sum is conserved.
        """
        vals = [0.0] * 20 + [10.0] * 10 + [0.0] * 20
        result = gaussian(vals, sigma=1.0)
        assert abs(sum(result) - sum(vals)) / sum(vals) < 0.05

    def test_wider_sigma_smoother(self):
        """Larger sigma must produce a smoother (flatter) result around a spike."""
        vals = [0.0] * 15 + [10.0] + [0.0] * 15
        r_narrow = gaussian(vals, sigma=1.0)
        r_wide = gaussian(vals, sigma=4.0)
        # Wide kernel peak must be lower than narrow kernel peak
        assert r_wide[15] < r_narrow[15]

    def test_sigma_zero_returns_original(self):
        vals = [1.0, 2.0, 3.0]
        result = gaussian(vals, sigma=0)
        assert result == vals

    def test_empty(self):
        assert gaussian([], sigma=2.0) == []


# ── Dispatch ─────────────────────────────────────────────────────────


class TestSmoothDispatch:
    def test_none_returns_input_unchanged(self):
        vals = [1.0, 2.0, 3.0]
        assert smooth(vals, method="none") == vals

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown smoothing"):
            smooth([1.0], method="invalid")

    def test_all_methods_return_same_length(self):
        vals = [float(v) for v in range(1, 21)]
        for method, kwargs in [
            ("ema", {"weight": 0.5}),
            ("moving_average", {"window": 3}),
            ("gaussian", {"sigma": 1.0}),
            ("none", {}),
        ]:
            result = smooth(vals, method=method, **kwargs)
            assert len(result) == len(vals), f"method={method} changed length"

    def test_all_methods_finite(self):
        """No method should produce NaN or Inf for reasonable input."""
        vals = [float(v) for v in range(1, 21)]
        for method, kwargs in [
            ("ema", {"weight": 0.99}),
            ("moving_average", {"window": 7}),
            ("gaussian", {"sigma": 5.0}),
        ]:
            result = smooth(vals, method=method, **kwargs)
            assert all(
                math.isfinite(v) for v in result
            ), f"method={method} has non-finite"
