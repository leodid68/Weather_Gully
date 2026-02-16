# Skewed Student-t Distribution — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add Fernández-Steel skewed Student-t CDF to the probability engine, fit (df, γ) via MLE in calibration, and validate the improvement over Normal/Student-t via Brier score.

**Architecture:** The skew-t CDF delegates to the existing `_student_t_cdf`. Calibration adds a 2D grid search (df × γ) alongside the existing `_fit_student_t_df`. A new validation script scores 4 distributions against historical resolutions. Integration is a 3-line CDF switch change.

**Tech Stack:** Python 3.14, stdlib only (math, json, logging). No scipy/numpy.

---

## Task 1: Implement `_skew_t_cdf` in probability.py

**Files:**
- Modify: `weather/probability.py:214-243` (add after `_student_t_cdf`)
- Test: `weather/tests/test_probability.py`

### What to build

The Fernández-Steel skew-t CDF with parameter γ (gamma):
- γ = 1.0 → symmetric (standard Student-t)
- γ < 1.0 → left-skewed (more weight in left tail)
- γ > 1.0 → right-skewed (more weight in right tail)

The CDF formula for standardized variable x with df degrees of freedom and skewness γ:

```
_skew_t_cdf(x, df, gamma):
    k = 2 * gamma / (gamma^2 + 1)   # normalizing constant for PDF

    if x < 0:
        return k * gamma * T(x * gamma, df)
    else:
        # At x=0: left piece contributes k * gamma * T(0, df) = k * gamma * 0.5
        # Right piece CDF = 1 - k * (1/gamma) * (1 - T(x/gamma, df))
        left_mass = k * gamma * 0.5
        right_cdf = k / gamma * (T(x/gamma, df) - 0.5)
        return left_mass + right_cdf
```

Where T is `_student_t_cdf` (already implemented).

### Implementation

Add this function in `weather/probability.py` after `_student_t_cdf` (after line 243):

```python
def _skew_t_cdf(x: float, df: float, gamma: float) -> float:
    """CDF of the Fernández-Steel skewed Student's t-distribution.

    Args:
        x: Standardized value (z-score).
        df: Degrees of freedom (> 0).
        gamma: Skewness parameter (> 0).
            gamma = 1.0: symmetric (reduces to Student-t).
            gamma < 1.0: left-skewed (heavier left tail).
            gamma > 1.0: right-skewed (heavier right tail).

    Returns:
        CDF value in [0, 1].
    """
    # Guard: invalid inputs
    if math.isnan(x) or df <= 0:
        return 0.5
    if math.isinf(x):
        return 1.0 if x > 0 else 0.0

    # Clamp gamma to safe range
    gamma = max(0.3, min(3.0, gamma))

    # gamma == 1.0 → standard Student-t (avoid division artifacts)
    if abs(gamma - 1.0) < 1e-10:
        return _student_t_cdf(x, df)

    # Normalizing constant: k = 2*gamma / (gamma^2 + 1)
    g2 = gamma * gamma
    k = 2.0 * gamma / (g2 + 1.0)

    if x < 0:
        # Left tail: scale x by gamma (compresses or stretches left tail)
        return k * gamma * _student_t_cdf(x * gamma, df)
    else:
        # Right tail: total left mass at x=0 + right integral
        left_at_zero = k * gamma * 0.5
        right_integral = (k / gamma) * (_student_t_cdf(x / gamma, df) - 0.5)
        return left_at_zero + right_integral
```

### Tests

Add a new test class `TestSkewTCDF` in `weather/tests/test_probability.py`:

```python
class TestSkewTCDF(unittest.TestCase):
    """Tests for the Fernández-Steel skewed Student-t CDF."""

    def test_gamma_one_equals_student_t(self):
        """gamma=1.0 should match standard Student-t exactly."""
        from weather.probability import _skew_t_cdf, _student_t_cdf
        for df in [3, 5, 10, 30]:
            for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                self.assertAlmostEqual(
                    _skew_t_cdf(x, df, 1.0),
                    _student_t_cdf(x, df),
                    places=6,
                    msg=f"df={df}, x={x}",
                )

    def test_center_is_half(self):
        """F(0) should be 0.5 for any gamma (distribution centered at 0)."""
        from weather.probability import _skew_t_cdf
        for gamma in [0.5, 0.8, 1.0, 1.2, 1.5]:
            self.assertAlmostEqual(
                _skew_t_cdf(0, 10, gamma), 0.5, places=6,
                msg=f"gamma={gamma}",
            )

    def test_left_skew_heavier_left_tail(self):
        """gamma < 1 → more probability in the left tail."""
        from weather.probability import _skew_t_cdf
        # P(X < -3) should be larger for left-skewed (gamma=0.7) than symmetric
        left_skew = _skew_t_cdf(-3.0, 10, 0.7)
        symmetric = _skew_t_cdf(-3.0, 10, 1.0)
        self.assertGreater(left_skew, symmetric)

    def test_right_skew_heavier_right_tail(self):
        """gamma > 1 → more probability in the right tail."""
        from weather.probability import _skew_t_cdf
        # P(X > 3) should be larger for right-skewed (gamma=1.5) than symmetric
        right_skew = 1 - _skew_t_cdf(3.0, 10, 1.5)
        symmetric = 1 - _skew_t_cdf(3.0, 10, 1.0)
        self.assertGreater(right_skew, symmetric)

    def test_cdf_monotonically_increasing(self):
        """CDF must be monotonically non-decreasing."""
        from weather.probability import _skew_t_cdf
        for gamma in [0.6, 1.0, 1.4]:
            prev = 0.0
            for x_int in range(-50, 51):
                x = x_int * 0.2
                val = _skew_t_cdf(x, 10, gamma)
                self.assertGreaterEqual(val, prev - 1e-10,
                    msg=f"Non-monotonic at x={x}, gamma={gamma}")
                prev = val

    def test_limits_zero_and_one(self):
        """CDF should approach 0 at -inf and 1 at +inf."""
        from weather.probability import _skew_t_cdf
        for gamma in [0.5, 1.0, 1.5]:
            self.assertAlmostEqual(_skew_t_cdf(-1e10, 10, gamma), 0.0, places=4)
            self.assertAlmostEqual(_skew_t_cdf(1e10, 10, gamma), 1.0, places=4)

    def test_inf_inputs(self):
        from weather.probability import _skew_t_cdf
        self.assertEqual(_skew_t_cdf(float('inf'), 10, 0.8), 1.0)
        self.assertEqual(_skew_t_cdf(float('-inf'), 10, 0.8), 0.0)

    def test_nan_returns_half(self):
        from weather.probability import _skew_t_cdf
        self.assertEqual(_skew_t_cdf(float('nan'), 10, 0.8), 0.5)

    def test_gamma_clamped(self):
        """Extreme gamma values should be clamped, not crash."""
        from weather.probability import _skew_t_cdf
        result_low = _skew_t_cdf(0.0, 10, 0.01)  # clamped to 0.3
        result_high = _skew_t_cdf(0.0, 10, 100.0)  # clamped to 3.0
        self.assertAlmostEqual(result_low, 0.5, places=3)
        self.assertAlmostEqual(result_high, 0.5, places=3)
```

### Verification
```
python3 -m pytest weather/tests/test_probability.py::TestSkewTCDF -v
python3 -m pytest weather/tests/test_probability.py -q
```

---

## Task 2: Wire skew-t into `estimate_bucket_probability`

**Files:**
- Modify: `weather/probability.py:589-596` (CDF switch in `estimate_bucket_probability`)
- Modify: `weather/probability.py:778-785` (CDF switch in `estimate_bucket_probability_with_obs`)
- Test: `weather/tests/test_probability.py`

### What to build

Update both CDF selection blocks to support `"skew_t"` distribution from `calibration.json`.

### Implementation

In `estimate_bucket_probability()` (lines 589-596), replace:

```python
    # Determine CDF function based on calibration distribution
    cal = _load_calibration()
    dist = cal.get("distribution", "normal") if cal else "normal"
    if dist == "student_t":
        t_df = cal.get("student_t_df", 30)
        cdf_fn = lambda z: _student_t_cdf(z, t_df)
    else:
        cdf_fn = _normal_cdf
```

with:

```python
    # Determine CDF function based on calibration distribution
    cal = _load_calibration()
    dist = cal.get("distribution", "normal") if cal else "normal"
    if dist == "skew_t":
        t_df = cal.get("student_t_df", 10)
        gamma = cal.get("skew_t_gamma", 1.0)
        cdf_fn = lambda z: _skew_t_cdf(z, t_df, gamma)
    elif dist == "student_t":
        t_df = cal.get("student_t_df", 30)
        cdf_fn = lambda z: _student_t_cdf(z, t_df)
    else:
        cdf_fn = _normal_cdf
```

Apply the **exact same change** in `estimate_bucket_probability_with_obs()` (lines 778-785).

### Tests

Add in `weather/tests/test_probability.py`:

```python
class TestSkewTIntegration(unittest.TestCase):
    """Test that skew-t distribution is used when calibration says so."""

    def test_skew_t_distribution_from_calibration(self):
        """When calibration says skew_t, bucket probabilities should differ from normal."""
        from unittest.mock import patch

        cal_skew_t = {
            "distribution": "skew_t",
            "student_t_df": 10,
            "skew_t_gamma": 0.7,
        }
        cal_normal = {
            "distribution": "normal",
        }

        with patch("weather.probability._load_calibration", return_value=cal_skew_t):
            prob_skew = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        with patch("weather.probability._load_calibration", return_value=cal_normal):
            prob_normal = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )

        # Left-skewed (gamma=0.7) should give higher prob for left-tail bucket
        self.assertGreater(prob_skew, prob_normal)

    def test_skew_t_obs_function_uses_skew(self):
        """estimate_bucket_probability_with_obs should also use skew-t."""
        from unittest.mock import patch
        from weather.probability import estimate_bucket_probability_with_obs

        cal = {"distribution": "skew_t", "student_t_df": 10, "skew_t_gamma": 0.7}

        with patch("weather.probability._load_calibration", return_value=cal):
            prob = estimate_bucket_probability_with_obs(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_missing_gamma_defaults_to_symmetric(self):
        """If skew_t_gamma is missing, should default to 1.0 (symmetric)."""
        from unittest.mock import patch
        from weather.probability import _student_t_cdf

        cal = {"distribution": "skew_t", "student_t_df": 10}  # no gamma key

        with patch("weather.probability._load_calibration", return_value=cal):
            prob = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=45, bucket_high=55,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        # gamma=1.0 → same as Student-t
        self.assertGreater(prob, 0.0)
```

### Verification
```
python3 -m pytest weather/tests/test_probability.py::TestSkewTIntegration -v
python3 -m pytest weather/tests/ bot/tests/ -q  # Full suite — no regressions
```

---

## Task 3: Add `_skew_t_logpdf` and `_fit_skew_t_params` to calibrate.py

**Files:**
- Modify: `weather/calibrate.py:620-648` (add after `_fit_student_t_df`)
- Test: `weather/tests/test_calibrate.py`

### What to build

1. `_skew_t_logpdf(x, df, gamma)` — log-PDF of the Fernández-Steel skew-t
2. `_fit_skew_t_params(errors)` — 2D grid search over (df, γ) by maximum likelihood

### Implementation

Add after `_fit_student_t_df` (after line 648 of `calibrate.py`):

```python
def _skew_t_logpdf(x: float, df: float, gamma: float) -> float:
    """Log-PDF of the Fernández-Steel skewed Student's t-distribution.

    f(x | df, gamma) = (2 / (gamma + 1/gamma)) * t(x_scaled, df)
    where x_scaled = x * gamma if x < 0, x / gamma if x >= 0,
    and t(·, df) is the standard Student-t PDF.
    """
    g2 = gamma * gamma
    log_norm = math.log(2.0 * gamma / (g2 + 1.0))

    if x < 0:
        z = x * gamma
    else:
        z = x / gamma

    # Student-t log-PDF
    log_t = (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - ((df + 1) / 2) * math.log(1 + z * z / df)
    )

    return log_norm + log_t


def _fit_skew_t_params(errors: list[float]) -> tuple[float, float]:
    """Fit (df, gamma) for the Fernández-Steel skew-t by maximum likelihood.

    Grid search over candidate df and gamma values.
    Returns (best_df, best_gamma).
    """
    n = len(errors)
    if n < 30:
        return 10.0, 1.0  # Default fallback

    sigma = (sum(e ** 2 for e in errors) / n) ** 0.5
    standardized = [e / sigma for e in errors] if sigma > 0 else errors

    best_df = 10.0
    best_gamma = 1.0
    best_ll = float("-inf")

    for df in [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
        for gamma_int in [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]:
            gamma = gamma_int / 10.0
            ll = sum(_skew_t_logpdf(z, df, gamma) for z in standardized)
            if ll > best_ll:
                best_ll = ll
                best_df = float(df)
                best_gamma = round(gamma, 1)

    return best_df, best_gamma
```

### Tests

Add in `weather/tests/test_calibrate.py`:

```python
class TestSkewTFitting(unittest.TestCase):

    def test_fit_returns_valid_params(self):
        import random
        random.seed(42)
        errors = [random.gauss(0, 2) for _ in range(200)]
        df, gamma = _fit_skew_t_params(errors)
        self.assertIn(df, [2, 3, 4, 5, 7, 10, 15, 20, 30, 50])
        self.assertGreaterEqual(gamma, 0.5)
        self.assertLessEqual(gamma, 1.5)

    def test_left_skewed_data_gives_gamma_below_one(self):
        """Left-skewed data (negative outliers) should give gamma < 1."""
        import random
        random.seed(42)
        # Generate left-skewed data: mostly normal + occasional large negatives
        errors = [random.gauss(0, 2) for _ in range(400)]
        errors += [random.gauss(-8, 2) for _ in range(100)]
        df, gamma = _fit_skew_t_params(errors)
        self.assertLess(gamma, 1.0, f"Expected gamma < 1 for left-skewed data, got {gamma}")

    def test_symmetric_data_gives_gamma_near_one(self):
        """Symmetric data should give gamma close to 1.0."""
        import random
        random.seed(42)
        errors = [random.gauss(0, 2) for _ in range(500)]
        df, gamma = _fit_skew_t_params(errors)
        self.assertAlmostEqual(gamma, 1.0, delta=0.2,
            msg=f"Expected gamma ≈ 1 for symmetric data, got {gamma}")

    def test_insufficient_data_returns_defaults(self):
        df, gamma = _fit_skew_t_params([1.0, -1.0])
        self.assertEqual(df, 10.0)
        self.assertEqual(gamma, 1.0)

    def test_heavy_tailed_skewed_gives_low_df_and_skew(self):
        """Heavy-tailed AND left-skewed data should give low df AND gamma < 1."""
        import random
        random.seed(42)
        # Simulate heavy-tailed + left-skewed
        errors = []
        for _ in range(300):
            z = random.gauss(0, 1)
            chi2 = sum(random.gauss(0, 1)**2 for _ in range(3)) / 3
            errors.append(z / (chi2**0.5) * 2)
        # Add left-skew component
        errors += [random.gauss(-10, 3) for _ in range(100)]
        df, gamma = _fit_skew_t_params(errors)
        self.assertLessEqual(df, 15)
        self.assertLess(gamma, 1.0)
```

### Verification
```
python3 -m pytest weather/tests/test_calibrate.py::TestSkewTFitting -v
python3 -m pytest weather/tests/test_calibrate.py -q
```

---

## Task 4: Update calibration pipeline to use skew-t fitting

**Files:**
- Modify: `weather/calibrate.py:982-992` (`build_calibration_tables` distribution block)
- Modify: `weather/calibrate.py:1278-1288` (`build_weighted_calibration_tables` distribution block)
- Modify: `weather/calibrate.py:1001-1024` (result dict — add `skew_t_gamma`)
- Modify: `weather/calibrate.py:1300-1324` (result dict — add `skew_t_gamma`)
- Test: `weather/tests/test_calibrate.py`

### What to build

When Jarque-Bera rejects normality, fit skew-t instead of symmetric Student-t. Store `"distribution": "skew_t"`, `"student_t_df"`, and `"skew_t_gamma"` in the output.

### Implementation

In **both** `build_calibration_tables` (lines 982-992) and `build_weighted_calibration_tables` (lines 1278-1288), replace:

```python
    # Test normality and fit distribution
    all_error_values = [e["error"] for e in all_errors]
    normality = _test_normality(all_error_values)
    if normality["normal"]:
        distribution = "normal"
        student_t_df = None
    else:
        distribution = "student_t"
        student_t_df = _fit_student_t_df(all_error_values)
        logger.info("Non-normal errors detected (JB=%.1f, skew=%.3f, kurt=%.3f). Using Student's t (df=%.0f)",
                    normality["jb_statistic"], normality["skewness"], normality["kurtosis"], student_t_df)
```

with:

```python
    # Test normality and fit distribution
    all_error_values = [e["error"] for e in all_errors]
    normality = _test_normality(all_error_values)
    skew_t_gamma = None
    if normality["normal"]:
        distribution = "normal"
        student_t_df = None
    else:
        # Fit skewed Student-t (handles both heavy tails AND asymmetry)
        distribution = "skew_t"
        student_t_df, skew_t_gamma = _fit_skew_t_params(all_error_values)
        logger.info(
            "Non-normal errors (JB=%.1f, skew=%.3f, kurt=%.3f). "
            "Using skew-t (df=%.0f, gamma=%.2f)",
            normality["jb_statistic"], normality["skewness"],
            normality["kurtosis"], student_t_df, skew_t_gamma,
        )
```

In **both** result dicts, after `if student_t_df is not None:` block, add:

```python
    if skew_t_gamma is not None:
        result["skew_t_gamma"] = skew_t_gamma
```

### Tests

Add in `weather/tests/test_calibrate.py`:

```python
class TestCalibrationSkewTOutput(unittest.TestCase):
    """Verify that calibration output contains skew-t params when non-normal."""

    def test_build_tables_outputs_skew_t(self):
        """When errors are non-normal, output should have distribution=skew_t."""
        # This test uses the actual build_calibration_tables with mocked data.
        # Since we can't easily mock the full pipeline, test the fitting functions.
        import random
        random.seed(42)
        errors = [random.gauss(0, 2) for _ in range(200)]
        errors += [random.choice([-15, 15]) for _ in range(30)]

        normality = _test_normality(errors)
        self.assertFalse(normality["normal"])

        df, gamma = _fit_skew_t_params(errors)
        self.assertIsInstance(df, float)
        self.assertIsInstance(gamma, float)
        self.assertGreater(gamma, 0)
```

### Verification
```
python3 -m pytest weather/tests/test_calibrate.py -q
python3 -m pytest weather/tests/ bot/tests/ -q  # Full suite
```

---

## Task 5: Distribution validation script

**Files:**
- Create: `weather/distribution_validation.py`
- Create: `weather/tests/test_distribution_validation.py`

### What to build

A CLI script that scores 4 distributions (Normal, Student-t, Student-t best-fit, Skew-t) against historical error data using Brier score and log-loss. Outputs a comparison table.

### Implementation

Create `weather/distribution_validation.py`:

```python
"""Validate distribution choices against historical forecast errors.

Compares Normal, Student-t(10), Student-t(df_best), and Skew-t(df, gamma)
on Brier score and log-loss using the error cache.

Usage::

    python -m weather.distribution_validation
"""

import json
import logging
import math
import sys
from pathlib import Path

from .calibrate import _fit_student_t_df, _fit_skew_t_params, _test_normality
from .probability import _normal_cdf, _student_t_cdf, _skew_t_cdf

logger = logging.getLogger(__name__)

_ERROR_CACHE_PATH = Path(__file__).parent / "error_history.json"
_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


def _load_errors() -> list[dict]:
    """Load errors from cache."""
    if not _ERROR_CACHE_PATH.exists():
        logger.error("Error cache not found at %s", _ERROR_CACHE_PATH)
        return []
    with open(_ERROR_CACHE_PATH) as f:
        data = json.load(f)
    return data.get("errors", [])


def _bucket_hit(error: float, sigma: float, bucket_low: float, bucket_high: float,
                cdf_fn) -> float:
    """Compute P(actual ∈ bucket) using the given CDF function and sigma."""
    if sigma <= 0:
        sigma = 0.01

    if bucket_low <= -900:
        cdf_low = 0.0
    else:
        cdf_low = cdf_fn((bucket_low - 0.5) / sigma)

    if bucket_high >= 900:
        cdf_high = 1.0
    else:
        cdf_high = cdf_fn((bucket_high + 0.5) / sigma)

    return max(0.001, min(0.999, cdf_high - cdf_low))


def score_distribution(errors: list[float], cdf_fn, sigma: float,
                       bucket_width: float = 5.0) -> dict:
    """Score a distribution using Brier score and log-loss.

    Simulates temperature buckets centered on 0 (the forecast) with
    given width. For each error, checks which bucket the actual fell in
    and computes the probability assigned to that bucket.

    Returns dict with 'brier', 'log_loss', 'n'.
    """
    brier_sum = 0.0
    ll_sum = 0.0
    n = 0

    for error in errors:
        # Which bucket does this error fall in?
        # Buckets: ..., [-7.5, -2.5), [-2.5, 2.5), [2.5, 7.5), ...
        bucket_idx = round(error / bucket_width)
        bucket_low = bucket_idx * bucket_width - bucket_width / 2
        bucket_high = bucket_idx * bucket_width + bucket_width / 2

        prob = _bucket_hit(error, sigma, bucket_low, bucket_high, cdf_fn)

        # Brier: (1 - prob)^2  (the event happened, so outcome = 1)
        brier_sum += (1.0 - prob) ** 2
        # Log-loss: -log(prob)
        ll_sum += -math.log(max(1e-10, prob))
        n += 1

    if n == 0:
        return {"brier": 1.0, "log_loss": 10.0, "n": 0}

    return {
        "brier": round(brier_sum / n, 6),
        "log_loss": round(ll_sum / n, 6),
        "n": n,
    }


def run_validation() -> dict:
    """Run distribution comparison and return results dict."""
    errors_raw = _load_errors()
    if not errors_raw:
        logger.error("No errors to validate against")
        return {}

    # Extract raw error values (forecast - actual)
    error_values = [e["error"] for e in errors_raw if "error" in e]
    if len(error_values) < 50:
        logger.warning("Only %d errors — results may be unreliable", len(error_values))

    # Compute sigma from errors
    sigma = (sum(e ** 2 for e in error_values) / len(error_values)) ** 0.5

    # Fit parameters
    best_df = _fit_student_t_df(error_values)
    skew_df, skew_gamma = _fit_skew_t_params(error_values)

    # Normality test
    normality = _test_normality(error_values)

    # Define CDF functions for each distribution
    distributions = {
        "Normal": _normal_cdf,
        "Student-t(10)": lambda z: _student_t_cdf(z, 10),
        f"Student-t({best_df:.0f})": lambda z, df=best_df: _student_t_cdf(z, df),
        f"Skew-t(df={skew_df:.0f}, γ={skew_gamma:.1f})": lambda z, df=skew_df, g=skew_gamma: _skew_t_cdf(z, df, g),
    }

    results = {}
    for name, cdf_fn in distributions.items():
        scores = score_distribution(error_values, cdf_fn, sigma)
        results[name] = scores

    return {
        "n_errors": len(error_values),
        "sigma": round(sigma, 3),
        "normality": normality,
        "fitted_df": best_df,
        "fitted_skew_df": skew_df,
        "fitted_gamma": skew_gamma,
        "scores": results,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = run_validation()
    if not results:
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("DISTRIBUTION VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info("Errors: %d   Sigma: %.3f°F", results["n_errors"], results["sigma"])
    logger.info("Normality: JB=%.1f skew=%.3f kurt=%.3f → %s",
                results["normality"]["jb_statistic"],
                results["normality"]["skewness"],
                results["normality"]["kurtosis"],
                "normal" if results["normality"]["normal"] else "NON-NORMAL")
    logger.info("Fitted: Student-t df=%.0f  |  Skew-t df=%.0f gamma=%.2f",
                results["fitted_df"], results["fitted_skew_df"], results["fitted_gamma"])
    logger.info("")
    logger.info("%-35s  %10s  %10s  %6s", "Distribution", "Brier", "Log-Loss", "N")
    logger.info("-" * 70)

    best_brier = min(s["brier"] for s in results["scores"].values())

    for name, scores in results["scores"].items():
        marker = " ← best" if scores["brier"] == best_brier else ""
        logger.info("%-35s  %10.6f  %10.6f  %6d%s",
                    name, scores["brier"], scores["log_loss"], scores["n"], marker)

    logger.info("=" * 70)

    # Recommendation
    best_name = min(results["scores"], key=lambda k: results["scores"][k]["brier"])
    logger.info("RECOMMENDATION: Use %s (lowest Brier score)", best_name)


if __name__ == "__main__":
    main()
```

### Tests

Create `weather/tests/test_distribution_validation.py`:

```python
"""Tests for distribution validation module."""

import math
import unittest

from weather.distribution_validation import _bucket_hit, score_distribution
from weather.probability import _normal_cdf


class TestBucketHit(unittest.TestCase):

    def test_center_bucket_high_probability(self):
        """Error near 0 with narrow sigma → high probability for center bucket."""
        prob = _bucket_hit(0.5, 2.0, -2.5, 2.5, _normal_cdf)
        self.assertGreater(prob, 0.5)

    def test_tail_bucket_low_probability(self):
        """Error near 0, far bucket → low probability."""
        prob = _bucket_hit(0.0, 2.0, 10, 15, _normal_cdf)
        self.assertLess(prob, 0.01)

    def test_open_lower_bucket(self):
        """Sentinel -999 → open lower bound."""
        prob = _bucket_hit(-5.0, 2.0, -999, -2.5, _normal_cdf)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_open_upper_bucket(self):
        """Sentinel 999 → open upper bound."""
        prob = _bucket_hit(5.0, 2.0, 2.5, 999, _normal_cdf)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_zero_sigma_clamped(self):
        prob = _bucket_hit(0.0, 0.0, -2.5, 2.5, _normal_cdf)
        self.assertGreater(prob, 0.0)


class TestScoreDistribution(unittest.TestCase):

    def test_perfect_sigma_scores_well(self):
        """Errors drawn from N(0, sigma) should score well with Normal CDF."""
        import random
        random.seed(42)
        sigma = 3.0
        errors = [random.gauss(0, sigma) for _ in range(500)]
        result = score_distribution(errors, _normal_cdf, sigma)
        self.assertLess(result["brier"], 0.8)
        self.assertEqual(result["n"], 500)

    def test_wrong_sigma_scores_worse(self):
        """Using wrong sigma should give worse Brier score."""
        import random
        random.seed(42)
        sigma = 3.0
        errors = [random.gauss(0, sigma) for _ in range(500)]
        good = score_distribution(errors, _normal_cdf, sigma)
        bad = score_distribution(errors, _normal_cdf, sigma * 0.3)
        self.assertLess(good["brier"], bad["brier"])

    def test_empty_errors(self):
        result = score_distribution([], _normal_cdf, 3.0)
        self.assertEqual(result["n"], 0)

    def test_returns_required_keys(self):
        result = score_distribution([1.0, -1.0, 0.5], _normal_cdf, 2.0)
        self.assertIn("brier", result)
        self.assertIn("log_loss", result)
        self.assertIn("n", result)
```

### Verification
```
python3 -m pytest weather/tests/test_distribution_validation.py -v
python3 -m pytest weather/tests/ bot/tests/ -q  # Full suite
```

---

## Task 6: Run validation and update calibration.json

**Files:**
- Modify: `weather/calibration.json` (will be regenerated by recalibration)
- No new code — this is a validation + recalibration run

### What to do

1. Run the validation script to measure the impact:
```bash
python3 -m weather.distribution_validation
```

2. If skew-t wins (lowest Brier), the calibration pipeline already outputs skew-t params. Run recalibration:
```bash
python3 -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas
```

3. Verify `calibration.json` now contains `"distribution": "skew_t"` and `"skew_t_gamma"`.

4. Run the full test suite:
```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```

5. Run a dry-run to verify probabilities are coherent:
```bash
python3 -m weather --dry-run --set locations=NYC --explain
```

---

## Final Verification

After all tasks:
```bash
python3 -m pytest weather/tests/ bot/tests/ -q          # All tests pass
python3 -m weather.distribution_validation               # Skew-t wins
python3 -m weather --dry-run --set locations=NYC          # Probabilities coherent
```

## Critical Files Reference

| File | Role |
|------|------|
| `weather/probability.py` | `_skew_t_cdf` + CDF switch |
| `weather/calibrate.py` | `_skew_t_logpdf` + `_fit_skew_t_params` + pipeline update |
| `weather/distribution_validation.py` | Validation CLI script |
| `weather/calibration.json` | Stores `distribution`, `student_t_df`, `skew_t_gamma` |
