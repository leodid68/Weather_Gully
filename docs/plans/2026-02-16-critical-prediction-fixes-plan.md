# Critical Prediction Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fictitious horizon growth model with real data from Open-Meteo Previous Runs API, fix backtest lookahead bias, and implement Student's t-distribution for fat tails.

**Architecture:** New Previous Runs API client provides horizon-dependent forecasts. Calibration computes real RMSE by horizon. Backtest uses horizon-specific forecasts. CDF switches to Student's t if normality is rejected.

**Tech Stack:** Python 3.14 stdlib (urllib, math, json). No new dependencies.

---

### Task 1: Previous Runs API Client

Create `weather/previous_runs.py` — client for Open-Meteo Previous Runs API that fetches hourly forecasts at multiple horizons and computes daily max/min.

**Files:**
- Create: `weather/previous_runs.py`
- Create: `weather/tests/test_previous_runs.py`

**Step 1: Write the client**

```python
"""Open-Meteo Previous Runs API client — horizon-dependent forecasts."""

import json
import logging
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX

logger = logging.getLogger(__name__)

_BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_USER_AGENT = "WeatherGully/1.0"
_CHUNK_DAYS = 90
_MODELS = "gfs_seamless,ecmwf_ifs025"


def _fetch_json(url: str, max_retries: int = 3) -> dict | None:
    """Fetch JSON with retry + backoff."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT})
            with urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                logger.error("Previous Runs API fetch failed: %s", exc)
                return None
    return None


def _hourly_to_daily_max_min(
    times: list[str],
    values: list[float | None],
    tz_offset_hours: int = 0,
) -> dict[str, tuple[float, float]]:
    """Convert hourly temperatures to daily (max, min) using timezone offset.

    Returns: {date_str: (daily_max, daily_min)}
    """
    from collections import defaultdict
    daily: dict[str, list[float]] = defaultdict(list)
    for i, t_str in enumerate(times):
        if values[i] is None:
            continue
        # Parse ISO time and apply timezone offset
        dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
        dt_local = dt + timedelta(hours=tz_offset_hours)
        date_key = dt_local.strftime("%Y-%m-%d")
        daily[date_key].append(values[i])

    result = {}
    for date_key, temps in daily.items():
        if len(temps) >= 12:  # Need at least 12 hours to compute meaningful max/min
            result[date_key] = (round(max(temps), 1), round(min(temps), 1))
    return result


def _tz_offset_for_location(tz_name: str) -> int:
    """Approximate UTC offset in hours for US timezones (no DST — use standard)."""
    offsets = {
        "America/New_York": -5,
        "America/Chicago": -6,
        "America/Denver": -7,
        "America/Los_Angeles": -8,
    }
    return offsets.get(tz_name, -5)


def fetch_previous_runs(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    horizons: list[int] | None = None,
    tz_name: str = "America/New_York",
) -> dict[int, dict[str, dict]]:
    """Fetch forecasts at multiple horizons from Previous Runs API.

    Args:
        lat, lon: Station coordinates.
        start_date, end_date: Date range (YYYY-MM-DD).
        horizons: List of horizon days [0, 1, 2, 3, 5, 7]. Default: [0,1,2,3,5,7].
        tz_name: Timezone name for daily max/min computation.

    Returns:
        {horizon: {date_str: {"gfs_high": float, "gfs_low": float,
                               "ecmwf_high": float, "ecmwf_low": float}}}
    """
    if horizons is None:
        horizons = [0, 1, 2, 3, 5, 7]

    tz_offset = _tz_offset_for_location(tz_name)

    # Build variable list
    hourly_vars = []
    for h in horizons:
        suffix = "" if h == 0 else f"_previous_day{h}"
        hourly_vars.append(f"temperature_2m{suffix}")

    # Chunk the date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    result: dict[int, dict[str, dict]] = {h: {} for h in horizons}

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end_dt)

        url = (
            f"{_BASE_URL}?latitude={lat}&longitude={lon}"
            f"&hourly={','.join(hourly_vars)}"
            f"&models={_MODELS}"
            f"&start_date={chunk_start.strftime('%Y-%m-%d')}"
            f"&end_date={chunk_end.strftime('%Y-%m-%d')}"
            f"&temperature_unit=fahrenheit"
        )

        data = _fetch_json(url)
        if not data or "hourly" not in data:
            logger.warning("No data for chunk %s to %s", chunk_start, chunk_end)
            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(1)
            continue

        hourly = data["hourly"]
        times = hourly.get("time", [])

        for h in horizons:
            suffix = "" if h == 0 else f"_previous_day{h}"

            for model, prefix in [("gfs_seamless", "gfs"), ("ecmwf_ifs025", "ecmwf")]:
                key = f"temperature_2m{suffix}_{model}"
                values = hourly.get(key, [])
                if not values:
                    continue

                daily = _hourly_to_daily_max_min(times, values, tz_offset)
                for date_str, (hi, lo) in daily.items():
                    if date_str not in result[h]:
                        result[h][date_str] = {}
                    result[h][date_str][f"{prefix}_high"] = hi
                    result[h][date_str][f"{prefix}_low"] = lo

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1)  # Rate limiting

    return result
```

**Step 2: Write tests**

```python
"""Tests for weather.previous_runs — Previous Runs API client."""

import unittest
from unittest.mock import patch, MagicMock
from weather.previous_runs import (
    _hourly_to_daily_max_min,
    _tz_offset_for_location,
    fetch_previous_runs,
)


class TestHourlyToDailyMaxMin(unittest.TestCase):

    def test_basic_conversion(self):
        times = [f"2025-06-01T{h:02d}:00" for h in range(24)]
        values = [60.0 + h for h in range(24)]  # 60 to 83
        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertIn("2025-06-01", result)
        self.assertAlmostEqual(result["2025-06-01"][0], 83.0)  # max
        self.assertAlmostEqual(result["2025-06-01"][1], 60.0)  # min

    def test_timezone_offset(self):
        # UTC times that span two local days with -5h offset
        times = [f"2025-06-01T{h:02d}:00" for h in range(24)]
        times += [f"2025-06-02T{h:02d}:00" for h in range(24)]
        values = list(range(48))
        result = _hourly_to_daily_max_min(times, values, tz_offset=-5)
        # Should have entries for local dates
        self.assertTrue(len(result) >= 1)

    def test_nulls_skipped(self):
        times = [f"2025-06-01T{h:02d}:00" for h in range(24)]
        values = [70.0 if h != 12 else None for h in range(24)]
        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertIn("2025-06-01", result)

    def test_insufficient_hours_excluded(self):
        times = [f"2025-06-01T{h:02d}:00" for h in range(6)]
        values = [70.0] * 6
        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        # Only 6 hours — should be excluded (need >= 12)
        self.assertNotIn("2025-06-01", result)


class TestTzOffset(unittest.TestCase):
    def test_known_timezones(self):
        self.assertEqual(_tz_offset_for_location("America/New_York"), -5)
        self.assertEqual(_tz_offset_for_location("America/Chicago"), -6)
        self.assertEqual(_tz_offset_for_location("America/Los_Angeles"), -8)

    def test_unknown_defaults_to_eastern(self):
        self.assertEqual(_tz_offset_for_location("Unknown/Tz"), -5)


class TestFetchPreviousRuns(unittest.TestCase):

    @patch("weather.previous_runs._fetch_json")
    def test_basic_fetch(self, mock_fetch):
        """fetch_previous_runs should return parsed horizon-keyed data."""
        times = [f"2025-06-01T{h:02d}:00" for h in range(24)]
        mock_fetch.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m_gfs_seamless": [70.0 + h * 0.5 for h in range(24)],
                "temperature_2m_ecmwf_ifs025": [69.0 + h * 0.5 for h in range(24)],
                "temperature_2m_previous_day1_gfs_seamless": [68.0 + h * 0.5 for h in range(24)],
                "temperature_2m_previous_day1_ecmwf_ifs025": [67.0 + h * 0.5 for h in range(24)],
            }
        }

        result = fetch_previous_runs(40.77, -73.87, "2025-06-01", "2025-06-01",
                                     horizons=[0, 1], tz_name="America/New_York")
        self.assertIn(0, result)
        self.assertIn(1, result)
        mock_fetch.assert_called_once()

    @patch("weather.previous_runs._fetch_json")
    def test_empty_response(self, mock_fetch):
        mock_fetch.return_value = None
        result = fetch_previous_runs(40.77, -73.87, "2025-06-01", "2025-06-01",
                                     horizons=[0])
        self.assertEqual(result, {0: {}})
```

**Step 3: Run tests**

Run: `python3 -m pytest weather/tests/test_previous_runs.py -v`
Expected: all PASS

**Step 4: Commit**

```bash
git commit -m "feat: add Previous Runs API client for horizon-dependent forecasts"
```

---

### Task 2: Horizon-Real Calibration

Replace the fictitious growth model with real RMSE by horizon in `weather/calibrate.py`.

**Files:**
- Modify: `weather/calibrate.py`
- Modify: `weather/tests/test_calibrate.py`

**Step 1: Add `compute_horizon_errors()` function**

New function that takes Previous Runs data + actuals and computes errors per horizon:

```python
def compute_horizon_errors(
    previous_runs: dict[int, dict[str, dict]],
    actuals: dict[str, dict],
    locations: list[str],
    location_coords: dict[str, tuple[float, float]],
) -> list[dict]:
    """Compute forecast errors at each horizon from Previous Runs data.

    Returns list of error records with 'horizon' field added.
    """
    errors = []
    for horizon, forecasts_by_date in previous_runs.items():
        for date_str, forecast_data in forecasts_by_date.items():
            actual = actuals.get(date_str)
            if not actual:
                continue
            for metric in ["high", "low"]:
                actual_temp = actual.get(metric)
                if actual_temp is None:
                    continue
                for model_key, model_name in [("gfs", "gfs"), ("ecmwf", "ecmwf")]:
                    forecast_temp = forecast_data.get(f"{model_key}_{metric}")
                    if forecast_temp is None:
                        continue
                    error = forecast_temp - actual_temp
                    errors.append({
                        "location": ...,  # from reverse lookup
                        "target_date": date_str,
                        "month": int(date_str.split("-")[1]),
                        "metric": metric,
                        "model": model_name,
                        "horizon": horizon,
                        "forecast": forecast_temp,
                        "actual": actual_temp,
                        "error": error,
                    })
    return errors
```

**Step 2: Replace `_expand_sigma_by_horizon()` with `_compute_sigma_by_horizon()`**

```python
def _compute_sigma_by_horizon(errors: list[dict], max_horizon: int = 10) -> dict[str, float]:
    """Compute real sigma at each horizon from error data.

    For horizons with data: sigma = sqrt(mean(error^2))
    For horizons without data: linear interpolation/extrapolation
    """
    from collections import defaultdict
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for e in errors:
        by_horizon[e["horizon"]].append(e["error"])

    real_sigmas = {}
    for h, errs in sorted(by_horizon.items()):
        if len(errs) >= 20:  # min samples
            sigma = (sum(e**2 for e in errs) / len(errs)) ** 0.5
            real_sigmas[h] = sigma

    # Interpolate/extrapolate for missing horizons 0-10
    result = {}
    known = sorted(real_sigmas.keys())
    for h in range(max_horizon + 1):
        if h in real_sigmas:
            result[str(h)] = round(real_sigmas[h], 2)
        elif known:
            # Linear interpolation between nearest known points
            below = [k for k in known if k < h]
            above = [k for k in known if k > h]
            if below and above:
                k1, k2 = below[-1], above[0]
                t = (h - k1) / (k2 - k1)
                result[str(h)] = round(real_sigmas[k1] + t * (real_sigmas[k2] - real_sigmas[k1]), 2)
            elif below:
                # Extrapolate from last two known
                if len(below) >= 2:
                    k1, k2 = below[-2], below[-1]
                    rate = (real_sigmas[k2] - real_sigmas[k1]) / (k2 - k1)
                    result[str(h)] = round(real_sigmas[k2] + rate * (h - k2), 2)
                else:
                    result[str(h)] = round(real_sigmas[below[-1]] * 1.1, 2)
            elif above:
                result[str(h)] = round(real_sigmas[above[0]] * 0.9, 2)
    return result
```

**Step 3: Wire into `build_calibration_tables()` and `build_weighted_calibration_tables()`**

Add optional `horizon_errors` parameter. When provided, use `_compute_sigma_by_horizon()` instead of `_expand_sigma_by_horizon()`.

**Step 4: Write tests**

```python
class TestComputeSigmaByHorizon(unittest.TestCase):
    def test_real_sigmas_computed(self):
        errors = [{"horizon": 0, "error": e} for e in [1.0, -1.0, 2.0, -2.0] * 10]
        errors += [{"horizon": 3, "error": e} for e in [3.0, -3.0, 4.0, -4.0] * 10]
        result = _compute_sigma_by_horizon(errors)
        self.assertIn("0", result)
        self.assertIn("3", result)
        # Horizon 3 should have larger sigma than horizon 0
        self.assertGreater(float(result["3"]), float(result["0"]))

    def test_interpolation_works(self):
        errors = [{"horizon": 0, "error": e} for e in [1.0, -1.0] * 20]
        errors += [{"horizon": 7, "error": e} for e in [5.0, -5.0] * 20]
        result = _compute_sigma_by_horizon(errors)
        # Horizon 3 should be interpolated between 0 and 7
        self.assertIn("3", result)
        self.assertGreater(float(result["3"]), float(result["0"]))
        self.assertLess(float(result["3"]), float(result["7"]))
```

**Step 5: Run tests**

Run: `python3 -m pytest weather/tests/test_calibrate.py -v`

**Step 6: Commit**

```bash
git commit -m "feat: replace fictitious horizon growth model with real RMSE from Previous Runs API"
```

---

### Task 3: Backtest Without Lookahead

Modify `weather/backtest.py` to use `fetch_previous_runs()` for horizon-specific forecasts.

**Files:**
- Modify: `weather/backtest.py`
- Modify: `weather/tests/test_backtest.py`

**Step 1: Replace forecast source**

In `run_backtest()`, replace:
```python
forecasts = get_historical_forecasts(lat, lon, forecast_start, end_date, tz_name=tz_name)
```

With:
```python
from .previous_runs import fetch_previous_runs
prev_runs = fetch_previous_runs(lat, lon, start_date, end_date,
                                 horizons=[horizon], tz_name=tz_name)
forecasts_by_date = prev_runs.get(horizon, {})
```

Then iterate over `forecasts_by_date` instead of the nested dict structure.

**Step 2: Update the iteration loop**

The current loop is:
```python
for target_date_str, actual_data in actuals.items():
    target_forecasts = forecasts.get(target_date_str, {})
    run_forecasts = target_forecasts.get(target_date_str)
```

Replace with:
```python
for target_date_str, actual_data in actuals.items():
    run_forecasts = forecasts_by_date.get(target_date_str)
```

**Step 3: Update tests**

Update `TestRunBacktest` to mock `fetch_previous_runs` instead of `get_historical_forecasts`:

```python
@patch("weather.backtest.fetch_previous_runs")
@patch("weather.backtest.get_historical_actuals")
def test_basic_backtest(self, mock_actuals, mock_prev_runs):
    mock_prev_runs.return_value = {
        1: {
            "2025-06-04": {
                "gfs_high": 82.0, "gfs_low": 65.0,
                "ecmwf_high": 84.0, "ecmwf_low": 66.0,
            },
        },
    }
    mock_actuals.return_value = {
        "2025-06-04": {"high": 83.0, "low": 64.5},
    }
    result = run_backtest(locations=["NYC"], start_date="2025-06-04",
                          end_date="2025-06-04", horizon=1, entry_threshold=0.0)
    self.assertGreater(len(result.trades), 0)
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_backtest.py -v`

**Step 5: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`

**Step 6: Commit**

```bash
git commit -m "fix: eliminate backtest lookahead bias using Previous Runs API horizon-specific forecasts"
```

---

### Task 4: Distribution Test + Student's t-CDF

Test normality of forecast errors and switch to Student's t-distribution if fat tails detected.

**Files:**
- Modify: `weather/probability.py` — add `_student_t_cdf()`, modify `estimate_bucket_probability()`
- Modify: `weather/calibrate.py` — add normality test, Student's t df fitting
- Modify: `weather/tests/test_probability.py`
- Modify: `weather/tests/test_calibrate.py`

**Step 1: Add Jarque-Bera test to calibrate.py**

```python
def _test_normality(errors: list[float]) -> dict:
    """Jarque-Bera test for normality of forecast errors.

    Returns: {"normal": bool, "jb_statistic": float, "p_approx": str,
              "skewness": float, "kurtosis": float}
    """
    n = len(errors)
    if n < 30:
        return {"normal": True, "jb_statistic": 0, "p_approx": "insufficient_data",
                "skewness": 0, "kurtosis": 3}
    mean = sum(errors) / n
    centered = [e - mean for e in errors]
    m2 = sum(c**2 for c in centered) / n
    m3 = sum(c**3 for c in centered) / n
    m4 = sum(c**4 for c in centered) / n
    sigma = m2 ** 0.5
    if sigma < 1e-10:
        return {"normal": True, "jb_statistic": 0, "p_approx": "zero_variance",
                "skewness": 0, "kurtosis": 3}
    skewness = m3 / (sigma ** 3)
    kurtosis = m4 / (sigma ** 4)
    jb = (n / 6) * (skewness**2 + (kurtosis - 3)**2 / 4)
    # chi-squared critical values: 5.99 (5%), 9.21 (1%)
    normal = jb < 5.99
    p_approx = ">0.05" if normal else ("<0.01" if jb > 9.21 else "<0.05")
    return {"normal": normal, "jb_statistic": round(jb, 2),
            "p_approx": p_approx, "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3)}
```

**Step 2: Add Student's t-CDF to probability.py**

```python
def _student_t_cdf(x: float, df: float) -> float:
    """Student's t CDF using regularized incomplete beta function.

    For df > 100, falls back to normal CDF (negligible difference).
    """
    if df > 100:
        return _normal_cdf(x)
    t2 = x * x
    # Use regularized incomplete beta: I_x(a, b)
    # P(T <= x) = 1 - 0.5 * I_{df/(df+t2)}(df/2, 0.5) if x > 0
    #           = 0.5 * I_{df/(df+t2)}(df/2, 0.5)       if x <= 0
    z = df / (df + t2)
    beta_val = _regularized_beta(z, df / 2, 0.5)
    if x >= 0:
        return 1.0 - 0.5 * beta_val
    else:
        return 0.5 * beta_val


def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    # Lentz's algorithm for continued fraction representation
    ...  # Standard numerical implementation
```

**Step 3: Fit df by MLE in calibrate.py**

```python
def _fit_student_t_df(errors: list[float]) -> float:
    """Fit degrees of freedom for Student's t by maximizing log-likelihood.

    Grid search over candidate df values.
    """
    import math
    sigma = (sum(e**2 for e in errors) / len(errors)) ** 0.5
    standardized = [e / sigma for e in errors]
    best_df = 30
    best_ll = float("-inf")
    for df in [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]:
        ll = sum(_student_t_logpdf(z, df) for z in standardized)
        if ll > best_ll:
            best_ll = ll
            best_df = df
    return best_df
```

**Step 4: Wire into calibration pipeline**

In `build_calibration_tables()`, after computing errors:
1. Run `_test_normality()`
2. If not normal, fit `_fit_student_t_df()`
3. Add `"distribution"` and `"student_t_df"` to calibration.json

In `estimate_bucket_probability()`:
1. Load `distribution` from calibration
2. Use `_student_t_cdf` or `_normal_cdf` accordingly

**Step 5: Write tests**

- Test `_test_normality()` with normal data (should pass) and heavy-tailed data (should reject)
- Test `_student_t_cdf()` against known values (df=1 → Cauchy, df→∞ → Normal)
- Test that `estimate_bucket_probability()` uses the right CDF based on calibration

**Step 6: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`

**Step 7: Commit**

```bash
git commit -m "feat: add Student's t-distribution for fat-tailed forecast errors"
```

---

### Task 5: Integration + Recalibration Update

Wire everything together: update recalibration pipeline and run a full calibration with real horizon data.

**Files:**
- Modify: `weather/recalibrate.py` — use `fetch_previous_runs()` for error computation
- Modify: `weather/error_cache.py` — add horizon field to error records

**Step 1: Update error cache format**

Add `"horizon"` field to error records in `fetch_new_errors()`. Use `fetch_previous_runs()` instead of `get_historical_forecasts()`.

**Step 2: Update recalibration pipeline**

In `run_recalibration()`:
- Fetch Previous Runs data for horizons [0, 1, 2, 3, 5, 7]
- Compute horizon-dependent errors
- Pass to `build_weighted_calibration_tables()` with horizon data
- Run normality test on errors
- Add distribution info to calibration.json

**Step 3: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`

**Step 4: Run calibration**

```bash
python3 -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas
```

Verify:
- calibration.json has real horizon-dependent sigma (not growth-model derived)
- Distribution field present (normal or student_t)
- Log shows sigma values increasing with horizon

**Step 5: Commit**

```bash
git commit -m "feat: integrate horizon-real calibration and Student's t into recalibration pipeline"
```
