# Auto-Recalibration Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a weekly cron-driven recalibration pipeline that updates `calibration.json` from a 90-day sliding window of METAR errors with exponential weighting and parameter guard rails.

**Architecture:** A standalone `weather/recalibrate.py` script reads an incremental error cache (`error_history.json`), fetches only new days from METAR/Open-Meteo, recomputes all calibration params with exponential decay weighting (half-life 30d), clamps params within guard rails, writes `calibration.json` atomically (hot-reloaded by the bot), and logs each run for audit.

**Tech Stack:** Python 3.14 stdlib (json, math, os, tempfile, argparse, logging, datetime). Reuses `weather/historical.py` for METAR/Open-Meteo fetching and `weather/calibrate.py` computation functions (with new weighted variants).

---

### Task 1: Incremental Error Cache

Build `weather/error_cache.py` — manages `error_history.json` with load, append, prune, and incremental fetch.

**Files:**
- Create: `weather/error_cache.py`
- Create: `weather/tests/test_error_cache.py`
- Modify: `.gitignore` — add `weather/error_history.json` and `weather/recalibration_log/`

**Context:** The error cache stores computed forecast errors as a JSON file. Each record has the same format as `compute_forecast_errors()` output in `weather/calibrate.py:94-109`. The cache tracks `last_fetched` per location so subsequent runs only fetch new days. Records older than 365 days are pruned.

**Step 1: Write tests for error cache**

Create `weather/tests/test_error_cache.py`:

```python
"""Tests for weather.error_cache — incremental error history."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from weather.error_cache import (
    load_error_cache,
    save_error_cache,
    prune_old_errors,
    fetch_new_errors,
    CACHE_FORMAT_VERSION,
)


class TestLoadSaveCache(unittest.TestCase):

    def test_load_missing_file_returns_empty(self):
        cache = load_error_cache("/nonexistent/path.json")
        self.assertEqual(cache["errors"], [])
        self.assertEqual(cache["last_fetched"], {})
        self.assertEqual(cache["version"], CACHE_FORMAT_VERSION)

    def test_round_trip(self):
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [
                {"location": "NYC", "target_date": "2026-01-15", "month": 1,
                 "metric": "high", "model": "gfs",
                 "forecast": 42.0, "actual": 43.2, "error": -1.2,
                 "model_spread": 1.5},
            ],
            "last_fetched": {"NYC": "2026-01-15"},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            save_error_cache(cache, tmp)
            loaded = load_error_cache(tmp)
            self.assertEqual(len(loaded["errors"]), 1)
            self.assertEqual(loaded["last_fetched"]["NYC"], "2026-01-15")
        finally:
            os.unlink(tmp)

    def test_load_corrupt_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json{{{")
            tmp = f.name
        try:
            cache = load_error_cache(tmp)
            self.assertEqual(cache["errors"], [])
        finally:
            os.unlink(tmp)


class TestPruneOldErrors(unittest.TestCase):

    def test_prunes_old_keeps_recent(self):
        today = datetime.now().strftime("%Y-%m-%d")
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [
                {"target_date": today, "location": "NYC"},
                {"target_date": old_date, "location": "NYC"},
            ],
            "last_fetched": {},
        }
        pruned = prune_old_errors(cache, max_age_days=365)
        self.assertEqual(len(pruned["errors"]), 1)
        self.assertEqual(pruned["errors"][0]["target_date"], today)

    def test_empty_cache_no_crash(self):
        cache = {"version": CACHE_FORMAT_VERSION, "errors": [], "last_fetched": {}}
        pruned = prune_old_errors(cache)
        self.assertEqual(pruned["errors"], [])


class TestFetchNewErrors(unittest.TestCase):

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_fetches_from_last_fetched_plus_one(self, mock_compute):
        mock_compute.return_value = [
            {"location": "NYC", "target_date": "2026-02-12", "month": 2,
             "metric": "high", "model": "gfs",
             "forecast": 40.0, "actual": 41.0, "error": -1.0,
             "model_spread": 1.0},
        ]
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {"NYC": "2026-02-11"},
        }
        updated = fetch_new_errors(cache, locations=["NYC"],
                                   reference_date="2026-02-14")
        # Should have called with start_date="2026-02-12", end_date="2026-02-12"
        call_args = mock_compute.call_args
        self.assertEqual(call_args.kwargs["start_date"], "2026-02-12")
        self.assertEqual(call_args.kwargs["end_date"], "2026-02-12")
        self.assertEqual(len(updated["errors"]), 1)
        self.assertEqual(updated["last_fetched"]["NYC"], "2026-02-12")

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_bootstrap_fetches_90_days(self, mock_compute):
        mock_compute.return_value = []
        cache = {"version": CACHE_FORMAT_VERSION, "errors": [], "last_fetched": {}}
        fetch_new_errors(cache, locations=["NYC"], reference_date="2026-02-14")
        call_args = mock_compute.call_args
        # Bootstrap: should go back ~90 days
        self.assertEqual(call_args.kwargs["start_date"], "2025-11-16")

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_skips_location_when_up_to_date(self, mock_compute):
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {"NYC": "2026-02-12"},  # today-2 = 2026-02-12
        }
        fetch_new_errors(cache, locations=["NYC"], reference_date="2026-02-14")
        mock_compute.assert_not_called()

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_unknown_location_skipped(self, mock_compute):
        cache = {"version": CACHE_FORMAT_VERSION, "errors": [], "last_fetched": {}}
        fetch_new_errors(cache, locations=["UNKNOWN"], reference_date="2026-02-14")
        mock_compute.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_error_cache.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Implement error cache module**

Create `weather/error_cache.py`:

```python
"""Incremental error history cache for auto-recalibration.

Stores forecast errors in ``error_history.json`` with per-location
``last_fetched`` tracking so only new days need to be fetched.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .calibrate import _compute_errors_with_metar
from .config import LOCATIONS

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1
_DEFAULT_CACHE_PATH = str(Path(__file__).parent / "error_history.json")
_BOOTSTRAP_DAYS = 90
_BUFFER_DAYS = 2  # Skip last 2 days (METAR may be incomplete)


def load_error_cache(path: str = _DEFAULT_CACHE_PATH) -> dict:
    """Load error cache from disk. Returns empty structure if missing/corrupt."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "errors" in data:
            data.setdefault("version", CACHE_FORMAT_VERSION)
            data.setdefault("last_fetched", {})
            return data
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        pass
    return {"version": CACHE_FORMAT_VERSION, "errors": [], "last_fetched": {}}


def save_error_cache(cache: dict, path: str = _DEFAULT_CACHE_PATH) -> None:
    """Write error cache atomically (tmpfile + rename)."""
    import os
    import tempfile

    cache["version"] = CACHE_FORMAT_VERSION
    dir_path = str(Path(path).parent)
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_old_errors(cache: dict, max_age_days: int = 365) -> dict:
    """Remove error records older than max_age_days."""
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    cache["errors"] = [e for e in cache["errors"] if e.get("target_date", "") >= cutoff]
    return cache


def fetch_new_errors(
    cache: dict,
    locations: list[str],
    reference_date: str | None = None,
) -> dict:
    """Fetch new errors for each location since last_fetched.

    Args:
        cache: Current error cache dict.
        locations: Location keys to fetch.
        reference_date: Override "today" for testing (YYYY-MM-DD).

    Returns:
        Updated cache with new errors appended and last_fetched updated.
    """
    if reference_date:
        today = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        today = datetime.now()

    end_date = (today - timedelta(days=_BUFFER_DAYS)).strftime("%Y-%m-%d")

    for loc in locations:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.warning("Unknown location %s — skipping", loc)
            continue

        station = loc_data.get("station")
        if not station:
            logger.warning("No METAR station for %s — skipping", loc)
            continue

        last = cache["last_fetched"].get(loc)
        if last:
            start_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            start_date = start_dt.strftime("%Y-%m-%d")
        else:
            # Bootstrap: go back BOOTSTRAP_DAYS from end_date
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=_BOOTSTRAP_DAYS)
            start_date = start_dt.strftime("%Y-%m-%d")

        if start_date > end_date:
            logger.info("%s already up to date (last_fetched=%s)", loc, last)
            continue

        logger.info("Fetching errors for %s: %s to %s", loc, start_date, end_date)
        new_errors = _compute_errors_with_metar(
            location=loc,
            lat=loc_data["lat"],
            lon=loc_data["lon"],
            station=station,
            start_date=start_date,
            end_date=end_date,
            tz_name=loc_data.get("tz", "America/New_York"),
        )

        cache["errors"].extend(new_errors)

        # Update last_fetched to the latest target_date we got data for
        if new_errors:
            max_date = max(e["target_date"] for e in new_errors)
            cache["last_fetched"][loc] = max(end_date, max_date)
        else:
            cache["last_fetched"][loc] = end_date

    return cache
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_error_cache.py -v`
Expected: all PASS

**Step 5: Update .gitignore**

Add to `.gitignore`:
```
weather/error_history.json
weather/recalibration_log/
```

**Step 6: Commit**

```bash
git add weather/error_cache.py weather/tests/test_error_cache.py .gitignore
git commit -m "feat: add incremental error cache for auto-recalibration"
```

---

### Task 2: Weighted Calibration Functions

Add exponentially-weighted variants of the calibration computations in `weather/calibrate.py`. These reuse the existing logic but accept sample weights.

**Files:**
- Modify: `weather/calibrate.py` — add `_weighted_base_sigma()`, `_weighted_model_weights()`, `_weighted_seasonal_factors()`, `_weighted_platt()`, `_weighted_adaptive_factors()`, and `build_weighted_calibration_tables()`
- Modify: `weather/tests/test_calibrate.py` — add `TestWeightedCalibration`

**Context:** The existing `_compute_base_sigma()` (line 344), `compute_model_weights()` (line 274), `_compute_adaptive_factors()` (line 372), and `_fit_platt_from_errors()` (line 482) operate on unweighted lists. We add new functions that take a `weights` dict mapping `target_date` to weight. The weighting formula is `w(t) = exp(-ln(2) * age_days / half_life)`.

**Step 1: Write tests for weighted calibration**

Add to `weather/tests/test_calibrate.py`:

```python
class TestWeightedCalibration(unittest.TestCase):

    def _make_dated_errors(self, n_days=60, base_error=2.0):
        """Generate errors with dates spread over n_days ending today."""
        from datetime import datetime, timedelta
        errors = []
        today = datetime.now()
        for day_offset in range(n_days):
            date = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            month = (today - timedelta(days=day_offset)).month
            for metric in ["high", "low"]:
                for model in ["gfs", "ecmwf"]:
                    err = base_error * (1 if model == "gfs" else -0.8)
                    errors.append({
                        "location": "NYC",
                        "target_date": date,
                        "month": month,
                        "metric": metric,
                        "model": model,
                        "forecast": 50.0 + err,
                        "actual": 50.0,
                        "error": err,
                        "model_spread": abs(base_error * 0.5),
                    })
        return errors

    def test_compute_weights_recent_higher(self):
        from weather.calibrate import compute_exponential_weights
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        weights = compute_exponential_weights([today, old], half_life=30)
        self.assertGreater(weights[today], weights[old])
        self.assertAlmostEqual(weights[today], 1.0, places=1)

    def test_weighted_base_sigma_differs_from_unweighted(self):
        from weather.calibrate import _weighted_base_sigma, _compute_base_sigma
        errors = self._make_dated_errors(n_days=60)
        from weather.calibrate import compute_exponential_weights
        dates = list({e["target_date"] for e in errors})
        weights = compute_exponential_weights(dates, half_life=30)
        sigma_w = _weighted_base_sigma(errors, weights)
        sigma_u = _compute_base_sigma(errors)
        # Both should be positive and in reasonable range
        self.assertGreater(sigma_w, 0)
        self.assertGreater(sigma_u, 0)
        # They may differ slightly due to weighting
        self.assertAlmostEqual(sigma_w, sigma_u, delta=1.0)

    def test_build_weighted_tables_has_all_keys(self):
        from weather.calibrate import build_weighted_calibration_tables
        errors = self._make_dated_errors(n_days=30)
        result = build_weighted_calibration_tables(errors, ["NYC"], half_life=30)
        self.assertIn("global_sigma", result)
        self.assertIn("location_sigma", result)
        self.assertIn("seasonal_factors", result)
        self.assertIn("model_weights", result)
        self.assertIn("adaptive_sigma", result)
        self.assertIn("platt_scaling", result)
        self.assertIn("metadata", result)

    def test_effective_samples_in_metadata(self):
        from weather.calibrate import build_weighted_calibration_tables
        errors = self._make_dated_errors(n_days=30)
        result = build_weighted_calibration_tables(errors, ["NYC"], half_life=30)
        self.assertIn("samples_effective", result["metadata"])
        self.assertGreater(result["metadata"]["samples_effective"], 0)
        self.assertLessEqual(result["metadata"]["samples_effective"],
                             result["metadata"]["samples"])
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_calibrate.py::TestWeightedCalibration -v`
Expected: FAIL (ImportError — functions don't exist yet)

**Step 3: Implement weighted calibration functions**

Add to `weather/calibrate.py` (after the existing `build_calibration_tables` function, before `main()`):

```python
# ---------------------------------------------------------------------------
# Exponential weighting for auto-recalibration
# ---------------------------------------------------------------------------

def compute_exponential_weights(
    dates: list[str],
    half_life: float = 30.0,
    reference_date: str | None = None,
) -> dict[str, float]:
    """Compute exponential decay weights for a list of dates.

    w(t) = exp(-ln(2) * age_days / half_life)

    Args:
        dates: List of date strings (YYYY-MM-DD).
        half_life: Half-life in days (default 30).
        reference_date: "Today" for age computation (default: actual today).

    Returns:
        Dict mapping date string to weight in (0, 1].
    """
    if reference_date:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()

    weights: dict[str, float] = {}
    ln2 = math.log(2)
    for d in dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            age = max(0, (ref - dt).days)
            weights[d] = math.exp(-ln2 * age / half_life)
        except ValueError:
            weights[d] = 0.0
    return weights


def _weighted_base_sigma(
    errors: list[dict],
    weights: dict[str, float],
) -> float:
    """Compute weighted base sigma: sqrt(sum(w * error^2) / sum(w))."""
    sum_w = 0.0
    sum_w_err_sq = 0.0
    for e in errors:
        w = weights.get(e.get("target_date", ""), 1.0)
        sum_w += w
        sum_w_err_sq += w * e["error"] ** 2
    if sum_w < 1e-9:
        return 1.5  # Fallback
    return math.sqrt(sum_w_err_sq / sum_w)


def _weighted_model_weights(
    errors: list[dict],
    weights: dict[str, float],
    group_by: str = "location",
) -> dict[str, dict[str, float]]:
    """Compute model weights using weighted RMSE (inverse-RMSE weighting)."""
    from collections import defaultdict

    # Group: (group_key, model) -> list of (w, forecast, actual)
    paired: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for err in errors:
        group_key = str(err.get(group_by, "global"))
        model = err["model"]
        w = weights.get(err.get("target_date", ""), 1.0)
        paired[group_key][model].append((w, err["forecast"], err["actual"]))

    results: dict[str, dict[str, float]] = {}
    model_map = {"gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025"}

    for group_key, model_data in paired.items():
        model_rmse: dict[str, float] = {}
        for model, triples in model_data.items():
            sum_w = sum(t[0] for t in triples)
            if sum_w < 1e-9:
                continue
            wmse = sum(t[0] * (t[1] - t[2]) ** 2 for t in triples) / sum_w
            model_rmse[model] = math.sqrt(wmse)

        if not model_rmse:
            continue

        inv_rmse = {m: 1.0 / r for m, r in model_rmse.items() if r > 0}
        total_inv = sum(inv_rmse.values())
        if total_inv == 0:
            continue

        weights_dict: dict[str, float] = {}
        for model, inv in inv_rmse.items():
            canonical = model_map.get(model, model)
            weights_dict[canonical] = round(inv / total_inv, 3)

        noaa_weight = 0.20
        remaining = 1.0 - noaa_weight
        for k in weights_dict:
            weights_dict[k] = round(weights_dict[k] * remaining, 3)
        weights_dict["noaa"] = noaa_weight

        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: round(v / total, 3) for k, v in weights_dict.items()}

        results[group_key] = weights_dict

    return results


def build_weighted_calibration_tables(
    all_errors: list[dict],
    locations: list[str],
    half_life: float = 30.0,
    reference_date: str | None = None,
) -> dict:
    """Build calibration tables with exponential decay weighting.

    Same output format as ``build_calibration_tables`` but weights
    recent errors more heavily.
    """
    dates = list({e["target_date"] for e in all_errors})
    w = compute_exponential_weights(dates, half_life=half_life,
                                    reference_date=reference_date)

    # Effective sample count (sum of weights)
    samples_effective = sum(w.get(e.get("target_date", ""), 1.0) for e in all_errors)

    # Weighted global sigma
    global_base = _weighted_base_sigma(all_errors, w)
    global_sigma = _expand_sigma_by_horizon(global_base)

    # Weighted seasonal factors
    monthly_errors: dict[str, list[dict]] = defaultdict(list)
    for e in all_errors:
        monthly_errors[str(e.get("month", 0))].append(e)

    monthly_sigma: dict[str, float] = {}
    for month_key, errs in monthly_errors.items():
        s = _weighted_base_sigma(errs, w)
        if s > 0:
            monthly_sigma[month_key] = round(s, 2)

    if monthly_sigma:
        # Weighted mean sigma for normalization
        mean_sigma = sum(monthly_sigma.values()) / len(monthly_sigma)
        seasonal_factors = {
            m: round(s / mean_sigma, 3) if mean_sigma > 0 else 1.0
            for m, s in monthly_sigma.items()
        }
    else:
        seasonal_factors = {}

    # Per-location sigma
    location_sigma: dict[str, dict] = {}
    location_seasonal: dict[str, dict] = {}
    for loc in locations:
        loc_errors = [e for e in all_errors if e["location"] == loc]
        if not loc_errors:
            continue
        loc_base = _weighted_base_sigma(loc_errors, w)
        location_sigma[loc] = _expand_sigma_by_horizon(loc_base)

        loc_monthly: dict[str, list[dict]] = defaultdict(list)
        for e in loc_errors:
            loc_monthly[str(e.get("month", 0))].append(e)
        loc_monthly_sigma = {}
        for mk, errs in loc_monthly.items():
            s = _weighted_base_sigma(errs, w)
            if s > 0:
                loc_monthly_sigma[mk] = round(s, 2)
        if loc_monthly_sigma:
            loc_mean = sum(loc_monthly_sigma.values()) / len(loc_monthly_sigma)
            location_seasonal[loc] = {
                m: round(s / loc_mean, 3) if loc_mean > 0 else 1.0
                for m, s in loc_monthly_sigma.items()
            }

    # Weighted model weights
    model_weights = _weighted_model_weights(all_errors, w, group_by="location")

    # Adaptive factors + Platt (reuse existing — they already work on the
    # error list; weighting these further adds minimal value for 90-day window)
    adaptive_factors = _compute_adaptive_factors(all_errors)
    platt_params = _fit_platt_from_errors(all_errors)

    mean_spread = _compute_mean_model_spread(all_errors)
    error_dates = [e["target_date"] for e in all_errors]
    date_range = [min(error_dates), max(error_dates)] if error_dates else []

    return {
        "global_sigma": global_sigma,
        "location_sigma": location_sigma,
        "seasonal_factors": seasonal_factors,
        "location_seasonal": location_seasonal,
        "model_weights": model_weights,
        "adaptive_sigma": adaptive_factors,
        "platt_scaling": platt_params,
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "samples": len(all_errors),
            "samples_effective": round(samples_effective, 1),
            "date_range": date_range,
            "locations": locations,
            "base_sigma_global": round(global_base, 2),
            "mean_model_spread": round(mean_spread, 2),
            "horizon_growth_model": "NWP linear: sigma(h) = base * growth(h)",
            "weighting": f"exponential half-life {half_life}d",
        },
    }
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_calibrate.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add weather/calibrate.py weather/tests/test_calibrate.py
git commit -m "feat: add weighted calibration functions for auto-recalibration"
```

---

### Task 3: Guard Rails (Parameter Clamping)

Add parameter clamp constants and a `clamp_calibration()` function that enforces bounds on all calibration parameters.

**Files:**
- Create: `weather/guard_rails.py`
- Create: `weather/tests/test_guard_rails.py`

**Context:** After `build_weighted_calibration_tables()` produces a calibration dict, `clamp_calibration()` enforces physical bounds on every parameter. If any value hits a clamp, it's recorded in a `clamped` list returned alongside the result.

**Step 1: Write tests**

Create `weather/tests/test_guard_rails.py`:

```python
"""Tests for weather.guard_rails — parameter clamping."""

import unittest

from weather.guard_rails import clamp_calibration, PARAM_BOUNDS


class TestClampCalibration(unittest.TestCase):

    def _make_calibration(self, **overrides):
        """Build a minimal valid calibration dict with optional overrides."""
        cal = {
            "global_sigma": {"0": 1.84, "5": 4.91, "10": 11.03},
            "location_sigma": {"NYC": {"0": 2.38, "5": 6.35, "10": 14.27}},
            "seasonal_factors": {"1": 1.08, "7": 0.93},
            "location_seasonal": {"NYC": {"1": 1.01, "7": 0.83}},
            "model_weights": {"NYC": {"gfs_seamless": 0.51, "ecmwf_ifs025": 0.29, "noaa": 0.20}},
            "adaptive_sigma": {
                "underdispersion_factor": 1.3,
                "spread_to_sigma_factor": 0.78,
                "ema_to_sigma_factor": 1.04,
                "samples": 500,
            },
            "platt_scaling": {"a": 0.73, "b": 0.32},
            "metadata": {"base_sigma_global": 1.84},
        }
        for key, value in overrides.items():
            # Support dotted keys like "platt_scaling.a"
            parts = key.split(".")
            target = cal
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = value
        return cal

    def test_valid_calibration_unchanged(self):
        cal = self._make_calibration()
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped_list, [])
        self.assertAlmostEqual(clamped_cal["metadata"]["base_sigma_global"], 1.84)

    def test_base_sigma_too_low(self):
        cal = self._make_calibration(**{"metadata.base_sigma_global": 0.3})
        cal["global_sigma"]["0"] = 0.3
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertGreaterEqual(clamped_cal["metadata"]["base_sigma_global"], PARAM_BOUNDS["base_sigma"][0])
        self.assertGreater(len(clamped_list), 0)

    def test_base_sigma_too_high(self):
        cal = self._make_calibration(**{"metadata.base_sigma_global": 5.0})
        cal["global_sigma"]["0"] = 5.0
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertLessEqual(clamped_cal["metadata"]["base_sigma_global"], PARAM_BOUNDS["base_sigma"][1])

    def test_platt_a_clamped(self):
        cal = self._make_calibration(**{"platt_scaling.a": 3.0})
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertLessEqual(clamped_cal["platt_scaling"]["a"], PARAM_BOUNDS["platt_a"][1])
        self.assertTrue(any("platt_a" in c["param"] for c in clamped_list))

    def test_platt_b_clamped(self):
        cal = self._make_calibration(**{"platt_scaling.b": -2.0})
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertGreaterEqual(clamped_cal["platt_scaling"]["b"], PARAM_BOUNDS["platt_b"][0])

    def test_seasonal_factor_clamped(self):
        cal = self._make_calibration()
        cal["seasonal_factors"]["1"] = 3.0  # Way too high
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertLessEqual(clamped_cal["seasonal_factors"]["1"], PARAM_BOUNDS["seasonal_factor"][1])

    def test_model_weight_clamped(self):
        cal = self._make_calibration()
        cal["model_weights"]["NYC"]["gfs_seamless"] = 0.75  # Too high
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertLessEqual(clamped_cal["model_weights"]["NYC"]["gfs_seamless"],
                             PARAM_BOUNDS["model_weight"][1])

    def test_spread_to_sigma_clamped(self):
        cal = self._make_calibration(**{"adaptive_sigma.spread_to_sigma_factor": 0.1})
        clamped_cal, clamped_list = clamp_calibration(cal)
        self.assertGreaterEqual(clamped_cal["adaptive_sigma"]["spread_to_sigma_factor"],
                                PARAM_BOUNDS["spread_to_sigma"][0])
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_guard_rails.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement guard_rails.py**

Create `weather/guard_rails.py`:

```python
"""Parameter guard rails for calibration — prevents drift into unreasonable values."""

import logging

logger = logging.getLogger(__name__)

# (min, max) bounds for each parameter
PARAM_BOUNDS = {
    "base_sigma": (1.0, 4.0),
    "seasonal_factor": (0.5, 2.0),
    "model_weight": (0.15, 0.70),
    "platt_a": (0.3, 2.0),
    "platt_b": (-1.0, 1.0),
    "spread_to_sigma": (0.3, 1.5),
    "ema_to_sigma": (0.5, 2.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_calibration(cal: dict) -> tuple[dict, list[dict]]:
    """Apply guard rails to all calibration parameters.

    Returns:
        (clamped_calibration, list_of_clamped_entries)
        Each clamped entry: {"param": name, "original": value, "clamped": value}
    """
    clamped_list: list[dict] = []

    def _record(param: str, original: float, clamped: float):
        if original != clamped:
            clamped_list.append({
                "param": param,
                "original": round(original, 4),
                "clamped": round(clamped, 4),
            })
            logger.warning("Guard rail: %s = %.4f clamped to %.4f", param, original, clamped)

    # base_sigma (in metadata and global_sigma["0"])
    lo, hi = PARAM_BOUNDS["base_sigma"]
    base = cal.get("metadata", {}).get("base_sigma_global", 1.84)
    clamped_base = _clamp(base, lo, hi)
    _record("base_sigma", base, clamped_base)
    if base != clamped_base:
        cal["metadata"]["base_sigma_global"] = clamped_base
        # Rescale global_sigma proportionally
        ratio = clamped_base / base if base > 0 else 1.0
        for h in cal.get("global_sigma", {}):
            cal["global_sigma"][h] = round(cal["global_sigma"][h] * ratio, 2)

    # seasonal_factors
    lo, hi = PARAM_BOUNDS["seasonal_factor"]
    for month, factor in cal.get("seasonal_factors", {}).items():
        c = _clamp(factor, lo, hi)
        _record(f"seasonal_factor[{month}]", factor, c)
        cal["seasonal_factors"][month] = c

    for loc, loc_s in cal.get("location_seasonal", {}).items():
        for month, factor in loc_s.items():
            c = _clamp(factor, lo, hi)
            _record(f"location_seasonal[{loc}][{month}]", factor, c)
            loc_s[month] = c

    # model_weights (only GFS/ECMWF, not NOAA which is fixed)
    lo, hi = PARAM_BOUNDS["model_weight"]
    for loc, weights in cal.get("model_weights", {}).items():
        for model in ["gfs_seamless", "ecmwf_ifs025"]:
            if model in weights:
                w = weights[model]
                c = _clamp(w, lo, hi)
                _record(f"model_weight[{loc}][{model}]", w, c)
                weights[model] = c

    # platt_scaling
    platt = cal.get("platt_scaling", {})
    if "a" in platt:
        lo, hi = PARAM_BOUNDS["platt_a"]
        c = _clamp(platt["a"], lo, hi)
        _record("platt_a", platt["a"], c)
        platt["a"] = c
    if "b" in platt:
        lo, hi = PARAM_BOUNDS["platt_b"]
        c = _clamp(platt["b"], lo, hi)
        _record("platt_b", platt["b"], c)
        platt["b"] = c

    # adaptive_sigma
    adaptive = cal.get("adaptive_sigma", {})
    if "spread_to_sigma_factor" in adaptive:
        lo, hi = PARAM_BOUNDS["spread_to_sigma"]
        v = adaptive["spread_to_sigma_factor"]
        c = _clamp(v, lo, hi)
        _record("spread_to_sigma", v, c)
        adaptive["spread_to_sigma_factor"] = c
    if "ema_to_sigma_factor" in adaptive:
        lo, hi = PARAM_BOUNDS["ema_to_sigma"]
        v = adaptive["ema_to_sigma_factor"]
        c = _clamp(v, lo, hi)
        _record("ema_to_sigma", v, c)
        adaptive["ema_to_sigma_factor"] = c

    return cal, clamped_list
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_guard_rails.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add weather/guard_rails.py weather/tests/test_guard_rails.py
git commit -m "feat: add parameter guard rails for calibration clamping"
```

---

### Task 4: Recalibrate Orchestrator

Build `weather/recalibrate.py` — the CLI entry point that ties together error cache, weighted calibration, guard rails, atomic write, and audit logging.

**Files:**
- Create: `weather/recalibrate.py`
- Create: `weather/tests/test_recalibrate.py`

**Context:** This is the main script invoked by cron: `python -m weather.recalibrate --locations NYC,Chicago,...`. It loads the error cache, fetches new errors, filters to a 90-day window, computes weighted calibration, applies guard rails, writes `calibration.json` atomically, and saves a recalibration log. If fewer than 100 effective weighted samples, it aborts without writing.

**Step 1: Write tests**

Create `weather/tests/test_recalibrate.py`:

```python
"""Tests for weather.recalibrate — orchestrator."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from weather.recalibrate import (
    filter_window,
    run_recalibration,
    _compute_delta,
    MIN_EFFECTIVE_SAMPLES,
)


class TestFilterWindow(unittest.TestCase):

    def test_filters_to_window(self):
        errors = [
            {"target_date": "2026-02-10"},
            {"target_date": "2025-10-01"},  # > 90 days ago from 2026-02-14
            {"target_date": "2026-01-15"},
        ]
        filtered = filter_window(errors, window_days=90,
                                 reference_date="2026-02-14")
        dates = [e["target_date"] for e in filtered]
        self.assertIn("2026-02-10", dates)
        self.assertIn("2026-01-15", dates)
        self.assertNotIn("2025-10-01", dates)

    def test_empty_errors(self):
        self.assertEqual(filter_window([], window_days=90), [])


class TestComputeDelta(unittest.TestCase):

    def test_delta_between_calibrations(self):
        old = {"global_sigma": {"0": 1.84}, "platt_scaling": {"a": 0.73, "b": 0.32}}
        new = {"global_sigma": {"0": 1.90}, "platt_scaling": {"a": 0.75, "b": 0.30}}
        delta = _compute_delta(old, new)
        self.assertIn("base_sigma", delta)
        self.assertAlmostEqual(delta["base_sigma"], 0.06, places=2)

    def test_delta_no_old(self):
        new = {"global_sigma": {"0": 1.90}, "platt_scaling": {"a": 0.75, "b": 0.30}}
        delta = _compute_delta({}, new)
        self.assertEqual(delta, {})


class TestRunRecalibration(unittest.TestCase):

    def _make_errors(self, n_days=30):
        """Generate plausible error records for testing."""
        errors = []
        today = datetime.now()
        for day_offset in range(n_days):
            date = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            month = (today - timedelta(days=day_offset)).month
            for metric in ["high", "low"]:
                for model in ["gfs", "ecmwf"]:
                    errors.append({
                        "location": "NYC",
                        "target_date": date,
                        "month": month,
                        "metric": metric,
                        "model": model,
                        "forecast": 50.0 + (2.0 if model == "gfs" else -1.6),
                        "actual": 50.0,
                        "error": 2.0 if model == "gfs" else -1.6,
                        "model_spread": 1.5,
                    })
        return errors

    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.load_error_cache")
    @patch("weather.recalibrate.save_error_cache")
    def test_successful_recalibration(self, mock_save_cache, mock_load, mock_fetch):
        errors = self._make_errors(n_days=30)
        cache = {"version": 1, "errors": errors, "last_fetched": {"NYC": "2026-02-12"}}
        mock_load.return_value = cache
        mock_fetch.return_value = cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = os.path.join(tmpdir, "calibration.json")
            log_dir = os.path.join(tmpdir, "recalibration_log")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=cal_path,
                log_dir=log_dir,
            )

            self.assertTrue(result["success"])
            self.assertTrue(os.path.exists(cal_path))
            self.assertTrue(os.path.isdir(log_dir))

            with open(cal_path) as f:
                cal = json.load(f)
            self.assertIn("global_sigma", cal)

    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.load_error_cache")
    @patch("weather.recalibrate.save_error_cache")
    def test_insufficient_samples_aborts(self, mock_save_cache, mock_load, mock_fetch):
        # Only 2 days of data — way below MIN_EFFECTIVE_SAMPLES
        errors = self._make_errors(n_days=2)
        cache = {"version": 1, "errors": errors, "last_fetched": {}}
        mock_load.return_value = cache
        mock_fetch.return_value = cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = os.path.join(tmpdir, "calibration.json")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=cal_path,
                log_dir=os.path.join(tmpdir, "log"),
            )

            self.assertFalse(result["success"])
            self.assertFalse(os.path.exists(cal_path))
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_recalibrate.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement recalibrate.py**

Create `weather/recalibrate.py`:

```python
"""Auto-recalibration pipeline — weekly cron entry point.

Usage::

    python -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas

Loads incremental error cache, fetches new METAR errors, recomputes
all calibration parameters with exponential weighting on a 90-day
sliding window, applies guard rails, and writes calibration.json
atomically.
"""

import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .calibrate import build_weighted_calibration_tables
from .error_cache import (
    load_error_cache,
    save_error_cache,
    prune_old_errors,
    fetch_new_errors,
    _DEFAULT_CACHE_PATH,
)
from .guard_rails import clamp_calibration

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = str(Path(__file__).parent / "calibration.json")
_DEFAULT_LOG_DIR = str(Path(__file__).parent / "recalibration_log")

MIN_EFFECTIVE_SAMPLES = 100
WINDOW_DAYS = 90
HALF_LIFE = 30.0


def filter_window(
    errors: list[dict],
    window_days: int = WINDOW_DAYS,
    reference_date: str | None = None,
) -> list[dict]:
    """Keep only errors within the sliding window."""
    if reference_date:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()
    cutoff = (ref - timedelta(days=window_days)).strftime("%Y-%m-%d")
    return [e for e in errors if e.get("target_date", "") >= cutoff]


def _compute_delta(old_cal: dict, new_cal: dict) -> dict:
    """Compute parameter deltas between old and new calibration."""
    if not old_cal:
        return {}
    delta = {}
    old_base = old_cal.get("global_sigma", {}).get("0", 0)
    new_base = new_cal.get("global_sigma", {}).get("0", 0)
    if old_base and new_base:
        delta["base_sigma"] = round(new_base - old_base, 4)

    old_platt = old_cal.get("platt_scaling", {})
    new_platt = new_cal.get("platt_scaling", {})
    if old_platt.get("a") and new_platt.get("a"):
        delta["platt_a"] = round(new_platt["a"] - old_platt["a"], 4)
    if old_platt.get("b") is not None and new_platt.get("b") is not None:
        delta["platt_b"] = round(new_platt["b"] - old_platt["b"], 4)

    return delta


def _load_existing_calibration(path: str) -> dict:
    """Load existing calibration.json for delta computation."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_atomic(data: dict, path: str) -> None:
    """Write JSON atomically via tmpfile + rename."""
    dir_path = str(Path(path).parent)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_log(log_entry: dict, log_dir: str) -> None:
    """Save recalibration log entry to log_dir/YYYY-MM-DD.json."""
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(log_dir, f"{date_str}.json")
    with open(path, "w") as f:
        json.dump(log_entry, f, indent=2)
    logger.info("Recalibration log saved to %s", path)


def run_recalibration(
    locations: list[str],
    cache_path: str = _DEFAULT_CACHE_PATH,
    output_path: str = _DEFAULT_OUTPUT,
    log_dir: str = _DEFAULT_LOG_DIR,
    reference_date: str | None = None,
) -> dict:
    """Run the full recalibration pipeline.

    Returns dict with ``success``, ``samples``, ``samples_effective``, etc.
    """
    t0 = time.monotonic()

    # 1. Load and update error cache
    cache = load_error_cache(cache_path)
    cache = fetch_new_errors(cache, locations, reference_date=reference_date)
    cache = prune_old_errors(cache)
    save_error_cache(cache, cache_path)

    # 2. Filter to window
    windowed = filter_window(cache["errors"], WINDOW_DAYS, reference_date)
    logger.info("Windowed errors: %d (from %d total cached)", len(windowed), len(cache["errors"]))

    if not windowed:
        logger.warning("No errors in window — aborting recalibration")
        return {"success": False, "reason": "no_data", "samples": 0}

    # 3. Build weighted calibration
    calibration = build_weighted_calibration_tables(
        windowed, locations, half_life=HALF_LIFE, reference_date=reference_date,
    )

    # 4. Check minimum samples
    effective = calibration["metadata"].get("samples_effective", 0)
    if effective < MIN_EFFECTIVE_SAMPLES:
        logger.warning(
            "Insufficient effective samples (%.1f < %d) — aborting",
            effective, MIN_EFFECTIVE_SAMPLES,
        )
        return {
            "success": False,
            "reason": "insufficient_samples",
            "samples": len(windowed),
            "samples_effective": effective,
        }

    # 5. Apply guard rails
    calibration, clamped = clamp_calibration(calibration)

    # 6. Compute delta from previous calibration
    old_cal = _load_existing_calibration(output_path)
    delta = _compute_delta(old_cal, calibration)

    # 7. Write calibration.json atomically
    _write_atomic(calibration, output_path)
    logger.info("Calibration written to %s", output_path)

    # 8. Save recalibration log
    elapsed = time.monotonic() - t0
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start": windowed[-1]["target_date"] if windowed else "",
            "end": windowed[0]["target_date"] if windowed else "",
        },
        "samples_total": len(windowed),
        "samples_effective": effective,
        "params": {
            "base_sigma": calibration["metadata"]["base_sigma_global"],
            "platt": calibration["platt_scaling"],
            "model_weights": calibration["model_weights"],
            "seasonal_factors": calibration["seasonal_factors"],
            "adaptive_sigma": {
                k: v for k, v in calibration["adaptive_sigma"].items()
                if k != "samples"
            },
        },
        "clamped": clamped,
        "delta_from_previous": delta,
        "fetch_stats": {
            "total_cached_errors": len(cache["errors"]),
            "elapsed_s": round(elapsed, 1),
        },
    }
    _save_log(log_entry, log_dir)

    return {
        "success": True,
        "samples": len(windowed),
        "samples_effective": effective,
        "clamped_count": len(clamped),
        "delta": delta,
    }


def main() -> None:
    """CLI entry point for auto-recalibration."""
    parser = argparse.ArgumentParser(
        description="Auto-recalibrate weather model from recent METAR data",
    )
    parser.add_argument(
        "--locations", type=str,
        default="NYC,Chicago,Miami,Seattle,Atlanta,Dallas",
        help="Comma-separated location keys",
    )
    parser.add_argument(
        "--cache", type=str, default=_DEFAULT_CACHE_PATH,
        help=f"Error cache path (default: {_DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=_DEFAULT_OUTPUT,
        help=f"Calibration output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-dir", type=str, default=_DEFAULT_LOG_DIR,
        help=f"Recalibration log directory (default: {_DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loc_keys = [l.strip() for l in args.locations.split(",")]
    result = run_recalibration(
        locations=loc_keys,
        cache_path=args.cache,
        output_path=args.output,
        log_dir=args.log_dir,
    )

    if result["success"]:
        logger.info(
            "Recalibration complete: %d samples (%.1f effective), %d clamped",
            result["samples"], result["samples_effective"], result["clamped_count"],
        )
        if result.get("delta"):
            logger.info("Delta from previous: %s", result["delta"])
    else:
        logger.warning("Recalibration aborted: %s", result.get("reason", "unknown"))


if __name__ == "__main__":
    main()
```

**Step 4: Add `__main__` support**

Create `weather/recalibrate.py` already has `if __name__ == "__main__"` but Python needs `__main__.py` routing or the module needs to be invocable. The `if __name__` block at the bottom handles `python -m weather.recalibrate` since Python will look for `weather/recalibrate.py` directly when invoked as a module.

**Step 5: Run tests**

Run: `python3 -m pytest weather/tests/test_recalibrate.py -v`
Expected: all PASS

**Step 6: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: all pass (including existing tests unaffected)

**Step 7: Commit**

```bash
git add weather/recalibrate.py weather/tests/test_recalibrate.py
git commit -m "feat: add recalibration orchestrator with guard rails and logging"
```

---

### Task 5: Integration Test & .gitignore

Verify the full pipeline works end-to-end with mocked API calls, and finalize .gitignore.

**Files:**
- Modify: `.gitignore`
- No new files — this is a verification/cleanup task

**Step 1: Run the full test suite**

```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```

Expected: all pass

**Step 2: Verify CLI help works**

```bash
python3 -m weather.recalibrate --help
```

Expected: Shows help with `--locations`, `--cache`, `--output`, `--log-dir` options.

**Step 3: Verify .gitignore entries**

Ensure `.gitignore` contains:
```
weather/error_history.json
weather/recalibration_log/
```

**Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: add error_history and recalibration_log to gitignore"
```

---

Plan complete and saved to `docs/plans/2026-02-16-auto-recalibration-plan.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
