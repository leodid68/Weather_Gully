# Remaining Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 remaining improvements: numerical robustness tests, Open-Meteo TTL cache, inter-location correlation matrix, and enriched CLI report.

**Architecture:** Each task is independent and can be implemented in any order. Tasks 1-2 are isolated additions. Task 3 touches calibrate→probability→strategy→config. Task 4 enriches the existing report module.

**Tech Stack:** Python 3.14 stdlib only (no new dependencies). Unittest for tests.

---

### Task 1: Numerical Robustness Tests for Beta & Student's t

**Files:**
- Modify: `weather/tests/test_probability.py:543-589` (add to existing edge case section)
- Possibly modify: `weather/probability.py:118-206` (fix any bugs found)

**Step 1: Write the failing tests**

Add a new class `TestNumericalRobustnessExtended` after the existing `TestStudentTCDFEdgeCases` class (line 590) in `weather/tests/test_probability.py`:

```python
class TestNumericalRobustnessExtended(unittest.TestCase):
    """Extended numerical robustness tests for beta function and Student's t."""

    def test_beta_a_near_zero(self):
        """Beta with a≈0 should not crash."""
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1e-10, 1.0)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_beta_b_near_zero(self):
        """Beta with b≈0 should not crash."""
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1.0, 1e-10)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_beta_x_near_zero(self):
        """Beta with x≈0 should return ≈0."""
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1e-300, 5.0, 5.0)
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_beta_x_near_one(self):
        """Beta with x≈1 should return ≈1."""
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1.0 - 1e-15, 5.0, 5.0)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_beta_both_params_tiny(self):
        """Beta with both a,b near zero should not crash."""
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1e-8, 1e-8)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_student_t_df_half(self):
        """Student's t with df=0.5 at x=0 should return 0.5."""
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0.0, 0.5)
        self.assertAlmostEqual(result, 0.5, places=3)

    def test_student_t_df_1000_converges_to_normal(self):
        """Student's t with df=1000 should converge to normal CDF."""
        from weather.probability import _student_t_cdf, _normal_cdf
        # At z=1.96, normal CDF ≈ 0.975
        result = _student_t_cdf(1.96, 1000)
        normal = _normal_cdf(1.96)
        self.assertAlmostEqual(result, normal, places=3)

    def test_student_t_extreme_1e10(self):
        """Student's t at x=1e10 should be ≈1.0 without crash."""
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1e10, 10)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_student_t_extreme_neg_1e10(self):
        """Student's t at x=-1e10 should be ≈0.0 without crash."""
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(-1e10, 10)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_beta_non_convergence_logs_warning(self):
        """Continued fraction that doesn't converge should log a warning."""
        from weather.probability import _regularized_incomplete_beta
        # Very small max_iter to force non-convergence
        result = _regularized_incomplete_beta(0.3, 5.0, 5.0, max_iter=1)
        # Should still return a number (not crash), even if inaccurate
        self.assertTrue(0.0 <= result <= 1.0)

    def test_bucket_probability_with_extreme_forecast(self):
        """Bucket probability should handle extreme forecast without crash."""
        from weather.probability import estimate_bucket_probability
        # Forecast way outside any bucket
        prob = estimate_bucket_probability(200.0, 40, 44, "2026-02-20", apply_seasonal=False, horizon_override=0)
        self.assertAlmostEqual(prob, 0.0, places=2)

    def test_bucket_probability_with_zero_sigma(self):
        """Bucket probability with sigma_override=0 should not crash."""
        from weather.probability import estimate_bucket_probability
        prob = estimate_bucket_probability(42.0, 40, 44, "2026-02-20", sigma_override=0.0)
        # sigma clamped to 0.01, so should return a valid probability
        self.assertTrue(0.0 <= prob <= 1.0)
```

**Step 2: Run tests to verify they pass (or find bugs)**

Run: `python3 -m pytest weather/tests/test_probability.py::TestNumericalRobustnessExtended -v`
Expected: Most should PASS since guards already exist. If any FAIL, fix in next step.

**Step 3: Fix any discovered bugs**

If `test_beta_x_near_zero` fails because `1e-300` underflows in `math.log()`, add a guard in `_regularized_incomplete_beta` at line 142 of `probability.py`:

```python
    try:
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    except (OverflowError, ValueError):
        return 0.0 if x < 0.5 else 1.0
```

**Step 4: Run full test suite**

Run: `python3 -m pytest weather/tests/test_probability.py -q`
Expected: All pass.

**Step 5: Commit**

```bash
git add weather/tests/test_probability.py weather/probability.py
git commit -m "test: add extended numerical robustness tests for beta/Student's t"
```

---

### Task 2: TTL Cache for Open-Meteo

**Files:**
- Modify: `weather/open_meteo.py:1-18` (add cache module vars) and `:172-289` (integrate into `get_open_meteo_forecast_multi`)
- Modify: `weather/tests/test_open_meteo.py` (add cache tests)

**Step 1: Write the failing tests**

Add a new class at the end of `weather/tests/test_open_meteo.py`:

```python
class TestForecastCache(unittest.TestCase):
    """Tests for TTL cache in get_open_meteo_forecast_multi."""

    def setUp(self):
        """Reset cache before each test."""
        import weather.open_meteo as om
        om._forecast_cache.clear()

    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        from weather.open_meteo import _cache_key
        k1 = _cache_key("40.7,-73.8", "America/New_York")
        k2 = _cache_key("40.7,-73.8", "America/New_York")
        self.assertEqual(k1, k2)

    def test_cache_key_different_coords(self):
        """Different coordinates produce different keys."""
        from weather.open_meteo import _cache_key
        k1 = _cache_key("40.7,-73.8", "America/New_York")
        k2 = _cache_key("41.9,-87.9", "America/Chicago")
        self.assertNotEqual(k1, k2)

    @patch("weather.open_meteo._fetch_json")
    def test_cache_hit_skips_fetch(self, mock_fetch):
        """Second call with same args should use cache, not refetch."""
        from weather.open_meteo import get_open_meteo_forecast_multi
        import weather.open_meteo as om

        mock_data = {"daily": {
            "time": ["2026-02-16"],
            "temperature_2m_max_gfs_seamless": [50.0],
            "temperature_2m_min_gfs_seamless": [30.0],
            "temperature_2m_max_ecmwf_ifs025": [52.0],
            "temperature_2m_min_ecmwf_ifs025": [28.0],
        }}
        mock_fetch.return_value = mock_data

        locs = {"NYC": {"lat": 40.7, "lon": -73.8, "tz": "America/New_York"}}
        result1 = get_open_meteo_forecast_multi(locs)
        result2 = get_open_meteo_forecast_multi(locs)

        # _fetch_json should only be called once (second call hits cache)
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertEqual(result1, result2)

    @patch("weather.open_meteo.time")
    @patch("weather.open_meteo._fetch_json")
    def test_cache_expires_after_ttl(self, mock_fetch, mock_time):
        """Cache entry should expire after TTL seconds."""
        from weather.open_meteo import get_open_meteo_forecast_multi, _CACHE_TTL
        import weather.open_meteo as om

        mock_data = {"daily": {
            "time": ["2026-02-16"],
            "temperature_2m_max_gfs_seamless": [50.0],
            "temperature_2m_min_gfs_seamless": [30.0],
            "temperature_2m_max_ecmwf_ifs025": [52.0],
            "temperature_2m_min_ecmwf_ifs025": [28.0],
        }}
        mock_fetch.return_value = mock_data

        # First call at t=1000
        mock_time.time.return_value = 1000.0
        mock_time.sleep = lambda x: None
        locs = {"NYC": {"lat": 40.7, "lon": -73.8, "tz": "America/New_York"}}
        get_open_meteo_forecast_multi(locs)

        # Second call at t=1000+TTL+1 → expired, should refetch
        mock_time.time.return_value = 1000.0 + _CACHE_TTL + 1
        get_open_meteo_forecast_multi(locs)

        self.assertEqual(mock_fetch.call_count, 2)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_open_meteo.py::TestForecastCache -v`
Expected: FAIL (no `_cache_key`, `_forecast_cache`, `_CACHE_TTL` yet).

**Step 3: Implement the cache**

At the top of `weather/open_meteo.py`, after line 16 (`from ._ssl import SSL_CTX as _SSL_CTX`), add:

```python
_forecast_cache: dict[str, tuple[dict, float]] = {}  # key → (result_data, timestamp)
_CACHE_TTL = 900  # 15 minutes in seconds


def _cache_key(coords: str, tz: str) -> str:
    """Deterministic cache key from coordinates and timezone."""
    return f"{coords}|{tz}"
```

In `get_open_meteo_forecast_multi()`, at line 203 (before `for tz, group in tz_groups.items():`), wrap the fetch in a cache check:

```python
    for tz, group in tz_groups.items():
        lats = ",".join(str(loc["lat"]) for _, loc in group)
        lons = ",".join(str(loc["lon"]) for _, loc in group)
        cache_k = _cache_key(f"{lats},{lons}", tz)

        # Check cache
        now = time.time()
        if cache_k in _forecast_cache:
            cached_entries, cached_at = _forecast_cache[cache_k]
            if now - cached_at < _CACHE_TTL:
                logger.info("Open-Meteo cache hit for tz=%s (%d locations)", tz, len(group))
                for idx, (name, _) in enumerate(group):
                    result[name] = cached_entries.get(name, {})
                continue

        # ... existing fetch logic ...

        # After parsing, store in cache
        group_results = {name: result[name] for name, _ in group if name in result}
        _forecast_cache[cache_k] = (group_results, now)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_open_meteo.py -v`
Expected: All PASS.

**Step 5: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass.

**Step 6: Commit**

```bash
git add weather/open_meteo.py weather/tests/test_open_meteo.py
git commit -m "feat: add TTL cache (15min) for Open-Meteo forecasts"
```

---

### Task 3: Inter-Location Correlation

This is the largest task. It has 4 sub-tasks.

#### Task 3a: Config — add correlation parameters

**Files:**
- Modify: `weather/config.py:116` (add after max_open_positions)

**Step 1: Add config fields**

After line 116 in `weather/config.py` (`max_open_positions: int = 15`):

```python
    # Inter-location correlation (sizing reduction)
    correlation_threshold: float = 0.5    # Ignore correlations below this
    correlation_discount: float = 0.5     # How much to reduce sizing (0=ignore, 1=full)
```

**Step 2: Run existing tests**

Run: `python3 -m pytest weather/tests/test_config.py -q`
Expected: All pass (new fields have defaults).

**Step 3: Commit**

```bash
git add weather/config.py
git commit -m "feat: add correlation_threshold and correlation_discount config"
```

#### Task 3b: Calibration — compute correlation matrix

**Files:**
- Modify: `weather/calibrate.py` (add `_compute_correlation_matrix` function)
- Create: `weather/tests/test_correlation.py`

**Step 1: Write the failing test**

Create `weather/tests/test_correlation.py`:

```python
"""Tests for inter-location forecast error correlation."""

import unittest

from weather.calibrate import _compute_correlation_matrix


class TestCorrelationMatrix(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        """Identical errors → correlation = 1.0."""
        errors = [
            {"location": "NYC", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "NYC", "target_date": "2025-01-16", "month": 1, "error": -1.0},
            {"location": "NYC", "target_date": "2025-01-17", "month": 1, "error": 3.0},
            {"location": "Chicago", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "Chicago", "target_date": "2025-01-16", "month": 1, "error": -1.0},
            {"location": "Chicago", "target_date": "2025-01-17", "month": 1, "error": 3.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"  # sorted alphabetically
        self.assertIn(key, matrix)
        self.assertAlmostEqual(matrix[key]["DJF"], 1.0, places=2)

    def test_zero_correlation(self):
        """Uncorrelated errors → correlation ≈ 0."""
        errors = [
            {"location": "NYC", "target_date": "2025-07-01", "month": 7, "error": 1.0},
            {"location": "NYC", "target_date": "2025-07-02", "month": 7, "error": -1.0},
            {"location": "NYC", "target_date": "2025-07-03", "month": 7, "error": 1.0},
            {"location": "NYC", "target_date": "2025-07-04", "month": 7, "error": -1.0},
            {"location": "Miami", "target_date": "2025-07-01", "month": 7, "error": -1.0},
            {"location": "Miami", "target_date": "2025-07-02", "month": 7, "error": 1.0},
            {"location": "Miami", "target_date": "2025-07-03", "month": 7, "error": -1.0},
            {"location": "Miami", "target_date": "2025-07-04", "month": 7, "error": 1.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Miami"], errors)
        key = "Miami|NYC"
        self.assertIn(key, matrix)
        self.assertAlmostEqual(matrix[key]["JJA"], -1.0, places=2)

    def test_insufficient_data_omitted(self):
        """Pairs with < 5 shared dates in a season should be omitted."""
        errors = [
            {"location": "NYC", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "Chicago", "target_date": "2025-01-15", "month": 1, "error": 1.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"
        # Not enough shared dates for DJF
        self.assertTrue(key not in matrix or "DJF" not in matrix.get(key, {}))

    def test_seasonal_grouping(self):
        """Errors in different seasons produce separate entries."""
        errors = []
        for month, season_months in [(1, range(1, 4)), (7, range(7, 10))]:
            for day in range(1, 11):
                date = f"2025-{month:02d}-{day:02d}"
                errors.append({"location": "NYC", "target_date": date, "month": month, "error": float(day)})
                errors.append({"location": "Chicago", "target_date": date, "month": month, "error": float(day) * 0.9})
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"
        self.assertIn(key, matrix)
        # Both seasons should have entries
        self.assertIn("DJF", matrix[key])
        self.assertIn("JJA", matrix[key])
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest weather/tests/test_correlation.py -v`
Expected: ImportError — `_compute_correlation_matrix` doesn't exist yet.

**Step 3: Implement the function**

Add to `weather/calibrate.py` after `_compute_mean_model_spread` (after line 590):

```python
# Season mapping: month → season code
_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    sx = (sum((x - mx) ** 2 for x in xs) / (n - 1)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / (n - 1)) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _compute_correlation_matrix(
    locations: list[str],
    errors: list[dict],
    min_samples: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute Pearson correlation of forecast errors between location pairs by season.

    Args:
        locations: List of location keys.
        errors: Error records with "location", "target_date", "month", "error".
        min_samples: Minimum shared dates per season to include (default 5).

    Returns:
        {"LocA|LocB": {"DJF": 0.72, "MAM": 0.45, ...}, ...}
        Location pair keys are alphabetically sorted.
    """
    # Group errors: (location, target_date) → error
    by_loc_date: dict[tuple[str, str], float] = {}
    date_months: dict[str, int] = {}
    for e in errors:
        loc = e["location"]
        date = e["target_date"]
        by_loc_date[(loc, date)] = e["error"]
        date_months[date] = e["month"]

    result: dict[str, dict[str, float]] = {}

    for i, loc_a in enumerate(locations):
        for loc_b in locations[i + 1:]:
            pair_key = "|".join(sorted([loc_a, loc_b]))

            # Find shared dates
            dates_a = {d for (l, d) in by_loc_date if l == loc_a}
            dates_b = {d for (l, d) in by_loc_date if l == loc_b}
            shared = dates_a & dates_b

            # Group shared dates by season
            season_dates: dict[str, list[str]] = {}
            for d in shared:
                season = _MONTH_TO_SEASON.get(date_months.get(d, 1), "DJF")
                season_dates.setdefault(season, []).append(d)

            pair_seasons: dict[str, float] = {}
            for season, dates in season_dates.items():
                if len(dates) < min_samples:
                    continue
                xs = [by_loc_date[(loc_a, d)] for d in dates]
                ys = [by_loc_date[(loc_b, d)] for d in dates]
                corr = _pearson_correlation(xs, ys)
                pair_seasons[season] = round(corr, 3)

            if pair_seasons:
                result[pair_key] = pair_seasons

    return result
```

**Step 4: Integrate into `build_calibration_tables` and `build_weighted_calibration_tables`**

In `build_calibration_tables` (line 850), add before the `result = {` dict:

```python
    # Inter-location correlation matrix
    correlation_matrix = _compute_correlation_matrix(locations, all_errors)
```

And add `"correlation_matrix": correlation_matrix,` to the result dict.

Same for `build_weighted_calibration_tables` (line 1122).

**Step 5: Run tests**

Run: `python3 -m pytest weather/tests/test_correlation.py weather/tests/test_calibrate.py -v`
Expected: All pass.

**Step 6: Commit**

```bash
git add weather/calibrate.py weather/tests/test_correlation.py
git commit -m "feat: compute inter-location correlation matrix in calibration"
```

#### Task 3c: Loading — expose `get_correlation` in probability.py

**Files:**
- Modify: `weather/probability.py` (add `get_correlation` function after line 68)
- Modify: `weather/tests/test_probability.py` (add test)

**Step 1: Write the failing test**

Add at end of `weather/tests/test_probability.py`:

```python
class TestGetCorrelation(unittest.TestCase):
    def test_returns_correlation_for_season(self):
        import weather.probability as prob
        cal_data = {
            "correlation_matrix": {
                "Chicago|NYC": {"DJF": 0.72, "JJA": 0.45},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            # January → DJF
            corr = prob.get_correlation("NYC", "Chicago", 1)
            self.assertAlmostEqual(corr, 0.72, places=2)
            # July → JJA
            corr = prob.get_correlation("NYC", "Chicago", 7)
            self.assertAlmostEqual(corr, 0.45, places=2)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)

    def test_returns_zero_for_unknown_pair(self):
        import weather.probability as prob
        cal_data = {"correlation_matrix": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            corr = prob.get_correlation("NYC", "Timbuktu", 1)
            self.assertEqual(corr, 0.0)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest weather/tests/test_probability.py::TestGetCorrelation -v`
Expected: FAIL — `get_correlation` doesn't exist.

**Step 3: Implement `get_correlation`**

Add to `weather/probability.py` after `_load_calibration()` (after line 68):

```python
# Season mapping for correlation lookup
_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def get_correlation(loc1: str, loc2: str, month: int) -> float:
    """Get calibrated correlation between two locations for a given month.

    Returns 0.0 if no calibration data is available for this pair/season.
    """
    cal = _load_calibration()
    matrix = cal.get("correlation_matrix", {})
    pair_key = "|".join(sorted([loc1, loc2]))
    pair_data = matrix.get(pair_key, {})
    season = _MONTH_TO_SEASON.get(month, "DJF")
    return pair_data.get(season, 0.0)
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_probability.py::TestGetCorrelation -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "feat: add get_correlation() for inter-location lookup"
```

#### Task 3d: Strategy — apply correlation discount to sizing

**Files:**
- Modify: `weather/strategy.py:937-944` (adjust sizing)
- Create: `weather/tests/test_correlation_sizing.py`

**Step 1: Write the failing test**

Create `weather/tests/test_correlation_sizing.py`:

```python
"""Tests for correlation-based sizing discount in strategy."""

import unittest
from weather.strategy import _apply_correlation_discount
from weather.config import Config


class TestCorrelationDiscount(unittest.TestCase):

    def test_no_open_positions_no_discount(self):
        """No open trades → no discount."""
        adjusted = _apply_correlation_discount(
            base_size=2.0,
            location="NYC",
            month=1,
            open_locations=[],
            config=Config(correlation_threshold=0.5, correlation_discount=0.5),
        )
        self.assertEqual(adjusted, 2.0)

    def test_correlated_position_reduces_sizing(self):
        """Open position in correlated location → sizing reduced."""
        with unittest.mock.patch("weather.strategy.get_correlation", return_value=0.8):
            adjusted = _apply_correlation_discount(
                base_size=2.0,
                location="NYC",
                month=1,
                open_locations=["Chicago"],
                config=Config(correlation_threshold=0.5, correlation_discount=0.5),
            )
        # 2.0 * (1 - 0.8 * 0.5) = 2.0 * 0.6 = 1.2
        self.assertAlmostEqual(adjusted, 1.2, places=2)

    def test_below_threshold_no_discount(self):
        """Correlation below threshold → no discount."""
        with unittest.mock.patch("weather.strategy.get_correlation", return_value=0.3):
            adjusted = _apply_correlation_discount(
                base_size=2.0,
                location="NYC",
                month=1,
                open_locations=["Miami"],
                config=Config(correlation_threshold=0.5, correlation_discount=0.5),
            )
        self.assertEqual(adjusted, 2.0)

    def test_multiple_correlated_uses_max(self):
        """Multiple correlated positions → use max correlation for discount."""
        def mock_corr(l1, l2, m):
            if sorted([l1, l2]) == ["Chicago", "NYC"]:
                return 0.8
            if sorted([l1, l2]) == ["Dallas", "NYC"]:
                return 0.6
            return 0.0

        with unittest.mock.patch("weather.strategy.get_correlation", side_effect=mock_corr):
            adjusted = _apply_correlation_discount(
                base_size=2.0,
                location="NYC",
                month=1,
                open_locations=["Chicago", "Dallas"],
                config=Config(correlation_threshold=0.5, correlation_discount=0.5),
            )
        # Max corr = 0.8 with Chicago: 2.0 * (1 - 0.8 * 0.5) = 1.2
        self.assertAlmostEqual(adjusted, 1.2, places=2)
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest weather/tests/test_correlation_sizing.py -v`
Expected: FAIL — `_apply_correlation_discount` doesn't exist.

**Step 3: Implement the discount function**

In `weather/strategy.py`, add after line 31 (after `logger = ...`):

```python
from .probability import get_correlation
```

Add a new function before `run_weather_strategy`:

```python
def _apply_correlation_discount(
    base_size: float,
    location: str,
    month: int,
    open_locations: list[str],
    config: Config,
) -> float:
    """Reduce position size when correlated positions are already open.

    Uses the max correlation across all open locations. If it exceeds
    ``config.correlation_threshold``, multiplies sizing by
    ``(1 - max_corr * config.correlation_discount)``.
    """
    if not open_locations:
        return base_size

    max_corr = 0.0
    for open_loc in open_locations:
        if open_loc == location:
            continue
        corr = get_correlation(location, open_loc, month)
        if corr > max_corr:
            max_corr = corr

    if max_corr < config.correlation_threshold:
        return base_size

    factor = 1.0 - max_corr * config.correlation_discount
    adjusted = base_size * max(0.1, factor)  # Floor at 10% of original size
    logger.info("Correlation discount: %s corr=%.2f → sizing %.2f → %.2f",
                location, max_corr, base_size, adjusted)
    return round(adjusted, 2)
```

**Step 4: Integrate into `run_weather_strategy`**

In `run_weather_strategy`, after `position_size = compute_position_size(...)` (line 944), add:

```python
            # Correlation discount: reduce sizing for correlated open positions
            open_locations = list({t.location for t in state.trades.values() if t.location})
            forecast_month = int(date_str.split("-")[1])
            position_size = _apply_correlation_discount(
                position_size, location, forecast_month, open_locations, config,
            )
```

**Step 5: Run tests**

Run: `python3 -m pytest weather/tests/test_correlation_sizing.py weather/tests/test_strategy.py -v`
Expected: All pass.

**Step 6: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass.

**Step 7: Commit**

```bash
git add weather/strategy.py weather/tests/test_correlation_sizing.py
git commit -m "feat: apply correlation discount to sizing for correlated locations"
```

---

### Task 4: Enriched CLI Report

**Files:**
- Modify: `weather/report.py` (rewrite main and add helper functions)
- Modify: `weather/tests/test_report.py` (add tests)

**Step 1: Write the failing tests**

Create or extend `weather/tests/test_report.py`:

```python
"""Tests for enriched CLI report."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from weather.report import (
    _format_pnl,
    _format_position_row,
    format_report,
)


class TestPnlFormatting(unittest.TestCase):

    def test_positive_pnl(self):
        result = _format_pnl(2.15)
        self.assertIn("+$2.15", result)

    def test_negative_pnl(self):
        result = _format_pnl(-1.50)
        self.assertIn("-$1.50", result)

    def test_zero_pnl(self):
        result = _format_pnl(0.0)
        self.assertIn("$0.00", result)


class TestPositionRow(unittest.TestCase):

    def test_basic_row(self):
        row = _format_position_row(
            location="NYC",
            bucket="48-52°F",
            side="YES",
            entry_price=0.35,
            unrealized=0.07,
            days_left=2,
        )
        self.assertIn("NYC", row)
        self.assertIn("48-52°F", row)
        self.assertIn("YES", row)


class TestFormatReport(unittest.TestCase):

    def test_empty_state_no_crash(self):
        """Report should not crash with completely empty state."""
        from weather.state import TradingState
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Weather Gully Report", output)
        self.assertIn("Open Positions", output)

    def test_report_includes_circuit_breaker(self):
        from weather.state import TradingState
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Circuit Breaker", output)
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest weather/tests/test_report.py -v`
Expected: FAIL — `_format_pnl`, `_format_position_row`, `format_report` don't exist.

**Step 3: Implement the enriched report**

Rewrite `weather/report.py`:

```python
"""CLI report -- enriched trading dashboard with positions, P&L, metrics, calibration.

Usage: python3 -m weather.report [--state-file PATH] [--trade-log PATH] [--watch N]
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import brier_score, calibration_table, sharpe_ratio, win_rate, average_edge
from .state import TradingState

# ANSI color codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _format_pnl(value: float) -> str:
    """Format P&L with color: green for positive, red for negative."""
    if value > 0:
        return f"{_GREEN}+${value:.2f}{_RESET}"
    elif value < 0:
        return f"{_RED}-${abs(value):.2f}{_RESET}"
    return f"$0.00"


def _format_position_row(
    location: str,
    bucket: str,
    side: str,
    entry_price: float,
    unrealized: float,
    days_left: int,
) -> str:
    """Format a single position row for the report."""
    pnl_str = _format_pnl(unrealized)
    return f"  {location:<8s} {bucket:<10s} {side:<4s} ${entry_price:.2f}  {pnl_str}  ({days_left}d left)"


def _load_trade_log(path: str = "weather/trade_log.jsonl") -> list[dict]:
    """Load JSONL trade log."""
    entries = []
    p = Path(path)
    if not p.exists():
        return entries
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _get_calibration_age() -> tuple[int | None, str]:
    """Get calibration.json age in days and status string."""
    cal_path = Path(__file__).parent / "calibration.json"
    if not cal_path.exists():
        return None, f"{_RED}MISSING{_RESET}"
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        gen = cal.get("metadata", {}).get("generated") or cal.get("_metadata", {}).get("generated_at")
        if gen:
            gen_dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - gen_dt).days
            if age > 90:
                return age, f"{_RED}{age}d — STALE{_RESET}"
            elif age > 30:
                return age, f"{_YELLOW}{age}d — consider recalibrating{_RESET}"
            return age, f"{_GREEN}{age}d — OK{_RESET}"
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    return None, f"{_YELLOW}unknown{_RESET}"


def format_report(
    state: TradingState,
    trade_log: list[dict] | None = None,
) -> str:
    """Generate the full report as a string."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    lines = []

    lines.append(f"\n{_BOLD}{'=' * 55}{_RESET}")
    lines.append(f"{_BOLD}  Weather Gully Report{_RESET}")
    lines.append(f"  {_DIM}Last update: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}{_RESET}")
    lines.append(f"{_BOLD}{'=' * 55}{_RESET}")

    # ── Open Positions ──
    lines.append(f"\n{_BOLD}── Open Positions ({len(state.trades)}) ──{_RESET}")
    if not state.trades:
        lines.append(f"  {_DIM}No open positions{_RESET}")
    else:
        for mid, trade in state.trades.items():
            days_left = 0
            if trade.forecast_date:
                try:
                    end = datetime.strptime(trade.forecast_date, "%Y-%m-%d").date()
                    days_left = max(0, (end - now.date()).days)
                except ValueError:
                    pass
            lines.append(_format_position_row(
                location=trade.location or "?",
                bucket=trade.outcome_name[:12],
                side=trade.side.upper(),
                entry_price=trade.cost_basis,
                unrealized=0.0,  # No live price available in CLI
                days_left=days_left,
            ))

    # ── Today's P&L ──
    daily_pnl = state.get_daily_pnl(today)
    lines.append(f"\n{_BOLD}── Today's P&L ──{_RESET}")
    lines.append(f"  Realized: {_format_pnl(daily_pnl)}")

    # ── Metrics ──
    resolved = [p for p in state.predictions.values() if p.resolved and p.actual_outcome is not None]
    lines.append(f"\n{_BOLD}── Metrics ({len(resolved)} resolved) ──{_RESET}")
    if resolved:
        preds = [(p.our_probability, p.actual_outcome) for p in resolved]
        bs = brier_score(preds)
        wr = win_rate([1.0 if o else -1.0 for _, o in preds])
        lines.append(f"  Brier: {bs:.4f}" if bs is not None else "  Brier: N/A")
        lines.append(f"  Win rate: {wr:.0%}" if wr is not None else "  Win rate: N/A")

        if trade_log:
            returns = []
            edges = []
            for entry in trade_log:
                prob = entry.get("prob_platt", 0)
                price = entry.get("market_price", 0)
                if price > 0:
                    edges.append(prob - price)
            ae = average_edge(edges)
            if ae is not None:
                lines.append(f"  Avg edge: {ae:.1%}")
    else:
        lines.append(f"  {_DIM}No resolved predictions yet{_RESET}")

    # ── Calibration ──
    age, status = _get_calibration_age()
    lines.append(f"\n{_BOLD}── Calibration ──{_RESET}")
    lines.append(f"  Age: {status}")

    # ── Circuit Breaker ──
    lines.append(f"\n{_BOLD}── Circuit Breaker ──{_RESET}")
    positions_today = state.positions_opened_today(today)
    lines.append(f"  Daily loss: {_format_pnl(daily_pnl)} / $10.00")
    lines.append(f"  Positions today: {positions_today} / 20")
    if state.last_circuit_break:
        lines.append(f"  Last break: {state.last_circuit_break}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Weather Gully trading report")
    parser.add_argument("--state-file", default="weather/paper_state.json")
    parser.add_argument("--trade-log", default="weather/trade_log.jsonl")
    parser.add_argument("--watch", type=int, default=0, metavar="N",
                        help="Refresh every N seconds (0 = once)")
    args = parser.parse_args()

    def _render():
        state = TradingState.load(args.state_file)
        trade_log = _load_trade_log(args.trade_log)
        print(format_report(state, trade_log))

    if args.watch > 0:
        # Graceful Ctrl+C
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            os.system("clear")
            _render()
            time.sleep(args.watch)
    else:
        _render()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_report.py -v`
Expected: All pass.

**Step 5: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass.

**Step 6: Commit**

```bash
git add weather/report.py weather/tests/test_report.py
git commit -m "feat: enriched CLI report with positions, P&L, metrics, calibration, --watch"
```

---

## Final Verification

Run full test suite after all tasks:

```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```

Expected: All tests pass with ~370+ tests.
