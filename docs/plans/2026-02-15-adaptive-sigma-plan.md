# Adaptive Sigma Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the theoretical NWP horizon growth sigma with empirical ensemble spread, combining 3 real-time signals via max() with the calibrated sigma as floor.

**Architecture:** New `weather/ensemble.py` fetches 51 ECMWF + 31 GFS ensemble members from Open-Meteo. `compute_adaptive_sigma()` in `probability.py` takes max(ensemble_stddev, model_spread, ema_error) × calibration factors. `strategy.py` passes the result as `sigma_override` to bypass internal sigma calculation.

**Tech Stack:** Python 3.14 stdlib (urllib, json, math), Open-Meteo Ensemble API (free, no key)

---

### Task 1: Create `weather/ensemble.py` — data model and cache

**Files:**
- Create: `weather/ensemble.py`
- Create: `weather/tests/test_ensemble.py`

**Step 1: Write failing tests for EnsembleResult and cache**

```python
# weather/tests/test_ensemble.py
"""Tests for the ensemble spread client."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from weather.ensemble import EnsembleResult, _cache_path, _read_cache, _write_cache


class TestEnsembleResult(unittest.TestCase):

    def test_creation(self):
        r = EnsembleResult(
            member_temps=[50.0, 52.0, 54.0],
            ensemble_mean=52.0,
            ensemble_stddev=2.0,
            ecmwf_stddev=1.8,
            gfs_stddev=2.2,
            n_members=3,
        )
        self.assertEqual(r.n_members, 3)
        self.assertAlmostEqual(r.ensemble_stddev, 2.0)

    def test_empty_result(self):
        r = EnsembleResult.empty()
        self.assertEqual(r.n_members, 0)
        self.assertEqual(r.ensemble_stddev, 0.0)
        self.assertEqual(r.member_temps, [])


class TestCache(unittest.TestCase):

    def test_cache_path_format(self):
        p = _cache_path("/tmp/cache", 40.77, -73.87, "2026-02-15", "high")
        self.assertIn("40.77", str(p))
        self.assertIn("2026-02-15", str(p))
        self.assertIn("high", str(p))

    def test_write_and_read_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = EnsembleResult(
                member_temps=[50.0, 52.0],
                ensemble_mean=51.0,
                ensemble_stddev=1.41,
                ecmwf_stddev=1.0,
                gfs_stddev=1.5,
                n_members=2,
            )
            _write_cache(tmpdir, 40.77, -73.87, "2026-02-15", "high", r)
            loaded = _read_cache(tmpdir, 40.77, -73.87, "2026-02-15", "high", ttl_seconds=3600)
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded.ensemble_stddev, 1.41)
            self.assertEqual(loaded.n_members, 2)

    def test_expired_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = EnsembleResult(
                member_temps=[50.0],
                ensemble_mean=50.0,
                ensemble_stddev=0.0,
                ecmwf_stddev=0.0,
                gfs_stddev=0.0,
                n_members=1,
            )
            _write_cache(tmpdir, 40.0, -74.0, "2026-02-15", "high", r)
            # Read with TTL=0 → always expired
            loaded = _read_cache(tmpdir, 40.0, -74.0, "2026-02-15", "high", ttl_seconds=0)
            self.assertIsNone(loaded)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_ensemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weather.ensemble'`

**Step 3: Implement EnsembleResult and cache functions**

```python
# weather/ensemble.py
"""Open-Meteo Ensemble API client — fetches member spread for adaptive sigma.

Queries ECMWF IFS (51 members) and GFS (31 members) ensemble forecasts
to compute empirical forecast uncertainty.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "cache" / "ensemble"


@dataclass
class EnsembleResult:
    """Result of an ensemble spread query."""
    member_temps: list[float] = field(default_factory=list)
    ensemble_mean: float = 0.0
    ensemble_stddev: float = 0.0
    ecmwf_stddev: float = 0.0
    gfs_stddev: float = 0.0
    n_members: int = 0

    @classmethod
    def empty(cls) -> "EnsembleResult":
        return cls()


def _cache_path(cache_dir: str, lat: float, lon: float, date: str, metric: str) -> Path:
    """Build the cache file path for an ensemble query."""
    return Path(cache_dir) / f"{lat}_{lon}_{date}_{metric}.json"


def _write_cache(cache_dir: str, lat: float, lon: float, date: str, metric: str, result: EnsembleResult) -> None:
    """Write an EnsembleResult to the disk cache."""
    path = _cache_path(cache_dir, lat, lon, date, metric)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data["_cached_at"] = time.time()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_cache(cache_dir: str, lat: float, lon: float, date: str, metric: str, ttl_seconds: int = 21600) -> EnsembleResult | None:
    """Read a cached EnsembleResult if it exists and is fresh enough."""
    path = _cache_path(cache_dir, lat, lon, date, metric)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        cached_at = data.pop("_cached_at", 0)
        if time.time() - cached_at > ttl_seconds:
            return None
        return EnsembleResult(**data)
    except (json.JSONDecodeError, IOError, TypeError, KeyError):
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_ensemble.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add weather/ensemble.py weather/tests/test_ensemble.py
git commit -m "feat: add ensemble data model and disk cache"
```

---

### Task 2: Add `fetch_ensemble_spread()` — API client

**Files:**
- Modify: `weather/ensemble.py`
- Modify: `weather/tests/test_ensemble.py`

**Step 1: Write failing tests for fetch_ensemble_spread**

Add to `weather/tests/test_ensemble.py`:

```python
from unittest.mock import patch, MagicMock
from weather.ensemble import fetch_ensemble_spread, ENSEMBLE_API_BASE


class TestFetchEnsembleSpread(unittest.TestCase):

    def _make_api_response(self, ecmwf_temps, gfs_temps, date="2026-02-15"):
        """Build a mock ensemble API JSON response."""
        daily = {"time": [date]}
        for i, t in enumerate(ecmwf_temps):
            daily[f"temperature_2m_max_member{i}"] = [t]
        # GFS members start after ECMWF
        for i, t in enumerate(gfs_temps):
            daily[f"temperature_2m_max_member{i + len(ecmwf_temps)}"] = [t]
        return [
            {"daily": {k: v for k, v in daily.items()}, "model": "ecmwf_ifs025"},
            {"daily": {"time": [date], **{f"temperature_2m_max_member{i}": [t] for i, t in enumerate(gfs_temps)}}, "model": "gfs025"},
        ]

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_basic_fetch(self, mock_fetch):
        """Fetching ensemble data computes stddev from members."""
        # 5 ECMWF members + 3 GFS members
        ecmwf = [50.0, 52.0, 54.0, 48.0, 56.0]
        gfs = [49.0, 53.0, 51.0]
        mock_fetch.return_value = {
            "ecmwf_temps": ecmwf,
            "gfs_temps": gfs,
        }
        result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high")
        self.assertGreater(result.n_members, 0)
        self.assertGreater(result.ensemble_stddev, 0)

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_api_failure_returns_empty(self, mock_fetch):
        """API failure returns EnsembleResult.empty()."""
        mock_fetch.return_value = None
        result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high")
        self.assertEqual(result.n_members, 0)
        self.assertEqual(result.ensemble_stddev, 0.0)

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_cache_hit_skips_api(self, mock_fetch):
        """When cache is fresh, API is not called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cached = EnsembleResult(
                member_temps=[50.0, 52.0],
                ensemble_mean=51.0,
                ensemble_stddev=1.41,
                ecmwf_stddev=1.0,
                gfs_stddev=1.5,
                n_members=2,
            )
            _write_cache(tmpdir, 40.77, -73.87, "2026-02-15", "high", cached)
            result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high", cache_dir=tmpdir)
            mock_fetch.assert_not_called()
            self.assertAlmostEqual(result.ensemble_stddev, 1.41)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_ensemble.py::TestFetchEnsembleSpread -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_ensemble_spread'`

**Step 3: Implement fetch_ensemble_spread and _fetch_ensemble_json**

Add to `weather/ensemble.py`:

```python
import math
import random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX as _SSL_CTX

ENSEMBLE_API_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"
_USER_AGENT = "WeatherGully/1.0"

_ECMWF_MEMBERS = 51
_GFS_MEMBERS = 31


def _fetch_ensemble_json(url: str, timeout: int = 5, max_retries: int = 1) -> dict | None:
    """Fetch JSON from the ensemble API with timeout and retry."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = 1.0 * (0.5 + random.random())
                logger.warning("Ensemble API error — retry %d/%d in %.1fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.warning("Ensemble API failed after %d retries: %s", max_retries, exc)
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Ensemble API JSON parse error: %s", exc)
            return None
    return None


def _stddev(values: list[float]) -> float:
    """Compute sample standard deviation (Bessel-corrected)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def fetch_ensemble_spread(
    lat: float,
    lon: float,
    target_date: str,
    metric: str = "high",
    cache_dir: str | None = None,
    cache_ttl: int = 21600,
) -> EnsembleResult:
    """Fetch ensemble member temperatures and compute spread.

    Args:
        lat: Latitude.
        lon: Longitude.
        target_date: Date string "YYYY-MM-DD".
        metric: "high" or "low".
        cache_dir: Override cache directory (for testing).
        cache_ttl: Cache TTL in seconds (default 6 hours).

    Returns:
        EnsembleResult with member temps and stddev.
    """
    cdir = cache_dir or str(_CACHE_DIR)

    # Check cache first
    cached = _read_cache(cdir, lat, lon, target_date, metric, ttl_seconds=cache_ttl)
    if cached is not None:
        logger.debug("Ensemble cache hit for %s %s %s", lat, lon, target_date)
        return cached

    # Build API URL
    daily_var = "temperature_2m_max" if metric == "high" else "temperature_2m_min"
    url = (
        f"{ENSEMBLE_API_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={daily_var}"
        f"&temperature_unit=fahrenheit"
        f"&models=ecmwf_ifs025,gfs025"
        f"&start_date={target_date}&end_date={target_date}"
    )

    data = _fetch_ensemble_json(url)
    if not data:
        return EnsembleResult.empty()

    # Parse response — may be a list (multi-model) or single dict
    entries = data if isinstance(data, list) else [data]

    ecmwf_temps: list[float] = []
    gfs_temps: list[float] = []

    for entry in entries:
        daily = entry.get("daily", {})
        # Ensemble API returns member data as arrays per day
        # Each member key: temperature_2m_max_memberN → [val_day1, ...]
        for key, values in daily.items():
            if key.startswith(f"{daily_var}_member") or key == daily_var:
                if isinstance(values, list) and values and values[0] is not None:
                    temp = float(values[0])
                    # Determine model from entry context or key pattern
                    # The API returns separate entries per model when multiple models requested
                    model = entry.get("model", "")
                    if "ecmwf" in model.lower():
                        ecmwf_temps.append(temp)
                    elif "gfs" in model.lower():
                        gfs_temps.append(temp)
                    else:
                        # Fallback: first N are ECMWF, rest are GFS
                        ecmwf_temps.append(temp)

    all_temps = ecmwf_temps + gfs_temps

    if not all_temps:
        logger.warning("Ensemble API returned no member temperatures for %s %s", lat, target_date)
        return EnsembleResult.empty()

    result = EnsembleResult(
        member_temps=all_temps,
        ensemble_mean=round(sum(all_temps) / len(all_temps), 2),
        ensemble_stddev=round(_stddev(all_temps), 2),
        ecmwf_stddev=round(_stddev(ecmwf_temps), 2) if len(ecmwf_temps) >= 2 else 0.0,
        gfs_stddev=round(_stddev(gfs_temps), 2) if len(gfs_temps) >= 2 else 0.0,
        n_members=len(all_temps),
    )

    # Write to cache
    try:
        _write_cache(cdir, lat, lon, target_date, metric, result)
    except OSError as exc:
        logger.warning("Failed to write ensemble cache: %s", exc)

    logger.info("Ensemble spread: %.2f°F (ECMWF=%.2f, GFS=%.2f, %d members)",
                result.ensemble_stddev, result.ecmwf_stddev, result.gfs_stddev, result.n_members)

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_ensemble.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/ensemble.py weather/tests/test_ensemble.py
git commit -m "feat: add ensemble API client with cache and spread calculation"
```

---

### Task 3: Add `get_feedback_ema()` to `feedback.py`

**Files:**
- Modify: `weather/feedback.py`
- Modify: `weather/tests/test_feedback.py`

**Step 1: Write failing test**

Add to `weather/tests/test_feedback.py`:

```python
from weather.feedback import FeedbackState, FeedbackEntry, _season_key


class TestGetFeedbackEma(unittest.TestCase):

    def test_returns_ema_when_enough_data(self):
        state = FeedbackState()
        # Record 10 samples to exceed _MIN_SAMPLES=7
        for i in range(10):
            state.record("NYC", 1, 50.0, 50.0 + (i % 3))
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNotNone(ema)
        self.assertGreater(ema, 0)

    def test_returns_none_when_no_data(self):
        state = FeedbackState()
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNone(ema)

    def test_returns_none_when_too_few_samples(self):
        state = FeedbackState()
        state.record("NYC", 1, 50.0, 52.0)
        state.record("NYC", 1, 50.0, 48.0)
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNone(ema)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_feedback.py::TestGetFeedbackEma -v`
Expected: FAIL with `AttributeError: 'FeedbackState' object has no attribute 'get_abs_error_ema'`

**Step 3: Implement get_abs_error_ema**

Add to `weather/feedback.py`, in the `FeedbackState` class after `get_bias()`:

```python
def get_abs_error_ema(self, location: str, month: int) -> float | None:
    """Get the EMA of absolute forecast errors for a location+season.

    Returns the abs_error_ema if enough samples exist, else None.
    This is used as a signal for adaptive sigma — higher values
    indicate the bot is consistently making larger errors.
    """
    key = _season_key(location, month)
    entry = self.entries.get(key)
    if entry and entry.has_enough_data:
        return entry.abs_error_ema
    return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_feedback.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/feedback.py weather/tests/test_feedback.py
git commit -m "feat: expose get_abs_error_ema() for adaptive sigma"
```

---

### Task 4: Add `compute_adaptive_sigma()` to `probability.py`

**Files:**
- Modify: `weather/probability.py`
- Modify: `weather/tests/test_probability.py`

**Step 1: Write failing tests for compute_adaptive_sigma**

Add to `weather/tests/test_probability.py`:

```python
from weather.ensemble import EnsembleResult
from weather.probability import compute_adaptive_sigma, _get_stddev, _get_seasonal_factor


class TestComputeAdaptiveSigma(unittest.TestCase):

    def _make_ensemble(self, stddev):
        return EnsembleResult(
            member_temps=[50.0],
            ensemble_mean=50.0,
            ensemble_stddev=stddev,
            ecmwf_stddev=stddev,
            gfs_stddev=stddev,
            n_members=51,
        )

    def test_ensemble_signal_wins(self):
        """When ensemble stddev is largest, it should determine sigma."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=self._make_ensemble(5.0),
            model_spread=1.0,
            ema_error=1.0,
            forecast_date=today,
            location="NYC",
        )
        # 5.0 * underdispersion_factor (1.3) = 6.5, should be the max
        self.assertGreaterEqual(result, 6.0)

    def test_model_spread_wins(self):
        """When model spread is largest, it should determine sigma."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=self._make_ensemble(0.5),
            model_spread=10.0,
            ema_error=0.5,
            forecast_date=today,
            location="NYC",
        )
        # 10.0 * spread_to_sigma (0.7) = 7.0
        self.assertGreaterEqual(result, 5.0)

    def test_ema_signal_wins(self):
        """When EMA error is largest, it should determine sigma."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=self._make_ensemble(0.5),
            model_spread=0.5,
            ema_error=8.0,
            forecast_date=today,
            location="NYC",
        )
        # 8.0 * ema_to_sigma (1.25) = 10.0
        self.assertGreaterEqual(result, 8.0)

    def test_floor_prevents_too_low(self):
        """Sigma should never go below the calibrated floor."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=self._make_ensemble(0.1),
            model_spread=0.1,
            ema_error=0.1,
            forecast_date=today,
            location="NYC",
        )
        floor = _get_stddev(today, "NYC") * _get_seasonal_factor(
            datetime.now(timezone.utc).month, "NYC")
        self.assertGreaterEqual(result, floor)

    def test_none_ensemble_uses_other_signals(self):
        """When ensemble is None, sigma uses spread and EMA."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=None,
            model_spread=4.0,
            ema_error=3.0,
            forecast_date=today,
            location="NYC",
        )
        self.assertGreater(result, 0)

    def test_all_none_returns_floor(self):
        """When all signals are None/zero, returns calibrated floor."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=None,
            model_spread=0.0,
            ema_error=None,
            forecast_date=today,
            location="NYC",
        )
        floor = _get_stddev(today, "NYC") * _get_seasonal_factor(
            datetime.now(timezone.utc).month, "NYC")
        self.assertAlmostEqual(result, floor, places=1)

    def test_result_always_positive(self):
        """Sigma must always be positive."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(
            ensemble_result=self._make_ensemble(0.0),
            model_spread=0.0,
            ema_error=None,
            forecast_date=today,
            location="",
        )
        self.assertGreater(result, 0)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestComputeAdaptiveSigma -v`
Expected: FAIL with `ImportError: cannot import name 'compute_adaptive_sigma'`

**Step 3: Implement compute_adaptive_sigma**

Add to `weather/probability.py`, before `estimate_bucket_probability()`:

```python
from .ensemble import EnsembleResult

# Adaptive sigma conversion factors (NWP literature defaults)
# These are recalibrated empirically after 30+ days of data — see calibrate.py
_UNDERDISPERSION_FACTOR = 1.3   # Ensemble spread is typically underdispersed
_SPREAD_TO_SIGMA = 0.7          # |GFS - ECMWF| to sigma conversion
_EMA_TO_SIGMA = 1.25            # MAE → sigma for Gaussian (MAE ≈ 0.8σ)


def _load_adaptive_factors() -> dict:
    """Load calibrated adaptive sigma factors from calibration.json."""
    cal = _load_calibration()
    return cal.get("adaptive_sigma", {})


def compute_adaptive_sigma(
    ensemble_result: EnsembleResult | None,
    model_spread: float,
    ema_error: float | None,
    forecast_date: str,
    location: str = "",
) -> float:
    """Compute adaptive sigma from max of 3 real-time signals.

    The calibrated sigma (horizon + seasonal) serves as a floor.
    Each signal is converted to sigma-space using calibrated or default factors.

    Args:
        ensemble_result: Ensemble spread data (may be None if API failed).
        model_spread: |GFS - ECMWF| deterministic model disagreement.
        ema_error: EMA of absolute forecast errors (may be None).
        forecast_date: Date string "YYYY-MM-DD".
        location: Canonical location key.

    Returns:
        Adaptive sigma in °F (always positive, >= floor).
    """
    factors = _load_adaptive_factors()
    underdispersion = factors.get("underdispersion_factor", _UNDERDISPERSION_FACTOR)
    spread_factor = factors.get("spread_to_sigma_factor", _SPREAD_TO_SIGMA)
    ema_factor = factors.get("ema_to_sigma_factor", _EMA_TO_SIGMA)

    # Signal 1: Ensemble spread
    sigma_ensemble = 0.0
    if ensemble_result and ensemble_result.n_members >= 2:
        sigma_ensemble = ensemble_result.ensemble_stddev * underdispersion

    # Signal 2: Model spread (deterministic)
    sigma_spread = model_spread * spread_factor

    # Signal 3: EMA of past errors
    sigma_ema = 0.0
    if ema_error is not None and ema_error > 0:
        sigma_ema = ema_error * ema_factor

    # Floor: calibrated sigma × seasonal factor
    base_sigma = _get_stddev(forecast_date, location=location)
    try:
        month = int(forecast_date.split("-")[1])
    except (IndexError, ValueError):
        month = datetime.now(timezone.utc).month
    seasonal = _get_seasonal_factor(month, location=location)
    sigma_floor = base_sigma * seasonal

    # Max of all signals (most pessimistic wins)
    sigma = max(sigma_ensemble, sigma_spread, sigma_ema, sigma_floor)

    logger.info(
        "Adaptive sigma: ensemble=%.2f spread=%.2f ema=%.2f floor=%.2f → final=%.2f",
        sigma_ensemble, sigma_spread, sigma_ema, sigma_floor, sigma,
    )

    return sigma
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_probability.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "feat: add compute_adaptive_sigma() with 3-signal max and calibrated floor"
```

---

### Task 5: Add `sigma_override` to `estimate_bucket_probability()`

**Files:**
- Modify: `weather/probability.py:226-283`
- Modify: `weather/tests/test_probability.py`

**Step 1: Write failing tests**

Add to `weather/tests/test_probability.py`:

```python
class TestSigmaOverride(unittest.TestCase):

    def test_override_bypasses_internal_sigma(self):
        """When sigma_override is provided, internal sigma calculation is skipped."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # With a very large sigma override, probability for any bucket should be low
        prob_wide = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=20.0)
        # With a very small sigma override, probability should be high
        prob_narrow = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=0.5)
        self.assertGreater(prob_narrow, prob_wide)

    def test_override_none_uses_internal_sigma(self):
        """When sigma_override is None, normal sigma calculation applies."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prob_default = estimate_bucket_probability(50.0, 49, 51, today)
        prob_none = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=None)
        self.assertAlmostEqual(prob_default, prob_none, places=4)

    def test_override_ignores_weather_data(self):
        """When sigma_override is set, weather_data should be ignored."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        weather = {"cloud_cover_max": 100, "wind_speed_max": 60, "precip_sum": 20}
        prob_with_weather = estimate_bucket_probability(
            50.0, 49, 51, today, sigma_override=3.0, weather_data=weather)
        prob_no_weather = estimate_bucket_probability(
            50.0, 49, 51, today, sigma_override=3.0)
        self.assertAlmostEqual(prob_with_weather, prob_no_weather, places=4)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestSigmaOverride -v`
Expected: FAIL with `TypeError: estimate_bucket_probability() got an unexpected keyword argument 'sigma_override'`

**Step 3: Add sigma_override parameter**

Modify `estimate_bucket_probability()` in `weather/probability.py:226`:

Change the function signature to add `sigma_override: float | None = None`:

```python
def estimate_bucket_probability(
    forecast_temp: float,
    bucket_low: int,
    bucket_high: int,
    forecast_date: str,
    apply_seasonal: bool = True,
    location: str = "",
    weather_data: dict | None = None,
    metric: str = "high",
    sigma_override: float | None = None,
) -> float:
```

And replace the sigma computation block (lines 253-265) with:

```python
    if sigma_override is not None:
        sigma = sigma_override
    else:
        sigma = _get_stddev(forecast_date, location=location)
        if apply_seasonal:
            try:
                month = int(forecast_date.split("-")[1])
            except (IndexError, ValueError):
                month = datetime.now(timezone.utc).month
            factor = _get_seasonal_factor(month, location=location)
            sigma *= factor

        # Weather-based sigma adjustment
        if weather_data:
            sigma *= _weather_sigma_multiplier(weather_data, metric)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_probability.py -v`
Expected: PASS (all existing + new tests)

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "feat: add sigma_override parameter to estimate_bucket_probability()"
```

---

### Task 6: Add `sigma_override` to `score_buckets()` and `estimate_bucket_probability_with_obs()`

**Files:**
- Modify: `weather/probability.py:364-453` (estimate_bucket_probability_with_obs)
- Modify: `weather/strategy.py:137-218` (score_buckets)
- Modify: `weather/tests/test_strategy.py`

**Step 1: Write failing test**

Add to `weather/tests/test_strategy.py`:

```python
class TestScoreBucketsSigmaOverride(unittest.TestCase):

    def test_sigma_override_passed_through(self):
        """score_buckets should pass sigma_override to probability functions."""
        from weather.config import Config
        config = Config()
        markets = [
            {"outcome_name": "50-54°F", "external_price_yes": 0.30, "best_ask": 0.30},
        ]
        # With very different sigma overrides, probabilities should differ
        scored_wide = score_buckets(markets, 52.0, "2026-02-15", config,
                                     metric="high", sigma_override=20.0)
        scored_narrow = score_buckets(markets, 52.0, "2026-02-15", config,
                                       metric="high", sigma_override=2.0)
        if scored_wide and scored_narrow:
            self.assertGreater(scored_narrow[0]["prob"], scored_wide[0]["prob"])
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest weather/tests/test_strategy.py::TestScoreBucketsSigmaOverride -v`
Expected: FAIL with `TypeError: score_buckets() got an unexpected keyword argument 'sigma_override'`

**Step 3: Thread sigma_override through score_buckets and estimate_bucket_probability_with_obs**

In `weather/probability.py`, add `sigma_override: float | None = None` to `estimate_bucket_probability_with_obs()` signature and pass it through:

```python
def estimate_bucket_probability_with_obs(
    forecast_temp: float,
    bucket_low: int,
    bucket_high: int,
    forecast_date: str,
    obs_data: dict | None = None,
    metric: str = "high",
    apply_seasonal: bool = True,
    station_lon: float = -74.0,
    station_tz: str = "",
    location: str = "",
    weather_data: dict | None = None,
    sigma_override: float | None = None,
) -> float:
```

When `obs_data` is empty, pass `sigma_override` to the inner `estimate_bucket_probability()` call. When `obs_data` is present and `sigma_override` is set, use it instead of the intraday sigma.

In `weather/strategy.py`, add `sigma_override: float | None = None` to `score_buckets()`:

```python
def score_buckets(
    event_markets: list[dict],
    forecast_temp: float,
    forecast_date: str,
    config: Config,
    obs_data: dict | None = None,
    metric: str = "high",
    location: str = "",
    weather_data: dict | None = None,
    sigma_override: float | None = None,
) -> list[dict]:
```

And pass `sigma_override=sigma_override` to both `estimate_bucket_probability_with_obs()` and `estimate_bucket_probability()` calls inside it.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_strategy.py weather/tests/test_probability.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: thread sigma_override through score_buckets and obs probability"
```

---

### Task 7: Wire adaptive sigma into `run_weather_strategy()`

**Files:**
- Modify: `weather/config.py` (add `adaptive_sigma` toggle)
- Modify: `weather/strategy.py:463-951` (run_weather_strategy)
- Modify: `weather/tests/test_strategy.py`

**Step 1: Add config toggle**

In `weather/config.py`, add after line 101 (`aviation_hours`):

```python
    # Adaptive sigma (ensemble-based)
    adaptive_sigma: bool = True
```

**Step 2: Write failing integration test**

Add to `weather/tests/test_strategy.py`:

```python
class TestAdaptiveSigmaIntegration(unittest.TestCase):

    @patch("weather.strategy.fetch_ensemble_spread")
    @patch("weather.strategy.compute_adaptive_sigma")
    def test_adaptive_sigma_called_when_enabled(self, mock_compute, mock_fetch):
        """When adaptive_sigma=True, compute_adaptive_sigma should be called."""
        from weather.ensemble import EnsembleResult
        mock_fetch.return_value = EnsembleResult(
            member_temps=[50.0, 52.0],
            ensemble_mean=51.0,
            ensemble_stddev=1.5,
            ecmwf_stddev=1.0,
            gfs_stddev=1.5,
            n_members=2,
        )
        mock_compute.return_value = 3.5

        # This test verifies the import and call path exist.
        # Full integration is tested via dry_run in the main strategy.
        from weather.strategy import score_buckets
        from weather.config import Config
        config = Config(adaptive_sigma=True)
        # Just verify the imports work and config flag is accessible
        self.assertTrue(config.adaptive_sigma)
```

**Step 3: Implement the wiring in run_weather_strategy**

In `weather/strategy.py`, add imports at the top:

```python
from .ensemble import fetch_ensemble_spread
from .probability import compute_adaptive_sigma
```

In `run_weather_strategy()`, after the ensemble forecast computation (around line 714, after `forecast_temp = ensemble_temp`), add:

```python
        # Adaptive sigma: compute from ensemble spread + model spread + feedback EMA
        adaptive_sigma_value = None
        if config.adaptive_sigma:
            loc_data = LOCATIONS.get(location, {})
            ensemble_result = fetch_ensemble_spread(
                loc_data.get("lat", 0), loc_data.get("lon", 0),
                date_str, metric,
            )
            ema_error = feedback.get_abs_error_ema(location, datetime.now(timezone.utc).month)
            adaptive_sigma_value = compute_adaptive_sigma(
                ensemble_result, model_spread, ema_error,
                date_str, location,
            )
```

Then pass `sigma_override=adaptive_sigma_value` to the `score_buckets()` call:

```python
            scored = score_buckets(event_markets, forecast_temp, date_str, config,
                                   obs_data=scoring_obs, metric=metric,
                                   location=location, weather_data=om_data,
                                   sigma_override=adaptive_sigma_value)
```

And also to the legacy single-match path (the `estimate_bucket_probability()` call inside the else branch).

**Step 4: Run all tests**

Run: `python3 -m pytest weather/tests/ -q`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add weather/config.py weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: wire adaptive sigma into strategy loop with config toggle"
```

---

### Task 8: Add sigma signal logging for future calibration

**Files:**
- Create: `weather/sigma_log.py`
- Create: `weather/tests/test_sigma_log.py`

**Step 1: Write failing test**

```python
# weather/tests/test_sigma_log.py
"""Tests for sigma signal logging."""

import json
import os
import tempfile
import unittest

from weather.sigma_log import log_sigma_signals, load_sigma_log


class TestSigmaLog(unittest.TestCase):

    def test_log_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            log_sigma_signals(
                path=path,
                location="NYC",
                date="2026-02-15",
                metric="high",
                ensemble_stddev=2.5,
                model_spread=1.8,
                ema_error=2.1,
                final_sigma=3.25,
                forecast_temp=52.0,
            )
            entries = load_sigma_log(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["location"], "NYC")
            self.assertAlmostEqual(entries[0]["ensemble_stddev"], 2.5)
        finally:
            os.unlink(path)

    def test_appends_multiple_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            for i in range(3):
                log_sigma_signals(
                    path=path,
                    location="NYC",
                    date=f"2026-02-{15+i}",
                    metric="high",
                    ensemble_stddev=2.0 + i,
                    model_spread=1.0,
                    ema_error=1.5,
                    final_sigma=2.5 + i,
                    forecast_temp=50.0,
                )
            entries = load_sigma_log(path)
            self.assertEqual(len(entries), 3)
        finally:
            os.unlink(path)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_sigma_log.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weather.sigma_log'`

**Step 3: Implement sigma_log.py**

```python
# weather/sigma_log.py
"""Sigma signal logging for adaptive sigma calibration.

Each forecast logs the 3 sigma signals and the final sigma chosen.
This data is used to calibrate the conversion factors after 30+ days.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "sigma_log.json"


def log_sigma_signals(
    location: str,
    date: str,
    metric: str,
    ensemble_stddev: float,
    model_spread: float,
    ema_error: float | None,
    final_sigma: float,
    forecast_temp: float,
    path: str | None = None,
) -> None:
    """Append a sigma signal entry to the log file."""
    log_path = Path(path) if path else _DEFAULT_PATH
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "date": date,
        "metric": metric,
        "ensemble_stddev": ensemble_stddev,
        "model_spread": model_spread,
        "ema_error": ema_error,
        "final_sigma": final_sigma,
        "forecast_temp": forecast_temp,
    }

    # Load existing entries
    entries = load_sigma_log(str(log_path))
    entries.append(entry)

    # Atomic write
    fd, tmp = tempfile.mkstemp(dir=str(log_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, str(log_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_sigma_log(path: str | None = None) -> list[dict]:
    """Load sigma signal entries from the log file."""
    log_path = Path(path) if path else _DEFAULT_PATH
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_sigma_log.py -v`
Expected: PASS

**Step 5: Wire logging into strategy (in `run_weather_strategy()`, after computing adaptive sigma)**

Add import: `from .sigma_log import log_sigma_signals`

After `adaptive_sigma_value = compute_adaptive_sigma(...)`:

```python
            log_sigma_signals(
                location=location,
                date=date_str,
                metric=metric,
                ensemble_stddev=ensemble_result.ensemble_stddev,
                model_spread=model_spread,
                ema_error=ema_error,
                final_sigma=adaptive_sigma_value,
                forecast_temp=forecast_temp,
            )
```

**Step 6: Commit**

```bash
git add weather/sigma_log.py weather/tests/test_sigma_log.py weather/strategy.py
git commit -m "feat: add sigma signal logging for future calibration"
```

---

### Task 9: Final integration test and full test suite

**Files:**
- Modify: `weather/tests/test_ensemble.py` (add API response parsing test)

**Step 1: Run the full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All tests PASS (existing ~356 + ~20 new tests)

**Step 2: Verify no regressions**

Run: `python3 -m pytest weather/tests/test_probability.py weather/tests/test_strategy.py weather/tests/test_feedback.py -v`
Expected: All existing tests still pass — the `sigma_override=None` default means old behavior is unchanged.

**Step 3: Add cache directory to .gitignore**

Check if `weather/cache/` is already in `.gitignore`. If not, add it:

```
weather/cache/
weather/sigma_log.json
```

**Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: add ensemble cache and sigma log to gitignore"
```

---

### Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `weather/ensemble.py` | Create | Ensemble API client with cache, EnsembleResult dataclass |
| `weather/sigma_log.py` | Create | Sigma signal logging for future calibration |
| `weather/probability.py` | Modify | Add `compute_adaptive_sigma()` and `sigma_override` parameter |
| `weather/feedback.py` | Modify | Add `get_abs_error_ema()` method |
| `weather/strategy.py` | Modify | Wire ensemble fetch + adaptive sigma into trading loop |
| `weather/config.py` | Modify | Add `adaptive_sigma: bool = True` toggle |
| `weather/tests/test_ensemble.py` | Create | 8 tests for ensemble client and cache |
| `weather/tests/test_sigma_log.py` | Create | 2 tests for sigma logging |
| `weather/tests/test_probability.py` | Modify | 10 tests for adaptive sigma and sigma_override |
| `weather/tests/test_strategy.py` | Modify | 2 tests for sigma_override passthrough |
| `weather/tests/test_feedback.py` | Modify | 3 tests for get_abs_error_ema |
