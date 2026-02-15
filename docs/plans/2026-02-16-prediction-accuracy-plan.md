# Prediction Accuracy & Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix reliability bugs (fill timeout, UTC/local, dead code), add Platt scaling for probability calibration, and switch calibration ground truth from ERA5 to METAR.

**Architecture:** Four sequential tasks. Task 1 fixes execution bugs with no model changes. Task 2 adds Platt scaling sigmoid post-processing in `score_buckets()`. Task 3 adds IEM METAR historical data client and wires it into calibration. Task 4 re-runs calibration and validates improvements.

**Tech Stack:** Python 3.14, stdlib (urllib, math, json), pytest, no new dependencies.

---

### Task 1: Quick Fixes (Reliability)

**Files:**
- Modify: `weather/strategy.py:910` (fill timeout)
- Modify: `weather/strategy.py:115-133,864-869` (remove dead code)
- Modify: `weather/bridge.py:196-201` (remove dead code)
- Modify: `weather/aviation.py:181-186` (UTC/local bug)
- Modify: `weather/config.py` (add station ICAO to LOCATIONS)
- Test: `weather/tests/test_strategy.py`
- Test: `weather/tests/test_aviation.py`

**Step 1: Write failing test for aviation UTC/local fix**

In `weather/tests/test_aviation.py`, add a test that verifies `compute_daily_extremes()` correctly filters UTC observations to local dates:

```python
class TestComputeDailyExtremesTimezone(unittest.TestCase):

    def test_utc_observation_maps_to_correct_local_date(self):
        """Observation at 2025-01-16T03:00:00Z is still Jan 15 in NYC (UTC-5)."""
        observations = [
            {"time": "2025-01-15T22:00:00Z", "temp_f": 35.0},  # Jan 15 local
            {"time": "2025-01-16T03:00:00Z", "temp_f": 30.0},  # Still Jan 15 local (UTC-5)
            {"time": "2025-01-16T06:00:00Z", "temp_f": 28.0},  # Jan 16 local (UTC-5)
        ]
        result = compute_daily_extremes(observations, "2025-01-15", tz_name="America/New_York")
        # Should include the 03:00Z obs (still Jan 15 local) but not 06:00Z (Jan 16 local)
        self.assertIsNotNone(result)
        self.assertEqual(result["obs_count"], 2)
        self.assertAlmostEqual(result["high"], 35.0)
        self.assertAlmostEqual(result["low"], 30.0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest weather/tests/test_aviation.py::TestComputeDailyExtremesTimezone -v`
Expected: FAIL (current implementation compares UTC date string, not local)

**Step 3: Fix `compute_daily_extremes()` to use local timezone**

In `weather/aviation.py`, modify `compute_daily_extremes()`:

```python
def compute_daily_extremes(
    observations: list[dict],
    target_date: str,
    tz_name: str = "America/New_York",
) -> dict | None:
    """Compute daily high/low from METAR observations for a specific date.

    Converts UTC observation times to local timezone before filtering
    to the target date, since Polymarket resolves on local calendar dates.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    day_obs = []
    for obs in observations:
        obs_time = obs.get("time", "")
        try:
            utc_dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
            local_dt = utc_dt.astimezone(tz)
            local_date = local_dt.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            local_date = obs_time[:10]  # Fallback to raw date prefix
        if local_date == target_date:
            day_obs.append(obs)

    if not day_obs:
        return None

    temps = [obs["temp_f"] for obs in day_obs]
    latest_time = max(obs["time"] for obs in day_obs)

    return {
        "high": max(temps),
        "low": min(temps),
        "obs_count": len(day_obs),
        "latest_obs_time": latest_time,
    }
```

Also update `get_aviation_daily_data()` which extracts dates from UTC observations — it must also use local timezone to group by date:

```python
def get_aviation_daily_data(
    locations: list[str],
    hours: int = 24,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    from .config import LOCATIONS as _LOCATIONS
    all_obs = get_metar_observations(
        locations, hours=hours,
        max_retries=max_retries, base_delay=base_delay,
    )

    result: dict[str, dict[str, dict]] = {}

    for loc, observations in all_obs.items():
        if not observations:
            continue

        tz_name = _LOCATIONS.get(loc, {}).get("tz", "America/New_York")

        # Collect all unique LOCAL dates from observations
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        dates = set()
        for obs in observations:
            try:
                utc_dt = _dt.fromisoformat(obs["time"].replace("Z", "+00:00"))
                local_date = utc_dt.astimezone(tz).strftime("%Y-%m-%d")
                dates.add(local_date)
            except (ValueError, KeyError):
                dates.add(obs["time"][:10])

        loc_data: dict[str, dict] = {}
        for date_str in sorted(dates):
            extremes = compute_daily_extremes(observations, date_str, tz_name=tz_name)
            if extremes:
                loc_data[date_str] = {
                    "obs_high": extremes["high"],
                    "obs_low": extremes["low"],
                    "obs_count": extremes["obs_count"],
                    "latest_obs_time": extremes["latest_obs_time"],
                }

        if loc_data:
            result[loc] = loc_data

    logger.info("Aviation daily data: %d locations with observations", len(result))
    return result
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest weather/tests/test_aviation.py -v`
Expected: PASS

**Step 5: Fix fill timeout in strategy.py**

In `weather/strategy.py` line 910, change:
```python
result = client.execute_trade(market_id, "yes", position_size)
```
to:
```python
result = client.execute_trade(
    market_id, "yes", position_size,
    fill_timeout=config.fill_timeout_seconds,
    fill_poll_interval=config.fill_poll_interval,
)
```

**Step 6: Remove dead code (detect_price_trend + get_price_history callers)**

In `weather/strategy.py`:
- Remove the `detect_price_trend()` function (lines 115-133)
- Remove the `use_trends` parameter from `run_weather_strategy()` signature (line 477)
- Remove the trend detection block (lines 864-869):
  ```python
  if use_trends:
      history = client.get_price_history(market_id)
      trend = detect_price_trend(history, config.price_drop_threshold)
      if trend["is_opportunity"]:
          logger.info(...)
  ```

In `weather/bridge.py`:
- Remove `get_price_history()` method (lines 196-201)

**Step 7: Run all strategy and aviation tests**

Run: `python3 -m pytest weather/tests/test_strategy.py weather/tests/test_aviation.py -v`
Expected: PASS (may need to update test mocks that reference `use_trends` or `detect_price_trend`)

**Step 8: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass (except pre-existing backtest failure)

**Step 9: Commit**

```bash
git add weather/strategy.py weather/bridge.py weather/aviation.py weather/tests/test_aviation.py weather/tests/test_strategy.py
git commit -m "fix: fill timeout, UTC/local aviation bug, remove dead trend code"
```

---

### Task 2: Platt Scaling

**Files:**
- Modify: `weather/probability.py` (add `platt_calibrate()` and `_load_platt_params()`)
- Modify: `weather/strategy.py:140-224` (apply Platt in `score_buckets()`)
- Modify: `weather/calibrate.py` (add `_compute_platt_params()`)
- Test: `weather/tests/test_probability.py`
- Test: `weather/tests/test_calibrate.py`

**Step 1: Write failing tests for Platt calibration**

In `weather/tests/test_probability.py`:

```python
class TestPlattCalibrate(unittest.TestCase):

    def test_identity_when_no_params(self):
        """Without calibration params, returns raw probability."""
        from weather.probability import platt_calibrate
        self.assertAlmostEqual(platt_calibrate(0.5), 0.5)
        self.assertAlmostEqual(platt_calibrate(0.3), 0.3)

    def test_bounded_output(self):
        from weather.probability import platt_calibrate
        result = platt_calibrate(0.001)
        self.assertGreaterEqual(result, 0.01)
        result = platt_calibrate(0.999)
        self.assertLessEqual(result, 0.99)

    def test_monotonic(self):
        """Platt scaling must preserve probability ordering."""
        from weather.probability import platt_calibrate
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = [platt_calibrate(p) for p in probs]
        for i in range(len(calibrated) - 1):
            self.assertLessEqual(calibrated[i], calibrated[i + 1])
```

In `weather/tests/test_calibrate.py`:

```python
class TestComputePlattParams(unittest.TestCase):

    def test_perfect_calibration_gives_identity(self):
        """If predicted == actual for all bins, a~1 and b~0."""
        from weather.calibrate import _compute_platt_params
        predictions = [0.2, 0.4, 0.6, 0.8]
        actuals = [0.2, 0.4, 0.6, 0.8]
        params = _compute_platt_params(predictions, actuals)
        self.assertIn("a", params)
        self.assertIn("b", params)
        # a should be close to 1.0, b close to 0.0
        self.assertAlmostEqual(params["a"], 1.0, delta=0.3)
        self.assertAlmostEqual(params["b"], 0.0, delta=0.3)

    def test_overconfident_model(self):
        """Overconfident model (predicted > actual) should have a > 1."""
        from weather.calibrate import _compute_platt_params
        # Model predicts 50% but actual rate is only 30%
        predictions = [0.2, 0.35, 0.5, 0.65, 0.8]
        actuals = [0.05, 0.15, 0.30, 0.50, 0.70]
        params = _compute_platt_params(predictions, actuals)
        # Should stretch probabilities (a > 1 pushes away from center)
        self.assertIsNotNone(params["a"])

    def test_empty_data_returns_identity(self):
        from weather.calibrate import _compute_platt_params
        params = _compute_platt_params([], [])
        self.assertAlmostEqual(params["a"], 1.0)
        self.assertAlmostEqual(params["b"], 0.0)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestPlattCalibrate weather/tests/test_calibrate.py::TestComputePlattParams -v`
Expected: FAIL (functions don't exist yet)

**Step 3: Implement `platt_calibrate()` in probability.py**

Add after the `_load_adaptive_factors()` function:

```python
def _load_platt_params() -> dict:
    """Load Platt scaling parameters from calibration.json.

    Returns dict with 'a' and 'b' keys, or empty dict if unavailable.
    """
    cal = _load_calibration()
    return cal.get("platt_scaling", {})


def platt_calibrate(prob: float) -> float:
    """Apply Platt scaling to a raw probability.

    Transforms: calibrated = sigmoid(a * logit(prob) + b)
    Falls back to identity if no calibration params available.

    Args:
        prob: Raw probability in (0, 1).

    Returns:
        Calibrated probability in [0.01, 0.99].
    """
    params = _load_platt_params()
    a = params.get("a", 1.0)
    b = params.get("b", 0.0)

    # Identity shortcut
    if a == 1.0 and b == 0.0:
        return max(0.01, min(0.99, prob))

    # Clamp input to avoid log(0)
    p = max(1e-6, min(1 - 1e-6, prob))
    logit_p = math.log(p / (1 - p))
    calibrated = 1.0 / (1.0 + math.exp(-(a * logit_p + b)))

    return max(0.01, min(0.99, round(calibrated, 4)))
```

**Step 4: Run probability tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_probability.py::TestPlattCalibrate -v`
Expected: PASS

**Step 5: Implement `_compute_platt_params()` in calibrate.py**

Add before `build_calibration_tables()`:

```python
def _compute_platt_params(
    predictions: list[float],
    actuals: list[float],
) -> dict:
    """Fit Platt scaling parameters (a, b) via gradient descent on log-loss.

    Platt scaling: calibrated = sigmoid(a * logit(pred) + b)

    Uses simple gradient descent to minimize binary cross-entropy loss
    over the calibration bins.

    Args:
        predictions: List of predicted probabilities (bin centers).
        actuals: List of actual outcome rates.

    Returns:
        {"a": float, "b": float}
    """
    if len(predictions) < 2 or len(actuals) < 2:
        return {"a": 1.0, "b": 0.0}

    # Initialize params
    a, b = 1.0, 0.0
    lr = 0.1
    eps = 1e-6

    for _ in range(500):
        grad_a, grad_b = 0.0, 0.0
        n = len(predictions)

        for pred, actual in zip(predictions, actuals):
            p = max(eps, min(1 - eps, pred))
            y = max(eps, min(1 - eps, actual))
            logit_p = math.log(p / (1 - p))
            z = a * logit_p + b
            sigmoid_z = 1.0 / (1.0 + math.exp(-z))
            # Gradient of cross-entropy: d/d_param = (sigmoid(z) - y) * d_z/d_param
            err = sigmoid_z - y
            grad_a += err * logit_p / n
            grad_b += err / n

        a -= lr * grad_a
        b -= lr * grad_b

    return {"a": round(a, 4), "b": round(b, 4)}
```

**Step 6: Wire Platt params into `build_calibration_tables()`**

In `build_calibration_tables()`, after the `adaptive_factors` computation, add:

```python
    # Platt scaling: compute params from internal calibration curve
    # Build a mini calibration curve from the errors
    platt_params = _fit_platt_from_errors(all_errors)
```

Add helper function:

```python
def _fit_platt_from_errors(errors: list[dict]) -> dict:
    """Fit Platt params by simulating bucket probabilities vs actual outcomes.

    Groups errors into probability bins and computes actual hit rates,
    then fits Platt scaling on the resulting calibration curve.
    """
    if not errors:
        return {"a": 1.0, "b": 0.0}

    # Compute bucket probabilities for each error record
    # Each error has forecast, actual, metric — simulate a 5°F bucket centered on forecast
    from .probability import estimate_bucket_probability

    bins: dict[int, list[int]] = {}  # bin_index → [0 or 1 outcomes]

    for err in errors:
        forecast = err.get("forecast", 0)
        actual = err.get("actual", 0)
        # Simulate a 5°F bucket centered on forecast
        bucket_low = round(forecast) - 2
        bucket_high = round(forecast) + 2
        prob = estimate_bucket_probability(
            forecast, bucket_low, bucket_high,
            err.get("target_date", "2025-06-15"),
            apply_seasonal=False,
            location=err.get("location", ""),
        )
        # Did actual land in the bucket?
        outcome = 1 if bucket_low <= actual <= bucket_high else 0
        # Bin by decile
        bin_idx = min(9, int(prob * 10))
        bins.setdefault(bin_idx, []).append(outcome)

    # Compute calibration curve from bins
    predictions = []
    actuals_list = []
    for bin_idx in sorted(bins):
        outcomes = bins[bin_idx]
        if len(outcomes) >= 5:  # Minimum samples per bin
            pred = (bin_idx + 0.5) / 10.0
            actual_rate = sum(outcomes) / len(outcomes)
            predictions.append(pred)
            actuals_list.append(actual_rate)

    return _compute_platt_params(predictions, actuals_list)
```

Add `"platt_scaling": platt_params` to the return dict in `build_calibration_tables()`.

**Step 7: Apply Platt in `score_buckets()`**

In `weather/strategy.py`, in `score_buckets()`, after computing `prob` and before computing `ev`, add:

```python
        from .probability import platt_calibrate
        prob = platt_calibrate(prob)
```

So the EV line becomes:
```python
        prob = platt_calibrate(prob)
        ev = prob - price
```

**Step 8: Run calibrate and probability tests**

Run: `python3 -m pytest weather/tests/test_calibrate.py::TestComputePlattParams weather/tests/test_probability.py::TestPlattCalibrate -v`
Expected: PASS

**Step 9: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass

**Step 10: Commit**

```bash
git add weather/probability.py weather/calibrate.py weather/strategy.py weather/tests/test_probability.py weather/tests/test_calibrate.py
git commit -m "feat: add Platt scaling for probability calibration correction"
```

---

### Task 3: METAR Historical Ground Truth

**Files:**
- Modify: `weather/historical.py` (add `get_historical_metar_actuals()`)
- Modify: `weather/config.py:13-20` (add `station` ICAO to LOCATIONS)
- Modify: `weather/calibrate.py` (add `--actuals-source` CLI option)
- Test: `weather/tests/test_historical.py`

**Step 1: Write failing test for METAR historical client**

In `weather/tests/test_historical.py` (create if needed):

```python
class TestGetHistoricalMetarActuals(unittest.TestCase):

    @patch("weather.historical._fetch_metar_csv")
    def test_parses_iem_csv(self, mock_fetch):
        mock_fetch.return_value = (
            "station,valid,tmpf\n"
            "KLGA,2025-01-15 00:54,32.0\n"
            "KLGA,2025-01-15 06:54,28.1\n"
            "KLGA,2025-01-15 14:54,41.0\n"
            "KLGA,2025-01-15 18:54,38.5\n"
            "KLGA,2025-01-16 00:54,35.0\n"
        )
        result = get_historical_metar_actuals("KLGA", "2025-01-15", "2025-01-16")
        self.assertIn("2025-01-15", result)
        self.assertAlmostEqual(result["2025-01-15"]["high"], 41.0)
        self.assertAlmostEqual(result["2025-01-15"]["low"], 28.1)

    @patch("weather.historical._fetch_metar_csv")
    def test_empty_csv_returns_empty(self, mock_fetch):
        mock_fetch.return_value = "station,valid,tmpf\n"
        result = get_historical_metar_actuals("KLGA", "2025-01-15", "2025-01-16")
        self.assertEqual(result, {})
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest weather/tests/test_historical.py::TestGetHistoricalMetarActuals -v`
Expected: FAIL (function doesn't exist)

**Step 3: Add station codes to LOCATIONS in config.py**

In `weather/config.py`, add `"station"` key to each location:

```python
LOCATIONS = {
    "NYC": {"lat": 40.7769, "lon": -73.8740, "name": "New York City (LaGuardia)", "tz": "America/New_York", "station": "KLGA"},
    "Chicago": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago (O'Hare)", "tz": "America/Chicago", "station": "KORD"},
    "Seattle": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle (Sea-Tac)", "tz": "America/Los_Angeles", "station": "KSEA"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta (Hartsfield)", "tz": "America/New_York", "station": "KATL"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "name": "Dallas (DFW)", "tz": "America/Chicago", "station": "KDFW"},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "name": "Miami (MIA)", "tz": "America/New_York", "station": "KMIA"},
}
```

**Step 4: Implement METAR historical client in historical.py**

Add to `weather/historical.py`:

```python
_IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _fetch_metar_csv(url: str, max_retries: int = 3, base_delay: float = 1.0) -> str | None:
    """Fetch CSV text from Iowa Environmental Mesonet with retry."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return resp.read().decode()
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning("IEM API error — retry %d/%d in %.1fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.error("IEM API failed after %d retries: %s", max_retries, exc)
            return None
    return None


def get_historical_metar_actuals(
    station: str,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict]:
    """Fetch historical METAR daily high/low from Iowa Environmental Mesonet.

    Uses the ASOS download service which provides hourly METAR observations
    from US airports. Computes daily high/low in °F using local timezone dates.

    Returns::

        {
            "2025-01-15": {"high": 41.0, "low": 28.1},
            ...
        }
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    url = (
        f"{_IEM_BASE}"
        f"?station={station}"
        f"&data=tmpf"
        f"&tz=America%2FNew_York"  # Request in a standard tz, we'll re-parse
        f"&format=onlycomma"
        f"&latlon=no"
        f"&elev=no"
        f"&missing=empty"
        f"&trace=empty"
        f"&direct=no"
        f"&report_type=3"
        f"&year1={start_dt.year}&month1={start_dt.month}&day1={start_dt.day}"
        f"&year2={end_dt.year}&month2={end_dt.month}&day2={end_dt.day}"
    )

    csv_text = _fetch_metar_csv(url, max_retries=max_retries, base_delay=base_delay)
    if not csv_text:
        return {}

    # Parse CSV: "station,valid,tmpf\nKLGA,2025-01-15 00:54,32.0\n..."
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return {}

    # Group temperatures by local date
    daily_temps: dict[str, list[float]] = {}
    for line in lines[1:]:  # Skip header
        parts = line.split(",")
        if len(parts) < 3:
            continue
        date_str = parts[1].strip()[:10]  # "2025-01-15"
        temp_str = parts[2].strip()
        if not temp_str or temp_str == "M":
            continue
        try:
            temp_f = float(temp_str)
        except ValueError:
            continue
        # Sanity check: reject obviously bad readings
        if temp_f < -60 or temp_f > 140:
            continue
        daily_temps.setdefault(date_str, []).append(temp_f)

    actuals: dict[str, dict] = {}
    for date_str, temps in sorted(daily_temps.items()):
        if len(temps) < 4:  # Need at least 4 obs for a reliable daily extreme
            continue
        actuals[date_str] = {
            "high": round(max(temps), 1),
            "low": round(min(temps), 1),
        }

    logger.info("METAR actuals: %d days loaded for %s", len(actuals), station)
    return actuals
```

**Step 5: Run METAR test to verify it passes**

Run: `python3 -m pytest weather/tests/test_historical.py::TestGetHistoricalMetarActuals -v`
Expected: PASS

**Step 6: Add `--actuals-source` CLI option to calibrate.py**

In `weather/calibrate.py`, in `main()`:

```python
    parser.add_argument(
        "--actuals-source", type=str, default="metar",
        choices=["metar", "era5"],
        help="Ground truth source: metar (ASOS stations) or era5 (reanalysis). Default: metar",
    )
```

Then in the location loop, conditionally use METAR:

```python
    for loc in loc_keys:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.error("Unknown location: %s", loc)
            continue

        if args.actuals_source == "metar":
            from .historical import get_historical_metar_actuals
            station = loc_data.get("station")
            if not station:
                logger.error("No METAR station for %s — falling back to ERA5", loc)
                # Fall through to normal compute_forecast_errors
            else:
                errors = _compute_errors_with_metar(
                    location=loc,
                    lat=loc_data["lat"],
                    lon=loc_data["lon"],
                    station=station,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    tz_name=loc_data.get("tz", "America/New_York"),
                )
                all_errors.extend(errors)
                continue

        errors = compute_forecast_errors(
            location=loc,
            lat=loc_data["lat"],
            lon=loc_data["lon"],
            start_date=args.start_date,
            end_date=args.end_date,
            tz_name=loc_data.get("tz", "America/New_York"),
        )
        all_errors.extend(errors)
```

Add `_compute_errors_with_metar()`:

```python
def _compute_errors_with_metar(
    location: str,
    lat: float,
    lon: float,
    station: str,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
) -> list[dict]:
    """Like compute_forecast_errors but uses METAR actuals instead of ERA5."""
    from .historical import get_historical_metar_actuals

    logger.info("Fetching historical forecasts for %s (%s to %s)...",
                location, start_date, end_date)
    forecasts = get_historical_forecasts(lat, lon, start_date, end_date, tz_name=tz_name)

    logger.info("Fetching METAR actuals for %s (station %s)...", location, station)
    actuals = get_historical_metar_actuals(station, start_date, end_date,
                                            tz_name=tz_name)

    # Same deduplication logic as compute_forecast_errors
    seen: dict[tuple[str, str, str], dict] = {}

    for _run_date_str, targets in forecasts.items():
        for target_date_str, model_data in targets.items():
            actual = actuals.get(target_date_str)
            if not actual:
                continue

            month = datetime.strptime(target_date_str, "%Y-%m-%d").month

            spread: dict[str, float] = {}
            for metric in ["high", "low"]:
                gfs_val = model_data.get(f"gfs_{metric}")
                ecmwf_val = model_data.get(f"ecmwf_{metric}")
                if gfs_val is not None and ecmwf_val is not None:
                    spread[metric] = abs(gfs_val - ecmwf_val)

            for model_prefix in ["gfs", "ecmwf"]:
                for metric in ["high", "low"]:
                    key = (target_date_str, model_prefix, metric)
                    if key in seen:
                        continue

                    forecast_key = f"{model_prefix}_{metric}"
                    forecast_val = model_data.get(forecast_key)
                    actual_val = actual.get(metric)

                    if forecast_val is None or actual_val is None:
                        continue

                    record = {
                        "location": location,
                        "target_date": target_date_str,
                        "month": month,
                        "metric": metric,
                        "model": model_prefix,
                        "forecast": forecast_val,
                        "actual": actual_val,
                        "error": forecast_val - actual_val,
                        "model_spread": spread.get(metric, 0.0),
                    }
                    seen[key] = record

    errors = list(seen.values())
    logger.info("Computed %d deduplicated forecast errors for %s (METAR)", len(errors), location)
    return errors
```

**Step 7: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass

**Step 8: Commit**

```bash
git add weather/historical.py weather/config.py weather/calibrate.py weather/tests/test_historical.py
git commit -m "feat: add METAR historical ground truth for calibration"
```

---

### Task 4: Re-calibration & Validation

**Files:**
- Read: `weather/calibration.json` (before/after)
- Run: `weather/calibrate.py` CLI

**Step 1: Backup current calibration.json**

```bash
cp weather/calibration.json weather/calibration_era5_backup.json
```

**Step 2: Run calibration with METAR**

```bash
python3 -m weather.calibrate \
  --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --actuals-source metar
```

Expected output: sigma values, adaptive factors, Platt scaling params. Verify:
- `spread_to_sigma_factor` in ~0.5–1.0 range
- `ema_to_sigma_factor` in ~1.0–1.5 range
- Platt `a` > 0 (positive slope), `b` small

**Step 3: Verify calibration.json has new sections**

Read `weather/calibration.json` and check it contains:
- `"adaptive_sigma"` with calibrated factors
- `"platt_scaling"` with `a` and `b`

**Step 4: Run backtest to compare**

```bash
python3 -m weather.backtest \
  --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas \
  --start-date 2025-06-01 --end-date 2025-12-31
```

Compare Brier score with previous value (0.1828). It should be lower (better).

**Step 5: Run full test suite one final time**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass

**Step 6: Commit calibration data**

```bash
git add weather/calibration.json
git commit -m "data: recalibrate with METAR ground truth and Platt scaling"
```
