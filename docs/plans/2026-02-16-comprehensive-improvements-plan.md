# Weather Gully Comprehensive Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 12 improvements across robustness, risk management, alpha, and optimization — preparing the bot for live trading on Polymarket.

**Architecture:** Three-phase rollout. Phase 1 hardens the bot (numerical guards, circuit breaker, logging). Phase 2 unlocks new alpha (NO positions, feedback decay, metrics). Phase 3 optimizes performance (cache, parallelization, correlation). Each task is TDD: write failing test → implement → verify → commit.

**Tech Stack:** Python 3.14 stdlib only (no new deps). `unittest` for tests. `concurrent.futures` for parallelization. JSON file storage for state.

---

## Phase 1 — Robustness & Risk Management

### Task 1: Numerical guards for `_regularized_incomplete_beta`

**Files:**
- Modify: `weather/probability.py:103-157`
- Test: `weather/tests/test_probability.py`

**Step 1: Write failing tests**

Add to `weather/tests/test_probability.py`:

```python
class TestRegularizedIncompleteBeta(unittest.TestCase):
    """Direct tests for _regularized_incomplete_beta edge cases."""

    def test_invalid_a_zero(self):
        from weather.probability import _regularized_incomplete_beta
        # a=0 should not crash — return 0.0
        result = _regularized_incomplete_beta(0.5, 0, 1)
        self.assertIsInstance(result, float)

    def test_invalid_b_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1, 0)
        self.assertIsInstance(result, float)

    def test_invalid_a_negative(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, -1, 1)
        self.assertEqual(result, 0.0)

    def test_x_near_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1e-20, 5, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_x_near_one(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1 - 1e-20, 5, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestRegularizedIncompleteBeta -v`
Expected: FAIL (crash on `math.lgamma(0)` for `a=0`/`b=0` tests)

**Step 3: Implement guards in `_regularized_incomplete_beta`**

In `weather/probability.py`, add guards at the top of the function (after `x <= 0` / `x >= 1` checks):

```python
def _regularized_incomplete_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Guard: invalid parameters
    if a <= 0 or b <= 0:
        logger.warning("Invalid beta params a=%.4f b=%.4f — returning 0.0", a, b)
        return 0.0

    # Clamp x away from exact 0/1 to avoid log domain errors
    x = max(1e-15, min(1 - 1e-15, x))

    # ... rest of existing code ...

    # After the for loop, add convergence warning:
    else:
        # Loop completed without break — didn't converge
        logger.warning("Incomplete beta did not converge after %d iterations (x=%.6f, a=%.4f, b=%.4f)",
                        max_iter, x, a, b)

    return front * f
```

Note: The `else` clause on the `for` loop runs only if the loop finishes without `break`.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest weather/tests/test_probability.py::TestRegularizedIncompleteBeta -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "fix: add numerical guards to _regularized_incomplete_beta"
```

---

### Task 2: Numerical guards for `_student_t_cdf`

**Files:**
- Modify: `weather/probability.py:160-174`
- Test: `weather/tests/test_probability.py`

**Step 1: Write failing tests**

Add to `weather/tests/test_probability.py`:

```python
class TestStudentTCDFEdgeCases(unittest.TestCase):
    """Edge cases for _student_t_cdf robustness."""

    def test_extreme_positive(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1e10, 10)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_extreme_negative(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(-1e10, 10)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_positive_inf(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('inf'), 10)
        self.assertEqual(result, 1.0)

    def test_negative_inf(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('-inf'), 10)
        self.assertEqual(result, 0.0)

    def test_nan_returns_half(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('nan'), 10)
        self.assertEqual(result, 0.5)

    def test_df_zero(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1.0, 0)
        self.assertEqual(result, 0.5)

    def test_df_negative(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1.0, -5)
        self.assertEqual(result, 0.5)

    def test_df_fractional(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0, 0.5)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_df_very_large(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0, 1000)
        self.assertAlmostEqual(result, 0.5, places=4)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestStudentTCDFEdgeCases -v`
Expected: FAIL (inf/nan/df<=0 cases)

**Step 3: Implement guards in `_student_t_cdf`**

```python
def _student_t_cdf(x: float, df: float) -> float:
    """CDF of standard Student's t-distribution."""
    # Guard: invalid inputs
    if math.isnan(x) or df <= 0:
        logger.warning("Invalid student_t_cdf input: x=%s df=%s — returning 0.5", x, df)
        return 0.5
    if math.isinf(x):
        return 1.0 if x > 0 else 0.0

    if df > 100:
        return _normal_cdf(x)
    t2 = x * x
    z = df / (df + t2)
    ibeta = _regularized_incomplete_beta(z, df / 2.0, 0.5)
    if x >= 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_probability.py::TestStudentTCDFEdgeCases weather/tests/test_probability.py::TestStudentTCDF -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "fix: add numerical guards to _student_t_cdf (inf, nan, df<=0)"
```

---

### Task 3: Cap probability output and sigma upper bound

**Files:**
- Modify: `weather/probability.py:485` and `weather/probability.py:673`
- Test: `weather/tests/test_probability.py`

**Step 1: Write failing test**

```python
class TestProbabilityOutputBounds(unittest.TestCase):
    """Ensure estimate_bucket_probability always returns [0, 1]."""

    def test_never_exceeds_one(self):
        # With very tight sigma around center of a wide bucket, could theoretically exceed 1.0
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=-999, bucket_high=999,
            forecast_date="2026-02-16", sigma_override=0.001,
        )
        self.assertLessEqual(result, 1.0)

    def test_zero_sigma_returns_valid(self):
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=0.0,
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_huge_sigma_returns_valid(self):
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=1000.0,
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
```

**Step 2: Run tests — verify fail or pass**

Run: `python3 -m pytest weather/tests/test_probability.py::TestProbabilityOutputBounds -v`

**Step 3: Add cap in both functions**

In `weather/probability.py`, `estimate_bucket_probability` line 485:
```python
    prob = max(0.0, min(1.0, cdf_high - cdf_low))
    return round(prob, 4)
```

Same change in `estimate_bucket_probability_with_obs` line 673:
```python
    prob = max(0.0, min(1.0, cdf_high - cdf_low))
    return round(prob, 4)
```

Add sigma upper bound after `sigma <= 0` guard (both functions):
```python
    if sigma <= 0:
        sigma = 0.01
    sigma = min(sigma, 50.0)
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_probability.py::TestProbabilityOutputBounds -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "fix: cap probability output to [0,1] and add sigma upper bound"
```

---

### Task 4: Logging for calibration fallback chain

**Files:**
- Modify: `weather/probability.py:23-53` (`_load_calibration`) and `weather/probability.py:212-241` (`_get_stddev`)
- Test: `weather/tests/test_probability.py`

**Step 1: Write failing test**

```python
class TestCalibrationLogging(unittest.TestCase):
    """Verify calibration fallback generates warnings."""

    @patch("weather.probability._CALIBRATION_PATH")
    def test_missing_file_logs_warning(self, mock_path):
        import weather.probability as prob
        mock_path.exists.return_value = False
        mock_path.stat.side_effect = OSError("No file")
        # Reset cache to force reload
        prob._calibration_cache = None
        prob._calibration_mtime = 0.0
        with self.assertLogs("weather.probability", level="WARNING") as cm:
            prob._load_calibration()
        self.assertTrue(any("not found" in msg or "fallback" in msg for msg in cm.output))
```

**Step 2: Run test — should fail (no warning logged currently)**

Run: `python3 -m pytest weather/tests/test_probability.py::TestCalibrationLogging -v`

**Step 3: Add warning logs**

In `_load_calibration()`, after the `if _CALIBRATION_PATH.exists():` block's else branch:

```python
    if _CALIBRATION_PATH.exists():
        # ... existing try/except ...
    else:
        logger.warning("Calibration file not found at %s — using hardcoded fallbacks", _CALIBRATION_PATH)

    _calibration_cache = {}
    _calibration_mtime = current_mtime
    return _calibration_cache
```

In `_get_stddev()`, add DEBUG logging for which sigma source was used:

```python
    # 1. Location-specific sigma
    if location and cal:
        loc_sigma = cal.get("location_sigma", {}).get(location, {})
        if horizon_key in loc_sigma:
            logger.debug("Using calibrated location sigma for %s h=%s: %.2f", location, horizon_key, float(loc_sigma[horizon_key]))
            return float(loc_sigma[horizon_key])

    # 2. Global calibrated sigma
    if cal:
        global_sigma = cal.get("global_sigma", {})
        if horizon_key in global_sigma:
            logger.debug("Using global calibrated sigma for h=%s: %.2f (no location data for %s)", horizon_key, float(global_sigma[horizon_key]), location)
            return float(global_sigma[horizon_key])

    # 3. Hardcoded fallback
    logger.debug("Using hardcoded fallback sigma for h=%d", days_ahead)
```

**Step 4: Run test**

Run: `python3 -m pytest weather/tests/test_probability.py::TestCalibrationLogging -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "feat: add warning logging for calibration fallback chain"
```

---

### Task 5: Calibration age validation

**Files:**
- Modify: `weather/probability.py:23-53`
- Test: `weather/tests/test_probability.py`

**Step 1: Write failing test**

```python
class TestCalibrationAge(unittest.TestCase):
    """Verify stale calibration generates warnings."""

    def test_old_calibration_warns(self):
        import weather.probability as prob
        import time
        # Create temp calibration file with old metadata
        old_meta = {
            "_metadata": {
                "generated_at": "2025-06-01T00:00:00Z",
            },
            "global_sigma": {"0": 2.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_meta, f)
            tmp_path = f.name
        try:
            orig_path = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            with self.assertLogs("weather.probability", level="WARNING") as cm:
                prob._load_calibration()
            self.assertTrue(any("days old" in msg for msg in cm.output))
        finally:
            prob._CALIBRATION_PATH = orig_path
            prob._calibration_cache = None
            os.unlink(tmp_path)
```

**Step 2: Run test — should fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestCalibrationAge -v`

**Step 3: Add age check in `_load_calibration`**

After successfully loading the JSON (inside the try block, after `logger.info`):

```python
            # Check calibration age
            generated_at = _calibration_cache.get("_metadata", {}).get("generated_at")
            if not generated_at:
                generated_at = _calibration_cache.get("metadata", {}).get("generated_at")
            if generated_at:
                try:
                    gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - gen_dt).days
                    if age_days > 90:
                        logger.error("Calibration is %d days old — STALE (regenerate with: python3 -m weather.calibrate)", age_days)
                    elif age_days > 30:
                        logger.warning("Calibration is %d days old — consider re-running calibrate", age_days)
                except (ValueError, TypeError):
                    pass
```

**Step 4: Run test**

Run: `python3 -m pytest weather/tests/test_probability.py::TestCalibrationAge -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/tests/test_probability.py
git commit -m "feat: warn when calibration data is stale (>30 days)"
```

---

### Task 6: Circuit breaker — config parameters

**Files:**
- Modify: `weather/config.py:38-111`
- Test: `weather/tests/test_config.py` (check if exists, else create minimal test)

**Step 1: Add new config fields**

In `weather/config.py`, add to the `Config` dataclass after `fill_poll_interval`:

```python
    # Circuit breaker (risk management)
    daily_loss_limit: float = 10.0           # Stop trading after $X daily loss
    max_positions_per_day: int = 20           # Max new positions per calendar day
    cooldown_hours_after_max_loss: float = 24.0  # Hours to wait after circuit break
    max_open_positions: int = 15              # Max simultaneous open positions
```

**Step 2: Write test**

```python
class TestCircuitBreakerConfig(unittest.TestCase):
    def test_defaults_exist(self):
        from weather.config import Config
        c = Config()
        self.assertEqual(c.daily_loss_limit, 10.0)
        self.assertEqual(c.max_positions_per_day, 20)
        self.assertEqual(c.cooldown_hours_after_max_loss, 24.0)
        self.assertEqual(c.max_open_positions, 15)
```

**Step 3: Run test**

Run: `python3 -m pytest weather/tests/test_config.py::TestCircuitBreakerConfig -v` (or inline in test_probability.py if no test_config.py)

**Step 4: Commit**

```bash
git add weather/config.py weather/tests/test_config.py
git commit -m "feat: add circuit breaker config parameters"
```

---

### Task 7: Circuit breaker — state tracking

**Files:**
- Modify: `weather/state.py`
- Test: `weather/tests/test_state.py`

**Step 1: Write failing tests**

```python
class TestCircuitBreakerState(unittest.TestCase):
    def test_daily_pnl_default_zero(self):
        state = TradingState()
        self.assertEqual(state.get_daily_pnl("2026-02-16"), 0.0)

    def test_record_daily_pnl(self):
        state = TradingState()
        state.record_daily_pnl("2026-02-16", -3.50)
        state.record_daily_pnl("2026-02-16", -2.00)
        self.assertAlmostEqual(state.get_daily_pnl("2026-02-16"), -5.50)

    def test_positions_opened_today(self):
        state = TradingState()
        state.record_position_opened("2026-02-16")
        state.record_position_opened("2026-02-16")
        self.assertEqual(state.positions_opened_today("2026-02-16"), 2)

    def test_circuit_break_timestamp(self):
        state = TradingState()
        self.assertIsNone(state.last_circuit_break)
        state.last_circuit_break = "2026-02-16T12:00:00+00:00"
        self.assertIsNotNone(state.last_circuit_break)

    def test_daily_pnl_serialization(self):
        state = TradingState()
        state.record_daily_pnl("2026-02-16", -5.0)
        state.record_position_opened("2026-02-16")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            loaded = TradingState.load(path)
            self.assertAlmostEqual(loaded.get_daily_pnl("2026-02-16"), -5.0)
            self.assertEqual(loaded.positions_opened_today("2026-02-16"), 1)
        finally:
            os.unlink(path)
```

**Step 2: Run tests — should fail (methods don't exist yet)**

Run: `python3 -m pytest weather/tests/test_state.py::TestCircuitBreakerState -v`

**Step 3: Add circuit breaker state to `TradingState`**

In `weather/state.py`, add new fields to the dataclass:

```python
@dataclass
class TradingState:
    trades: dict[str, TradeRecord] = field(default_factory=dict)
    last_run: str = ""
    previous_forecasts: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, PredictionRecord] = field(default_factory=dict)
    event_positions: dict[str, str] = field(default_factory=dict)
    daily_observations: dict[str, dict] = field(default_factory=dict)
    # Circuit breaker
    daily_pnl: dict[str, float] = field(default_factory=dict)          # date_str → cumulative P&L
    daily_positions_count: dict[str, int] = field(default_factory=dict) # date_str → count
    last_circuit_break: str | None = None
```

Add methods:

```python
    def get_daily_pnl(self, date_str: str) -> float:
        return self.daily_pnl.get(date_str, 0.0)

    def record_daily_pnl(self, date_str: str, amount: float) -> None:
        self.daily_pnl[date_str] = self.daily_pnl.get(date_str, 0.0) + amount

    def positions_opened_today(self, date_str: str) -> int:
        return self.daily_positions_count.get(date_str, 0)

    def record_position_opened(self, date_str: str) -> None:
        self.daily_positions_count[date_str] = self.daily_positions_count.get(date_str, 0) + 1
```

Update `save()` to include the new fields in the data dict:

```python
    "daily_pnl": self.daily_pnl,
    "daily_positions_count": self.daily_positions_count,
    "last_circuit_break": self.last_circuit_break,
```

Update `load()` to read them back:

```python
    daily_pnl = data.get("daily_pnl", {})
    daily_positions_count = data.get("daily_positions_count", {})
    last_circuit_break = data.get("last_circuit_break")
```

And pass them to the constructor.

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_state.py::TestCircuitBreakerState -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/state.py weather/tests/test_state.py
git commit -m "feat: add circuit breaker state tracking (daily PnL, position count)"
```

---

### Task 8: Circuit breaker — strategy enforcement

**Files:**
- Modify: `weather/strategy.py:454-490` (top of `run_weather_strategy`) and `~870-880` (trade execution)
- Test: `weather/tests/test_strategy.py`

**Step 1: Write failing tests**

```python
class TestCircuitBreaker(unittest.TestCase):
    def test_daily_loss_stops_trading(self):
        """If daily P&L exceeds limit, no trades should execute."""
        # Setup: state with -$11 daily P&L
        state = TradingState()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state.record_daily_pnl(today, -11.0)
        config = Config()
        config.daily_loss_limit = 10.0
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)
        self.assertIn("daily loss", reason.lower())

    def test_max_positions_stops_trading(self):
        state = TradingState()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for _ in range(21):
            state.record_position_opened(today)
        config = Config()
        config.max_positions_per_day = 20
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_cooldown_blocks_after_circuit_break(self):
        state = TradingState()
        state.last_circuit_break = datetime.now(timezone.utc).isoformat()
        config = Config()
        config.cooldown_hours_after_max_loss = 24.0
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_max_open_positions_blocks(self):
        state = TradingState()
        for i in range(16):
            state.record_trade(f"market_{i}", f"outcome_{i}", "yes", 0.1, 10)
        config = Config()
        config.max_open_positions = 15
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_no_block_when_within_limits(self):
        state = TradingState()
        config = Config()
        blocked, reason = check_circuit_breaker(state, config)
        self.assertFalse(blocked)
```

**Step 2: Run tests — should fail**

Run: `python3 -m pytest weather/tests/test_strategy.py::TestCircuitBreaker -v`

**Step 3: Implement `check_circuit_breaker` in `strategy.py`**

Add new function after `check_context_safeguards`:

```python
def check_circuit_breaker(state: TradingState, config: Config) -> tuple[bool, str]:
    """Check global circuit breaker conditions.

    Returns (blocked: bool, reason: str). If blocked, all trading should halt.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Daily loss limit
    daily_pnl = state.get_daily_pnl(today)
    if daily_pnl <= -config.daily_loss_limit:
        return True, f"Daily loss limit hit: ${daily_pnl:.2f} (limit -${config.daily_loss_limit:.2f})"

    # 2. Max positions per day
    positions_today = state.positions_opened_today(today)
    if positions_today >= config.max_positions_per_day:
        return True, f"Max positions per day: {positions_today} (limit {config.max_positions_per_day})"

    # 3. Cooldown after last circuit break
    if state.last_circuit_break:
        try:
            last_break = datetime.fromisoformat(state.last_circuit_break)
            hours_since = (datetime.now(timezone.utc) - last_break).total_seconds() / 3600
            if hours_since < config.cooldown_hours_after_max_loss:
                return True, f"Cooldown active: {hours_since:.1f}h since circuit break (need {config.cooldown_hours_after_max_loss}h)"
        except (ValueError, TypeError):
            pass

    # 4. Max open positions
    if len(state.trades) >= config.max_open_positions:
        return True, f"Max open positions: {len(state.trades)} (limit {config.max_open_positions})"

    return False, ""
```

Then call it at the top of `run_weather_strategy`, after state loading and before the event loop:

```python
    # Circuit breaker check
    blocked, reason = check_circuit_breaker(state, config)
    if blocked:
        logger.warning("CIRCUIT BREAKER: %s", reason)
        state.last_circuit_break = datetime.now(timezone.utc).isoformat()
        state.save(save_path)
        return
```

Also update the trade execution block (line ~896) to call `state.record_position_opened(today_str)` after a successful trade.

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_strategy.py::TestCircuitBreaker -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: All pass

**Step 6: Commit**

```bash
git add weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: add global circuit breaker (daily loss, max positions, cooldown)"
```

---

## Phase 2 — Alpha Enhancement

### Task 9: Score NO-side opportunities in `score_buckets`

**Files:**
- Modify: `weather/strategy.py:121-209` (`score_buckets`)
- Test: `weather/tests/test_strategy.py`

**Step 1: Write failing tests**

```python
class TestScoreBucketsNO(unittest.TestCase):
    def test_no_side_scored_when_overpriced(self):
        """A bucket with prob=0.10 and price=0.80 should generate a NO opportunity."""
        markets = [{
            "id": "m1", "outcome_name": "90°F or higher",
            "best_ask": 0.80, "external_price_yes": 0.80,
        }]
        config = Config()
        config.min_ev_threshold = 0.03
        config.min_probability = 0.05
        with patch("weather.strategy.parse_temperature_bucket", return_value=(90, 999)):
            with patch("weather.strategy.estimate_bucket_probability", return_value=0.10):
                with patch("weather.strategy.platt_calibrate", side_effect=lambda p: p):
                    scored = score_buckets(markets, 70.0, "2026-02-20", config, metric="high")
        no_entries = [s for s in scored if s.get("side") == "no"]
        self.assertTrue(len(no_entries) > 0)
        self.assertGreater(no_entries[0]["ev"], 0)
```

**Step 2: Run test — should fail**

Run: `python3 -m pytest weather/tests/test_strategy.py::TestScoreBucketsNO -v`

**Step 3: Add NO-side scoring to `score_buckets`**

In `weather/strategy.py`, modify `score_buckets` to also evaluate the NO side. After computing `prob` and `prob_raw`, add:

```python
        # YES side (existing logic)
        ev_yes = prob * (1.0 - config.trading_fees) - price
        if prob >= config.min_probability:
            scored.append({
                "market": market,
                "bucket": bucket,
                "outcome_name": outcome_name,
                "prob": prob,
                "prob_raw": prob_raw,
                "price": price,
                "ev": ev_yes,
                "side": "yes",
            })

        # NO side: if bucket is overpriced, buying NO is profitable
        no_prob = 1.0 - prob    # P(not in this bucket)
        no_price = 1.0 - price  # Price of NO token
        if no_price >= MIN_TICK_SIZE and no_prob >= config.min_probability:
            ev_no = no_prob * (1.0 - config.trading_fees) - no_price
            if ev_no > 0:
                scored.append({
                    "market": market,
                    "bucket": bucket,
                    "outcome_name": outcome_name,
                    "prob": no_prob,
                    "prob_raw": 1.0 - prob_raw,
                    "price": no_price,
                    "ev": ev_no,
                    "side": "no",
                })
```

Remove the old `scored.append(...)` block that didn't include `"side"` and ensure all existing entries include `"side": "yes"`.

**Step 4: Run test**

Run: `python3 -m pytest weather/tests/test_strategy.py::TestScoreBucketsNO -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: score NO-side opportunities in score_buckets"
```

---

### Task 10: Execute NO-side trades in strategy loop

**Files:**
- Modify: `weather/strategy.py:823-955` (trade execution block)
- Test: `weather/tests/test_strategy.py`

**Step 1: Write test (integration-style with mocks)**

```python
class TestExecuteNOTrade(unittest.TestCase):
    def test_no_trade_uses_correct_side(self):
        """When executing a NO-side entry, the bridge should receive side='no'."""
        # This is a higher-level integration test; verify the side param flows through
        # by checking that state.record_trade gets called with side="no"
        from weather.state import TradingState
        state = TradingState()
        state.record_trade("m1", "bucket_name", "no", 0.20, 10.0,
                           location="NYC", forecast_date="2026-02-20")
        self.assertEqual(state.trades["m1"].side, "no")
```

**Step 2: Implement**

In the trade execution block (`strategy.py` ~line 823-955), change:
- Use `entry["side"]` instead of hardcoded `"yes"` for `client.execute_trade` and `state.record_trade`
- For NO trades: `position_size` uses `compute_position_size(no_prob, no_price, ...)`
- The `entry_threshold` check should compare against the correct price (YES price for YES, NO price for NO)

Key changes:

```python
    side = entry.get("side", "yes")
    # ...
    result = client.execute_trade(market_id, side, position_size, ...)
    # ...
    state.record_trade(
        market_id=market_id,
        outcome_name=outcome_name,
        side=side,
        cost_basis=price,
        # ...
    )
```

**Step 3: Run full tests**

Run: `python3 -m pytest weather/tests/ -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: execute NO-side trades (buy NO token for overpriced buckets)"
```

---

### Task 11: Sell existing YES positions on edge inversion

**Files:**
- Modify: `weather/strategy.py` (in `check_exit_opportunities` or new function)
- Test: `weather/tests/test_strategy.py`

**Step 1: Write test**

```python
class TestEdgeInversionExit(unittest.TestCase):
    def test_sell_when_model_disagrees(self):
        """If we hold YES at 0.10 and current prob=0.05 while market=0.15, should exit."""
        from weather.strategy import should_exit_on_edge_inversion
        should_exit = should_exit_on_edge_inversion(
            our_prob=0.05, market_price=0.15, cost_basis=0.10, side="yes",
        )
        self.assertTrue(should_exit)

    def test_no_exit_when_edge_holds(self):
        should_exit = should_exit_on_edge_inversion(
            our_prob=0.30, market_price=0.10, cost_basis=0.08, side="yes",
        )
        self.assertFalse(should_exit)
```

**Step 2: Implement**

```python
def should_exit_on_edge_inversion(
    our_prob: float,
    market_price: float,
    cost_basis: float,
    side: str = "yes",
    min_loss_to_trigger: float = 0.02,
) -> bool:
    """Check if the edge has inverted, suggesting we should exit."""
    if side == "yes":
        # We bought YES. Exit if: market values it higher than our model says
        # AND we can sell at profit or small loss
        return market_price > our_prob and market_price >= cost_basis - min_loss_to_trigger
    else:
        # We bought NO. Exit if model now agrees with YES side
        return our_prob > (1 - market_price) and (1 - market_price) >= cost_basis - min_loss_to_trigger
```

Integrate into `check_exit_opportunities`.

**Step 3: Run tests**

Run: `python3 -m pytest weather/tests/test_strategy.py -v -k "EdgeInversion"`
Expected: PASS

**Step 4: Commit**

```bash
git add weather/strategy.py weather/tests/test_strategy.py
git commit -m "feat: exit positions on edge inversion (model probability shifted)"
```

---

### Task 12: Externalize horizon growth factors

**Files:**
- Modify: `weather/calibrate.py:48-60`
- Modify: `weather/probability.py:212-241` (`_get_stddev`)
- Test: `weather/tests/test_probability.py`

**Step 1: Write test**

```python
class TestHorizonGrowthFromCalibration(unittest.TestCase):
    def test_reads_growth_from_json(self):
        """If calibration.json has horizon_growth, use it instead of hardcoded."""
        cal_data = {
            "global_sigma": {"0": 2.0},
            "horizon_growth": {"0": 1.0, "1": 1.5, "2": 2.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            import weather.probability as prob
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            # horizon 1 should use growth 1.5: 2.0 * 1.5 = 3.0
            sigma = prob._get_stddev("2026-02-17", horizon_override=1)
            self.assertAlmostEqual(sigma, 3.0, places=1)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            os.unlink(tmp_path)
```

**Step 2: Run test — should fail**

Run: `python3 -m pytest weather/tests/test_probability.py::TestHorizonGrowthFromCalibration -v`

**Step 3: Implement**

In `weather/probability.py`, modify `_get_stddev` to use `horizon_growth` from calibration:

```python
def _get_stddev(forecast_date: str, location: str = "", horizon_override: int | None = None) -> float:
    days_ahead = horizon_override if horizon_override is not None else get_horizon_days(forecast_date)
    cal = _load_calibration()
    horizon_key = str(days_ahead)

    # 1. Location-specific sigma (already horizon-expanded)
    if location and cal:
        loc_sigma = cal.get("location_sigma", {}).get(location, {})
        if horizon_key in loc_sigma:
            return float(loc_sigma[horizon_key])

    # 2. Global base sigma × horizon growth factor
    if cal:
        global_sigma = cal.get("global_sigma", {})
        if horizon_key in global_sigma:
            return float(global_sigma[horizon_key])
        # Fallback: base sigma (h=0) × growth factor from JSON
        base_sigma_str = global_sigma.get("0")
        horizon_growth = cal.get("horizon_growth", {})
        if base_sigma_str and horizon_key in horizon_growth:
            return float(base_sigma_str) * float(horizon_growth[horizon_key])

    # 3. Hardcoded fallback
    if days_ahead <= 10:
        return _HORIZON_STDDEV.get(days_ahead, 11.8)
    return min(18.0, 11.8 + 0.7 * (days_ahead - 10))
```

In `weather/calibrate.py`, add horizon_growth to the output JSON in `build_calibration_tables`:

```python
    result["horizon_growth"] = {str(h): _horizon_growth_factor(h) for h in range(11)}
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_probability.py::TestHorizonGrowthFromCalibration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/probability.py weather/calibrate.py weather/tests/test_probability.py
git commit -m "feat: externalize horizon growth factors to calibration.json"
```

---

### Task 13: Exponential decay in feedback loop

**Files:**
- Modify: `weather/feedback.py`
- Test: `weather/tests/test_feedback.py`

**Step 1: Write failing tests**

```python
class TestFeedbackDecay(unittest.TestCase):
    def test_stale_entry_decays(self):
        """An entry not updated in 100+ days should be treated as no data."""
        from weather.feedback import FeedbackEntry, FeedbackState
        state = FeedbackState()
        # Manually create an old entry
        entry = FeedbackEntry(bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
                              last_updated="2025-10-01T00:00:00+00:00")
        state.entries["NYC|winter"] = entry
        # With decay, 140+ days old → should return None
        result = state.get_bias("NYC", 1)
        self.assertIsNone(result)

    def test_recent_entry_not_decayed(self):
        from weather.feedback import FeedbackEntry, FeedbackState
        from datetime import datetime, timezone
        state = FeedbackState()
        entry = FeedbackEntry(bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
                              last_updated=datetime.now(timezone.utc).isoformat())
        state.entries["NYC|winter"] = entry
        result = state.get_bias("NYC", 1)
        self.assertIsNotNone(result)
        # Should be close to 2.0 (minimal decay)
        self.assertAlmostEqual(result, 2.0, delta=0.5)
```

**Step 2: Run test — should fail (no `last_updated` field)**

Run: `python3 -m pytest weather/tests/test_feedback.py::TestFeedbackDecay -v`

**Step 3: Implement decay**

In `weather/feedback.py`:

1. Add `last_updated` field to `FeedbackEntry`:
```python
@dataclass
class FeedbackEntry:
    bias_ema: float = 0.0
    abs_error_ema: float = 0.0
    sample_count: int = 0
    last_updated: str = ""  # ISO timestamp of last update
```

2. Update `FeedbackEntry.update` to set timestamp:
```python
    def update(self, forecast_temp: float, actual_temp: float) -> None:
        from datetime import datetime, timezone
        error = forecast_temp - actual_temp
        abs_error = abs(error)
        self.sample_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()
        # ... rest unchanged ...
```

3. Add decay method:
```python
_HALF_LIFE_DAYS = 30.0
_DECAY_FLOOR = 0.1  # Below this, treat as no data

    def decay_factor(self) -> float:
        """Compute time-based decay factor. Returns 0.0-1.0."""
        if not self.last_updated:
            return 0.0
        try:
            from datetime import datetime, timezone
            last = datetime.fromisoformat(self.last_updated)
            days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
            return 0.5 ** (days / _HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            return 0.0
```

4. Modify `get_bias` and `get_abs_error_ema` to apply decay:
```python
    def get_bias(self, location: str, month: int) -> float | None:
        key = _season_key(location, month)
        entry = self.entries.get(key)
        if entry and entry.has_enough_data:
            decay = entry.decay_factor()
            if decay < _DECAY_FLOOR:
                return None
            return entry.bias_ema * decay
        return None
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_feedback.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add weather/feedback.py weather/tests/test_feedback.py
git commit -m "feat: add exponential decay to feedback loop (half-life 30 days)"
```

---

### Task 14: Metrics module

**Files:**
- Create: `weather/metrics.py`
- Test: `weather/tests/test_metrics.py`

**Step 1: Write failing tests**

```python
class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        from weather.metrics import brier_score
        predictions = [(1.0, True), (0.0, False), (1.0, True)]
        self.assertAlmostEqual(brier_score(predictions), 0.0)

    def test_worst_predictions(self):
        from weather.metrics import brier_score
        predictions = [(0.0, True), (1.0, False)]
        self.assertAlmostEqual(brier_score(predictions), 1.0)

    def test_empty_returns_none(self):
        from weather.metrics import brier_score
        self.assertIsNone(brier_score([]))


class TestSharpeRatio(unittest.TestCase):
    def test_positive_sharpe(self):
        from weather.metrics import sharpe_ratio
        returns = [0.05, 0.03, 0.04, 0.06, 0.02]
        result = sharpe_ratio(returns)
        self.assertGreater(result, 0)

    def test_empty_returns_none(self):
        from weather.metrics import sharpe_ratio
        self.assertIsNone(sharpe_ratio([]))


class TestCalibrationTable(unittest.TestCase):
    def test_bins_predictions(self):
        from weather.metrics import calibration_table
        predictions = [(0.15, True), (0.15, False), (0.85, True), (0.85, True)]
        table = calibration_table(predictions)
        # Bin 0.1-0.2 should have 50% actual
        self.assertIn("0.1-0.2", table)
        self.assertAlmostEqual(table["0.1-0.2"]["actual_freq"], 0.5)
```

**Step 2: Run tests — should fail**

Run: `python3 -m pytest weather/tests/test_metrics.py -v`

**Step 3: Implement `weather/metrics.py`**

```python
"""Forecast quality metrics — Brier score, Sharpe ratio, calibration table."""

import math


def brier_score(predictions: list[tuple[float, bool]]) -> float | None:
    """Brier score: mean((prob - outcome)^2). Lower is better. 0 = perfect."""
    if not predictions:
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in predictions) / len(predictions)


def sharpe_ratio(returns: list[float], annualize_factor: float = 365.0) -> float | None:
    """Annualized Sharpe ratio from a list of per-trade returns."""
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-10
    return (mean_r / std) * math.sqrt(annualize_factor)


def win_rate(returns: list[float]) -> float | None:
    """Fraction of trades with positive return."""
    if not returns:
        return None
    return sum(1 for r in returns if r > 0) / len(returns)


def average_edge(edges: list[float]) -> float | None:
    """Mean absolute edge |prob - market_price| on trades taken."""
    if not edges:
        return None
    return sum(abs(e) for e in edges) / len(edges)


def calibration_table(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> dict[str, dict]:
    """Bin predictions and compute actual frequency per bin."""
    bins: dict[str, list[bool]] = {}
    step = 1.0 / n_bins
    for prob, outcome in predictions:
        bin_idx = min(int(prob / step), n_bins - 1)
        lo = round(bin_idx * step, 1)
        hi = round((bin_idx + 1) * step, 1)
        key = f"{lo}-{hi}"
        bins.setdefault(key, []).append(outcome)

    table = {}
    for key, outcomes in sorted(bins.items()):
        n = len(outcomes)
        actual = sum(1 for o in outcomes if o) / n if n else 0
        table[key] = {"count": n, "actual_freq": round(actual, 4)}
    return table
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add weather/metrics.py weather/tests/test_metrics.py
git commit -m "feat: add metrics module (Brier, Sharpe, calibration table, win rate)"
```

---

### Task 15: CLI report

**Files:**
- Create: `weather/report.py`
- Test: Manual (CLI output)

**Step 1: Implement `weather/report.py`**

```python
"""CLI report — print trading state summary, P&L, and forecast quality metrics.

Usage: python3 -m weather.report [--state-file PATH] [--feedback-file PATH]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .metrics import brier_score, calibration_table, sharpe_ratio, win_rate
from .state import TradingState


def _load_trade_log(path: str = "weather/trade_log.jsonl") -> list[dict]:
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


def main():
    parser = argparse.ArgumentParser(description="Weather Gully trading report")
    parser.add_argument("--state-file", default="weather/paper_state.json")
    parser.add_argument("--trade-log", default="weather/trade_log.jsonl")
    args = parser.parse_args()

    state = TradingState.load(args.state_file)
    trade_log = _load_trade_log(args.trade_log)

    print("=" * 60)
    print("  WEATHER GULLY — TRADING REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Open positions
    print(f"\nOpen positions: {len(state.trades)}")
    for mid, trade in state.trades.items():
        print(f"  {trade.location} {trade.forecast_date} {trade.outcome_name} "
              f"[{trade.side}] @ ${trade.cost_basis:.2f} ({trade.shares:.0f} shares)")

    # Calibration stats
    cal_stats = state.get_calibration_stats()
    print(f"\nResolved predictions: {cal_stats['count']}")
    if cal_stats["brier"] is not None:
        print(f"  Brier score: {cal_stats['brier']:.4f}")
        print(f"  Accuracy:    {cal_stats['accuracy']:.1%}")

    # Predictions for metrics
    resolved = [p for p in state.predictions.values() if p.resolved and p.actual_outcome is not None]
    if resolved:
        preds = [(p.our_probability, p.actual_outcome) for p in resolved]
        bs = brier_score(preds)
        print(f"  Brier (full): {bs:.4f}" if bs is not None else "")

        cal = calibration_table(preds)
        print("\n  Calibration table:")
        for bin_key, data in cal.items():
            print(f"    {bin_key}: n={data['count']}, actual={data['actual_freq']:.1%}")

    # Trade log stats
    if trade_log:
        print(f"\nTrade log entries: {len(trade_log)}")

    # Daily P&L
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = state.get_daily_pnl(today)
    print(f"\nDaily P&L ({today}): ${daily:+.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Test manually**

Run: `python3 -m weather.report --state-file weather/paper_state.json`
Expected: Formatted report output

**Step 3: Commit**

```bash
git add weather/report.py
git commit -m "feat: add CLI report (positions, P&L, Brier, calibration table)"
```

---

## Phase 3 — Optimization

### Task 16: Cache for API responses

**Files:**
- Modify: `weather/open_meteo.py:50-70`
- Test: `weather/tests/test_open_meteo.py`

**Step 1: Write failing test**

```python
class TestFetchCache(unittest.TestCase):
    def test_cached_response_not_refetched(self):
        from weather.open_meteo import _fetch_json_cached, _cache
        _cache.clear()
        url = "https://example.com/test"
        with patch("weather.open_meteo._fetch_json", return_value={"temp": 50}) as mock:
            result1 = _fetch_json_cached(url)
            result2 = _fetch_json_cached(url)
        self.assertEqual(mock.call_count, 1)  # Only one actual fetch
        self.assertEqual(result1, result2)
```

**Step 2: Implement**

In `weather/open_meteo.py`, add at module level:

```python
import time as _time

_cache: dict[str, tuple[float, dict | None]] = {}
_CACHE_TTL = 900  # 15 minutes


def _fetch_json_cached(url: str, ttl: float = _CACHE_TTL, **kwargs) -> dict | None:
    """Fetch JSON with in-memory TTL cache."""
    now = _time.monotonic()
    if url in _cache:
        cached_time, cached_data = _cache[url]
        if now - cached_time < ttl:
            logger.debug("Cache hit for %s", url[:80])
            return cached_data
    data = _fetch_json(url, **kwargs)
    _cache[url] = (now, data)
    return data
```

Then replace `_fetch_json` calls in `get_open_meteo_forecast` and `get_open_meteo_forecast_multi` with `_fetch_json_cached`.

**Step 3: Run tests**

Run: `python3 -m pytest weather/tests/test_open_meteo.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add weather/open_meteo.py weather/tests/test_open_meteo.py
git commit -m "feat: add in-memory TTL cache for Open-Meteo API responses (15 min)"
```

---

### Task 17: Correlation-adjusted position sizing

**Files:**
- Create: `weather/correlation.py`
- Modify: `weather/strategy.py`
- Test: `weather/tests/test_correlation.py`

**Step 1: Write failing tests**

```python
class TestCorrelationMatrix(unittest.TestCase):
    def test_self_correlation_is_one(self):
        from weather.correlation import get_correlation
        self.assertEqual(get_correlation("NYC", "NYC"), 1.0)

    def test_symmetric(self):
        from weather.correlation import get_correlation
        self.assertEqual(get_correlation("NYC", "Atlanta"), get_correlation("Atlanta", "NYC"))

    def test_nyc_atlanta_moderate(self):
        from weather.correlation import get_correlation
        corr = get_correlation("NYC", "Atlanta")
        self.assertGreater(corr, 0.3)
        self.assertLess(corr, 0.7)

    def test_seattle_independent(self):
        from weather.correlation import get_correlation
        corr = get_correlation("Seattle", "NYC")
        self.assertLess(corr, 0.2)


class TestCorrelationAdjustedSizing(unittest.TestCase):
    def test_reduces_size_with_correlated_positions(self):
        from weather.correlation import adjust_position_size
        # Already holding NYC position, entering Atlanta (corr ~0.5)
        adjusted = adjust_position_size(
            base_size=10.0,
            new_location="Atlanta",
            existing_locations=["NYC"],
        )
        self.assertLess(adjusted, 10.0)
        self.assertGreater(adjusted, 5.0)

    def test_no_reduction_with_uncorrelated(self):
        from weather.correlation import adjust_position_size
        adjusted = adjust_position_size(
            base_size=10.0,
            new_location="Seattle",
            existing_locations=["NYC"],
        )
        self.assertAlmostEqual(adjusted, 10.0, delta=1.0)
```

**Step 2: Implement `weather/correlation.py`**

```python
"""Inter-location correlation for position sizing adjustment."""

# Static correlation matrix (seasonal average, derived from geographic proximity)
_CORRELATION = {
    ("NYC", "Atlanta"): 0.50,
    ("NYC", "Chicago"): 0.40,
    ("NYC", "Miami"): 0.25,
    ("NYC", "Dallas"): 0.20,
    ("NYC", "Seattle"): 0.10,
    ("Chicago", "Atlanta"): 0.30,
    ("Chicago", "Dallas"): 0.20,
    ("Chicago", "Miami"): 0.15,
    ("Chicago", "Seattle"): 0.10,
    ("Atlanta", "Miami"): 0.35,
    ("Atlanta", "Dallas"): 0.30,
    ("Atlanta", "Seattle"): 0.10,
    ("Dallas", "Miami"): 0.20,
    ("Dallas", "Seattle"): 0.10,
    ("Miami", "Seattle"): 0.05,
}

_PENALTY_FACTOR = 0.5  # How much correlation reduces sizing


def get_correlation(loc_a: str, loc_b: str) -> float:
    if loc_a == loc_b:
        return 1.0
    key = (loc_a, loc_b) if (loc_a, loc_b) in _CORRELATION else (loc_b, loc_a)
    return _CORRELATION.get(key, 0.0)


def adjust_position_size(
    base_size: float,
    new_location: str,
    existing_locations: list[str],
    penalty: float = _PENALTY_FACTOR,
) -> float:
    if not existing_locations:
        return base_size
    max_corr = max(get_correlation(new_location, loc) for loc in existing_locations)
    return base_size * (1.0 - max_corr * penalty)
```

**Step 3: Run tests**

Run: `python3 -m pytest weather/tests/test_correlation.py -v`
Expected: PASS

**Step 4: Integrate into strategy.py**

In the trade execution block, before `compute_position_size`, add:

```python
    # Correlation-adjusted sizing
    existing_locations = list({t.location for t in state.trades.values() if t.location})
    if existing_locations:
        from .correlation import adjust_position_size
        position_size = adjust_position_size(position_size, location, existing_locations)
```

**Step 5: Commit**

```bash
git add weather/correlation.py weather/tests/test_correlation.py weather/strategy.py
git commit -m "feat: add inter-location correlation matrix for position sizing"
```

---

### Task 18: Final integration test & full suite

**Step 1: Run full test suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: ALL PASS

**Step 2: Run paper trading dry run**

Run: `python3 -m weather.paper_trade --verbose`
Expected: No crashes, circuit breaker logs visible, NO opportunities logged

**Step 3: Run CLI report**

Run: `python3 -m weather.report`
Expected: Formatted output

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: integration verification after comprehensive improvements"
```

---

## Summary

| Task | Phase | Description | Files |
|------|-------|-------------|-------|
| 1 | P1 | Guards for `_regularized_incomplete_beta` | probability.py, tests |
| 2 | P1 | Guards for `_student_t_cdf` | probability.py, tests |
| 3 | P1 | Cap probability output + sigma bound | probability.py, tests |
| 4 | P1 | Logging for calibration fallback | probability.py, tests |
| 5 | P1 | Calibration age validation | probability.py, tests |
| 6 | P1 | Circuit breaker config | config.py, tests |
| 7 | P1 | Circuit breaker state | state.py, tests |
| 8 | P1 | Circuit breaker enforcement | strategy.py, tests |
| 9 | P2 | Score NO-side opportunities | strategy.py, tests |
| 10 | P2 | Execute NO-side trades | strategy.py, tests |
| 11 | P2 | Exit on edge inversion | strategy.py, tests |
| 12 | P2 | Externalize horizon growth | calibrate.py, probability.py |
| 13 | P2 | Feedback decay | feedback.py, tests |
| 14 | P2 | Metrics module | metrics.py, tests |
| 15 | P2 | CLI report | report.py |
| 16 | P3 | API response cache | open_meteo.py, tests |
| 17 | P3 | Correlation sizing | correlation.py, strategy.py |
| 18 | P3 | Integration verification | all |
