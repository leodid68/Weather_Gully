# Prediction Methodology Audit — Free Wins Design

## Goal

Fix 5 low-effort, high-impact issues identified by a full audit of the temperature prediction pipeline. Each fix improves prediction accuracy or financial correctness with minimal risk.

## Fixes

### 1. Feedback uses forecast month (not current month)

**File:** `weather/strategy.py` lines 727, 744

**Problem:** `datetime.now(timezone.utc).month` returns the current month. If we're on Jan 31 trading a Feb 2 forecast, we look up January's feedback bias instead of February's. Systematic mismatch during month transitions.

**Fix:** Extract month from `date_str` (the forecast target date) instead of `datetime.now()`.

```python
# Before
ema_error = feedback.get_abs_error_ema(location, datetime.now(timezone.utc).month)
feedback_bias = feedback.get_bias(location, datetime.now(timezone.utc).month)

# After
forecast_month = int(date_str.split("-")[1])
ema_error = feedback.get_abs_error_ema(location, forecast_month)
feedback_bias = feedback.get_bias(location, forecast_month)
```

### 2. Include Polymarket trading fees in EV calculation

**Files:** `weather/config.py`, `weather/strategy.py`

**Problem:** `ev = prob - price` ignores Polymarket's ~2% fee on gains. A 3% apparent edge is really 1% after fees. We trade marginal opportunities that are actually unprofitable.

**Fix:** Add `trading_fees: float = 0.02` to Config. Adjust EV calculation:
```python
# net payout after fees = (1.0 - fees) if we win, -price if we lose
# ev = prob * (1.0 - fees) - price  (replaces prob - price)
ev = prob * (1.0 - config.trading_fees) - price
```

This requires threading `trading_fees` through to `score_buckets()`.

### 3. Temperature rounding precision: round(temp) → round(temp, 1)

**File:** `weather/open_meteo.py` lines 131, 133, 139, 141, 250, 252, 257, 259

**Problem:** `round(gfs_high)` rounds to integer, losing up to 0.5°F. On a 5°F bucket, that's 10% of bucket width — enough to shift a borderline probability meaningfully.

**Fix:** `round(gfs_high, 1)` everywhere (8 occurrences).

### 4. Default fill_timeout = 30s (verify order fills)

**File:** `weather/bridge.py`

**Problem:** `fill_timeout=0.0` means no fill verification. If a fill is partial, the state records requested shares, not actual shares. Exposure tracking is wrong.

**Fix:** Change default from `0.0` to `30.0` in both `place_order()` and `sell_position()` signatures.

### 5. Reduce aviation obs weight at day-0

**File:** `weather/strategy.py` line 670

**Problem:** `config.aviation_obs_weight * 2.0 = 0.80` at day-0. After renormalization with GFS(0.30)+ECMWF(0.50), the METAR observation gets ~44% of ensemble weight. A single current-temperature observation shouldn't dominate two NWP forecast models.

**Fix:** Remove the `* 2.0` multiplier. Keep the base weight (0.40) for day-0, same as day-1. The observation still gets meaningful weight (~29% after renormalization) without dominating.

## Success Criteria

1. All 5 fixes applied
2. Full test suite passes (497+ tests)
3. EV now accounts for fees (verifiable in logs)
4. Feedback uses correct forecast month
5. Temperature precision improved to 0.1°F
