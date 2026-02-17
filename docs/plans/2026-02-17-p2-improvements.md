# P2 Improvements: Trade Resolution, Other Bucket, Model Conviction, Cross-Temporal

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** (1) Auto-resolve trades in trade_log via METAR observations, (2) Adjust no_prob for "Other" bucket, (3) Add model conviction signal, (4) Add cross-temporal arb signal.

**Tech Stack:** Python 3.14, stdlib, pas de deps externes.

---

## Task 1: Auto-resolve trades in trade_log

**Files:**
- Modify: `weather/paper_trade.py` — call `resolve_trades()` after METAR observations
- Modify: `weather/trade_log.py` — ensure `resolve_trades()` handles sentinel buckets correctly
- Create: `weather/tests/test_trade_log_resolution.py`

### What to build

Currently `trade_log.py` has `resolve_trades(actuals)` that takes `{"2026-02-17": {"high": 43.0}}` but it's never called. Wire it into `paper_trade.py` after predictions are resolved, using `state.daily_obs` as the actuals source.

### Implementation

**In `weather/paper_trade.py`**, in `_async_main()`, after `_feed_feedback(state, feedback, kalman=kalman)` (around line 178), add:

```python
# Auto-resolve trade log entries using daily observations
from .trade_log import resolve_trades
actuals: dict[str, dict[str, float]] = {}
for key, obs in state.daily_obs.items():
    # key is "location:YYYY-MM-DD", obs has obs_high/obs_low
    parts = key.split(":", 1)
    if len(parts) != 2:
        continue
    _, date_str = parts
    for metric in ("high", "low"):
        temp = obs.get(f"obs_{metric}")
        if temp is not None:
            actuals.setdefault(date_str, {})[metric] = temp
if actuals:
    resolved_count = resolve_trades(actuals)
    if resolved_count:
        logger.info("Resolved %d trade_log entries via METAR observations", resolved_count)
```

**In `weather/trade_log.py`**, verify `resolve_trades()` handles sentinel buckets (-999, 999):
- Current code at line 85: `lo <= actual <= hi`. With sentinels: `-999 <= 43 <= 41` → True (correct for "41 or below"). Verify edge cases.

### Tests

- `test_resolve_trades_win`: bucket [42, 43], actual 42.5 → outcome=1, pnl=(1-price)*shares
- `test_resolve_trades_loss`: bucket [42, 43], actual 45.0 → outcome=0, pnl=-price*shares
- `test_resolve_sentinel_below`: bucket [-999, 41], actual 39 → outcome=1
- `test_resolve_sentinel_above`: bucket [46, 999], actual 50 → outcome=1
- `test_resolve_skips_already_resolved`: outcome already set → skip
- `test_paper_trade_wires_resolution`: actuals dict built correctly from daily_obs

### Verification
```
python3 -m pytest weather/tests/test_trade_log_resolution.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 2: Adjust no_prob for "Other" bucket

**Files:**
- Modify: `weather/strategy.py` — in `score_buckets()`, detect "Other" buckets and adjust no_prob
- Create: `weather/tests/test_other_bucket.py`

### What to build

In `score_buckets()`, before the main loop, scan `event_markets` for non-parseable bucket names (these are "Other" or "No winner" buckets). Sum their YES prices as `p_other`. Then in the NO side scoring, use `no_prob = 1.0 - prob - p_other` instead of `no_prob = 1.0 - prob`.

### Implementation

**In `score_buckets()` (strategy.py around line 220)**, add before the main loop:

```python
# Detect "Other" bucket price from non-parseable markets
p_other = 0.0
for m in event_markets:
    oname = m.get("outcome_name", "")
    if not parse_temperature_bucket(oname):
        ask_str = m.get("best_ask") or m.get("external_price_yes") or "0"
        try:
            p = float(ask_str)
        except (ValueError, TypeError):
            p = 0.0
        if 0 < p < 1.0:
            p_other += p
```

Then in the NO side block (around line 311):

```python
no_prob = 1.0 - prob - p_other  # Account for "Other" bucket
no_prob = max(no_prob, 0.0)     # Clamp to non-negative
```

### Tests

- `test_no_prob_adjusted_with_other`: event with 3 temp buckets + 1 "Other" at $0.05 → no_prob reduced by 0.05
- `test_no_prob_no_other`: event with only temp buckets → no_prob = 1.0 - prob (unchanged)
- `test_no_prob_clamped_to_zero`: edge case where prob + p_other > 1.0 → no_prob = 0.0
- `test_other_bucket_detected`: market with outcome_name "Other" → correctly identified

### Verification
```
python3 -m pytest weather/tests/test_other_bucket.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 3: Model conviction signal

**Files:**
- Modify: `weather/strategy.py` — add conviction-based edge boost when dominant model aligns with forecast
- Modify: `weather/open_meteo.py` — expose per-model temperatures from ensemble computation
- Create: `weather/tests/test_model_conviction.py`

### What to build

When models disagree (already detected), check if the dominant model (highest weight in calibration.json > 0.5) aligns with a specific bucket. If the dominant model's temperature falls within the tradeable bucket, apply a conviction multiplier to the edge (1.5x), which boosts position sizing via Kelly. If the dominant model disagrees with the bucket, reduce to 0.5x.

### Implementation

**In `weather/open_meteo.py`**, modify `compute_ensemble_forecast()` to also return model-level temperatures. Add a new function:

```python
def get_dominant_model_info(location: str, noaa_temp: float | None,
                            om_data: dict | None, metric: str = "high"
                            ) -> tuple[str, float | None, float]:
    """Return (model_name, model_temp, model_weight) for the dominant model.

    Returns ("", None, 0.0) if no dominant model.
    """
    weights = _get_model_weights(location)
    best_name, best_weight = "", 0.0
    for name, w in weights.items():
        if w > best_weight:
            best_name, best_weight = name, w

    if best_weight < 0.4:  # No clear dominant model
        return "", None, 0.0

    # Get the dominant model's temperature
    if best_name == "noaa" and noaa_temp is not None:
        return best_name, noaa_temp, best_weight
    elif om_data and best_name in om_data:
        key = "temperature_2m_max" if metric == "high" else "temperature_2m_min"
        temp = om_data[best_name].get(key)
        if temp is not None:
            return best_name, temp, best_weight
    return best_name, None, best_weight
```

**In `weather/strategy.py`**, after model disagreement detection and before scoring, add:

```python
# Model conviction: check if dominant model aligns with tradeable bucket
conviction_multiplier = 1.0
if config.multi_source and model_disagreement:
    from .open_meteo import get_dominant_model_info
    dom_name, dom_temp, dom_weight = get_dominant_model_info(
        location, noaa_temp, om_data, metric)
    if dom_temp is not None and dom_weight >= 0.5:
        logger.info("Model conviction: %s=%.0f°F (weight=%.2f)", dom_name, dom_temp, dom_weight)
```

Then in the scoring loop, for each tradeable bucket, check if the dominant model's temperature falls within the bucket:

```python
if conviction_multiplier != 1.0:
    ev *= conviction_multiplier
    # Also adjust position sizing via a factor on the edge
```

Actually, simpler approach: after computing `position_size` via Kelly, multiply by conviction:

```python
# Model conviction sizing adjustment
if model_disagreement and dom_temp is not None and dom_weight >= 0.5:
    lo, hi = bucket
    bc = hi if lo < -900 else (lo if hi > 900 else (lo + hi) / 2.0)
    if abs(dom_temp - bc) <= sigma_override * 0.5:
        # Dominant model agrees with this bucket → boost
        conviction_multiplier = 1.5
        logger.info("Model conviction boost: dominant %s agrees with %s", dom_name, outcome_name)
    else:
        # Dominant model disagrees → reduce
        conviction_multiplier = 0.5
        logger.info("Model conviction penalty: dominant %s disagrees with %s", dom_name, outcome_name)
    position_size *= conviction_multiplier
```

### Tests

- `test_get_dominant_model_info_gfs`: GFS weight 0.85 → returns GFS temp
- `test_get_dominant_model_info_noaa`: NOAA weight 0.5 → returns NOAA temp
- `test_get_dominant_model_info_no_dominant`: all weights < 0.4 → returns empty
- `test_conviction_boost_agrees`: dominant model temp within bucket → 1.5x
- `test_conviction_penalty_disagrees`: dominant model temp far from bucket → 0.5x
- `test_no_conviction_without_disagreement`: no model_disagreement → no adjustment

### Verification
```
python3 -m pytest weather/tests/test_model_conviction.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 4: Cross-temporal arbitrage signal

**Files:**
- Modify: `weather/strategy.py` — add cross-temporal price comparison after scoring
- Create: `weather/tests/test_cross_temporal.py`

### What to build

For each location, compare the price of the same bucket across different forecast dates (J+1 vs J+2, etc.). If the price at J+2 is significantly higher than expected given the sigma growth from J+1 to J+2, it's a signal that J+2 is overpriced (or J+1 is underpriced).

Use the price_snapshots_mr data (mean-reversion tracker) which already stores per-bucket prices keyed by `location|date|metric|lo,hi`.

### Implementation

Add a helper function in `strategy.py`:

```python
def detect_cross_temporal_signal(
    events: dict[str, list[dict]],
    location: str,
    metric: str,
    price_tracker: "PriceTracker | None",
) -> list[dict]:
    """Detect cross-temporal arbitrage signals.

    Compares the same bucket across different dates for the same location.
    Returns list of signals: {"bucket": (lo, hi), "date_near": str,
    "date_far": str, "price_near": float, "price_far": float, "expected_ratio": float}.
    """
    if not price_tracker:
        return []

    # Group buckets by (location, metric, bucket) across dates
    bucket_prices: dict[tuple, list[tuple[str, float]]] = {}  # (lo, hi) → [(date, price), ...]
    for event_id, event_markets in events.items():
        for m in event_markets:
            event_info = parse_weather_event(m.get("event_name", ""))
            if not event_info:
                continue
            if event_info["location"] != location or event_info["metric"] != metric:
                continue
            bucket = _parse_bucket(m.get("outcome_name", ""), location)
            if not bucket:
                continue
            ask_str = m.get("best_ask") or "0"
            try:
                price = float(ask_str)
            except (ValueError, TypeError):
                continue
            if price <= 0:
                continue
            bucket_prices.setdefault(bucket, []).append((event_info["date"], price))

    signals = []
    for bucket, date_prices in bucket_prices.items():
        if len(date_prices) < 2:
            continue
        date_prices.sort()  # Sort by date
        for i in range(len(date_prices) - 1):
            date_near, price_near = date_prices[i]
            date_far, price_far = date_prices[i + 1]
            # Expected ratio: price should be lower at longer horizon (more uncertainty)
            # But only if price_near is significant (> $0.05)
            if price_near < 0.05:
                continue
            ratio = price_far / price_near
            # If far date price is > 80% of near date price, it's potentially overpriced
            # (sigma grows ~33% per day, so probability should spread more)
            if ratio > 0.85 and price_far > 0.10:
                signals.append({
                    "bucket": bucket,
                    "date_near": date_near,
                    "date_far": date_far,
                    "price_near": price_near,
                    "price_far": price_far,
                    "ratio": ratio,
                })
    return signals
```

In the main loop, after scoring for a location, call this and log any signals found:

```python
# Cross-temporal arbitrage signals
ct_signals = detect_cross_temporal_signal(events, location, metric, price_tracker)
for sig in ct_signals:
    logger.info(
        "Cross-temporal signal %s %s: %s — J+near=%s $%.2f vs J+far=%s $%.2f (ratio=%.2f)",
        location, sig["bucket"], metric,
        sig["date_near"], sig["price_near"],
        sig["date_far"], sig["price_far"], sig["ratio"],
    )
```

This is informational logging only for now — not auto-traded. The signal can be used for manual review or future automation.

### Tests

- `test_cross_temporal_detects_overpriced_far`: near=$0.30, far=$0.28 (ratio 0.93) → signal
- `test_cross_temporal_skips_low_price`: near=$0.03, far=$0.02 → skipped (< $0.05)
- `test_cross_temporal_no_signal_when_ratio_low`: near=$0.30, far=$0.15 (ratio 0.5) → no signal
- `test_cross_temporal_multiple_dates`: 3 dates → checks all pairs
- `test_cross_temporal_empty_when_no_tracker`: price_tracker=None → empty

### Verification
```
python3 -m pytest weather/tests/test_cross_temporal.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Final Verification

```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
python3 -m weather.paper_trade --set locations=NYC --explain  # Verify all P2 features
```
