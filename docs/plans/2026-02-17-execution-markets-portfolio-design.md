# Execution, Low Temp Markets & Portfolio Management

## Problem

Three weaknesses in the current bot:

1. **Execution filtering** — The fixed 15% slippage threshold rejects too many trades. In a NYC dry-run, 4/8 tradeable buckets were filtered for "slippage too high". These are limit orders already; the issue is wide bid-ask spreads on low-liquidity markets.

2. **Missing markets** — The bot only trades "Highest temperature" events. "Lowest temperature" markets exist on Polymarket with the same bucket structure, same cities, same APIs. The parsing already recognizes `"lowest"` but the strategy loop only scores `"high"`.

3. **Portfolio sizing** — Each trade is sized independently via quarter-Kelly on the full balance, ignoring existing exposure. With 10 open positions ($20 exposure), the 11th trade is still sized on $50. The correlation discount never triggers (threshold 0.7, max observed correlation 0.674).

## Solution

### 1. Adaptive Execution

Replace the fixed slippage threshold with edge-aware filtering and depth-aware sizing.

**A. Edge-proportional slippage tolerance:**

```
effective_threshold = min(slippage_max_pct, edge * slippage_edge_ratio)
```

With `slippage_edge_ratio = 0.5` (new config field, default 0.5): a trade with 20% edge tolerates 10% spread; a trade with 5% edge tolerates 2.5% spread. High-EV trades justify worse liquidity.

**B. Orderbook depth sizing:**

Before placing an order, compute available liquidity in the top N levels of the orderbook:

```python
def compute_available_depth(book_side: list[dict], max_levels: int = 5) -> float:
    return sum(float(level["size"]) * float(level["price"]) for level in book_side[:max_levels])
```

If `position_size > available_depth * depth_fill_ratio` (new config, default 0.5), reduce position size to `available_depth * depth_fill_ratio` instead of skipping entirely. Transforms "skip" into smaller trades.

**C. VWAP limit pricing:**

Instead of `asks[0]["price"]`, compute volume-weighted average price across levels needed to fill our size:

```python
def compute_vwap(book_side: list[dict], target_usd: float) -> float:
    filled = 0.0
    cost = 0.0
    for level in book_side:
        price = float(level["price"])
        available = float(level["size"]) * price
        take = min(available, target_usd - filled)
        cost += take
        filled += take
        if filled >= target_usd:
            break
    return cost / filled if filled > 0 else float(book_side[0]["price"])
```

Place limit order at VWAP price instead of best ask. Better average fill on larger orders.

**New config fields:**
- `slippage_edge_ratio: float = 0.5` — edge multiplier for adaptive slippage
- `depth_fill_ratio: float = 0.5` — max fraction of orderbook depth to consume
- `vwap_max_levels: int = 5` — orderbook levels to scan for VWAP

### 2. Low Temperature Markets

**What already works (no changes needed):**
- `parse_weather_event()` returns `metric="low"` for "Lowest temperature" events
- `compute_ensemble_forecast(metric="low")` uses `gfs_low`, `ecmwf_low`
- `estimate_bucket_probability()` — same skew-t CDF, same sigma logic
- `get_historical_metar_actuals()` returns `{"high": X, "low": Y}`
- NOAA forecasts include low temperatures

**Changes needed:**

A. New config field:
```python
trade_metrics: list[str] = ["high"]  # Add "low" to enable low temp trading
```

B. Strategy loop modification — in `run_weather_strategy()`, where events are scored (around line 941), the current code extracts `metric` from `parse_weather_event()` but the NOAA/Open-Meteo cache lookup and ensemble computation already use the metric field. The main change: the scoring loop currently processes all events but only fetches NOAA high or low based on the parsed metric. Verify this path works end-to-end for `metric="low"`.

C. Backtest validation — run backtest with `trade_metrics=["high", "low"]` to verify low temp Brier score is comparable. If low temp sigma is significantly different, add `low_sigma_factor` to calibration.

**Expected impact:** ~2x more tradeable events (each city/date has both high and low markets). Same model, same APIs, minimal code change.

### 3. Portfolio Budget & Correlation

**A. Dynamic budget sizing:**

Replace in `compute_position_size()`:
```python
# Old: uses full balance
size = kelly * balance

# New: uses available budget
available = max(0, balance - current_exposure)
size = kelly * available
```

Add `current_exposure` parameter to `compute_position_size()`. The strategy already tracks exposure via `state.trades` — sum of `trade.shares * trade.entry_price` for open positions.

When exposure is high, new trades are naturally smaller. When all positions are closed, full balance is available. This converges toward optimal aggregate Kelly without matrix math.

**B. Lower correlation threshold:**

Change default `correlation_threshold` from 0.7 to 0.3. Current correlation data shows values like NYC|Atlanta DJF=0.174, Dallas|Seattle SON=0.674. At threshold 0.3, pairs above 0.3 get discounted:
- Dallas|Seattle SON (0.674): discount = 1 - 0.674*0.5 = 0.663x
- Atlanta|NYC SON (0.281): no discount (below 0.3)

**C. Same-location horizon penalty:**

When we already hold a position on the same location within 2 days, apply a `same_location_discount`:

```python
if any(existing.location == location and abs(existing.horizon - horizon) <= 2
       for existing in open_positions):
    size *= same_location_discount  # default 0.5
```

NYC high Feb 17 and NYC high Feb 18 are highly correlated — halve the second position.

**D. Cumulative correlation penalty:**

Replace `max(correlations)` with sum-based penalty:

```python
total_corr = sum(corr for corr in correlations if corr > threshold)
factor = max(0.1, 1.0 - total_corr * discount)
```

Having 3 moderately correlated positions (0.3 each) is worse than 1. The sum captures this.

**New config fields:**
- `same_location_discount: float = 0.5` — size reduction for same-city near-horizon
- `same_location_horizon_window: int = 2` — days within which same-location penalty applies

**Changed defaults:**
- `correlation_threshold`: 0.7 → 0.3

## Architecture

All changes are in existing files — no new modules.

| File | Changes |
|------|---------|
| `weather/config.py` | Add 5 new config fields, change 1 default |
| `weather/bridge.py` | Add `compute_available_depth()`, `compute_vwap()`, modify `execute_trade()` |
| `weather/strategy.py` | Adaptive slippage in `check_context_safeguards()`, depth sizing, low temp loop, budget sizing, cumulative correlation |
| `weather/sizing.py` | Add `current_exposure` param to `compute_position_size()` |
| `weather/calibration.json` | No changes (correlation data already exists) |

## Constraints

- 100% stdlib (no new dependencies)
- No breaking changes to existing API signatures (add optional params only)
- All features gated by config (can be disabled individually)
- Existing tests must continue to pass

## Expected Impact

| Feature | Estimated Impact |
|---------|-----------------|
| Adaptive slippage | +30-50% more trades executed (fewer filtered) |
| Depth sizing | Smaller but non-zero trades instead of skips |
| VWAP pricing | ~1-3% better average fill price |
| Low temp markets | ~2x more market opportunities |
| Dynamic budget | Prevents over-leverage at high exposure |
| Correlation fixes | Active portfolio risk reduction (currently dormant) |

## Testing Strategy

- Unit tests for each new function (`compute_vwap`, `compute_available_depth`, budget sizing)
- Integration tests: mock orderbook with varying depth → verify adaptive sizing
- Backtest: compare with/without low temp, with/without dynamic budget
- Full test suite must pass (876+ tests)
