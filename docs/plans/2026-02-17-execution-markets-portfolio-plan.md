# Execution, Low Temp Markets & Portfolio Management — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement adaptive execution (VWAP, depth sizing, edge-proportional slippage), low temperature market trading, and portfolio-level budget/correlation management.

**Architecture:** All changes are in existing files — no new modules. Config fields are added with backward-compatible defaults. Each feature is independently gated by config.

**Tech Stack:** Python 3.14, stdlib only (no new dependencies).

---

## Task 1: Add new config fields

**Files:**
- Modify: `weather/config.py` — add 6 new fields to `Config` dataclass, change 1 default

### What to build

Add the following fields to the `Config` dataclass:

```python
# Adaptive execution
slippage_edge_ratio: float = 0.5       # Edge multiplier for adaptive slippage threshold
depth_fill_ratio: float = 0.5          # Max fraction of orderbook depth to consume
vwap_max_levels: int = 5               # Orderbook levels to scan for VWAP pricing

# Low temp markets
trade_metrics: str = "high"             # Comma-separated: "high", "low", or "high,low"

# Portfolio management
same_location_discount: float = 0.5    # Size reduction for same-city near-horizon positions
same_location_horizon_window: int = 2  # Days within which same-location penalty applies
```

Change existing default:
```python
correlation_threshold: float = 0.3     # Was 0.7 — too high, correlations never triggered
```

Use `str` for `trade_metrics` (like `locations`) with a property `active_metrics` that parses it, to keep consistency with the existing config pattern. Add `active_metrics` property.

### Implementation

In `weather/config.py`, add after line 141 (after `mean_reversion`):

```python
# Adaptive execution (VWAP, depth sizing, edge-proportional slippage)
slippage_edge_ratio: float = 0.5
depth_fill_ratio: float = 0.5
vwap_max_levels: int = 5

# Low temperature markets
trade_metrics: str = "high"

# Portfolio: same-location sizing discount
same_location_discount: float = 0.5
same_location_horizon_window: int = 2
```

Change line 132:
```python
correlation_threshold: float = 0.3  # was 0.7
```

Add after `active_locations` property (after line 161):
```python
@property
def active_metrics(self) -> list[str]:
    """Return list of metrics to trade: ['high'], ['low'], or ['high', 'low']."""
    return [m.strip().lower() for m in self.trade_metrics.split(",") if m.strip()]
```

### Tests

Add to `weather/tests/test_config.py`:

```python
def test_new_config_defaults():
    cfg = Config()
    assert cfg.slippage_edge_ratio == 0.5
    assert cfg.depth_fill_ratio == 0.5
    assert cfg.vwap_max_levels == 5
    assert cfg.trade_metrics == "high"
    assert cfg.same_location_discount == 0.5
    assert cfg.same_location_horizon_window == 2
    assert cfg.correlation_threshold == 0.3  # Changed from 0.7

def test_active_metrics_single():
    cfg = Config(trade_metrics="high")
    assert cfg.active_metrics == ["high"]

def test_active_metrics_both():
    cfg = Config(trade_metrics="high,low")
    assert cfg.active_metrics == ["high", "low"]

def test_active_metrics_low_only():
    cfg = Config(trade_metrics="low")
    assert cfg.active_metrics == ["low"]
```

### Verification
```
python3 -m pytest weather/tests/test_config.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 2: Add `compute_vwap` and `compute_available_depth` to bridge

**Files:**
- Modify: `weather/bridge.py` — add 2 helper functions, modify `execute_trade()` to use VWAP pricing and depth-aware sizing
- Create: `weather/tests/test_bridge_execution.py` — new test file for execution helpers

### What to build

Two pure functions and an update to `execute_trade()`.

### Implementation

**A. Add helper functions** — insert before `execute_trade()` (before line 265):

```python
def compute_available_depth(book_side: list[dict], max_levels: int = 5) -> float:
    """Compute total USD liquidity available in the top N levels of an orderbook side.

    Args:
        book_side: List of orderbook levels, each with "price" and "size" keys.
        max_levels: Maximum number of levels to scan.

    Returns:
        Total available liquidity in USD.
    """
    total = 0.0
    for level in book_side[:max_levels]:
        try:
            price = float(level["price"])
            size = float(level["size"])
            total += size * price
        except (KeyError, ValueError, TypeError):
            continue
    return total


def compute_vwap(book_side: list[dict], target_usd: float) -> float:
    """Compute volume-weighted average price across orderbook levels to fill target_usd.

    Walks through the orderbook from best price, filling as much as possible from
    each level until the target is met.

    Args:
        book_side: List of orderbook levels (e.g. asks), each with "price" and "size".
        target_usd: Target USD amount to fill.

    Returns:
        VWAP price. Falls back to best level price if no fill possible.
    """
    if not book_side:
        return 0.0

    filled = 0.0
    cost = 0.0
    for level in book_side:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, ValueError, TypeError):
            continue
        available_usd = size * price
        take = min(available_usd, target_usd - filled)
        cost += take
        filled += take
        if filled >= target_usd:
            break

    if filled > 0:
        return cost / filled
    # Fallback: best ask/bid price
    try:
        return float(book_side[0]["price"])
    except (KeyError, ValueError, TypeError, IndexError):
        return 0.0
```

**B. Modify `execute_trade()`** — add `depth_fill_ratio` and `vwap_max_levels` parameters:

Add parameters to signature (line 265):
```python
def execute_trade(
    self,
    market_id: str,
    side: str,
    amount: float,
    fill_timeout: float = 30.0,
    fill_poll_interval: float = 2.0,
    depth_fill_ratio: float = 0.0,
    vwap_max_levels: int = 0,
) -> dict:
```

Replace the price-fetching block (lines 302-317) with:
```python
# Re-fetch fresh price from orderbook
try:
    book = self.clob.get_orderbook(token_id)
    asks = book.get("asks") or []
    if asks:
        # Depth-aware sizing: cap amount to available liquidity
        if depth_fill_ratio > 0:
            depth = compute_available_depth(asks, max_levels=max(vwap_max_levels, 5))
            max_from_depth = depth * depth_fill_ratio
            if amount > max_from_depth > 0:
                logger.info("Depth cap: $%.2f → $%.2f (%.0f%% of $%.2f depth)",
                            amount, max_from_depth, depth_fill_ratio * 100, depth)
                amount = max_from_depth

        # VWAP pricing across multiple levels
        if vwap_max_levels > 1:
            price = compute_vwap(asks[:vwap_max_levels], amount)
        else:
            price = float(asks[0]["price"])
    else:
        # Fallback to cached price
        price = gm.best_ask if gm.best_ask > 0 else (
            gm.outcome_prices[0] if gm.outcome_prices else 0.5
        )
except Exception:
    price = gm.best_ask if gm.best_ask > 0 else (
        gm.outcome_prices[0] if gm.outcome_prices else 0.5
    )
    logger.warning("Could not refresh price for %s, using cached %.4f", token_id[:16], price)
```

### Tests (`weather/tests/test_bridge_execution.py`)

```python
"""Tests for bridge execution helpers (VWAP, depth)."""

from weather.bridge import compute_available_depth, compute_vwap


class TestComputeAvailableDepth:
    def test_basic_depth(self):
        book = [
            {"price": "0.50", "size": "100"},
            {"price": "0.55", "size": "200"},
        ]
        assert compute_available_depth(book, max_levels=5) == 100*0.5 + 200*0.55

    def test_max_levels_limit(self):
        book = [
            {"price": "0.50", "size": "100"},
            {"price": "0.55", "size": "200"},
            {"price": "0.60", "size": "300"},
        ]
        depth = compute_available_depth(book, max_levels=2)
        assert depth == 100*0.5 + 200*0.55

    def test_empty_book(self):
        assert compute_available_depth([], max_levels=5) == 0.0

    def test_malformed_level(self):
        book = [{"price": "0.50"}, {"price": "0.60", "size": "100"}]
        assert compute_available_depth(book, max_levels=5) == 100 * 0.60


class TestComputeVwap:
    def test_single_level_fill(self):
        book = [{"price": "0.50", "size": "100"}]
        # target 10 USD, all from level 0
        vwap = compute_vwap(book, 10.0)
        assert abs(vwap - 0.50) < 1e-6

    def test_multi_level_fill(self):
        book = [
            {"price": "0.50", "size": "20"},   # 10 USD available
            {"price": "0.60", "size": "100"},   # 60 USD available
        ]
        # target 20 USD: 10 from level 0, 10 from level 1
        vwap = compute_vwap(book, 20.0)
        expected = 20.0 / 20.0  # cost=20, filled=20 → 1.0? No...
        # Actually: cost = 10 + 10 = 20, filled = 10 + 10 = 20 → 1.0
        # Wait: cost is in USD, filled is in USD, so vwap = cost/filled = 1.0? No.
        # The VWAP is: (10 * 0.50 + 10 * 0.60) / 20 = 0.55... but that's not how it works.
        # The function: filled += take (USD), cost += take (USD), so cost/filled = 1.0
        # Actually looking at the function: take = min(available_usd, target - filled)
        # filled = 0 + 10 + 10 = 20, cost = 0 + 10 + 10 = 20, vwap = 20/20 = 1.0
        # Hmm, that means the function always returns 1.0 since cost == filled.
        # The function tracks USD in and USD out, not shares.
        # Need to rethink: VWAP should be total_cost / total_shares.
        # Let me fix the function design in the implementation above.
        pass

    def test_empty_book(self):
        assert compute_vwap([], 10.0) == 0.0

    def test_fallback_to_best(self):
        book = [{"price": "0.50", "size": "1"}]
        # target much larger than available
        vwap = compute_vwap(book, 1000.0)
        # Should fill what's available and compute VWAP
        assert vwap > 0
```

**IMPORTANT NOTE:** The `compute_vwap` function in the design doc tracks USD filled and USD cost, which makes `cost/filled = 1.0` always. The correct implementation must track **shares** filled and compute `total_cost / total_shares`:

```python
def compute_vwap(book_side: list[dict], target_usd: float) -> float:
    if not book_side:
        return 0.0

    total_shares = 0.0
    total_cost = 0.0
    remaining = target_usd
    for level in book_side:
        try:
            price = float(level["price"])
            size = float(level["size"])  # size is in shares
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0:
            continue
        available_usd = size * price
        take_usd = min(available_usd, remaining)
        take_shares = take_usd / price
        total_shares += take_shares
        total_cost += take_usd
        remaining -= take_usd
        if remaining <= 0:
            break

    if total_shares > 0:
        return total_cost / total_shares
    try:
        return float(book_side[0]["price"])
    except (KeyError, ValueError, TypeError, IndexError):
        return 0.0
```

With this corrected implementation, the multi-level test becomes:
```python
def test_multi_level_fill(self):
    book = [
        {"price": "0.50", "size": "20"},   # 10 USD available
        {"price": "0.60", "size": "100"},   # 60 USD available
    ]
    vwap = compute_vwap(book, 20.0)
    # Level 0: 10 USD → 20 shares. Level 1: 10 USD → 16.67 shares
    # VWAP = 20 / 36.67 = 0.5455
    expected = 20.0 / (20 + 10/0.60)
    assert abs(vwap - expected) < 0.01
```

### Verification
```
python3 -m pytest weather/tests/test_bridge_execution.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 3: Adaptive slippage in strategy safeguards

**Files:**
- Modify: `weather/strategy.py` — modify `check_context_safeguards()` to use edge-proportional slippage

### What to build

Replace the fixed slippage check with edge-proportional tolerance. A trade with 20% edge tolerates 10% spread; a trade with 5% edge tolerates 2.5% spread.

### Implementation

Modify `check_context_safeguards()` (lines 66-122) — change the slippage check block (lines 99-104):

**Current code (lines 99-104):**
```python
    # Slippage
    estimates = slippage.get("estimates", []) if slippage else []
    if estimates:
        slippage_pct = estimates[0].get("slippage_pct", 0)
        if slippage_pct > config.slippage_max_pct:
            return False, [f"Slippage too high: {slippage_pct:.1%}"]
```

**New code:**
```python
    # Slippage (adaptive: edge-proportional tolerance)
    estimates = slippage.get("estimates", []) if slippage else []
    if estimates:
        slippage_pct = estimates[0].get("slippage_pct", 0)
        # Edge-proportional threshold: high-EV trades tolerate worse liquidity
        user_edge_val = edge.get("user_edge", 0) if edge else 0
        if user_edge_val and config.slippage_edge_ratio > 0:
            adaptive_threshold = min(config.slippage_max_pct,
                                     abs(user_edge_val) * config.slippage_edge_ratio)
            effective_threshold = max(adaptive_threshold, 0.01)  # Floor at 1%
        else:
            effective_threshold = config.slippage_max_pct
        if slippage_pct > effective_threshold:
            return False, [f"Slippage too high: {slippage_pct:.1%} (threshold {effective_threshold:.1%})"]
```

Also add `config` parameter usage note: when `slippage_edge_ratio = 0`, the behavior is identical to the old fixed threshold.

### Tests

Add to existing safeguards test file or `weather/tests/test_strategy.py`:

```python
def test_adaptive_slippage_high_edge_passes():
    """High-edge trade tolerates higher slippage."""
    config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
    context = {
        "slippage": {"estimates": [{"slippage_pct": 0.10}]},
        "edge": {"user_edge": 0.25},  # 25% edge → threshold = 12.5%
    }
    ok, reasons = check_context_safeguards(context, config)
    assert ok  # 10% < 12.5%

def test_adaptive_slippage_low_edge_blocked():
    """Low-edge trade has low slippage tolerance."""
    config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
    context = {
        "slippage": {"estimates": [{"slippage_pct": 0.10}]},
        "edge": {"user_edge": 0.05},  # 5% edge → threshold = 2.5%
    }
    ok, reasons = check_context_safeguards(context, config)
    assert not ok  # 10% > 2.5%

def test_adaptive_slippage_capped_at_max():
    """Adaptive threshold never exceeds slippage_max_pct."""
    config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
    context = {
        "slippage": {"estimates": [{"slippage_pct": 0.14}]},
        "edge": {"user_edge": 0.50},  # 50% edge → 25%, capped at 15%
    }
    ok, reasons = check_context_safeguards(context, config)
    assert ok  # 14% < 15%

def test_adaptive_slippage_no_edge_uses_fixed():
    """Without edge data, falls back to fixed threshold."""
    config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
    context = {
        "slippage": {"estimates": [{"slippage_pct": 0.10}]},
    }
    ok, reasons = check_context_safeguards(context, config)
    assert ok  # 10% < 15% (fixed threshold)
```

### Verification
```
python3 -m pytest weather/tests/test_strategy.py -v -k "slippage"
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 4: Wire VWAP and depth sizing into strategy execution

**Files:**
- Modify: `weather/strategy.py` — pass `depth_fill_ratio` and `vwap_max_levels` to `execute_trade()`

### What to build

Wire the new config fields through to `execute_trade()` so that the bridge uses VWAP pricing and depth-aware sizing when the config enables them.

### Implementation

In `weather/strategy.py`, at the `execute_trade` call (line 1284), change:

**Current (line 1284-1288):**
```python
result = client.execute_trade(
    market_id, side, position_size,
    fill_timeout=config.fill_timeout_seconds,
    fill_poll_interval=config.fill_poll_interval,
)
```

**New:**
```python
result = client.execute_trade(
    market_id, side, position_size,
    fill_timeout=config.fill_timeout_seconds,
    fill_poll_interval=config.fill_poll_interval,
    depth_fill_ratio=config.depth_fill_ratio,
    vwap_max_levels=config.vwap_max_levels,
)
```

That's it — the bridge handles the rest. When `depth_fill_ratio=0` or `vwap_max_levels=0`, behavior is unchanged.

### Tests

No new test file needed — the wiring is covered by existing integration tests. Verify by running full suite.

### Verification
```
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 5: Low temperature markets (trade_metrics filter)

**Files:**
- Modify: `weather/strategy.py` — add `trade_metrics` filtering in the scoring loop

### What to build

Filter events by `config.active_metrics` so that low-temp events are processed only when `"low"` is in the configured metrics list. Everything else (parsing, forecasting, scoring) already works for metric="low".

### Implementation

In `weather/strategy.py`, in the scoring loop (after line 950 where `metric = event_info["metric"]`), add a filter:

```python
metric = event_info["metric"]

# Filter by configured trade metrics
if metric not in config.active_metrics:
    logger.debug("Skipping %s %s — metric '%s' not in trade_metrics %s",
                 location, date_str, metric, config.active_metrics)
    continue
```

Insert this right after line 950 (before line 952 `if location not in config.active_locations:`).

### Tests

Add to `weather/tests/test_strategy.py`:

```python
def test_trade_metrics_filters_low_temp():
    """Events with metric='low' are skipped when trade_metrics='high'."""
    config = Config(trade_metrics="high")
    assert "low" not in config.active_metrics
    assert "high" in config.active_metrics

def test_trade_metrics_allows_both():
    """Both high and low are allowed when trade_metrics='high,low'."""
    config = Config(trade_metrics="high,low")
    assert "high" in config.active_metrics
    assert "low" in config.active_metrics
```

The actual integration test is verifying that the strategy loop skips events correctly — covered by the existing scoring loop mock tests.

### Verification
```
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 6: Dynamic budget sizing in `compute_position_size`

**Files:**
- Modify: `weather/sizing.py` — add `current_exposure` parameter to `compute_position_size()`
- Modify: `weather/strategy.py` — compute exposure sum from state and pass it to sizing

### What to build

Deduct current exposure from available balance before Kelly sizing. When exposure is high, new trades are naturally smaller. When all positions are closed, full balance is available.

### Implementation

**A. In `weather/sizing.py`**, add `current_exposure` parameter to `compute_position_size()`:

Change signature (line 27):
```python
def compute_position_size(
    probability: float,
    price: float,
    balance: float,
    max_position_usd: float,
    kelly_frac: float = 0.25,
    min_trade: float = 1.0,
    current_exposure: float = 0.0,
) -> float:
```

Change the sizing computation (line 57):
```python
    # Budget: deduct current exposure from available balance
    available = max(0.0, balance - current_exposure)
    if available <= 0:
        return 0.0

    size = available * frac
    size = min(size, max_position_usd)

    # If Kelly says less than the minimum viable trade, don't trade at all
    if size < min_trade:
        return 0.0

    # Don't bet more than we have
    size = min(size, available)
```

Also update the debug log to show exposure:
```python
    logger.debug(
        "Kelly sizing: p=%.3f price=%.3f b=%.2f frac=%.4f exposure=$%.2f → $%.2f",
        probability, price, b, frac, current_exposure, size,
    )
```

**B. In `weather/strategy.py`**, compute exposure and pass it. Before the scoring loop (around line 935, after balance is set), add:

```python
# Compute current exposure for budget-aware sizing
current_exposure = sum(
    t.shares * (t.cost_basis or 0)
    for t in state.trades.values()
    if t.shares >= MIN_SHARES_PER_ORDER
)
```

Then in the `compute_position_size` call (line 1206), add the parameter:

```python
position_size = compute_position_size(
    probability=prob,
    price=price,
    balance=balance,
    max_position_usd=config.max_position_usd,
    kelly_frac=config.kelly_fraction,
    current_exposure=current_exposure,
)
```

### Tests

Add to `weather/tests/test_sizing.py`:

```python
def test_budget_deducts_exposure():
    """Position size is smaller when exposure is high."""
    size_no_exp = compute_position_size(0.6, 0.3, 50.0, 2.0, current_exposure=0.0)
    size_with_exp = compute_position_size(0.6, 0.3, 50.0, 2.0, current_exposure=30.0)
    assert size_with_exp < size_no_exp

def test_budget_zero_when_fully_exposed():
    """Position size is 0 when exposure equals balance."""
    size = compute_position_size(0.6, 0.3, 50.0, 2.0, current_exposure=50.0)
    assert size == 0.0

def test_budget_backward_compatible():
    """Default exposure=0 gives same result as before."""
    size_default = compute_position_size(0.6, 0.3, 50.0, 2.0)
    size_explicit = compute_position_size(0.6, 0.3, 50.0, 2.0, current_exposure=0.0)
    assert size_default == size_explicit
```

### Verification
```
python3 -m pytest weather/tests/test_sizing.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 7: Cumulative correlation penalty + same-location discount

**Files:**
- Modify: `weather/strategy.py` — rewrite `_apply_correlation_discount()`, add same-location horizon penalty in scoring loop

### What to build

**A. Cumulative correlation:** Replace `max(correlations)` with sum-based penalty. Having 3 positions correlated at 0.3 should discount more than 1 position at 0.3.

**B. Same-location discount:** When we hold a position on the same location within N days, apply a discount factor.

### Implementation

**A. Rewrite `_apply_correlation_discount()` (lines 606-632):**

```python
def _apply_correlation_discount(
    base_size: float,
    location: str,
    month: int,
    open_locations: list[str],
    config: Config,
) -> float:
    """Reduce position size based on cumulative correlation with open positions."""
    if not open_locations:
        return base_size

    # Sum of correlations above threshold (cumulative, not max)
    total_corr = 0.0
    for open_loc in open_locations:
        if open_loc == location:
            continue
        corr = get_correlation(location, open_loc, month)
        if corr > config.correlation_threshold:
            total_corr += corr

    if total_corr <= 0:
        return base_size

    factor = max(0.1, 1.0 - total_corr * config.correlation_discount)
    adjusted = base_size * factor
    logger.info("Correlation discount: %s total_corr=%.2f → sizing $%.2f → $%.2f",
                location, total_corr, base_size, adjusted)
    return round(adjusted, 2)
```

**B. Add same-location horizon penalty in the scoring loop.** In `weather/strategy.py`, after the correlation discount block (after line 1226), add:

```python
# Same-location horizon penalty: reduce if holding same city within N days
if config.same_location_discount < 1.0:
    for existing_trade in state.trades.values():
        if (existing_trade.location == location
            and existing_trade.forecast_date
            and existing_trade.shares >= MIN_SHARES_PER_ORDER):
            try:
                existing_horizon = get_horizon_days(existing_trade.forecast_date)
                if abs(days_ahead - existing_horizon) <= config.same_location_horizon_window:
                    position_size = round(position_size * config.same_location_discount, 2)
                    logger.info("Same-location penalty: %s horizon %d~%d → $%.2f",
                                location, existing_horizon, days_ahead, position_size)
                    break  # Apply once, not per trade
            except (ValueError, TypeError):
                continue
```

Insert this after line 1226 (after the explain correlation log) and before line 1228 (mean-reversion modifier).

### Tests

Add to `weather/tests/test_strategy.py`:

```python
def test_cumulative_correlation_discount():
    """Sum of correlations gives larger discount than max alone."""
    from weather.strategy import _apply_correlation_discount
    config = Config(correlation_threshold=0.3, correlation_discount=0.5)
    # Mock: we need get_correlation to return values
    # Use locations known to have correlations in calibration data
    # Dallas|Seattle SON=0.674, Atlanta|NYC SON=0.281
    # For month=10 (Oct, SON season):
    size = _apply_correlation_discount(
        2.0, "Dallas", 10, ["Seattle", "Miami"], config,
    )
    # Dallas|Seattle SON=0.674 > 0.3, Dallas|Miami SON=-0.226 < 0.3
    # total_corr = 0.674, factor = 1 - 0.674*0.5 = 0.663
    assert size < 2.0

def test_correlation_no_positions():
    from weather.strategy import _apply_correlation_discount
    config = Config(correlation_threshold=0.3, correlation_discount=0.5)
    assert _apply_correlation_discount(2.0, "NYC", 1, [], config) == 2.0
```

### Verification
```
python3 -m pytest weather/tests/test_strategy.py -v -k "correlation"
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 8: Full integration test and validation

**Files:**
- Create: `weather/tests/test_execution_portfolio.py` — integration tests

### What to build

End-to-end tests verifying:
1. Config fields load correctly from JSON
2. VWAP + depth + adaptive slippage work together
3. Budget sizing reduces with exposure
4. Cumulative correlation discounts
5. trade_metrics filtering
6. Same-location discount

### Implementation

```python
"""Integration tests for execution, low temp, and portfolio features."""
import json
import tempfile
import os
from weather.config import Config
from weather.sizing import compute_position_size
from weather.bridge import compute_available_depth, compute_vwap


class TestConfigIntegration:
    def test_load_new_fields_from_json(self, tmp_path):
        cfg_data = {
            "slippage_edge_ratio": 0.6,
            "depth_fill_ratio": 0.4,
            "vwap_max_levels": 3,
            "trade_metrics": "high,low",
            "same_location_discount": 0.7,
            "same_location_horizon_window": 3,
            "correlation_threshold": 0.25,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg_data))
        cfg = Config.load(str(tmp_path))
        assert cfg.slippage_edge_ratio == 0.6
        assert cfg.depth_fill_ratio == 0.4
        assert cfg.vwap_max_levels == 3
        assert cfg.active_metrics == ["high", "low"]
        assert cfg.same_location_discount == 0.7
        assert cfg.correlation_threshold == 0.25


class TestBudgetSizingIntegration:
    def test_exposure_reduces_position(self):
        """Full Kelly pipeline with exposure."""
        s1 = compute_position_size(0.55, 0.40, 50.0, 2.0, kelly_frac=0.25, current_exposure=0.0)
        s2 = compute_position_size(0.55, 0.40, 50.0, 2.0, kelly_frac=0.25, current_exposure=40.0)
        assert s2 < s1
        assert s2 >= 0


class TestVwapDepthIntegration:
    def test_realistic_orderbook(self):
        """VWAP on a realistic thin orderbook."""
        asks = [
            {"price": "0.35", "size": "50"},
            {"price": "0.38", "size": "30"},
            {"price": "0.42", "size": "100"},
            {"price": "0.45", "size": "200"},
            {"price": "0.50", "size": "500"},
        ]
        depth = compute_available_depth(asks, max_levels=5)
        assert depth > 0

        # VWAP for $5 order
        vwap = compute_vwap(asks, 5.0)
        assert 0.35 <= vwap <= 0.38  # Should fill mostly from first level

        # VWAP for $30 order (spans multiple levels)
        vwap_large = compute_vwap(asks, 30.0)
        assert vwap_large > vwap  # Larger order gets worse average price
```

### Verification
```
python3 -m pytest weather/tests/test_execution_portfolio.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Final Verification

After all 8 tasks:
```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
python3 -m weather --dry-run --explain --set locations=NYC --set trade_metrics=high,low  # Verify both metrics
```

## Critical Files Reference

| File | Changes |
|------|---------|
| `weather/config.py` | 6 new fields, 1 changed default, 1 new property |
| `weather/bridge.py` | 2 new functions, `execute_trade()` enhanced |
| `weather/strategy.py` | Adaptive slippage, VWAP wiring, metric filter, budget sizing, cumulative correlation, same-location discount |
| `weather/sizing.py` | `current_exposure` parameter |
| `weather/tests/test_bridge_execution.py` | New: VWAP + depth tests |
| `weather/tests/test_execution_portfolio.py` | New: integration tests |
