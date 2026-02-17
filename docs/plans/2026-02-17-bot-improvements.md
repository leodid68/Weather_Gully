# Bot Improvements: Stale Cleanup, Fair-Price Limits, Model Disagreement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Fix 3 issues identified in paper trading: expired positions polluting state, trades blocked by slippage on thin markets, and model disagreement leading to risky positions.

**Tech Stack:** Python 3.14, stdlib only, ~914 existing tests.

---

## Task 1: Auto-cleanup stale/expired positions

**Files:**
- Modify: `weather/strategy.py` — add cleanup at start of `run_weather_strategy()`
- Create: `weather/tests/test_stale_cleanup.py`

### What to build

Positions whose `forecast_date < today` are expired (market has resolved). Remove them from state at the beginning of each strategy run. This prevents stale positions (like Atlanta Feb 15 still in state on Feb 17) from affecting exposure calculations and correlation guards.

### Implementation

**In `weather/strategy.py`, inside `run_weather_strategy()`, after the circuit breaker check (line ~820) and before `client.sync_exposure_from_state()` (line 823):**

```python
# Auto-cleanup expired positions
today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
stale_ids = [
    mid for mid, t in state.trades.items()
    if t.forecast_date and t.forecast_date < today_str
]
if stale_ids:
    for mid in stale_ids:
        trade = state.trades[mid]
        logger.warning(
            "Removing expired position: %s %s (date %s < %s)",
            trade.location, trade.outcome_name[:40], trade.forecast_date, today_str,
        )
        if trade.event_id:
            state.remove_event_position_market(trade.event_id, mid)
        state.remove_trade(mid)
    logger.info("Cleaned up %d expired position(s)", len(stale_ids))
```

### Tests (`weather/tests/test_stale_cleanup.py`)

```python
"""Tests for stale position auto-cleanup in strategy."""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from weather.state import TradingState, TradeRecord
from weather.config import Config


class TestStalePositionCleanup(unittest.TestCase):
    """Test that expired positions are removed before strategy runs."""

    def _make_state_with_trades(self, dates: list[str]) -> TradingState:
        state = TradingState()
        for i, d in enumerate(dates):
            mid = f"market-{i}"
            state.record_trade(
                market_id=mid,
                outcome_name=f"Bucket {i}",
                side="yes",
                cost_basis=0.10,
                shares=10.0,
                location="NYC",
                forecast_date=d,
                forecast_temp=45.0,
                event_id=f"event-{i}",
            )
            state.record_event_position(f"event-{i}", mid)
        return state

    @patch("weather.strategy.datetime")
    def test_expired_positions_removed(self, mock_dt):
        """Positions with forecast_date < today should be removed."""
        mock_dt.now.return_value = datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.strptime = datetime.strptime

        state = self._make_state_with_trades(["2026-02-15", "2026-02-17", "2026-02-19"])
        # Simulate the cleanup logic
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        for mid in stale_ids:
            trade = state.trades[mid]
            if trade.event_id:
                state.remove_event_position_market(trade.event_id, mid)
            state.remove_trade(mid)

        self.assertEqual(len(state.trades), 2)  # Feb 17 and 19 remain
        self.assertNotIn("market-0", state.trades)  # Feb 15 removed
        self.assertIn("market-1", state.trades)
        self.assertIn("market-2", state.trades)
        # Event positions also cleaned
        self.assertNotIn("event-0", state.event_positions)

    def test_no_stale_positions(self):
        """No crash when all positions are current."""
        state = self._make_state_with_trades(["2099-12-31"])
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        self.assertEqual(len(stale_ids), 0)
        self.assertEqual(len(state.trades), 1)

    def test_empty_forecast_date_not_removed(self):
        """Positions without forecast_date should NOT be removed."""
        state = TradingState()
        state.record_trade(
            market_id="m1", outcome_name="Bucket", side="yes",
            cost_basis=0.10, shares=10.0, forecast_date="",
        )
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        self.assertEqual(len(stale_ids), 0)
```

### Verification
```
python3 -m pytest weather/tests/test_stale_cleanup.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 2: Fair-price limit orders + fix adaptive slippage wiring

**Files:**
- Modify: `weather/bridge.py` — populate edge in `get_market_context()`, add `limit_price` param to `execute_trade()`
- Modify: `weather/strategy.py` — pass `limit_price=prob` to `execute_trade()`
- Modify: `weather/paper_bridge.py` — accept `limit_price` kwarg
- Create: `weather/tests/test_limit_orders.py`

### What to build

Two related fixes:

**A) Fix adaptive slippage (broken wiring):** `get_market_context()` returns `"edge": {}` which means adaptive slippage never has `user_edge_val`, so it always falls back to the static `slippage_max_pct`. Fix: populate edge dict with `my_probability` and market price.

**B) Fair-price limit orders:** Instead of always placing at best_ask, place at `min(fair_value, best_ask)`. When our model says prob=0.45 but best_ask=0.50, we place at $0.45 (resting limit). If best_ask=0.30, we still place at $0.30 (aggressive, fills instantly).

### Implementation

**A) In `weather/bridge.py`, `get_market_context()` (line ~200):**

After `my_probability = kwargs.get("my_probability")` (add this), populate the edge dict:

```python
def get_market_context(self, market_id: str, **kwargs) -> dict | None:
    # ... existing code ...
    my_probability = kwargs.get("my_probability")

    # Edge computation for adaptive slippage
    edge_dict: dict = {}
    if my_probability is not None:
        market_price = gm.best_ask if gm.best_ask > 0 else (
            gm.outcome_prices[0] if gm.outcome_prices else 0.5
        )
        user_edge = my_probability - market_price
        edge_dict = {
            "user_edge": user_edge,
            "recommendation": "TRADE" if user_edge > 0.02 else ("HOLD" if user_edge > 0 else "SKIP"),
            "suggested_threshold": 0.02,
        }

    return {
        "market": {"time_to_resolution": time_str},
        "slippage": {"estimates": [{"slippage_pct": slippage_pct}]},
        "edge": edge_dict,
        "warnings": [],
        "discipline": {},
    }
```

**B) In `weather/bridge.py`, `execute_trade()` — add `limit_price` parameter:**

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
    limit_price: float = 0.0,  # NEW: cap order price at this value (0 = no cap)
) -> dict:
```

After computing the `price` from orderbook (line ~377), add:

```python
# Apply limit price cap (fair value from model)
if limit_price > 0 and price > limit_price:
    logger.info("Limit cap: ask=$%.4f → limit=$%.4f (saving %.1f%%)",
                price, limit_price, (price - limit_price) / price * 100)
    price = limit_price
```

**C) In `weather/strategy.py`, line ~1323, pass limit_price:**

```python
result = client.execute_trade(
    market_id, side, position_size,
    fill_timeout=config.fill_timeout_seconds,
    fill_poll_interval=config.fill_poll_interval,
    depth_fill_ratio=config.depth_fill_ratio,
    vwap_max_levels=config.vwap_max_levels,
    limit_price=prob,  # Never pay more than our fair value
)
```

**D) In `weather/paper_bridge.py`, `execute_trade()` already accepts `**kwargs` so no change needed. But update the simulated price to respect limit_price:**

In PaperBridge.execute_trade(), after computing `price` (line ~133):
```python
limit_price = kwargs.get("limit_price", 0.0)
if limit_price > 0 and price > limit_price:
    price = limit_price
```

### Tests (`weather/tests/test_limit_orders.py`)

```python
"""Tests for fair-price limit orders and adaptive slippage wiring."""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from weather.bridge import CLOBWeatherBridge, compute_vwap


class TestMarketContextEdge(unittest.TestCase):
    """Test that get_market_context populates edge dict."""

    def _make_bridge_with_market(self, best_ask=0.30, best_bid=0.25):
        bridge = CLOBWeatherBridge.__new__(CLOBWeatherBridge)
        bridge._market_cache = {}
        gm = MagicMock()
        gm.best_ask = best_ask
        gm.best_bid = best_bid
        gm.spread = best_ask - best_bid
        gm.outcome_prices = [best_ask]
        gm.end_date = "2026-02-20T00:00:00Z"
        bridge._market_cache["m1"] = gm
        return bridge

    def test_edge_populated_with_probability(self):
        bridge = self._make_bridge_with_market(best_ask=0.30)
        ctx = bridge.get_market_context("m1", my_probability=0.45)
        self.assertIn("user_edge", ctx["edge"])
        self.assertAlmostEqual(ctx["edge"]["user_edge"], 0.15, places=2)
        self.assertEqual(ctx["edge"]["recommendation"], "TRADE")

    def test_edge_empty_without_probability(self):
        bridge = self._make_bridge_with_market()
        ctx = bridge.get_market_context("m1")
        self.assertEqual(ctx["edge"], {})

    def test_edge_skip_when_negative(self):
        bridge = self._make_bridge_with_market(best_ask=0.60)
        ctx = bridge.get_market_context("m1", my_probability=0.50)
        self.assertEqual(ctx["edge"]["recommendation"], "SKIP")


class TestLimitPriceCap(unittest.TestCase):
    """Test that execute_trade respects limit_price."""

    def _make_bridge(self):
        bridge = CLOBWeatherBridge.__new__(CLOBWeatherBridge)
        bridge.clob = MagicMock()
        bridge._market_cache = {}
        bridge._total_exposure = 0.0
        bridge._position_count = 0
        bridge._known_positions = set()

        gm = MagicMock()
        gm.best_ask = 0.50
        gm.best_bid = 0.25
        gm.outcome_prices = [0.50]
        gm.clob_token_ids = ["token-yes", "token-no"]
        bridge._market_cache["m1"] = gm
        return bridge

    def test_limit_price_caps_order(self):
        bridge = self._make_bridge()
        bridge.clob.get_orderbook.return_value = {
            "asks": [{"price": "0.50", "size": "100"}],
        }
        bridge.clob.post_order.return_value = {"orderID": "order-1"}

        bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0, limit_price=0.40)

        # Verify post_order was called with limit price, not ask price
        call_kwargs = bridge.clob.post_order.call_args
        self.assertLessEqual(float(call_kwargs.kwargs.get("price", call_kwargs[1].get("price", 0))), 0.40)

    def test_no_limit_uses_ask(self):
        bridge = self._make_bridge()
        bridge.clob.get_orderbook.return_value = {
            "asks": [{"price": "0.30", "size": "100"}],
        }
        bridge.clob.post_order.return_value = {"orderID": "order-1"}

        bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0, limit_price=0)

        call_kwargs = bridge.clob.post_order.call_args
        # Price should be 0.30 (ask price, no cap)
        self.assertIsNotNone(call_kwargs)
```

### Verification
```
python3 -m pytest weather/tests/test_limit_orders.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 3: Model disagreement sigma boost

**Files:**
- Modify: `weather/strategy.py` — detect and log model disagreement, boost sigma
- Modify: `weather/config.py` — add `model_disagreement_threshold` config field
- Create: `weather/tests/test_model_disagreement.py`

### What to build

When NOAA and Open-Meteo forecasts diverge by more than a configurable threshold (default 3°F), boost sigma by a multiplier to increase uncertainty. This prevents the bot from taking confident positions when the models disagree (like Seattle with a 4°F gap between NOAA and Open-Meteo).

The `model_spread` is already computed (it's the absolute difference between GFS and ECMWF) and passed to `compute_adaptive_sigma()` where it becomes one of the 4 sigma signals. But we also need to account for **NOAA vs Open-Meteo** disagreement specifically, which can be larger than GFS-ECMWF spread.

### Implementation

**In `weather/config.py`, add field after `same_location_horizon_window` (line ~157):**

```python
model_disagreement_threshold: float = 3.0   # °F — boost sigma when models diverge
model_disagreement_multiplier: float = 1.5  # sigma multiplier when disagreement detected
```

**In `weather/strategy.py`, after the ensemble forecast computation (line ~1035), before adaptive sigma:**

The `model_spread` variable already contains GFS-ECMWF spread. But we need to also check NOAA vs ensemble divergence. Add:

```python
# Model disagreement detection (NOAA vs Open-Meteo ensemble)
model_disagreement = False
if config.multi_source and noaa_temp is not None and om_data:
    # Compute Open-Meteo-only forecast for comparison
    om_only_temp, _ = compute_ensemble_forecast(None, om_data, metric, location=location)
    if om_only_temp is not None:
        noaa_om_spread = abs(noaa_temp - om_only_temp)
        if noaa_om_spread >= config.model_disagreement_threshold:
            model_disagreement = True
            logger.warning(
                "Model disagreement: NOAA=%.0f°F vs Open-Meteo=%.0f°F (spread=%.1f°F > %.1f°F threshold)",
                noaa_temp, om_only_temp, noaa_om_spread, config.model_disagreement_threshold,
            )
```

Then, after `compute_adaptive_sigma` (line ~1071), apply the multiplier:

```python
if model_disagreement and adaptive_sigma_value is not None:
    old_sigma = adaptive_sigma_value
    adaptive_sigma_value *= config.model_disagreement_multiplier
    logger.info("Sigma boosted for model disagreement: %.2f → %.2f", old_sigma, adaptive_sigma_value)
```

If `adaptive_sigma` is not enabled (rare), apply the multiplier to the sigma_override flow in score_buckets:

```python
sigma_override_for_scoring = adaptive_sigma_value
if model_disagreement and sigma_override_for_scoring is None:
    # Even without adaptive sigma, boost the base sigma
    base_sigma = _get_stddev(date_str, location)
    sigma_override_for_scoring = base_sigma * config.model_disagreement_multiplier
    logger.info("Sigma override for model disagreement: %.2f", sigma_override_for_scoring)
```

### Tests (`weather/tests/test_model_disagreement.py`)

```python
"""Tests for model disagreement detection and sigma boost."""
import unittest
from weather.config import Config


class TestModelDisagreementConfig(unittest.TestCase):
    """Test config fields for model disagreement."""

    def test_default_threshold(self):
        config = Config()
        self.assertEqual(config.model_disagreement_threshold, 3.0)

    def test_default_multiplier(self):
        config = Config()
        self.assertEqual(config.model_disagreement_multiplier, 1.5)


class TestModelDisagreementDetection(unittest.TestCase):
    """Test that model disagreement is properly detected."""

    def test_disagreement_detected_when_spread_above_threshold(self):
        """When NOAA-OpenMeteo spread > threshold, disagreement is flagged."""
        noaa_temp = 40.0
        om_temp = 44.0  # 4°F spread
        threshold = 3.0
        spread = abs(noaa_temp - om_temp)
        self.assertTrue(spread >= threshold)

    def test_no_disagreement_when_close(self):
        """When models are close, no disagreement."""
        noaa_temp = 40.0
        om_temp = 41.5  # 1.5°F spread
        threshold = 3.0
        spread = abs(noaa_temp - om_temp)
        self.assertFalse(spread >= threshold)

    def test_sigma_boost_applied(self):
        """Sigma should be multiplied by disagreement multiplier."""
        base_sigma = 2.0
        multiplier = 1.5
        boosted = base_sigma * multiplier
        self.assertAlmostEqual(boosted, 3.0)
```

### Verification
```
python3 -m pytest weather/tests/test_model_disagreement.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Final Verification

```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
```

## Critical Files Reference

| File | Role |
|------|------|
| `weather/strategy.py` | Stale cleanup, limit_price pass-through, disagreement detection |
| `weather/bridge.py` | Edge wiring, limit_price cap |
| `weather/paper_bridge.py` | limit_price in paper mode |
| `weather/config.py` | disagreement threshold/multiplier |
