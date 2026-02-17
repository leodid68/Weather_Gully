# Maker Orders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement hybrid taker/maker execution for Polymarket weather markets, with a continuous order manager daemon for fill detection and TTL expiry.

**Architecture:** Strategy loop decides taker vs maker per trade based on edge/spread. Maker orders post GTC postOnly bids and are tracked in `pending_orders.json`. A separate daemon polls for fills, cancels expired orders, and reconciles on startup.

**Tech Stack:** Python 3.14, stdlib (`fcntl`, `json`, `concurrent.futures`, `time`), no new deps.

---

## Task 1: PendingOrders state management (`weather/pending_state.py`)

**Files:**
- Create: `weather/pending_state.py`
- Create: `weather/tests/test_pending_state.py`

### What to build

A `PendingOrders` class that manages `pending_orders.json` with file locking (same pattern as `state.py:state_lock()`). Each pending order is a dict:

```json
{
    "order_id": "0xabc...",
    "market_id": "0x123...",
    "token_id": "456...",
    "side": "yes",
    "price": 0.08,
    "size": 25.0,
    "amount_usd": 2.00,
    "submitted_at": "2026-02-17T10:30:00Z",
    "ttl_seconds": 900,
    "location": "NYC",
    "outcome_name": "41°F or below",
    "forecast_date": "2026-02-18",
    "prob": 0.25
}
```

### Implementation

```python
"""Pending maker orders state — file-backed with fcntl locking."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def pending_lock(path: str):
    """Exclusive file lock for pending orders (same pattern as state.state_lock)."""
    lock_path = path + ".lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError:
        logger.error("Failed to acquire pending orders lock")
        raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


class PendingOrders:
    """Thread-safe pending maker orders backed by JSON file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._orders: list[dict] = []

    def load(self) -> None:
        """Load pending orders from disk (call inside pending_lock)."""
        try:
            with open(self.path) as f:
                self._orders = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._orders = []

    def save(self) -> None:
        """Atomic save to disk (call inside pending_lock)."""
        dir_path = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._orders, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add(self, order: dict) -> None:
        """Add a pending order."""
        self._orders.append(order)

    def remove(self, order_id: str) -> dict | None:
        """Remove and return order by order_id, or None if not found."""
        for i, o in enumerate(self._orders):
            if o.get("order_id") == order_id:
                return self._orders.pop(i)
        return None

    def get_by_market(self, market_id: str) -> dict | None:
        """Return pending order for a market, or None."""
        for o in self._orders:
            if o.get("market_id") == market_id:
                return o
        return None

    def has_market(self, market_id: str) -> bool:
        """Check if there's a pending order for this market."""
        return any(o.get("market_id") == market_id for o in self._orders)

    def total_exposure(self) -> float:
        """Sum of all pending order amounts."""
        return sum(o.get("amount_usd", 0.0) for o in self._orders)

    @property
    def orders(self) -> list[dict]:
        """Read-only access to current orders."""
        return list(self._orders)

    def __len__(self) -> int:
        return len(self._orders)
```

### Tests (`weather/tests/test_pending_state.py`)

```python
import json
import os
import tempfile
import pytest
from weather.pending_state import PendingOrders, pending_lock

@pytest.fixture
def tmp_path_file(tmp_path):
    return str(tmp_path / "pending_orders.json")

def _sample_order(**overrides):
    base = {
        "order_id": "ox1", "market_id": "m1", "token_id": "t1",
        "side": "yes", "price": 0.10, "size": 20.0, "amount_usd": 2.0,
        "submitted_at": "2026-02-17T10:00:00Z", "ttl_seconds": 900,
        "location": "NYC", "outcome_name": "41°F or below",
        "forecast_date": "2026-02-18", "prob": 0.25,
    }
    base.update(overrides)
    return base

class TestPendingOrders:
    def test_add_and_len(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        assert len(po) == 0
        po.add(_sample_order())
        assert len(po) == 1

    def test_save_load_roundtrip(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order())
        po.save()
        po2 = PendingOrders(tmp_path_file)
        po2.load()
        assert len(po2) == 1
        assert po2.orders[0]["order_id"] == "ox1"

    def test_remove_existing(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(order_id="a"))
        po.add(_sample_order(order_id="b"))
        removed = po.remove("a")
        assert removed is not None
        assert removed["order_id"] == "a"
        assert len(po) == 1

    def test_remove_nonexistent(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        assert po.remove("nope") is None

    def test_has_market(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(market_id="m1"))
        assert po.has_market("m1") is True
        assert po.has_market("m2") is False

    def test_get_by_market(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(market_id="m1"))
        result = po.get_by_market("m1")
        assert result is not None
        assert result["market_id"] == "m1"
        assert po.get_by_market("m2") is None

    def test_total_exposure(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(amount_usd=2.0))
        po.add(_sample_order(order_id="ox2", market_id="m2", amount_usd=3.5))
        assert po.total_exposure() == pytest.approx(5.5)

    def test_atomic_save(self, tmp_path_file):
        """Save is atomic — interrupted write doesn't corrupt."""
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order())
        po.save()
        # File exists and is valid JSON
        with open(tmp_path_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_load_missing_file(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()  # Should not raise
        assert len(po) == 0

    def test_load_corrupt_file(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("{invalid json")
        po = PendingOrders(tmp_path_file)
        po.load()  # Should not raise
        assert len(po) == 0

class TestPendingLock:
    def test_lock_acquire_release(self, tmp_path_file):
        with pending_lock(tmp_path_file):
            pass  # Should not raise

    def test_lock_creates_lockfile(self, tmp_path_file):
        with pending_lock(tmp_path_file):
            assert os.path.exists(tmp_path_file + ".lock")
```

### Verification
```bash
python3 -m pytest weather/tests/test_pending_state.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 2: Add `post_only` parameter to CLOB client

**Files:**
- Modify: `polymarket/client.py` — `post_order()` method (line 293)
- Create: `weather/tests/test_post_only.py`

### What to build

Add `post_only: bool = False` parameter to `post_order()` and wire it into the request body instead of the hardcoded `False`.

### Implementation

In `polymarket/client.py`, modify `post_order()` signature (line 245):

Add `post_only: bool = False` parameter after `order_type`.

At line 293, change:
```python
# Before:
"postOnly": False,
# After:
"postOnly": post_only,
```

### Tests

```python
"""Test postOnly flag propagation in CLOB client."""
import json
from unittest.mock import patch, MagicMock
import pytest

class TestPostOnlyFlag:
    def test_post_only_false_by_default(self):
        """Default behavior: postOnly=False."""
        from polymarket.client import PolymarketCLOB
        client = PolymarketCLOB.__new__(PolymarketCLOB)
        client.host = "https://example.com"
        client.api_key = "test"
        client.api_secret = "test"
        client.api_passphrase = "test"
        client.chain_id = 137

        with patch.object(client, '_request') as mock_req:
            mock_req.return_value = {"orderID": "test123", "status": "LIVE"}
            client.post_order("token1", "BUY", 0.50, 10.0)
            call_args = mock_req.call_args
            body = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("body", {})
            # Check the body was passed with postOnly=False
            assert body.get("postOnly") is False

    def test_post_only_true_when_set(self):
        """postOnly=True is passed through."""
        from polymarket.client import PolymarketCLOB
        client = PolymarketCLOB.__new__(PolymarketCLOB)
        client.host = "https://example.com"
        client.api_key = "test"
        client.api_secret = "test"
        client.api_passphrase = "test"
        client.chain_id = 137

        with patch.object(client, '_request') as mock_req:
            mock_req.return_value = {"orderID": "test123", "status": "LIVE"}
            client.post_order("token1", "BUY", 0.50, 10.0, post_only=True)
            call_args = mock_req.call_args
            body = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("body", {})
            assert body.get("postOnly") is True
```

### Verification
```bash
python3 -m pytest weather/tests/test_post_only.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 3: Add maker config fields

**Files:**
- Modify: `weather/config.py` — Config dataclass
- Modify: `weather/config.json` — Add defaults

### What to build

Add 4 new maker order config fields to the `Config` dataclass:

```python
# Maker orders (hybrid taker/maker execution)
maker_edge_threshold: float = 0.05     # Edge below this → use maker
maker_spread_threshold: float = 0.10   # Spread above this → use maker
maker_ttl_seconds: int = 900           # 15 min before cancel
maker_tick_buffer: int = 1             # Ticks above best bid
```

Add after line 158 (after `model_disagreement_multiplier`).

Also add to `config.json`:
```json
"maker_edge_threshold": 0.05,
"maker_spread_threshold": 0.10,
"maker_ttl_seconds": 900,
"maker_tick_buffer": 1
```

### Verification
```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 4: Add `execute_maker_order()` to bridge

**Files:**
- Modify: `weather/bridge.py` — Add `execute_maker_order()` method
- Create: `weather/tests/test_maker_bridge.py`

### What to build

Add `execute_maker_order()` to `CLOBWeatherBridge` that posts a GTC postOnly bid and returns immediately (no fill wait).

### Implementation

Add to `CLOBWeatherBridge` after `execute_sell()`:

```python
def execute_maker_order(
    self,
    market_id: str,
    side: str,
    amount: float,
    maker_price: float,
) -> dict:
    """Post a GTC postOnly bid. Returns immediately, no fill wait.

    Returns dict with keys: success, posted, order_id, price, size, token_id.
    If CLOB rejects postOnly (would cross spread), returns {"posted": False}.
    """
    gm = self._market_cache.get(market_id)
    if not gm or not gm.clob_token_ids:
        return {"success": False, "posted": False, "error": "no market data"}

    token_id = gm.clob_token_ids[0]
    price = round(maker_price, 2)
    size = round(amount / price, 1) if price > 0 else 0.0

    if size < MIN_SHARES_PER_ORDER:
        return {"success": False, "posted": False, "error": "size below minimum"}

    clob_side = "BUY" if side.lower() in ("yes", "buy") else "SELL"

    try:
        result = self.clob.post_order(
            token_id=token_id,
            side=clob_side,
            price=price,
            size=size,
            neg_risk=True,
            order_type="GTC",
            post_only=True,
        )
        order_id = result.get("orderID", "")
        status = result.get("status", "")

        if not order_id:
            # CLOB rejected — likely would cross spread
            logger.info("Maker order rejected (postOnly would cross): %s", result)
            return {"success": False, "posted": False, "error": "rejected"}

        logger.info(
            "Maker order posted: %s %.1f shares @ $%.2f (order %s)",
            clob_side, size, price, order_id,
        )
        return {
            "success": True,
            "posted": True,
            "order_id": order_id,
            "price": price,
            "size": size,
            "token_id": token_id,
        }

    except Exception as exc:
        logger.warning("Maker order failed: %s", exc)
        return {"success": False, "posted": False, "error": str(exc)}
```

Import `MIN_SHARES_PER_ORDER` from `weather.config` at the top of `bridge.py` if not already imported.

### Tests (`weather/tests/test_maker_bridge.py`)

Test with mocked CLOB client:
- `test_maker_order_success`: Mock `post_order` returns valid orderID → `posted=True`
- `test_maker_order_rejected_crosses_spread`: Mock returns empty orderID → `posted=False`
- `test_maker_order_below_min_shares`: Amount too small → `posted=False`
- `test_maker_order_no_market_data`: Missing market cache → `posted=False`
- `test_maker_order_post_only_flag`: Verify `post_only=True` is passed to `clob.post_order`
- `test_maker_order_exception_handling`: Mock raises → graceful failure

### Verification
```bash
python3 -m pytest weather/tests/test_maker_bridge.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 5: Taker/maker decision in strategy

**Files:**
- Modify: `weather/strategy.py` — Trade execution block (~lines 1340-1400)
- Create: `weather/tests/test_maker_decision.py`

### What to build

Replace the current unconditional taker execution with a taker/maker decision:

```
Edge = prob - price
If edge > maker_edge_threshold AND spread < maker_spread_threshold:
    → TAKER (existing path)
Else:
    → MAKER: post bid at min(prob, best_bid + tick_size)
    → If best_bid + tick >= best_ask → no room, fall back to taker
    → Add to pending_orders
    → Do NOT call record_trade()
```

### Implementation

1. At the top of `strategy.py`, add import:
```python
from .pending_state import PendingOrders, pending_lock
```

2. In `run_weather_strategy()` signature, add parameters:
```python
pending: PendingOrders | None = None,
```

3. Before the main event loop, load pending orders:
```python
pending_path = str(Path(config_dir) / "pending_orders.json") if config_dir else "pending_orders.json"
if pending is None:
    pending = PendingOrders(pending_path)
    with pending_lock(pending_path):
        pending.load()
```

4. In the budget calculation, add pending exposure:
```python
pending_exposure = pending.total_exposure() if pending else 0.0
effective_exposure = current_exposure + pending_exposure
```

5. In the duplicate guard (where it checks `state.trades`), also check pending:
```python
if market_id in state.trades:
    continue
if pending and pending.has_market(market_id):
    logger.debug("Pending maker order exists for %s — skip", market_id)
    continue
```

6. Replace the trade execution block with taker/maker decision:

```python
# --- Taker vs Maker decision ---
book = client.get_orderbook(token_id) if hasattr(client, '_market_cache') else None
best_bid = 0.0
best_ask = price  # fallback
spread = 1.0
if book:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if bids:
        best_bid = float(bids[0]["price"])
    if asks:
        best_ask = float(asks[0]["price"])
    spread = best_ask - best_bid

edge = prob - price
use_taker = (edge > config.maker_edge_threshold and spread < config.maker_spread_threshold)

if not use_taker:
    # Maker path
    tick = 0.01
    maker_price = min(prob, best_bid + tick * config.maker_tick_buffer)
    maker_price = round(maker_price, 2)

    # No room for maker if bid + tick >= ask
    if best_bid + tick >= best_ask:
        logger.info("No room for maker (bid $%.2f + tick >= ask $%.2f) — fallback to taker", best_bid, best_ask)
        use_taker = True
    elif maker_price <= 0:
        use_taker = True

if not use_taker:
    # Execute maker order
    if dry_run:
        logger.info("[DRY RUN] Would post MAKER %s $%.2f @ $%.2f on '%s'",
                     side.upper(), position_size, maker_price, outcome_name)
    else:
        result = client.execute_maker_order(market_id, side, position_size, maker_price)
        if result.get("posted"):
            order_entry = {
                "order_id": result["order_id"],
                "market_id": market_id,
                "token_id": result.get("token_id", ""),
                "side": side,
                "price": result["price"],
                "size": result["size"],
                "amount_usd": position_size,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "ttl_seconds": config.maker_ttl_seconds,
                "location": location,
                "outcome_name": outcome_name,
                "forecast_date": date_str,
                "prob": prob,
            }
            with pending_lock(pending_path):
                pending.load()
                pending.add(order_entry)
                pending.save()
            logger.info("Maker order queued: %s $%.2f @ $%.2f", side.upper(), position_size, maker_price)
            trades_executed += 1
        else:
            logger.info("Maker order rejected — fallback to taker")
            use_taker = True

if use_taker:
    # Existing taker execution (unchanged)
    ...existing code...
```

### Tests (`weather/tests/test_maker_decision.py`)

- `test_high_edge_low_spread_uses_taker`: edge=0.10 > 0.05, spread=0.05 < 0.10 → taker
- `test_low_edge_uses_maker`: edge=0.03 < 0.05 → maker
- `test_high_spread_uses_maker`: spread=0.15 > 0.10 → maker
- `test_no_room_for_maker_falls_back`: bid + tick >= ask → taker
- `test_maker_price_capped_at_prob`: maker_price = min(prob, bid+tick)
- `test_pending_exposure_counted_in_budget`: pending $5 + state $10 = $15 effective
- `test_duplicate_market_in_pending_skipped`: market already in pending → skip
- `test_maker_rejected_falls_back_to_taker`: execute_maker_order returns posted=False → taker

### Verification
```bash
python3 -m pytest weather/tests/test_maker_decision.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 6: Paper bridge maker support

**Files:**
- Modify: `weather/paper_bridge.py` — Add `execute_maker_order()` method
- Add tests to `weather/tests/test_maker_decision.py`

### What to build

Add `execute_maker_order()` to `PaperBridge` that simulates maker behavior. If `maker_price >= best_bid` (realistic fill scenario), simulate immediate fill. Otherwise, return posted but not filled (for order manager to handle).

### Implementation

Add to `PaperBridge` class:

```python
def execute_maker_order(
    self,
    market_id: str,
    side: str,
    amount: float,
    maker_price: float,
) -> dict:
    """Simulate maker order. In paper mode, assume immediate fill if price is competitive."""
    import uuid
    gm = self._market_cache.get(market_id)
    if not gm:
        return {"success": False, "posted": False, "error": "no market data"}

    price = round(maker_price, 2)
    if price <= 0:
        return {"success": False, "posted": False, "error": "invalid price"}

    size = round(amount / price, 1)
    order_id = f"paper-maker-{uuid.uuid4().hex[:8]}"

    logger.info("[PAPER] Maker order: %s $%.2f @ $%.2f on '%s'",
                side.upper(), amount, price, market_id[:16])

    return {
        "success": True,
        "posted": True,
        "order_id": order_id,
        "price": price,
        "size": size,
        "token_id": gm.clob_token_ids[0] if gm.clob_token_ids else "",
    }
```

### Tests

- `test_paper_maker_order_success`: Returns posted=True with valid order_id
- `test_paper_maker_order_no_market`: Missing market → posted=False

### Verification
```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 7: Order manager daemon (`weather/order_manager.py`)

**Files:**
- Create: `weather/order_manager.py`
- Create: `weather/tests/test_order_manager.py`

### What to build

A daemon that polls every 10 seconds:
1. Load pending orders
2. Check each order: TTL expired? → cancel. Filled? → record_trade. Partial? → update.
3. On startup, reconcile pending vs `get_open_orders()`

### Implementation

```python
"""Order manager daemon — monitors pending maker orders for fills and TTL expiry."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .pending_state import PendingOrders, pending_lock
from .state import TradingState, state_lock

logger = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds


def reconcile_on_startup(
    clob,
    pending: PendingOrders,
    pending_path: str,
) -> int:
    """Reconcile pending_orders.json with actual CLOB open orders on startup.

    Remove stale entries (orders that no longer exist on CLOB).
    Returns number of stale entries cleaned.
    """
    try:
        open_orders = clob.get_open_orders()
    except Exception as exc:
        logger.warning("Failed to fetch open orders for reconciliation: %s", exc)
        return 0

    open_ids = {o.get("id") or o.get("orderID") for o in open_orders}
    cleaned = 0

    with pending_lock(pending_path):
        pending.load()
        stale = [o for o in pending.orders if o["order_id"] not in open_ids]
        for o in stale:
            logger.info("Reconcile: removing stale pending order %s", o["order_id"])
            pending.remove(o["order_id"])
            cleaned += 1
        if cleaned:
            pending.save()

    return cleaned


def poll_once(
    clob,
    pending: PendingOrders,
    pending_path: str,
    state: TradingState,
    state_path: str,
) -> tuple[int, int, int]:
    """Single poll iteration. Returns (fills, cancels, errors)."""
    fills = 0
    cancels = 0
    errors = 0
    now = datetime.now(timezone.utc)

    with pending_lock(pending_path):
        pending.load()
        to_remove: list[str] = []
        to_record: list[dict] = []

        for order in list(pending.orders):
            order_id = order["order_id"]

            # 1. TTL check
            submitted = datetime.fromisoformat(order["submitted_at"])
            elapsed = (now - submitted).total_seconds()
            if elapsed > order.get("ttl_seconds", 900):
                logger.info("TTL expired for order %s (%.0fs)", order_id, elapsed)
                try:
                    clob.cancel_order(order_id)
                except Exception as exc:
                    logger.debug("Cancel failed (may already be gone): %s", exc)
                to_remove.append(order_id)
                cancels += 1
                continue

            # 2. Check fill status
            try:
                status_resp = clob.get_order(order_id)
            except Exception as exc:
                logger.debug("Get order failed for %s: %s", order_id, exc)
                errors += 1
                continue

            status = status_resp.get("status", "")
            size_matched = float(status_resp.get("size_matched", 0))
            original_size = float(status_resp.get("original_size") or order.get("size", 0))

            if status == "MATCHED" or (original_size > 0 and size_matched >= original_size * 0.99):
                # Fully filled
                logger.info("Order %s FILLED: %.1f shares @ $%.2f",
                            order_id, size_matched, order["price"])
                to_record.append({
                    "market_id": order["market_id"],
                    "outcome_name": order["outcome_name"],
                    "side": order["side"],
                    "cost_basis": order["price"],
                    "shares": size_matched or order["size"],
                    "location": order.get("location", ""),
                    "forecast_date": order.get("forecast_date", ""),
                    "prob": order.get("prob"),
                })
                to_remove.append(order_id)
                fills += 1

            elif status == "CANCELLED":
                logger.info("Order %s cancelled externally", order_id)
                to_remove.append(order_id)
                cancels += 1

            elif size_matched > 0 and size_matched < original_size * 0.99:
                # Partial fill — update remaining
                logger.info("Order %s partial fill: %.1f / %.1f",
                            order_id, size_matched, original_size)

        # Apply removals
        for oid in to_remove:
            pending.remove(oid)
        pending.save()

    # Record fills in trading state (outside pending lock to avoid deadlock)
    if to_record:
        with state_lock(state_path):
            state_data = TradingState.load(state_path)
            for rec in to_record:
                state_data.record_trade(**rec)
            state_data.save(state_path)

    return fills, cancels, errors


def run_manager(
    clob,
    config_dir: str = "",
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Main daemon loop. Runs until KeyboardInterrupt."""
    config_dir = config_dir or str(Path(__file__).parent)
    config = Config.load(config_dir)

    pending_path = str(Path(config_dir) / "pending_orders.json")
    state_path = str(Path(config_dir) / config.state_file)

    pending = PendingOrders(pending_path)
    state = TradingState.load(state_path)

    logger.info("Order manager starting — poll every %ds", poll_interval)

    # Startup reconciliation
    cleaned = reconcile_on_startup(clob, pending, pending_path)
    if cleaned:
        logger.info("Reconciliation: cleaned %d stale entries", cleaned)

    while True:
        try:
            fills, cancels, errors = poll_once(clob, pending, pending_path, state, state_path)
            if fills or cancels:
                logger.info("Poll: %d fills, %d cancels, %d errors", fills, cancels, errors)
        except Exception as exc:
            logger.error("Poll error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # In standalone mode, needs CLOB client
    from .bridge import CLOBWeatherBridge
    config_dir = str(Path(__file__).parent)
    config = Config.load(config_dir)
    creds = config.load_api_creds(config_dir)
    if not creds:
        print("No creds.json found — cannot start order manager", file=sys.stderr)
        sys.exit(1)

    bridge = CLOBWeatherBridge(config=config, api_creds=creds)
    run_manager(bridge.clob, config_dir=config_dir)
```

### Tests (`weather/tests/test_order_manager.py`)

```python
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import pytest
from weather.order_manager import poll_once, reconcile_on_startup
from weather.pending_state import PendingOrders

def _make_order(**overrides):
    base = {
        "order_id": "ox1", "market_id": "m1", "token_id": "t1",
        "side": "yes", "price": 0.10, "size": 20.0, "amount_usd": 2.0,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "ttl_seconds": 900,
        "location": "NYC", "outcome_name": "41°F or below",
        "forecast_date": "2026-02-18", "prob": 0.25,
    }
    base.update(overrides)
    return base

class TestPollOnce:
    def test_fill_detected(self, tmp_path):
        """Filled order → record_trade + remove from pending."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "MATCHED", "size_matched": 20.0, "original_size": 20.0}

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 1
        assert cancels == 0
        po.load()
        assert len(po) == 0

    def test_ttl_expired(self, tmp_path):
        """Expired order → cancel + remove."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
        po.add(_make_order(order_id="ox1", submitted_at=old_time, ttl_seconds=900))
        po.save()

        mock_clob = MagicMock()

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1
        mock_clob.cancel_order.assert_called_once_with("ox1")

    def test_cancelled_externally(self, tmp_path):
        """Externally cancelled → cleanup."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "CANCELLED", "size_matched": 0}

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1
        po.load()
        assert len(po) == 0

    def test_still_live_no_change(self, tmp_path):
        """Live order, not expired → no action."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "LIVE", "size_matched": 0, "original_size": 20}

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 0 and cancels == 0
        po.load()
        assert len(po) == 1

class TestReconciliation:
    def test_removes_stale(self, tmp_path):
        """Stale entries not in CLOB open orders → removed."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.add(_make_order(order_id="ox2", market_id="m2"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_open_orders.return_value = [{"id": "ox2"}]  # ox1 not present

        cleaned = reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 1
        po.load()
        assert len(po) == 1
        assert po.orders[0]["order_id"] == "ox2"

    def test_no_stale(self, tmp_path):
        """All entries match CLOB → nothing removed."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_open_orders.return_value = [{"id": "ox1"}]

        cleaned = reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 0
```

### Verification
```bash
python3 -m pytest weather/tests/test_order_manager.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 8: Integration tests

**Files:**
- Create: `weather/tests/test_maker_integration.py`

### What to build

End-to-end test of the full maker cycle:
1. Strategy posts maker order → pending_orders updated
2. Order manager detects fill → record_trade called, pending cleaned
3. Strategy detects pending → skips duplicate
4. TTL expiry → capital freed

### Tests

```python
"""Integration tests for the full maker order lifecycle."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from weather.pending_state import PendingOrders, pending_lock
from weather.order_manager import poll_once, reconcile_on_startup

def _make_order(**overrides):
    base = {
        "order_id": "ox1", "market_id": "m1", "token_id": "t1",
        "side": "yes", "price": 0.10, "size": 20.0, "amount_usd": 2.0,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "ttl_seconds": 900,
        "location": "NYC", "outcome_name": "41°F or below",
        "forecast_date": "2026-02-18", "prob": 0.25,
    }
    base.update(overrides)
    return base

class TestFullMakerCycle:
    def test_post_fill_record(self, tmp_path):
        """Full cycle: post maker → fill → record in state."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")

        # 1. Strategy posts maker order
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()
        assert len(po) == 1

        # 2. Order manager detects fill
        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "MATCHED", "size_matched": 20.0, "original_size": 20.0}

        fills, _, _ = poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 1

        # 3. Pending cleaned
        po.load()
        assert len(po) == 0

    def test_pending_blocks_duplicate(self, tmp_path):
        """Market already in pending → has_market returns True."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(market_id="m1"))
        po.save()

        assert po.has_market("m1") is True
        assert po.has_market("m2") is False

    def test_ttl_frees_capital(self, tmp_path):
        """TTL expired → order cancelled, exposure freed."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
        po.add(_make_order(order_id="ox1", submitted_at=old_time, ttl_seconds=900, amount_usd=5.0))
        po.save()

        initial_exposure = po.total_exposure()
        assert initial_exposure == pytest.approx(5.0)

        mock_clob = MagicMock()
        _, cancels, _ = poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1

        po.load()
        assert po.total_exposure() == pytest.approx(0.0)

    def test_pending_exposure_in_budget(self, tmp_path):
        """Pending orders counted in effective exposure."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(amount_usd=3.0))
        po.add(_make_order(order_id="ox2", market_id="m2", amount_usd=4.0))

        assert po.total_exposure() == pytest.approx(7.0)
```

### Verification
```bash
python3 -m pytest weather/tests/test_maker_integration.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Final Verification

After all 8 tasks:
```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
```

## Critical Files Reference

| File | Role |
|------|------|
| `weather/pending_state.py` | NEW: Pending orders state with file lock |
| `weather/order_manager.py` | NEW: Daemon for fill detection/TTL/reconciliation |
| `polymarket/client.py` | MOD: `post_only` parameter on `post_order()` |
| `weather/bridge.py` | MOD: `execute_maker_order()` method |
| `weather/paper_bridge.py` | MOD: `execute_maker_order()` simulation |
| `weather/strategy.py` | MOD: Taker/maker decision logic |
| `weather/config.py` | MOD: 4 new maker config fields |
