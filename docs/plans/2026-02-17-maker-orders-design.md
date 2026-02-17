# Maker Orders Strategy — Design

> Validated 2026-02-17. Next step: implementation plan via writing-plans skill.

## Goal

Eliminate slippage on Polymarket weather markets by switching from pure taker execution (buy at ask) to a hybrid taker/maker system with a continuous order management daemon.

## Architecture

Two independent processes communicating via shared file state:

```
┌─────────────────────┐          ┌─────────────────────┐
│   Strategy Loop     │          │   Order Manager      │
│   (every 5 min)     │          │   (poll every 10s)   │
│                     │          │                      │
│  1. Score buckets   │  writes  │  1. Load pending     │
│  2. Decide taker    │────────→ │  2. Check fills      │
│     or maker        │          │  3. Cancel expired   │
│  3. Taker → fill    │ pending  │  4. Record trades    │
│     immediate       │ _orders  │  5. Reconcile        │
│  4. Maker → post    │  .json   │                      │
│     bid, add to     │←─────────│                      │
│     pending state   │ updates  │                      │
└─────────────────────┘          └─────────────────────┘
```

**Shared state**: `pending_orders.json` with `fcntl.flock` file locking. Each entry:

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

## Taker vs Maker Decision

```
Edge = prob - price
         │
    ┌────┴────┐
    │ edge >  │
    │  5% AND │── YES ──→ TAKER (immediate fill, record_trade())
    │spread < │
    │  10%    │
    └────┬────┘
         │ NO
         ▼
       MAKER
       post bid at min(prob, best_bid + tick_size)
       add to pending_orders.json
```

**Maker bid pricing**: `maker_price = min(prob, best_bid + tick_size)`
- 1 tick above best bid for queue priority
- Never above our probability (preserves edge)
- If `best_bid + tick >= best_ask` → no room for maker, fall back to taker

**`postOnly: True`** on all maker orders — CLOB rejects if the order would cross the spread, preventing accidental taker fills.

## Config Parameters

```python
maker_edge_threshold: float = 0.05    # below → maker
maker_spread_threshold: float = 0.10  # above → maker
maker_ttl_seconds: int = 900          # 15 min before cancel
maker_tick_buffer: int = 1            # ticks above best bid
```

## Order Manager Daemon

Continuous process polling every 10 seconds:

```python
while True:
    pending = load_pending_orders()  # file lock

    for order in pending:
        # 1. TTL expired?
        if now - order.submitted_at > order.ttl_seconds:
            clob.cancel_order(order.order_id)
            remove_from_pending(order)
            continue

        # 2. Check fill status
        status = clob.get_order(order.order_id)

        if status == "MATCHED":
            state.record_trade(...)
            remove_from_pending(order)
        elif status == "CANCELLED":
            remove_from_pending(order)
        elif size_matched > 0:
            update_pending(order, remaining=original - matched)

    save_pending_orders()  # file lock
    sleep(10)
```

**Startup reconciliation**: on launch, the manager reads `pending_orders.json` and reconciles with `clob.get_open_orders()`. Stale entries are cleaned, missed fills are recovered.

**Launch**:
```bash
# Terminal 1: strategy loop
python3 -m weather.paper_loop

# Terminal 2: order manager
python3 -m weather.order_manager
```

## Code Changes

### `polymarket/client.py`
- Add `post_only: bool = False` parameter to `post_order()`, pass as `"postOnly"` in JSON body.

### `weather/bridge.py`
- New `execute_maker_order(market_id, side, amount, maker_price)` → posts GTC postOnly bid, returns `{"order_id", "price", "size", "posted"}`. Does NOT wait for fill.

### `weather/strategy.py`
- In the trade loop: check edge/spread → taker or maker decision.
- Maker path: call `execute_maker_order()`, add to pending. No `record_trade()`.
- Guard: skip if `market_id` already in `state.trades` OR `pending_orders`.
- Budget: `effective_exposure = state_exposure + pending_exposure`.

### `weather/paper_bridge.py`
- New `execute_maker_order()` simulation: if `maker_price >= best_bid` → immediate fill, else add to pending for manager to simulate.

### `weather/order_manager.py` (new)
- Daemon process: poll loop, fill detection, TTL cancel, crash reconciliation.

### `weather/pending_state.py` (new)
- `PendingOrders` class: load/save `pending_orders.json` with file lock, add/remove/query.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Strategy crash with pending orders | Manager continues managing them. On restart, strategy checks pending before posting duplicates |
| Manager crash | On restart, reconciles pending vs `get_open_orders()`. Recovers missed fills |
| Both crash | Manager startup reconciliation from `get_open_orders()` is source of truth |
| CLOB rejects postOnly (would cross) | `execute_maker_order()` returns `{"posted": False}`, strategy skips |
| Partial fill at TTL | Manager cancels remainder, records partial position |
| Market resolved with open order | CLOB auto-cancels. Manager detects CANCELLED, cleans up |
| Double position (taker + maker same market) | Strategy checks `state.trades` AND `pending_orders` before posting |
| Capital locked by pending makers | Pending total counted as exposure: `available = max_exposure - state_exposure - pending_exposure` |

## Testing

### Unit tests
- Taker/maker decision logic (edge/spread thresholds)
- Maker price calculation (`min(prob, best_bid + tick)`)
- No maker when no room (bid + tick >= ask)
- `postOnly` flag propagation
- Pending exposure counted in budget
- No duplicate market guard

### Order manager tests
- Fill detection → record_trade + remove pending
- TTL expiry → cancel + remove
- Partial fill → record partial, cancel remainder
- Crash reconciliation (pending vs get_open_orders)
- Market cancellation → cleanup
- File lock contention (strategy + manager concurrent access)

### Integration tests
- Full maker cycle: post → manager detects fill → position recorded
- Maker expire: post → TTL → cancel → capital freed
