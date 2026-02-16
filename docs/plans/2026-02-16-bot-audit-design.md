# Bot Audit — Cleanup & Critical Fixes Design

## Goal

Clean up dead code in the bot and weather integration layer, then fix the 4 most urgent reliability/financial risk issues identified by the audit.

## Phase 1: Dead Code Cleanup

Remove 6 dead/vestigial elements:

### 1. `weather/state.py` — `analyzed_markets`
- Remove `analyzed_markets: set` attribute from `TradingState`
- Remove `mark_analyzed()` and `was_analyzed()` methods
- Remove serialization/deserialization of `analyzed_markets` in `to_dict()`/`from_dict()`
- Remove any tests that only test these methods

### 2. `weather/state.py` — `pending_orders`
- Remove `PendingOrder` dataclass
- Remove `pending_orders: dict` attribute from `TradingState`
- Remove serialization/deserialization in `to_dict()`/`from_dict()`
- Remove any tests that only test `PendingOrder`

### 3. `weather/bridge.py` — `get_positions()` always returns `[]`
- Remove the method entirely (it's misleading — positions are tracked in TradingState)
- Check if anything calls it and remove those call sites

### 4. `bot/scoring.py` — `edge_confidence()`
- Remove the function
- Remove any tests that only test it

### 5. `bot/retry.py` — entire module
- Remove `bot/retry.py`
- Remove `bot/tests/test_retry.py` (if it exists and only tests retry)
- Remove any imports of `with_retry` from other modules

### 6. `weather/config.py` — `price_drop_threshold`
- Remove the orphaned config field from the `Config` dataclass

## Phase 2: Critical Fixes

### Fix 1: Thread-safe circuit breaker

**File:** `polymarket/client.py`

**Problem:** `_CircuitBreaker._failure_count` and `state` are mutated from multiple threads (ThreadPoolExecutor) without synchronization.

**Fix:** Add `threading.Lock()` to `_CircuitBreaker.__init__()`. Wrap `record_success()`, `record_failure()`, and `allow_request()` with `self._lock`.

```python
def __init__(self, ...):
    ...
    self._lock = threading.Lock()

def record_success(self):
    with self._lock:
        self._failure_count = 0
        self.state = self.CLOSED

def record_failure(self):
    with self._lock:
        self._failure_count += 1
        if self._failure_count >= self._threshold:
            self.state = self.OPEN
            self._opened_at = time.monotonic()
```

### Fix 2: Weather state file locking

**File:** `bot/strategy.py` — `_run_weather_pipeline()`

**Problem:** Weather state loaded/saved without acquiring file lock. Concurrent processes can corrupt the file.

**Fix:** Import `state_lock` from `weather.state` and wrap the pipeline call:

```python
from weather.state import state_lock

def _run_weather_pipeline(...):
    with state_lock(weather_state_path):
        weather_state = WeatherState.load(weather_state_path)
        run_weather_strategy(...)
```

### Fix 3: Persist feedback state

**File:** `weather/strategy.py`

**Problem:** `FeedbackState.load()` is called but `feedback.save()` is never called. EMA corrections lost on restart.

**Fix:** Add `feedback.save()` at the end of `run_weather_strategy()`, after all processing is complete but before the function returns.

### Fix 4: Unified exposure tracking

**File:** `bot/strategy.py` — `_run_weather_pipeline()`

**Problem:** Weather bridge receives `max_exposure` without subtracting the bot's current exposure. Both pipelines can independently max out exposure.

**Fix:** Calculate bot exposure before launching weather pipeline, pass remaining budget:

```python
bot_exposure = sum(
    t.price * t.size for t in state.trades.values()
    if getattr(t, 'memo', None) not in ('pending_exit', 'pending_fill')
)
remaining = max(0, config.max_total_exposure - bot_exposure)
bridge = CLOBWeatherBridge(
    clob_client=client,
    max_exposure=remaining,
)
```

## Success Criteria

1. All dead code removed, no remaining references
2. Full test suite passes after cleanup
3. Circuit breaker mutations wrapped in lock
4. Weather state loaded/saved under file lock
5. Feedback state persisted after each strategy run
6. Weather bridge exposure = total_max - bot_current_exposure
7. No new test failures
