# Bot Audit — Cleanup & Critical Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove 6 dead code elements from the bot/weather codebase, then fix 4 critical reliability and financial risk issues.

**Architecture:** Phase 1 removes dead code with careful test updates. Phase 2 adds threading lock, file lock, feedback persistence, and unified exposure. Each fix is small and independently testable.

**Tech Stack:** Python 3.14 stdlib (threading, json). No new dependencies.

---

### Task 1: Dead Code Cleanup — weather/state.py

Remove `analyzed_markets`, `PendingOrder`, and `pending_orders` from `TradingState`.

**Files:**
- Modify: `weather/state.py` — remove lines related to `analyzed_markets`, `PendingOrder`, `pending_orders`
- Modify: `weather/tests/test_state_extended.py` — remove tests for `pending_orders` and `analyzed_markets`
- Modify: `weather/tests/test_strategy.py` — remove assertions on `analyzed_markets`

**Step 1: Remove dead code from weather/state.py**

Remove these elements:
- `PendingOrder` dataclass (lines 16-48)
- `analyzed_markets: set[str]` field (line 136)
- `pending_orders: dict[str, PendingOrder]` field (line 147)
- `mark_analyzed()` method (lines 181-182)
- `was_analyzed()` method (lines 184-185)
- `analyzed_markets` in `to_dict()` — serialization line
- `pending_orders` in `to_dict()` (line 292)
- `analyzed_markets` in `from_dict()` — deserialization
- `pending_orders` in `from_dict()` (lines 330-333)
- `analyzed_markets` parameter in `cls(...)` constructor call
- `pending_orders` parameter in `cls(...)` constructor call (line 342)

**Step 2: Update tests**

In `weather/tests/test_state_extended.py`:
- Remove all test methods that only test `pending_orders` or `analyzed_markets`
- Update fixtures that include `"pending_orders"` or `"analyzed_markets"` keys — remove those keys
- If a test checks round-trip serialization, just remove the `pending_orders`/`analyzed_markets` assertions

In `weather/tests/test_strategy.py`:
- Remove line 362 or any assertion referencing `analyzed_markets`

**Step 3: Run tests**

Run: `python3 -m pytest weather/tests/test_state_extended.py weather/tests/test_strategy.py -v`
Expected: all PASS

**Step 4: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: all pass

**Step 5: Commit**

```bash
git commit -m "refactor: remove dead code from weather state (analyzed_markets, pending_orders, PendingOrder)"
```

---

### Task 2: Dead Code Cleanup — bridge, scoring, retry, config

Remove `get_positions()` stub, `edge_confidence()`, `bot/retry.py`, and `price_drop_threshold`.

**Files:**
- Modify: `weather/bridge.py` — remove `get_positions()` method (lines 140-145)
- Modify: `weather/strategy.py` — remove call to `client.get_positions()` in `positions_only` branch (line 485)
- Modify: `weather/tests/test_bridge.py` — remove test for `get_positions` (line 109)
- Modify: `weather/tests/test_strategy.py` — remove mock setup for `get_positions` (lines 144, 220)
- Modify: `bot/scoring.py` — remove `edge_confidence()` function (lines 71-89)
- Modify: `bot/tests/test_scoring.py` — remove tests for `edge_confidence` (lines around 86-100)
- Delete: `bot/retry.py`
- Modify: `bot/tests/test_daemon.py` — remove imports and tests for `with_retry` (lines 22-79, etc.)
- Modify: `weather/config.py` — remove `price_drop_threshold` field (line 80)

**Step 1: Remove get_positions() from bridge and strategy**

In `weather/bridge.py`: delete `get_positions()` method (lines 140-145).

In `weather/strategy.py`: find the `positions_only` branch that calls `client.get_positions()` (line 485). Replace with a comment or remove the branch if it's unused. If the `positions_only` parameter still has value (showing weather state trades), change it to read from `state.trades` directly instead.

In `weather/tests/test_bridge.py`: remove the test asserting `get_positions()` returns `[]` (line 109).

In `weather/tests/test_strategy.py`: remove `bridge.get_positions.return_value = ...` mock setup (lines 144, 220).

**Step 2: Remove edge_confidence() from scoring**

In `bot/scoring.py`: delete `edge_confidence()` (lines 71-89).

In `bot/tests/test_scoring.py`: delete tests for `edge_confidence` and remove its import.

**Step 3: Delete bot/retry.py**

Delete the file `bot/retry.py`.

In `bot/tests/test_daemon.py`: remove `from bot.retry import with_retry` and all test methods that only test `with_retry`. Keep other daemon tests intact.

**Step 4: Remove price_drop_threshold from config**

In `weather/config.py`: delete `price_drop_threshold: float = 0.10` (line 80).

**Step 5: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`
Expected: all pass

**Step 6: Commit**

```bash
git commit -m "refactor: remove dead code (get_positions stub, edge_confidence, retry module, price_drop_threshold)"
```

---

### Task 3: Fix — Thread-safe Circuit Breaker

Add `threading.Lock()` to `_CircuitBreaker` in `polymarket/client.py`.

**Files:**
- Modify: `polymarket/client.py` — lines 27-72
- Modify or create: `bot/tests/test_circuit_breaker.py` (or add to existing test file)

**Step 1: Write test for thread safety**

Create a test that exercises the circuit breaker from multiple threads simultaneously. The test should verify that concurrent `record_failure()` calls correctly increment `_failure_count` and transition to OPEN state without race conditions.

```python
import threading
from polymarket.client import _CircuitBreaker

class TestCircuitBreakerThreadSafety(unittest.TestCase):
    def test_concurrent_failures_open_circuit(self):
        cb = _CircuitBreaker(failure_threshold=10, recovery_timeout=60)
        threads = []
        for _ in range(20):
            t = threading.Thread(target=cb.record_failure)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # After 20 failures (threshold=10), should be OPEN
        self.assertEqual(cb.state, cb.OPEN)
        self.assertFalse(cb.allow_request())

    def test_concurrent_success_resets(self):
        cb = _CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        # Force to OPEN
        for _ in range(5):
            cb.record_failure()
        self.assertEqual(cb.state, cb.OPEN)
        # Simulate half-open probe success from multiple threads
        cb.state = cb.HALF_OPEN
        threads = [threading.Thread(target=cb.record_success) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(cb.state, cb.CLOSED)
        self.assertEqual(cb._failure_count, 0)
```

**Step 2: Implement the lock**

In `polymarket/client.py`, modify `_CircuitBreaker`:

Add `import threading` at the top of the file.

In `__init__()`: add `self._lock = threading.Lock()`

Wrap `record_success()`, `record_failure()`, and `allow_request()` bodies with `with self._lock:`.

```python
def record_success(self) -> None:
    with self._lock:
        if self.state == self.HALF_OPEN:
            logger.info("Circuit breaker → CLOSED (probe succeeded)")
        self._failure_count = 0
        self.state = self.CLOSED

def record_failure(self) -> None:
    with self._lock:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning("Circuit breaker → OPEN after %d failures", self._failure_count)

def allow_request(self) -> bool:
    with self._lock:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN (recovery probe)")
                return True
            return False
        return True  # HALF_OPEN
```

**Step 3: Run tests**

Run: `python3 -m pytest bot/tests/test_circuit_breaker.py -v`
Expected: PASS

**Step 4: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`

**Step 5: Commit**

```bash
git commit -m "fix: add threading lock to circuit breaker for thread safety"
```

---

### Task 4: Fix — Weather State Locking + Feedback Persistence

Add file locking around weather state in `_run_weather_pipeline()` and persist feedback state.

**Files:**
- Modify: `bot/strategy.py` — `_run_weather_pipeline()` (lines 106-160)
- Modify: `weather/strategy.py` — end of `run_weather_strategy()` (around line 959)
- Create or modify: tests to verify feedback.save() is called

**Step 1: Add file lock to weather pipeline**

In `bot/strategy.py`, in `_run_weather_pipeline()`:

Check if `weather.state` has a `state_lock` context manager. If it does, wrap the weather state load + run + save in it:

```python
from weather.state import state_lock  # if it exists

def _run_weather_pipeline(...):
    ...
    weather_state_path = ...
    with state_lock(weather_state_path):
        weather_state = WeatherState.load(weather_state_path)
        run_weather_strategy(...)
```

If `state_lock` doesn't exist in `weather/state.py`, create a simple file-lock context manager using `fcntl.flock` (Unix) or a `.lock` file approach.

**Step 2: Add feedback.save() to weather strategy**

In `weather/strategy.py`, at the end of `run_weather_strategy()`, just before the final `state.save(save_path)` (line 959), add:

```python
# Persist feedback state (EMA corrections)
try:
    feedback.save()
except Exception as exc:
    logger.warning("Failed to save feedback state: %s", exc)
```

**Step 3: Write test for feedback persistence**

In `weather/tests/test_strategy.py`, add a test verifying that `feedback.save()` is called during `run_weather_strategy()`:

```python
def test_feedback_state_saved(self):
    """run_weather_strategy should persist feedback state."""
    # ... setup mocks ...
    with patch.object(feedback, 'save') as mock_save:
        run_weather_strategy(...)
    mock_save.assert_called_once()
```

**Step 4: Run tests**

Run: `python3 -m pytest weather/tests/test_strategy.py bot/tests/ -q`

**Step 5: Commit**

```bash
git commit -m "fix: add weather state locking and feedback persistence"
```

---

### Task 5: Fix — Unified Exposure Tracking

Pass remaining exposure budget (total - bot exposure) to the weather bridge.

**Files:**
- Modify: `bot/strategy.py` — `_run_weather_pipeline()` (line 126)

**Step 1: Calculate bot exposure before weather pipeline**

In `bot/strategy.py`, in `_run_weather_pipeline()`, before creating the `CLOBWeatherBridge`, calculate the current bot exposure:

```python
# Calculate remaining exposure budget for weather
bot_exposure = sum(
    (1.0 - t.price) * t.size if getattr(t, 'side', 'BUY') == 'SELL' else t.price * t.size
    for t in state.trades.values()
    if getattr(t, 'memo', None) not in ('pending_exit', 'pending_fill')
)
remaining_exposure = max(0.0, config.max_total_exposure - bot_exposure)
```

**Step 2: Pass remaining exposure to bridge**

Change line 126 from:
```python
bridge = CLOBWeatherBridge(
    clob_client=client,
    gamma_client=gamma,
    max_exposure=config.max_total_exposure,
)
```
To:
```python
bridge = CLOBWeatherBridge(
    clob_client=client,
    gamma_client=gamma,
    max_exposure=remaining_exposure,
)
logger.info("Weather pipeline: bot_exposure=$%.2f, remaining=$%.2f",
            bot_exposure, remaining_exposure)
```

**Step 3: Write test**

In `bot/tests/test_strategy.py` (or create if needed), add a test that verifies the weather bridge receives reduced exposure when bot has positions:

```python
@patch("bot.strategy.CLOBWeatherBridge")
def test_weather_bridge_gets_remaining_exposure(self, mock_bridge_cls):
    state = TradingState()
    state.trades = {"pos1": TradeRecord(price=0.5, size=10.0, ...)}  # $5 exposure
    config = Config(max_total_exposure=50.0, ...)
    _run_weather_pipeline(client, config, state, dry_run=True, state_path="...")
    # Bridge should receive max_exposure = 50 - 5 = 45
    call_kwargs = mock_bridge_cls.call_args.kwargs
    self.assertAlmostEqual(call_kwargs["max_exposure"], 45.0, delta=1.0)
```

**Step 4: Run full suite**

Run: `python3 -m pytest weather/tests/ bot/tests/ -q`

**Step 5: Commit**

```bash
git commit -m "fix: pass remaining exposure budget to weather bridge (unified tracking)"
```
