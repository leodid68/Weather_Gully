# Async I/O + Multi-Source Forecast — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Migrate the bot to full async I/O (aiohttp/httpx.AsyncClient), add Open-Meteo multi-model + Weather Underground sources, and implement dynamic model weighting.

**Architecture:** Bottom-up migration: shared async HTTP client → weather modules → CLOB/Gamma clients → bridge → strategy → entry points. New sources (WU, multi-model) added during migration. Tests adapted last.

**Tech Stack:** Python 3.14, aiohttp, httpx (AsyncClient), pytest-asyncio.

---

## Task 1: Shared async HTTP client (`weather/http_client.py`)

**Files:**
- Create: `weather/http_client.py`
- Create: `weather/tests/test_http_client.py`

### What to build

A shared async HTTP client built on `aiohttp` with SSL (certifi), retry with exponential backoff, and connection pooling. This replaces all `urllib.request.urlopen()` calls in weather modules.

### Implementation

```python
"""Shared async HTTP client — replaces urllib across weather modules."""

from __future__ import annotations

import json
import logging
import random
import ssl
from typing import Any

import aiohttp
import certifi

logger = logging.getLogger(__name__)

_session: aiohttp.ClientSession | None = None


def _make_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


_SSL_CTX = _make_ssl_context()


async def get_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session."""
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(ssl=_SSL_CTX, limit=20)
        _session = aiohttp.ClientSession(connector=connector)
    return _session


async def close_session() -> None:
    """Close the shared session (call at shutdown)."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def fetch_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict | list | None:
    """Async GET that returns parsed JSON, with retry and backoff.

    Returns None on failure (never raises).
    """
    session = await get_session()
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    logger.debug("HTTP %d from %s — retry in %.1fs", resp.status, url[:80], delay)
                    import asyncio
                    await asyncio.sleep(delay)
                    continue
                logger.warning("HTTP %d from %s", resp.status, url[:80])
                return None
        except (aiohttp.ClientError, TimeoutError) as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.debug("Fetch error %s — retry in %.1fs: %s", url[:80], delay, exc)
                import asyncio
                await asyncio.sleep(delay)
            else:
                logger.warning("Fetch failed after %d attempts: %s — %s", max_retries, url[:80], exc)
                return None
    return None


async def post_json(
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
) -> dict | None:
    """Async POST that returns parsed JSON. Returns None on failure."""
    session = await get_session()
    try:
        async with session.post(
            url, json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status in (200, 201):
                return await resp.json(content_type=None)
            logger.warning("POST %d from %s", resp.status, url[:80])
            return None
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning("POST failed: %s — %s", url[:80], exc)
        return None
```

### Tests

Test with `aiohttp` test utilities or mock:
- `test_fetch_json_success`: Mock server returns 200 → parsed JSON
- `test_fetch_json_retry_on_500`: Mock returns 500, then 200 → retries and succeeds
- `test_fetch_json_timeout`: Mock times out → returns None
- `test_session_singleton`: Two calls to `get_session()` return same object
- `test_close_session`: After close, next call creates new session

### Verification
```bash
pip install aiohttp
python3 -m pytest weather/tests/test_http_client.py -v
```

---

## Task 2: Weather Underground client (`weather/wu.py`)

**Files:**
- Create: `weather/wu.py`
- Create: `weather/tests/test_wu.py`

### What to build

Async Weather Underground forecast client with 60-min disk cache. Uses the TWC (The Weather Company) API v3.

### Implementation

```python
"""Weather Underground forecast client — async with disk cache."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from .http_client import fetch_json

logger = logging.getLogger(__name__)

WU_API_BASE = "https://api.weather.com/v3/wx/forecast/daily/5day"
_CACHE_DIR = Path(__file__).parent / "cache" / "wu"
_DEFAULT_TTL = 3600  # 60 min


def _cache_path(cache_dir: Path, lat: float, lon: float) -> Path:
    return cache_dir / f"{lat:.2f}_{lon:.2f}.json"


def _read_cache(path: Path, ttl: int) -> dict | None:
    try:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def get_wu_forecast(
    lat: float,
    lon: float,
    api_key: str,
    cache_dir: Path | None = None,
    cache_ttl: int = _DEFAULT_TTL,
) -> dict[str, dict] | None:
    """Fetch 5-day WU forecast. Returns {date_str: {"high": F, "low": F}} or None.

    Uses disk cache to stay within 500 calls/day free tier.
    """
    cache_dir = cache_dir or _CACHE_DIR
    cp = _cache_path(cache_dir, lat, lon)
    cached = _read_cache(cp, cache_ttl)
    if cached is not None:
        logger.debug("WU cache hit for %.2f,%.2f", lat, lon)
        return cached

    url = WU_API_BASE
    params = {
        "geocode": f"{lat:.4f},{lon:.4f}",
        "format": "json",
        "units": "e",  # imperial (°F)
        "language": "en-US",
        "apiKey": api_key,
    }

    data = await fetch_json(url, params=params, timeout=15, max_retries=2)
    if not data:
        logger.warning("WU forecast failed for %.2f,%.2f", lat, lon)
        return None

    # Parse TWC response
    try:
        days = data.get("dayOfWeek", [])
        dates = data.get("validTimeLocal", [])
        highs = data.get("temperatureMax", [])
        lows = data.get("temperatureMin", [])

        result: dict[str, dict] = {}
        for i, date_str in enumerate(dates):
            if not date_str:
                continue
            day = date_str[:10]  # "2026-02-18T07:00:00-0500" → "2026-02-18"
            high = highs[i] if i < len(highs) and highs[i] is not None else None
            low = lows[i] if i < len(lows) and lows[i] is not None else None
            if high is not None or low is not None:
                result[day] = {"high": high, "low": low}

        if result:
            _write_cache(cp, result)
            logger.info("WU forecast: %d days for %.2f,%.2f", len(result), lat, lon)

        return result or None

    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("WU parse error: %s", exc)
        return None
```

### Tests

- `test_wu_forecast_parse`: Mock fetch_json returns sample TWC response → correct parsing
- `test_wu_cache_hit`: Cache exists and fresh → no HTTP call
- `test_wu_cache_miss`: Cache expired → HTTP call made
- `test_wu_api_failure`: fetch_json returns None → returns None gracefully
- `test_wu_no_api_key`: Empty key → returns None

### Verification
```bash
python3 -m pytest weather/tests/test_wu.py -v
```

---

## Task 3: Config fields for new sources

**Files:**
- Modify: `weather/config.py` — Add fields
- Modify: `weather/config.json` — Add defaults

### What to build

New config fields:

```python
# Weather Underground
wu_api_key: str = ""           # From creds.json or env
wu_cache_ttl: int = 3600       # 60 min cache

# NOAA caching
noaa_cache_ttl: int = 900      # 15 min cache

# Dynamic model weighting
dynamic_weights: bool = True   # Use performance-based weights instead of fixed
wu_weight_bonus: float = 0.20  # +20% weight bonus for WU (resolution source)
```

Add after the maker order fields in Config dataclass. Add to `config.json`.

### Verification
```bash
python3 -m pytest weather/tests/test_config.py -v
```

---

## Task 4: Migrate `weather/open_meteo.py` to async + multi-models

**Files:**
- Modify: `weather/open_meteo.py`
- Modify: `weather/tests/test_open_meteo.py`

### What to build

1. Replace `_fetch_json(url)` (urllib) with `await fetch_json(url)` from `http_client`
2. Make `get_open_meteo_forecast()` and `get_open_meteo_forecast_multi()` async
3. Add new models to the API call: `ukmo_seamless`, `jma_seamless`, `arpege_seamless`, `gem_seamless`, `bom_access_global`
4. `compute_ensemble_forecast()` stays sync (pure computation, no I/O)

**Key changes:**
- `async def get_open_meteo_forecast(...)` — `await fetch_json(url)` instead of `_fetch_json(url)`
- `async def get_open_meteo_forecast_multi(...)` — same pattern
- Remove `_fetch_json()` function (replaced by `http_client.fetch_json`)
- Remove `import urllib` and `from ._ssl import SSL_CTX`
- Add models to URL params: `&models=ecmwf_ifs025,gfs_seamless,ukmo_seamless,jma_seamless,arpege_seamless,gem_seamless,bom_access_global`
- Parse new model data alongside existing GFS/ECMWF

**Tests:** Add `@pytest.mark.asyncio` to all async tests, mock `http_client.fetch_json`.

### Verification
```bash
python3 -m pytest weather/tests/test_open_meteo.py -v
```

---

## Task 5: Migrate `weather/noaa.py` to async + add cache

**Files:**
- Modify: `weather/noaa.py`
- Modify: `weather/tests/test_noaa.py`

### What to build

1. Replace urllib with `await fetch_json()` from `http_client`
2. Make `get_noaa_forecast()` async
3. Add 15-min disk cache (same pattern as ensemble.py)

**Key changes:**
- `async def get_noaa_forecast(...)` — two `await fetch_json()` calls (points → forecast)
- Add `_CACHE_DIR = Path(__file__).parent / "cache" / "noaa"` with 15 min TTL
- Remove `_fetch_json()`, remove urllib imports

### Verification
```bash
python3 -m pytest weather/tests/test_noaa.py -v
```

---

## Task 6: Migrate `weather/aviation.py` and `weather/ensemble.py` to async

**Files:**
- Modify: `weather/aviation.py`
- Modify: `weather/ensemble.py`
- Modify: `weather/tests/test_aviation.py`
- Modify: `weather/tests/test_ensemble.py` (if exists)

### What to build

1. Aviation: Replace urllib with `await fetch_json()`, make `get_metar_observations()` and `get_aviation_daily_data()` async
2. Ensemble: Replace urllib with `await fetch_json()`, make `fetch_ensemble_spread()` async

Both are straightforward — same pattern as Tasks 4-5.

### Verification
```bash
python3 -m pytest weather/tests/test_aviation.py weather/tests/test_ensemble.py -v
```

---

## Task 7: Migrate `polymarket/client.py` to async

**Files:**
- Modify: `polymarket/client.py`
- Modify: `polymarket/tests/` or `bot/tests/` (wherever CLOB tests are)

### What to build

Migrate from `httpx.Client` (sync) to `httpx.AsyncClient`. All methods become async.

**Key changes:**
- `self._http = httpx.AsyncClient(base_url=base_url, timeout=15)` (line ~114)
- `async def _request(...)` — `await self._http.request(...)` instead of sync
- All callers become async: `async def post_order(...)`, `async def cancel_order(...)`, `async def get_orderbook(...)`, `async def get_order(...)`, `async def get_open_orders(...)`, etc.
- `async def close()` — `await self._http.aclose()`
- Keep circuit breaker logic unchanged

Also migrate `PublicClient` (read-only, used in dry-run mode) to async.

### Verification
```bash
python3 -m pytest bot/tests/ -v
```

---

## Task 8: Migrate `bot/gamma.py` to async

**Files:**
- Modify: `bot/gamma.py`
- Modify: `bot/tests/` (gamma tests)

### What to build

Same pattern as Task 7: `httpx.Client` → `httpx.AsyncClient`.

**Key changes:**
- `self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)`
- `async def fetch_markets(...)`, `async def fetch_events(...)`, `async def check_resolution(...)`
- `async def close()` — `await self._http.aclose()`

### Verification
```bash
python3 -m pytest bot/tests/ -v
```

---

## Task 9: Migrate `weather/bridge.py` and `weather/paper_bridge.py` to async

**Files:**
- Modify: `weather/bridge.py`
- Modify: `weather/paper_bridge.py`
- Modify: `weather/tests/test_bridge.py`, `weather/tests/test_maker_bridge.py`

### What to build

All bridge methods become async since they call async CLOB/Gamma methods.

**Key changes in bridge.py:**
- `async def execute_trade(...)` — `await self.clob.post_order(...)`, `await self.clob.get_orderbook(...)`
- `async def execute_sell(...)` — same pattern
- `async def execute_maker_order(...)` — same pattern
- `async def verify_fill(...)` — `await asyncio.sleep(poll_interval)` instead of `time.sleep()`
- `async def cancel_order(...)` — `await self.clob.cancel_order(...)`
- `async def fetch_weather_markets(...)` — `await self.gamma.fetch_events(...)`
- `async def get_portfolio(...)` — `await self.clob.get_positions(...)`

**Key changes in paper_bridge.py:**
- All methods get `async def` prefix but remain simulation (no real I/O)
- `execute_trade`, `execute_sell`, `execute_maker_order` become async

### Verification
```bash
python3 -m pytest weather/tests/test_bridge.py weather/tests/test_maker_bridge.py -v
```

---

## Task 10: Migrate `weather/strategy.py` to async + WU + dynamic weights

**Files:**
- Modify: `weather/strategy.py`
- Modify: `weather/feedback.py` — Add `model_errors` tracking
- Modify: `weather/tests/test_strategy.py`

### What to build

The biggest task — migrates the main loop to async.

**Key changes:**

1. `async def run_weather_strategy(...)` — the main function becomes async

2. Replace `ThreadPoolExecutor` parallel fetches with `asyncio.gather()`:
```python
# Before (lines 846-851):
with ThreadPoolExecutor(max_workers=2) as pool:
    fut_portfolio = pool.submit(client.get_portfolio)
    fut_markets = pool.submit(client.fetch_weather_markets)
    portfolio = fut_portfolio.result()
    markets = fut_markets.result()

# After:
portfolio, markets = await asyncio.gather(
    client.get_portfolio(),
    client.fetch_weather_markets(),
)
```

3. Replace forecast ThreadPoolExecutor (lines 912-953):
```python
# After:
noaa_results = await asyncio.gather(
    *[get_noaa_forecast(loc, LOCATIONS) for loc in active_locs],
    return_exceptions=True,
)
om_result, aviation_result, wu_result = await asyncio.gather(
    get_open_meteo_forecast_multi(loc_subset),
    get_aviation_daily_data(active_locs),
    _fetch_wu_forecasts(active_locs, config),  # NEW
    return_exceptions=True,
)
```

4. Replace exit orderbook ThreadPoolExecutor with `asyncio.gather()`:
```python
# In check_exit_opportunities:
orderbooks_list = await asyncio.gather(
    *[client.clob.get_orderbook(tid) for _, tid in orderbook_targets],
    return_exceptions=True,
)
```

5. Add WU integration in the forecast combining section
6. Add dynamic weight computation in `compute_ensemble_forecast()` or alongside it

7. `async def check_exit_opportunities(...)` — becomes async
8. `async def _check_stop_loss_reversals(...)` — becomes async
9. `async def _emergency_exit_losers(...)` — becomes async

**Dynamic weights in feedback.py:**
- Add `model_errors: dict[str, dict]` field to `FeedbackState` — tracks EMA error per model per location
- Add `record_model_error(location, model, error)` method
- Add `get_model_weights(location) -> dict[str, float]` method — returns `1/EMA²` normalized weights

### Verification
```bash
python3 -m pytest weather/tests/test_strategy.py -v
```

---

## Task 11: Migrate entry points and order manager

**Files:**
- Modify: `weather/__main__.py`
- Modify: `weather/paper_trade.py`
- Modify: `weather/paper_loop.py`
- Modify: `weather/order_manager.py`

### What to build

Wrap everything in `asyncio.run()`.

**__main__.py:**
```python
async def async_main():
    bridge = _build_bridge(config, live=args.live)
    try:
        ...
        await run_weather_strategy(client=bridge, ...)
    finally:
        await close_session()
        bridge.close()

def main():
    ...
    asyncio.run(async_main())
```

**paper_trade.py:** Same pattern — `asyncio.run()` wrapper.

**paper_loop.py:** Subprocess-based so no change to the loop itself, but `paper_trade.py` (which it calls via subprocess) will be async internally.

**order_manager.py:**
```python
async def poll_once_async(...):
    ...
    status_resp = await clob.get_order(order_id)
    ...
    await clob.cancel_order(order_id)

async def run_manager_async(...):
    while True:
        await poll_once_async(...)
        await asyncio.sleep(poll_interval)

if __name__ == "__main__":
    asyncio.run(run_manager_async(...))
```

### Verification
```bash
python3 -m weather --dry-run --set locations=NYC  # Verify CLI works
```

---

## Task 12: Fix all tests for async

**Files:**
- Modify: All test files in `weather/tests/` and `bot/tests/`

### What to build

Adapt all tests that call async functions:

1. Add `@pytest.mark.asyncio` decorator to tests calling async functions
2. Add `async def` to test functions that use `await`
3. For mocks, use `AsyncMock` instead of `MagicMock` where needed:
```python
from unittest.mock import AsyncMock
mock_clob = AsyncMock()
mock_clob.get_orderbook.return_value = {"bids": [...]}
```
4. Ensure `pytest-asyncio` is installed (already a dev dep)

**Tests that need adaptation:**
- `test_strategy.py` — all tests calling `run_weather_strategy`
- `test_bridge.py`, `test_maker_bridge.py` — bridge method tests
- `test_open_meteo.py`, `test_noaa.py`, `test_aviation.py` — forecast tests
- `test_order_manager.py` — poll_once tests
- `bot/tests/` — CLOB and Gamma client tests

Tests that DON'T change:
- `test_probability.py`, `test_sizing.py`, `test_feedback.py` — pure computation, no I/O
- `test_pending_state.py` — file I/O only, not async
- `test_maker_decision.py` — pure logic tests

### Verification
```bash
python3 -m pytest weather/tests/ bot/tests/ -q  # Full suite must pass
```

---

## Final Verification

After all 12 tasks:
```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
python3 -m weather --dry-run --set locations=NYC  # Verify async CLI
python3 -m weather.paper_trade --set locations=NYC  # Verify paper trading
```

## Critical Files Reference

| File | Role |
|------|------|
| `weather/http_client.py` | NEW: Shared aiohttp client |
| `weather/wu.py` | NEW: Weather Underground client |
| `weather/open_meteo.py` | MOD: async + 6 new models |
| `weather/noaa.py` | MOD: async + 15 min cache |
| `weather/aviation.py` | MOD: async |
| `weather/ensemble.py` | MOD: async |
| `polymarket/client.py` | MOD: httpx.AsyncClient |
| `bot/gamma.py` | MOD: httpx.AsyncClient |
| `weather/bridge.py` | MOD: all methods async |
| `weather/paper_bridge.py` | MOD: all methods async |
| `weather/strategy.py` | MOD: asyncio.gather + WU + dynamic weights |
| `weather/feedback.py` | MOD: model_errors tracking |
| `weather/order_manager.py` | MOD: async poll loop |
| `weather/__main__.py` | MOD: asyncio.run() |
| `weather/paper_trade.py` | MOD: asyncio.run() |
