# Async I/O + Multi-Source Forecast — Design

> Validated 2026-02-17. Next step: implementation plan via writing-plans skill.

## Goal

Migrate the entire bot to async I/O (`aiohttp`) and add new forecast sources (Open-Meteo multi-model, Weather Underground, Meteo-France) with dynamic model weighting.

## Architecture

Full async pipeline: shared `aiohttp.ClientSession` replaces all `urllib`/`httpx` calls. `asyncio.gather()` replaces `ThreadPoolExecutor` for parallel fetches. New sources added on the async foundation.

## 1. Shared Async HTTP Client

`weather/http_client.py` — singleton `aiohttp.ClientSession` with connection pooling, retry, and timeout.

```python
class HttpClient:
    _session: aiohttp.ClientSession | None = None

    @classmethod
    async def get(cls, url, params=None, headers=None, timeout=30) -> dict | str:
        ...

    @classmethod
    async def post(cls, url, body=None, headers=None) -> dict:
        ...

    @classmethod
    async def close(cls):
        ...
```

**Dependency added:** `aiohttp` (only new external dep).

## 2. New Forecast Sources

### A. Open-Meteo Multi-Model (free, same API)

Currently 2-3 models. Expanded to ~8 with a single API call (`&models=ecmwf,gfs,ukmo,...`):

| Model | Coverage | Resolution | Notes |
|-------|----------|------------|-------|
| `ecmwf_ifs025` | Global | 25km | Already used |
| `gfs_seamless` | Global | 25km | Already used |
| `ukmo_seamless` | Global | 10km | UK Met Office |
| `jma_seamless` | Global | 20km | Japan Met Agency |
| `arpege_seamless` | Europe | 10km | Meteo-France ARPEGE |
| `meteofrance_seamless` | Europe | 2.5km | Meteo-France AROME (high-res) |
| `gem_seamless` | Global | 15km | Environment Canada |
| `bom_access_global` | Global | 25km | Bureau of Meteorology Australia |

### B. Weather Underground

Polymarket's resolution source — direct edge.

- **API:** `https://api.weather.com/v3/wx/forecast/daily/5day`
- **Key:** Free tier via wunderground.com signup, stored in `creds.json` as `wu_api_key`
- **Rate limit:** 500 calls/day → **60 min cache** mandatory
- **Data:** 5-day forecast, daily high/low
- **Weight bonus:** +20% on computed weight (resolution source)

### C. Meteo-France

Already available via Open-Meteo (`meteofrance_seamless`, `arpege_seamless`). No separate API needed.

## 3. Async Pipeline

**Before (sync + threads):**
```
main() -> run_weather_strategy() [sync]
  └── ThreadPoolExecutor
       ├── thread: NOAA
       ├── thread: Open-Meteo
       └── thread: Aviation
```

**After (full async):**
```
main() -> asyncio.run(run_weather_strategy()) [async]
  └── asyncio.gather(
       fetch_noaa(...),
       fetch_open_meteo_multi(...),
       fetch_aviation(...),
       fetch_wu(...),
     )
  └── asyncio.gather(
       clob.get_orderbook(token1),
       clob.get_orderbook(token2),
       ...
     )
```

All functions become async: `run_weather_strategy`, `check_exit_opportunities`, `execute_trade`, `execute_sell`, `execute_maker_order`, `poll_once`.

Entry points use `asyncio.run()`.

## 4. Dynamic Model Weighting

Replace fixed weights with performance-based weights:

```
weight(model) = 1 / EMA_error(model)^2
Normalized so sum = 1.0
```

- Extend `weather/feedback.py` to track error **per model** (`model_errors` dict)
- After each market resolution, compare each model's forecast vs actual
- Recalculate weights each run
- WU gets +20% bonus on computed weight (resolution source)
- **Fallback:** Use fixed weights until 10+ resolutions accumulated

## 5. Error Handling & Rate Limits

| Source | Rate Limit | Cache TTL | Fallback |
|--------|-----------|-----------|----------|
| Open-Meteo (multi-model) | 10k/day | 15 min (existing) | Keep last cache |
| NOAA | ~50 req/min | 15 min (new) | Skip, other models suffice |
| Aviation METAR | No limit | None | Skip, weight 0 |
| Weather Underground | 500/day | 60 min (mandatory) | Skip, redistribute weight |
| Ensemble API | No limit | 6h (existing) | Keep last cache |

```python
results = await asyncio.gather(
    fetch_noaa(...),
    fetch_open_meteo_multi(...),
    fetch_aviation(...),
    fetch_wu(...),
    return_exceptions=True,  # no crash if one source fails
)
```

Bot must function with **1 source available**. If only ECMWF works, trade with wider sigma.

## 6. Files Changed

### Created
| File | Role |
|------|------|
| `weather/http_client.py` | Shared async HTTP client (aiohttp) |
| `weather/wu.py` | Weather Underground client (5-day forecast, 60 min cache) |

### Modified
| File | Change |
|------|--------|
| `weather/open_meteo.py` | async + 6 new models |
| `weather/noaa.py` | async + 15 min cache |
| `weather/aviation.py` | async |
| `weather/ensemble.py` | async |
| `weather/strategy.py` | async, asyncio.gather(), dynamic weights |
| `weather/bridge.py` | async execute_trade/sell/maker_order |
| `weather/paper_bridge.py` | async methods |
| `polymarket/client.py` | httpx -> aiohttp, all methods async |
| `bot/gamma.py` | httpx -> aiohttp, all methods async |
| `weather/__main__.py` | asyncio.run() |
| `weather/paper_trade.py` | asyncio.run() |
| `weather/paper_loop.py` | asyncio.run() |
| `weather/order_manager.py` | async poll_once, asyncio.sleep() |
| `weather/feedback.py` | model_errors field for dynamic weighting |
| `weather/config.py` | wu_cache_ttl, noaa_cache_ttl, dynamic_weights |

### Unchanged
- Scoring, sizing, exits, maker/taker logic — identical
- State file formats — identical
- 981 existing tests — adapted with @pytest.mark.asyncio, same coverage
