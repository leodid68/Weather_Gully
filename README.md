# Weather Gully

Automated weather trading bot for [Polymarket](https://polymarket.com) — trades temperature prediction markets using a multi-source ensemble forecast (NOAA + Open-Meteo + METAR observations) via CLOB direct (no SDK dependency).

Polymarket lists daily temperature markets for 6 US cities: *"Will the highest temperature in NYC on March 15 be 55°F or higher?"* with buckets like `[50-54]`, `[55-59]`, etc. This bot estimates the true probability for each bucket using weather forecasts and real-time airport observations, compares it to the market price, and trades when it finds a significant edge.

---

## Table of Contents

- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Weather Data Sources](#weather-data-sources)
- [Probability Model](#probability-model)
- [Trading Strategy](#trading-strategy)
- [Supported Locations](#supported-locations)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key Algorithms](#key-algorithms)

---

## Architecture

```
Weather_Gully/
├── polymarket/      # CLOB client (EIP-712 signing, HMAC L2 auth, WebSocket)
├── bot/             # General trading bot (signals, Kelly sizing, multi-choice arb, daemon)
└── weather/         # Weather strategy (ensemble forecasts, METAR obs, bucket scoring)
```

The project is organized into three independent Python packages that compose into a single trading pipeline:

### `polymarket/` — Low-Level CLOB Client

Direct implementation of the Polymarket CLOB API protocol. Zero SDK dependencies.

| Module | Description |
|--------|-------------|
| `client.py` | Synchronous REST client with connection pooling (httpx) and retry |
| `order.py` | EIP-712 order construction and signing (CTF Exchange struct) |
| `auth.py` | L1 API key derivation (EIP-712 typed signature) + L2 HMAC request signing |
| `ws.py` | Async WebSocket client with auto-reconnect and exponential backoff |
| `approve.py` | Token approval transactions for CTF Exchange |
| `constants.py` | Chain ID, contract addresses, API endpoints |

**Key implementation details:**
- Orders are signed using EIP-712 typed data (`Order` struct with maker, taker, tokenId, makerAmount, takerAmount, etc.)
- L2 authentication uses HMAC-SHA256 over `{timestamp}{method}{path}{body}` with a derived API secret
- Compact JSON serialization (`separators=(",",":")`) is critical for HMAC validation matching the server
- Weather markets use `neg_risk=True` (NEG_RISK_CTF_EXCHANGE contract)

### `bot/` — General-Purpose Trading Bot

Market-agnostic trading infrastructure with 4 signal detection methods.

| Module | Description |
|--------|-------------|
| `gamma.py` | Gamma API client — market discovery, metadata, multi-choice detection (`negRisk` grouping) |
| `scanner.py` | Market scanning with Gamma (primary) and CLOB fallback, liquidity grading (A/B/C) |
| `signals.py` | 4 edge-detection methods: longshot bias, arbitrage, microstructure, multi-choice arb |
| `scoring.py` | Calibration scoring — tracks prediction accuracy over time |
| `sizing.py` | Kelly criterion position sizing with risk limits |
| `strategy.py` | Main strategy loop — scans markets, detects signals, sizes and executes trades |
| `daemon.py` | Continuous operation mode with SIGTERM/SIGINT graceful shutdown, health check file |
| `config.py` | Bot-specific configuration (separate from weather config) |
| `state.py` | Persistent state tracking (positions, P&L, trade history) |

**Signal detection methods:**
1. **Longshot bias** — Empirical mispricing at price extremes (based on Becker 2024 analysis of 72.1M Kalshi trades)
2. **Arbitrage** — YES + NO orderbook violations of the $1.00 invariant
3. **Microstructure** — Order-book imbalance and spread analysis
4. **Multi-choice arbitrage** — Sum of YES prices across grouped `negRisk` markets != $1.00

### `weather/` — Weather-Specific Strategy

The core intelligence layer. Combines multiple weather data sources into a probability estimate for each temperature bucket, then trades accordingly.

| Module | Description |
|--------|-------------|
| `strategy.py` | Main strategy loop — fetches data, scores buckets, manages entries/exits/stop-losses |
| `bridge.py` | `CLOBWeatherBridge` adapter — connects weather strategy to CLOB + Gamma APIs |
| `noaa.py` | NOAA Weather API client (api.weather.gov) — official US forecast |
| `open_meteo.py` | Open-Meteo client — GFS + ECMWF multi-model forecasts (free, no API key) |
| `aviation.py` | Aviation Weather API client — METAR real-time observations from airport stations |
| `probability.py` | Normal CDF bucket probability, horizon-dependent sigma, observation-adjusted estimates |
| `parsing.py` | Event name parser — extracts location, date, metric (high/low) from market titles |
| `sizing.py` | Kelly criterion sizing with weather-specific position caps |
| `config.py` | Weather-specific configuration (thresholds, locations, toggles) |
| `state.py` | Persistent state (predictions, daily observations, trade history) |

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKET DISCOVERY                             │
│  Gamma API → filter "temperature" events → parse location/date      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                    PARALLEL DATA FETCH                               │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────────┐  │
│  │   NOAA   │  │  Open-Meteo   │  │  Aviation Weather (METAR)     │  │
│  │ weather  │  │  GFS + ECMWF  │  │  Real-time airport obs        │  │
│  │  .gov    │  │  (free API)   │  │  (aviationweather.gov)        │  │
│  └────┬─────┘  └──────┬───────┘  └──────────────┬────────────────┘  │
│       │               │                         │                    │
│  point forecast   model forecasts          obs high/low              │
│  (high/low °F)    (daily high/low)         (running extremes)        │
└───────┼───────────────┼─────────────────────────┼────────────────────┘
        │               │                         │
┌───────▼───────────────▼─────────────────────────▼────────────────────┐
│                    ENSEMBLE FORECAST                                  │
│                                                                       │
│  Weighted average:  ECMWF (50%) + GFS (30%) + NOAA (20%)             │
│  + Aviation obs (dynamic weight: ×2 day-J, ×1 J+1, ×0 J+2+)         │
│                                                                       │
│  Result: single temperature forecast with confidence spread           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   BUCKET PROBABILITY                                  │
│                                                                       │
│  P(temp ∈ [bucket_low, bucket_high]) via Normal CDF                  │
│  σ = f(horizon_days, season, time_of_day, observations)              │
│                                                                       │
│  Day J-7: σ = 6.0°F     Day J 12:00: σ = 2.0°F                      │
│  Day J-1: σ = 2.0°F     Day J 15:00: σ = 1.0°F                      │
│  Day J morning: σ = 3.0°F  Day J 17:00: σ = 0.5°F                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                    EDGE DETECTION                                     │
│                                                                       │
│  edge = our_probability - market_price                                │
│  Filter: edge > entry_threshold (default 15¢)                        │
│  EV = probability × (1/price - 1) × kelly_fraction                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                      SAFEGUARDS                                       │
│                                                                       │
│  ✓ Slippage check (best ask vs mid, max 15%)                         │
│  ✓ Time to resolution (min 2 hours)                                  │
│  ✓ Correlation guard (max 1 position per event)                      │
│  ✓ Flip-flop discipline (no re-entry after recent exit)              │
│  ✓ Price trend detection (avoid catching falling knives)             │
│  ✓ Max trades per run (default 5)                                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   SIZING & EXECUTION                                  │
│                                                                       │
│  Size: fractional Kelly (25%) with position caps ($2 default)        │
│  Order: CLOB limit order at best ask (BUY) or best bid (SELL)        │
│  Contract: NEG_RISK_CTF_EXCHANGE (neg_risk=True for weather)         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   POSITION MONITORING                                 │
│                                                                       │
│  ✓ Dynamic exit thresholds (tighter as resolution approaches)        │
│  ✓ Stop-loss on forecast reversal (5°F shift away from bucket)       │
│  ✓ Observation-aware stop-loss (after 14:00 local for highs)         │
│  ✓ Persistent state (survives restarts)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Weather Data Sources

### 1. NOAA (api.weather.gov)

The official US National Weather Service forecast. Provides point forecasts (high/low temperature) for specific GPS coordinates. Free, no API key required.

- **Endpoint:** `https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast`
- **Data:** Daily high/low temperature forecasts, 7-day outlook
- **Weight in ensemble:** 20%

### 2. Open-Meteo (api.open-meteo.com)

Multi-model forecast API aggregating global weather models. Free, no API key required.

- **Models used:** ECMWF IFS (50% weight) + GFS Seamless (30% weight)
- **Data:** Daily `temperature_2m_max` and `temperature_2m_min` per model
- **Total ensemble weight:** 80% (50% ECMWF + 30% GFS)

### 3. Aviation Weather — METAR Observations (aviationweather.gov)

Real-time actual temperature observations from the exact airport stations used by Polymarket for market resolution. This is the **ground-truth data source**.

- **Endpoint:** `https://aviationweather.gov/api/data/metar?ids=KLGA,KORD&format=json&hours=24`
- **Data:** Hourly METAR reports with actual temperature (°C, converted to °F)
- **Station mapping:** NYC→KLGA, Chicago→KORD, Seattle→KSEA, Atlanta→KATL, Dallas→KDFW, Miami→KMIA
- **Weight in ensemble:** Dynamic — ×2 on resolution day (day-J), ×1 on J+1, ×0 for J+2 and beyond
- **Key advantage:** On resolution day, the running observed high/low constrains the forecast. If it's 15:00 local and the running high is 52°F, a market for "55°F or higher" at 20¢ is almost certainly overpriced.

### Ensemble Calculation

```python
# Base weights (without observations)
ECMWF: 50%  |  GFS: 30%  |  NOAA: 20%

# With observations on resolution day
ECMWF: 50%  |  GFS: 30%  |  NOAA: 20%  |  METAR: 80% (2 × 0.40 base weight)

# All weights are renormalized to sum to 1.0
```

---

## Probability Model

The probability that the actual temperature falls within a given bucket `[low, high]` is estimated using a **normal distribution** centered on the ensemble forecast temperature:

```
P(temp ∈ [low, high]) = Φ((high + 0.5 - forecast) / σ) - Φ((low - 0.5 - forecast) / σ)
```

Where `Φ` is the standard normal CDF and `σ` (standard deviation) depends on:

### Forecast Horizon (days ahead)

| Days Ahead | σ (°F) | Interpretation |
|-----------|--------|----------------|
| 0 (today) | 1.5 | Very confident |
| 1 | 2.0 | |
| 2 | 2.5 | |
| 3 | 3.0 | |
| 5 | 4.0 | Moderate uncertainty |
| 7 | 6.0 | |
| 10 | 9.0 | High uncertainty |

### Seasonal Adjustment

Winter months (Dec-Feb) have a 0.90× accuracy factor (forecasts are less reliable due to storms and cold fronts). Summer months (Jun-Sep) have a 1.00× factor. This widens σ in harder seasons.

### Intraday Observation-Based σ (Resolution Day)

When METAR observations are available on the resolution day, σ shrinks dramatically as the day progresses:

| Local Time | σ for "high" | σ for "low" |
|-----------|-------------|-------------|
| Before 10:00 | 3.0°F | 0.5°F (low already occurred) |
| 10:00–14:00 | 2.0°F | 1.0°F |
| 14:00–17:00 | 1.0°F | 0.5°F |
| After 17:00 | 0.5°F (peak passed) | 0.5°F |

Timezone-aware: UTC offset is derived from station longitude (KLGA→-5, KORD→-6, KSEA→-8).

### Forecast Constraining

On resolution day, the observed running extreme constrains the forecast:
- **High:** actual high cannot be lower than the running observed max (temperature may still go up)
- **Low:** actual low cannot be higher than the running observed min (temperature may still drop)

---

## Trading Strategy

### Entry Criteria

1. **Edge threshold:** `our_probability - market_price > 0.15` (15¢ minimum edge)
2. **EV threshold:** Expected value > 3¢ after Kelly sizing
3. **Horizon:** Within `max_days_ahead` (default 7 days)
4. **Safeguards pass:** Slippage < 15%, time to resolution > 2h, no correlation conflict

### Position Sizing

**Fractional Kelly Criterion** (quarter-Kelly for conservatism):

```
f* = kelly_fraction × (p × b - q) / b

where:
  p = our probability estimate
  b = net odds = (1 / price) - 1
  q = 1 - p
  kelly_fraction = 0.25 (configurable)
```

Position capped at `max_position_usd` (default $2.00) and minimum `MIN_SHARES_PER_ORDER` (5 shares).

### Exit & Stop-Loss

- **Dynamic exits:** Exit threshold tightens as resolution approaches
- **Forecast reversal stop-loss:** If the forecast shifts by > 5°F away from our bucket, exit
- **Observation-aware stop-loss:** On resolution day, uses actual observed temperature instead of forecast (only after 14:00 local for highs, 08:00 for lows, to avoid premature exits from naturally-low morning readings)
- **Correlation guard:** Maximum 1 position per event (prevents correlated losses across buckets of the same market)

---

## Supported Locations

| City | Airport | ICAO | Coordinates | UTC Offset |
|------|---------|------|-------------|------------|
| NYC | LaGuardia | KLGA | 40.78, -73.87 | -5 (Eastern) |
| Chicago | O'Hare | KORD | 41.97, -87.91 | -6 (Central) |
| Seattle | Sea-Tac | KSEA | 47.45, -122.31 | -8 (Pacific) |
| Atlanta | Hartsfield-Jackson | KATL | 33.64, -84.43 | -5 (Eastern) |
| Dallas | DFW | KDFW | 32.90, -97.04 | -6 (Central) |
| Miami | MIA | KMIA | 25.80, -80.29 | -5 (Eastern) |

These match the exact stations Polymarket uses for market resolution.

---

## Installation

### Prerequisites

- Python 3.11+
- An Ethereum wallet with USDC on Polygon (for live trading only)

### Setup

```bash
git clone https://github.com/your-user/Weather_Gully.git
cd Weather_Gully

# Install dependencies
pip install httpx eth-account eth-abi eth-utils websockets certifi

# Verify installation
python3 -m pytest -q
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `httpx` | HTTP client for Gamma API and CLOB REST |
| `eth-account` | EIP-712 order signing and key derivation |
| `eth-abi` | ABI encoding for Order struct |
| `eth-utils` | keccak256 hashing |
| `websockets` | Real-time WebSocket market data |
| `certifi` | SSL certificates (macOS compatibility) |

**No SDK dependencies:** The NOAA, Open-Meteo, and Aviation Weather APIs use Python's built-in `urllib` — no additional HTTP library needed for weather data.

---

## Configuration

Config is loaded with priority: `config.json` > environment variables > defaults.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `POLY_PRIVATE_KEY` | Ethereum private key (hex, 0x-prefixed). Required for `--live` |
| `WEATHER_ENTRY_THRESHOLD` | Minimum edge to enter a position |
| `WEATHER_EXIT_THRESHOLD` | Edge at which to exit |
| `WEATHER_MAX_POSITION` | Maximum USD per position |
| `WEATHER_SIZING_PCT` | Sizing percentage |
| `WEATHER_MAX_TRADES` | Max trades per run |
| `WEATHER_LOCATIONS` | Comma-separated city list |

### Config Fields

Set via `--set key=value` or `config.json`:

| Field | Default | Description |
|-------|---------|-------------|
| **Entry/Exit** | | |
| `entry_threshold` | 0.15 | Minimum edge (probability - price) to enter |
| `exit_threshold` | 0.45 | Edge below which to exit |
| `min_ev_threshold` | 0.03 | Minimum expected value in USD |
| **Sizing** | | |
| `max_position_usd` | 2.00 | Maximum position size in USD |
| `sizing_pct` | 0.05 | Sizing as percentage of balance |
| `kelly_fraction` | 0.25 | Kelly criterion fraction (0.25 = quarter-Kelly) |
| `max_exposure` | 50.00 | Maximum total portfolio exposure |
| **Locations** | | |
| `locations` | NYC | Comma-separated cities to trade |
| **Strategy** | | |
| `max_days_ahead` | 7 | Maximum forecast horizon (days) |
| `seasonal_adjustments` | true | Apply seasonal accuracy factors |
| `adjacent_buckets` | true | Consider adjacent bucket positions |
| `dynamic_exits` | true | Tighten exit threshold near resolution |
| `multi_source` | true | Enable Open-Meteo ensemble (GFS + ECMWF) |
| `correlation_guard` | true | Max 1 position per event |
| `stop_loss_reversal` | true | Exit on forecast reversal |
| `stop_loss_reversal_threshold` | 5.0 | Temperature shift (°F) to trigger stop-loss |
| **Aviation (METAR)** | | |
| `aviation_obs` | true | Enable real-time METAR observations |
| `aviation_obs_weight` | 0.40 | Base weight for observations in ensemble |
| `aviation_hours` | 24 | Hours of METAR history to fetch |
| **Safeguards** | | |
| `slippage_max_pct` | 0.15 | Max allowed slippage (best ask vs mid) |
| `time_to_resolution_min_hours` | 2 | Min hours before resolution to trade |
| `price_drop_threshold` | 0.10 | Price drop that triggers caution |
| `forecast_change_threshold` | 3.0 | Forecast change (°F) triggering re-evaluation |
| **Rate Limiting** | | |
| `max_trades_per_run` | 5 | Maximum trades per strategy run |
| `max_retries` | 3 | HTTP retry count |
| `retry_base_delay` | 1.0 | Base delay for exponential backoff |

---

## Usage

### Weather Bot (Standalone)

```bash
# Dry-run — fetch markets, show opportunities (no trades)
python3 -m weather

# Trade all 6 locations
python3 -m weather --set locations=NYC,Chicago,Seattle,Atlanta,Dallas,Miami

# Live trading (requires POLY_PRIVATE_KEY)
export POLY_PRIVATE_KEY=0x...
python3 -m weather --live

# Verbose logging (DEBUG level)
python3 -m weather --verbose

# Structured JSON logs (for log aggregation)
python3 -m weather --json-log

# Show current positions only
python3 -m weather --positions

# Show current config
python3 -m weather --config

# Disable METAR observations (forecast-only mode)
python3 -m weather --no-aviation

# Disable context safeguards
python3 -m weather --no-safeguards

# Disable price trend detection
python3 -m weather --no-trends

# Set config values
python3 -m weather --set entry_threshold=0.20 --set max_position_usd=5.00

# Combine flags
python3 -m weather --live --verbose --set locations=NYC,Chicago
```

### General Bot

```bash
# Scan all markets for signals
python3 -m bot --scan

# Scan weather markets specifically
python3 -m bot --weather

# Show detected signals (longshot bias, arb, microstructure)
python3 -m bot --signals

# Dry-run strategy
python3 -m bot

# Live trading
python3 -m bot --live

# Daemon mode (continuous, with graceful shutdown)
python3 -m bot --daemon --live
```

### CLOB Client (Direct)

```bash
# List available markets
python3 -m polymarket markets

# View orderbook for a token
python3 -m polymarket book <token_id>

# Get current price
python3 -m polymarket price <token_id>
```

---

## Testing

```bash
# Run all 319 tests
python3 -m pytest -v

# Weather tests only
python3 -m pytest weather/tests/ -v

# Bot tests only
python3 -m pytest bot/tests/ -v

# Polymarket client tests only
python3 -m pytest polymarket/tests/ -v

# Specific test module
python3 -m pytest weather/tests/test_aviation.py -v
python3 -m pytest weather/tests/test_probability.py -v
python3 -m pytest weather/tests/test_strategy.py -v
```

### Test Coverage

| Package | Tests | Coverage |
|---------|-------|----------|
| `weather/` | Strategy, bridge, NOAA parsing, Open-Meteo, aviation/METAR, probability, sizing, state, parsing | ~200 tests |
| `bot/` | Scanner, signals, scoring, sizing, daemon, Gamma API | ~80 tests |
| `polymarket/` | Order signing, HMAC auth, client REST | ~40 tests |

All external API calls are mocked. Test fixtures in `weather/tests/fixtures/` include realistic METAR responses.

---

## Project Structure

```
Weather_Gully/
├── polymarket/
│   ├── __init__.py
│   ├── __main__.py          # CLI: markets, book, price
│   ├── client.py            # CLOB REST client (httpx, retry, HMAC auth)
│   ├── order.py             # EIP-712 order construction & signing
│   ├── auth.py              # L1 key derivation + L2 HMAC signing
│   ├── ws.py                # WebSocket client (auto-reconnect)
│   ├── approve.py           # Token approval for CTF Exchange
│   ├── constants.py         # Chain ID, contract addresses, URLs
│   └── tests/
│       ├── test_auth.py
│       ├── test_client.py
│       └── test_order.py
│
├── bot/
│   ├── __init__.py
│   ├── __main__.py          # CLI: scan, signals, daemon, live
│   ├── gamma.py             # Gamma API client (market discovery)
│   ├── scanner.py           # Market scanning & liquidity grading
│   ├── signals.py           # 4 signal methods (longshot, arb, micro, multi-choice)
│   ├── scoring.py           # Calibration scoring
│   ├── sizing.py            # Kelly criterion (generic bot)
│   ├── strategy.py          # Main strategy loop (generic)
│   ├── daemon.py            # Continuous mode with graceful shutdown
│   ├── config.py            # Bot-specific config
│   ├── state.py             # Bot state persistence
│   ├── retry.py             # Retry utilities
│   └── tests/
│       ├── test_daemon.py
│       ├── test_gamma.py
│       ├── test_scanner.py
│       ├── test_scoring.py
│       ├── test_signals.py
│       └── test_sizing.py
│
├── weather/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── strategy.py          # Main strategy loop (weather-specific)
│   ├── bridge.py            # CLOBWeatherBridge adapter
│   ├── noaa.py              # NOAA Weather API client
│   ├── open_meteo.py        # Open-Meteo multi-model client
│   ├── aviation.py          # Aviation Weather API (METAR observations)
│   ├── probability.py       # Bucket probability estimation (Normal CDF)
│   ├── parsing.py           # Event name parsing (location, date, metric)
│   ├── sizing.py            # Kelly criterion (weather-specific)
│   ├── config.py            # Weather config (dataclass, JSON, env vars)
│   ├── state.py             # Persistent state (predictions, observations)
│   └── tests/
│       ├── test_strategy.py
│       ├── test_bridge.py
│       ├── test_aviation.py
│       ├── test_probability.py
│       ├── test_open_meteo.py
│       ├── test_parsing.py
│       ├── test_sizing.py
│       ├── test_state_extended.py
│       └── fixtures/
│           └── metar_response.json
│
└── README.md
```

---

## Key Algorithms

### 1. Ensemble Forecast Weighting

```python
# Sources and weights
weights = {"ecmwf_ifs025": 0.50, "gfs_seamless": 0.30, "noaa": 0.20}

# With METAR observations (resolution day)
if aviation_obs_temp is not None:
    weights["metar"] = aviation_obs_weight  # 0.40 base, ×2 on day-J
    # All weights renormalized to sum to 1.0

forecast = sum(temp * weight for temp, weight in sources) / sum(weights)
```

### 2. Bucket Probability via Normal CDF

```python
P(temp ∈ [low, high]) = Φ((high + 0.5 - μ) / σ) - Φ((low - 0.5 - μ) / σ)

# μ = ensemble forecast temperature
# σ = standard deviation (horizon, season, time-of-day dependent)
# ±0.5 continuity correction (integer bucket boundaries)
# Sentinel values: -999 = open below, 999 = open above
```

### 3. Kelly Criterion Sizing

```python
b = (1 / price) - 1          # net odds
f_full = (p * b - q) / b     # full Kelly fraction
f = f_full * 0.25            # quarter-Kelly (conservative)
size = min(f * balance, max_position_usd)
```

### 4. Observation-Constrained Forecast

```python
# On resolution day, observed extremes bound the forecast
if metric == "high":
    effective_temp = max(observed_high, model_forecast)
    # "The daily high can't be lower than what we already saw"
elif metric == "low":
    effective_temp = min(observed_low, model_forecast)
    # "The daily low can't be higher than what we already saw"
```

### 5. Intraday Sigma Reduction (Resolution Day)

```python
# As the day progresses, less temperature change is possible
# σ shrinks → bucket probabilities become more concentrated
# → larger edge signals → more confident trades

# Example: 15:00 local, running high = 52°F
# Market "55°F or higher" at 20¢
# σ = 1.0°F → P(≥55) ≈ 0.3% → massive SELL signal
```

---

## How It Works (End-to-End)

1. **Fetch markets** — Gamma API returns active weather markets (temperature buckets per city/date)
2. **Parallel data fetch** — NOAA, Open-Meteo (GFS + ECMWF), and METAR observations fetched concurrently via `ThreadPoolExecutor`
3. **Ensemble forecast** — Weighted average across all sources, with METAR getting dynamic weight based on resolution proximity
4. **Score buckets** — Normal CDF estimates probability for each temperature range, using observation-adjusted sigma on resolution day
5. **Find edge** — Compare our probability vs market price; filter by EV threshold
6. **Safeguards** — Check slippage, time to resolution, correlation guard, flip-flop discipline, price trends
7. **Size** — Fractional Kelly criterion (quarter-Kelly) with position caps
8. **Trade** — CLOB limit order at best ask/bid (`neg_risk=True` for weather markets)
9. **Monitor** — Dynamic exit thresholds, stop-loss on forecast reversal, observation-aware stop-loss on resolution day
10. **Persist state** — Save predictions, observations, and trade history to JSON for restart resilience
