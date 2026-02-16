# Weather Gully

Automated weather trading bot for [Polymarket](https://polymarket.com) — trades temperature prediction markets using a multi-source ensemble forecast (NOAA + Open-Meteo + METAR observations) with empirically calibrated uncertainty, via CLOB direct (no SDK dependency).

Polymarket lists daily temperature markets for 6 US cities: *"Will the highest temperature in NYC on March 15 be 55°F or higher?"* with buckets like `[50-54]`, `[55-59]`, etc. This bot estimates the true probability for each bucket using weather forecasts and real-time airport observations, compares it to the market price, and trades when it finds a significant edge.

---

## Table of Contents

- [Code Quality](#code-quality)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Weather Data Sources](#weather-data-sources)
- [Probability Model](#probability-model)
- [Calibration System](#calibration-system)
- [Paper Trading](#paper-trading)
- [Backtesting](#backtesting)
- [Trading Strategy](#trading-strategy)
- [Execution & Fill Verification](#execution--fill-verification)
- [Risk Management](#risk-management)
- [Supported Locations](#supported-locations)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Usage Recommendations](#usage-recommendations)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key Algorithms](#key-algorithms)

---

## Code Quality

This codebase has been through **4 rounds of comprehensive code audit**, covering all 3 modules (`weather/`, `bot/`, `polymarket/`). ~50 bugs were identified and fixed across the audit rounds, including 2 critical probability distribution bugs found in the latest audit.

**Current status:** 0 critical issues, 0 important issues remaining. 548 tests, all green.

| Category | Examples of fixes applied |
|----------|-------------------------|
| **Security** | Private keys filtered from logs, API credentials use underscore-prefixed attributes with `__repr__` redaction |
| **Order integrity** | Decimal arithmetic (no float rounding), zero-amount order guards, salt stringified, tick size validated |
| **Robustness** | Atomic file writes (tempfile + `os.replace`), file locking (`fcntl.flock`), circuit breaker (5xx only), retry with exponential backoff |
| **Trading logic** | Kelly/exposure formulas correct for BUY and SELL, stop-loss uses state (not API), seasonal month parsed from forecast date, pending orders verified before removal |
| **Data pipeline** | Historical actuals chunked for >90 day ranges, Open-Meteo multi-location batching (6 cities in 3 API calls), Bessel's correction on model spread |
| **Probability math** | Regularized incomplete beta `* a` bug (flattened all Student's t distributions), horizon override for backtest/calibration (was using horizon=0 for all past dates) |

---

## Architecture

```
Weather_Gully/
├── polymarket/      # CLOB client (EIP-712 signing, HMAC L2 auth, WebSocket)
├── bot/             # General trading bot (signals, Kelly sizing, multi-choice arb, daemon)
└── weather/         # Weather strategy (ensemble forecasts, METAR obs, bucket scoring, calibration)
```

The project is organized into three independent Python packages that compose into a single trading pipeline:

### `polymarket/` — Low-Level CLOB Client

Direct implementation of the Polymarket CLOB API protocol. Zero SDK dependencies.

| Module | Description |
|--------|-------------|
| `client.py` | Synchronous REST client with connection pooling (httpx), retry, and HMAC auth inside retry loop |
| `order.py` | EIP-712 order construction and signing (CTF Exchange struct) |
| `auth.py` | L1 API key derivation (EIP-712 typed signature) + L2 HMAC request signing |
| `public.py` | Read-only public client (no authentication, for dry-run mode) |
| `ws.py` | Async WebSocket client with auto-reconnect and exponential backoff |
| `approve.py` | Token approval transactions for CTF Exchange |
| `constants.py` | Chain ID, contract addresses, API endpoints |

**Key implementation details:**
- Orders are signed using EIP-712 typed data (`Order` struct with maker, taker, tokenId, makerAmount, takerAmount, etc.)
- L2 authentication uses HMAC-SHA256 over `{timestamp}{method}{path}{body}` with a derived API secret
- HMAC headers are rebuilt inside the retry loop to ensure fresh timestamps on retries
- Compact JSON serialization (`separators=(",",":")`) is critical for HMAC validation matching the server
- Weather markets use `neg_risk=True` (NEG_RISK_CTF_EXCHANGE contract)
- Fill detection uses approximate float comparison (`abs(matched - original) < 1e-9`)

### `bot/` — General-Purpose Trading Bot

Market-agnostic trading infrastructure with 4 signal detection methods.

| Module | Description |
|--------|-------------|
| `gamma.py` | Gamma API client — market discovery, metadata, multi-choice detection (`negRisk` grouping) |
| `scanner.py` | Market scanning with Gamma (primary) and CLOB fallback, liquidity grading (A/B/C) |
| `signals.py` | 4 edge-detection methods: longshot bias, arbitrage, microstructure, multi-choice arb |
| `scoring.py` | Calibration scoring — Brier score, log score, calibration curves |
| `sizing.py` | Kelly criterion position sizing with risk limits (BUY and SELL formulas) |
| `strategy.py` | Main strategy loop — scans markets, detects signals, sizes and executes trades |
| `daemon.py` | Continuous operation mode with SIGTERM/SIGINT graceful shutdown, health check, interruptible sleep |
| `config.py` | Bot-specific configuration (separate from weather config) |
| `state.py` | Persistent state tracking (positions, P&L, trade history, calibration) |

**Signal detection methods:**
1. **Longshot bias** — Empirical mispricing at price extremes (based on Becker 2024 analysis of 72.1M Kalshi trades). Bias corrections range from -0.008 to +0.008, with a default min_edge of 0.004
2. **Arbitrage** — YES + NO orderbook violations of the $1.00 invariant (buy-both or sell-both)
3. **Microstructure** — Order-book imbalance and spread analysis (Kyle's lambda)
4. **Multi-choice arbitrage** — Sum of YES prices across grouped `negRisk` markets != $1.00

### `weather/` — Weather-Specific Strategy

The core intelligence layer. Combines multiple weather data sources into a probability estimate for each temperature bucket, then trades accordingly.

| Module | Description |
|--------|-------------|
| `strategy.py` | Main strategy loop — fetches data, scores buckets, manages entries/exits/stop-losses |
| `bridge.py` | `CLOBWeatherBridge` adapter — connects weather strategy to CLOB + Gamma APIs, with fill verification |
| `noaa.py` | NOAA Weather API client (api.weather.gov) — official US forecast |
| `open_meteo.py` | Open-Meteo client — GFS + ECMWF multi-model forecasts with auxiliary weather variables |
| `aviation.py` | Aviation Weather API client — METAR real-time observations from airport stations |
| `probability.py` | Student's t-CDF bucket probability, calibrated sigma, Platt scaling, observation-adjusted estimates, weather-based sigma adjustments |
| `parsing.py` | Event name parser — extracts location, date, metric (high/low) from market titles |
| `sizing.py` | Kelly criterion sizing with weather-specific position caps |
| `config.py` | Weather-specific configuration (thresholds, locations, toggles) |
| `state.py` | Persistent state (predictions, daily observations, trade history, pending orders) |
| `historical.py` | Open-Meteo historical forecast and ERA5 reanalysis client (for calibration) |
| `previous_runs.py` | Open-Meteo Previous Runs API client — horizon-specific historical forecasts for real RMSE computation |
| `metar_actuals.py` | Historical METAR ground-truth actuals (airport obs, more accurate than ERA5 reanalysis) |
| `calibrate.py` | Calibration script — computes empirical sigma, model weights, Platt scaling, adaptive sigma factors from historical data |
| `recalibrate.py` | Auto-recalibration orchestrator — rolling 90-day window, error cache, guard rails, delta logging |
| `error_cache.py` | Persistent error cache for incremental recalibration (append-only, prunable) |
| `backtest.py` | Backtesting engine — simulates strategy on historical data with Brier scoring, supports real price snapshots |
| `paper_bridge.py` | `PaperBridge` — simulated execution wrapper (real prices, no orders submitted) |
| `paper_trade.py` | Paper trading CLI — runs the real strategy with simulated execution, records price snapshots |
| `calibration.json` | Generated calibration tables (sigma by horizon/location, seasonal factors, model weights) |

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
│  │  .gov    │  │  + cloud/wind │  │  (aviationweather.gov)        │  │
│  │          │  │  + precip     │  │                               │  │
│  └────┬─────┘  └──────┬───────┘  └──────────────┬────────────────┘  │
│       │               │                         │                    │
│  point forecast   model forecasts          obs high/low              │
│  (high/low °F)    + auxiliary vars         (running extremes)        │
└───────┼───────────────┼─────────────────────────┼────────────────────┘
        │               │                         │
┌───────▼───────────────▼─────────────────────────▼────────────────────┐
│                    ENSEMBLE FORECAST                                  │
│                                                                       │
│  Weighted average: weights from calibration.json or defaults          │
│  Typical: ECMWF (50%) + GFS (30%) + NOAA (20%)                      │
│  + Aviation obs (dynamic weight: ×2 day-J, ×1 J+1, ×0 J+2+)         │
│                                                                       │
│  Result: single temperature forecast + model spread diagnostic        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               CALIBRATED BUCKET PROBABILITY                          │
│                                                                       │
│  P(temp ∈ [bucket_low, bucket_high]) via Normal CDF                  │
│  σ = calibrated_base_sigma × horizon_growth × seasonal_factor        │
│  + weather-based adjustment (cloud >80%, wind >40km/h, precip >10mm) │
│                                                                       │
│  Calibration chain: location_sigma → global_sigma → hardcoded        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                    EDGE DETECTION                                     │
│                                                                       │
│  edge = our_probability - market_price (using best_ask when avail)   │
│  Filter: edge > entry_threshold (default 15c)                        │
│  EV = prob - price > min_ev_threshold (default 3c)                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                      SAFEGUARDS                                       │
│                                                                       │
│  - Slippage check (best ask vs mid, max 15%)                         │
│  - Time to resolution (min 2 hours)                                  │
│  - Correlation guard (max 1 position per event)                      │
│  - Flip-flop discipline (no re-entry after recent exit)              │
│  - Price trend detection (avoid catching falling knives)             │
│  - Max trades per run (default 5)                                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               SIZING & EXECUTION                                      │
│                                                                       │
│  Size: fractional Kelly (25%) with position caps ($2 default)        │
│  BUY:  shares = usd / price                                         │
│  SELL: shares = usd / (1 - price)                                    │
│  Order: CLOB limit order at best ask (BUY) or best bid (SELL)        │
│  Fill: verify via polling, cancel unfilled, adjust for partial fills  │
│  Contract: NEG_RISK_CTF_EXCHANGE (neg_risk=True for weather)         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   POSITION MONITORING                                 │
│                                                                       │
│  - Dynamic exit thresholds (tighter as resolution approaches)        │
│  - Stop-loss on forecast reversal (5°F shift away from bucket)       │
│  - Observation-aware stop-loss (after 14:00 local for highs)         │
│  - State-tracked exits (positions from state, not API)               │
│  - Pending order deduplication (skip positions with pending exits)    │
│  - Persistent state (survives restarts)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Weather Data Sources

### 1. NOAA (api.weather.gov)

The official US National Weather Service forecast. Provides point forecasts (high/low temperature) for specific GPS coordinates. Free, no API key required.

- **Endpoint:** `https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast`
- **Data:** Daily high/low temperature forecasts, 7-day outlook
- **Weight in ensemble:** 20% (default, adjustable via calibration)

### 2. Open-Meteo (api.open-meteo.com)

Multi-model forecast API aggregating global weather models. Free, no API key required.

- **Models used:** ECMWF IFS (50% weight) + GFS Seamless (30% weight)
- **Data:** Daily `temperature_2m_max` and `temperature_2m_min` per model
- **Auxiliary variables:** `cloud_cover_max`, `wind_speed_10m_max`, `wind_gusts_10m_max`, `precipitation_sum`, `precipitation_probability_max` — used for dynamic sigma adjustment
- **Total ensemble weight:** 80% (50% ECMWF + 30% GFS)
- **Calibrated weights:** Per-location weights from `calibration.json` (ECMWF consistently outperforms GFS, with inverse-RMSE weighting)

### 3. Aviation Weather — METAR Observations (aviationweather.gov)

Real-time actual temperature observations from the exact airport stations used by Polymarket for market resolution. This is the **ground-truth data source**.

- **Endpoint:** `https://aviationweather.gov/api/data/metar?ids=KLGA,KORD&format=json&hours=24`
- **Data:** Hourly METAR reports with actual temperature (°C, converted to °F)
- **Station mapping:** NYC→KLGA, Chicago→KORD, Seattle→KSEA, Atlanta→KATL, Dallas→KDFW, Miami→KMIA
- **Weight in ensemble:** Dynamic — ×2 on resolution day (day-J), ×1 on J+1, ×0 for J+2 and beyond
- **Key advantage:** On resolution day, the running observed high/low constrains the forecast. If it's 15:00 local and the running high is 52°F, a market for "55°F or higher" at 20c is almost certainly overpriced.

### Ensemble Calculation

```python
# Base weights (from calibration.json or defaults)
ECMWF: 50%  |  GFS: 30%  |  NOAA: 20%

# With observations on resolution day
ECMWF: 50%  |  GFS: 30%  |  NOAA: 20%  |  METAR: 80% (2 x 0.40 base weight)

# All weights are renormalized to sum to 1.0
# Per-location weights override global when available
```

---

## Probability Model

The probability that the actual temperature falls within a given bucket `[low, high]` is estimated using a **Student's t-distribution** (df=10) centered on the ensemble forecast temperature, with **Platt scaling** for market calibration:

```
P_raw(temp in [low, high]) = T_df((high + 0.5 - forecast) / sigma) - T_df((low - 0.5 - forecast) / sigma)
P_calibrated = sigmoid(a * logit(P_raw) + b)     # Platt scaling
```

Where `T_df` is the Student's t-CDF (df=10, heavier tails than Gaussian — better fits real forecast error distribution which has kurtosis=4.4), and `sigma` depends on multiple factors:

### 1. Calibrated Horizon Sigma

Sigma grows with forecast horizon using **real RMSE** computed from the Open-Meteo Previous Runs API — actual forecast errors measured at each horizon, not a theoretical growth model.

| Horizon (days) | Global sigma | NYC    | Miami  | Seattle |
|----------------|-------------|--------|--------|---------|
| 0 (today)      | 2.34°F      | 3.64°F | 1.82°F | 1.81°F  |
| 1              | 4.00°F      | 6.25°F | 3.07°F | 2.65°F  |
| 3              | 4.60°F      | 5.92°F | 3.47°F | 2.71°F  |
| 5              | 5.30°F      | 6.31°F | 4.12°F | 3.12°F  |
| 7              | 7.54°F      | 8.44°F | 5.12°F | 4.32°F  |
| 10             | 10.89°F     | 11.63°F| 6.61°F | 6.12°F  |

Calibrated from ~2,100 weighted samples (exponential half-life 30 days) across 6 locations with METAR ground-truth actuals. Per-location sigma tables capture climate-specific error profiles (e.g., Miami is much more predictable than NYC/Chicago).

### 2. Seasonal Adjustment

Sigma is adjusted by an empirically derived seasonal factor (from `calibration.json`). Winter months have higher sigma (more uncertain), summer months have lower sigma.

The adjustment is applied as: `sigma /= seasonal_factor` — so a factor of 0.90 (winter) widens sigma by ~11%.

### 3. Weather-Based Sigma Adjustment

Auxiliary weather conditions from Open-Meteo dynamically adjust sigma:

| Condition | Effect on sigma |
|-----------|----------------|
| Cloud cover > 80% (for highs) | +10% |
| Wind > 40 km/h | +8% |
| Precipitation > 10mm (for highs) | +12% |

### 4. Intraday Observation-Based Sigma (Resolution Day)

When METAR observations are available on the resolution day, sigma shrinks dramatically as the day progresses:

| Local Time | sigma for "high" | sigma for "low" |
|-----------|-----------------|-----------------|
| Before 10:00 | 3.0°F | 0.5°F (low already occurred) |
| 10:00-14:00 | 2.0°F | 1.0°F |
| 14:00-17:00 | 1.0°F | 0.5°F |
| After 17:00 | 0.5°F (peak passed) | 0.5°F |

Timezone-aware: uses IANA timezone from location config (NYC → America/New_York, etc.).

### 5. Forecast Constraining

On resolution day, the observed running extreme constrains the forecast:
- **High:** actual high cannot be lower than the running observed max (temperature may still go up)
- **Low:** actual low cannot be higher than the running observed min (temperature may still drop)

### Sigma Lookup Chain

The system uses a three-level fallback for sigma values:

1. **Location-specific calibrated sigma** (`calibration.json → location_sigma → NYC → horizon 3`)
2. **Global calibrated sigma** (`calibration.json → global_sigma → horizon 3`)
3. **Hardcoded defaults** (built-in `_HORIZON_STDDEV` table)

This ensures the bot works out-of-the-box (hardcoded fallback) while benefiting from calibration data when available.

### 7. Platt Scaling (Market Calibration)

Raw probability estimates are post-processed through **Platt scaling** to align with observed market frequencies:

```
P_calibrated = sigmoid(a * logit(P_raw) + b)
```

Current calibrated parameters: `a=0.6205`, `b=0.6942` (fitted from ~2,100 weighted forecast error samples). This corrects systematic biases in the probability model — e.g., if raw probabilities of 30% historically win 35% of the time, Platt scaling adjusts upward.

### 8. Adaptive Sigma (Live Adjustment)

In addition to the static calibration, sigma can be adjusted in real-time using three signals:

| Signal | Factor | Description |
|--------|--------|-------------|
| **Ensemble underdispersion** | 1.30 | NWP ensembles systematically underestimate spread |
| **Model spread** | `spread × 0.648` | When GFS and ECMWF disagree, widen sigma proportionally |
| **Recent error EMA** | `ema × 1.053` | Exponential moving average of recent absolute errors |

These factors are empirically calibrated from historical data (not literature defaults).

---

## Calibration System

The calibration pipeline replaces hardcoded sigma/seasonal tables with values empirically derived from real forecast error data. Two modes: **full calibration** (from scratch) and **auto-recalibration** (incremental rolling window).

### How It Works

1. **Fetch historical data** — `weather/historical.py` retrieves archived model forecasts (Open-Meteo) and ERA5/METAR actuals for a given date range
2. **Compute forecast errors** — `weather/calibrate.py` computes `forecast - actual` for each (target_date, model, metric) tuple, deduplicated
3. **Derive base sigma** — Weighted standard deviation of errors (exponential half-life 30 days = more weight on recent data)
4. **Real RMSE by horizon** — `weather/previous_runs.py` fetches horizon-specific forecasts from the Previous Runs API; RMSE computed per horizon using METAR ground-truth actuals
5. **Compute per-location and seasonal factors** — Separate sigma and seasonal adjustments for each city
6. **Compute model weights** — Inverse-RMSE weighting per location (lower RMSE = higher weight)
7. **Fit Platt scaling** — Logistic regression on (predicted probability, actual outcome) pairs
8. **Fit adaptive sigma factors** — `spread_to_sigma_factor` and `ema_to_sigma_factor` from historical model spread and error EMA
9. **Distribution selection** — Jarque-Bera test determines Normal vs Student's t (currently t with df=10)
10. **Generate `calibration.json`** — Output file consumed by `probability.py` at runtime

### Auto-Recalibration

The `weather/recalibrate.py` module provides incremental recalibration on a rolling 90-day window:

```bash
# Run recalibration (fetches new errors, updates calibration.json)
python3 -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas
```

Features:
- **Error cache** (`error_cache.py`) — persistent, append-only, with incremental fetching (only fetches new days)
- **Guard rails** — minimum 50 effective samples required, delta clamping on sigma/Platt params
- **Delta logging** — each recalibration logs the change vs previous calibration to `recalibration_log/`
- **Horizon errors** — fetches Previous Runs data + METAR actuals for real RMSE computation

### Running Calibration

```bash
# Calibrate on a full year for all 6 locations
python3 -m weather.calibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --start-date 2025-01-01 \
  --end-date 2026-01-01

# Calibrate for a single location
python3 -m weather.calibrate --locations NYC --start-date 2025-01-01 --end-date 2026-01-01

# Custom output path
python3 -m weather.calibrate --locations NYC --start-date 2025-06-01 --end-date 2025-12-31 --output my_calibration.json
```

### Output Format (`calibration.json`)

```json
{
  "global_sigma": {"0": 2.34, "1": 4.00, "3": 4.60, "5": 5.30, "7": 7.54, "10": 10.89},
  "location_sigma": {
    "NYC": {"0": 3.64, "1": 6.25, "3": 5.92, ...},
    "Miami": {"0": 1.82, "1": 3.07, "3": 3.47, ...}
  },
  "seasonal_factors": {"1": 0.944, "2": 1.258, "11": 0.907, "12": 0.891},
  "location_seasonal": {
    "NYC": {"1": 0.918, "2": 1.498, ...}
  },
  "model_weights": {
    "NYC": {"noaa": 0.20, "gfs_seamless": 0.486, "ecmwf_ifs025": 0.314}
  },
  "adaptive_sigma": {
    "underdispersion_factor": 1.3,
    "spread_to_sigma_factor": 0.648,
    "ema_to_sigma_factor": 1.053,
    "samples": 176
  },
  "platt_scaling": {"a": 0.6205, "b": 0.6942},
  "distribution": "student_t",
  "student_t_df": 10.0,
  "normality_test": {
    "normal": false,
    "jb_statistic": 2154.21,
    "skewness": -1.098,
    "kurtosis": 4.434
  },
  "metadata": {
    "generated": "2026-02-16T11:23:57+00:00",
    "samples": 2112,
    "samples_effective": 852.1,
    "weighting": "exponential half-life 30.0d",
    "base_sigma_global": 2.51,
    "horizon_growth_model": "real RMSE from Previous Runs data"
  }
}
```

### Key Design Decision: Real RMSE from Previous Runs API

Open-Meteo's standard forecast API does **not** store historical model runs. However, the **Previous Runs API** (`previous-runs-api.open-meteo.com`) archives forecasts from past model runs, enabling direct measurement of forecast error at each horizon.

The solution uses **real RMSE by horizon**:
- For each horizon (0, 1, 2, 3, 5, 7 days), fetch the actual forecast that was made at that lead time
- Compare against **METAR ground-truth observations** (airport stations matching Polymarket's resolution source)
- Compute weighted RMSE per horizon per location (exponential half-life 30 days)
- Fallback: if Previous Runs data is unavailable, uses NWP growth model as backup

---

## Paper Trading

Paper trading runs the **exact same strategy code** as live trading, but intercepts all order execution — no real orders are submitted. It uses real Polymarket prices via the Gamma and CLOB APIs (read-only) and records price snapshots for later use in backtesting.

### How It Works

```
GammaClient → real markets       PublicClient → real orderbooks
                ↓                                    ↓
        PaperBridge (wraps CLOBWeatherBridge)
          ├─ fetch_weather_markets()  → delegates to real bridge
          ├─ execute_trade()          → simulated (no CLOB call)
          ├─ execute_sell()           → simulated (no CLOB call)
          └─ save_snapshots()         → price_snapshots.json
                ↓
        run_weather_strategy()  (identical code path)
                ↓
        paper_state.json (trades, predictions, forecasts)
```

**Safety:** `PublicClient` has no `post_order()` method — even if the PaperBridge fails to intercept, no real order can be placed.

### Running Paper Trading

```bash
# Basic paper trading (all default locations)
python3 -m weather.paper_trade

# Verbose mode (DEBUG logging)
python3 -m weather.paper_trade --verbose

# Specific locations
python3 -m weather.paper_trade --set locations=NYC,Chicago

# Disable safeguards (for testing)
python3 -m weather.paper_trade --no-safeguards
```

### What It Does

1. Loads config (no private key needed)
2. Builds a read-only bridge wrapped in `PaperBridge`
3. Loads `paper_state.json` (separate from real trading state)
4. **Resolves past predictions** via `GammaClient.check_resolution()` and prints P&L summary
5. Runs `run_weather_strategy()` with `dry_run=False` — trades go through `PaperBridge`
6. Saves price snapshots to `price_snapshots.json` (append-only)
7. Saves paper state

### Generated Files

| File | Description |
|------|-------------|
| `weather/paper_state.json` | Paper trading state (trades, predictions, forecasts) — separate from real `weather_state.json` |
| `weather/price_snapshots.json` | Append-only history of real Polymarket prices, used by the backtest engine |

---

## Backtesting

The backtesting engine simulates the strategy on historical data to validate parameters and measure prediction quality.

### Pricing Model

The backtest supports two pricing modes:

1. **Real price snapshots** (from paper trading) — when `--snapshot-path` is provided, uses actual recorded Polymarket prices for matching buckets
2. **Probabilistic model** (default fallback) — simulates a semi-efficient market where prices are correlated with true probabilities plus Gaussian noise (~5% model/market disagreement): `price = clamp(prob + N(0, 0.05), 0.02, 0.98)`. Deterministic for reproducibility (seeded RNG).

This replaces the previous uniform `1/N` pricing (~$0.14 for all buckets) which was unrealistic and inflated ROI.

### Running a Backtest

```bash
# Standard backtest (probabilistic pricing model)
python3 -m weather.backtest \
  --locations NYC,Chicago \
  --start-date 2025-06-01 \
  --end-date 2025-12-31 \
  --horizon 3 \
  --output backtest_report.json

# Backtest with real price snapshots (after accumulating paper trading data)
python3 -m weather.backtest \
  --locations NYC \
  --start-date 2026-02-01 \
  --end-date 2026-02-15 \
  --snapshot-path weather/price_snapshots.json
```

### What It Does

For each day in the range:
1. Fetch historical model forecasts for the target date
2. Compute ensemble forecast using `compute_ensemble_forecast()`
3. Generate temperature buckets around the forecast
4. Estimate probability for each bucket via `estimate_bucket_probability()`
5. Price the bucket: use real snapshot if available, otherwise probabilistic model
6. Simulate a trade if EV > entry_threshold
7. Resolve against ERA5 actual temperatures

### Output Metrics

| Metric | Description |
|--------|-------------|
| **Brier score** | Mean squared error of probability estimates (lower = better, <0.25 is good) |
| **Accuracy** | Fraction of trades that won |
| **Total P&L** | Cumulative profit/loss |
| **ROI** | P&L / total amount risked |
| **Max drawdown** | Largest peak-to-trough decline |
| **Sharpe ratio** | Risk-adjusted return (sample variance) |
| **Calibration curve** | Reliability diagram (predicted vs actual frequency per bin) |

### Backtest Results (2024-2025, 6 cities, horizon=3)

```
Total trades:      9,677
Brier score:       0.1277
Accuracy:          19.8%
Total P&L:         +$268
ROI:               16.2%
Max drawdown:      $16.23
Sharpe ratio:      0.08
```

**Overfitting validation** (split backtests):
- 2024 (100% out-of-sample): 16.3% ROI
- 2025 (partial in-sample): 16.1% ROI
- Summer 2024 (max out-of-sample): 19.4% ROI
- Winter 2025 (in-sample): 12.2% ROI

OOS performance equals or exceeds in-sample — no overfitting detected.

### Interpreting Results

- **Brier score < 0.15**: Excellent calibration (current model achieves 0.128)
- **Brier score 0.15-0.25**: Good — profitable if combined with proper sizing
- **Brier score > 0.25**: Overconfident predictions — consider widening sigma or re-calibrating
- **Calibration curve**: Each point should lie close to the diagonal. Points above the diagonal mean you are underconfident (opportunity), below means overconfident (danger)

---

## Trading Strategy

### Entry Criteria

1. **Edge threshold:** `our_probability - market_price > entry_threshold` (default 0.15)
2. **EV threshold:** Expected value > `min_ev_threshold` (default 0.03)
3. **Price ceiling:** `market_price < entry_threshold` (skip already-expensive buckets)
4. **Horizon:** Within `max_days_ahead` (default 7 days)
5. **Safeguards pass:** Slippage < 15%, time to resolution > 2h, no correlation conflict
6. **Minimum shares:** Position size covers at least `MIN_SHARES_PER_ORDER` (5 shares)

### Position Sizing

**Fractional Kelly Criterion** (quarter-Kelly for conservatism):

```
BUY:  f* = (p - price) / (1 - price)
SELL: f* = (price - p) / price

position_usd = bankroll x f* x kelly_fraction
shares_BUY  = position_usd / price
shares_SELL = position_usd / (1 - price)
```

Position capped at `max_position_usd` (default $2.00) and `max_exposure` (default $50.00).

### Best Ask Pricing

The strategy uses `best_ask` from the orderbook for edge calculation when available, falling back to `external_price_yes` (Gamma mid-price). This gives a more accurate execution price and avoids phantom edges from stale mid-prices.

### Exit & Stop-Loss

- **Dynamic exits:** Exit threshold tightens as resolution approaches (closer = lower target, always requires 5c profit above cost basis)
- **Forecast reversal stop-loss:** If the forecast shifts by > 5°F away from our bucket, exit
- **Observation-aware stop-loss:** On resolution day, uses actual observed temperature instead of forecast (only after 14:00 local for highs, 08:00 for lows, to avoid premature exits)
- **Correlation guard:** Maximum 1 position per event (prevents correlated losses across buckets of the same market)
- **State-tracked exits:** Exit logic reads positions from state (not from the API, which returns [] for weather positions tracked locally)

---

## Execution & Fill Verification

### Order Flow

1. **Re-fetch price** — Before submitting, re-fetch the orderbook to get the freshest best ask/bid
2. **Submit CLOB limit order** — `neg_risk=True` for weather markets
3. **Verify fill** — Poll `GET /data/order/{id}` until MATCHED, CANCELLED, or timeout
4. **Handle partial fills** — If partially filled, cancel the remaining portion and record the actual filled amount
5. **Cancel unfilled** — If not filled within `fill_timeout_seconds` (default 30s), cancel the order
6. **Track exposure** — Exposure is tracked using `price x actual_shares` (not the originally requested amount)

### Fill States

| Status | Action |
|--------|--------|
| `MATCHED` | Fully filled — record trade, update exposure |
| Partially filled | Cancel remainder, record actual shares filled |
| Timeout | Cancel order, do not record trade |
| `CANCELLED` | Check if any portion was filled before cancellation |

### Pending Orders

The state tracks `PendingOrder` records for orders submitted but not yet confirmed. Positions with `memo="pending_exit"` are skipped by the exit/stop-loss logic to avoid duplicate exit orders.

---

## Risk Management

### Position-Level Limits

| Check | Default | Description |
|-------|---------|-------------|
| Max position size | $2.00 | Maximum USD per trade |
| Min shares per order | 5 | Polymarket minimum |
| Min tick size | $0.01 | Minimum price increment |
| Kelly fraction | 0.25 | Quarter-Kelly for conservatism |

### Portfolio-Level Limits (bot module)

| Check | Default | Description |
|-------|---------|-------------|
| Max total exposure | $50.00 | Sum of all open position costs |
| Max open positions | 10 | Prevents over-diversification |
| Max daily loss | $10.00 | Realized + unrealized PnL circuit-breaker |
| Max trades per run | 5 | Rate limiting |

### Market-Level Safeguards

| Check | Default | Description |
|-------|---------|-------------|
| Slippage | < 15% | Best ask vs mid-price |
| Time to resolution | > 2 hours | Don't trade too close to settlement |
| Correlation guard | 1 per event | Max 1 position per temperature event |
| Forecast change alert | 3.0°F | Re-evaluate when forecast shifts significantly |
| Price drop detection | 10% | Identify falling-knife scenarios |

### State Persistence

- **Atomic writes:** State is written to a temp file then atomically renamed (`os.replace`) to prevent corruption on crash
- **File locking:** `fcntl.flock(LOCK_EX)` prevents concurrent bot runs from corrupting state
- **Pruning:** Old predictions, observations, and analyzed markets are automatically pruned to prevent unbounded growth

---

## Supported Locations

| City | Airport | ICAO | Coordinates | Timezone |
|------|---------|------|-------------|----------|
| NYC | LaGuardia | KLGA | 40.78, -73.87 | America/New_York |
| Chicago | O'Hare | KORD | 41.97, -87.91 | America/Chicago |
| Seattle | Sea-Tac | KSEA | 47.45, -122.31 | America/Los_Angeles |
| Atlanta | Hartsfield-Jackson | KATL | 33.64, -84.43 | America/New_York |
| Dallas | DFW | KDFW | 32.90, -97.04 | America/Chicago |
| Miami | MIA | KMIA | 25.80, -80.29 | America/New_York |

These match the exact stations Polymarket uses for market resolution.

---

## Installation

### Prerequisites

- Python 3.11+ (developed on Python 3.14)
- An Ethereum wallet with USDC on Polygon (for live trading only)

### Setup

```bash
git clone <repo-url>
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
| `adjacent_buckets` | true | Score all buckets (vs. single-match) |
| `dynamic_exits` | true | Tighten exit threshold near resolution |
| `multi_source` | true | Enable Open-Meteo ensemble (GFS + ECMWF) |
| `correlation_guard` | true | Max 1 position per event |
| `stop_loss_reversal` | true | Exit on forecast reversal |
| `stop_loss_reversal_threshold` | 5.0 | Temperature shift (°F) to trigger stop-loss |
| **Aviation (METAR)** | | |
| `aviation_obs` | true | Enable real-time METAR observations |
| `aviation_obs_weight` | 0.40 | Base weight for observations in ensemble |
| `aviation_hours` | 24 | Hours of METAR history to fetch |
| **Execution** | | |
| `fill_timeout_seconds` | 30.0 | Seconds to wait for fill verification |
| `fill_poll_interval` | 2.0 | Seconds between fill status polls |
| **Safeguards** | | |
| `slippage_max_pct` | 0.15 | Max allowed slippage |
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

# Show calibration stats (Brier score, log score)
python3 -m bot --calibration

# Dry-run strategy
python3 -m bot

# Live trading
python3 -m bot --live

# Daemon mode (continuous, with graceful shutdown)
python3 -m bot --daemon --live

# Check daemon health
python3 -m bot --health
```

### Calibration

```bash
# Full calibration (all locations, 1 year)
python3 -m weather.calibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --start-date 2025-01-01 --end-date 2026-01-01

# Quick calibration (single location, 6 months)
python3 -m weather.calibrate --locations NYC \
  --start-date 2025-06-01 --end-date 2025-12-31
```

### Paper Trading

```bash
# Paper trade with real Polymarket prices (no orders submitted)
python3 -m weather.paper_trade --verbose

# Specific locations
python3 -m weather.paper_trade --set locations=NYC,Chicago

# Accumulate price snapshots over time for realistic backtests
# (run daily via cron or manually)
python3 -m weather.paper_trade
```

### Backtesting

```bash
# Backtest NYC over 6 months (probabilistic pricing)
python3 -m weather.backtest --locations NYC \
  --start-date 2025-06-01 --end-date 2025-12-31 \
  --output report.json

# Backtest with real price snapshots from paper trading
python3 -m weather.backtest --locations NYC \
  --start-date 2026-02-01 --end-date 2026-02-15 \
  --snapshot-path weather/price_snapshots.json

# Backtest with custom horizon
python3 -m weather.backtest --locations NYC,Chicago \
  --start-date 2025-06-01 --end-date 2025-12-31 \
  --horizon 5
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

## Usage Recommendations

### Step-by-Step Getting Started

Follow these steps **in order** — each one builds on the previous.

#### Step 1: Install and verify

```bash
git clone <repo-url> && cd Weather_Gully
pip install httpx eth-account eth-abi eth-utils websockets certifi
python3 -m pytest -q  # 548 tests should pass
```

#### Step 2: First dry-run

```bash
python3 -m weather --dry-run --verbose
```

Read the output carefully. You should see:
- `Fetched N markets from M events` — Polymarket connection works
- `NOAA forecast for NYC: 7 days` — NOAA API works
- `Open-Meteo: 6 locations in 3 request(s)` — Open-Meteo works
- `METAR: N observations across 6 stations` — aviation API works
- `Bucket: XX-YY°F @ $0.xx — prob=XX.X% EV=0.xxx` — probability model works
- `Safeguard blocked: ...` — safeguards correctly prevent reckless trades

If any API fails, the bot logs warnings and continues with available data sources.

#### Step 3: Calibrate

```bash
python3 -m weather.calibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --start-date 2025-01-01 --end-date 2026-01-01
```

This takes 2-5 minutes (chunked API calls). It produces `weather/calibration.json` with empirical sigma values, seasonal factors, and model weights. **Without calibration, the bot uses hardcoded defaults that are reasonable but not optimized.**

Run dry-run again after calibration — you should see `Loaded calibration data (8760 samples)` in the logs.

#### Step 4: Paper trade (1-2 weeks)

```bash
# Run daily (or set up a cron job)
python3 -m weather.paper_trade --verbose
```

Paper trading uses real Polymarket prices but submits no orders. It:
- Records trades in `weather/paper_state.json` (separate from real state)
- Saves price snapshots to `weather/price_snapshots.json`
- Resolves past predictions and prints a P&L summary

After a few days, you'll have a track record to evaluate.

#### Step 5: Backtest

```bash
# With real price snapshots from paper trading
python3 -m weather.backtest --locations NYC \
  --start-date 2026-02-01 --end-date 2026-02-15 \
  --snapshot-path weather/price_snapshots.json

# Or with synthetic pricing on historical data
python3 -m weather.backtest --locations NYC,Chicago \
  --start-date 2025-06-01 --end-date 2025-12-31 \
  --output backtest_report.json
```

**Target metrics before going live:**
- Brier score < 0.25 (probability estimates are well-calibrated)
- Positive total P&L
- Max drawdown < 30% of total risked

#### Step 6: Go live (small)

```bash
# Set up credentials
export POLY_PRIVATE_KEY=0x...

# Ensure on-chain approvals are in place
python3 -m polymarket.approve

# Start with 1 location, low limits
python3 -m weather --live \
  --set locations=NYC \
  --set max_position_usd=2.00 \
  --set max_exposure=20.00 \
  --verbose
```

Monitor the first few trades closely. Once comfortable, expand:

```bash
# All 6 locations with higher limits
python3 -m weather --live \
  --set locations=NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --set max_position_usd=5.00 \
  --set max_exposure=50.00
```

#### Step 7: Automate with cron

```bash
# Edit crontab
crontab -e

# Run every 30 minutes from 8am to 10pm UTC, Monday-Sunday
*/30 8-22 * * * cd /path/to/Weather_Gully && python3 -m weather --live --json-log >> /var/log/weather_gully.log 2>&1
```

The file lock prevents concurrent runs — if a previous run is still active, the new one exits cleanly.

For continuous operation, use daemon mode instead:

```bash
# Start daemon (runs in background with auto-restart)
nohup python3 -m bot --daemon --live --weather > /var/log/weather_gully.log 2>&1 &

# Check health
python3 -m bot --health
```

---

### Optimal Timing

| Window | Horizon | Sigma | Edge Quality | Recommendation |
|--------|---------|-------|--------------|----------------|
| **Day-J after 14:00 local** | 0 days | 0.5-1.0°F | Excellent | Best time to trade. METAR observations constrain the forecast, creating high-confidence mispriced tails. |
| **Day-J morning** | 0 days | 2.0-3.0°F | Good | Intraday sigma is still moderate. Only trade if edge > 20c. |
| **Day J-1** | 1 day | ~2.6°F | Good | Forecasts are fairly tight. Markets haven't fully adjusted. |
| **Day J-2 to J-3** | 2-3 days | 3.3-3.9°F | Moderate | Tradeable only with large discrepancies (edge > 15c). |
| **Day J-5+** | 5+ days | >5°F | Weak | Sigma too wide for reliable probability estimates. Skip unless edge > 25c. |

**Key insight:** The bot's biggest advantage is on **resolution day afternoons**. At 15:00 local time, if the running observed high is 52°F and the market for "55°F or higher" is at 20c, the intraday sigma (0.5-1.0°F) tells you this bucket has only ~2% probability — a massive edge.

**Run frequency:**
- **Cron:** Every 30 minutes during market hours captures intraday observation updates
- **Daemon:** `run_interval_seconds=300` (5 minutes) is a good default
- **Manual:** At minimum, run once in the morning and once in the afternoon for each resolution day

---

### Risk Tuning Guide

#### Conservative (recommended for start)

```bash
python3 -m weather --live \
  --set kelly_fraction=0.25 \
  --set max_position_usd=2.00 \
  --set max_exposure=20.00 \
  --set entry_threshold=0.15
```

- Quarter-Kelly limits downside variance
- $2 max per position limits individual trade risk
- $20 total exposure means max 10 concurrent positions
- 15c entry threshold = only high-conviction trades

#### Moderate (after 2+ weeks of profitable paper trading)

```bash
python3 -m weather --live \
  --set kelly_fraction=0.25 \
  --set max_position_usd=5.00 \
  --set max_exposure=50.00 \
  --set entry_threshold=0.12
```

#### Aggressive (experienced, with proven Brier < 0.20)

```bash
python3 -m weather --live \
  --set kelly_fraction=0.35 \
  --set max_position_usd=10.00 \
  --set max_exposure=100.00 \
  --set entry_threshold=0.10
```

**Never exceed `kelly_fraction=0.50`** — above half-Kelly, variance grows exponentially while expected return plateaus.

#### Key Parameters Explained

| Parameter | Effect of raising | Effect of lowering |
|-----------|------------------|--------------------|
| `entry_threshold` | Fewer trades, higher average edge | More trades, includes marginal opportunities |
| `kelly_fraction` | Larger position sizes, higher variance | Smaller positions, smoother equity curve |
| `max_position_usd` | More capital per trade | Limits individual trade exposure |
| `max_exposure` | More total capital at risk | Limits aggregate portfolio risk |
| `stop_loss_reversal_threshold` | Tolerates more forecast drift | Exits faster on reversal (may be premature in winter) |
| `slippage_max_pct` | Accepts wider spreads | Requires tighter spreads (may miss opportunities) |

---

### Location-Specific Strategy

| City | Calibrated Sigma | Best Season | Liquidity | Notes |
|------|-----------------|-------------|-----------|-------|
| **NYC** | 2.15°F | Summer | Highest | Best starting location. Coastal — higher winter uncertainty (nor'easters). Tightest spreads on Polymarket. |
| **Chicago** | 1.83°F | Fall | Good | Continental climate — large swings in spring/fall. Good calibration data. |
| **Miami** | 1.94°F | Year-round | Moderate | Subtropical — smallest daily variation, most stable forecasts. Tight markets but smaller edges. |
| **Seattle** | 1.73°F | Summer | Moderate | Marine — narrow range but cloud cover creates forecast challenges. Lowest sigma = tightest probability estimates. |
| **Atlanta** | 1.89°F | Fall | Good | Moderate difficulty. Severe weather days (thunderstorms) can create large errors. |
| **Dallas** | 1.99°F | Summer | Good | Hot summers are very predictable. Winter cold fronts add uncertainty. |

**Recommendations:**
- Start with **NYC only** (highest liquidity, most trades available)
- Add **Chicago + Atlanta** after 1 week (good liquidity, different climate)
- Add **Miami + Dallas + Seattle** once comfortable (full coverage)
- **Seasonal re-calibration:** Winter months (Dec-Feb) have higher sigma — the bot adjusts automatically via `seasonal_factors`, but re-running calibration quarterly improves accuracy

---

### Reading the Logs

The bot outputs structured logs. Here's how to interpret key messages:

#### Healthy Operation

```
Fetched 452 active weather markets           # Market discovery successful
Open-Meteo: 6 locations in 3 request(s)     # API batching working (6 locations, 3 API calls)
METAR: 160 observations across 6 stations   # Real-time observations available
Ensemble forecast: 48.0°F (NOAA=47°F, spread=1.3°F)  # Low spread = models agree
Bucket: 46-47°F @ $0.08 — prob=34.8% EV=0.269        # Good edge found
[DRY RUN] Would buy ...                     # Trade correctly blocked in dry-run
```

#### Warnings to Watch

```
Safeguard blocked: Slippage too high: 45%    # Spread too wide — normal for illiquid buckets
Safeguard blocked: Resolves in 0.0h — too soon  # Market resolving now — correct to skip
Price $0.30 above entry threshold — skip     # Price already too high — no edge
Exit check: 2 trade(s) have no cached market data  # Stale positions — may need manual review
STOP-LOSS: NYC forecast shifted 7.2°F       # Forecast reversed — position exited
```

#### Problems to Investigate

```
Open-Meteo failed after 3 retries           # API rate limit or outage — bot continues with NOAA only
NOAA forecast for NYC: 0 days               # NOAA API down — ensemble uses Open-Meteo + METAR only
Trade failed: Circuit breaker OPEN           # 5+ consecutive server errors — wait 60s cooldown
Could not verify fill for order              # Exchange API timeout — check position manually
```

---

### Monitoring & Maintenance

#### Daily Checks

1. **Review trade log:** Look for unexpected stop-losses or failed fills
2. **Check positions:** `python3 -m weather --positions` shows open positions
3. **Verify state file:** `weather/weather_state.json` should contain current trades

#### Weekly Checks

1. **Check calibration scores:** `python3 -m bot --calibration`
   - Brier score < 0.25 = good calibration
   - Brier score > 0.30 = re-calibrate immediately
2. **Review paper trade P&L** (if running in parallel): `python3 -m weather.paper_trade`
3. **Check for stale pending orders:** Look for `Removing stale pending_fill` in logs

#### Quarterly Maintenance

1. **Re-calibrate** (auto-recalibration uses rolling 90-day window):
   ```bash
   # Incremental recalibration (recommended — fast, uses error cache)
   python3 -m weather.recalibrate --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami

   # Full recalibration from scratch (slower — re-fetches all historical data)
   python3 -m weather.calibrate \
     --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
     --start-date <12-months-ago> --end-date <today>
   ```
2. **Update dependencies:** `pip install --upgrade httpx eth-account eth-abi`
3. **Prune state file:** Old predictions are auto-pruned, but if the file grows >1MB, delete and restart clean

#### State File Recovery

If the state file becomes corrupted (extremely rare thanks to atomic writes):

```bash
# Back up the corrupted file
cp weather/weather_state.json weather/weather_state.json.bak

# Delete it — the bot starts fresh with no positions
rm weather/weather_state.json

# IMPORTANT: Manually close any open positions on Polymarket
# The bot won't know about positions from the deleted state
```

#### Daemon Management

```bash
# Start
nohup python3 -m bot --daemon --live --weather > weather.log 2>&1 &

# Check status
python3 -m bot --health

# Graceful stop (SIGTERM — completes current cycle)
kill $(cat /tmp/weather_bot.pid)

# Emergency stop (SIGINT)
kill -INT $(cat /tmp/weather_bot.pid)

# View recent logs
tail -f weather.log | grep -E "INFO|WARNING|ERROR"
```

---

### What to Avoid

| Mistake | Why It's Dangerous | What to Do Instead |
|---------|-------------------|-------------------|
| `--live` without dry-run first | You don't know what the bot would trade | Always `--dry-run` first on a new config |
| `kelly_fraction > 0.50` | Exponential variance growth, risk of ruin | Stay at 0.25 (quarter-Kelly) |
| `--no-safeguards` in live mode | Removes slippage, time-to-resolution, correlation checks | Only use for testing/paper trading |
| Trading without `calibration.json` | Hardcoded sigma is an average — not optimal for any specific city | Run `python3 -m weather.calibrate` first |
| Multiple instances on same state file | File lock prevents corruption but causes skipped runs | Use separate state files or daemon mode |
| Ignoring model spread > 5°F | GFS and ECMWF disagree = high uncertainty | Bot logs spread — manually review trades when spread is high |
| Going live on day 1 | No track record to validate | Paper trade for 1-2 weeks first |
| Setting `max_exposure` too high | One bad day can wipe out gains | Start with $20, raise to $50 after profitability confirmed |
| Ignoring stale cache warnings | Positions may not be exitable | Check `Exit check: N trade(s) have no cached market data` warnings |
| Running during API outages | Partial data leads to poor probability estimates | Bot degrades gracefully but consider pausing if 2+ sources are down |

---

### Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `0 weather markets found` | Gamma API down or no active weather events | Wait — Polymarket may not have events for today |
| `Open-Meteo failed after 3 retries` | Rate limited (free API, 10K req/day) | Wait 1-2 minutes. Bot batches locations to minimize calls (6 locations = 3 requests). |
| `No tradeable buckets above EV threshold` | No edge — market prices already efficient | Normal. Bot is working correctly — no trade is the right call. |
| `Safeguard blocked: Resolves in 0.0h` | Market is resolving right now | Correct behavior — don't trade at resolution time |
| `Circuit breaker OPEN` | 5+ consecutive CLOB API server errors | Wait 60s. If persistent, check Polymarket status. |
| `Could not verify fill` | Exchange API slow | Order may be filled — check Polymarket UI. Bot records as `pending_fill`. |
| `State lock: another instance running` | Concurrent bot run | Wait for the other instance to finish, or kill it. |
| All tests fail with `ModuleNotFoundError` | Dependencies not installed | Run `pip install httpx eth-account eth-abi eth-utils websockets certifi` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | macOS Python missing certificates | Run `pip install certifi`. The bot auto-falls back but logs a CRITICAL warning. |

---

### Advanced: Multi-Strategy Setup

You can run the weather bot alongside the general market bot for diversified edge:

```bash
# Terminal 1: Weather strategy (temperature markets)
python3 -m weather --live --set locations=NYC,Chicago,Miami

# Terminal 2: General bot (longshot bias, arbitrage, microstructure)
python3 -m bot --live --daemon

# They use separate state files and don't conflict
# Weather: weather/weather_state.json
# Bot:     state.json
```

### Advanced: Custom Calibration Periods

```bash
# Summer-only calibration (most stable, good for summer trading)
python3 -m weather.calibrate --locations NYC \
  --start-date 2025-06-01 --end-date 2025-09-01

# Winter-only (most challenging, calibrates for worst case)
python3 -m weather.calibrate --locations NYC,Chicago \
  --start-date 2024-12-01 --end-date 2025-03-01

# Rolling 6-month (recent data, captures model improvements)
python3 -m weather.calibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --start-date 2025-08-01 --end-date 2026-02-01
```

Choose the calibration window based on what you're trading:
- **Trading in summer?** Calibrate on last summer's data
- **Year-round?** Use a full year for the most robust sigma
- **Conservative?** Calibrate on winter data (highest sigma = widest uncertainty bands)

---

## Testing

```bash
# Run all 548 tests
python3 -m pytest -q

# Weather tests only
python3 -m pytest weather/tests/ -v

# Bot tests only
python3 -m pytest bot/tests/ -v

# Specific test modules
python3 -m pytest weather/tests/test_probability.py -v
python3 -m pytest weather/tests/test_calibrate.py -v
python3 -m pytest weather/tests/test_backtest.py -v
python3 -m pytest weather/tests/test_bridge.py -v
python3 -m pytest weather/tests/test_strategy.py -v
```

### Test Coverage

| Package | Tests | Key coverage |
|---------|-------|-------------|
| `weather/` | ~365 tests | Strategy, bridge, paper bridge, NOAA, Open-Meteo (single + multi-location), aviation/METAR, probability (Student's t, Platt scaling), calibration, recalibration, previous runs, METAR actuals, error cache, backtesting, sizing, state, parsing |
| `bot/` | ~137 tests | Scanner, signals, scoring, sizing, daemon, Gamma API, strategy, state |
| `polymarket/` | ~46 tests | Order signing, HMAC auth, client REST, circuit breaker, fill detection |

All external API calls are mocked. Test fixtures in `weather/tests/fixtures/` include realistic METAR responses.

---

## Project Structure

```
Weather_Gully/
├── polymarket/
│   ├── __init__.py
│   ├── __main__.py          # CLI: markets, book, price
│   ├── client.py            # CLOB REST client (httpx, retry, HMAC auth)
│   ├── public.py            # Read-only public client (no auth, dry-run)
│   ├── order.py             # EIP-712 order construction & signing
│   ├── auth.py              # L1 key derivation + L2 HMAC signing
│   ├── ws.py                # WebSocket client (auto-reconnect)
│   ├── approve.py           # Token approval for CTF Exchange
│   ├── constants.py         # Chain ID, contract addresses, URLs
│   └── tests/
│
├── bot/
│   ├── __init__.py
│   ├── __main__.py          # CLI: scan, signals, daemon, live, health
│   ├── gamma.py             # Gamma API client (market discovery)
│   ├── scanner.py           # Market scanning & liquidity grading
│   ├── signals.py           # 4 signal methods (longshot, arb, micro, multi-choice)
│   ├── scoring.py           # Calibration scoring (Brier, log, curves)
│   ├── sizing.py            # Kelly criterion (BUY + SELL formulas)
│   ├── strategy.py          # Main strategy loop (generic)
│   ├── daemon.py            # Continuous mode with graceful shutdown
│   ├── config.py            # Bot-specific config
│   ├── state.py             # Bot state persistence (with file locking)
│   └── tests/
│
├── weather/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── strategy.py          # Main strategy loop (weather-specific)
│   ├── bridge.py            # CLOBWeatherBridge adapter (with fill verification)
│   ├── noaa.py              # NOAA Weather API client
│   ├── open_meteo.py        # Open-Meteo multi-model client (+ auxiliary vars)
│   ├── aviation.py          # Aviation Weather API (METAR observations)
│   ├── probability.py       # Bucket probability (calibrated sigma, weather adjustments)
│   ├── parsing.py           # Event name parsing (location, date, metric)
│   ├── sizing.py            # Kelly criterion (weather-specific)
│   ├── config.py            # Weather config (dataclass, JSON, env vars)
│   ├── state.py             # Persistent state (predictions, obs, pending orders)
│   ├── historical.py        # Open-Meteo historical + ERA5 client (calibration)
│   ├── previous_runs.py     # Open-Meteo Previous Runs API (horizon-specific forecasts)
│   ├── metar_actuals.py     # Historical METAR ground-truth actuals
│   ├── calibrate.py         # Calibration script (sigma, Platt, adaptive factors, model weights)
│   ├── recalibrate.py       # Auto-recalibration orchestrator (rolling window, guard rails)
│   ├── error_cache.py       # Persistent error cache for incremental recalibration
│   ├── backtest.py          # Backtesting engine (Brier score, drawdown, Sharpe, snapshot pricing)
│   ├── paper_bridge.py      # PaperBridge — simulated execution wrapper
│   ├── paper_trade.py       # Paper trading CLI (python -m weather.paper_trade)
│   ├── calibration.json     # Generated calibration tables
│   ├── _ssl.py              # SSL context (certifi → system → unverified fallback)
│   └── tests/
│       ├── test_strategy.py
│       ├── test_bridge.py
│       ├── test_paper_bridge.py
│       ├── test_aviation.py
│       ├── test_probability.py
│       ├── test_open_meteo.py
│       ├── test_parsing.py
│       ├── test_sizing.py
│       ├── test_state_extended.py
│       ├── test_historical.py
│       ├── test_previous_runs.py
│       ├── test_metar_actuals.py
│       ├── test_calibrate.py
│       ├── test_recalibrate.py
│       ├── test_error_cache.py
│       ├── test_backtest.py
│       └── fixtures/
│
└── README.md
```

---

## Key Algorithms

### 1. Ensemble Forecast Weighting

```python
# Weights from calibration.json (per-location) or defaults
weights = {"ecmwf_ifs025": 0.50, "gfs_seamless": 0.30, "noaa": 0.20}

# With METAR observations (resolution day)
if aviation_obs_temp is not None:
    weights["metar"] = aviation_obs_weight  # 0.40 base, x2 on day-J
    # All weights renormalized to sum to 1.0

forecast = sum(temp * weight for temp, weight in sources) / sum(weights)
```

### 2. Calibrated Bucket Probability

```python
# sigma lookup chain: location → global → hardcoded
sigma = calibration["location_sigma"][location][horizon]  # or fallback

# Seasonal adjustment
sigma /= seasonal_factor[month]  # factor < 1.0 widens sigma

# Weather-based adjustment
if cloud_cover > 80%: sigma *= 1.10
if wind > 40 km/h:    sigma *= 1.08
if precip > 10mm:     sigma *= 1.12

# Guard against degenerate sigma
sigma = max(sigma, 0.01)

# Student's t-CDF probability (df=10, heavier tails than Gaussian)
z_lo = (low - 0.5 - forecast) / sigma
z_hi = (high + 0.5 - forecast) / sigma
P_raw = student_t_cdf(z_hi, df=10) - student_t_cdf(z_lo, df=10)

# Platt scaling (market calibration)
P = sigmoid(0.6205 * logit(P_raw) + 0.6942)
```

### 3. Kelly Criterion Sizing

```python
# BUY: profit if price rises to 1, lose if drops to 0
f_buy  = (p - price) / (1 - price) * kelly_fraction
shares_buy = position_usd / price

# SELL: profit if price drops to 0, lose if rises to 1
f_sell = (price - p) / price * kelly_fraction
shares_sell = position_usd / (1 - price)
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

### 5. Real RMSE Calibration

```python
# 1. Fetch horizon-specific forecasts from Previous Runs API
for horizon in [0, 1, 2, 3, 5, 7]:
    forecasts = fetch_previous_runs(lat, lon, start, end, horizons=[horizon])
    actuals = get_historical_metar_actuals(station, start, end)

# 2. Compute weighted RMSE per horizon (half-life 30 days)
for horizon, errors in horizon_errors.items():
    weights = exp(-age / half_life)  # recent errors count more
    sigma[horizon] = sqrt(weighted_mean(error², weights))

# 3. Distribution selection via Jarque-Bera test
if jb_statistic > 5.99:  # p < 0.05
    distribution = "student_t"
    df = fit_student_t_df(errors)  # currently df=10

# 4. Platt scaling from probability-outcome pairs
a, b = logistic_regression(predicted_probs, actual_outcomes)
```

### 6. Dynamic Exit Threshold

```python
base_target = min(cost_basis * 2.0, 0.80)

if hours_to_resolution < 6:   time_factor = 0.6  # take profit early
elif hours < 24:               time_factor = 0.8
elif hours < 72:               time_factor = 0.9
else:                          time_factor = 1.0

threshold = cost_basis + (base_target - cost_basis) * time_factor
threshold = max(threshold, cost_basis + 0.05)  # minimum 5c profit
```
