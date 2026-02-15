# Weather Gully

Automated weather trading bot for [Polymarket](https://polymarket.com) — trades temperature prediction markets using a multi-source ensemble forecast (NOAA + Open-Meteo + METAR observations) with empirically calibrated uncertainty, via CLOB direct (no SDK dependency).

Polymarket lists daily temperature markets for 6 US cities: *"Will the highest temperature in NYC on March 15 be 55°F or higher?"* with buckets like `[50-54]`, `[55-59]`, etc. This bot estimates the true probability for each bucket using weather forecasts and real-time airport observations, compares it to the market price, and trades when it finds a significant edge.

---

## Table of Contents

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
| `probability.py` | Normal CDF bucket probability, calibrated sigma, observation-adjusted estimates, weather-based sigma adjustments |
| `parsing.py` | Event name parser — extracts location, date, metric (high/low) from market titles |
| `sizing.py` | Kelly criterion sizing with weather-specific position caps |
| `config.py` | Weather-specific configuration (thresholds, locations, toggles) |
| `state.py` | Persistent state (predictions, daily observations, trade history, pending orders) |
| `historical.py` | Open-Meteo historical forecast and ERA5 reanalysis client (for calibration) |
| `calibrate.py` | Calibration script — computes empirical sigma and model weights from historical data |
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

The probability that the actual temperature falls within a given bucket `[low, high]` is estimated using a **normal distribution** centered on the ensemble forecast temperature:

```
P(temp in [low, high]) = phi((high + 0.5 - forecast) / sigma) - phi((low - 0.5 - forecast) / sigma)
```

Where `phi` is the standard normal CDF and `sigma` (standard deviation) depends on multiple factors:

### 1. Calibrated Horizon Sigma

Sigma grows with forecast horizon using a **hybrid approach**: an empirically calibrated base sigma multiplied by NWP (Numerical Weather Prediction) growth factors.

```
sigma(horizon) = base_sigma x growth_factor(horizon)
```

| Horizon (days) | Growth Factor | Example sigma (base=1.96) |
|----------------|---------------|---------------------------|
| 0 (today)      | 1.00          | 1.96°F                    |
| 1              | 1.33          | 2.61°F                    |
| 2              | 1.67          | 3.27°F                    |
| 3              | 2.00          | 3.92°F                    |
| 5              | 2.67          | 5.23°F                    |
| 7              | 4.00          | 7.84°F                    |
| 10             | 6.00          | 11.76°F                   |

The base sigma is empirically computed from ~8,760 forecast-vs-actual comparison samples across 6 locations and a full year of data (2025). Per-location sigmas are available (e.g., NYC: 2.15°F, Seattle: 1.73°F).

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

---

## Calibration System

The calibration pipeline replaces hardcoded sigma/seasonal tables with values empirically derived from real forecast error data.

### How It Works

1. **Fetch historical data** — `weather/historical.py` retrieves archived model forecasts (Open-Meteo) and ERA5 reanalysis actuals for a given date range, using chunked 90-day API requests
2. **Compute forecast errors** — `weather/calibrate.py` computes `forecast - actual` for each (target_date, model, metric) tuple, deduplicated to avoid counting the same error multiple times
3. **Derive base sigma** — Standard deviation of all forecast errors = empirical base sigma
4. **Apply horizon growth model** — `sigma(h) = base_sigma x growth_factor(h)` using NWP-derived growth factors
5. **Compute per-location and seasonal factors** — Separate sigma and seasonal adjustments for each city
6. **Compute model weights** — Inverse-RMSE weighting per location (lower RMSE = higher weight)
7. **Generate `calibration.json`** — Output file consumed by `probability.py` at runtime

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
  "global_sigma": {"0": 1.96, "1": 2.61, "2": 3.27, ...},
  "location_sigma": {
    "NYC": {"0": 2.15, "1": 2.86, ...},
    "Miami": {"0": 1.94, "1": 2.58, ...}
  },
  "seasonal_factors": {"1": 0.88, "6": 1.05, ...},
  "location_seasonal": {
    "NYC": {"1": 0.85, "7": 1.08, ...}
  },
  "model_weights": {
    "NYC": {"noaa": 0.20, "gfs_seamless": 0.25, "ecmwf_ifs025": 0.55}
  },
  "metadata": {
    "generated": "2026-01-15T12:00:00+00:00",
    "samples": 8760,
    "base_sigma_global": 1.96,
    "mean_model_spread": 2.3,
    "horizon_growth_model": "NWP linear: sigma(h) = base * growth(h)"
  }
}
```

### Key Design Decision: Hybrid Approach

Open-Meteo's free API does **not** store historical model runs — querying for the forecast from 5 days ago returns the same forecast as today. This means we cannot directly measure how error grows with horizon from the API alone.

The solution is a **hybrid approach**:
- **Empirical base sigma** from real forecast-vs-actual comparisons (what the API does provide)
- **NWP horizon growth model** derived from NOAA/NWS model verification statistics (how error grows with lead time is well-established in meteorological literature)

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

### Interpreting Results

- **Brier score < 0.20**: Excellent calibration — your probability estimates closely match reality
- **Brier score 0.20-0.25**: Good — profitable if combined with proper sizing
- **Brier score > 0.25**: Overconfident predictions — consider widening sigma or adjusting thresholds
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

- Python 3.11+
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

### Getting Started

1. **Start with dry-run mode** — Always run without `--live` first to understand what the bot would trade. Check the logs for opportunity detection, probability estimates, and edge calculations.

2. **Calibrate before trading** — Run `python3 -m weather.calibrate` to generate `calibration.json`. Without it, the bot uses hardcoded defaults that are reasonable but not optimized for current conditions.

3. **Paper trade first** — Run `python3 -m weather.paper_trade` daily to simulate trading with real Polymarket prices. This accumulates price snapshots and tracks predictions for resolution. Review the P&L summary after a few days.

4. **Backtest** — Use `python3 -m weather.backtest` to validate on historical data. After accumulating paper trading data, use `--snapshot-path weather/price_snapshots.json` for realistic pricing. Aim for a Brier score < 0.25 and a positive total P&L before going live.

5. **Start with 1 location** — Begin with `--set locations=NYC` (highest liquidity) before adding more cities. Each location adds API calls and potential positions.

### Optimal Timing

- **Best edge:** Resolution day (day-J), especially after 14:00 local time. Intraday sigma drops to 0.5-1.0°F, enabling high-confidence trades. The observed running extreme constrains the forecast, often creating mispriced tails.

- **Good edge:** Day before resolution (J-1), when the ensemble forecast is already fairly tight (sigma ~2.6°F) and markets haven't fully adjusted.

- **Weak edge:** Day J-5 and beyond. Sigma grows rapidly (>5°F), making probability estimates wide. Only trade if there's a very large discrepancy between your estimate and the market price.

- **Run frequency:** For a cron-based setup, run every 30-60 minutes during market hours. For daemon mode, the `run_interval_seconds` config controls the cycle (default: 300s). More frequent runs capture intraday observation updates.

### Risk Tuning

- **Conservative start:** Keep defaults (`kelly_fraction=0.25`, `max_position_usd=2.00`, `max_exposure=50.00`). Quarter-Kelly is intentionally conservative — full Kelly is theoretically optimal but has extreme variance.

- **Entry threshold:** The default 0.15 (15c edge) is strict. Lower to 0.10 for more trades with smaller edges, or raise to 0.20 for fewer but higher-conviction trades.

- **Stop-loss threshold:** The default 5°F shift is reasonable for most markets. In volatile winter weather, consider raising to 7°F to avoid premature stop-losses from normal forecast oscillation.

- **Correlation guard:** Keep enabled (`correlation_guard=true`). Without it, the bot could take positions on multiple buckets of the same event, creating correlated risk.

### Location-Specific Notes

- **NYC (LaGuardia):** Highest volume, tightest spreads. Best liquidity for all bucket positions. Coastal location means higher winter uncertainty (nor'easters, ocean influence).

- **Chicago (O'Hare):** Continental climate — large daily temperature swings, especially in spring/fall. Forecast uncertainty is moderate. Good liquidity.

- **Miami:** Subtropical — smallest daily variation, most stable forecasts (low sigma). Markets tend to be tighter but the edge is correspondingly smaller.

- **Seattle:** Marine climate — narrow temperature range but cloud cover and maritime influence create forecast challenges. Higher sigma for highs due to cloud cover effects.

- **Dallas/Atlanta:** Good liquidity with moderate forecast difficulty. Summer forecasts are very stable (high confidence), but severe weather days can create large errors.

### Monitoring & Maintenance

- **Check logs regularly** — Look for `STOP-LOSS`, `EXIT`, and `Trade failed` messages. Frequent stop-losses may indicate the sigma is too narrow or the entry threshold is too low.

- **Re-calibrate periodically** — Run calibration quarterly or when you notice Brier scores degrading. Weather model accuracy varies seasonally and improves as NWP models are updated.

- **Monitor Brier score** — Use `python3 -m bot --calibration` to check prediction quality. A degrading Brier score signals that your probability model needs recalibration.

- **State file health** — The state file (`weather_state.json`) is written atomically and locked for concurrent access. If it becomes corrupted (rare), simply delete it — the bot will start fresh with no open positions.

- **Daemon health check** — `python3 -m bot --health` checks PID file and heartbeat. If the heartbeat is stale (>5 minutes), the daemon may be stuck.

### What to Avoid

- **Don't disable safeguards in live mode** — The `--no-safeguards` flag exists for testing. In production, slippage checks and time-to-resolution guards prevent bad trades.

- **Don't trade without calibration on new locations** — Hardcoded defaults are averages. If Polymarket adds new cities, calibrate on historical data before trading there.

- **Don't set `kelly_fraction` > 0.5** — Higher Kelly fractions increase theoretical returns but also dramatically increase drawdown variance. Quarter-Kelly (0.25) is the recommended ceiling.

- **Don't ignore the model spread** — When GFS and ECMWF disagree by >5°F, forecast uncertainty is much higher than sigma alone suggests. The bot logs model spread for this reason.

- **Don't run multiple bot instances on the same state file** — The file lock prevents corruption, but two instances fighting for the lock will cause skipped runs. Use separate state files if you need parallel instances.

---

## Testing

```bash
# Run all 380 tests
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
| `weather/` | ~254 tests | Strategy, bridge, paper bridge, NOAA, Open-Meteo, aviation/METAR, probability, calibration, backtesting, sizing, state, parsing |
| `bot/` | ~80 tests | Scanner, signals, scoring, sizing, daemon, Gamma API |
| `polymarket/` | ~46 tests | Order signing, HMAC auth, client REST, fill detection |

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
│   ├── calibrate.py         # Calibration script (empirical sigma, model weights)
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
│       ├── test_calibrate.py
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

# Normal CDF probability
P = phi((high + 0.5 - forecast) / sigma) - phi((low - 0.5 - forecast) / sigma)
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

### 5. Hybrid Calibration

```python
# 1. Compute base sigma from real forecast errors
errors = [forecast - actual for each (target_date, model, metric)]
base_sigma = stddev(errors)  # e.g., 1.96°F

# 2. Expand to all horizons using NWP growth model
growth = {0: 1.00, 1: 1.33, 2: 1.67, 3: 2.00, ..., 10: 6.00}
sigma[h] = base_sigma * growth[h]

# 3. Per-location and seasonal adjustments are computed similarly
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
