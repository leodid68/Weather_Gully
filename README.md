# Weather Gully

Automated weather trading bot for Polymarket — trades temperature markets using NOAA + Open-Meteo ensemble forecasts via CLOB direct (no SDK dependency).

## Architecture

```
Weather_Gully/
├── polymarket/      # CLOB client (EIP-712 signing, HMAC L2 auth, WebSocket)
├── bot/             # General trading bot (signals, Kelly sizing, multi-choice arb, daemon)
└── weather/         # Weather strategy (NOAA/Open-Meteo forecasts, bucket scoring, safeguards)
```

**Three packages, one pipeline:**

- `polymarket/` — Low-level Polymarket CLOB client. Order construction, EIP-712 signing, L2 HMAC authentication, REST + WebSocket.
- `bot/` — General-purpose trading bot. 4 signal methods (longshot bias, arbitrage, microstructure, multi-choice arb), Kelly sizing, Gamma API market discovery, daemon mode, calibration scoring.
- `weather/` — Weather-specific strategy. NOAA + Open-Meteo multi-model ensemble forecasts, normal CDF bucket probability, seasonal adjustments, dynamic exits, correlation guard, stop-loss on forecast reversal.

The `CLOBWeatherBridge` adapter connects the weather strategy to the CLOB + Gamma APIs.

## Usage

### Weather bot (standalone)

```bash
# Dry-run — fetch markets and show opportunities
python -m weather

# All 6 locations
python -m weather --set locations=NYC,Chicago,Seattle,Atlanta,Dallas,Miami

# Live trading
python -m weather --live

# Verbose
python -m weather --verbose
```

### General bot

```bash
# Scan all markets
python -m bot --scan

# Scan weather markets specifically
python -m bot --weather

# Show detected signals
python -m bot --signals

# Dry-run strategy
python -m bot

# Live trading
python -m bot --live

# Daemon mode (continuous)
python -m bot --daemon --live
```

### CLOB client

```bash
# List markets
python -m polymarket markets

# Orderbook
python -m polymarket book <token_id>

# Price
python -m polymarket price <token_id>
```

## Configuration

Config is loaded with priority: `config.json` > environment variables > defaults.

Key environment variables:
- `POLY_PRIVATE_KEY` — Ethereum private key for CLOB authentication (required for `--live`)

Key config fields (set via `--set key=value`):
- `entry_threshold` — Minimum edge to enter (default: 0.15 for weather, 0.10 for bot)
- `max_position_usd` — Max USD per position (default: $2.00 weather, $5.00 bot)
- `kelly_fraction` — Kelly criterion fraction (default: 0.25)
- `locations` — Comma-separated cities: NYC, Chicago, Seattle, Atlanta, Dallas, Miami

## Supported locations

| City | Station | Coordinates |
|------|---------|-------------|
| NYC | LaGuardia (LGA) | 40.78, -73.87 |
| Chicago | O'Hare (ORD) | 41.97, -87.91 |
| Seattle | Sea-Tac (SEA) | 47.45, -122.31 |
| Atlanta | Hartsfield (ATL) | 33.64, -84.43 |
| Dallas | DFW | 32.90, -97.04 |
| Miami | MIA | 25.80, -80.29 |

## Tests

```bash
# All tests
python -m pytest weather/tests/ bot/tests/ polymarket/tests/ -v

# Weather only
python -m pytest weather/tests/ -v

# Bot only
python -m pytest bot/tests/ -v
```

295 tests covering: bridge adapter, strategy integration, NOAA parsing, probability estimation, Kelly sizing, signal detection, order signing, HMAC authentication.

## Dependencies

- `httpx` — HTTP client (Gamma API, CLOB REST)
- `eth-account` — EIP-712 order signing
- `py-clob-client` utilities — Order struct, CTF exchange
- `certifi` — SSL certificates (macOS)

## How it works

1. **Fetch markets** — Gamma API returns active weather markets (temperature buckets per city/date)
2. **Forecast** — NOAA + Open-Meteo ensemble provides weighted temperature forecast with confidence spread
3. **Score buckets** — Normal CDF estimates probability for each temperature range
4. **Find edge** — Compare our probability vs market price; filter by EV threshold
5. **Safeguards** — Check slippage, time to resolution, flip-flop discipline
6. **Size** — Fractional Kelly criterion with position caps
7. **Trade** — CLOB limit order at best ask/bid (`neg_risk=True` for weather markets)
8. **Monitor** — Dynamic exit thresholds, stop-loss on forecast reversal, correlation guard
