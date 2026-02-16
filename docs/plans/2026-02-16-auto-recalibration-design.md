# Auto-Recalibration Pipeline — Design

## Goal

Replace the static `calibration.json` (one-shot snapshot of 2025) with a self-updating pipeline that recalibrates all model parameters weekly on a 90-day sliding window with exponential weighting, so the model adapts to changing weather patterns and model performance.

## Architecture

```
cron (weekly, Sunday 06:00)
  python -m weather.recalibrate
       │
       ▼
  error_history.json  ◄──  METAR (IEM) + Open-Meteo forecasts
  (incremental cache)       (fetch only new days)
       │
       ▼  90-day window + exponential weighting (half-life 30 days)
  Full recomputation:
    • base sigma + horizon expansion
    • seasonal factors (weighted)
    • model weights GFS/ECMWF (weighted RMSE)
    • Platt scaling (a, b)
    • adaptive factors (spread→sigma, ema→sigma)
    • parameter clamps (guard rails)
       │
       ▼  atomic write (tmpfile + os.rename)
  calibration.json  ──►  Bot hot-reloads via mtime check
       │
       ▼
  recalibration_log/2026-02-16.json  (audit trail)
```

The bot requires zero changes — it already hot-reloads `calibration.json` via mtime-based cache invalidation in `_load_calibration()`.

## Incremental Error Cache

### Format: `weather/error_history.json`

```json
{
  "errors": [
    {
      "location": "NYC", "target_date": "2026-02-10", "month": 2,
      "metric": "high", "model": "gfs",
      "forecast": 42.0, "actual": 43.2, "error": -1.2,
      "model_spread": 1.5
    }
  ],
  "last_fetched": {
    "NYC": "2026-02-10",
    "Chicago": "2026-02-10"
  }
}
```

### Incremental Fetch Logic

1. Read `last_fetched[location]` from cache
2. Fetch METAR actuals + Open-Meteo forecasts from `last_fetched + 1` to `today - 2` (2-day buffer for incomplete METAR data)
3. Compute errors, append to cache
4. Update `last_fetched`
5. Prune errors older than 365 days (keep 1 year for audit, even though window is 90 days)

### First Run Bootstrap

If `error_history.json` doesn't exist, fetch the full 90-day window from scratch (~30s). Subsequent runs fetch only the delta (~5-10s).

## Exponential Weighting

Each error record receives weight: `w(t) = exp(-ln(2) * age_days / 30)`

| Age | Weight |
|-----|--------|
| 0 days | 1.00 |
| 30 days | 0.50 |
| 60 days | 0.25 |
| 90 days | 0.125 |

This weight is applied to **all** computations:
- **Weighted sigma**: `sqrt(sum(w * error^2) / sum(w))`
- **Weighted RMSE** for model weights: `sqrt(sum(w * (forecast - actual)^2) / sum(w))`
- **Weighted Platt**: gradient descent with sample weights in the loss function
- **Weighted seasonal factors**: `weighted_monthly_sigma / weighted_global_sigma`

## Parameter Guard Rails

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| `base_sigma` | 1.0 | 4.0 | Physical bounds for day-0 forecast error (°F) |
| `seasonal_factor` | 0.5 | 2.0 | Beyond 2x signals data problem |
| `model_weight` (GFS/ECMWF) | 0.15 | 0.70 | No single model dominates that much; NOAA fixed at 0.20 |
| `platt_a` | 0.3 | 2.0 | Outside → fundamental model breakdown |
| `platt_b` | -1.0 | 1.0 | Extreme offset → structural problem |
| `spread_to_sigma` | 0.3 | 1.5 | Physical bounds |
| `ema_to_sigma` | 0.5 | 2.0 | Physical bounds |

If any parameter hits a clamp, a warning is logged in the recalibration log with the unclamped value.

## Minimum Sample Threshold

If the 90-day window has < 100 effective weighted samples, the recalibration **does not replace** `calibration.json`. It logs a warning and exits. This prevents bad calibration on insufficient data (e.g., bot just started, or API outage).

## Recalibration Log

Each run produces `weather/recalibration_log/YYYY-MM-DD.json`:

```json
{
  "timestamp": "2026-02-16T06:00:12Z",
  "window": {"start": "2025-11-18", "end": "2026-02-14"},
  "samples_total": 540,
  "samples_effective": 312.5,
  "params": {
    "base_sigma": 1.84,
    "platt": {"a": 0.73, "b": 0.32},
    "model_weights": {"NYC": {"gfs": 0.51, "ecmwf": 0.29, "noaa": 0.20}},
    "seasonal_factors": {"1": 1.08, "2": 1.02},
    "adaptive_sigma": {"spread_to_sigma": 0.78, "ema_to_sigma": 1.04}
  },
  "clamped": [],
  "delta_from_previous": {
    "base_sigma": -0.02,
    "platt_a": +0.01
  },
  "fetch_stats": {
    "new_errors": 42,
    "fetch_time_s": 8.3
  }
}
```

## Components

### New files
- `weather/recalibrate.py` — CLI entry point + orchestration
- `weather/error_cache.py` — Incremental error cache management
- `weather/tests/test_recalibrate.py` — Tests
- `weather/tests/test_error_cache.py` — Tests

### Modified files
- `weather/calibrate.py` — Extract weighted variants of existing functions (weighted sigma, weighted RMSE, weighted Platt)
- `.gitignore` — Add `error_history.json`, `recalibration_log/`

### Unchanged
- `weather/probability.py` — Already hot-reloads calibration.json
- `weather/strategy.py` — No changes needed
- `weather/config.py` — No changes needed (cron is external)

## Cron Setup (macOS launchd)

```bash
# Add to crontab:
0 6 * * 0 cd /Users/leodidier/Weather_Gully && python3 -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas 2>> weather/recalibration.log
```

## Success Criteria

1. `python3 -m weather.recalibrate` produces a valid `calibration.json` with all params within clamps
2. Incremental fetch adds only new days (verified by `last_fetched` timestamps)
3. Exponential weighting verified: same data with different ages produces different params
4. Guard rails prevent drift: artificially extreme errors don't push params past clamps
5. Minimum sample threshold prevents calibration on insufficient data
6. Bot hot-reloads new params without restart
7. Full test suite passes after recalibration
