# Design: Adaptive Sigma from Ensemble Spread

**Date:** 2026-02-15
**Status:** Approved
**Problem:** Sigma is too narrow (overconfident) and not reactive to real-time uncertainty signals.

## Approach

Replace the theoretical NWP horizon growth model with empirical sigma from ensemble member spread. Use `max()` of three independent signals so the most pessimistic estimate always wins.

## Three Sigma Signals

### Signal 1 — Ensemble Spread (primary)

New module `weather/ensemble.py` fetches daily `temperature_2m_max/min` from the Open-Meteo Ensemble API (`https://ensemble-api.open-meteo.com/v1/ensemble`):
- ECMWF IFS 0.25 (51 members, 15 days)
- GFS 0.25 (31 members, 10 days)

Conversion: `sigma_ensemble = ensemble_stddev * underdispersion_factor`

Cache: JSON files in `weather/cache/ensemble/` keyed by `{lat}_{lon}_{date}_{metric}.json`, TTL 6 hours.

### Signal 2 — Model Spread (GFS vs ECMWF deterministic)

Already computed in `compute_ensemble_forecast()` but currently unused.

Conversion: `sigma_spread = |gfs - ecmwf| * spread_to_sigma_factor`

### Signal 3 — EMA of Past Errors

Already tracked in `feedback.py` as `abs_error_ema` but never fed back into sigma.

Conversion: `sigma_ema = abs_error_ema * ema_to_sigma_factor`

## Combining Signals

```python
sigma_final = max(sigma_ensemble, sigma_spread, sigma_ema, sigma_floor)
```

Where `sigma_floor = base_sigma(horizon, location) * seasonal_factor(month, location)` from the existing calibration. The calibrated sigma serves as a floor, never a ceiling.

## Calibrating Conversion Factors

### Initial values (NWP literature fallbacks)
- `underdispersion_factor` = 1.3
- `spread_to_sigma_factor` = 0.7
- `ema_to_sigma_factor` = 1.25

### Empirical calibration procedure
After 30+ days of data collection:
1. Log `(ensemble_stddev, model_spread, abs_error_ema, actual_error)` per forecast
2. For each signal: `factor = sqrt(mean(actual_error^2) / mean(signal^2))`
3. Store calibrated factors in `calibration.json` under `adaptive_sigma` key

### Logging
Each forecast logs all signals to `weather/sigma_log.json` for periodic recalibration.

## Integration

### `weather/strategy.py`
```python
ensemble_temp, model_spread = compute_ensemble_forecast(...)
ensemble_result = fetch_ensemble_spread(lat, lon, target_date, metric)
ema_error = get_feedback_ema(location, metric)

adaptive_sigma = compute_adaptive_sigma(
    ensemble_result, model_spread, ema_error,
    forecast_date, location, month
)

probs = score_buckets(ensemble_temp, buckets, forecast_date, ...,
                      sigma_override=adaptive_sigma)
```

### `weather/probability.py`
- `estimate_bucket_probability()` gains `sigma_override: float | None` parameter
- When provided, replaces all internal sigma calculation (horizon + seasonal + weather multiplier)
- `_weather_sigma_multiplier()` is removed — ensemble captures weather-regime uncertainty natively

### `weather/feedback.py`
- Expose `get_feedback_ema(location, metric) -> float`

## Fallback Chain

1. Ensemble spread (API call) — may fail
2. Model spread (deterministic GFS vs ECMWF) — always available
3. EMA of errors (feedback history) — available if history exists
4. Calibrated sigma (calibration.json) — always available as floor

If the ensemble API times out (>5s), continue with signals 2-4. Log a warning when any signal is missing.

## Tests

- Unit tests for each signal in isolation
- Integration tests for all 8 combinations of missing signals
- Property: `sigma_final >= sigma_floor` always holds
- Mock ensemble API for all tests
- End-to-end test through `score_buckets()` with `sigma_override`
