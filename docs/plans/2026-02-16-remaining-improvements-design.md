# Remaining Improvements Design

Date: 2026-02-16

## Context

Most items from `improve.md` are already implemented (10/15). This design covers the 4 remaining items (notifications excluded per user preference).

## 1. Numerical Robustness Tests

**File:** `weather/tests/test_probability.py`

New class `TestNumericalRobustness`:
- `_student_t_cdf(±1e10, df=10)` → ~1.0 / ~0.0, no crash
- `_regularized_incomplete_beta(0.5, 1e-10, 1.0)` and `(0.5, 1.0, 1e-10)` → no crash
- `_regularized_incomplete_beta(1e-300, 5, 5)` → ~0; `(1-1e-15, 5, 5)` → ~1
- `_student_t_cdf(0, df=0.5)` → 0.5; `(1.96, df=1000)` → ~0.975
- Continued fraction non-convergence warning logged

Fix any bugs discovered.

## 2. TTL Cache for Open-Meteo

**File:** `weather/open_meteo.py`

```
_forecast_cache: dict[str, tuple[dict, float]] = {}
_CACHE_TTL = 900  # 15 minutes
```

- `_cache_key(latitudes, longitudes, timezone) -> str`
- `_get_cached(key) -> dict | None` — returns data if age < TTL
- `_set_cache(key, data)` — stores with current timestamp
- Integrated into `get_open_meteo_forecast_multi()` before HTTP fetch
- Log INFO on hit, DEBUG on miss

**Tests** (`weather/tests/test_open_meteo.py`):
- Cache hit returns same data without refetch
- Cache expires after TTL
- Different coordinates produce different keys

## 3. Inter-Location Correlation

### 3a. Calibration (calibrate.py)

New function `_compute_correlation_matrix(locations, errors_by_location)`:
- Pearson correlation of forecast errors between all location pairs
- Grouped by season (DJF, MAM, JJA, SON)
- Output: `{"correlation_matrix": {"NYC|Chicago": {"DJF": 0.72, ...}, ...}}`
- Stored in `calibration.json`

### 3b. Loading (probability.py)

- Load matrix in `_load_calibration()`
- Expose `get_correlation(loc1, loc2, month) -> float`

### 3c. Strategy integration (strategy.py)

- Before sizing, check open positions on other locations
- If correlation > `correlation_threshold`: reduce sizing
- Formula: `adjusted_size = base_size * (1 - max_corr * correlation_discount)`

### 3d. Config (config.py)

```python
correlation_threshold: float = 0.5
correlation_discount: float = 0.5
```

### Tests
- Pearson on synthetic data
- Sizing reduced when correlation exceeds threshold
- Correlation = 0 does not affect sizing

## 4. Enriched CLI Report

**File:** `weather/report.py`

### 4a. Live positions table
- Location, bucket, side, entry price, current price, unrealized P&L, time remaining
- ANSI colors: green for profit, red for loss

### 4b. Metrics summary
- 7-day rolling Brier, win rate, Sharpe, average edge

### 4c. Calibration drift
- `calibration.json` age, sigma drift vs observed (7 days)
- Warning if drift > 20%

### 4d. --watch mode
- `--watch N` refreshes every N seconds
- `os.system('clear')` + reprint, Ctrl+C to quit

### Tests
- Report generates with empty data
- P&L formatting positive/negative
- Watch mode first cycle no crash
