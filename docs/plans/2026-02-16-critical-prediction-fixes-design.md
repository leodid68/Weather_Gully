# Critical Prediction Fixes — Design

## Goal

Eliminate the 3 critical flaws in the prediction pipeline:
1. Replace fictitious horizon growth model with real RMSE by horizon from Open-Meteo Previous Runs API
2. Fix backtest lookahead bias by using horizon-specific historical forecasts
3. Test Gaussian assumption and switch to Student's t-distribution if fat tails detected

## Data Source: Open-Meteo Previous Runs API

**Base URL:** `https://previous-runs-api.open-meteo.com/v1/forecast`

**Key capability:** Returns `temperature_2m_previous_dayN` (N=1-7) — the forecast that was issued N days before each date. This gives us real horizon-dependent forecasts.

**Verified:** API returns different values for different horizons (not deduplicated like Historical Forecast API). Data available from January 2024 onwards for both GFS and ECMWF. No nulls observed.

**Limitation:** Only hourly variables available (no daily max/min aggregates). We compute daily max/min from the 24h hourly data ourselves.

## Component 1: Previous Runs Client (`weather/previous_runs.py`)

New module that fetches hourly forecasts from the Previous Runs API at specified horizons.

```python
def fetch_previous_runs(
    lat: float, lon: float,
    start_date: str, end_date: str,
    horizons: list[int] = [0, 1, 2, 3, 5, 7],
    models: list[str] = ["gfs_seamless", "ecmwf_ifs025"],
    tz_name: str = "America/New_York",
) -> dict[int, dict[str, dict]]:
    """Fetch forecasts at multiple horizons.

    Returns: {horizon: {date_str: {gfs_high, gfs_low, ecmwf_high, ecmwf_low}}}
    """
```

- Chunks requests to 90-day windows (API limit)
- Computes daily max/min from hourly data using timezone-aware boundaries
- Uses `temperature_unit=fahrenheit`
- Shared SSL context from `weather/_ssl.py`
- Rate limiting: 1s delay between chunks

## Component 2: Horizon-Real Calibration

Modify `weather/calibrate.py` to compute RMSE at each horizon directly instead of using the growth model.

**Current (broken):**
```
base_sigma = RMSE(horizon_0_errors)
sigma[h] = base_sigma * hardcoded_growth[h]  # fictitious
```

**New:**
```
for horizon in [0, 1, 2, 3, 5, 7]:
    sigma[horizon] = RMSE(errors_at_horizon_h)  # real data
# Interpolate for horizons 4, 6, 8, 9, 10
```

The error records gain a `horizon` field. `compute_forecast_errors()` accepts forecasts from `fetch_previous_runs()` instead of the deduplicated Historical Forecast API.

`_expand_sigma_by_horizon()` is replaced by a lookup table of real sigmas, with linear interpolation for missing horizons (4, 6, 8-10) and extrapolation beyond horizon 7 using the observed growth rate.

## Component 3: Backtest Without Lookahead

Modify `weather/backtest.py` to use `fetch_previous_runs()` instead of `get_historical_forecasts()`.

**Current (lookahead):**
```python
forecasts = get_historical_forecasts(lat, lon, start, end)
# Returns same-day forecast for ALL dates — horizon 0 always
```

**New:**
```python
forecasts = fetch_previous_runs(lat, lon, start, end, horizons=[horizon])
# Returns the forecast that was issued `horizon` days before each date
```

The rest of the backtest logic stays the same. The simulated market price is still synthetic but now the FORECASTS are realistic — we're using what the model actually predicted N days before.

## Component 4: Distribution Test + Student's t

### Step 1: Test normality of forecast errors

Add `_test_normality(errors)` to `weather/calibrate.py` that runs a Jarque-Bera test on the residuals. If p < 0.05, the normal assumption is rejected.

Jarque-Bera test statistic (stdlib-only):
```python
n = len(errors)
skewness = sum((e/sigma)**3 for e in errors) / n
kurtosis = sum((e/sigma)**4 for e in errors) / n
jb = (n/6) * (skewness**2 + (kurtosis - 3)**2 / 4)
# JB > 5.99 → reject normality at 5% level (chi-squared, df=2)
```

### Step 2: Fit Student's t degrees of freedom

If normality rejected, fit `df` parameter by maximum likelihood:
```python
# df that minimizes negative log-likelihood of errors under Student's t
# Grid search over df = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
```

### Step 3: Replace CDF in probability.py

Add `_student_t_cdf(x, df)` using the regularized incomplete beta function (implementable with stdlib `math`).

In `estimate_bucket_probability()`:
```python
if cal.get("distribution") == "student_t":
    df = cal["student_t_df"]
    cdf_fn = lambda x: _student_t_cdf(x, df)
else:
    cdf_fn = _normal_cdf
```

The calibration.json gains two new fields:
- `"distribution": "normal" | "student_t"`
- `"student_t_df": float` (only when distribution is student_t)

## Success Criteria

1. Previous Runs API client tested with real data from 2024-2025
2. Horizon-dependent sigma calibrated from real RMSE (not growth model)
3. Backtest uses horizon-specific forecasts (no lookahead)
4. Normality test run on error distribution
5. If fat tails: Student's t-CDF with calibrated df replaces normal CDF
6. Full test suite passes
7. Recalibration pipeline updated to use new horizon data
