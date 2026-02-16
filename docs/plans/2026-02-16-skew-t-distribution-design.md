# Skewed Student-t Distribution Upgrade

## Problem

Current calibration data (2136 errors, 6 US cities, 90-day window) shows:
- **Excess kurtosis = 4.822** — very heavy tails (normal = 0)
- **Skewness = -1.101** — significant left skew (cold surprises > hot surprises)
- **Student-t(10) kurtosis = 1.0** — captures only ~20% of the observed tail weight
- Student-t ignores skewness entirely

Temperature forecast errors are both heavy-tailed AND asymmetric. The current Student-t(10) handles the first but not the second.

## Solution: Fernandez-Steel Skew-t

Replace the symmetric Student-t CDF with the Fernandez-Steel skewed Student-t, which adds a single skewness parameter gamma:

- gamma = 1.0: symmetric (reduces to standard Student-t)
- gamma < 1.0: left-skewed (heavier left tail — cold surprises)
- gamma > 1.0: right-skewed (heavier right tail — hot surprises)

**CDF formula** (based on existing `_student_t_cdf`):

```
skew_t_cdf(x, df, gamma) =
    if x < 0: (2 * gamma^2 / (1 + gamma^2)) * T(x * gamma, df)
    if x >= 0: (2 / (1 + gamma^2)) * T(x / gamma, df) - (1 - gamma^2)/(1 + gamma^2) + 1 - 1/(1+gamma^2)
```

Where T is the standard Student-t CDF (already implemented).

## Architecture

### Phase 1: Validation + Global Skew-t

1. **Validation script** (`weather/distribution_validation.py`):
   - Loads historical errors from error cache
   - Computes bucket probabilities with 4 distributions: Normal, Student-t(10), Student-t(df_best), Skew-t(df, gamma)
   - Scores each against actual resolutions via Brier score + log-loss
   - Outputs comparison table with recommendation

2. **Skew-t CDF** (in `weather/probability.py`):
   - `_skew_t_cdf(x, df, gamma)` — pure stdlib, delegates to `_student_t_cdf`
   - Plugs into existing CDF switch in `estimate_bucket_probability()` and `estimate_bucket_probability_with_obs()`

3. **Calibration fit** (in `weather/calibrate.py`):
   - `_fit_skew_t_params(errors)` — 2D grid search (df x gamma) by MLE
   - df grid: {2, 3, 4, 5, 7, 10, 15, 20, 30, 50}
   - gamma grid: {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5}
   - Output: `"distribution": "skew_t"`, `"student_t_df"`, `"skew_t_gamma"` in calibration.json

4. **Integration** — 3-line change in CDF switch:
   ```python
   if dist == "skew_t":
       cdf_fn = lambda z: _skew_t_cdf(z, t_df, gamma)
   ```

### Phase 2: Full-Year Recalibration (future)

- Run `python3 -m weather.calibrate` on all 14 cities, Jan-Dec 2025
- ~700+ errors per city
- Enables per-city (df, gamma) fitting in Phase 3

### Phase 3: Per-City Skew-t (future)

- Fit (df, gamma) per location if >200 errors available
- Store in `calibration.json` under `"per_location_distribution"`
- Fallback to global (df, gamma) if insufficient data

## Constraints

- 100% stdlib (no scipy, no numpy)
- Skew-t CDF computes via existing `_student_t_cdf` + arithmetic
- No breaking changes to existing API (`estimate_bucket_probability` signature unchanged)
- Guard rails: gamma clamped to [0.3, 3.0], df clamped to [2, 100]

## Expected Impact

With skewness = -1.1:
- Left-tail buckets ("44F or below") get ~5-15% more probability mass
- Right-tail buckets ("60F or above") get ~5-10% less probability mass
- Center buckets change by ~1-3%
- Net effect: more accurate pricing on extreme temperature events

## Files Modified

| File | Change |
|------|--------|
| `weather/probability.py` | Add `_skew_t_cdf()`, update CDF switch |
| `weather/calibrate.py` | Add `_skew_t_logpdf()`, `_fit_skew_t_params()` |
| `weather/distribution_validation.py` | New: validation CLI script |
| `weather/calibration.json` | New fields: `skew_t_gamma` |
| `weather/tests/test_probability.py` | Tests for `_skew_t_cdf` |
| `weather/tests/test_distribution_validation.py` | Tests for validation logic |
