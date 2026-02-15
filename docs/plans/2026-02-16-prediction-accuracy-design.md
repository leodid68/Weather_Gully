# Prediction Accuracy & Reliability Improvements

**Date**: 2026-02-16
**Scope**: Quick fixes + Platt scaling + METAR ground truth
**Approach**: A (focused high-impact, ~2-3 sessions)

## Problem Statement

Audit of the weather prediction pipeline revealed 5 critical issues:

1. **Calibration curve bias** — backtest shows systematic overconfidence for probabilities < 50% and underconfidence > 65%
2. **ERA5 vs METAR mismatch** — calibration uses ERA5 reanalysis (31km grid) but Polymarket resolves on METAR station data. Potential 0.5-1.5°F systematic bias.
3. **Fill verification disabled** — `fill_timeout=0.0` means trades are counted as executed even if the order never matches
4. **UTC/local bug in aviation.py** — `compute_daily_extremes()` compares UTC obs_time with local target_date
5. **Dead code** — `get_price_history()` always returns `[]`, all trend detection is inactive

## Design

### Part 1: Quick Fixes (Reliability)

#### 1a. Fill timeout
- In `strategy.py`, pass `fill_timeout=config.fill_timeout_seconds` to `client.execute_trade()`
- Config already has `fill_timeout_seconds = 30.0`

#### 1b. UTC/local bug
- In `aviation.py:compute_daily_extremes()`, convert `obs_time` to local timezone before comparing with `target_date`
- Use the location's `tz_name` for conversion

#### 1c. Dead code removal
- Remove `detect_price_trend()` from `strategy.py`
- Remove `get_price_history()` from `bridge.py`
- Remove any callers/references

### Part 2: Platt Scaling

**Principle**: post-hoc sigmoid correction on raw probabilities.

```
prob_calibrated = 1 / (1 + exp(-(a * logit(prob_raw) + b)))
```

Where `a` and `b` minimize log-loss on backtest (predicted, actual) pairs.

**Implementation**:
- New function `platt_calibrate(prob: float) -> float` in `probability.py`
- Parameters `a`, `b` stored in `calibration.json` under `"platt_scaling": {"a": ..., "b": ...}`
- Computed in `calibrate.py` via `_compute_platt_params(predictions, actuals)`
- Applied in `score_buckets()` before EV calculation (raw proba stays in logs for debug)
- Fallback: if params absent from calibration.json, return raw proba
- Output bounded to [0.01, 0.99]

### Part 3: METAR Historical Ground Truth

**Data source**: Iowa Environmental Mesonet (IEM) — free public METAR archive.

**Implementation**:
- New function `get_historical_metar_actuals(station, start_date, end_date)` in `historical.py`
- Parses IEM CSV, extracts daily high/low in °F
- Station mapping in `config.py`:
  - NYC → KLGA, Chicago → KORD, Miami → KMIA
  - Seattle → KSEA, Atlanta → KATL, Dallas → KDFW

**Calibrate.py changes**:
- New CLI option `--actuals-source metar|era5` (default: `metar`)
- When `metar`, calls `get_historical_metar_actuals()` instead of `get_historical_actuals()`
- ERA5 remains as fallback

### Part 4: Re-calibration & Validation

1. Run calibration with METAR: `python3 -m weather.calibrate --actuals-source metar ...`
2. New `calibration.json` includes: METAR-calibrated sigma + platt_scaling params + adaptive_sigma
3. Run backtest, compare Brier score before/after

**Success criteria**:
- Platt params produce a reasonable monotone correction
- Brier score improved vs baseline
- `spread_to_sigma_factor` and `ema_to_sigma_factor` remain in sane ranges with METAR ground truth

## Out of Scope (Future)

- Student-t distribution (add later if tail performance still poor)
- Horizon-dependent stop-loss (Approach B)
- Train/validation split (Approach C)
- Portfolio-level correlation guard
- Automatic recalibration
