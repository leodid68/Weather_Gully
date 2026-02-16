"""Calibration script — compute empirical sigma and model weights from historical data.

Uses a **hybrid approach** because Open-Meteo's free previous-runs API does not
store historical model runs (all queries return the latest forecast, regardless
of when queried). This means we cannot directly observe how forecast error grows
with horizon from the API alone.

Strategy:
  1. Fetch archived forecasts and ERA5 actuals for a date range.
  2. Deduplicate errors — one error per (target_date, model, metric).
  3. Compute the **base sigma** (empirical stddev of model error at horizon 0).
  4. Apply a **horizon growth model** derived from NWP verification literature
     to generate sigma for all horizons 0-10.
  5. Compute per-location, per-season factors and model weights.

Usage::

    python -m weather.calibrate --locations NYC --start-date 2025-01-01 --end-date 2026-01-01
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import LOCATIONS
from .historical import get_historical_actuals, get_historical_forecasts, get_historical_metar_actuals

logger = logging.getLogger(__name__)

# Output path for calibration data
_DEFAULT_OUTPUT = str(Path(__file__).parent / "calibration.json")

# ---------------------------------------------------------------------------
# Horizon growth model
# ---------------------------------------------------------------------------
# NWP forecast error grows with horizon. These growth factors are derived from
# the original hardcoded _HORIZON_STDDEV table in probability.py, which was
# based on NOAA/NWS model verification statistics:
#   h=0 → 1.5°F, h=5 → 4.0°F, h=10 → 9.0°F
# Ratios (relative to h=0): linear at 0.5°F/day for h≤5, then 1.0°F/day for h>5.
#
# sigma(h) = base_sigma * _HORIZON_GROWTH[h]
_HORIZON_GROWTH = {
    0: 1.00,
    1: 1.33,
    2: 1.67,
    3: 2.00,
    4: 2.33,
    5: 2.67,
    6: 3.33,
    7: 4.00,
    8: 4.67,
    9: 5.33,
    10: 6.00,
}


def _horizon_growth_factor(horizon: int) -> float:
    """Growth factor for a given horizon (days ahead).

    For horizons beyond 10, extrapolates linearly at the day 6-10 rate.
    """
    if horizon in _HORIZON_GROWTH:
        return _HORIZON_GROWTH[horizon]
    return 6.00 + 0.67 * (horizon - 10)


def compute_forecast_errors(
    location: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
) -> list[dict]:
    """Compute **deduplicated** forecast errors comparing archived forecasts to ERA5.

    Because Open-Meteo's free API returns the same forecast for a given target
    date regardless of when it was queried, we deduplicate by
    ``(target_date, model, metric)`` to avoid counting the same error multiple
    times across different "run dates".

    Each record gets ``horizon=0`` since we only have one forecast per target
    date.  The horizon growth is applied later in ``build_calibration_tables``.

    Additionally computes model spread (|GFS - ECMWF|) per target date for
    uncertainty analysis.

    Returns a list of error records::

        [
            {
                "location": "NYC",
                "target_date": "2025-01-16",
                "month": 1,
                "metric": "high",
                "model": "gfs",
                "forecast": 42.0,
                "actual": 43.2,
                "error": -1.2,
                "model_spread": 1.5,
            },
            ...
        ]
    """
    logger.info("Fetching historical forecasts for %s (%s to %s)...",
                location, start_date, end_date)
    forecasts = get_historical_forecasts(lat, lon, start_date, end_date, tz_name=tz_name)

    logger.info("Fetching ERA5 actuals for %s...", location)
    actuals = get_historical_actuals(lat, lon, start_date, end_date, tz_name=tz_name)

    # Collect all forecasts per target date, deduplicating across run dates.
    # Key: (target_date, model_prefix, metric)
    seen: dict[tuple[str, str, str], dict] = {}

    for _run_date_str, targets in forecasts.items():
        for target_date_str, model_data in targets.items():
            actual = actuals.get(target_date_str)
            if not actual:
                continue

            month = datetime.strptime(target_date_str, "%Y-%m-%d").month

            # Compute model spread for this target date
            spread: dict[str, float] = {}
            for metric in ["high", "low"]:
                gfs_val = model_data.get(f"gfs_{metric}")
                ecmwf_val = model_data.get(f"ecmwf_{metric}")
                if gfs_val is not None and ecmwf_val is not None:
                    spread[metric] = abs(gfs_val - ecmwf_val)

            for model_prefix in ["gfs", "ecmwf"]:
                for metric in ["high", "low"]:
                    key = (target_date_str, model_prefix, metric)
                    if key in seen:
                        continue

                    forecast_key = f"{model_prefix}_{metric}"
                    forecast_val = model_data.get(forecast_key)
                    actual_val = actual.get(metric)

                    if forecast_val is None or actual_val is None:
                        continue

                    record = {
                        "location": location,
                        "target_date": target_date_str,
                        "month": month,
                        "metric": metric,
                        "model": model_prefix,
                        "forecast": forecast_val,
                        "actual": actual_val,
                        "error": forecast_val - actual_val,
                        "model_spread": spread.get(metric, 0.0),
                    }
                    seen[key] = record

    errors = list(seen.values())
    logger.info("Computed %d deduplicated forecast errors for %s", len(errors), location)
    return errors


def _compute_errors_with_metar(
    location: str,
    lat: float,
    lon: float,
    station: str,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
) -> list[dict]:
    """Compute deduplicated forecast errors using METAR actuals instead of ERA5.

    Same logic as ``compute_forecast_errors`` but fetches ground truth from
    Iowa Environmental Mesonet (IEM) ASOS station data, which matches the
    actual Polymarket resolution source.

    Returns the same error record format as ``compute_forecast_errors``.
    """
    logger.info("Fetching historical forecasts for %s (%s to %s)...",
                location, start_date, end_date)
    forecasts = get_historical_forecasts(lat, lon, start_date, end_date, tz_name=tz_name)

    logger.info("Fetching METAR actuals for %s (station %s)...", location, station)
    actuals = get_historical_metar_actuals(station, start_date, end_date, tz_name=tz_name)

    # Deduplicate: one error per (target_date, model, metric)
    seen: dict[tuple[str, str, str], dict] = {}

    for _run_date_str, targets in forecasts.items():
        for target_date_str, model_data in targets.items():
            actual = actuals.get(target_date_str)
            if not actual:
                continue

            month = datetime.strptime(target_date_str, "%Y-%m-%d").month

            # Compute model spread for this target date
            spread: dict[str, float] = {}
            for metric in ["high", "low"]:
                gfs_val = model_data.get(f"gfs_{metric}")
                ecmwf_val = model_data.get(f"ecmwf_{metric}")
                if gfs_val is not None and ecmwf_val is not None:
                    spread[metric] = abs(gfs_val - ecmwf_val)

            for model_prefix in ["gfs", "ecmwf"]:
                for metric in ["high", "low"]:
                    key = (target_date_str, model_prefix, metric)
                    if key in seen:
                        continue

                    forecast_key = f"{model_prefix}_{metric}"
                    forecast_val = model_data.get(forecast_key)
                    actual_val = actual.get(metric)

                    if forecast_val is None or actual_val is None:
                        continue

                    record = {
                        "location": location,
                        "target_date": target_date_str,
                        "month": month,
                        "metric": metric,
                        "model": model_prefix,
                        "forecast": forecast_val,
                        "actual": actual_val,
                        "error": forecast_val - actual_val,
                        "model_spread": spread.get(metric, 0.0),
                    }
                    seen[key] = record

    errors = list(seen.values())
    logger.info("Computed %d deduplicated forecast errors (METAR) for %s", len(errors), location)
    return errors


def compute_empirical_sigma(
    errors: list[dict],
    group_by: str = "month",
) -> dict[str, float]:
    """Compute standard deviation of forecast errors grouped by a key.

    Args:
        errors: List of error records from ``compute_forecast_errors``.
        group_by: Key to group by (``"month"``, ``"model"``, ``"location"``, etc.).

    Returns:
        Dict mapping group key to empirical sigma (stddev of errors).
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for err in errors:
        key = str(err.get(group_by, "unknown"))
        groups[key].append(err["error"])

    result: dict[str, float] = {}
    for key, errs in sorted(groups.items()):
        if len(errs) < 3:
            continue
        n = len(errs)
        mean = sum(errs) / n
        variance = sum((e - mean) ** 2 for e in errs) / (n - 1)  # Bessel's correction
        result[key] = round(math.sqrt(variance), 2)

    return result


def compute_model_weights(
    errors: list[dict],
    group_by: str = "location",
    grid_step: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Compute optimal model weights per group via 2-D grid search on MAE.

    Searches the full simplex ``(w_gfs, w_ecmwf, w_noaa)`` where each weight
    is in ``[0, 1]`` and they sum to 1.  Because NOAA has no historical
    forecast archive, its contribution is proxied by the model consensus
    (midpoint of GFS and ECMWF).  The ensemble is then::

        ensemble = w_gfs * gfs + w_ecmwf * ecmwf + w_noaa * (gfs + ecmwf) / 2

    This gives the NOAA axis real meaning: allocating weight to the proxy
    acts as *shrinkage towards the model consensus*, which can reduce
    variance when one model is noisy.

    Args:
        errors: List of error records (must contain ``"model"``, ``"forecast"``,
            ``"actual"``, ``"target_date"``, ``"metric"`` and the *group_by* key).
        group_by: Key to group by (typically ``"location"``).
        grid_step: Step size for each axis of the 2-D grid (default 0.01 = 1%).

    Returns:
        Dict mapping group key to ``{"gfs_seamless": w, "ecmwf_ifs025": w,
        "noaa": w}`` with weights summing to 1.0.
    """
    # Align GFS & ECMWF forecasts by (group, target_date, metric)
    aligned: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)

    for err in errors:
        group_key = str(err.get(group_by, "global"))
        obs_key = (err["target_date"], err["metric"])
        if obs_key not in aligned[group_key]:
            aligned[group_key][obs_key] = {"actual": err["actual"]}
        aligned[group_key][obs_key][err["model"]] = err["forecast"]

    results: dict[str, dict[str, float]] = {}

    steps = int(round(1.0 / grid_step)) + 1

    for group_key, observations in aligned.items():
        paired = [
            obs for obs in observations.values()
            if "gfs" in obs and "ecmwf" in obs
        ]
        if len(paired) < 5:
            continue

        # Pre-compute arrays for speed
        gfs_vals = [obs["gfs"] for obs in paired]
        ecmwf_vals = [obs["ecmwf"] for obs in paired]
        actual_vals = [obs["actual"] for obs in paired]
        n = len(paired)

        best_mae = float("inf")
        best_w = (0.5, 0.5, 0.0)  # (gfs, ecmwf, noaa)

        # Pre-compute NOAA proxy = model consensus (midpoint GFS/ECMWF)
        noaa_proxy = [(gfs_vals[k] + ecmwf_vals[k]) / 2.0 for k in range(n)]

        # 2-D grid: w_gfs ∈ [0,1], w_noaa ∈ [0,1], w_ecmwf = 1 - w_gfs - w_noaa ≥ 0
        for i in range(steps):
            w_gfs = round(i * grid_step, 4)
            max_noaa = round(1.0 - w_gfs, 4)
            noaa_steps = int(round(max_noaa / grid_step)) + 1
            for j in range(noaa_steps):
                w_noaa = round(j * grid_step, 4)
                w_ecmwf = round(1.0 - w_gfs - w_noaa, 4)
                if w_ecmwf < -1e-9:
                    continue

                total_ae = 0.0
                for k in range(n):
                    ens = (w_gfs * gfs_vals[k]
                           + w_ecmwf * ecmwf_vals[k]
                           + w_noaa * noaa_proxy[k])
                    total_ae += abs(ens - actual_vals[k])
                mae = total_ae / n

                if mae < best_mae:
                    best_mae = mae
                    best_w = (w_gfs, w_ecmwf, w_noaa)

        weights: dict[str, float] = {
            "gfs_seamless": round(max(0.0, best_w[0]), 3),
            "ecmwf_ifs025": round(max(0.0, best_w[1]), 3),
            "noaa": round(max(0.0, best_w[2]), 3),
        }

        # Normalize to exactly 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}

        logger.info(
            "Optimal weights for %s: GFS=%.1f%% ECMWF=%.1f%% NOAA=%.1f%% "
            "(MAE=%.3f°F, n=%d obs)",
            group_key, weights["gfs_seamless"] * 100,
            weights["ecmwf_ifs025"] * 100, weights["noaa"] * 100,
            best_mae, n,
        )

        results[group_key] = weights

    return results


def _compute_base_sigma(errors: list[dict]) -> float:
    """Compute base sigma (stddev of all errors, deduplicated)."""
    vals = [e["error"] for e in errors]
    if len(vals) < 3:
        return 1.5  # Fallback
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)  # Bessel's correction
    return math.sqrt(variance)


def _expand_sigma_by_horizon(base_sigma: float) -> dict[str, float]:
    """Generate sigma for horizons 0-10 using the growth model.

    sigma(h) = base_sigma * _HORIZON_GROWTH[h]
    """
    return {
        str(h): round(base_sigma * _horizon_growth_factor(h), 2)
        for h in range(11)
    }


def _compute_sigma_by_horizon(errors: list[dict], max_horizon: int = 10) -> dict[str, float]:
    """Compute real sigma at each horizon from error data.

    For horizons with data: sigma = sqrt(mean(error^2)) (RMSE).
    For horizons without data: linear interpolation/extrapolation.
    Requires at least 20 samples per horizon for a data point.

    Args:
        errors: List of error records, each containing ``"horizon"`` (int)
            and ``"error"`` (float) fields.
        max_horizon: Maximum horizon to compute (default 10).

    Returns:
        ``{"0": sigma_0, "1": sigma_1, ...}`` rounded to 2 decimals.
    """
    # Group errors by horizon
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for e in errors:
        h = e.get("horizon")
        if h is not None:
            by_horizon[h].append(e["error"])

    # Compute RMSE for horizons with >= 20 samples
    known: dict[int, float] = {}
    for h, errs in sorted(by_horizon.items()):
        if len(errs) >= 20:
            rmse = math.sqrt(sum(e ** 2 for e in errs) / len(errs))
            known[h] = rmse

    if not known:
        # No horizon has enough data — fall back to growth model
        base = _compute_base_sigma(errors) if errors else 1.5
        return _expand_sigma_by_horizon(base)

    # Fill all horizons 0..max_horizon via interpolation/extrapolation
    known_horizons = sorted(known.keys())
    result: dict[str, float] = {}

    for h in range(max_horizon + 1):
        if h in known:
            result[str(h)] = round(known[h], 2)
            continue

        # Find surrounding known horizons for interpolation
        lower = [kh for kh in known_horizons if kh < h]
        upper = [kh for kh in known_horizons if kh > h]

        if lower and upper:
            # Interpolate between nearest lower and upper
            h_lo = lower[-1]
            h_hi = upper[0]
            sigma_lo = known[h_lo]
            sigma_hi = known[h_hi]
            fraction = (h - h_lo) / (h_hi - h_lo)
            sigma = sigma_lo + fraction * (sigma_hi - sigma_lo)
            result[str(h)] = round(sigma, 2)
        elif lower:
            # Extrapolate beyond last known horizon
            if len(known_horizons) >= 2:
                # Use growth rate between last two known horizons
                h_prev = known_horizons[-2]
                h_last = known_horizons[-1]
                sigma_prev = known[h_prev]
                sigma_last = known[h_last]
                rate = (sigma_last - sigma_prev) / (h_last - h_prev)
                sigma = sigma_last + rate * (h - h_last)
                result[str(h)] = round(max(sigma, 0.01), 2)
            else:
                # Only one known horizon — use its value
                result[str(h)] = round(known[known_horizons[-1]], 2)
        elif upper:
            # Extrapolate below first known horizon
            if len(known_horizons) >= 2:
                h_first = known_horizons[0]
                h_second = known_horizons[1]
                sigma_first = known[h_first]
                sigma_second = known[h_second]
                rate = (sigma_second - sigma_first) / (h_second - h_first)
                sigma = sigma_first + rate * (h - h_first)
                result[str(h)] = round(max(sigma, 0.01), 2)
            else:
                result[str(h)] = round(known[known_horizons[0]], 2)

    return result


def compute_horizon_errors(
    previous_runs_data: dict[int, dict[str, dict]],
    actuals: dict[str, dict],
    location: str,
) -> list[dict]:
    """Compute forecast errors at each horizon from Previous Runs data.

    Compares forecasts issued at different horizons against observed actuals
    to produce error records that include a ``"horizon"`` field.  Each record
    has the same fields as :func:`compute_forecast_errors` returns, plus
    ``"horizon": int``.

    Args:
        previous_runs_data: ``{horizon: {date_str: {gfs_high, gfs_low,
            ecmwf_high, ecmwf_low}}}`` — output of
            :func:`weather.previous_runs.fetch_previous_runs`.
        actuals: ``{date_str: {high, low}}`` — observed temperatures.
        location: Location key (e.g. ``"NYC"``), stored in each record.

    Returns:
        List of error records with ``"horizon"`` field.
    """
    errors: list[dict] = []

    for horizon, dates in previous_runs_data.items():
        for date_str, model_data in dates.items():
            actual = actuals.get(date_str)
            if not actual:
                continue

            month = datetime.strptime(date_str, "%Y-%m-%d").month

            # Compute model spread for this target date
            spread: dict[str, float] = {}
            for metric in ["high", "low"]:
                gfs_val = model_data.get(f"gfs_{metric}")
                ecmwf_val = model_data.get(f"ecmwf_{metric}")
                if gfs_val is not None and ecmwf_val is not None:
                    spread[metric] = abs(gfs_val - ecmwf_val)

            for model_prefix in ["gfs", "ecmwf"]:
                for metric in ["high", "low"]:
                    forecast_key = f"{model_prefix}_{metric}"
                    forecast_val = model_data.get(forecast_key)
                    actual_val = actual.get(metric)

                    if forecast_val is None or actual_val is None:
                        continue

                    record = {
                        "location": location,
                        "target_date": date_str,
                        "month": month,
                        "metric": metric,
                        "model": model_prefix,
                        "forecast": forecast_val,
                        "actual": actual_val,
                        "error": forecast_val - actual_val,
                        "model_spread": spread.get(metric, 0.0),
                        "horizon": int(horizon),
                    }
                    errors.append(record)

    logger.info(
        "Computed %d horizon errors for %s across %d horizons",
        len(errors), location, len(previous_runs_data),
    )
    return errors


def _test_normality(errors: list[float]) -> dict:
    """Jarque-Bera test for normality of forecast errors.

    Returns dict with:
        normal: bool — True if normality NOT rejected at 5% level
        jb_statistic: float
        skewness: float
        kurtosis: float (excess kurtosis = kurtosis - 3)
    """
    n = len(errors)
    if n < 30:
        return {"normal": True, "jb_statistic": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    mean = sum(errors) / n
    centered = [e - mean for e in errors]
    m2 = sum(c**2 for c in centered) / n
    m3 = sum(c**3 for c in centered) / n
    m4 = sum(c**4 for c in centered) / n
    sigma = m2 ** 0.5
    if sigma < 1e-10:
        return {"normal": True, "jb_statistic": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    skewness = m3 / (sigma ** 3)
    kurtosis = m4 / (sigma ** 4)
    excess_kurtosis = kurtosis - 3.0
    jb = (n / 6.0) * (skewness**2 + excess_kurtosis**2 / 4.0)
    # chi-squared(2) critical value at 5%: 5.991
    normal = jb < 5.991
    return {
        "normal": normal,
        "jb_statistic": round(jb, 2),
        "skewness": round(skewness, 3),
        "kurtosis": round(excess_kurtosis, 3),
    }


def _student_t_logpdf(x: float, df: float) -> float:
    """Log-PDF of standard Student's t-distribution."""
    return (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - ((df + 1) / 2) * math.log(1 + x**2 / df)
    )


def _fit_student_t_df(errors: list[float]) -> float:
    """Fit degrees of freedom for Student's t by maximum likelihood.

    Grid search over candidate df values. Returns the df that maximizes
    the log-likelihood of the standardized errors.
    """
    n = len(errors)
    sigma = (sum(e**2 for e in errors) / n) ** 0.5
    standardized = [e / sigma for e in errors] if sigma > 0 else errors

    best_df = 30.0
    best_ll = float("-inf")

    for df in [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]:
        ll = sum(_student_t_logpdf(z, df) for z in standardized)
        if ll > best_ll:
            best_ll = ll
            best_df = float(df)

    return best_df


def _compute_mean_model_spread(errors: list[dict]) -> float:
    """Average model spread across all errors (for diagnostics)."""
    spreads = [e.get("model_spread", 0.0) for e in errors if e.get("model_spread")]
    return sum(spreads) / len(spreads) if spreads else 0.0


_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    sx = (sum((x - mx) ** 2 for x in xs) / (n - 1)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / (n - 1)) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _compute_correlation_matrix(
    locations: list[str],
    errors: list[dict],
    min_samples: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute Pearson correlation of forecast errors between location pairs by season.

    Args:
        locations: List of location keys.
        errors: Error records with "location", "target_date", "month", "error".
        min_samples: Minimum shared dates per season to include (default 5).

    Returns:
        {"LocA|LocB": {"DJF": 0.72, "MAM": 0.45, ...}, ...}
        Location pair keys are alphabetically sorted.
    """
    by_loc_date: dict[tuple[str, str], float] = {}
    date_months: dict[str, int] = {}
    for e in errors:
        loc = e["location"]
        date = e["target_date"]
        by_loc_date[(loc, date)] = e["error"]
        date_months[date] = e["month"]

    result: dict[str, dict[str, float]] = {}

    for i, loc_a in enumerate(locations):
        for loc_b in locations[i + 1:]:
            pair_key = "|".join(sorted([loc_a, loc_b]))
            dates_a = {d for (l, d) in by_loc_date if l == loc_a}
            dates_b = {d for (l, d) in by_loc_date if l == loc_b}
            shared = dates_a & dates_b

            season_dates: dict[str, list[str]] = {}
            for d in shared:
                season = _MONTH_TO_SEASON.get(date_months.get(d, 1), "DJF")
                season_dates.setdefault(season, []).append(d)

            pair_seasons: dict[str, float] = {}
            for season, dates in season_dates.items():
                if len(dates) < min_samples:
                    continue
                xs = [by_loc_date[(loc_a, d)] for d in dates]
                ys = [by_loc_date[(loc_b, d)] for d in dates]
                corr = _pearson_correlation(xs, ys)
                pair_seasons[season] = round(corr, 3)

            if pair_seasons:
                result[pair_key] = pair_seasons

    return result


def _compute_adaptive_factors(errors: list[dict]) -> dict:
    """Compute calibrated adaptive sigma conversion factors from historical data.

    Derives ``spread_to_sigma_factor`` and ``ema_to_sigma_factor`` empirically
    from the relationship between model spread / EMA of errors and the actual
    observed forecast errors.

    Args:
        errors: List of error records from ``compute_forecast_errors``.

    Returns:
        Dict with ``underdispersion_factor`` (literature default),
        ``spread_to_sigma_factor``, ``ema_to_sigma_factor``, and ``samples``.
    """
    # Group by (target_date, metric) to get one weighted error + one spread per day
    day_groups: dict[tuple[str, str], dict] = {}
    for err in errors:
        key = (err["target_date"], err["metric"])
        if key not in day_groups:
            day_groups[key] = {"errors": [], "spread": err.get("model_spread", 0.0)}
        day_groups[key]["errors"].append(err)

    if not day_groups:
        return {
            "underdispersion_factor": 1.3,
            "spread_to_sigma_factor": 0.7,
            "ema_to_sigma_factor": 1.25,
            "samples": 0,
        }

    # Compute weighted ensemble error per day (weighted by inverse-RMSE style)
    # For simplicity: average the absolute errors across models for each day
    day_records: list[dict] = []
    for (target_date, metric), group in sorted(day_groups.items()):
        abs_errors = [abs(e["error"]) for e in group["errors"]]
        weighted_abs_error = sum(abs_errors) / len(abs_errors)
        day_records.append({
            "target_date": target_date,
            "metric": metric,
            "abs_error": weighted_abs_error,
            "spread": group["spread"],
        })

    # --- spread_to_sigma_factor ---
    # sqrt(mean(error²) / mean(spread²))
    sum_err_sq = sum(r["abs_error"] ** 2 for r in day_records)
    sum_spread_sq = sum(r["spread"] ** 2 for r in day_records)
    n = len(day_records)

    if sum_spread_sq > 0:
        spread_to_sigma = math.sqrt(sum_err_sq / sum_spread_sq)
    else:
        spread_to_sigma = 0.7  # Fallback

    # --- ema_to_sigma_factor ---
    # Simulate EMA of absolute errors over sorted time series, then compare
    alpha = 0.15
    sorted_records = sorted(day_records, key=lambda r: (r["target_date"], r["metric"]))

    sum_ema_sq = 0.0
    ema = sorted_records[0]["abs_error"] if sorted_records else 1.0
    for rec in sorted_records:
        sum_ema_sq += ema ** 2
        ema = alpha * rec["abs_error"] + (1 - alpha) * ema

    if sum_ema_sq > 0:
        ema_to_sigma = math.sqrt(sum_err_sq / sum_ema_sq)
    else:
        ema_to_sigma = 1.25  # Fallback

    logger.info("Adaptive factors: spread_to_sigma=%.3f, ema_to_sigma=%.3f (n=%d)",
                spread_to_sigma, ema_to_sigma, n)

    return {
        "underdispersion_factor": 1.3,
        "spread_to_sigma_factor": round(spread_to_sigma, 3),
        "ema_to_sigma_factor": round(ema_to_sigma, 3),
        "samples": n,
    }


def _compute_platt_params(predictions: list[float], actuals: list[float]) -> dict:
    """Fit Platt scaling parameters (a, b) via gradient descent on log-loss.
    Platt scaling: calibrated = sigmoid(a * logit(pred) + b)
    """
    if len(predictions) < 2 or len(actuals) < 2:
        return {"a": 1.0, "b": 0.0}

    a, b = 1.0, 0.0
    lr = 0.1
    eps = 1e-6

    for _ in range(500):
        grad_a, grad_b = 0.0, 0.0
        n = len(predictions)
        for pred, actual in zip(predictions, actuals):
            p = max(eps, min(1 - eps, pred))
            y = max(eps, min(1 - eps, actual))
            logit_p = math.log(p / (1 - p))
            z = max(-500, min(500, a * logit_p + b))
            sigmoid_z = 1.0 / (1.0 + math.exp(-z))
            err = sigmoid_z - y
            grad_a += err * logit_p / n
            grad_b += err / n
        a -= lr * grad_a
        b -= lr * grad_b

    return {"a": round(a, 4), "b": round(b, 4)}


def _fit_platt_from_errors(errors: list[dict]) -> dict:
    """Fit Platt params by simulating bucket probabilities vs actual outcomes.
    Groups errors into probability bins and computes actual hit rates,
    then fits Platt scaling on the resulting calibration curve.
    """
    if not errors:
        return {"a": 1.0, "b": 0.0}

    from .probability import estimate_bucket_probability

    bins: dict[int, list[int]] = {}
    for err in errors:
        forecast = err.get("forecast", 0)
        actual = err.get("actual", 0)
        bucket_low = round(forecast) - 2
        bucket_high = round(forecast) + 2
        prob = estimate_bucket_probability(
            forecast, bucket_low, bucket_high,
            err.get("target_date", "2025-06-15"),
            apply_seasonal=False,
            location=err.get("location", ""),
            horizon_override=err.get("horizon"),
        )
        outcome = 1 if bucket_low <= actual <= bucket_high else 0
        bin_idx = min(9, int(prob * 10))
        bins.setdefault(bin_idx, []).append(outcome)

    predictions = []
    actuals_list = []
    for bin_idx in sorted(bins):
        outcomes = bins[bin_idx]
        if len(outcomes) >= 5:
            pred = (bin_idx + 0.5) / 10.0
            actual_rate = sum(outcomes) / len(outcomes)
            predictions.append(pred)
            actuals_list.append(actual_rate)

    return _compute_platt_params(predictions, actuals_list)


def build_calibration_tables(
    all_errors: list[dict],
    locations: list[str],
    horizon_errors: list[dict] | None = None,
) -> dict:
    """Build the full calibration.json structure from all errors.

    Uses a **hybrid approach**: empirical base sigma from the data, then either
    real RMSE by horizon (when ``horizon_errors`` is provided) or the NWP
    horizon growth model (fallback) to generate sigma for all horizons 0-10.

    Args:
        all_errors: Combined error records from all locations.
        locations: List of location keys processed.
        horizon_errors: Optional list of error records with ``"horizon"`` field
            from :func:`compute_horizon_errors`.  When provided, real RMSE per
            horizon is used instead of the fictitious growth model.

    Returns:
        Dict ready to be serialized to ``calibration.json``.
    """
    # Global base sigma (one value from all deduplicated errors)
    global_base = _compute_base_sigma(all_errors)
    logger.info("Global base sigma: %.2f°F", global_base)

    # Expand to full horizon table
    if horizon_errors is not None:
        global_sigma = _compute_sigma_by_horizon(horizon_errors)
        horizon_model = "real RMSE from Previous Runs data"
        logger.info("Using real RMSE by horizon (global)")
    else:
        global_sigma = _expand_sigma_by_horizon(global_base)
        horizon_model = "NWP linear: sigma(h) = base * growth(h)"

    # Seasonal factors (relative sigma by month)
    # Factor > 1.0 means this month is MORE uncertain than average (multiply sigma up).
    # Factor < 1.0 means LESS uncertain (multiply sigma down).
    # Therefore: factor = monthly_sigma / mean_sigma (not inverted).
    monthly_sigma = compute_empirical_sigma(all_errors, group_by="month")
    if monthly_sigma:
        mean_sigma = sum(monthly_sigma.values()) / len(monthly_sigma)
        seasonal_factors = {
            m: round(s / mean_sigma, 3) if mean_sigma > 0 else 1.0
            for m, s in monthly_sigma.items()
        }
    else:
        seasonal_factors = {}

    # Per-location sigma (base + growth or real RMSE)
    location_sigma: dict[str, dict] = {}
    location_seasonal: dict[str, dict] = {}
    for loc in locations:
        loc_errors = [e for e in all_errors if e["location"] == loc]
        if not loc_errors:
            continue

        if horizon_errors is not None:
            loc_horizon_errors = [e for e in horizon_errors if e.get("location") == loc]
            if loc_horizon_errors:
                location_sigma[loc] = _compute_sigma_by_horizon(loc_horizon_errors)
                logger.info("  %s: using real RMSE by horizon", loc)
            else:
                loc_base = _compute_base_sigma(loc_errors)
                location_sigma[loc] = _expand_sigma_by_horizon(loc_base)
                logger.info("  %s base sigma: %.2f°F (fallback — no horizon data)", loc, loc_base)
        else:
            loc_base = _compute_base_sigma(loc_errors)
            location_sigma[loc] = _expand_sigma_by_horizon(loc_base)
            logger.info("  %s base sigma: %.2f°F", loc, loc_base)

        loc_monthly_sigma = compute_empirical_sigma(loc_errors, group_by="month")
        if loc_monthly_sigma:
            loc_mean = sum(loc_monthly_sigma.values()) / len(loc_monthly_sigma)
            location_seasonal[loc] = {
                m: round(s / loc_mean, 3) if loc_mean > 0 else 1.0
                for m, s in loc_monthly_sigma.items()
            }

    # Model weights per location
    model_weights = compute_model_weights(all_errors, group_by="location")

    # Model spread diagnostics
    mean_spread = _compute_mean_model_spread(all_errors)
    logger.info("Mean model spread (|GFS - ECMWF|): %.2f°F", mean_spread)

    # Adaptive sigma factors
    adaptive_factors = _compute_adaptive_factors(all_errors)

    # Platt scaling
    platt_params = _fit_platt_from_errors(all_errors)

    # Test normality and fit distribution
    all_error_values = [e["error"] for e in all_errors]
    normality = _test_normality(all_error_values)
    if normality["normal"]:
        distribution = "normal"
        student_t_df = None
    else:
        distribution = "student_t"
        student_t_df = _fit_student_t_df(all_error_values)
        logger.info("Non-normal errors detected (JB=%.1f, skew=%.3f, kurt=%.3f). Using Student's t (df=%.0f)",
                    normality["jb_statistic"], normality["skewness"], normality["kurtosis"], student_t_df)

    # Inter-location correlation matrix
    correlation_matrix = _compute_correlation_matrix(locations, all_errors)

    # Metadata
    dates = [e["target_date"] for e in all_errors]
    date_range = [min(dates), max(dates)] if dates else []

    result = {
        "global_sigma": global_sigma,
        "location_sigma": location_sigma,
        "seasonal_factors": seasonal_factors,
        "location_seasonal": location_seasonal,
        "model_weights": model_weights,
        "adaptive_sigma": adaptive_factors,
        "platt_scaling": platt_params,
        "correlation_matrix": correlation_matrix,
        "distribution": distribution,
        "normality_test": normality,
        "horizon_growth": {str(h): round(_horizon_growth_factor(h), 2) for h in range(11)},
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "samples": len(all_errors),
            "date_range": date_range,
            "locations": locations,
            "base_sigma_global": round(global_base, 2),
            "mean_model_spread": round(mean_spread, 2),
            "horizon_growth_model": horizon_model,
        },
    }
    if student_t_df is not None:
        result["student_t_df"] = student_t_df

    return result


def compute_exponential_weights(
    dates: list[str],
    half_life: float = 30.0,
    reference_date: str | None = None,
) -> dict[str, float]:
    """Compute exponential decay weights for a list of date strings.

    Formula: w(t) = exp(-ln(2) * age_days / half_life)

    Args:
        dates: List of date strings in YYYY-MM-DD format.
        half_life: Half-life in days (default 30).
        reference_date: Override "today" for testing (YYYY-MM-DD string).
            Defaults to ``datetime.now()``.

    Returns:
        Dict mapping date string to weight in (0, 1].
    """
    if reference_date is not None:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()

    ln2 = math.log(2)
    weights: dict[str, float] = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        age_days = (ref - dt).days
        if age_days < 0:
            age_days = 0
        weights[d] = math.exp(-ln2 * age_days / half_life)
    return weights


def _weighted_base_sigma(errors: list[dict], weights: dict[str, float]) -> float:
    """Compute base sigma using exponentially-weighted errors.

    Formula: sqrt(sum(w * error^2) / sum(w))

    Falls back to 1.5 if sum of weights is near zero.
    """
    sum_w = 0.0
    sum_w_err_sq = 0.0
    for e in errors:
        w = weights.get(e["target_date"], 0.0)
        sum_w += w
        sum_w_err_sq += w * e["error"] ** 2
    if sum_w < 1e-9:
        return 1.5
    return math.sqrt(sum_w_err_sq / sum_w)


def _weighted_model_weights(
    errors: list[dict],
    weights: dict[str, float],
    group_by: str = "location",
    grid_step: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Compute model weights per group via 2-D grid search on time-weighted MAE.

    Same as ``compute_model_weights()`` but applies exponential time-decay
    weights so recent observations count more heavily.
    """
    # Align GFS & ECMWF by (group, target_date, metric) with time-weight
    aligned: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)

    for err in errors:
        group_key = str(err.get(group_by, "global"))
        obs_key = (err["target_date"], err["metric"])
        if obs_key not in aligned[group_key]:
            aligned[group_key][obs_key] = {
                "actual": err["actual"],
                "tw": weights.get(err["target_date"], 0.0),
            }
        aligned[group_key][obs_key][err["model"]] = err["forecast"]

    results: dict[str, dict[str, float]] = {}
    steps = int(round(1.0 / grid_step)) + 1

    for group_key, observations in aligned.items():
        paired = [
            obs for obs in observations.values()
            if "gfs" in obs and "ecmwf" in obs and obs.get("tw", 0) > 1e-9
        ]
        if len(paired) < 5:
            continue

        gfs_vals = [obs["gfs"] for obs in paired]
        ecmwf_vals = [obs["ecmwf"] for obs in paired]
        actual_vals = [obs["actual"] for obs in paired]
        tw_vals = [obs["tw"] for obs in paired]
        total_tw = sum(tw_vals)
        noaa_proxy = [(gfs_vals[k] + ecmwf_vals[k]) / 2.0 for k in range(len(paired))]

        best_mae = float("inf")
        best_w = (0.5, 0.5, 0.0)

        for i in range(steps):
            w_gfs = round(i * grid_step, 4)
            max_noaa = round(1.0 - w_gfs, 4)
            noaa_steps = int(round(max_noaa / grid_step)) + 1
            for j in range(noaa_steps):
                w_noaa = round(j * grid_step, 4)
                w_ecmwf = round(1.0 - w_gfs - w_noaa, 4)
                if w_ecmwf < -1e-9:
                    continue

                sum_w_ae = 0.0
                for k in range(len(paired)):
                    ens = (w_gfs * gfs_vals[k]
                           + w_ecmwf * ecmwf_vals[k]
                           + w_noaa * noaa_proxy[k])
                    sum_w_ae += tw_vals[k] * abs(ens - actual_vals[k])
                mae = sum_w_ae / total_tw if total_tw > 1e-9 else float("inf")

                if mae < best_mae:
                    best_mae = mae
                    best_w = (w_gfs, w_ecmwf, w_noaa)

        model_weights: dict[str, float] = {
            "gfs_seamless": round(max(0.0, best_w[0]), 3),
            "ecmwf_ifs025": round(max(0.0, best_w[1]), 3),
            "noaa": round(max(0.0, best_w[2]), 3),
        }

        total = sum(model_weights.values())
        if total > 0:
            model_weights = {k: round(v / total, 3) for k, v in model_weights.items()}

        results[group_key] = model_weights

    return results


def build_weighted_calibration_tables(
    all_errors: list[dict],
    locations: list[str],
    half_life: float = 30.0,
    reference_date: str | None = None,
    horizon_errors: list[dict] | None = None,
) -> dict:
    """Build calibration tables with exponential decay weighting.

    Same output format as ``build_calibration_tables()`` but weights recent
    data more heavily using exponential decay.

    Args:
        all_errors: Combined error records from all locations.
        locations: List of location keys processed.
        half_life: Half-life in days for exponential weighting (default 30).
        reference_date: Override "today" for testing (YYYY-MM-DD string).
        horizon_errors: Optional list of error records with ``"horizon"`` field
            from :func:`compute_horizon_errors`.  When provided, real RMSE per
            horizon is used instead of the fictitious growth model.

    Returns:
        Dict ready to be serialized to ``calibration.json``.
    """
    # Compute exponential weights for all dates
    all_dates = list({e["target_date"] for e in all_errors})
    weights = compute_exponential_weights(all_dates, half_life, reference_date)

    # Global weighted base sigma
    global_base = _weighted_base_sigma(all_errors, weights)
    logger.info("Weighted global base sigma: %.2f°F (half-life=%.0fd)", global_base, half_life)

    # Expand to full horizon table
    if horizon_errors is not None:
        global_sigma = _compute_sigma_by_horizon(horizon_errors)
        horizon_model = "real RMSE from Previous Runs data"
        logger.info("Using real RMSE by horizon (global, weighted)")
    else:
        global_sigma = _expand_sigma_by_horizon(global_base)
        horizon_model = "NWP linear: sigma(h) = base * growth(h)"

    # Weighted seasonal factors
    # Group errors by month, compute weighted sigma per month
    monthly_errors: dict[str, list[dict]] = defaultdict(list)
    for e in all_errors:
        monthly_errors[str(e["month"])].append(e)

    monthly_sigma: dict[str, float] = {}
    for month_str, errs in monthly_errors.items():
        if len(errs) >= 3:
            monthly_sigma[month_str] = _weighted_base_sigma(errs, weights)

    if monthly_sigma:
        mean_sigma = sum(monthly_sigma.values()) / len(monthly_sigma)
        seasonal_factors = {
            m: round(s / mean_sigma, 3) if mean_sigma > 0 else 1.0
            for m, s in monthly_sigma.items()
        }
    else:
        seasonal_factors = {}

    # Per-location weighted sigma
    location_sigma: dict[str, dict] = {}
    location_seasonal: dict[str, dict] = {}
    for loc in locations:
        loc_errors = [e for e in all_errors if e["location"] == loc]
        if not loc_errors:
            continue

        if horizon_errors is not None:
            loc_horizon_errors = [e for e in horizon_errors if e.get("location") == loc]
            if loc_horizon_errors:
                location_sigma[loc] = _compute_sigma_by_horizon(loc_horizon_errors)
                logger.info("  %s: using real RMSE by horizon (weighted)", loc)
            else:
                loc_base = _weighted_base_sigma(loc_errors, weights)
                location_sigma[loc] = _expand_sigma_by_horizon(loc_base)
                logger.info("  %s weighted base sigma: %.2f°F (fallback)", loc, loc_base)
        else:
            loc_base = _weighted_base_sigma(loc_errors, weights)
            location_sigma[loc] = _expand_sigma_by_horizon(loc_base)
            logger.info("  %s weighted base sigma: %.2f°F", loc, loc_base)

        loc_monthly: dict[str, list[dict]] = defaultdict(list)
        for e in loc_errors:
            loc_monthly[str(e["month"])].append(e)

        loc_monthly_sigma: dict[str, float] = {}
        for month_str, errs in loc_monthly.items():
            if len(errs) >= 3:
                loc_monthly_sigma[month_str] = _weighted_base_sigma(errs, weights)

        if loc_monthly_sigma:
            loc_mean = sum(loc_monthly_sigma.values()) / len(loc_monthly_sigma)
            location_seasonal[loc] = {
                m: round(s / loc_mean, 3) if loc_mean > 0 else 1.0
                for m, s in loc_monthly_sigma.items()
            }

    # Weighted model weights
    model_weights = _weighted_model_weights(all_errors, weights, group_by="location")

    # Reuse existing unweighted functions for adaptive and Platt
    adaptive_factors = _compute_adaptive_factors(all_errors)
    platt_params = _fit_platt_from_errors(all_errors)

    # Test normality and fit distribution
    all_error_values = [e["error"] for e in all_errors]
    normality = _test_normality(all_error_values)
    if normality["normal"]:
        distribution = "normal"
        student_t_df = None
    else:
        distribution = "student_t"
        student_t_df = _fit_student_t_df(all_error_values)
        logger.info("Non-normal errors detected (JB=%.1f, skew=%.3f, kurt=%.3f). Using Student's t (df=%.0f)",
                    normality["jb_statistic"], normality["skewness"], normality["kurtosis"], student_t_df)

    # Effective sample count = sum of all weights
    sum_all_weights = sum(weights.get(e["target_date"], 0.0) for e in all_errors)

    # Inter-location correlation matrix
    correlation_matrix = _compute_correlation_matrix(locations, all_errors)

    # Metadata
    dates = [e["target_date"] for e in all_errors]
    date_range = [min(dates), max(dates)] if dates else []

    result = {
        "global_sigma": global_sigma,
        "location_sigma": location_sigma,
        "seasonal_factors": seasonal_factors,
        "location_seasonal": location_seasonal,
        "model_weights": model_weights,
        "adaptive_sigma": adaptive_factors,
        "platt_scaling": platt_params,
        "correlation_matrix": correlation_matrix,
        "distribution": distribution,
        "normality_test": normality,
        "horizon_growth": {str(h): round(_horizon_growth_factor(h), 2) for h in range(11)},
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "samples": len(all_errors),
            "samples_effective": round(sum_all_weights, 1),
            "weighting": f"exponential half-life {half_life}d",
            "date_range": date_range,
            "locations": locations,
            "base_sigma_global": round(global_base, 2),
            "horizon_growth_model": horizon_model,
        },
    }
    if student_t_df is not None:
        result["student_t_df"] = student_t_df

    return result


def main() -> None:
    """CLI entry point for calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate weather forecast sigma from historical data",
    )
    parser.add_argument(
        "--locations", type=str, default="NYC",
        help="Comma-separated location keys (default: NYC)",
    )
    parser.add_argument(
        "--start-date", type=str, required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", type=str, default=_DEFAULT_OUTPUT,
        help=f"Output path for calibration.json (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--actuals-source", type=str, default="metar",
        choices=["metar", "era5"],
        help="Ground truth source: metar (ASOS stations) or era5 (reanalysis). Default: metar",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loc_keys = [l.strip() for l in args.locations.split(",")]
    all_errors: list[dict] = []

    for loc in loc_keys:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.error("Unknown location: %s (available: %s)", loc, ", ".join(LOCATIONS))
            continue

        station = loc_data.get("station")
        if args.actuals_source == "metar" and station:
            errors = _compute_errors_with_metar(
                location=loc,
                lat=loc_data["lat"],
                lon=loc_data["lon"],
                station=station,
                start_date=args.start_date,
                end_date=args.end_date,
                tz_name=loc_data.get("tz", "America/New_York"),
            )
        else:
            if args.actuals_source == "metar" and not station:
                logger.warning("No METAR station for %s — falling back to ERA5", loc)
            errors = compute_forecast_errors(
                location=loc,
                lat=loc_data["lat"],
                lon=loc_data["lon"],
                start_date=args.start_date,
                end_date=args.end_date,
                tz_name=loc_data.get("tz", "America/New_York"),
            )
        all_errors.extend(errors)

    if not all_errors:
        logger.error("No forecast errors computed — check date range and API availability")
        sys.exit(1)

    calibration = build_calibration_tables(all_errors, loc_keys)

    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info("Calibration written to %s (%d samples)", args.output, len(all_errors))
    logger.info("Global sigma by horizon: %s", calibration["global_sigma"])
    logger.info("Model weights: %s", calibration["model_weights"])


if __name__ == "__main__":
    main()
