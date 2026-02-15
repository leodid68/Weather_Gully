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
from .historical import get_historical_actuals, get_historical_forecasts

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
) -> dict[str, dict[str, float]]:
    """Compute optimal model weights per group by minimizing RMSE via grid search.

    Args:
        errors: List of error records.
        group_by: Key to group by (typically ``"location"``).

    Returns:
        Dict mapping group key to model weight dict, e.g.::

            {"NYC": {"noaa": 0.20, "gfs_seamless": 0.30, "ecmwf_ifs025": 0.50}}
    """
    # Group errors by (group_key, target_date, metric) to get paired model predictions
    paired: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for err in errors:
        group_key = str(err.get(group_by, "global"))
        model = err["model"]
        paired[group_key][model].append((err["forecast"], err["actual"]))

    results: dict[str, dict[str, float]] = {}

    for group_key, model_data in paired.items():
        # Compute RMSE for each model
        model_rmse: dict[str, float] = {}
        for model, pairs in model_data.items():
            if not pairs:
                continue
            mse = sum((f - a) ** 2 for f, a in pairs) / len(pairs)
            model_rmse[model] = math.sqrt(mse)

        if not model_rmse:
            continue

        # Inverse-RMSE weighting (lower RMSE → higher weight)
        inv_rmse = {m: 1.0 / r for m, r in model_rmse.items() if r > 0}
        total_inv = sum(inv_rmse.values())

        if total_inv == 0:
            continue

        # Map to canonical model names
        model_map = {"gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025"}
        weights: dict[str, float] = {}
        for model, inv in inv_rmse.items():
            canonical = model_map.get(model, model)
            weights[canonical] = round(inv / total_inv, 3)

        # Add NOAA weight (default 0.20, reduce proportionally from others)
        noaa_weight = 0.20
        remaining = 1.0 - noaa_weight
        for k in weights:
            weights[k] = round(weights[k] * remaining, 3)
        weights["noaa"] = noaa_weight

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}

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


def _compute_mean_model_spread(errors: list[dict]) -> float:
    """Average model spread across all errors (for diagnostics)."""
    spreads = [e.get("model_spread", 0.0) for e in errors if e.get("model_spread")]
    return sum(spreads) / len(spreads) if spreads else 0.0


def build_calibration_tables(
    all_errors: list[dict],
    locations: list[str],
) -> dict:
    """Build the full calibration.json structure from all errors.

    Uses a **hybrid approach**: empirical base sigma from the data, then the
    NWP horizon growth model to generate sigma for all horizons 0-10.

    Args:
        all_errors: Combined error records from all locations.
        locations: List of location keys processed.

    Returns:
        Dict ready to be serialized to ``calibration.json``.
    """
    # Global base sigma (one value from all deduplicated errors)
    global_base = _compute_base_sigma(all_errors)
    logger.info("Global base sigma: %.2f°F", global_base)

    # Expand to full horizon table using NWP growth model
    global_sigma = _expand_sigma_by_horizon(global_base)

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

    # Per-location sigma (base + growth)
    location_sigma: dict[str, dict] = {}
    location_seasonal: dict[str, dict] = {}
    for loc in locations:
        loc_errors = [e for e in all_errors if e["location"] == loc]
        if not loc_errors:
            continue

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

    # Metadata
    dates = [e["target_date"] for e in all_errors]
    date_range = [min(dates), max(dates)] if dates else []

    return {
        "global_sigma": global_sigma,
        "location_sigma": location_sigma,
        "seasonal_factors": seasonal_factors,
        "location_seasonal": location_seasonal,
        "model_weights": model_weights,
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "samples": len(all_errors),
            "date_range": date_range,
            "locations": locations,
            "base_sigma_global": round(global_base, 2),
            "mean_model_spread": round(mean_spread, 2),
            "horizon_growth_model": "NWP linear: sigma(h) = base * growth(h)",
        },
    }


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loc_keys = [l.strip() for l in args.locations.split(",")]
    all_errors: list[dict] = []

    for loc in loc_keys:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.error("Unknown location: %s (available: %s)", loc, ", ".join(LOCATIONS))
            continue

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
