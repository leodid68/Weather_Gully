"""NOAA probability model — dynamic accuracy by horizon, season, and bucket width."""

import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# NOAA forecast accuracy curve (days ahead → probability of being correct)
_HORIZON_ACCURACY = {
    0: 0.97,
    1: 0.95,
    2: 0.90,
    3: 0.85,
    4: 0.80,
    5: 0.75,
    6: 0.70,
    7: 0.65,
    8: 0.60,
    9: 0.55,
    10: 0.50,
}

# Standard deviation of NOAA forecast error (°F) by horizon
# Used for bucket probability estimation via normal CDF
_HORIZON_STDDEV = {
    0: 1.5,
    1: 2.0,
    2: 2.5,
    3: 3.0,
    4: 3.5,
    5: 4.0,
    6: 5.0,
    7: 6.0,
    8: 7.0,
    9: 8.0,
    10: 9.0,
}

# Seasonal adjustment multipliers (month → factor)
# Winter forecasts are harder (storms, cold fronts), summer is more stable
_SEASONAL_FACTORS = {
    1: 0.90, 2: 0.90, 3: 0.95,  # Winter / early spring
    4: 0.95, 5: 1.00, 6: 1.00,  # Spring / early summer
    7: 1.00, 8: 1.00, 9: 1.00,  # Summer / early fall
    10: 0.95, 11: 0.95, 12: 0.90,  # Fall / winter
}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (stdlib, zero dependencies)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def get_horizon_days(forecast_date: str) -> int:
    """Number of days between now (UTC) and the forecast date."""
    try:
        target = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return 999  # Invalid date — will be filtered by max_days_ahead
    now = datetime.now(timezone.utc)
    delta = (target - now).total_seconds() / 86400
    return max(0, round(delta))


def get_noaa_probability(forecast_date: str, apply_seasonal: bool = True) -> float:
    """Dynamic NOAA probability based on forecast horizon and season.

    Returns a probability in [0.0, 1.0] representing how likely the NOAA
    point forecast is to be correct for the given date.
    """
    days_ahead = get_horizon_days(forecast_date)
    # Clamp to our table range, linearly interpolate beyond
    if days_ahead <= 10:
        base_prob = _HORIZON_ACCURACY.get(days_ahead, 0.50)
    else:
        base_prob = max(0.40, 0.50 - 0.02 * (days_ahead - 10))

    if apply_seasonal:
        month = datetime.now(timezone.utc).month
        factor = _SEASONAL_FACTORS.get(month, 1.0)
        base_prob *= factor

    return round(min(base_prob, 0.99), 4)


def _get_stddev(forecast_date: str) -> float:
    """Forecast error standard deviation for the given horizon."""
    days_ahead = get_horizon_days(forecast_date)
    if days_ahead <= 10:
        return _HORIZON_STDDEV.get(days_ahead, 9.0)
    return min(12.0, 9.0 + 0.5 * (days_ahead - 10))


def estimate_bucket_probability(
    forecast_temp: float,
    bucket_low: int,
    bucket_high: int,
    forecast_date: str,
    apply_seasonal: bool = True,
) -> float:
    """Estimate P(actual temperature ∈ [bucket_low, bucket_high]).

    Uses a normal distribution centered on the NOAA forecast temperature,
    with standard deviation based on the forecast horizon.

    Sentinel values: -999 = open below, 999 = open above.
    """
    sigma = _get_stddev(forecast_date)
    if apply_seasonal:
        month = datetime.now(timezone.utc).month
        factor = _SEASONAL_FACTORS.get(month, 1.0)
        # Wider uncertainty in harder seasons (inverse of accuracy boost)
        if factor < 1.0:
            sigma /= factor

    # CDF bounds
    if bucket_low <= -900:
        cdf_low = 0.0
    else:
        cdf_low = _normal_cdf((bucket_low - 0.5 - forecast_temp) / sigma)

    if bucket_high >= 900:
        cdf_high = 1.0
    else:
        cdf_high = _normal_cdf((bucket_high + 0.5 - forecast_temp) / sigma)

    prob = max(0.0, cdf_high - cdf_low)
    return round(prob, 4)
