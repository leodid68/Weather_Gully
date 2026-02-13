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


# ---------------------------------------------------------------------------
# Observation-adjusted probability (METAR integration)
# ---------------------------------------------------------------------------

def _intraday_sigma(latest_obs_time: str, metric: str, utc_offset: float = -5.0) -> float:
    """Dynamic sigma based on how late in the **local** day we have observations.

    The later in the day, the less the temperature can change, so sigma shrinks.

    Args:
        latest_obs_time: ISO 8601 timestamp (e.g. ``"2025-03-15T18:00:00Z"``).
        metric: ``"high"`` or ``"low"``.
        utc_offset: Hours offset from UTC for the station's local time
            (e.g. -5 for NYC, -6 for Chicago, -8 for Seattle).

    Sigma schedule (local time, for "high"):
        Before 10:00 local → 3.0°F  (morning, peak not yet reached)
        10:00–14:00 local  → 2.0°F  (midday, approaching peak)
        14:00–17:00 local  → 1.0°F  (afternoon, near peak)
        After 17:00 local  → 0.5°F  (evening, peak almost certainly passed)

    For "low", the logic is inverted (lows happen in early morning local time).
    """
    try:
        utc_hour = int(latest_obs_time[11:13])
    except (ValueError, IndexError):
        return 3.0  # Default: high uncertainty

    local_hour = (utc_hour + utc_offset) % 24

    if metric == "high":
        if local_hour < 10:
            return 3.0
        elif local_hour < 14:
            return 2.0
        elif local_hour < 17:
            return 1.0
        return 0.5
    else:
        # For lows: minimum typically occurs in early morning (04-08 local)
        if local_hour < 6:
            return 3.0
        elif local_hour < 10:
            return 2.0
        elif local_hour < 14:
            return 1.0
        return 0.5


def constrained_forecast(
    obs_extreme: float,
    model_forecast: float,
    metric: str,
) -> float:
    """Constrain the model forecast using the observed running extreme.

    For "high": the actual daily high cannot be lower than the running
    observed high (temperature may still go up, but can't go down).

    For "low": the actual daily low cannot be higher than the running
    observed low (temperature may still drop, but the low can't increase).
    """
    if metric == "high":
        return max(obs_extreme, model_forecast)
    else:
        return min(obs_extreme, model_forecast)


def _utc_offset_from_lon(lon: float) -> float:
    """Approximate UTC offset from longitude (US stations only)."""
    if lon > -85:
        return -5.0   # Eastern (NYC, Atlanta, Miami)
    elif lon > -100:
        return -6.0   # Central (Chicago, Dallas)
    elif lon > -115:
        return -7.0   # Mountain
    return -8.0        # Pacific (Seattle)


def estimate_bucket_probability_with_obs(
    forecast_temp: float,
    bucket_low: int,
    bucket_high: int,
    forecast_date: str,
    obs_data: dict | None = None,
    metric: str = "high",
    apply_seasonal: bool = True,
    station_lon: float = -74.0,
) -> float:
    """Estimate bucket probability with observation-based uncertainty reduction.

    When ``obs_data`` is provided (from METAR), the forecast is constrained
    by the running observed extreme and sigma is reduced based on the time
    of the latest observation.

    Args:
        forecast_temp: Ensemble forecast temperature.
        bucket_low: Lower bound of the bucket (sentinel -999 = open below).
        bucket_high: Upper bound of the bucket (sentinel 999 = open above).
        forecast_date: Date string ``"YYYY-MM-DD"``.
        obs_data: Optional dict with keys ``obs_high``, ``obs_low``,
            ``latest_obs_time``, ``obs_count``.
        metric: ``"high"`` or ``"low"`` — which extreme this event tracks.
        apply_seasonal: Whether to apply seasonal adjustments.
        station_lon: Longitude of the station (for timezone-aware sigma).

    Returns:
        Probability in [0.0, 1.0].
    """
    if not obs_data or obs_data.get("obs_count", 0) == 0:
        return estimate_bucket_probability(
            forecast_temp, bucket_low, bucket_high,
            forecast_date, apply_seasonal=apply_seasonal,
        )

    latest_obs_time = obs_data.get("latest_obs_time", "")
    obs_key = f"obs_{metric}"
    obs_extreme = obs_data.get(obs_key)

    if obs_extreme is None:
        return estimate_bucket_probability(
            forecast_temp, bucket_low, bucket_high,
            forecast_date, apply_seasonal=apply_seasonal,
        )

    # Constrain forecast by observed extreme
    effective_temp = constrained_forecast(obs_extreme, forecast_temp, metric)

    # Use intraday sigma (tighter than horizon-based sigma on resolution day)
    utc_offset = _utc_offset_from_lon(station_lon)
    sigma = _intraday_sigma(latest_obs_time, metric, utc_offset=utc_offset)

    if apply_seasonal:
        month = datetime.now(timezone.utc).month
        factor = _SEASONAL_FACTORS.get(month, 1.0)
        if factor < 1.0:
            sigma /= factor

    # CDF bounds
    if bucket_low <= -900:
        cdf_low = 0.0
    else:
        cdf_low = _normal_cdf((bucket_low - 0.5 - effective_temp) / sigma)

    if bucket_high >= 900:
        cdf_high = 1.0
    else:
        cdf_high = _normal_cdf((bucket_high + 0.5 - effective_temp) / sigma)

    prob = max(0.0, cdf_high - cdf_low)
    return round(prob, 4)
