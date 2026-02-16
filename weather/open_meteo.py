"""Open-Meteo multi-model forecast client (free, no API key).

Fetches GFS and ECMWF forecasts and returns a combined/ensemble view.
"""

import json
import logging
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX as _SSL_CTX

logger = logging.getLogger(__name__)

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Model weights for ensemble averaging (ECMWF generally more accurate)
MODEL_WEIGHTS = {
    "ecmwf_ifs025": 0.50,
    "gfs_seamless": 0.30,
    "noaa": 0.20,
}

_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


def _get_model_weights(location: str = "") -> dict[str, float]:
    """Get model weights, preferring calibrated per-location weights.

    Lookup chain: calibration.json location weights → default MODEL_WEIGHTS.
    """
    if location:
        from .probability import _load_calibration
        cal = _load_calibration()
        if cal:
            loc_weights = cal.get("model_weights", {}).get(location)
            if loc_weights:
                return loc_weights
    return MODEL_WEIGHTS

_MODELS = "gfs_seamless,ecmwf_ifs025"


_USER_AGENT = "WeatherGully/1.0"


def _fetch_json(url: str, max_retries: int = 3, base_delay: float = 1.0) -> dict | None:
    """Fetch JSON with retry."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning("Open-Meteo error — retry %d/%d in %.1fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.error("Open-Meteo failed after %d retries: %s", max_retries, exc)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Open-Meteo JSON parse error: %s", exc)
            return None
    return None


def get_open_meteo_forecast(
    lat: float,
    lon: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
    tz_name: str = "",
) -> dict[str, dict]:
    """Fetch multi-model forecasts from Open-Meteo.

    Returns::

        {
            "2025-03-15": {
                "gfs_high": 52, "gfs_low": 38,
                "ecmwf_high": 54, "ecmwf_low": 37,
            },
            ...
        }
    """
    if not tz_name:
        from .probability import _tz_from_lon
        tz_name = _tz_from_lon(lon)
    tz = tz_name
    # Auxiliary weather variables for sigma adjustment
    _AUX_VARS = (
        "cloud_cover_max,cloud_cover_mean,"
        "wind_speed_10m_max,wind_gusts_10m_max,"
        "precipitation_sum,precipitation_probability_max"
    )
    url = (
        f"{OPEN_METEO_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,{_AUX_VARS}"
        f"&temperature_unit=fahrenheit"
        f"&timezone={tz}"
        f"&models={_MODELS}"
        f"&forecast_days=10"
    )

    data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
    if not data or "daily" not in data:
        logger.error("Open-Meteo returned no daily data")
        return {}

    daily = data["daily"]
    dates = daily.get("time", [])

    forecasts: dict[str, dict] = {}

    for i, date_str in enumerate(dates):
        entry: dict = {}

        # GFS
        gfs_high = _safe_get(daily, "temperature_2m_max_gfs_seamless", i)
        gfs_low = _safe_get(daily, "temperature_2m_min_gfs_seamless", i)
        if gfs_high is not None:
            entry["gfs_high"] = round(gfs_high, 1)
        if gfs_low is not None:
            entry["gfs_low"] = round(gfs_low, 1)

        # ECMWF
        ecmwf_high = _safe_get(daily, "temperature_2m_max_ecmwf_ifs025", i)
        ecmwf_low = _safe_get(daily, "temperature_2m_min_ecmwf_ifs025", i)
        if ecmwf_high is not None:
            entry["ecmwf_high"] = round(ecmwf_high, 1)
        if ecmwf_low is not None:
            entry["ecmwf_low"] = round(ecmwf_low, 1)

        # Auxiliary weather variables (average across models where applicable)
        for aux_key, entry_key in [
            ("cloud_cover_max", "cloud_cover_max"),
            ("cloud_cover_mean", "cloud_cover_mean"),
            ("wind_speed_10m_max", "wind_speed_max"),
            ("wind_gusts_10m_max", "wind_gusts_max"),
            ("precipitation_sum", "precip_sum"),
            ("precipitation_probability_max", "precip_prob_max"),
        ]:
            # Try model-specific keys first, then plain key
            values = []
            for model_suffix in ["_gfs_seamless", "_ecmwf_ifs025"]:
                val = _safe_get(daily, aux_key + model_suffix, i)
                if val is not None:
                    values.append(val)
            if not values:
                val = _safe_get(daily, aux_key, i)
                if val is not None:
                    values.append(val)
            if values:
                entry[entry_key] = round(sum(values) / len(values), 1)

        if entry:
            forecasts[date_str] = entry

    logger.info("Open-Meteo: %d days of multi-model forecasts", len(forecasts))
    return forecasts


def get_open_meteo_forecast_multi(
    locations: dict[str, dict],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """Fetch multi-model forecasts for all locations in minimal API calls.

    Groups locations by timezone (Open-Meteo requires one tz per request),
    then issues one request per timezone group.

    Args:
        locations: Dict of location_name → {"lat": float, "lon": float, "tz": str}.

    Returns:
        {location_name: {date_str: {gfs_high, gfs_low, ecmwf_high, ...}}}
    """
    # Group locations by timezone
    tz_groups: dict[str, list[tuple[str, dict]]] = {}
    for name, loc_data in locations.items():
        tz = loc_data.get("tz", "")
        if not tz:
            from .probability import _tz_from_lon
            tz = _tz_from_lon(loc_data["lon"])
        tz_groups.setdefault(tz, []).append((name, loc_data))

    _AUX_VARS = (
        "cloud_cover_max,cloud_cover_mean,"
        "wind_speed_10m_max,wind_gusts_10m_max,"
        "precipitation_sum,precipitation_probability_max"
    )

    result: dict[str, dict[str, dict]] = {}

    for tz, group in tz_groups.items():
        lats = ",".join(str(loc["lat"]) for _, loc in group)
        lons = ",".join(str(loc["lon"]) for _, loc in group)

        url = (
            f"{OPEN_METEO_BASE}"
            f"?latitude={lats}&longitude={lons}"
            f"&daily=temperature_2m_max,temperature_2m_min,{_AUX_VARS}"
            f"&temperature_unit=fahrenheit"
            f"&timezone={tz}"
            f"&models={_MODELS}"
            f"&forecast_days=10"
        )

        data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
        if not data:
            for name, _ in group:
                result[name] = {}
            continue

        # Multi-location returns a list; single location returns a dict
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "daily" in data:
            entries = [data]
        else:
            for name, _ in group:
                result[name] = {}
            continue

        for idx, (name, _) in enumerate(group):
            if idx >= len(entries):
                result[name] = {}
                continue
            entry_data = entries[idx]
            daily = entry_data.get("daily", {})
            dates = daily.get("time", [])
            forecasts: dict[str, dict] = {}

            for i, date_str in enumerate(dates):
                entry: dict = {}

                gfs_high = _safe_get(daily, "temperature_2m_max_gfs_seamless", i)
                gfs_low = _safe_get(daily, "temperature_2m_min_gfs_seamless", i)
                if gfs_high is not None:
                    entry["gfs_high"] = round(gfs_high, 1)
                if gfs_low is not None:
                    entry["gfs_low"] = round(gfs_low, 1)

                ecmwf_high = _safe_get(daily, "temperature_2m_max_ecmwf_ifs025", i)
                ecmwf_low = _safe_get(daily, "temperature_2m_min_ecmwf_ifs025", i)
                if ecmwf_high is not None:
                    entry["ecmwf_high"] = round(ecmwf_high, 1)
                if ecmwf_low is not None:
                    entry["ecmwf_low"] = round(ecmwf_low, 1)

                for aux_key, entry_key in [
                    ("cloud_cover_max", "cloud_cover_max"),
                    ("cloud_cover_mean", "cloud_cover_mean"),
                    ("wind_speed_10m_max", "wind_speed_max"),
                    ("wind_gusts_10m_max", "wind_gusts_max"),
                    ("precipitation_sum", "precip_sum"),
                    ("precipitation_probability_max", "precip_prob_max"),
                ]:
                    values = []
                    for model_suffix in ["_gfs_seamless", "_ecmwf_ifs025"]:
                        val = _safe_get(daily, aux_key + model_suffix, i)
                        if val is not None:
                            values.append(val)
                    if not values:
                        val = _safe_get(daily, aux_key, i)
                        if val is not None:
                            values.append(val)
                    if values:
                        entry[entry_key] = round(sum(values) / len(values), 1)

                if entry:
                    forecasts[date_str] = entry

            result[name] = forecasts

    total_days = sum(len(v) for v in result.values())
    logger.info("Open-Meteo: %d locations in %d request(s), %d total forecast-days",
                len(result), len(tz_groups), total_days)
    return result


def _safe_get(daily: dict, key: str, index: int) -> float | None:
    """Safely get a value from the daily arrays."""
    arr = daily.get(key)
    if arr and index < len(arr):
        val = arr[index]
        if val is not None:
            return float(val)
    return None


def compute_ensemble_forecast(
    noaa_temp: float | None,
    open_meteo_data: dict | None,
    metric: str,
    aviation_obs_temp: float | None = None,
    aviation_obs_weight: float = 0.0,
    location: str = "",
) -> tuple[float | None, float]:
    """Combine NOAA + Open-Meteo + Aviation observations into a weighted ensemble.

    Args:
        noaa_temp: NOAA point forecast (may be None).
        open_meteo_data: Open-Meteo data for the date (gfs_high, ecmwf_high, etc.).
        metric: ``"high"`` or ``"low"``.
        aviation_obs_temp: Observed temperature from METAR (may be None).
        aviation_obs_weight: Weight for the aviation observation (default 0.0).
        location: Canonical location key for calibrated model weights.

    Returns:
        ``(ensemble_temp, model_spread)`` where model_spread is the std dev
        across available models (useful for adjusting confidence).
    """
    weights = _get_model_weights(location)
    temps: list[tuple[float, float]] = []  # (temp, weight)

    if noaa_temp is not None:
        temps.append((noaa_temp, weights.get("noaa", 0.20)))

    if open_meteo_data:
        gfs_key = f"gfs_{metric}"
        ecmwf_key = f"ecmwf_{metric}"

        gfs_val = open_meteo_data.get(gfs_key)
        if gfs_val is not None:
            temps.append((gfs_val, weights.get("gfs_seamless", 0.30)))

        ecmwf_val = open_meteo_data.get(ecmwf_key)
        if ecmwf_val is not None:
            temps.append((ecmwf_val, weights.get("ecmwf_ifs025", 0.50)))

    if aviation_obs_temp is not None and aviation_obs_weight > 0:
        temps.append((aviation_obs_temp, aviation_obs_weight))

    if not temps:
        return None, 0.0

    # Weighted average
    total_weight = sum(w for _, w in temps)
    ensemble = sum(t * w for t, w in temps) / total_weight

    # Model spread (sample std dev of raw temps — indicates uncertainty)
    raw_temps = [t for t, _ in temps]
    if len(raw_temps) >= 2:
        mean = sum(raw_temps) / len(raw_temps)
        variance = sum((t - mean) ** 2 for t in raw_temps) / (len(raw_temps) - 1)
        spread = variance ** 0.5
    else:
        spread = 0.0

    return round(ensemble, 1), round(spread, 2)
