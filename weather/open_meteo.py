"""Open-Meteo multi-model forecast client (free, no API key).

Fetches GFS, ECMWF and optional regional NWP model forecasts
and returns a combined/ensemble view.
"""

import logging
import time
from pathlib import Path

from .http_client import fetch_json

logger = logging.getLogger(__name__)

_forecast_cache: dict[str, tuple[dict, float]] = {}  # key → (result_data, timestamp)
_CACHE_TTL = 900  # 15 minutes in seconds


def _cache_key(coords: str, tz: str) -> str:
    """Deterministic cache key from coordinates and timezone."""
    return f"{coords}|{tz}"


OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Model weights for ensemble averaging
MODEL_WEIGHTS = {
    "ecmwf_ifs025": 0.50,
    "gfs_seamless": 0.30,
    "noaa": 0.20,
    "local": 0.20,  # Regional NWP model weight (international cities)
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

_BASE_MODELS = (
    "gfs_seamless,ecmwf_ifs025,"
    "ukmo_seamless,jma_seamless,arpege_seamless,gem_seamless,bom_access_global"
)


def _models_str(local_model: str = "") -> str:
    """Build the ``models=`` parameter, optionally including a regional NWP model."""
    if local_model:
        return f"{_BASE_MODELS},{local_model}"
    return _BASE_MODELS


async def get_open_meteo_forecast(
    lat: float,
    lon: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
    tz_name: str = "",
    local_model: str = "",
) -> dict[str, dict]:
    """Fetch multi-model forecasts from Open-Meteo.

    Returns::

        {
            "2025-03-15": {
                "gfs_high": 52, "gfs_low": 38,
                "ecmwf_high": 54, "ecmwf_low": 37,
                "local_high": 53, "local_low": 36,  # if local_model set
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
    models = _models_str(local_model)
    url = (
        f"{OPEN_METEO_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,{_AUX_VARS}"
        f"&temperature_unit=fahrenheit"
        f"&timezone={tz}"
        f"&models={models}"
        f"&forecast_days=10"
    )

    data = await fetch_json(url, max_retries=max_retries, base_delay=base_delay)
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

        # Additional global models
        for model_name in ("ukmo_seamless", "jma_seamless", "arpege_seamless",
                           "gem_seamless", "bom_access_global"):
            m_high = _safe_get(daily, f"temperature_2m_max_{model_name}", i)
            m_low = _safe_get(daily, f"temperature_2m_min_{model_name}", i)
            if m_high is not None:
                entry[f"{model_name}_high"] = round(m_high, 1)
            if m_low is not None:
                entry[f"{model_name}_low"] = round(m_low, 1)

        # Regional NWP model (international cities)
        if local_model:
            local_high = _safe_get(daily, f"temperature_2m_max_{local_model}", i)
            local_low = _safe_get(daily, f"temperature_2m_min_{local_model}", i)
            if local_high is not None:
                entry["local_high"] = round(local_high, 1)
            if local_low is not None:
                entry["local_low"] = round(local_low, 1)

        # Auxiliary weather variables (average across models where applicable)
        model_suffixes = ["_gfs_seamless", "_ecmwf_ifs025",
                          "_ukmo_seamless", "_jma_seamless", "_arpege_seamless",
                          "_gem_seamless", "_bom_access_global"]
        if local_model:
            model_suffixes.append(f"_{local_model}")
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
            for model_suffix in model_suffixes:
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

    if local_model:
        logger.info("Open-Meteo: %d days of multi-model forecasts (local=%s)",
                     len(forecasts), local_model)
    else:
        logger.info("Open-Meteo: %d days of multi-model forecasts", len(forecasts))
    return forecasts


async def get_open_meteo_forecast_multi(
    locations: dict[str, dict],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """Fetch multi-model forecasts for all locations in minimal API calls.

    Groups locations by ``(timezone, local_model)`` — Open-Meteo requires
    one tz per request and the ``models=`` list is shared across locations
    in the same request.

    Args:
        locations: Dict of location_name → {"lat", "lon", "tz", "local_model"?}.

    Returns:
        {location_name: {date_str: {gfs_high, gfs_low, ecmwf_high, ..., local_high?}}}
    """
    # Group locations by (timezone, local_model) so each request has a
    # consistent models= parameter.
    groups: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for name, loc_data in locations.items():
        tz = loc_data.get("tz", "")
        if not tz:
            from .probability import _tz_from_lon
            tz = _tz_from_lon(loc_data["lon"])
        lm = loc_data.get("local_model", "")
        groups.setdefault((tz, lm), []).append((name, loc_data))

    _AUX_VARS = (
        "cloud_cover_max,cloud_cover_mean,"
        "wind_speed_10m_max,wind_gusts_10m_max,"
        "precipitation_sum,precipitation_probability_max"
    )

    result: dict[str, dict[str, dict]] = {}

    for (tz, local_model), group in groups.items():
        lats = ",".join(str(loc["lat"]) for _, loc in group)
        lons = ",".join(str(loc["lon"]) for _, loc in group)

        # --- TTL cache check ---
        cache_k = _cache_key(f"{lats},{lons}", f"{tz}|{local_model}")
        cached = _forecast_cache.get(cache_k)
        if cached is not None:
            cached_data, cached_at = cached
            if time.time() - cached_at < _CACHE_TTL:
                logger.info("Open-Meteo cache hit for tz=%s (%d locations)",
                            tz, len(group))
                result.update(cached_data)
                continue

        models = _models_str(local_model)
        url = (
            f"{OPEN_METEO_BASE}"
            f"?latitude={lats}&longitude={lons}"
            f"&daily=temperature_2m_max,temperature_2m_min,{_AUX_VARS}"
            f"&temperature_unit=fahrenheit"
            f"&timezone={tz}"
            f"&models={models}"
            f"&forecast_days=10"
        )

        data = await fetch_json(url, max_retries=max_retries, base_delay=base_delay)
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

        group_results: dict[str, dict[str, dict]] = {}

        for idx, (name, _) in enumerate(group):
            if idx >= len(entries):
                group_results[name] = {}
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

                # Additional global models
                for model_name in ("ukmo_seamless", "jma_seamless", "arpege_seamless",
                                   "gem_seamless", "bom_access_global"):
                    m_high = _safe_get(daily, f"temperature_2m_max_{model_name}", i)
                    m_low = _safe_get(daily, f"temperature_2m_min_{model_name}", i)
                    if m_high is not None:
                        entry[f"{model_name}_high"] = round(m_high, 1)
                    if m_low is not None:
                        entry[f"{model_name}_low"] = round(m_low, 1)

                # Regional NWP model
                if local_model:
                    lm_high = _safe_get(daily, f"temperature_2m_max_{local_model}", i)
                    lm_low = _safe_get(daily, f"temperature_2m_min_{local_model}", i)
                    if lm_high is not None:
                        entry["local_high"] = round(lm_high, 1)
                    if lm_low is not None:
                        entry["local_low"] = round(lm_low, 1)

                model_suffixes = ["_gfs_seamless", "_ecmwf_ifs025",
                                  "_ukmo_seamless", "_jma_seamless", "_arpege_seamless",
                                  "_gem_seamless", "_bom_access_global"]
                if local_model:
                    model_suffixes.append(f"_{local_model}")
                for aux_key, entry_key in [
                    ("cloud_cover_max", "cloud_cover_max"),
                    ("cloud_cover_mean", "cloud_cover_mean"),
                    ("wind_speed_10m_max", "wind_speed_max"),
                    ("wind_gusts_10m_max", "wind_gusts_max"),
                    ("precipitation_sum", "precip_sum"),
                    ("precipitation_probability_max", "precip_prob_max"),
                ]:
                    values = []
                    for model_suffix in model_suffixes:
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

            group_results[name] = forecasts

        # Store in cache and merge into result
        _forecast_cache[cache_k] = (group_results, time.time())
        result.update(group_results)

    total_days = sum(len(v) for v in result.values())
    logger.info("Open-Meteo: %d locations in %d request(s), %d total forecast-days",
                len(result), len(groups), total_days)
    return result


def get_dominant_model_info(
    location: str,
    noaa_temp: float | None,
    om_data: dict | None,
    metric: str = "high",
) -> tuple[str, float | None, float]:
    """Return (model_name, model_temp, model_weight) for the highest-weight model.

    Returns ("", None, 0.0) if no model has weight >= 0.4.
    """
    weights = _get_model_weights(location)
    best_name, best_weight = "", 0.0
    for name, w in weights.items():
        if w > best_weight:
            best_name, best_weight = name, w

    if best_weight < 0.4:
        return "", None, 0.0

    # Get the dominant model's temperature
    if best_name == "noaa" and noaa_temp is not None:
        return best_name, noaa_temp, best_weight
    if om_data:
        if best_name == "gfs_seamless":
            key = f"gfs_{metric}"
        elif best_name == "ecmwf_ifs025":
            key = f"ecmwf_{metric}"
        else:
            key = f"local_{metric}"
        temp = om_data.get(key)
        if temp is not None:
            return best_name, temp, best_weight
    return best_name, None, best_weight


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
    """Combine NOAA + Open-Meteo + local NWP + Aviation into a weighted ensemble.

    For US cities the third source is NOAA; for international cities it is a
    regional NWP model (``local_high`` / ``local_low`` in *open_meteo_data*).

    Args:
        noaa_temp: NOAA point forecast (may be None).
        open_meteo_data: Open-Meteo data for the date (gfs_high, ecmwf_high,
            local_high?, etc.).
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

        # Regional NWP model (international cities — replaces NOAA role)
        local_val = open_meteo_data.get(f"local_{metric}")
        if local_val is not None:
            temps.append((local_val, weights.get("local", 0.20)))

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
