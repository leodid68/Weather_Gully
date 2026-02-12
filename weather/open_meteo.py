"""Open-Meteo multi-model forecast client (free, no API key).

Fetches GFS and ECMWF forecasts and returns a combined/ensemble view.
"""

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Model weights for ensemble averaging (ECMWF generally more accurate)
MODEL_WEIGHTS = {
    "ecmwf_ifs025": 0.50,
    "gfs_seamless": 0.30,
    "noaa": 0.20,
}

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
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
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


def _timezone_for_lon(lon: float) -> str:
    """Approximate US timezone from longitude."""
    if lon > -82:
        return "America/New_York"
    elif lon > -100:
        return "America/Chicago"
    elif lon > -115:
        return "America/Denver"
    return "America/Los_Angeles"


def get_open_meteo_forecast(
    lat: float,
    lon: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
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
    tz = _timezone_for_lon(lon)
    url = (
        f"{OPEN_METEO_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min"
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
            entry["gfs_high"] = round(gfs_high)
        if gfs_low is not None:
            entry["gfs_low"] = round(gfs_low)

        # ECMWF
        ecmwf_high = _safe_get(daily, "temperature_2m_max_ecmwf_ifs025", i)
        ecmwf_low = _safe_get(daily, "temperature_2m_min_ecmwf_ifs025", i)
        if ecmwf_high is not None:
            entry["ecmwf_high"] = round(ecmwf_high)
        if ecmwf_low is not None:
            entry["ecmwf_low"] = round(ecmwf_low)

        if entry:
            forecasts[date_str] = entry

    logger.info("Open-Meteo: %d days of multi-model forecasts", len(forecasts))
    return forecasts


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
) -> tuple[float | None, float]:
    """Combine NOAA + Open-Meteo into a weighted ensemble forecast.

    Args:
        noaa_temp: NOAA point forecast (may be None).
        open_meteo_data: Open-Meteo data for the date (gfs_high, ecmwf_high, etc.).
        metric: ``"high"`` or ``"low"``.

    Returns:
        ``(ensemble_temp, model_spread)`` where model_spread is the std dev
        across available models (useful for adjusting confidence).
    """
    temps: list[tuple[float, float]] = []  # (temp, weight)

    if noaa_temp is not None:
        temps.append((noaa_temp, MODEL_WEIGHTS.get("noaa", 0.20)))

    if open_meteo_data:
        gfs_key = f"gfs_{metric}"
        ecmwf_key = f"ecmwf_{metric}"

        gfs_val = open_meteo_data.get(gfs_key)
        if gfs_val is not None:
            temps.append((gfs_val, MODEL_WEIGHTS.get("gfs_seamless", 0.30)))

        ecmwf_val = open_meteo_data.get(ecmwf_key)
        if ecmwf_val is not None:
            temps.append((ecmwf_val, MODEL_WEIGHTS.get("ecmwf_ifs025", 0.50)))

    if not temps:
        return None, 0.0

    # Weighted average
    total_weight = sum(w for _, w in temps)
    ensemble = sum(t * w for t, w in temps) / total_weight

    # Model spread (std dev of raw temps — indicates uncertainty)
    raw_temps = [t for t, _ in temps]
    if len(raw_temps) >= 2:
        mean = sum(raw_temps) / len(raw_temps)
        variance = sum((t - mean) ** 2 for t in raw_temps) / len(raw_temps)
        spread = variance ** 0.5
    else:
        spread = 0.0

    return round(ensemble, 1), round(spread, 2)
