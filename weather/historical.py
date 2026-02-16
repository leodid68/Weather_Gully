"""Open-Meteo historical forecast and actuals client.

Fetches archived forecasts and ERA5 reanalysis data for calibration.
"""

import json
import logging
import random
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX as _SSL_CTX

logger = logging.getLogger(__name__)

_FORECAST_ARCHIVE_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
_ACTUALS_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
_IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_USER_AGENT = "WeatherGully/1.0"
_MODELS = "gfs_seamless,ecmwf_ifs025"


def _fetch_json(url: str, max_retries: int = 3, base_delay: float = 1.0) -> dict | None:
    """Fetch JSON with retry and exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning("Historical API error — retry %d/%d in %.1fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.error("Historical API failed after %d retries: %s", max_retries, exc)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Historical API JSON parse error: %s", exc)
            return None
    return None


def get_historical_forecasts(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """Fetch model forecasts from Open-Meteo for a historical range.

    Note: Open-Meteo's free API returns the *latest* model output for
    any given date, not the archived run from that date. This means all
    forecasts for a target date are identical regardless of when queried.
    We therefore fetch in large chunks (up to 3 months per request) to
    minimize API calls.

    Returns::

        {
            "2025-01-15": {  # run_date (= target_date since no true horizon)
                "2025-01-15": {"gfs_high": 42, "gfs_low": 30, ...},
            },
            ...
        }
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_forecasts: dict[str, dict[str, dict]] = {}
    _CHUNK_DAYS = 90  # Open-Meteo allows up to ~92 days per request

    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_dt)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        url = (
            f"{_FORECAST_ARCHIVE_BASE}"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&temperature_unit=fahrenheit"
            f"&timezone={tz_name}"
            f"&models={_MODELS}"
            f"&start_date={chunk_start_str}"
            f"&end_date={chunk_end_str}"
        )

        data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
        if data and "daily" in data:
            daily = data["daily"]
            dates = daily.get("time", [])

            for i, target_date in enumerate(dates):
                # Deduplicate: skip dates already fetched from a previous chunk
                if target_date in all_forecasts:
                    continue
                entry: dict = {}
                for model_prefix, model_key in [("gfs", "gfs_seamless"), ("ecmwf", "ecmwf_ifs025")]:
                    high_key = f"temperature_2m_max_{model_key}"
                    low_key = f"temperature_2m_min_{model_key}"
                    high_val = _safe_get(daily, high_key, i)
                    low_val = _safe_get(daily, low_key, i)
                    if high_val is not None:
                        entry[f"{model_prefix}_high"] = round(high_val, 1)
                    if low_val is not None:
                        entry[f"{model_prefix}_low"] = round(low_val, 1)
                if entry:
                    # Store as run_date = target_date (no real horizon distinction)
                    all_forecasts[target_date] = {target_date: entry}

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    logger.info("Historical forecasts: %d target dates loaded", len(all_forecasts))
    return all_forecasts


def get_historical_actuals(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict]:
    """Fetch ERA5 reanalysis actuals (ground truth) from Open-Meteo.

    Returns::

        {
            "2025-01-15": {"high": 43.2, "low": 29.5},
            "2025-01-16": {"high": 45.1, "low": 32.0},
            ...
        }
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    actuals: dict[str, dict] = {}
    _CHUNK_DAYS = 90  # Open-Meteo allows up to ~92 days per request

    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_dt)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        url = (
            f"{_ACTUALS_ARCHIVE_BASE}"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&temperature_unit=fahrenheit"
            f"&timezone={tz_name}"
            f"&start_date={chunk_start_str}"
            f"&end_date={chunk_end_str}"
        )

        data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
        if data and "daily" in data:
            daily = data["daily"]
            dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])
            lows = daily.get("temperature_2m_min", [])

            for i, date_str in enumerate(dates):
                if date_str in actuals:
                    continue
                entry: dict = {}
                if i < len(highs) and highs[i] is not None:
                    entry["high"] = round(float(highs[i]), 1)
                if i < len(lows) and lows[i] is not None:
                    entry["low"] = round(float(lows[i]), 1)
                if entry:
                    actuals[date_str] = entry

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    logger.info("Historical actuals: %d days loaded", len(actuals))
    return actuals


def _safe_get(daily: dict, key: str, index: int) -> float | None:
    """Safely get a value from the daily arrays."""
    arr = daily.get(key)
    if arr and index < len(arr):
        val = arr[index]
        if val is not None:
            return float(val)
    return None


def _fetch_metar_csv(url: str, max_retries: int = 3, base_delay: float = 1.0) -> str | None:
    """Fetch CSV text from IEM METAR archive with retry and exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "text/csv",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return resp.read().decode()
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning("IEM METAR API error — retry %d/%d in %.1fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.error("IEM METAR API failed after %d retries: %s", max_retries, exc)
            return None
    return None


def get_historical_metar_actuals(
    station: str,
    start_date: str,
    end_date: str,
    tz_name: str = "America/New_York",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict]:
    """Fetch METAR daily high/low from Iowa Environmental Mesonet (IEM).

    Returns::

        {
            "2025-01-15": {"high": 41.0, "low": 28.1},
            ...
        }

    Observations with temps < -60 or > 140 are rejected as bad readings.
    Days with fewer than 4 valid observations are skipped.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    tz_encoded = tz_name.replace("/", "%2F")
    url = (
        f"{_IEM_BASE}"
        f"?station={station}"
        f"&data=tmpf"
        f"&tz={tz_encoded}"
        f"&format=onlycomma"
        f"&latlon=no"
        f"&elev=no"
        f"&missing=empty"
        f"&trace=empty"
        f"&direct=no"
        f"&report_type=3"
        f"&year1={start_dt.year}&month1={start_dt.month}&day1={start_dt.day}"
        f"&year2={end_dt.year}&month2={end_dt.month}&day2={end_dt.day}"
    )

    csv_text = _fetch_metar_csv(url, max_retries=max_retries, base_delay=base_delay)
    if not csv_text:
        return {}

    # Parse CSV: group temperatures by date
    daily_temps: dict[str, list[float]] = {}
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return {}

    # First line is the header: station,valid,tmpf
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue

        date_str = parts[1][:10]  # Extract YYYY-MM-DD from "valid" column
        temp_str = parts[2].strip()
        if not temp_str:
            continue

        try:
            temp = float(temp_str)
        except ValueError:
            continue

        # Reject bad readings
        if temp < -60 or temp > 140:
            continue

        daily_temps.setdefault(date_str, []).append(temp)

    # Compute daily high/low, requiring at least 4 observations
    actuals: dict[str, dict] = {}
    for date_str, temps in daily_temps.items():
        if len(temps) < 4:
            continue
        actuals[date_str] = {
            "high": round(max(temps), 1),
            "low": round(min(temps), 1),
        }

    logger.info("METAR actuals: %d days loaded for station %s", len(actuals), station)
    return actuals
