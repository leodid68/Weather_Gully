"""Aviation Weather API client — METAR observations (real-time).

Fetches actual temperature observations from airport weather stations.
These are the same stations used by Polymarket for resolution,
making this the ground-truth data source.
"""

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX as _SSL_CTX

logger = logging.getLogger(__name__)

AVIATION_API_BASE = "https://aviationweather.gov/api/data/metar"
_USER_AGENT = "WeatherGully/1.0"

# Map Polymarket location names to ICAO station identifiers
STATION_MAP: dict[str, str] = {
    "NYC": "KLGA",       # LaGuardia Airport
    "Chicago": "KORD",   # O'Hare International
    "Seattle": "KSEA",   # Seattle-Tacoma International
    "Atlanta": "KATL",   # Hartsfield-Jackson
    "Dallas": "KDFW",    # Dallas/Fort Worth International
    "Miami": "KMIA",     # Miami International
}

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _fetch_json(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> list | None:
    """Fetch JSON from Aviation Weather API with retry on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code in _RETRYABLE_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Aviation API HTTP %d — retry %d/%d in %.1fs",
                    exc.code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Aviation API HTTP %d: %s", exc.code, url)
            return None
        except URLError as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Aviation API URL error — retry %d/%d in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc.reason,
                )
                time.sleep(delay)
                continue
            logger.error("Aviation API URL error: %s", exc.reason)
            return None
        except TimeoutError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Aviation API timeout — retry %d/%d in %.1fs",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Aviation API timeout: %s", url)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Aviation API JSON parse error: %s", exc)
            return None
    return None


def _celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round(c * 9.0 / 5.0 + 32.0, 1)


def get_metar_observations(
    locations: list[str],
    hours: int = 24,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, list[dict]]:
    """Fetch METAR observations for multiple stations in a single API call.

    Args:
        locations: List of canonical location names (e.g. ``["NYC", "Chicago"]``).
        hours: Number of hours of observations to fetch (default 24).
        max_retries: Number of retries on transient failures.
        base_delay: Base delay for exponential backoff.

    Returns:
        ``{location: [{"time": str, "temp_f": float, "raw": str}, ...]}``
        sorted by time ascending within each location.
    """
    # Map locations to station IDs
    stations: dict[str, str] = {}  # station_id → location_name
    for loc in locations:
        station = STATION_MAP.get(loc)
        if station:
            stations[station] = loc
        else:
            logger.warning("No METAR station mapped for location: %s", loc)

    if not stations:
        return {}

    ids_param = ",".join(stations.keys())
    url = f"{AVIATION_API_BASE}?ids={ids_param}&format=json&hours={hours}"

    data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
    if data is None:
        logger.error("Aviation API returned no data")
        return {}

    # Group observations by location
    result: dict[str, list[dict]] = {loc: [] for loc in locations if loc in stations.values()}

    for obs in data:
        station_id = obs.get("icaoId", "")
        loc_name = stations.get(station_id)
        if not loc_name:
            continue

        temp_c = obs.get("temp")
        obs_time = obs.get("reportTime") or obs.get("obsTime")

        if temp_c is None or obs_time is None:
            continue

        try:
            temp_c = float(temp_c)
        except (ValueError, TypeError):
            continue

        result[loc_name].append({
            "time": str(obs_time),
            "temp_f": _celsius_to_fahrenheit(temp_c),
            "temp_c": temp_c,
            "station": station_id,
            "raw": obs.get("rawOb", ""),
        })

    # Sort by time ascending
    for loc in result:
        result[loc].sort(key=lambda x: x["time"])

    total_obs = sum(len(v) for v in result.values())
    logger.info("METAR: %d observations across %d stations", total_obs, len(stations))
    return result


def compute_daily_extremes(
    observations: list[dict],
    target_date: str,
) -> dict | None:
    """Compute daily high/low from METAR observations for a specific date.

    Args:
        observations: List of observation dicts (from ``get_metar_observations``).
        target_date: Date string ``"YYYY-MM-DD"`` to filter observations.

    Returns:
        ``{"high": float, "low": float, "obs_count": int, "latest_obs_time": str}``
        or ``None`` if no observations match the target date.
    """
    day_obs = []
    for obs in observations:
        obs_time = obs.get("time", "")
        # Match observations whose date portion matches target_date
        if obs_time[:10] == target_date:
            day_obs.append(obs)

    if not day_obs:
        return None

    temps = [obs["temp_f"] for obs in day_obs]
    latest_time = max(obs["time"] for obs in day_obs)

    return {
        "high": max(temps),
        "low": min(temps),
        "obs_count": len(day_obs),
        "latest_obs_time": latest_time,
    }


def get_aviation_daily_data(
    locations: list[str],
    hours: int = 24,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """High-level: fetch METAR observations and compute daily extremes.

    Returns:
        ``{location: {date_str: {"obs_high": float, "obs_low": float,
        "obs_count": int, "latest_obs_time": str}}}``
    """
    all_obs = get_metar_observations(
        locations, hours=hours,
        max_retries=max_retries, base_delay=base_delay,
    )

    result: dict[str, dict[str, dict]] = {}

    for loc, observations in all_obs.items():
        if not observations:
            continue

        # Collect all unique dates from observations
        dates = sorted(set(obs["time"][:10] for obs in observations))
        loc_data: dict[str, dict] = {}

        for date_str in dates:
            extremes = compute_daily_extremes(observations, date_str)
            if extremes:
                loc_data[date_str] = {
                    "obs_high": extremes["high"],
                    "obs_low": extremes["low"],
                    "obs_count": extremes["obs_count"],
                    "latest_obs_time": extremes["latest_obs_time"],
                }

        if loc_data:
            result[loc] = loc_data

    logger.info("Aviation daily data: %d locations with observations", len(result))
    return result
