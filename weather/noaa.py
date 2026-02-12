"""NOAA Weather API client with retry and exponential backoff."""

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

NOAA_API_BASE = "https://api.weather.gov"
_USER_AGENT = "WeatherGully/1.0"

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _fetch_json(
    url: str,
    headers: dict | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict | None:
    """Fetch JSON from *url* with retry on transient errors."""
    hdrs = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
    if headers:
        hdrs.update(headers)

    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers=hdrs)
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code in _RETRYABLE_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "NOAA HTTP %d on %s — retry %d/%d in %.1fs",
                    exc.code, url, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            logger.error("NOAA HTTP %d: %s", exc.code, url)
            return None
        except URLError as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "NOAA URL error %s — retry %d/%d in %.1fs",
                    exc.reason, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            logger.error("NOAA URL error: %s", exc.reason)
            return None
        except TimeoutError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "NOAA timeout on %s — retry %d/%d in %.1fs",
                    url, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            logger.error("NOAA timeout: %s", url)
            return None
    return None


def get_noaa_forecast(
    location: str,
    locations: dict,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict]:
    """Get NOAA forecast for a location.

    Args:
        location: Canonical location key (e.g. ``"NYC"``).
        locations: ``LOCATIONS`` dict from config.
        max_retries: Number of retries on transient failures.
        base_delay: Base delay for exponential backoff.

    Returns:
        ``{date_str: {"high": int|None, "low": int|None}}``
    """
    if location not in locations:
        logger.warning("Unknown location: %s", location)
        return {}

    loc = locations[location]
    points_url = f"{NOAA_API_BASE}/points/{loc['lat']},{loc['lon']}"
    points_data = _fetch_json(points_url, max_retries=max_retries, base_delay=base_delay)

    if not points_data or "properties" not in points_data:
        logger.error("Failed to get NOAA grid for %s", location)
        return {}

    forecast_url = points_data["properties"].get("forecast")
    if not forecast_url:
        logger.error("No forecast URL for %s", location)
        return {}

    forecast_data = _fetch_json(forecast_url, max_retries=max_retries, base_delay=base_delay)
    if not forecast_data or "properties" not in forecast_data:
        logger.error("Failed to get NOAA forecast for %s", location)
        return {}

    periods = forecast_data["properties"].get("periods", [])
    forecasts: dict[str, dict] = {}

    for period in periods:
        start_time = period.get("startTime", "")
        if not start_time:
            continue

        date_str = start_time[:10]
        temp = period.get("temperature")
        is_daytime = period.get("isDaytime", True)

        if date_str not in forecasts:
            forecasts[date_str] = {"high": None, "low": None}

        if is_daytime:
            forecasts[date_str]["high"] = temp
        else:
            forecasts[date_str]["low"] = temp

    logger.info("NOAA forecast for %s: %d days", location, len(forecasts))
    return forecasts
