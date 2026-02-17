"""NOAA Weather API client — async with disk cache."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from .http_client import fetch_json

logger = logging.getLogger(__name__)

NOAA_API_BASE = "https://api.weather.gov"
_USER_AGENT = "WeatherGully/1.0"

_CACHE_DIR = Path(__file__).parent / "cache" / "noaa"
_DEFAULT_TTL = 900  # 15 min


def _cache_path(cache_dir: Path, lat: float, lon: float) -> Path:
    return cache_dir / f"{lat:.2f}_{lon:.2f}.json"


def _read_cache(path: Path, ttl: int) -> dict | None:
    try:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def get_noaa_forecast(
    location: str,
    locations: dict,
    max_retries: int = 3,
    base_delay: float = 1.0,
    cache_ttl: int = _DEFAULT_TTL,
    cache_dir: Path | None = None,
) -> dict[str, dict]:
    """Get NOAA forecast for a location.

    Args:
        location: Canonical location key (e.g. ``"NYC"``).
        locations: ``LOCATIONS`` dict from config.
        max_retries: Number of retries on transient failures.
        base_delay: Base delay for exponential backoff.
        cache_ttl: Cache time-to-live in seconds (default 900).
        cache_dir: Override cache directory (useful for tests).

    Returns:
        ``{date_str: {"high": int|None, "low": int|None}}``
    """
    if location not in locations:
        logger.warning("Unknown location: %s", location)
        return {}

    loc = locations[location]
    lat, lon = loc["lat"], loc["lon"]

    # --- cache check ---
    cdir = cache_dir or _CACHE_DIR
    cp = _cache_path(cdir, lat, lon)
    cached = _read_cache(cp, cache_ttl)
    if cached is not None:
        logger.debug("NOAA cache hit for %s (%.2f,%.2f)", location, lat, lon)
        return cached

    # --- points lookup ---
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
    points_url = f"{NOAA_API_BASE}/points/{lat},{lon}"
    points_data = await fetch_json(
        points_url, headers=headers,
        max_retries=max_retries, base_delay=base_delay,
    )

    if not points_data or "properties" not in points_data:
        logger.error("Failed to get NOAA grid for %s", location)
        return {}

    forecast_url = points_data["properties"].get("forecast")
    if not forecast_url:
        logger.error("No forecast URL for %s", location)
        return {}

    # --- forecast fetch ---
    forecast_data = await fetch_json(
        forecast_url, headers=headers,
        max_retries=max_retries, base_delay=base_delay,
    )
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

    # --- cache write ---
    if forecasts:
        _write_cache(cp, forecasts)

    logger.info("NOAA forecast for %s: %d days", location, len(forecasts))
    return forecasts
