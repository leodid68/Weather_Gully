"""Weather Underground forecast client — async with disk cache."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from .http_client import fetch_json

logger = logging.getLogger(__name__)

WU_API_BASE = "https://api.weather.com/v3/wx/forecast/daily/5day"
_CACHE_DIR = Path(__file__).parent / "cache" / "wu"
_DEFAULT_TTL = 3600  # 60 min


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


async def get_wu_forecast(
    lat: float,
    lon: float,
    api_key: str,
    cache_dir: Path | None = None,
    cache_ttl: int = _DEFAULT_TTL,
) -> dict[str, dict] | None:
    """Fetch 5-day WU forecast. Returns {date_str: {"high": F, "low": F}} or None.

    Uses disk cache to stay within 500 calls/day free tier.
    """
    if not api_key:
        return None

    cache_dir = cache_dir or _CACHE_DIR
    cp = _cache_path(cache_dir, lat, lon)
    cached = _read_cache(cp, cache_ttl)
    if cached is not None:
        logger.debug("WU cache hit for %.2f,%.2f", lat, lon)
        return cached

    url = WU_API_BASE
    params = {
        "geocode": f"{lat:.4f},{lon:.4f}",
        "format": "json",
        "units": "e",  # imperial (°F)
        "language": "en-US",
        "apiKey": api_key,
    }

    data = await fetch_json(url, params=params, timeout=15, max_retries=2)
    if not data:
        logger.warning("WU forecast failed for %.2f,%.2f", lat, lon)
        return None

    # Parse TWC response
    try:
        dates = data.get("validTimeLocal", [])
        highs = data.get("temperatureMax", [])
        lows = data.get("temperatureMin", [])

        result: dict[str, dict] = {}
        for i, date_str in enumerate(dates):
            if not date_str:
                continue
            day = date_str[:10]  # "2026-02-18T07:00:00-0500" → "2026-02-18"
            high = highs[i] if i < len(highs) and highs[i] is not None else None
            low = lows[i] if i < len(lows) and lows[i] is not None else None
            if high is not None or low is not None:
                result[day] = {"high": high, "low": low}

        if result:
            _write_cache(cp, result)
            logger.info("WU forecast: %d days for %.2f,%.2f", len(result), lat, lon)

        return result or None

    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("WU parse error: %s", exc)
        return None
