"""Open-Meteo Previous Runs API client for horizon-dependent forecasts.

Fetches what models predicted N days before each target date, using
hourly data converted to daily max/min with timezone-aware boundaries.
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

_PREVIOUS_RUNS_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
_USER_AGENT = "WeatherGully/1.0"
_MODELS = "gfs_seamless,ecmwf_ifs025"
_CHUNK_DAYS = 90
_MIN_HOURS_PER_DAY = 12


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
                logger.warning(
                    "Previous Runs API error — retry %d/%d in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.error(
                "Previous Runs API failed after %d retries: %s",
                max_retries, exc,
            )
            return None
        except json.JSONDecodeError as exc:
            logger.error("Previous Runs API JSON parse error: %s", exc)
            return None
    return None


# ── Timezone helpers ────────────────────────────────────────────────

_TZ_OFFSETS: dict[str, int] = {
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Los_Angeles": -8,
}


def _tz_offset_for_location(tz_name: str) -> int:
    """Map US timezone name to standard UTC offset (hours).

    Uses standard time (not DST) for consistency — this is accurate
    enough for daily max/min computation.

    Returns -5 (Eastern) as default for unknown timezones.
    """
    return _TZ_OFFSETS.get(tz_name, -5)


# ── Hourly → daily conversion ──────────────────────────────────────

def _hourly_to_daily_max_min(
    times: list[str],
    values: list[float | None],
    tz_offset: int,
) -> dict[str, tuple[float, float]]:
    """Convert hourly temperature readings to daily (max, min).

    Groups hourly readings by local date (applying *tz_offset* hours
    to the UTC timestamps).  Days with fewer than ``_MIN_HOURS_PER_DAY``
    valid readings are excluded from the result.

    Args:
        times: ISO datetime strings in UTC (e.g. ``"2025-01-15T00:00"``).
        values: Hourly temperature values; ``None`` entries are skipped.
        tz_offset: UTC offset in hours (e.g. ``-5`` for US Eastern).

    Returns:
        ``{date_str: (daily_max, daily_min)}``
    """
    daily_temps: dict[str, list[float]] = {}

    for iso_time, val in zip(times, values):
        if val is None:
            continue

        # Parse UTC time and apply offset to get local date
        utc_dt = datetime.strptime(iso_time[:16], "%Y-%m-%dT%H:%M")
        local_dt = utc_dt + timedelta(hours=tz_offset)
        date_str = local_dt.strftime("%Y-%m-%d")

        daily_temps.setdefault(date_str, []).append(val)

    result: dict[str, tuple[float, float]] = {}
    for date_str, temps in daily_temps.items():
        if len(temps) < _MIN_HOURS_PER_DAY:
            continue
        result[date_str] = (round(max(temps), 1), round(min(temps), 1))

    return result


# ── Main fetch function ─────────────────────────────────────────────

def _hourly_var(horizon: int) -> str:
    """Return the hourly variable name for a given horizon.

    Horizon 0 → ``temperature_2m``
    Horizon N → ``temperature_2m_previous_dayN``
    """
    if horizon == 0:
        return "temperature_2m"
    return f"temperature_2m_previous_day{horizon}"


def fetch_previous_runs(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    horizons: list[int] | None = None,
    tz_name: str = "America/New_York",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[int, dict[str, dict[str, float]]]:
    """Fetch horizon-dependent hourly forecasts and return daily max/min.

    Queries the Previous Runs API for each requested horizon, chunking
    date ranges into 90-day windows.  Hourly data is converted to daily
    max/min using timezone-aware boundaries.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: Start date (``YYYY-MM-DD``).
        end_date: End date (``YYYY-MM-DD``).
        horizons: List of forecast horizons (days ahead), e.g. ``[0, 1, 3, 7]``.
            Defaults to ``[0, 1, 2, 3, 5, 7]``.
        tz_name: IANA timezone name for daily boundary computation.
        max_retries: Max HTTP retries per request.
        base_delay: Base delay for exponential backoff.

    Returns::

        {
            0: {
                "2025-01-15": {"gfs_high": 42.0, "gfs_low": 30.1,
                                "ecmwf_high": 44.0, "ecmwf_low": 29.8},
                ...
            },
            1: { ... },
            ...
        }
    """
    if horizons is None:
        horizons = [0, 1, 2, 3, 5, 7]

    tz_offset = _tz_offset_for_location(tz_name)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Build the hourly variable list for all horizons at once
    hourly_vars = []
    for h in horizons:
        hourly_vars.append(_hourly_var(h))
    hourly_param = ",".join(hourly_vars)

    result: dict[int, dict[str, dict[str, float]]] = {h: {} for h in horizons}

    current = start_dt
    first_chunk = True
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_dt)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        # Rate limiting between chunks
        if not first_chunk:
            time.sleep(1)
        first_chunk = False

        url = (
            f"{_PREVIOUS_RUNS_BASE}"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly={hourly_param}"
            f"&models={_MODELS}"
            f"&temperature_unit=fahrenheit"
            f"&start_date={chunk_start_str}"
            f"&end_date={chunk_end_str}"
        )

        data = _fetch_json(url, max_retries=max_retries, base_delay=base_delay)
        if not data or "hourly" not in data:
            logger.warning(
                "Previous Runs API returned no hourly data for %s to %s",
                chunk_start_str, chunk_end_str,
            )
            current = chunk_end + timedelta(days=1)
            continue

        hourly = data["hourly"]
        times = hourly.get("time", [])

        for h in horizons:
            var_name = _hourly_var(h)

            for model_prefix, model_key in [("gfs", "gfs_seamless"), ("ecmwf", "ecmwf_ifs025")]:
                col_key = f"{var_name}_{model_key}"
                values = hourly.get(col_key, [])

                if not values:
                    continue

                daily = _hourly_to_daily_max_min(times, values, tz_offset)

                for date_str, (day_max, day_min) in daily.items():
                    if date_str not in result[h]:
                        result[h][date_str] = {}
                    result[h][date_str][f"{model_prefix}_high"] = day_max
                    result[h][date_str][f"{model_prefix}_low"] = day_min

        current = chunk_end + timedelta(days=1)

    total_days = sum(len(dates) for dates in result.values())
    logger.info(
        "Previous Runs: %d horizons, %d total date entries loaded",
        len(horizons), total_days,
    )
    return result
