"""Incremental error history cache for auto-recalibration.

Manages ``weather/error_history.json`` — a persistent store of forecast-vs-actual
error records.  New errors are fetched incrementally (only the date range since
the last fetch), and old records are pruned after ``max_age_days``.

The cache uses METAR actuals (via ``_compute_errors_with_metar``) to match the
Polymarket resolution source.
"""

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from .calibrate import _compute_errors_with_metar
from .config import LOCATIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_FORMAT_VERSION = 1
_DEFAULT_CACHE_PATH = str(Path(__file__).parent / "error_history.json")
_BOOTSTRAP_DAYS = 90
_BUFFER_DAYS = 2  # Skip last 2 days — METAR may be incomplete


def _empty_cache() -> dict:
    """Return a fresh, empty cache structure."""
    return {
        "version": CACHE_FORMAT_VERSION,
        "errors": [],
        "last_fetched": {},
    }


def load_error_cache(path: str = _DEFAULT_CACHE_PATH) -> dict:
    """Load the error cache from disk.

    Returns a valid cache dict.  If the file is missing or contains corrupt
    JSON, returns an empty cache with the current format version.
    """
    if not os.path.exists(path):
        logger.info("Error cache not found at %s — starting fresh", path)
        return _empty_cache()

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, OSError) as exc:
        logger.warning("Corrupt error cache at %s (%s) — starting fresh", path, exc)
        return _empty_cache()

    # Validate minimal structure
    if not isinstance(data, dict) or "version" not in data:
        logger.warning("Invalid error cache structure at %s — starting fresh", path)
        return _empty_cache()

    return data


def save_error_cache(cache: dict, path: str = _DEFAULT_CACHE_PATH) -> None:
    """Write the cache atomically (tmpfile + os.replace)."""
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, path)
        logger.info("Error cache saved to %s (%d records)", path, len(cache.get("errors", [])))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_old_errors(cache: dict, max_age_days: int = 365) -> dict:
    """Remove error records with ``target_date`` older than *max_age_days*.

    Returns the (mutated) cache dict for convenience.
    """
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
    before = len(cache["errors"])
    cache["errors"] = [
        e for e in cache["errors"]
        if e.get("target_date", "9999-99-99") >= cutoff
    ]
    pruned = before - len(cache["errors"])
    if pruned:
        logger.info("Pruned %d old error records (cutoff %s)", pruned, cutoff)
    return cache


def fetch_new_errors(
    cache: dict,
    locations: list[str],
    reference_date: date | None = None,
) -> dict:
    """Fetch new error records for each location and append to cache.

    For each location:
      - ``start_date`` = ``last_fetched[loc] + 1 day``, or ``reference_date -
        BOOTSTRAP_DAYS`` if no prior fetch.
      - ``end_date`` = ``reference_date - BUFFER_DAYS``.
      - If ``start_date > end_date``, the location is already up to date — skip.
      - Unknown locations (not in ``LOCATIONS`` or missing ``station``) are skipped.

    Calls :func:`weather.calibrate._compute_errors_with_metar` for the
    date range, appends new records to ``cache["errors"]``, and updates
    ``cache["last_fetched"][loc]``.

    Returns the (mutated) cache dict.
    """
    if reference_date is None:
        reference_date = date.today()

    end_date = reference_date - timedelta(days=_BUFFER_DAYS)

    for loc in locations:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.warning("Unknown location '%s' — skipping", loc)
            continue

        station = loc_data.get("station")
        if not station:
            logger.warning("No METAR station for '%s' — skipping", loc)
            continue

        last = cache["last_fetched"].get(loc)
        if last:
            start_date = date.fromisoformat(last) + timedelta(days=1)
        else:
            start_date = reference_date - timedelta(days=_BOOTSTRAP_DAYS)

        if start_date > end_date:
            logger.info("Location %s already up to date (last_fetched=%s)", loc, last)
            continue

        logger.info("Fetching errors for %s: %s to %s", loc, start_date, end_date)
        new_errors = _compute_errors_with_metar(
            location=loc,
            lat=loc_data["lat"],
            lon=loc_data["lon"],
            station=station,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            tz_name=loc_data.get("tz", "America/New_York"),
        )

        cache["errors"].extend(new_errors)
        cache["last_fetched"][loc] = end_date.isoformat()
        logger.info("Added %d new error records for %s", len(new_errors), loc)

    return cache
