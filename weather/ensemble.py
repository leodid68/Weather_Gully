"""Ensemble forecast data model and disk cache.

Stores per-request ensemble statistics (member temperatures, mean, stddev
per model) and provides a simple JSON disk cache with TTL-based expiry.
Includes an API client to fetch ensemble member spreads from Open-Meteo.
"""

import json
import logging
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._ssl import SSL_CTX as _SSL_CTX

logger = logging.getLogger(__name__)

ENSEMBLE_API_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"
_USER_AGENT = "WeatherGully/1.0"

_CACHE_DIR = Path(__file__).parent / "cache" / "ensemble"


@dataclass
class EnsembleResult:
    """Summary statistics from an ensemble forecast query."""

    member_temps: list[float] = field(default_factory=list)
    ensemble_mean: float = 0.0
    ensemble_stddev: float = 0.0
    ecmwf_stddev: float = 0.0
    gfs_stddev: float = 0.0
    n_members: int = 0

    @classmethod
    def empty(cls) -> "EnsembleResult":
        """Return a default (empty) EnsembleResult."""
        return cls()


def _cache_path(cache_dir: Path, lat: float, lon: float, date: str, metric: str) -> Path:
    """Build the cache file path for a given query.

    Format: ``{cache_dir}/{lat}_{lon}_{date}_{metric}.json``
    """
    return cache_dir / f"{lat}_{lon}_{date}_{metric}.json"


def _write_cache(
    cache_dir: Path,
    lat: float,
    lon: float,
    date: str,
    metric: str,
    result: EnsembleResult,
) -> None:
    """Atomically write an EnsembleResult to the disk cache.

    Creates parent directories if needed.  Uses tempfile + os.replace for
    crash-safe writes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, lat, lon, date, metric)

    data = asdict(result)
    data["_cached_at"] = time.time()

    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.debug("Ensemble cache written: %s", path)


def _read_cache(
    cache_dir: Path,
    lat: float,
    lon: float,
    date: str,
    metric: str,
    ttl_seconds: int = 21600,
) -> EnsembleResult | None:
    """Read an EnsembleResult from the disk cache.

    Returns ``None`` if the file is missing, corrupted, or older than
    *ttl_seconds* (default 6 hours).
    """
    path = _cache_path(cache_dir, lat, lon, date, metric)

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Ensemble cache read failed (%s): %s", path, exc)
        return None

    cached_at = data.pop("_cached_at", None)
    if cached_at is None:
        logger.warning("Ensemble cache missing _cached_at: %s", path)
        return None

    age = time.time() - cached_at
    if age > ttl_seconds:
        logger.debug("Ensemble cache expired (%.0fs old, ttl=%ds): %s", age, ttl_seconds, path)
        return None

    try:
        return EnsembleResult(**data)
    except TypeError as exc:
        logger.warning("Ensemble cache deserialization failed (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _fetch_ensemble_json(
    url: str, timeout: int = 5, max_retries: int = 1
) -> dict | list | None:
    """HTTP GET *url*, parse JSON, return result or ``None`` on failure.

    Uses exponential back-off with jitter between retries.
    """
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            })
            with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < max_retries:
                delay = (2 ** attempt) * (0.5 + random.random())
                logger.warning(
                    "Ensemble API error — retry %d/%d in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.error("Ensemble API failed after %d retries: %s", max_retries, exc)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Ensemble API JSON parse error: %s", exc)
            return None
    return None


def _stddev(values: list[float]) -> float:
    """Bessel-corrected sample standard deviation.

    Returns 0.0 when fewer than 2 values are provided.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def fetch_ensemble_spread(
    lat: float,
    lon: float,
    target_date: str,
    metric: str = "high",
    cache_dir: str | Path | None = None,
    cache_ttl: int = 21600,
) -> EnsembleResult:
    """Fetch ensemble member spread from Open-Meteo for a single date.

    Args:
        lat: Latitude.
        lon: Longitude.
        target_date: ISO date string (``YYYY-MM-DD``).
        metric: ``"high"`` (temperature_2m_max) or ``"low"`` (temperature_2m_min).
        cache_dir: Directory for disk cache. ``None`` disables caching.
        cache_ttl: Cache time-to-live in seconds (default 6 h).

    Returns:
        :class:`EnsembleResult` with member statistics, or ``EnsembleResult.empty()``
        on failure.
    """
    # --- disk cache check ---------------------------------------------------
    effective_cache_dir = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    cached = _read_cache(effective_cache_dir, lat, lon, target_date, metric, ttl_seconds=cache_ttl)
    if cached is not None:
        logger.debug("Ensemble cache hit for %s %s %s %s", lat, lon, target_date, metric)
        return cached

    # --- build API URL -------------------------------------------------------
    daily_var = "temperature_2m_max" if metric == "high" else "temperature_2m_min"
    url = (
        f"{ENSEMBLE_API_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={daily_var}"
        f"&temperature_unit=fahrenheit"
        f"&models=ecmwf_ifs025,gfs025"
        f"&start_date={target_date}&end_date={target_date}"
    )

    raw = _fetch_ensemble_json(url)
    if raw is None:
        return EnsembleResult.empty()

    # --- parse response ------------------------------------------------------
    # Open-Meteo returns a list of entries (one per model) for multi-model
    # queries, or a single dict for single-model queries.
    if isinstance(raw, dict):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        logger.error("Unexpected ensemble API response type: %s", type(raw))
        return EnsembleResult.empty()

    # Match member keys in both old and new API formats:
    #   Old: temperature_2m_max_member01
    #   New: temperature_2m_max_member01_ecmwf_ifs025_ensemble
    #   New: temperature_2m_max_member01_ncep_gefs025
    member_re = re.compile(rf"^{re.escape(daily_var)}_member(\d+)(?:_(.+))?$")

    ecmwf_temps: list[float] = []
    gfs_temps: list[float] = []

    for entry in entries:
        daily = entry.get("daily", {})
        model = entry.get("model", "")

        for key, values in daily.items():
            m = member_re.match(key)
            if m and values:
                temp = values[0]
                if temp is not None:
                    temp = float(temp)
                    # Determine model from key suffix (new format) or entry model field (old format)
                    key_suffix = m.group(2) or ""
                    if "ecmwf" in key_suffix or "ecmwf" in model:
                        ecmwf_temps.append(temp)
                    else:
                        gfs_temps.append(temp)

    all_temps = ecmwf_temps + gfs_temps
    n_members = len(all_temps)

    if n_members == 0:
        logger.warning("Ensemble API returned 0 member temperatures")
        return EnsembleResult.empty()

    ensemble_mean = sum(all_temps) / n_members
    ensemble_stddev = _stddev(all_temps)
    ecmwf_stddev = _stddev(ecmwf_temps)
    gfs_stddev = _stddev(gfs_temps)

    result = EnsembleResult(
        member_temps=all_temps,
        ensemble_mean=round(ensemble_mean, 2),
        ensemble_stddev=round(ensemble_stddev, 2),
        ecmwf_stddev=round(ecmwf_stddev, 2),
        gfs_stddev=round(gfs_stddev, 2),
        n_members=n_members,
    )

    # --- write to cache (best-effort) ----------------------------------------
    try:
        _write_cache(effective_cache_dir, lat, lon, target_date, metric, result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ensemble cache write failed: %s", exc)

    logger.info(
        "Ensemble spread: mean=%.1f stddev=%.2f ecmwf_std=%.2f gfs_std=%.2f n=%d",
        ensemble_mean, ensemble_stddev, ecmwf_stddev, gfs_stddev, n_members,
    )
    return result
