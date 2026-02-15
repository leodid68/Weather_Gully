"""Ensemble forecast data model and disk cache.

Stores per-request ensemble statistics (member temperatures, mean, stddev
per model) and provides a simple JSON disk cache with TTL-based expiry.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

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
