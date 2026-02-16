"""Recalibration orchestrator -- CLI entry point for the auto-recalibration pipeline.

Ties together:
  - ``weather.error_cache``  -- load / fetch / prune / save error history
  - ``weather.calibrate``    -- build weighted calibration tables
  - ``weather.guard_rails``  -- clamp parameters within physical bounds

Usage::

    python -m weather.recalibrate --locations NYC,Chicago,Miami,Seattle,Atlanta,Dallas
"""

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .calibrate import build_weighted_calibration_tables
from .error_cache import (
    _DEFAULT_CACHE_PATH,
    fetch_new_errors,
    load_error_cache,
    prune_old_errors,
    save_error_cache,
)
from .guard_rails import clamp_calibration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT = str(Path(__file__).parent / "calibration.json")
_DEFAULT_LOG_DIR = str(Path(__file__).parent / "recalibration_log")
MIN_EFFECTIVE_SAMPLES = 100
WINDOW_DAYS = 90
HALF_LIFE = 30.0

# Default locations (all 6 cities)
_ALL_LOCATIONS = ["NYC", "Chicago", "Miami", "Seattle", "Atlanta", "Dallas"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def filter_window(
    errors: list[dict],
    window_days: int = 90,
    reference_date: str | None = None,
) -> list[dict]:
    """Keep only errors with target_date within the window.

    Args:
        errors: List of error records.
        window_days: Number of days to look back from reference_date.
        reference_date: YYYY-MM-DD string. Defaults to now.

    Returns:
        Filtered list of error records.
    """
    if reference_date is not None:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()

    cutoff = (ref - timedelta(days=window_days)).strftime("%Y-%m-%d")

    return [e for e in errors if e.get("target_date", "") >= cutoff]


def _compute_delta(old_cal: dict, new_cal: dict) -> dict:
    """Compare key parameters between old and new calibration.

    Computes differences for ``global_sigma["0"]`` and
    ``platt_scaling.a`` / ``platt_scaling.b``.

    Returns:
        Dict with ``base_sigma``, ``platt_a``, ``platt_b`` diffs.
        Empty dict if *old_cal* is empty.
    """
    if not old_cal:
        return {}

    old_base = old_cal.get("global_sigma", {}).get("0", 0.0)
    new_base = new_cal.get("global_sigma", {}).get("0", 0.0)

    old_platt = old_cal.get("platt_scaling", {})
    new_platt = new_cal.get("platt_scaling", {})

    return {
        "base_sigma": round(new_base - old_base, 4),
        "platt_a": round(new_platt.get("a", 0.0) - old_platt.get("a", 0.0), 4),
        "platt_b": round(new_platt.get("b", 0.0) - old_platt.get("b", 0.0), 4),
    }


def _load_existing_calibration(path: str) -> dict:
    """Load existing calibration.json, returning {} on any error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, OSError, FileNotFoundError):
        return {}


def _write_atomic(data: dict, path: str) -> None:
    """Write *data* as JSON to *path* atomically (tmpfile + os.replace).

    Creates parent directories if they do not exist.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_log(log_entry: dict, log_dir: str) -> None:
    """Write a recalibration log entry to *log_dir*/YYYY-MM-DD.json."""
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(log_dir, f"{date_str}.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)
    logger.info("Recalibration log saved to %s", log_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_recalibration(
    locations: list[str],
    cache_path: str = _DEFAULT_CACHE_PATH,
    output_path: str = _DEFAULT_OUTPUT,
    log_dir: str = _DEFAULT_LOG_DIR,
    reference_date: str | None = None,
) -> dict:
    """Run the full recalibration pipeline.

    Steps:
      1. Load error cache, fetch new errors, prune old, save.
      2. Filter to WINDOW_DAYS window.
      3. Build weighted calibration tables.
      4. Check effective samples >= MIN_EFFECTIVE_SAMPLES.
      5. Apply guard rails.
      6. Compute delta from previous calibration.
      7. Write calibration.json atomically.
      8. Save recalibration log.

    Args:
        locations: List of location keys to calibrate.
        cache_path: Path to the error history cache file.
        output_path: Path to write calibration.json.
        log_dir: Directory for recalibration log files.
        reference_date: YYYY-MM-DD override for "today" (for testing).

    Returns:
        Dict with ``success`` bool and summary fields.
    """
    # 1. Load -> fetch -> prune -> save
    logger.info("Loading error cache from %s", cache_path)
    cache = load_error_cache(cache_path)

    from datetime import date as _date
    ref_date_obj = (
        _date.fromisoformat(reference_date)
        if reference_date
        else None
    )

    logger.info("Fetching new errors for %s", locations)
    cache = fetch_new_errors(cache, locations, reference_date=ref_date_obj)

    logger.info("Pruning old errors")
    cache = prune_old_errors(cache)

    logger.info("Saving error cache")
    save_error_cache(cache, cache_path)

    all_errors = cache.get("errors", [])
    fetch_stats = {
        "total_errors_in_cache": len(all_errors),
        "locations": locations,
    }

    # 2. Filter to window
    windowed = filter_window(
        all_errors, window_days=WINDOW_DAYS, reference_date=reference_date
    )
    logger.info(
        "Windowed errors: %d / %d (window=%d days)",
        len(windowed), len(all_errors), WINDOW_DAYS,
    )

    # 3. Build weighted calibration tables
    calibration = build_weighted_calibration_tables(
        windowed, locations,
        half_life=HALF_LIFE,
        reference_date=reference_date,
    )

    samples_total = calibration.get("metadata", {}).get("samples", 0)
    samples_effective = calibration.get("metadata", {}).get("samples_effective", 0)

    # 4. Check effective samples
    if samples_effective < MIN_EFFECTIVE_SAMPLES:
        reason_msg = (
            f"samples_effective={samples_effective:.1f} < "
            f"MIN_EFFECTIVE_SAMPLES={MIN_EFFECTIVE_SAMPLES}"
        )
        logger.warning("Insufficient samples: %s — aborting write", reason_msg)
        return {
            "success": False,
            "reason": "insufficient_samples",
            "samples": samples_total,
            "samples_effective": samples_effective,
            "detail": reason_msg,
        }

    # 5. Apply guard rails
    calibration, clamped_list = clamp_calibration(calibration)
    clamped_count = len(clamped_list)
    if clamped_count:
        logger.info("Guard rails clamped %d parameter(s)", clamped_count)

    # 6. Compute delta from previous calibration
    old_cal = _load_existing_calibration(output_path)
    delta = _compute_delta(old_cal, calibration)

    # 7. Write calibration.json atomically
    _write_atomic(calibration, output_path)
    logger.info("Calibration written to %s", output_path)

    # 8. Save recalibration log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "window_days": WINDOW_DAYS,
        "half_life": HALF_LIFE,
        "samples_total": samples_total,
        "samples_effective": samples_effective,
        "params": {
            "base_sigma_global": calibration.get("metadata", {}).get(
                "base_sigma_global"
            ),
            "platt_a": calibration.get("platt_scaling", {}).get("a"),
            "platt_b": calibration.get("platt_scaling", {}).get("b"),
        },
        "clamped": clamped_list,
        "delta": delta,
        "fetch_stats": fetch_stats,
    }
    _save_log(log_entry, log_dir)

    # 9. Return summary
    return {
        "success": True,
        "samples": samples_total,
        "samples_effective": samples_effective,
        "clamped_count": clamped_count,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for recalibration."""
    parser = argparse.ArgumentParser(
        description="Run auto-recalibration pipeline",
    )
    parser.add_argument(
        "--locations",
        type=str,
        default=",".join(_ALL_LOCATIONS),
        help="Comma-separated location keys (default: all 6 cities)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=_DEFAULT_CACHE_PATH,
        help=f"Error cache path (default: {_DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=f"Calibration output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=_DEFAULT_LOG_DIR,
        help=f"Log directory (default: {_DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loc_keys = [loc.strip() for loc in args.locations.split(",")]

    result = run_recalibration(
        locations=loc_keys,
        cache_path=args.cache,
        output_path=args.output,
        log_dir=args.log_dir,
    )

    if result["success"]:
        logger.info(
            "Recalibration complete: %d samples (%.1f effective), "
            "%d clamped, delta=%s",
            result["samples"],
            result["samples_effective"],
            result["clamped_count"],
            result["delta"],
        )
    else:
        logger.warning(
            "Recalibration aborted: %s (%s)",
            result["reason"],
            result.get("detail", ""),
        )


if __name__ == "__main__":
    main()
