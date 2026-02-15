"""Sigma signal logging for adaptive sigma calibration."""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "sigma_log.json"


def log_sigma_signals(
    location: str,
    date: str,
    metric: str,
    ensemble_stddev: float,
    model_spread: float,
    ema_error: float | None,
    final_sigma: float,
    forecast_temp: float,
    path: str | None = None,
) -> None:
    """Append a sigma signal entry to the log file."""
    log_path = Path(path) if path else _DEFAULT_PATH
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "date": date,
        "metric": metric,
        "ensemble_stddev": ensemble_stddev,
        "model_spread": model_spread,
        "ema_error": ema_error,
        "final_sigma": final_sigma,
        "forecast_temp": forecast_temp,
    }
    entries = load_sigma_log(str(log_path))
    entries.append(entry)
    fd, tmp = tempfile.mkstemp(dir=str(log_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, str(log_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_sigma_log(path: str | None = None) -> list[dict]:
    """Load sigma signal entries from the log file."""
    log_path = Path(path) if path else _DEFAULT_PATH
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
