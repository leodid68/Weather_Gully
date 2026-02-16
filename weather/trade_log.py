"""Structured trade log for model improvement.

Records (prob_raw, prob_platt, market_price, outcome) per trade,
enabling future Platt re-fitting, threshold tuning, and Kelly calibration.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "trade_log.json"


def log_trade(
    location: str,
    date: str,
    metric: str,
    bucket: tuple[int, int],
    prob_raw: float,
    prob_platt: float,
    market_price: float,
    position_usd: float,
    shares: float,
    forecast_temp: float,
    sigma: float | None = None,
    horizon: int | None = None,
    path: str | None = None,
) -> None:
    """Append a trade entry to the log file.

    Outcome is filled later via :func:`resolve_trades`.
    """
    log_path = Path(path) if path else _DEFAULT_PATH
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "date": date,
        "metric": metric,
        "bucket": list(bucket),
        "prob_raw": round(prob_raw, 5),
        "prob_platt": round(prob_platt, 5),
        "market_price": round(market_price, 4),
        "edge_predicted": round(prob_platt - market_price, 4),
        "position_usd": round(position_usd, 2),
        "shares": round(shares, 2),
        "forecast_temp": round(forecast_temp, 1),
        "sigma": round(sigma, 3) if sigma is not None else None,
        "horizon": horizon,
        "outcome": None,  # filled by resolve_trades
        "pnl": None,
    }
    entries = load_trade_log(str(log_path))
    entries.append(entry)
    _write(entries, log_path)


def resolve_trades(
    actuals: dict[str, dict[str, float]],
    path: str | None = None,
) -> int:
    """Resolve unresolved trades against actual temperatures.

    Args:
        actuals: ``{date: {"high": float, "low": float}}``.
        path: Trade log path.

    Returns:
        Number of trades resolved.
    """
    log_path = Path(path) if path else _DEFAULT_PATH
    entries = load_trade_log(str(log_path))
    resolved = 0
    for entry in entries:
        if entry.get("outcome") is not None:
            continue
        date = entry.get("date", "")
        metric = entry.get("metric", "")
        if date not in actuals or metric not in actuals[date]:
            continue
        actual = actuals[date][metric]
        lo, hi = entry["bucket"]
        won = lo <= actual <= hi
        entry["outcome"] = 1 if won else 0
        entry["actual_temp"] = actual
        price = entry["market_price"]
        entry["pnl"] = round((1.0 - price) if won else -price, 4)
        entry["edge_realized"] = round(entry["outcome"] - price, 4)
        resolved += 1

    if resolved:
        _write(entries, log_path)
        logger.info("Resolved %d trades in trade log", resolved)
    return resolved


def load_trade_log(path: str | None = None) -> list[dict]:
    """Load trade log entries."""
    log_path = Path(path) if path else _DEFAULT_PATH
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _write(entries: list[dict], log_path: Path) -> None:
    """Atomic write of trade log."""
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
