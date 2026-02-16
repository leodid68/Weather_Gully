"""Market price mean-reversion timing — track price snapshots, compute
rolling Z-scores, and derive sizing multipliers.

When a bucket's market price is depressed relative to its recent history
(Z < -1), we boost sizing (up to 1.5x).  When elevated (Z > +1), we
reduce sizing (down to 0.5x).  The ``should_favor_exit`` signal is
informational only (logged, not auto-acted upon).

State is persisted in ``price_snapshots_mr.json`` — deleting the file resets.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "price_snapshots_mr.json"

_WINDOW_SIZE = 150
_MIN_SNAPSHOTS = 5


# --------------------------------------------------------------------------
# PriceHistory — per-market rolling price series
# --------------------------------------------------------------------------

@dataclass
class PriceHistory:
    """Rolling price series for a single market key."""

    prices: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)

    def add(self, price: float, timestamp: str) -> None:
        """Append a price snapshot and trim to 2 * _WINDOW_SIZE."""
        self.prices.append(price)
        self.timestamps.append(timestamp)
        max_len = 2 * _WINDOW_SIZE
        if len(self.prices) > max_len:
            self.prices = self.prices[-max_len:]
            self.timestamps = self.timestamps[-max_len:]

    def z_score(self, current_price: float) -> float | None:
        """Compute rolling Z-score of *current_price* against the last window.

        Returns ``None`` if fewer than ``_MIN_SNAPSHOTS`` prices are stored.
        Returns ``0.0`` if the standard deviation is effectively zero.
        """
        window = self.prices[-_WINDOW_SIZE:]
        if len(window) < _MIN_SNAPSHOTS:
            return None

        n = len(window)
        mean = sum(window) / n
        variance = sum((p - mean) ** 2 for p in window) / (n - 1)
        stddev = variance ** 0.5

        if stddev < 0.001:
            return 0.0

        return (current_price - mean) / stddev

    def to_dict(self) -> dict:
        return {"prices": self.prices, "timestamps": self.timestamps}

    @classmethod
    def from_dict(cls, d: dict) -> "PriceHistory":
        return cls(
            prices=d.get("prices", []),
            timestamps=d.get("timestamps", []),
        )


# --------------------------------------------------------------------------
# PriceTracker — keyed collection of PriceHistory
# --------------------------------------------------------------------------

def _market_key(location: str, forecast_date: str, metric: str,
                bucket: tuple[int | float, int | float]) -> str:
    return f"{location}|{forecast_date}|{metric}|{bucket[0]},{bucket[1]}"


@dataclass
class PriceTracker:
    """Tracks price snapshots across all active markets."""

    histories: dict[str, PriceHistory] = field(default_factory=dict)

    # -- recording --

    def record_price(self, location: str, forecast_date: str, metric: str,
                     bucket: tuple[int | float, int | float], price: float,
                     timestamp: str | None = None) -> None:
        """Record a price snapshot for a market."""
        key = _market_key(location, forecast_date, metric, bucket)
        if key not in self.histories:
            self.histories[key] = PriceHistory()
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self.histories[key].add(price, ts)

    # -- Z-score queries --

    def get_z_score(self, location: str, forecast_date: str, metric: str,
                    bucket: tuple[int | float, int | float],
                    current_price: float) -> float | None:
        """Return the rolling Z-score for the given market, or ``None``."""
        key = _market_key(location, forecast_date, metric, bucket)
        hist = self.histories.get(key)
        if hist is None:
            return None
        return hist.z_score(current_price)

    def sizing_multiplier(self, location: str, forecast_date: str, metric: str,
                          bucket: tuple[int | float, int | float],
                          current_price: float) -> float:
        """Return a sizing multiplier in [0.5, 1.5] based on Z-score.

        * Z < -1  (depressed):  ``min(1.5, 1.0 + 0.25 * (-z - 1.0))``
        * Z > +1  (elevated):   ``max(0.5, 1.0 - 0.25 * (z - 1.0))``
        * Otherwise:            ``1.0``
        """
        z = self.get_z_score(location, forecast_date, metric, bucket, current_price)
        if z is None:
            return 1.0
        if z < -1.0:
            return min(1.5, 1.0 + 0.25 * (-z - 1.0))
        if z > 1.0:
            return max(0.5, 1.0 - 0.25 * (z - 1.0))
        return 1.0

    def should_favor_exit(self, location: str, forecast_date: str, metric: str,
                          bucket: tuple[int | float, int | float],
                          current_price: float) -> bool:
        """Return ``True`` when mean-reversion favors exiting (Z > 1.0)."""
        z = self.get_z_score(location, forecast_date, metric, bucket, current_price)
        return z is not None and z > 1.0

    # -- housekeeping --

    def prune(self, max_markets: int = 200) -> None:
        """Remove oldest histories if the tracker exceeds *max_markets*."""
        if len(self.histories) <= max_markets:
            return
        # Sort by most recent timestamp (oldest first), drop excess
        def _latest_ts(item: tuple[str, PriceHistory]) -> str:
            return item[1].timestamps[-1] if item[1].timestamps else ""
        sorted_items = sorted(self.histories.items(), key=_latest_ts)
        excess = len(sorted_items) - max_markets
        keys_to_remove = [k for k, _ in sorted_items[:excess]]
        for k in keys_to_remove:
            del self.histories[k]

    # -- persistence --

    def save(self, path: str | None = None) -> None:
        """Persist price tracker state to JSON (atomic write)."""
        save_path = Path(path) if path else _DEFAULT_PATH
        data = {key: hist.to_dict() for key, hist in self.histories.items()}
        fd, tmp = tempfile.mkstemp(dir=str(save_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(save_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.debug("Price tracker saved to %s (%d markets)", save_path, len(data))

    @classmethod
    def load(cls, path: str | None = None) -> "PriceTracker":
        """Load price tracker state from JSON.  Returns empty state if missing."""
        load_path = Path(path) if path else _DEFAULT_PATH
        tracker = cls()
        if not load_path.exists():
            return tracker
        try:
            with open(load_path) as f:
                data = json.load(f)
            for key, hist_data in data.items():
                tracker.histories[key] = PriceHistory.from_dict(hist_data)
            logger.info("Loaded price tracker: %d markets", len(tracker.histories))
        except (json.JSONDecodeError, IOError, TypeError) as exc:
            logger.warning("Failed to load price tracker: %s — starting fresh", exc)
        return tracker
