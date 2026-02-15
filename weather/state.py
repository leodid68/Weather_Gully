"""Persistent trading state between runs."""

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    """Tracks an order that has been submitted but not yet filled."""
    order_id: str
    market_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    timestamp: str
    token_id: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "market_id": self.market_id,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp,
            "token_id": self.token_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PendingOrder":
        return cls(
            order_id=d["order_id"],
            market_id=d.get("market_id", ""),
            side=d.get("side", "BUY"),
            price=d.get("price", 0.0),
            size=d.get("size", 0.0),
            timestamp=d.get("timestamp", ""),
            token_id=d.get("token_id", ""),
        )


@dataclass
class TradeRecord:
    market_id: str
    outcome_name: str
    side: str
    cost_basis: float
    shares: float
    timestamp: str
    location: str = ""
    forecast_date: str = ""
    forecast_temp: float | None = None
    metric: str = "high"  # "high" or "low"

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "outcome_name": self.outcome_name,
            "side": self.side,
            "cost_basis": self.cost_basis,
            "shares": self.shares,
            "timestamp": self.timestamp,
            "location": self.location,
            "forecast_date": self.forecast_date,
            "forecast_temp": self.forecast_temp,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        return cls(
            market_id=d["market_id"],
            outcome_name=d.get("outcome_name", ""),
            side=d.get("side", "yes"),
            cost_basis=d.get("cost_basis", 0.0),
            shares=d.get("shares", 0.0),
            timestamp=d.get("timestamp", ""),
            location=d.get("location", ""),
            forecast_date=d.get("forecast_date", ""),
            forecast_temp=d.get("forecast_temp"),
            metric=d.get("metric", "high"),
        )


@dataclass
class PredictionRecord:
    """Track a prediction for post-resolution calibration."""
    market_id: str
    event_id: str
    location: str
    forecast_date: str
    metric: str  # "high" or "low"
    our_probability: float
    forecast_temp: float
    bucket_low: int
    bucket_high: int
    timestamp: str = ""
    resolved: bool = False
    actual_outcome: bool | None = None  # True if our bucket won
    fed_to_feedback: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        return cls(
            market_id=d["market_id"],
            event_id=d.get("event_id", ""),
            location=d.get("location", ""),
            forecast_date=d.get("forecast_date", ""),
            metric=d.get("metric", "high"),
            our_probability=d.get("our_probability", 0.0),
            forecast_temp=d.get("forecast_temp", 0.0),
            bucket_low=d.get("bucket_low", 0),
            bucket_high=d.get("bucket_high", 0),
            timestamp=d.get("timestamp", ""),
            resolved=d.get("resolved", False),
            actual_outcome=d.get("actual_outcome"),
            fed_to_feedback=d.get("fed_to_feedback", False),
        )


@dataclass
class TradingState:
    trades: dict[str, TradeRecord] = field(default_factory=dict)  # market_id → TradeRecord
    analyzed_markets: set[str] = field(default_factory=set)
    last_run: str = ""
    # Forecast change detection: location+date → last known temp
    previous_forecasts: dict[str, float] = field(default_factory=dict)
    # Calibration: market_id → PredictionRecord
    predictions: dict[str, PredictionRecord] = field(default_factory=dict)
    # Event tracking for correlation guard: event_id → market_id
    event_positions: dict[str, str] = field(default_factory=dict)
    # Daily METAR observations: "location|date" → obs_data dict
    daily_observations: dict[str, dict] = field(default_factory=dict)
    # Pending orders: order_id → PendingOrder
    pending_orders: dict[str, PendingOrder] = field(default_factory=dict)

    def record_trade(
        self,
        market_id: str,
        outcome_name: str,
        side: str,
        cost_basis: float,
        shares: float,
        location: str = "",
        forecast_date: str = "",
        forecast_temp: float | None = None,
        metric: str = "high",
    ) -> None:
        self.trades[market_id] = TradeRecord(
            market_id=market_id,
            outcome_name=outcome_name,
            side=side,
            cost_basis=cost_basis,
            shares=shares,
            timestamp=datetime.now(timezone.utc).isoformat(),
            location=location,
            forecast_date=forecast_date,
            forecast_temp=forecast_temp,
            metric=metric,
        )

    def remove_trade(self, market_id: str) -> None:
        self.trades.pop(market_id, None)

    def get_cost_basis(self, market_id: str) -> float | None:
        rec = self.trades.get(market_id)
        return rec.cost_basis if rec else None

    def mark_analyzed(self, market_id: str) -> None:
        self.analyzed_markets.add(market_id)

    def was_analyzed(self, market_id: str) -> bool:
        return market_id in self.analyzed_markets

    # -- Forecast change detection --

    def store_forecast(self, location: str, date: str, metric: str, temp: float) -> None:
        """Store a forecast for later change detection."""
        key = f"{location}|{date}|{metric}"
        self.previous_forecasts[key] = temp

    def get_forecast_delta(self, location: str, date: str, metric: str, new_temp: float) -> float | None:
        """Return change from last stored forecast, or None if no previous."""
        key = f"{location}|{date}|{metric}"
        prev = self.previous_forecasts.get(key)
        if prev is None:
            return None
        return new_temp - prev

    # -- Calibration tracking --

    def record_prediction(self, pred: "PredictionRecord") -> None:
        self.predictions[pred.market_id] = pred

    def get_calibration_stats(self) -> dict:
        """Compute Brier-style calibration from resolved predictions."""
        resolved = [p for p in self.predictions.values() if p.resolved and p.actual_outcome is not None]
        if not resolved:
            return {"count": 0, "brier": None, "accuracy": None}
        brier_sum = sum((p.our_probability - (1.0 if p.actual_outcome else 0.0)) ** 2 for p in resolved)
        correct = sum(1 for p in resolved if p.actual_outcome)
        return {
            "count": len(resolved),
            "brier": round(brier_sum / len(resolved), 4),
            "accuracy": round(correct / len(resolved), 4),
        }

    # -- Correlation guard --

    def has_event_position(self, event_id: str) -> bool:
        """Check if we already hold a position in this event."""
        return event_id in self.event_positions

    def record_event_position(self, event_id: str, market_id: str) -> None:
        self.event_positions[event_id] = market_id

    def remove_event_position(self, event_id: str) -> None:
        self.event_positions.pop(event_id, None)

    # -- Daily observations (METAR) --

    def update_daily_obs(self, location: str, date: str, obs_data: dict) -> None:
        """Store or update daily observation data for a location/date."""
        key = f"{location}|{date}"
        self.daily_observations[key] = obs_data

    def get_daily_obs(self, location: str, date: str) -> dict | None:
        """Retrieve stored daily observation data, or None if not available."""
        key = f"{location}|{date}"
        return self.daily_observations.get(key)

    def prune(
        self,
        max_predictions: int = 500,
        max_analyzed: int = 1000,
        max_observations_days: int = 30,
        max_forecasts: int = 500,
    ) -> None:
        """Remove old entries to prevent unbounded state growth."""
        # Cap analyzed_markets (IDs are hex hashes without timestamps;
        # remove arbitrary entries to stay within bounds)
        if len(self.analyzed_markets) > max_analyzed:
            excess = len(self.analyzed_markets) - max_analyzed
            to_remove = list(self.analyzed_markets)[:excess]
            self.analyzed_markets -= set(to_remove)

        # Prune resolved predictions beyond cap
        resolved = [(k, v) for k, v in self.predictions.items() if v.resolved]
        if len(resolved) > max_predictions:
            resolved.sort(key=lambda kv: kv[1].timestamp, reverse=True)
            to_remove = {k for k, _ in resolved[max_predictions:]}
            for k in to_remove:
                del self.predictions[k]

        # Prune old daily_observations
        if len(self.daily_observations) > max_observations_days * 10:
            sorted_keys = sorted(self.daily_observations.keys())
            excess = len(sorted_keys) - max_observations_days * 10
            for k in sorted_keys[:excess]:
                del self.daily_observations[k]

        # Prune old previous_forecasts
        if len(self.previous_forecasts) > max_forecasts:
            sorted_keys = sorted(self.previous_forecasts.keys())
            excess = len(sorted_keys) - max_forecasts
            for k in sorted_keys[:excess]:
                del self.previous_forecasts[k]

    def save(self, path: str) -> None:
        """Atomic save: write to temp file then rename (prevents corruption on crash)."""
        self.prune()
        data = {
            "trades": {mid: rec.to_dict() for mid, rec in self.trades.items()},
            "analyzed_markets": sorted(self.analyzed_markets),
            "last_run": datetime.now(timezone.utc).isoformat(),
            "previous_forecasts": self.previous_forecasts,
            "predictions": {mid: rec.to_dict() for mid, rec in self.predictions.items()},
            "event_positions": self.event_positions,
            "daily_observations": self.daily_observations,
            "pending_orders": {oid: po.to_dict() for oid, po in self.pending_orders.items()},
        }
        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)  # Atomic on POSIX
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TradingState":
        p = Path(path)
        if not p.exists():
            logger.info("No state file found at %s, starting fresh", path)
            return cls()
        try:
            with open(p) as f:
                data = json.load(f)
            trades = {
                mid: TradeRecord.from_dict(rec)
                for mid, rec in data.get("trades", {}).items()
            }
            analyzed = set(data.get("analyzed_markets", []))
            previous_forecasts = data.get("previous_forecasts", {})
            predictions = {
                mid: PredictionRecord.from_dict(rec)
                for mid, rec in data.get("predictions", {}).items()
            }
            event_positions = data.get("event_positions", {})
            daily_observations = data.get("daily_observations", {})
            pending_orders = {
                oid: PendingOrder.from_dict(po)
                for oid, po in data.get("pending_orders", {}).items()
            }
            return cls(
                trades=trades,
                analyzed_markets=analyzed,
                last_run=data.get("last_run", ""),
                previous_forecasts=previous_forecasts,
                predictions=predictions,
                event_positions=event_positions,
                daily_observations=daily_observations,
                pending_orders=pending_orders,
            )
        except (json.JSONDecodeError, IOError, KeyError) as exc:
            logger.warning("Failed to load state from %s: %s — starting fresh", path, exc)
            return cls()


@contextlib.contextmanager
def state_lock(state_path: str):
    """Exclusive file lock around state access (fcntl.flock, LOCK_EX | LOCK_NB).

    Prevents concurrent bot runs from corrupting state.
    """
    lock_path = state_path + ".lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug("Acquired state lock: %s", lock_path)
        yield
    except OSError:
        logger.error("Failed to acquire state lock — another instance running?")
        raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
