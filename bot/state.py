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
class TradeRecord:
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    order_id: str = ""
    timestamp: str = ""
    memo: str = ""
    end_date: str = ""
    condition_id: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TradingState:
    trades: dict[str, TradeRecord] = field(default_factory=dict)
    last_run: str = ""
    pnl_history: list[dict] = field(default_factory=list)
    predictions: dict[str, dict] = field(default_factory=dict)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    # Weather-specific state
    previous_forecasts: dict[str, dict] = field(default_factory=dict)
    event_positions: dict[str, str] = field(default_factory=dict)

    def record_trade(self, **kwargs) -> None:
        kwargs.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        rec = TradeRecord(**kwargs)
        self.trades[rec.market_id] = rec

    def remove_trade(self, market_id: str) -> None:
        self.trades.pop(market_id, None)

    def open_positions(self) -> list[TradeRecord]:
        return list(self.trades.values())

    # ── Calibration tracking ──────────────────────────────────────────

    def record_prediction(
        self, market_id: str, our_prob: float, market_price: float,
    ) -> None:
        self.predictions[market_id] = {
            "our_prob": our_prob,
            "market_price": market_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
            "outcome": None,
        }

    def resolve_prediction(self, market_id: str, outcome: bool) -> None:
        if market_id in self.predictions:
            self.predictions[market_id]["resolved"] = True
            self.predictions[market_id]["outcome"] = int(outcome)

    def get_calibration(self) -> dict:
        """Compute Brier + Log score on resolved predictions."""
        from .scoring import brier_score, log_score

        preds, outcomes = [], []
        for p in self.predictions.values():
            if p.get("resolved") and p.get("outcome") is not None:
                preds.append(p["our_prob"])
                outcomes.append(p["outcome"])

        if not preds:
            return {"brier": None, "log": None, "n": 0}

        return {
            "brier": round(brier_score(preds, outcomes), 6),
            "log": round(log_score(preds, outcomes), 6),
            "n": len(preds),
        }

    # ── Closed trade history ─────────────────────────────────────────

    def record_closed_trade(
        self, trade: "TradeRecord", exit_price: float, pnl: float,
    ) -> None:
        """Append a closed trade to pnl_history."""
        self.pnl_history.append({
            "market_id": trade.market_id,
            "token_id": trade.token_id,
            "side": trade.side,
            "entry_price": trade.price,
            "exit_price": exit_price,
            "size": trade.size,
            "pnl": round(pnl, 4),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })

    # ── Daily PnL tracking ───────────────────────────────────────────

    def record_daily_pnl(self, amount: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + amount

    def get_today_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.daily_pnl.get(today, 0.0)

    def prune(self, max_pnl_history: int = 500, max_prediction_days: int = 90) -> None:
        """Remove old entries to prevent unbounded state growth."""
        # Cap pnl_history
        if len(self.pnl_history) > max_pnl_history:
            self.pnl_history = self.pnl_history[-max_pnl_history:]

        # Keep last N resolved predictions + all unresolved
        resolved = [(k, v) for k, v in self.predictions.items() if v.get("resolved")]
        if len(resolved) > max_pnl_history:
            # Sort by timestamp, keep newest
            resolved.sort(key=lambda kv: kv[1].get("timestamp", ""), reverse=True)
            to_remove = {k for k, _ in resolved[max_pnl_history:]}
            for k in to_remove:
                del self.predictions[k]

        # Prune daily_pnl older than 90 days
        if len(self.daily_pnl) > max_prediction_days:
            sorted_days = sorted(self.daily_pnl.keys())
            for day in sorted_days[:-max_prediction_days]:
                del self.daily_pnl[day]

    def save(self, path: str) -> None:
        self.prune()
        data = {
            "trades": {mid: rec.to_dict() for mid, rec in self.trades.items()},
            "last_run": datetime.now(timezone.utc).isoformat(),
            "pnl_history": self.pnl_history,
            "predictions": self.predictions,
            "daily_pnl": self.daily_pnl,
            "previous_forecasts": self.previous_forecasts,
            "event_positions": self.event_positions,
        }
        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except BaseException:
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
            logger.info("No state file at %s, starting fresh", path)
            return cls()
        try:
            with open(p) as f:
                data = json.load(f)
            trades = {
                mid: TradeRecord.from_dict(rec)
                for mid, rec in data.get("trades", {}).items()
            }
            return cls(
                trades=trades,
                last_run=data.get("last_run", ""),
                pnl_history=data.get("pnl_history", []),
                predictions=data.get("predictions", {}),
                daily_pnl=data.get("daily_pnl", {}),
                previous_forecasts=data.get("previous_forecasts", {}),
                event_positions=data.get("event_positions", {}),
            )
        except (json.JSONDecodeError, IOError, KeyError) as exc:
            logger.warning("Failed to load state from %s: %s — starting fresh", path, exc)
            return cls()


@contextlib.contextmanager
def state_lock(state_path: str, retries: int = 3, delay: float = 0.2):
    """Exclusive file lock around state access with brief retry.

    Prevents concurrent bot runs from corrupting state.
    Retries a few times with short delays before giving up.
    """
    import time

    lock_path = state_path + ".lock"
    fd = open(lock_path, "w")
    acquired = False
    try:
        for attempt in range(retries + 1):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                logger.debug("Acquired state lock: %s", lock_path)
                break
            except OSError:
                if attempt < retries:
                    time.sleep(delay)
                    continue
                logger.error("Failed to acquire state lock after %d attempts — another instance running?",
                             retries + 1)
                raise
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
