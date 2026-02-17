"""Pending maker orders state — file-backed with fcntl locking."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def pending_lock(path: str):
    """Exclusive file lock for pending orders (same pattern as state.state_lock)."""
    lock_path = path + ".lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError:
        logger.error("Failed to acquire pending orders lock")
        raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


class PendingOrders:
    """Thread-safe pending maker orders backed by JSON file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._orders: list[dict] = []

    def load(self) -> None:
        """Load pending orders from disk (call inside pending_lock)."""
        try:
            with open(self.path) as f:
                self._orders = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._orders = []

    def save(self) -> None:
        """Atomic save to disk (call inside pending_lock)."""
        dir_path = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._orders, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add(self, order: dict) -> None:
        """Add a pending order."""
        self._orders.append(order)

    def remove(self, order_id: str) -> dict | None:
        """Remove and return order by order_id, or None if not found."""
        for i, o in enumerate(self._orders):
            if o.get("order_id") == order_id:
                return self._orders.pop(i)
        return None

    def get_by_market(self, market_id: str) -> dict | None:
        """Return pending order for a market, or None."""
        for o in self._orders:
            if o.get("market_id") == market_id:
                return o
        return None

    def has_market(self, market_id: str) -> bool:
        """Check if there's a pending order for this market."""
        return any(o.get("market_id") == market_id for o in self._orders)

    def cleanup_expired(self) -> int:
        """Remove orders whose TTL has elapsed. Returns count removed."""
        now = datetime.now(timezone.utc)
        kept: list[dict] = []
        removed = 0
        for o in self._orders:
            submitted = o.get("submitted_at", "")
            ttl = o.get("ttl_seconds", 900)
            try:
                ts = datetime.fromisoformat(submitted)
                elapsed = (now - ts).total_seconds()
                if elapsed > ttl:
                    removed += 1
                    logger.info("Expired pending order %s (%.0fs old, ttl=%ds)",
                                o.get("order_id", "?"), elapsed, ttl)
                    continue
            except (ValueError, TypeError):
                removed += 1
                continue
            kept.append(o)
        self._orders = kept
        return removed

    def total_exposure(self) -> float:
        """Sum of all pending order amounts."""
        return sum(o.get("amount_usd", 0.0) for o in self._orders)

    @property
    def orders(self) -> list[dict]:
        """Read-only access to current orders."""
        return list(self._orders)

    def __len__(self) -> int:
        return len(self._orders)
