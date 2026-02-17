"""PaperBridge — simulated execution wrapper around CLOBWeatherBridge.

Delegates all read operations to the real bridge (Gamma/CLOB) and
intercepts write operations (execute_trade, execute_sell) so that no
real orders are submitted.  Records price snapshots for later use in
backtesting.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .bridge import CLOBWeatherBridge
from .parsing import parse_weather_event, parse_temperature_bucket

logger = logging.getLogger(__name__)


class PaperBridge:
    """Drop-in replacement for CLOBWeatherBridge that simulates execution.

    All reads (markets, orderbooks, context) go through the real bridge.
    All writes (trades, sells) are simulated locally.

    Args:
        real_bridge: An initialised CLOBWeatherBridge.
    """

    def __init__(self, real_bridge: CLOBWeatherBridge) -> None:
        self._real = real_bridge
        # Paper exposure tracking
        self._total_exposure: float = 0.0
        self._position_count: int = 0
        # Paper positions: market_id → {"shares": float, "cost_basis": float}
        self._paper_positions: dict[str, dict] = {}
        # Price snapshots collected during fetch_weather_markets()
        self._snapshots: list[dict] = []

    # ------------------------------------------------------------------
    # Proxied properties (strategy.py accesses these directly)
    # ------------------------------------------------------------------

    @property
    def _market_cache(self):
        return self._real._market_cache

    @property
    def clob(self):
        return self._real.clob

    @property
    def gamma(self):
        return self._real.gamma

    @property
    def max_exposure(self):
        return self._real.max_exposure

    # ------------------------------------------------------------------
    # Delegated read methods
    # ------------------------------------------------------------------

    def fetch_weather_markets(self) -> list[dict]:
        """Delegate to real bridge and record price snapshots."""
        markets = self._real.fetch_weather_markets()
        self._record_snapshots(markets)
        return markets

    def get_market_context(self, market_id: str, **kwargs) -> dict | None:
        return self._real.get_market_context(market_id, **kwargs)

    def get_positions(self) -> list:
        return []

    def get_position(self, market_id: str) -> dict | None:
        """Return paper position for stop-loss checks."""
        pos = self._paper_positions.get(market_id)
        if pos:
            return {"shares_yes": pos["shares"], "shares": pos["shares"]}
        return None

    def get_portfolio(self) -> dict:
        return {
            "balance_usdc": max(0.0, self._real.max_exposure - self._total_exposure),
            "total_exposure": self._total_exposure,
            "positions_count": self._position_count,
        }

    def sync_exposure_from_state(self, trades: dict) -> None:
        total = 0.0
        count = 0
        for trade in trades.values():
            cost_basis = getattr(trade, "cost_basis", 0.0)
            shares = getattr(trade, "shares", 0.0)
            total += cost_basis * shares
            count += 1
            # Rebuild paper positions from state
            market_id = getattr(trade, "market_id", "")
            if market_id:
                self._paper_positions[market_id] = {
                    "shares": shares,
                    "cost_basis": cost_basis,
                }
        self._total_exposure = total
        self._position_count = count
        if count > 0:
            logger.info("PaperBridge: synced exposure from state — $%.2f across %d positions",
                        total, count)

    # ------------------------------------------------------------------
    # Simulated write methods
    # ------------------------------------------------------------------

    def execute_trade(
        self,
        market_id: str,
        side: str,
        amount: float,
        fill_timeout: float = 0.0,
        fill_poll_interval: float = 2.0,
        **kwargs,
    ) -> dict:
        """Simulate a buy trade using cached market data."""
        gm = self._real._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        if side.lower() == "no":
            # NO price ≈ 1 - YES best_bid (neg-risk market)
            yes_bid = gm.best_bid if gm.best_bid > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            price = 1.0 - yes_bid
        else:
            price = gm.best_ask if gm.best_ask > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
        limit_price = kwargs.get("limit_price", 0.0)
        if limit_price > 0 and price > limit_price:
            price = limit_price
        if price <= 0:
            return {"error": "Invalid price", "success": False}

        shares = amount / price

        # Track paper position (weighted average cost basis)
        existing = self._paper_positions.get(market_id, {"shares": 0.0, "cost_basis": 0.0})
        old_shares = existing["shares"]
        new_total_shares = old_shares + shares
        if new_total_shares > 0:
            avg_cost = (old_shares * existing["cost_basis"] + shares * price) / new_total_shares
        else:
            avg_cost = price
        self._paper_positions[market_id] = {
            "shares": new_total_shares,
            "cost_basis": avg_cost,
        }
        self._total_exposure += price * shares
        if old_shares == 0:
            self._position_count += 1

        logger.info(
            "[PAPER] BUY %.1f shares of %s @ $%.4f ($%.2f)",
            shares, market_id[:16], price, amount,
        )

        return {
            "success": True,
            "shares_bought": shares,
            "trade_id": f"paper-{market_id[:8]}-{int(datetime.now(timezone.utc).timestamp())}",
        }

    def execute_sell(
        self,
        market_id: str,
        shares: float,
        fill_timeout: float = 0.0,
        fill_poll_interval: float = 2.0,
    ) -> dict:
        """Simulate a sell trade using cached market data."""
        gm = self._real._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        price = gm.best_bid if gm.best_bid > 0 else (
            gm.outcome_prices[0] if gm.outcome_prices else 0.5
        )
        if price <= 0:
            return {"error": "Invalid bid price", "success": False}

        # Update paper position
        existing = self._paper_positions.get(market_id, {"shares": 0.0, "cost_basis": 0.0})
        remaining = max(0.0, existing["shares"] - shares)
        if remaining <= 0:
            self._paper_positions.pop(market_id, None)
        else:
            self._paper_positions[market_id] = {
                "shares": remaining,
                "cost_basis": existing["cost_basis"],
            }
        self._total_exposure = max(0.0, self._total_exposure - price * shares)
        self._position_count = max(0, self._position_count - 1)

        logger.info(
            "[PAPER] SELL %.1f shares of %s @ $%.4f",
            shares, market_id[:16], price,
        )

        return {
            "success": True,
            "trade_id": f"paper-sell-{market_id[:8]}-{int(datetime.now(timezone.utc).timestamp())}",
        }

    def verify_fill(self, order_id: str, **kwargs) -> dict:
        return {
            "filled": True,
            "partial": False,
            "size_matched": 0.0,
            "original_size": 0.0,
            "status": "PAPER_FILLED",
        }

    def cancel_order(self, order_id: str) -> bool:
        return True

    # ------------------------------------------------------------------
    # Snapshot recording
    # ------------------------------------------------------------------

    def _record_snapshots(self, markets: list[dict]) -> None:
        """Record price snapshots from fetched markets."""
        now = datetime.now(timezone.utc).isoformat()
        for m in markets:
            event_name = m.get("event_name", "")
            event_info = parse_weather_event(event_name)
            outcome_name = m.get("outcome_name", "")
            bucket = parse_temperature_bucket(outcome_name)

            self._snapshots.append({
                "timestamp": now,
                "date": event_info["date"] if event_info else "",
                "location": event_info["location"] if event_info else "",
                "metric": event_info["metric"] if event_info else "",
                "bucket_name": outcome_name,
                "bucket_lo": bucket[0] if bucket else None,
                "bucket_hi": bucket[1] if bucket else None,
                "market_id": m.get("id", ""),
                "best_ask": m.get("best_ask", 0.0),
                "best_bid": m.get("best_bid", 0.0),
                "external_price_yes": m.get("external_price_yes", 0.0),
            })

    def save_snapshots(self, path: str) -> None:
        """Append collected snapshots to a JSON file (atomic write)."""
        if not self._snapshots:
            return

        file_path = Path(path)
        existing: list[dict] = []
        if file_path.exists():
            try:
                with open(file_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        existing.extend(self._snapshots)

        fd, tmp = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(existing, f, indent=2)
            os.replace(tmp, str(file_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        logger.info("Saved %d price snapshots to %s (total: %d)",
                     len(self._snapshots), path, len(existing))
        self._snapshots.clear()
