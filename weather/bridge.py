"""CLOBWeatherBridge — CLOB + Gamma adapter for weather/strategy.py.

Provides the semantic interface that weather/strategy.py expects,
routing all operations through the Polymarket CLOB and Gamma APIs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.gamma import GammaClient, GammaMarket
    from polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)


class CLOBWeatherBridge:
    """Adapter: Polymarket CLOB + Gamma API for weather strategy.

    Args:
        clob_client: Authenticated (or public) PolymarketClient for order
            submission and orderbook queries.
        gamma_client: GammaClient for market discovery and metadata.
        max_exposure: Maximum total exposure in USD (used for portfolio).
    """

    def __init__(
        self,
        clob_client: PolymarketClient,
        gamma_client: GammaClient,
        max_exposure: float = 50.0,
    ):
        self.clob = clob_client
        self.gamma = gamma_client
        self.max_exposure = max_exposure
        # Cache: condition_id → GammaMarket (populated by fetch_weather_markets)
        self._market_cache: dict[str, GammaMarket] = {}
        # Exposure tracking (deducted from max_exposure for portfolio)
        self._total_exposure: float = 0.0
        self._position_count: int = 0
        self._known_positions: set[str] = set()  # market_ids with open positions

    def sync_exposure_from_state(self, trades: dict) -> None:
        """Initialize exposure tracking from persisted weather state trades.

        Args:
            trades: Dict of market_id → TradeRecord (from weather.state).
        """
        total = 0.0
        count = 0
        for trade in trades.values():
            cost_basis = getattr(trade, "cost_basis", 0.0)
            shares = getattr(trade, "shares", 0.0)
            total += cost_basis * shares
            count += 1
        self._total_exposure = total
        self._position_count = count
        self._known_positions = set()
        for trade in trades.values():
            market_id = getattr(trade, "market_id", "")
            if market_id:
                self._known_positions.add(market_id)
        if count > 0:
            logger.info("Bridge: synced exposure from state — $%.2f across %d positions",
                        total, count)

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def fetch_weather_markets(self) -> list[dict]:
        """Fetch active weather markets via Gamma API.

        Returns list of dicts compatible with the weather strategy:
            {id, event_id, event_name, outcome_name, external_price_yes,
             token_id_yes, token_id_no, end_date, best_bid, best_ask, status}
        """
        events, gamma_markets = self.gamma.fetch_events_with_markets(
            tag_slug="weather", limit=100,
        )

        markets: list[dict] = []
        for gm in gamma_markets:
            if gm.closed or not gm.active:
                continue
            if not gm.clob_token_ids:
                continue

            # Use condition_id as market_id (unique per binary market)
            market_id = gm.condition_id

            # YES price is the first outcome price
            yes_price = gm.outcome_prices[0] if gm.outcome_prices else 0.0

            # Outcome name from group_item_title or question
            outcome_name = gm.group_item_title or gm.question

            token_id_yes = gm.clob_token_ids[0] if gm.clob_token_ids else ""
            token_id_no = gm.clob_token_ids[1] if len(gm.clob_token_ids) > 1 else ""

            # Cache for later lookups
            self._market_cache[market_id] = gm

            markets.append({
                "id": market_id,
                "event_id": gm.event_id,
                "event_name": gm.event_title,
                "outcome_name": outcome_name,
                "external_price_yes": yes_price,
                "token_id_yes": token_id_yes,
                "token_id_no": token_id_no,
                "end_date": gm.end_date,
                "best_bid": gm.best_bid,
                "best_ask": gm.best_ask,
                "status": "active",
            })

        logger.info("Bridge: fetched %d active weather markets", len(markets))
        return markets

    # ------------------------------------------------------------------
    # Portfolio & positions
    # ------------------------------------------------------------------

    def get_portfolio(self) -> dict:
        """Derive portfolio info from max_exposure minus tracked positions.

        Deducts exposure from open positions tracked via record_exposure().
        """
        return {
            "balance_usdc": max(0.0, self.max_exposure - self._total_exposure),
            "total_exposure": self._total_exposure,
            "positions_count": self._position_count,
        }

    def get_position(self, market_id: str) -> dict | None:
        """Return None — position tracking is done via state."""
        return None

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    def get_market_context(self, market_id: str, **kwargs) -> dict | None:
        """Build market context from CLOB orderbook data.

        Returns a dict compatible with check_context_safeguards():
            {market: {time_to_resolution}, slippage: {estimates: [...]},
             edge: {}, warnings: [], discipline: {}}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return None

        # Time to resolution
        time_str = ""
        if gm.end_date:
            try:
                end_dt = datetime.fromisoformat(gm.end_date.replace("Z", "+00:00"))
                delta = end_dt - datetime.now(timezone.utc)
                hours = max(0, delta.total_seconds() / 3600)
                days = int(hours // 24)
                remaining_hours = int(hours % 24)
                if days > 0:
                    time_str = f"{days}d {remaining_hours}h"
                else:
                    time_str = f"{remaining_hours}h"
            except (ValueError, TypeError):
                pass

        # Slippage estimate from spread
        spread = gm.spread
        bid_ask_sum = gm.best_bid + gm.best_ask
        mid = bid_ask_sum / 2 if bid_ask_sum > 0 else 0.0
        slippage_pct = spread / mid if mid > 0 else 0.0

        return {
            "market": {"time_to_resolution": time_str},
            "slippage": {"estimates": [{"slippage_pct": slippage_pct}]},
            "edge": {},
            "warnings": [],
            "discipline": {},
        }

    # ------------------------------------------------------------------
    # Fill verification
    # ------------------------------------------------------------------

    def verify_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll order status until filled, partially filled, or timeout.

        Returns:
            {"filled": bool, "partial": bool, "size_matched": float,
             "original_size": float, "status": str}
        """
        deadline = time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"
        size_matched = 0.0
        original_size = 0.0

        while time.monotonic() < deadline:
            try:
                order = self.clob.get_order(order_id)
                if not order:
                    time.sleep(poll_interval)
                    continue

                last_status = order.get("status", "UNKNOWN")
                size_matched = float(order.get("size_matched", 0))
                original_size = float(order.get("original_size", order.get("size", 0)))

                if last_status == "MATCHED":
                    return {
                        "filled": True,
                        "partial": False,
                        "size_matched": size_matched,
                        "original_size": original_size,
                        "status": last_status,
                    }
                elif last_status == "CANCELLED":
                    return {
                        "filled": size_matched > 0,
                        "partial": 0 < size_matched < original_size,
                        "size_matched": size_matched,
                        "original_size": original_size,
                        "status": last_status,
                    }
            except Exception as exc:
                logger.debug("verify_fill poll error: %s", exc)

            time.sleep(poll_interval)

        # Timeout — check if partially filled
        return {
            "filled": size_matched > 0,
            "partial": 0 < size_matched < original_size,
            "size_matched": size_matched,
            "original_size": original_size,
            "status": f"TIMEOUT ({last_status})",
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""
        try:
            self.clob.cancel_order(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def execute_trade(
        self,
        market_id: str,
        side: str,
        amount: float,
        fill_timeout: float = 0.0,
        fill_poll_interval: float = 2.0,
    ) -> dict:
        """Execute a buy trade via CLOB limit order at fresh best ask.

        Re-fetches the orderbook before trading to avoid stale prices.
        When ``fill_timeout > 0``, verifies fill and cancels unfilled orders.

        Args:
            market_id: condition_id of the market.
            side: "yes" or "no".
            amount: USD amount to spend.
            fill_timeout: Seconds to wait for fill verification (0 = skip).
            fill_poll_interval: Seconds between fill status polls.

        Returns:
            {"success": bool, "shares_bought": float, "trade_id": str}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        # Determine token ID
        if side.lower() == "yes":
            if not gm.clob_token_ids:
                return {"error": "No YES token ID", "success": False}
            token_id = gm.clob_token_ids[0]
        else:
            if len(gm.clob_token_ids) < 2:
                return {"error": "No NO token ID", "success": False}
            token_id = gm.clob_token_ids[1]

        # Re-fetch fresh price from orderbook
        try:
            book = self.clob.get_orderbook(token_id)
            asks = book.get("asks") or []
            if asks:
                price = float(asks[0]["price"])
            else:
                # Fallback to cached price
                price = gm.best_ask if gm.best_ask > 0 else (
                    gm.outcome_prices[0] if gm.outcome_prices else 0.5
                )
        except Exception:
            price = gm.best_ask if gm.best_ask > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            logger.warning("Could not refresh price for %s, using cached %.4f", token_id[:16], price)

        if price <= 0:
            return {"error": "Invalid price", "success": False}

        shares = amount / price

        try:
            result = self.clob.post_order(
                token_id=token_id,
                side="BUY",
                price=price,
                size=shares,
                neg_risk=True,  # Weather markets are always neg-risk
            )
            order_id = result.get("orderID", "")
            logger.info(
                "Bridge trade: BUY %.1f shares of %s @ %.4f (order %s)",
                shares, token_id[:16], price, order_id,
            )

            # Fill verification
            if fill_timeout > 0 and order_id:
                fill = self.verify_fill(order_id, fill_timeout, fill_poll_interval)
                if fill["filled"] and not fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.info("Order %s fully filled: %.1f shares", order_id, actual_shares)
                    shares = actual_shares
                elif fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.warning("Order %s partially filled: %.1f/%.1f shares — cancelling remainder",
                                   order_id, actual_shares, fill["original_size"])
                    self.cancel_order(order_id)
                    shares = actual_shares
                else:
                    logger.warning("Order %s not filled within %.1fs — cancelling", order_id, fill_timeout)
                    self.cancel_order(order_id)
                    return {
                        "success": False,
                        "error": f"Order not filled within {fill_timeout}s",
                        "trade_id": order_id,
                    }
            elif fill_timeout == 0:
                logger.debug("No fill verification — exposure is estimated, not confirmed")

            self._total_exposure += price * shares
            if market_id not in self._known_positions:
                self._position_count += 1
                self._known_positions.add(market_id)
            return {
                "success": True,
                "shares_bought": shares,
                "trade_id": order_id,
            }
        except Exception as exc:
            logger.error("Bridge trade failed: %s", exc)
            return {"error": str(exc), "success": False}

    def execute_sell(
        self,
        market_id: str,
        shares: float,
        fill_timeout: float = 0.0,
        fill_poll_interval: float = 2.0,
    ) -> dict:
        """Execute a sell trade via CLOB limit order at fresh best bid.

        Re-fetches the orderbook before selling to avoid stale prices.
        When ``fill_timeout > 0``, verifies fill and cancels unfilled orders.

        Args:
            market_id: condition_id of the market.
            shares: Number of shares to sell.
            fill_timeout: Seconds to wait for fill verification (0 = skip).
            fill_poll_interval: Seconds between fill status polls.

        Returns:
            {"success": bool, "trade_id": str}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        if not gm.clob_token_ids:
            return {"error": "No token ID", "success": False}

        token_id = gm.clob_token_ids[0]  # YES token

        # Re-fetch fresh price from orderbook
        try:
            book = self.clob.get_orderbook(token_id)
            bids = book.get("bids") or []
            if bids:
                price = float(bids[0]["price"])
            else:
                price = gm.best_bid if gm.best_bid > 0 else (
                    gm.outcome_prices[0] if gm.outcome_prices else 0.5
                )
        except Exception:
            price = gm.best_bid if gm.best_bid > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            logger.warning("Could not refresh bid for %s, using cached %.4f", token_id[:16], price)

        if price <= 0:
            return {"error": "Invalid bid price", "success": False}

        try:
            result = self.clob.post_order(
                token_id=token_id,
                side="SELL",
                price=price,
                size=shares,
                neg_risk=True,
            )
            order_id = result.get("orderID", "")
            logger.info(
                "Bridge sell: SELL %.1f shares of %s @ %.4f (order %s)",
                shares, token_id[:16], price, order_id,
            )

            # Fill verification
            actual_shares = shares
            if fill_timeout > 0 and order_id:
                fill = self.verify_fill(order_id, fill_timeout, fill_poll_interval)
                if fill["filled"] and not fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.info("Sell order %s fully filled: %.1f shares", order_id, actual_shares)
                elif fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.warning("Sell order %s partially filled: %.1f/%.1f — cancelling remainder",
                                   order_id, actual_shares, fill["original_size"])
                    self.cancel_order(order_id)
                else:
                    logger.warning("Sell order %s not filled within %.1fs — cancelling", order_id, fill_timeout)
                    self.cancel_order(order_id)
                    return {
                        "success": False,
                        "error": f"Sell order not filled within {fill_timeout}s",
                        "trade_id": order_id,
                    }
            elif fill_timeout == 0:
                logger.debug("No fill verification — exposure is estimated, not confirmed")

            self._total_exposure = max(0.0, self._total_exposure - price * actual_shares)
            self._known_positions.discard(market_id)
            self._position_count = max(0, self._position_count - 1)
            return {
                "success": True,
                "trade_id": order_id,
            }
        except Exception as exc:
            logger.error("Bridge sell failed: %s", exc)
            return {"error": str(exc), "success": False}
