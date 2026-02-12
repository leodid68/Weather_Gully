"""CLOBWeatherBridge — CLOB + Gamma adapter for weather/strategy.py.

Provides the semantic interface that weather/strategy.py expects,
routing all operations through the Polymarket CLOB and Gamma APIs.
"""

from __future__ import annotations

import logging
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
        """Derive portfolio info from max_exposure and state.

        Returns a synthetic portfolio based on max_exposure
        (the CLOB has no portfolio endpoint).
        """
        return {
            "balance_usdc": self.max_exposure,
            "total_exposure": 0.0,
            "positions_count": 0,
        }

    def get_positions(self) -> list:
        """Return empty list — positions are tracked in weather.state.TradingState.

        The weather strategy reads positions from state, not from the API.
        """
        return []

    def get_position(self, market_id: str) -> dict | None:
        """Return None — position tracking is done via state."""
        return None

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    def get_market_context(
        self, market_id: str, my_probability: float | None = None,
    ) -> dict | None:
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
        mid = (gm.best_bid + gm.best_ask) / 2 if (gm.best_bid + gm.best_ask) > 0 else 0.5
        slippage_pct = spread / mid if mid > 0 else 0.0

        return {
            "market": {"time_to_resolution": time_str},
            "slippage": {"estimates": [{"slippage_pct": slippage_pct}]},
            "edge": {},
            "warnings": [],
            "discipline": {},
        }

    def get_price_history(self, market_id: str) -> list:
        """Price history is not available on CLOB — return empty list.

        The existing strategy code handles empty history gracefully.
        """
        return []

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def execute_trade(self, market_id: str, side: str, amount: float) -> dict:
        """Execute a buy trade via CLOB limit order at best ask.

        Args:
            market_id: condition_id of the market.
            side: "yes" or "no".
            amount: USD amount to spend.

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
            price = gm.best_ask if gm.best_ask > 0 else gm.outcome_prices[0] if gm.outcome_prices else 0.5
        else:
            if len(gm.clob_token_ids) < 2:
                return {"error": "No NO token ID", "success": False}
            token_id = gm.clob_token_ids[1]
            no_price = gm.outcome_prices[1] if len(gm.outcome_prices) > 1 else 1.0 - (gm.outcome_prices[0] if gm.outcome_prices else 0.5)
            price = no_price

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
            return {
                "success": True,
                "shares_bought": shares,
                "trade_id": order_id,
            }
        except Exception as exc:
            logger.error("Bridge trade failed: %s", exc)
            return {"error": str(exc), "success": False}

    def execute_sell(self, market_id: str, shares: float) -> dict:
        """Execute a sell trade via CLOB limit order at best bid.

        Args:
            market_id: condition_id of the market.
            shares: Number of shares to sell.

        Returns:
            {"success": bool, "trade_id": str}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        if not gm.clob_token_ids:
            return {"error": "No token ID", "success": False}

        token_id = gm.clob_token_ids[0]  # YES token
        price = gm.best_bid if gm.best_bid > 0 else gm.outcome_prices[0] if gm.outcome_prices else 0.5

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
            return {
                "success": True,
                "trade_id": order_id,
            }
        except Exception as exc:
            logger.error("Bridge sell failed: %s", exc)
            return {"error": str(exc), "success": False}

    # ------------------------------------------------------------------
    # Risk monitoring (no-op — managed by bot/strategy instead)
    # ------------------------------------------------------------------

    def set_risk_monitor(self, market_id: str, side: str,
                         stop_loss_pct: float = 0.20,
                         take_profit_pct: float = 0.50) -> dict | None:
        """No-op — risk monitoring is done by the strategy, not the API."""
        return None

    def get_risk_monitors(self) -> dict | None:
        return None

    def remove_risk_monitor(self, market_id: str, side: str) -> dict:
        return {}
