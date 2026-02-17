"""Tests for CLOBWeatherBridge — CLOB+Gamma adapter for weather strategy."""

import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weather.bridge import CLOBWeatherBridge


def _make_gamma_market(**overrides):
    """Create a mock GammaMarket dataclass-like object."""
    defaults = {
        "id": "gm-1",
        "question": "What will be the highest temperature in NYC on March 15?",
        "condition_id": "cond-1",
        "slug": "nyc-temp-mar15",
        "outcomes": ["Yes", "No"],
        "outcome_prices": [0.35, 0.65],
        "clob_token_ids": ["token-yes-1", "token-no-1"],
        "volume": 5000.0,
        "volume_24hr": 200.0,
        "liquidity": 1000.0,
        "best_bid": 0.33,
        "best_ask": 0.37,
        "spread": 0.04,
        "end_date": "2025-03-15T23:00:00Z",
        "active": True,
        "closed": False,
        "neg_risk": True,
        "group_item_title": "50-54°F",
        "event_id": "evt-1",
        "event_title": "What will be the highest temperature in NYC on March 15?",
    }
    defaults.update(overrides)

    # Create a simple namespace object
    class GM:
        pass

    gm = GM()
    for k, v in defaults.items():
        setattr(gm, k, v)
    return gm


class TestFetchWeatherMarkets:

    @pytest.mark.asyncio
    async def test_returns_formatted_markets(self):
        gamma = AsyncMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = await bridge.fetch_weather_markets()
        assert len(markets) == 1
        m = markets[0]
        assert m["id"] == "cond-1"
        assert m["event_id"] == "evt-1"
        assert m["outcome_name"] == "50-54°F"
        assert abs(m["external_price_yes"] - 0.35) < 1e-6
        assert m["token_id_yes"] == "token-yes-1"
        assert m["token_id_no"] == "token-no-1"
        assert m["status"] == "active"

    @pytest.mark.asyncio
    async def test_skips_closed_markets(self):
        gamma = AsyncMock()
        gm = _make_gamma_market(closed=True)
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = await bridge.fetch_weather_markets()
        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_skips_no_token_ids(self):
        gamma = AsyncMock()
        gm = _make_gamma_market(clob_token_ids=[])
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = await bridge.fetch_weather_markets()
        assert len(markets) == 0


class TestGetPortfolio:

    def test_returns_max_exposure_as_balance(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
            max_exposure=100.0,
        )
        portfolio = bridge.get_portfolio()
        assert portfolio["balance_usdc"] == 100.0


class TestGetPosition:

    def test_get_position_returns_none(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        assert bridge.get_position("any-id") is None


class TestGetMarketContext:

    def test_returns_context_with_time(self):
        gamma = MagicMock()
        clob = MagicMock()
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        # Populate cache
        gm = _make_gamma_market(end_date="2025-03-15T23:00:00Z")
        bridge._market_cache["cond-1"] = gm

        context = bridge.get_market_context("cond-1")
        assert context is not None
        assert "market" in context
        assert "slippage" in context
        assert "warnings" in context

    def test_unknown_market_returns_none(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        assert bridge.get_market_context("unknown") is None


class TestExecuteTrade:

    @pytest.mark.asyncio
    async def test_buy_yes(self):
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "order-123"}
        # Mock orderbook so bridge re-fetches fresh price
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.10", "size": "100"}],
            "bids": [{"price": "0.09", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_ask=0.10)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_trade("cond-1", "yes", 2.00, fill_timeout=0)
        assert result["success"] is True
        assert result["trade_id"] == "order-123"
        assert abs(result["shares_bought"] - 2.00 / 0.10) < 1e-6

        # Verify CLOB was called correctly
        clob.post_order.assert_called_once()
        call_kwargs = clob.post_order.call_args
        assert call_kwargs[1]["token_id"] == "token-yes-1"
        assert call_kwargs[1]["side"] == "BUY"
        assert call_kwargs[1]["neg_risk"] is True

    @pytest.mark.asyncio
    async def test_unknown_market(self):
        bridge = CLOBWeatherBridge(
            clob_client=AsyncMock(),
            gamma_client=MagicMock(),
        )
        result = await bridge.execute_trade("unknown", "yes", 1.0, fill_timeout=0)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_clob_error_handled(self):
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.side_effect = Exception("Network error")
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.50", "size": "100"}],
            "bids": [{"price": "0.49", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market()
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_trade("cond-1", "yes", 2.00, fill_timeout=0)
        assert result["success"] is False
        assert "Network error" in result["error"]


class TestExecuteSell:

    @pytest.mark.asyncio
    async def test_sell(self):
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "sell-456"}
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.42", "size": "100"}],
            "bids": [{"price": "0.40", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_bid=0.40)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_sell("cond-1", 10.0, fill_timeout=0)
        assert result["success"] is True
        assert result["trade_id"] == "sell-456"

        clob.post_order.assert_called_once()
        call_kwargs = clob.post_order.call_args
        assert call_kwargs[1]["side"] == "SELL"
        assert abs(call_kwargs[1]["size"] - 10.0) < 1e-6

    @pytest.mark.asyncio
    async def test_sell_no_side_uses_no_token(self):
        """Selling with side='no' should use clob_token_ids[1]."""
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "sell-no-789"}
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.60", "size": "100"}],
            "bids": [{"price": "0.58", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_bid=0.58)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_sell("cond-1", 10.0, side="no", fill_timeout=0)
        assert result["success"] is True

        call_kwargs = clob.post_order.call_args
        # Should use the NO token (token-no-1)
        assert call_kwargs[1]["token_id"] == "token-no-1"
        assert call_kwargs[1]["side"] == "SELL"

    @pytest.mark.asyncio
    async def test_sell_yes_side_uses_yes_token(self):
        """Selling with side='yes' (default) should use clob_token_ids[0]."""
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "sell-yes-789"}
        clob.get_orderbook.return_value = {
            "bids": [{"price": "0.40", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_bid=0.40)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_sell("cond-1", 10.0, side="yes", fill_timeout=0)
        assert result["success"] is True

        call_kwargs = clob.post_order.call_args
        assert call_kwargs[1]["token_id"] == "token-yes-1"

    @pytest.mark.asyncio
    async def test_unknown_market(self):
        bridge = CLOBWeatherBridge(
            clob_client=AsyncMock(),
            gamma_client=MagicMock(),
        )
        result = await bridge.execute_sell("unknown", 10.0, fill_timeout=0)
        assert result["success"] is False


class TestVerifyFill:

    @pytest.mark.asyncio
    async def test_filled_immediately(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.get_order.return_value = {
            "status": "MATCHED",
            "size_matched": 20.0,
            "original_size": 20.0,
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        result = await bridge.verify_fill("order-1", timeout_seconds=5, poll_interval=0.1)

        assert result["filled"] is True
        assert result["partial"] is False
        assert abs(result["size_matched"] - 20.0) < 1e-6
        assert result["status"] == "MATCHED"

    @pytest.mark.asyncio
    async def test_partial_fill(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.get_order.return_value = {
            "status": "CANCELLED",
            "size_matched": 10.0,
            "original_size": 20.0,
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        result = await bridge.verify_fill("order-1", timeout_seconds=5, poll_interval=0.1)

        assert result["filled"] is True
        assert result["partial"] is True
        assert abs(result["size_matched"] - 10.0) < 1e-6
        assert abs(result["original_size"] - 20.0) < 1e-6

    @pytest.mark.asyncio
    async def test_timeout_unfilled(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.get_order.return_value = {
            "status": "LIVE",
            "size_matched": 0,
            "original_size": 20.0,
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        result = await bridge.verify_fill("order-1", timeout_seconds=0.3, poll_interval=0.1)

        assert result["filled"] is False
        assert result["partial"] is False
        assert "TIMEOUT" in result["status"]

    @pytest.mark.asyncio
    async def test_api_error_during_poll(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.get_order.side_effect = Exception("API error")

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        result = await bridge.verify_fill("order-1", timeout_seconds=0.3, poll_interval=0.1)

        # Should timeout gracefully
        assert result["filled"] is False


class TestCancelOrder:

    @pytest.mark.asyncio
    async def test_cancel_success(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())

        result = await bridge.cancel_order("order-1")
        assert result is True
        clob.cancel_order.assert_called_once_with("order-1")

    @pytest.mark.asyncio
    async def test_cancel_failure(self):
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.cancel_order.side_effect = Exception("Network error")

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        result = await bridge.cancel_order("order-1")
        assert result is False


class TestExecuteTradeWithFillVerification:

    @pytest.mark.asyncio
    async def test_trade_with_fill_timeout_zero_skips_verification(self):
        """fill_timeout=0 should behave like the old code."""
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "order-123"}
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.10", "size": "100"}],
            "bids": [{"price": "0.09", "size": "100"}],
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_ask=0.10)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_trade("cond-1", "yes", 2.00, fill_timeout=0)
        assert result["success"] is True
        # get_order should NOT have been called
        clob.get_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_trade_unfilled_gets_cancelled(self):
        gamma = MagicMock()
        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        clob.post_order.return_value = {"orderID": "order-123"}
        clob.get_orderbook.return_value = {
            "asks": [{"price": "0.10", "size": "100"}],
        }
        clob.get_order.return_value = {
            "status": "LIVE",
            "size_matched": 0,
            "original_size": 20.0,
        }

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_ask=0.10)
        bridge._market_cache["cond-1"] = gm

        result = await bridge.execute_trade("cond-1", "yes", 2.00,
                                       fill_timeout=0.3, fill_poll_interval=0.1)
        assert result["success"] is False
        # Should have tried to cancel
        clob.cancel_order.assert_called_once_with("order-123")


class TestBestAskInStrategy:
    """Verify that best_ask from bridge is used in scoring."""

    @pytest.mark.asyncio
    async def test_best_ask_field_in_market(self):
        """Bridge fetch_weather_markets should include best_ask."""
        gamma = AsyncMock()
        gm = _make_gamma_market(best_ask=0.37)
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = AsyncMock()
        clob.is_neg_risk.return_value = True
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        markets = await bridge.fetch_weather_markets()

        assert len(markets) == 1
        assert abs(markets[0]["best_ask"] - 0.37) < 1e-6


if __name__ == "__main__":
    unittest.main()
