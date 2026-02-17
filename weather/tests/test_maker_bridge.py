"""Tests for execute_maker_order in CLOBWeatherBridge."""

from unittest.mock import AsyncMock, MagicMock
import pytest


def _make_bridge(market_cache=None):
    """Create a CLOBWeatherBridge with a mocked CLOB client."""
    from weather.bridge import CLOBWeatherBridge

    bridge = CLOBWeatherBridge.__new__(CLOBWeatherBridge)
    bridge.clob = AsyncMock()
    bridge._market_cache = market_cache or {}
    bridge._total_exposure = 0.0
    bridge._position_count = 0
    bridge._known_positions = set()
    return bridge


def _make_gm(token_ids=None, best_bid=0.10, best_ask=0.15):
    """Create a mock GammaMarket."""
    gm = MagicMock()
    gm.clob_token_ids = token_ids or ["token_abc"]
    gm.best_bid = best_bid
    gm.best_ask = best_ask
    gm.outcome_prices = [best_bid]
    return gm


class TestExecuteMakerOrder:
    @pytest.mark.asyncio
    async def test_maker_order_success(self):
        """Valid maker order -> posted=True with order_id."""
        gm = _make_gm()
        bridge = _make_bridge({"m1": gm})
        bridge.clob.post_order.return_value = {"orderID": "ox123", "status": "LIVE"}

        result = await bridge.execute_maker_order("m1", "yes", 2.0, 0.10)
        assert result["success"] is True
        assert result["posted"] is True
        assert result["order_id"] == "ox123"
        assert result["price"] == 0.10
        assert result["size"] == 20.0

    @pytest.mark.asyncio
    async def test_maker_order_rejected_crosses_spread(self):
        """CLOB rejects (no orderID) -> posted=False."""
        gm = _make_gm()
        bridge = _make_bridge({"m1": gm})
        bridge.clob.post_order.return_value = {"errorMsg": "would cross"}

        result = await bridge.execute_maker_order("m1", "yes", 2.0, 0.10)
        assert result["success"] is False
        assert result["posted"] is False

    @pytest.mark.asyncio
    async def test_maker_order_below_min_shares(self):
        """Amount too small for min shares -> posted=False."""
        gm = _make_gm()
        bridge = _make_bridge({"m1": gm})

        # 0.01 / 0.10 = 0.1 shares < 5.0 min
        result = await bridge.execute_maker_order("m1", "yes", 0.01, 0.10)
        assert result["posted"] is False
        assert "minimum" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_maker_order_no_market_data(self):
        """Missing market cache -> posted=False."""
        bridge = _make_bridge({})

        result = await bridge.execute_maker_order("m_unknown", "yes", 2.0, 0.10)
        assert result["posted"] is False
        assert "no market" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_maker_order_post_only_flag(self):
        """Verify post_only=True is passed to clob.post_order."""
        gm = _make_gm()
        bridge = _make_bridge({"m1": gm})
        bridge.clob.post_order.return_value = {"orderID": "ox123", "status": "LIVE"}

        await bridge.execute_maker_order("m1", "yes", 2.0, 0.10)

        call_kwargs = bridge.clob.post_order.call_args
        assert call_kwargs[1]["post_only"] is True
        assert call_kwargs[1]["order_type"] == "GTC"

    @pytest.mark.asyncio
    async def test_maker_order_exception_handling(self):
        """Exception in post_order -> graceful failure."""
        gm = _make_gm()
        bridge = _make_bridge({"m1": gm})
        bridge.clob.post_order.side_effect = Exception("network error")

        result = await bridge.execute_maker_order("m1", "yes", 2.0, 0.10)
        assert result["success"] is False
        assert result["posted"] is False
        assert "network error" in result.get("error", "")
