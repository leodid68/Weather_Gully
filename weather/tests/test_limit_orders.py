"""Tests for fair-price limit orders and adaptive slippage wiring."""
import unittest
from unittest.mock import MagicMock

from weather.bridge import CLOBWeatherBridge


class TestMarketContextEdge(unittest.TestCase):
    """Test that get_market_context populates edge dict."""

    def _make_bridge_with_market(self, best_ask=0.30, best_bid=0.25):
        bridge = CLOBWeatherBridge.__new__(CLOBWeatherBridge)
        bridge._market_cache = {}
        gm = MagicMock()
        gm.best_ask = best_ask
        gm.best_bid = best_bid
        gm.spread = best_ask - best_bid
        gm.outcome_prices = [best_ask]
        gm.end_date = "2026-02-20T00:00:00Z"
        bridge._market_cache["m1"] = gm
        return bridge

    def test_edge_populated_with_probability(self):
        bridge = self._make_bridge_with_market(best_ask=0.30)
        ctx = bridge.get_market_context("m1", my_probability=0.45)
        self.assertIn("user_edge", ctx["edge"])
        self.assertAlmostEqual(ctx["edge"]["user_edge"], 0.15, places=2)
        self.assertEqual(ctx["edge"]["recommendation"], "TRADE")

    def test_edge_empty_without_probability(self):
        bridge = self._make_bridge_with_market()
        ctx = bridge.get_market_context("m1")
        self.assertEqual(ctx["edge"], {})

    def test_edge_skip_when_negative(self):
        bridge = self._make_bridge_with_market(best_ask=0.60)
        ctx = bridge.get_market_context("m1", my_probability=0.50)
        self.assertEqual(ctx["edge"]["recommendation"], "SKIP")

    def test_edge_hold_when_small(self):
        bridge = self._make_bridge_with_market(best_ask=0.30)
        ctx = bridge.get_market_context("m1", my_probability=0.31)
        self.assertEqual(ctx["edge"]["recommendation"], "HOLD")


class TestLimitPriceCap(unittest.TestCase):
    """Test that execute_trade respects limit_price."""

    def _make_bridge(self, ask_price="0.50"):
        bridge = CLOBWeatherBridge.__new__(CLOBWeatherBridge)
        bridge.clob = MagicMock()
        bridge._market_cache = {}
        bridge._total_exposure = 0.0
        bridge._position_count = 0
        bridge._known_positions = set()

        gm = MagicMock()
        gm.best_ask = float(ask_price)
        gm.best_bid = 0.25
        gm.outcome_prices = [float(ask_price)]
        gm.clob_token_ids = ["token-yes", "token-no"]
        bridge._market_cache["m1"] = gm

        bridge.clob.get_orderbook.return_value = {
            "asks": [{"price": ask_price, "size": "100"}],
        }
        bridge.clob.post_order.return_value = {"orderID": "order-1"}
        return bridge

    def test_limit_price_caps_order(self):
        bridge = self._make_bridge(ask_price="0.50")
        result = bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0, limit_price=0.40)
        self.assertTrue(result.get("success"))
        # Verify post_order was called with capped price
        call_args = bridge.clob.post_order.call_args
        actual_price = call_args[1]["price"] if "price" in call_args[1] else call_args.kwargs["price"]
        self.assertLessEqual(actual_price, 0.40)

    def test_no_limit_uses_ask(self):
        bridge = self._make_bridge(ask_price="0.30")
        result = bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0)
        self.assertTrue(result.get("success"))

    def test_limit_below_ask_uses_limit(self):
        bridge = self._make_bridge(ask_price="0.50")
        result = bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0, limit_price=0.35)
        call_args = bridge.clob.post_order.call_args
        actual_price = call_args[1]["price"] if "price" in call_args[1] else call_args.kwargs["price"]
        self.assertLessEqual(actual_price, 0.35)

    def test_limit_above_ask_uses_ask(self):
        """When limit > ask, we should use ask (better price)."""
        bridge = self._make_bridge(ask_price="0.30")
        result = bridge.execute_trade("m1", "yes", 1.0, fill_timeout=0, limit_price=0.50)
        call_args = bridge.clob.post_order.call_args
        actual_price = call_args[1]["price"] if "price" in call_args[1] else call_args.kwargs["price"]
        self.assertAlmostEqual(actual_price, 0.30, places=2)
