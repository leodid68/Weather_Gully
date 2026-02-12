"""Tests for CLOBWeatherBridge — CLOB+Gamma adapter for weather strategy."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

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


class TestFetchWeatherMarkets(unittest.TestCase):

    def test_returns_formatted_markets(self):
        gamma = MagicMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = bridge.fetch_weather_markets()
        self.assertEqual(len(markets), 1)
        m = markets[0]
        self.assertEqual(m["id"], "cond-1")
        self.assertEqual(m["event_id"], "evt-1")
        self.assertEqual(m["outcome_name"], "50-54°F")
        self.assertAlmostEqual(m["external_price_yes"], 0.35)
        self.assertEqual(m["token_id_yes"], "token-yes-1")
        self.assertEqual(m["token_id_no"], "token-no-1")
        self.assertEqual(m["status"], "active")

    def test_skips_closed_markets(self):
        gamma = MagicMock()
        gm = _make_gamma_market(closed=True)
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = bridge.fetch_weather_markets()
        self.assertEqual(len(markets), 0)

    def test_skips_no_token_ids(self):
        gamma = MagicMock()
        gm = _make_gamma_market(clob_token_ids=[])
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        markets = bridge.fetch_weather_markets()
        self.assertEqual(len(markets), 0)


class TestGetPortfolio(unittest.TestCase):

    def test_returns_max_exposure_as_balance(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
            max_exposure=100.0,
        )
        portfolio = bridge.get_portfolio()
        self.assertEqual(portfolio["balance_usdc"], 100.0)


class TestGetPositions(unittest.TestCase):

    def test_returns_empty(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        self.assertEqual(bridge.get_positions(), [])

    def test_get_position_returns_none(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        self.assertIsNone(bridge.get_position("any-id"))


class TestGetMarketContext(unittest.TestCase):

    def test_returns_context_with_time(self):
        gamma = MagicMock()
        clob = MagicMock()
        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)

        # Populate cache
        gm = _make_gamma_market(end_date="2025-03-15T23:00:00Z")
        bridge._market_cache["cond-1"] = gm

        context = bridge.get_market_context("cond-1")
        self.assertIsNotNone(context)
        self.assertIn("market", context)
        self.assertIn("slippage", context)
        self.assertIn("warnings", context)

    def test_unknown_market_returns_none(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        self.assertIsNone(bridge.get_market_context("unknown"))


class TestGetPriceHistory(unittest.TestCase):

    def test_returns_empty_list(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        self.assertEqual(bridge.get_price_history("any-id"), [])


class TestExecuteTrade(unittest.TestCase):

    def test_buy_yes(self):
        gamma = MagicMock()
        clob = MagicMock()
        clob.post_order.return_value = {"orderID": "order-123"}

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_ask=0.10)
        bridge._market_cache["cond-1"] = gm

        result = bridge.execute_trade("cond-1", "yes", 2.00)
        self.assertTrue(result["success"])
        self.assertEqual(result["trade_id"], "order-123")
        self.assertAlmostEqual(result["shares_bought"], 2.00 / 0.10)

        # Verify CLOB was called correctly
        clob.post_order.assert_called_once()
        call_kwargs = clob.post_order.call_args
        self.assertEqual(call_kwargs[1]["token_id"], "token-yes-1")
        self.assertEqual(call_kwargs[1]["side"], "BUY")
        self.assertTrue(call_kwargs[1]["neg_risk"])

    def test_unknown_market(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        result = bridge.execute_trade("unknown", "yes", 1.0)
        self.assertFalse(result["success"])

    def test_clob_error_handled(self):
        gamma = MagicMock()
        clob = MagicMock()
        clob.post_order.side_effect = Exception("Network error")

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market()
        bridge._market_cache["cond-1"] = gm

        result = bridge.execute_trade("cond-1", "yes", 2.00)
        self.assertFalse(result["success"])
        self.assertIn("Network error", result["error"])


class TestExecuteSell(unittest.TestCase):

    def test_sell(self):
        gamma = MagicMock()
        clob = MagicMock()
        clob.post_order.return_value = {"orderID": "sell-456"}

        bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        gm = _make_gamma_market(best_bid=0.40)
        bridge._market_cache["cond-1"] = gm

        result = bridge.execute_sell("cond-1", 10.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["trade_id"], "sell-456")

        clob.post_order.assert_called_once()
        call_kwargs = clob.post_order.call_args
        self.assertEqual(call_kwargs[1]["side"], "SELL")
        self.assertAlmostEqual(call_kwargs[1]["size"], 10.0)

    def test_unknown_market(self):
        bridge = CLOBWeatherBridge(
            clob_client=MagicMock(),
            gamma_client=MagicMock(),
        )
        result = bridge.execute_sell("unknown", 10.0)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
