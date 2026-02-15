"""Tests for PaperBridge — simulated execution wrapper."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from weather.paper_bridge import PaperBridge
from weather.bridge import CLOBWeatherBridge


def _make_gamma_market(**overrides):
    """Create a mock GammaMarket-like object."""
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

    class GM:
        pass

    gm = GM()
    for k, v in defaults.items():
        setattr(gm, k, v)
    return gm


class TestFetchDelegatesToRealBridge(unittest.TestCase):

    def test_fetch_delegates_to_real_bridge(self):
        gamma = MagicMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        markets = paper.fetch_weather_markets()
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0]["id"], "cond-1")
        gamma.fetch_events_with_markets.assert_called_once()


class TestFetchRecordsSnapshots(unittest.TestCase):

    def test_fetch_records_snapshots(self):
        gamma = MagicMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        paper.fetch_weather_markets()
        self.assertEqual(len(paper._snapshots), 1)
        snap = paper._snapshots[0]
        self.assertIn("timestamp", snap)
        self.assertAlmostEqual(snap["best_ask"], 0.37)
        self.assertAlmostEqual(snap["best_bid"], 0.33)
        self.assertEqual(snap["bucket_name"], "50-54°F")


class TestExecuteTradeNoClobCall(unittest.TestCase):

    def test_execute_trade_no_clob_call(self):
        gamma = MagicMock()
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        # Populate cache
        gm = _make_gamma_market(best_ask=0.40)
        real_bridge._market_cache["cond-1"] = gm

        result = paper.execute_trade("cond-1", "yes", 2.00)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["shares_bought"], 2.00 / 0.40)
        self.assertIn("paper-", result["trade_id"])

        # CLOB post_order should NOT have been called
        clob.post_order.assert_not_called()


class TestExecuteTradeUpdatesExposure(unittest.TestCase):

    def test_execute_trade_updates_exposure(self):
        gamma = MagicMock()
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        gm = _make_gamma_market(best_ask=0.50)
        real_bridge._market_cache["cond-1"] = gm

        self.assertAlmostEqual(paper._total_exposure, 0.0)
        paper.execute_trade("cond-1", "yes", 5.00)
        self.assertAlmostEqual(paper._total_exposure, 5.00)
        self.assertEqual(paper._position_count, 1)

        # Paper position should be tracked
        pos = paper.get_position("cond-1")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["shares_yes"], 10.0)

    def test_unknown_market_returns_error(self):
        gamma = MagicMock()
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        result = paper.execute_trade("unknown", "yes", 1.0)
        self.assertFalse(result["success"])


class TestExecuteSellSimulated(unittest.TestCase):

    def test_execute_sell_simulated(self):
        gamma = MagicMock()
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        gm = _make_gamma_market(best_ask=0.50, best_bid=0.48)
        real_bridge._market_cache["cond-1"] = gm

        # Buy first
        paper.execute_trade("cond-1", "yes", 5.00)
        # Sell
        result = paper.execute_sell("cond-1", 5.0)
        self.assertTrue(result["success"])
        self.assertIn("paper-sell-", result["trade_id"])

        # CLOB post_order should NOT have been called
        clob.post_order.assert_not_called()

        # Position should be reduced
        pos = paper.get_position("cond-1")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["shares_yes"], 5.0)  # 10 bought - 5 sold


class TestMarketCacheProxied(unittest.TestCase):

    def test_market_cache_proxied(self):
        gamma = MagicMock()
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        gm = _make_gamma_market()
        real_bridge._market_cache["cond-1"] = gm

        # PaperBridge should expose the same cache
        self.assertIs(paper._market_cache, real_bridge._market_cache)
        self.assertIn("cond-1", paper._market_cache)

    def test_clob_proxied(self):
        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=MagicMock())
        paper = PaperBridge(real_bridge)
        self.assertIs(paper.clob, clob)

    def test_gamma_proxied(self):
        gamma = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=MagicMock(), gamma_client=gamma)
        paper = PaperBridge(real_bridge)
        self.assertIs(paper.gamma, gamma)


class TestSaveSnapshotsFormat(unittest.TestCase):

    def test_save_snapshots_format(self):
        gamma = MagicMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        paper.fetch_weather_markets()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            paper.save_snapshots(tmp_path)

            with open(tmp_path) as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            snap = data[0]
            self.assertIn("timestamp", snap)
            self.assertIn("best_ask", snap)
            self.assertIn("best_bid", snap)
            self.assertIn("market_id", snap)
            self.assertIn("bucket_name", snap)
        finally:
            os.unlink(tmp_path)

    def test_save_appends_to_existing(self):
        gamma = MagicMock()
        gm = _make_gamma_market()
        gamma.fetch_events_with_markets.return_value = ([], [gm])

        clob = MagicMock()
        real_bridge = CLOBWeatherBridge(clob_client=clob, gamma_client=gamma)
        paper = PaperBridge(real_bridge)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"existing": True}], f)
            tmp_path = f.name

        try:
            paper.fetch_weather_markets()
            paper.save_snapshots(tmp_path)

            with open(tmp_path) as f:
                data = json.load(f)

            self.assertEqual(len(data), 2)  # 1 existing + 1 new
            self.assertTrue(data[0]["existing"])
        finally:
            os.unlink(tmp_path)

    def test_no_snapshots_no_write(self):
        real_bridge = CLOBWeatherBridge(clob_client=MagicMock(), gamma_client=MagicMock())
        paper = PaperBridge(real_bridge)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            os.unlink(tmp_path)  # Remove so we can check it's not created
            paper.save_snapshots(tmp_path)
            self.assertFalse(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestVerifyFillAndCancel(unittest.TestCase):

    def test_verify_fill_returns_success(self):
        real_bridge = CLOBWeatherBridge(clob_client=MagicMock(), gamma_client=MagicMock())
        paper = PaperBridge(real_bridge)

        result = paper.verify_fill("any-order-id")
        self.assertTrue(result["filled"])
        self.assertEqual(result["status"], "PAPER_FILLED")

    def test_cancel_returns_true(self):
        real_bridge = CLOBWeatherBridge(clob_client=MagicMock(), gamma_client=MagicMock())
        paper = PaperBridge(real_bridge)
        self.assertTrue(paper.cancel_order("any-order-id"))


class TestGetPortfolio(unittest.TestCase):

    def test_portfolio_reflects_exposure(self):
        real_bridge = CLOBWeatherBridge(
            clob_client=MagicMock(), gamma_client=MagicMock(), max_exposure=100.0,
        )
        paper = PaperBridge(real_bridge)

        gm = _make_gamma_market(best_ask=0.50)
        real_bridge._market_cache["cond-1"] = gm

        portfolio = paper.get_portfolio()
        self.assertAlmostEqual(portfolio["balance_usdc"], 100.0)

        paper.execute_trade("cond-1", "yes", 10.0)
        portfolio = paper.get_portfolio()
        self.assertAlmostEqual(portfolio["balance_usdc"], 90.0)


if __name__ == "__main__":
    unittest.main()
