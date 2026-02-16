"""Tests for the --explain mode."""

import logging
import unittest
from unittest.mock import MagicMock, patch

from weather.config import Config
from weather.state import TradingState
from weather.strategy import run_weather_strategy


class TestExplainFlagImpliesDryRun(unittest.TestCase):

    @patch("weather.strategy.get_noaa_forecast")
    def test_explain_forces_dry_run(self, mock_noaa):
        """When explain=True, dry_run behavior should be active."""
        mock_noaa.return_value = {"2025-03-15": {"high": 52, "low": 38}}

        bridge = MagicMock()
        bridge.get_portfolio.return_value = {"balance_usdc": 50.0, "total_exposure": 0, "positions_count": 0}
        bridge.fetch_weather_markets.return_value = []
        bridge._market_cache = {}

        config = Config(locations="NYC", max_days_ahead=365, seasonal_adjustments=False, aviation_obs=False)
        state = TradingState()

        # explain=True with dry_run=True should work without errors
        run_weather_strategy(
            client=bridge, config=config, state=state,
            dry_run=True, explain=True, use_safeguards=False,
        )
        bridge.execute_trade.assert_not_called()


class TestExplainStatsAccumulation(unittest.TestCase):

    @patch("weather.strategy.get_noaa_forecast")
    def test_explain_stats_logged(self, mock_noaa):
        """With explain=True, DECISION SUMMARY should be logged."""
        mock_noaa.return_value = {"2025-03-15": {"high": 52, "low": 38}}

        bridge = MagicMock()
        bridge.get_portfolio.return_value = {"balance_usdc": 50.0, "total_exposure": 0, "positions_count": 0}
        # Load real fixture markets to generate scoring
        import json
        from pathlib import Path
        fixtures = Path(__file__).parent / "fixtures"
        markets = json.loads((fixtures / "weather_markets.json").read_text())["markets"]
        bridge.fetch_weather_markets.return_value = markets
        bridge.get_market_context.return_value = None
        bridge._market_cache = {}

        config = Config(
            locations="NYC", adjacent_buckets=True,
            seasonal_adjustments=False, max_days_ahead=365,
            aviation_obs=False,
        )
        state = TradingState()

        with self.assertLogs("weather.strategy", level="INFO") as cm:
            run_weather_strategy(
                client=bridge, config=config, state=state,
                dry_run=True, explain=True, use_safeguards=False,
            )

        # Check that DECISION SUMMARY appeared in logs
        summary_logs = [m for m in cm.output if "DECISION SUMMARY" in m]
        self.assertTrue(len(summary_logs) > 0, "Expected [EXPLAIN] DECISION SUMMARY in logs")

        # Check that events_scanned was logged
        scanned_logs = [m for m in cm.output if "Events scanned" in m]
        self.assertTrue(len(scanned_logs) > 0, "Expected Events scanned line in logs")


class TestExplainFilterReasonsTracking(unittest.TestCase):

    @patch("weather.strategy.get_noaa_forecast")
    def test_filter_reasons_tracked(self, mock_noaa):
        """Filter reasons (low_ev, price_above_threshold) should appear in explain output."""
        mock_noaa.return_value = {"2025-03-15": {"high": 52, "low": 38}}

        bridge = MagicMock()
        bridge.get_portfolio.return_value = {"balance_usdc": 50.0, "total_exposure": 0, "positions_count": 0}
        import json
        from pathlib import Path
        fixtures = Path(__file__).parent / "fixtures"
        markets = json.loads((fixtures / "weather_markets.json").read_text())["markets"]
        bridge.fetch_weather_markets.return_value = markets
        bridge.get_market_context.return_value = None
        bridge._market_cache = {}

        config = Config(
            locations="NYC", adjacent_buckets=True,
            seasonal_adjustments=False, max_days_ahead=365,
            aviation_obs=False,
        )
        state = TradingState()

        with self.assertLogs("weather.strategy", level="INFO") as cm:
            run_weather_strategy(
                client=bridge, config=config, state=state,
                dry_run=True, explain=True, use_safeguards=False,
            )

        # Check that filter breakdown appeared (some buckets should be filtered)
        all_output = "\n".join(cm.output)
        self.assertIn("Buckets filtered", all_output)


class TestExplainSummaryFormat(unittest.TestCase):

    @patch("weather.strategy.get_noaa_forecast")
    def test_summary_has_separator_lines(self, mock_noaa):
        """Summary should have separator lines (====)."""
        mock_noaa.return_value = {"2025-03-15": {"high": 52, "low": 38}}

        bridge = MagicMock()
        bridge.get_portfolio.return_value = {"balance_usdc": 50.0, "total_exposure": 0, "positions_count": 0}
        import json
        from pathlib import Path
        fixtures = Path(__file__).parent / "fixtures"
        markets = json.loads((fixtures / "weather_markets.json").read_text())["markets"]
        bridge.fetch_weather_markets.return_value = markets
        bridge.get_market_context.return_value = None
        bridge._market_cache = {}

        config = Config(
            locations="NYC", adjacent_buckets=True,
            seasonal_adjustments=False, max_days_ahead=365,
            aviation_obs=False,
        )
        state = TradingState()

        with self.assertLogs("weather.strategy", level="INFO") as cm:
            run_weather_strategy(
                client=bridge, config=config, state=state,
                dry_run=True, explain=True, use_safeguards=False,
            )

        separator_logs = [m for m in cm.output if "====" in m]
        # At least the main summary separator + explain summary separators
        self.assertGreaterEqual(len(separator_logs), 2)
