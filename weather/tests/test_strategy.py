"""Integration tests for the strategy module — mocked bridge (CLOB) calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from weather.config import Config
from weather.bridge import CLOBWeatherBridge
from weather.state import TradingState
from weather.strategy import (
    check_context_safeguards,
    check_exit_opportunities,
    detect_price_trend,
    run_weather_strategy,
    score_buckets,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestCheckContextSafeguards(unittest.TestCase):

    def test_none_context_passes(self):
        ok, reasons = check_context_safeguards(None, Config())
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_resolved_market_blocked(self):
        ctx = {"warnings": ["MARKET RESOLVED"], "market": {}, "discipline": {}, "slippage": {}, "edge": {}}
        ok, reasons = check_context_safeguards(ctx, Config())
        self.assertFalse(ok)
        self.assertIn("Market already resolved", reasons)

    def test_severe_flipflop_blocked(self):
        ctx = {
            "warnings": [],
            "market": {},
            "discipline": {"warning_level": "severe", "flip_flop_warning": "3 trades in 24h"},
            "slippage": {},
            "edge": {},
        }
        ok, _ = check_context_safeguards(ctx, Config())
        self.assertFalse(ok)

    def test_too_soon_resolution_blocked(self):
        ctx = {
            "warnings": [],
            "market": {"time_to_resolution": "1h"},
            "discipline": {},
            "slippage": {},
            "edge": {},
        }
        ok, reasons = check_context_safeguards(ctx, Config())
        self.assertFalse(ok)

    def test_high_slippage_blocked(self):
        ctx = {
            "warnings": [],
            "market": {},
            "discipline": {},
            "slippage": {"estimates": [{"slippage_pct": 0.20}]},
            "edge": {},
        }
        ok, _ = check_context_safeguards(ctx, Config())
        self.assertFalse(ok)


class TestDetectPriceTrend(unittest.TestCase):

    def test_empty_history(self):
        result = detect_price_trend([])
        self.assertEqual(result["direction"], "unknown")

    def test_flat_market(self):
        history = [{"price_yes": 0.50}] * 100
        result = detect_price_trend(history)
        self.assertEqual(result["direction"], "flat")

    def test_dropping_market(self):
        history = [{"price_yes": 0.50}] * 50 + [{"price_yes": 0.35}] * 50
        result = detect_price_trend(history)
        self.assertEqual(result["direction"], "down")
        self.assertTrue(result["is_opportunity"])


class TestScoreBuckets(unittest.TestCase):

    def test_scores_sorted_by_ev(self):
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config = Config(adjacent_buckets=True, seasonal_adjustments=False)

        scored = score_buckets(event_markets, 52, "2025-03-15", config)
        self.assertGreater(len(scored), 0)

        # Should be sorted by EV descending
        evs = [s["ev"] for s in scored]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_center_bucket_highest_probability(self):
        """Bucket containing the forecast should have highest probability."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config = Config(adjacent_buckets=True, seasonal_adjustments=False)

        scored = score_buckets(event_markets, 52, "2025-03-15", config)
        # Find the bucket that contains 52°F (50-54)
        center = [s for s in scored if s["bucket"] == (50, 54)]
        others = [s for s in scored if s["bucket"] != (50, 54)]

        self.assertEqual(len(center), 1)
        if others:
            self.assertGreater(center[0]["prob"], max(o["prob"] for o in others))


class TestStrategyIntegration(unittest.TestCase):
    """Full dry-run with mocked bridge."""

    def _make_mock_bridge(self) -> MagicMock:
        bridge = MagicMock(spec=CLOBWeatherBridge)
        bridge.get_portfolio.return_value = {
            "balance_usdc": 50.0,
            "total_exposure": 10.0,
            "positions_count": 2,
        }
        bridge.fetch_weather_markets.return_value = _load_fixture("weather_markets.json")["markets"]
        bridge.get_positions.return_value = _load_fixture("weather_positions.json")["positions"]
        bridge.get_market_context.return_value = None  # No safeguards blocking
        bridge.get_price_history.return_value = []
        bridge.get_position.return_value = None
        return bridge

    @patch("weather.strategy.get_noaa_forecast")
    def test_dry_run_no_errors(self, mock_noaa):
        """Full strategy in dry-run should complete without errors."""
        mock_noaa.return_value = {
            "2025-03-15": {"high": 52, "low": 38},
            "2025-03-16": {"high": 58, "low": 42},
        }

        bridge = self._make_mock_bridge()
        config = Config(
            locations="NYC",
            adjacent_buckets=True,
            dynamic_exits=False,
            seasonal_adjustments=False,
            max_days_ahead=365,  # Don't filter by date in tests
        )
        state = TradingState()

        # Should not raise
        run_weather_strategy(
            client=bridge,
            config=config,
            state=state,
            dry_run=True,
            use_safeguards=False,
            use_trends=False,
        )

        # Markets should have been fetched
        bridge.fetch_weather_markets.assert_called_once()

    @patch("weather.strategy.get_noaa_forecast")
    def test_no_trade_when_no_edge(self, mock_noaa):
        """When all prices are above fair value, no trades."""
        mock_noaa.return_value = {"2025-03-15": {"high": 52, "low": 38}}

        # Set all prices very high (above probability)
        markets = _load_fixture("weather_markets.json")["markets"]
        for m in markets:
            m["external_price_yes"] = 0.95

        bridge = self._make_mock_bridge()
        bridge.fetch_weather_markets.return_value = markets

        config = Config(
            locations="NYC", adjacent_buckets=True,
            seasonal_adjustments=False, max_days_ahead=365,
        )
        state = TradingState()

        run_weather_strategy(
            client=bridge, config=config, state=state,
            dry_run=True, use_safeguards=False, use_trends=False,
        )

        # No trades should have been executed
        bridge.execute_trade.assert_not_called()


class TestActiveLocations(unittest.TestCase):
    """Regression: active_locations must map to canonical LOCATIONS keys."""

    def test_nyc_uppercase_input(self):
        cfg = Config(locations="NYC")
        self.assertEqual(cfg.active_locations, ["NYC"])

    def test_chicago_lowercase_input(self):
        cfg = Config(locations="chicago")
        self.assertEqual(cfg.active_locations, ["Chicago"])

    def test_multiple_locations_mixed_case(self):
        cfg = Config(locations="nyc,Chicago,seattle")
        self.assertEqual(cfg.active_locations, ["NYC", "Chicago", "Seattle"])

    def test_all_locations(self):
        cfg = Config(locations="NYC,Chicago,Seattle,Atlanta,Dallas,Miami")
        self.assertEqual(
            cfg.active_locations,
            ["NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami"],
        )


class TestStateSaveLoad(unittest.TestCase):
    """State roundtrip: save then load should preserve data."""

    def test_roundtrip(self):
        state = TradingState()
        state.record_trade(
            market_id="m-1", outcome_name="50-54", side="yes",
            cost_basis=0.10, shares=20.0, location="NYC",
        )
        state.mark_analyzed("m-2")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = TradingState.load(path)

        self.assertIn("m-1", loaded.trades)
        self.assertAlmostEqual(loaded.trades["m-1"].cost_basis, 0.10)
        self.assertAlmostEqual(loaded.trades["m-1"].shares, 20.0)
        self.assertIn("m-2", loaded.analyzed_markets)

        Path(path).unlink()

    def test_load_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{corrupted json!!")
            path = f.name

        state = TradingState.load(path)
        # Should return fresh state, not crash
        self.assertEqual(len(state.trades), 0)

        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
