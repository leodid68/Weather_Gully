"""Integration tests for the strategy module — mocked bridge (CLOB) calls."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from weather.config import Config
from weather.bridge import CLOBWeatherBridge
from weather.state import TradingState
from weather.strategy import (
    check_context_safeguards,
    check_exit_opportunities,
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


class TestMinProbabilityFilter(unittest.TestCase):
    """Test that min_probability filters low-probability buckets."""

    def test_min_probability_filters_low_prob(self):
        """Buckets with prob < min_probability should be excluded."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        # Set high min_probability to filter most buckets
        config = Config(adjacent_buckets=True, seasonal_adjustments=False, min_probability=0.90)

        scored = score_buckets(event_markets, 52, "2025-03-15", config)
        # With 0.90 threshold, few or no buckets should pass
        for s in scored:
            self.assertGreaterEqual(s["prob"], 0.90)

    def test_min_probability_zero_passes_all(self):
        """With min_probability=0, all valid buckets should pass."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config_zero = Config(adjacent_buckets=True, seasonal_adjustments=False, min_probability=0.0)
        config_default = Config(adjacent_buckets=True, seasonal_adjustments=False, min_probability=0.15)

        scored_zero = score_buckets(event_markets, 52, "2025-03-15", config_zero)
        scored_default = score_buckets(event_markets, 52, "2025-03-15", config_default)
        # Zero threshold should return at least as many as default
        self.assertGreaterEqual(len(scored_zero), len(scored_default))


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
        bridge.get_market_context.return_value = None  # No safeguards blocking
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
            dry_run=True, use_safeguards=False,
        )

        # No trades should have been executed
        bridge.execute_trade.assert_not_called()


class TestHealthCheck(unittest.TestCase):
    """Test that strategy aborts when no weather sources return data."""

    @patch("weather.strategy.get_open_meteo_forecast")
    @patch("weather.strategy.get_noaa_forecast")
    def test_no_sources_returns_early(self, mock_noaa, mock_om):
        """If both NOAA and Open-Meteo return empty, strategy should abort."""
        mock_noaa.return_value = {}
        mock_om.return_value = {}

        bridge = MagicMock(spec=CLOBWeatherBridge)
        bridge.get_portfolio.return_value = {"balance_usdc": 50.0, "total_exposure": 0, "positions_count": 0}
        bridge.fetch_weather_markets.return_value = _load_fixture("weather_markets.json")["markets"]
        bridge.get_position.return_value = None

        config = Config(locations="NYC", multi_source=True, max_days_ahead=365,
                        seasonal_adjustments=False, aviation_obs=False)
        state = TradingState()

        run_weather_strategy(
            client=bridge, config=config, state=state,
            dry_run=True, use_safeguards=False,
        )

        # No trades should have been attempted (early return)
        bridge.execute_trade.assert_not_called()


class TestScoreBucketsWithNewParams(unittest.TestCase):
    """Test that score_buckets passes location and weather_data correctly."""

    def test_location_param_accepted(self):
        """score_buckets should accept location without error."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config = Config(adjacent_buckets=True, seasonal_adjustments=False)

        scored = score_buckets(event_markets, 52, "2025-03-15", config,
                               location="NYC")
        self.assertGreater(len(scored), 0)

    def test_weather_data_param_accepted(self):
        """score_buckets should accept weather_data without error."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config = Config(adjacent_buckets=True, seasonal_adjustments=False)

        weather_data = {
            "cloud_cover_max": 90.0,
            "wind_speed_max": 50.0,
            "precip_sum": 15.0,
        }
        scored = score_buckets(event_markets, 52, "2025-03-15", config,
                               weather_data=weather_data)
        self.assertGreater(len(scored), 0)

    def test_best_ask_used_when_available(self):
        """When best_ask is available, it should be used for pricing."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]

        # Add best_ask to one market
        for m in event_markets:
            if m["outcome_name"] == "50-54°F":
                m["best_ask"] = 0.40  # Different from external_price_yes (0.35)
                break

        config = Config(adjacent_buckets=True, seasonal_adjustments=False)
        scored = score_buckets(event_markets, 52, "2025-03-15", config)

        # Find the 50-54°F bucket
        center = [s for s in scored if s["bucket"] == (50, 54)]
        self.assertEqual(len(center), 1)
        # Price should be the best_ask (0.40), not external_price_yes (0.35)
        self.assertAlmostEqual(center[0]["price"], 0.40, places=2)

    def test_best_ask_zero_falls_back_to_external(self):
        """When best_ask is 0, should fall back to external_price_yes."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]

        for m in event_markets:
            if m["outcome_name"] == "50-54°F":
                m["best_ask"] = 0  # Zero = not available
                break

        config = Config(adjacent_buckets=True, seasonal_adjustments=False)
        scored = score_buckets(event_markets, 52, "2025-03-15", config)

        center = [s for s in scored if s["bucket"] == (50, 54)]
        self.assertEqual(len(center), 1)
        # Should use external_price_yes (0.35)
        self.assertAlmostEqual(center[0]["price"], 0.35, places=2)


class TestScoreBucketsSigmaOverride(unittest.TestCase):

    def test_sigma_override_affects_probability(self):
        from weather.config import Config
        config = Config()
        config.min_probability = 0.0  # Don't filter out
        markets = [
            {"outcome_name": "50-54\u00b0F", "external_price_yes": 0.30, "best_ask": 0.30},
        ]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        scored_wide = score_buckets(markets, 52.0, today, config, metric="high", sigma_override=20.0)
        scored_narrow = score_buckets(markets, 52.0, today, config, metric="high", sigma_override=2.0)
        if scored_wide and scored_narrow:
            self.assertGreater(scored_narrow[0]["prob"], scored_wide[0]["prob"])


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

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = TradingState.load(path)

        self.assertIn("m-1", loaded.trades)
        self.assertAlmostEqual(loaded.trades["m-1"].cost_basis, 0.10)
        self.assertAlmostEqual(loaded.trades["m-1"].shares, 20.0)

        Path(path).unlink()

    def test_load_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{corrupted json!!")
            path = f.name

        state = TradingState.load(path)
        # Should return fresh state, not crash
        self.assertEqual(len(state.trades), 0)

        Path(path).unlink()


class TestAdaptiveSigmaConfig(unittest.TestCase):

    def test_config_default_true(self):
        from weather.config import Config
        config = Config()
        self.assertTrue(config.adaptive_sigma)

    def test_config_can_disable(self):
        from weather.config import Config
        config = Config(adaptive_sigma=False)
        self.assertFalse(config.adaptive_sigma)


if __name__ == "__main__":
    unittest.main()
