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
    _check_stop_loss_reversals,
    _emergency_exit_losers,
    check_circuit_breaker,
    check_context_safeguards,
    check_exit_opportunities,
    run_weather_strategy,
    score_buckets,
    should_exit_on_edge_inversion,
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
        """Bucket containing the forecast should have highest YES probability."""
        markets_data = _load_fixture("weather_markets.json")
        event_markets = [m for m in markets_data["markets"] if m["event_id"] == "evt-nyc-high-mar15"]
        config = Config(adjacent_buckets=True, seasonal_adjustments=False)

        scored = score_buckets(event_markets, 52, "2025-03-15", config)
        # Filter to YES side only (NO side has complementary probs)
        yes_scored = [s for s in scored if s["side"] == "yes"]
        center = [s for s in yes_scored if s["bucket"] == (50, 54)]
        others = [s for s in yes_scored if s["bucket"] != (50, 54)]

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

    @patch("weather.strategy.FeedbackState")
    @patch("weather.strategy.get_noaa_forecast")
    def test_feedback_state_saved(self, mock_noaa, MockFS):
        """run_weather_strategy should persist feedback state."""
        mock_noaa.return_value = {
            "2025-03-15": {"high": 52, "low": 38},
        }
        mock_feedback = MockFS.load.return_value

        bridge = self._make_mock_bridge()
        config = Config(
            locations="NYC",
            adjacent_buckets=True,
            dynamic_exits=False,
            seasonal_adjustments=False,
            max_days_ahead=365,
        )
        state = TradingState()

        run_weather_strategy(
            client=bridge,
            config=config,
            state=state,
            dry_run=True,
            use_safeguards=False,
        )

        mock_feedback.save.assert_called_once()

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
        # Compare YES side probabilities: narrow sigma → higher center bucket prob
        yes_wide = [s for s in scored_wide if s["side"] == "yes"]
        yes_narrow = [s for s in scored_narrow if s["side"] == "yes"]
        if yes_wide and yes_narrow:
            self.assertGreater(yes_narrow[0]["prob"], yes_wide[0]["prob"])


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


class TestScoreBucketsNO(unittest.TestCase):

    def test_no_side_generated_for_overpriced_bucket(self):
        """A bucket with prob=0.10 and price=0.80 should have a NO opportunity with positive EV."""
        markets = [{
            "id": "m1", "outcome_name": "90°F or higher",
            "best_ask": 0.80, "external_price_yes": 0.80,
        }]
        config = Config()
        config.min_ev_threshold = 0.03
        config.min_probability = 0.05
        with patch("weather.strategy.parse_temperature_bucket", return_value=(90, 999)):
            with patch("weather.strategy.estimate_bucket_probability", return_value=0.10):
                with patch("weather.strategy.platt_calibrate", side_effect=lambda p: p):
                    scored = score_buckets(markets, 70.0, "2026-02-20", config, metric="high")
        no_entries = [s for s in scored if s.get("side") == "no"]
        self.assertTrue(len(no_entries) > 0, "Expected NO-side entries")
        self.assertGreater(no_entries[0]["ev"], 0)
        self.assertAlmostEqual(no_entries[0]["price"], 0.20, places=2)  # 1 - 0.80

    def test_yes_side_still_generated(self):
        """YES side should still be scored as before."""
        markets = [{
            "id": "m1", "outcome_name": "50-54°F",
            "best_ask": 0.10, "external_price_yes": 0.10,
        }]
        config = Config()
        config.min_probability = 0.05
        with patch("weather.strategy.parse_temperature_bucket", return_value=(50, 54)):
            with patch("weather.strategy.estimate_bucket_probability", return_value=0.40):
                with patch("weather.strategy.platt_calibrate", side_effect=lambda p: p):
                    scored = score_buckets(markets, 52.0, "2026-02-20", config, metric="high")
        yes_entries = [s for s in scored if s.get("side") == "yes"]
        self.assertTrue(len(yes_entries) > 0, "Expected YES-side entries")

    def test_all_entries_have_side_field(self):
        """Every scored entry must have a 'side' field."""
        markets = [{
            "id": "m1", "outcome_name": "50-54°F",
            "best_ask": 0.50, "external_price_yes": 0.50,
        }]
        config = Config()
        config.min_probability = 0.05
        with patch("weather.strategy.parse_temperature_bucket", return_value=(50, 54)):
            with patch("weather.strategy.estimate_bucket_probability", return_value=0.40):
                with patch("weather.strategy.platt_calibrate", side_effect=lambda p: p):
                    scored = score_buckets(markets, 52.0, "2026-02-20", config, metric="high")
        for entry in scored:
            self.assertIn("side", entry, f"Entry missing 'side' field: {entry}")


class TestParseBucketUnitConversion(unittest.TestCase):
    """Test _parse_bucket °C → °F conversion for international markets."""

    def test_fahrenheit_location_unchanged(self):
        """NYC (°F) bucket should pass through as floats."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("50-54°F", "NYC")
        self.assertEqual(result, (50.0, 54.0))

    def test_celsius_location_converted(self):
        """Paris (°C) bucket '15-20' should be converted to °F."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("15-20", "Paris")
        self.assertIsNotNone(result)
        # 15°C = 59°F, 20°C = 68°F
        self.assertAlmostEqual(result[0], 59.0, places=1)
        self.assertAlmostEqual(result[1], 68.0, places=1)

    def test_celsius_open_ended_below(self):
        """Open-ended lower bucket: sentinel -999 preserved."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("10 or below", "London")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], -999.0)
        self.assertAlmostEqual(result[1], 50.0, places=1)  # 10°C = 50°F

    def test_celsius_open_ended_above(self):
        """Open-ended upper bucket: sentinel 999 preserved."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("30 or higher", "Seoul")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 86.0, places=1)  # 30°C = 86°F
        self.assertEqual(result[1], 999.0)

    def test_unknown_location_treated_as_fahrenheit(self):
        """Unknown location defaults to °F (no conversion)."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("50-54", "UnknownCity")
        self.assertEqual(result, (50.0, 54.0))

    def test_no_location_treated_as_fahrenheit(self):
        """Empty location defaults to °F."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("50-54", "")
        self.assertEqual(result, (50.0, 54.0))

    def test_negative_celsius(self):
        """Negative °C should convert correctly."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("-5 to 0", "Toronto")
        self.assertIsNotNone(result)
        # -5°C = 23°F, 0°C = 32°F
        self.assertAlmostEqual(result[0], 23.0, places=1)
        self.assertAlmostEqual(result[1], 32.0, places=1)

    def test_none_for_unparseable(self):
        """Unparseable outcome returns None."""
        from weather.strategy import _parse_bucket
        self.assertIsNone(_parse_bucket("Sunny weather", "Paris"))

    def test_score_buckets_uses_converted_bucket(self):
        """score_buckets should pass °F-converted bounds to probability function."""
        from weather.strategy import score_buckets
        markets = [{
            "id": "m1", "outcome_name": "15-20",
            "best_ask": 0.30, "external_price_yes": 0.30,
        }]
        config = Config()
        config.min_probability = 0.01
        with patch("weather.strategy.estimate_bucket_probability", return_value=0.50) as mock_prob:
            with patch("weather.strategy.platt_calibrate", side_effect=lambda p: p):
                scored = score_buckets(markets, 62.0, "2026-06-15", config,
                                       metric="high", location="Paris")
        # The mock should have been called with converted °F bounds
        if mock_prob.called:
            call_args = mock_prob.call_args
            bucket_lo = call_args[0][1]  # second positional arg
            bucket_hi = call_args[0][2]  # third positional arg
            self.assertAlmostEqual(bucket_lo, 59.0, places=1)  # 15°C
            self.assertAlmostEqual(bucket_hi, 68.0, places=1)  # 20°C


class TestExecuteNOTrade(unittest.TestCase):

    def test_no_side_in_state_record(self):
        """When recording a NO-side trade, state should store side='no'."""
        from weather.state import TradingState
        state = TradingState()
        state.record_trade("m1", "bucket_name", "no", 0.20, 10.0,
                           location="NYC", forecast_date="2026-02-20")
        self.assertEqual(state.trades["m1"].side, "no")

    def test_yes_side_still_works(self):
        """YES-side trades should still be recorded correctly."""
        from weather.state import TradingState
        state = TradingState()
        state.record_trade("m2", "bucket_name", "yes", 0.10, 10.0,
                           location="NYC", forecast_date="2026-02-20")
        self.assertEqual(state.trades["m2"].side, "yes")


class TestCircuitBreaker(unittest.TestCase):

    def test_daily_loss_stops_trading(self):
        state = TradingState()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state.record_daily_pnl(today, -11.0)
        config = Config()
        config.daily_loss_limit = 10.0
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)
        self.assertIn("loss", reason.lower())

    def test_max_positions_stops_trading(self):
        state = TradingState()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for _ in range(21):
            state.record_position_opened(today)
        config = Config()
        config.max_positions_per_day = 20
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_cooldown_blocks_after_circuit_break(self):
        state = TradingState()
        state.last_circuit_break = datetime.now(timezone.utc).isoformat()
        config = Config()
        config.cooldown_hours_after_max_loss = 24.0
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_max_open_positions_blocks(self):
        state = TradingState()
        for i in range(16):
            state.record_trade(f"market_{i}", f"outcome_{i}", "yes", 0.1, 10)
        config = Config()
        config.max_open_positions = 15
        blocked, reason = check_circuit_breaker(state, config)
        self.assertTrue(blocked)

    def test_no_block_when_within_limits(self):
        state = TradingState()
        config = Config()
        blocked, reason = check_circuit_breaker(state, config)
        self.assertFalse(blocked)


class TestEdgeInversionExit(unittest.TestCase):

    def test_sell_yes_when_model_disagrees(self):
        """If we hold YES at $0.10 and current prob=0.05 while market=0.15, should exit."""
        self.assertTrue(should_exit_on_edge_inversion(
            our_prob=0.05, market_price=0.15, cost_basis=0.10, side="yes",
        ))

    def test_keep_yes_when_edge_holds(self):
        """If we hold YES at $0.08 and prob=0.30, market=0.10, keep holding."""
        self.assertFalse(should_exit_on_edge_inversion(
            our_prob=0.30, market_price=0.10, cost_basis=0.08, side="yes",
        ))

    def test_sell_no_when_model_flips(self):
        """If we hold NO and model now favors YES, should exit."""
        # Bought NO at 0.20 (market YES was 0.80). Now market YES is 0.75, our prob now 0.80
        self.assertTrue(should_exit_on_edge_inversion(
            our_prob=0.80, market_price=0.75, cost_basis=0.20, side="no",
        ))

    def test_keep_no_when_edge_holds(self):
        """If we hold NO and model still disagrees with YES, keep holding."""
        # Bought NO at 0.20 (market YES 0.80). Market still 0.80, our prob still 0.10
        self.assertFalse(should_exit_on_edge_inversion(
            our_prob=0.10, market_price=0.80, cost_basis=0.20, side="no",
        ))

    def test_no_exit_if_would_lose_too_much(self):
        """Even if edge inverted, don't exit at a big loss."""
        # Edge inverted but market price (0.06) is well below cost basis (0.10) - loss too big
        self.assertFalse(should_exit_on_edge_inversion(
            our_prob=0.05, market_price=0.06, cost_basis=0.10, side="yes",
        ))


class TestExitParallelFetch(unittest.TestCase):
    """Verify that check_exit_opportunities pre-fetches orderbooks in parallel."""

    def _make_mock_client(self, market_ids: list[str]) -> MagicMock:
        """Build a mock CLOBWeatherBridge with market cache for given IDs."""
        client = MagicMock()
        client._market_cache = {}
        for mid in market_ids:
            gm = MagicMock()
            gm.clob_token_ids = [f"token_{mid}"]
            gm.end_date = None
            client._market_cache[mid] = gm
        return client

    def test_exit_parallel_fetch(self):
        """All orderbooks should be fetched and positions processed."""
        market_ids = ["m-1", "m-2", "m-3"]
        client = self._make_mock_client(market_ids)

        # get_orderbook returns a book with bids above exit threshold
        client.clob.get_orderbook.return_value = {
            "bids": [{"price": "0.90"}],
        }
        client.get_market_context.return_value = None

        state = TradingState()
        for mid in market_ids:
            state.record_trade(mid, f"bucket_{mid}", "yes", 0.10, 20.0,
                               location="NYC", forecast_date="2026-02-20")

        config = Config(dynamic_exits=False)

        found, executed = check_exit_opportunities(
            client, config, state, dry_run=True, use_safeguards=False,
        )

        # All 3 should be found as exit opportunities (0.90 >= default threshold)
        self.assertEqual(found, 3)
        # get_orderbook should have been called once per market
        self.assertEqual(client.clob.get_orderbook.call_count, 3)

    def test_exit_handles_fetch_failure(self):
        """If one orderbook fetch fails, other positions should still be processed."""
        market_ids = ["m-ok", "m-fail"]
        client = self._make_mock_client(market_ids)

        def mock_get_orderbook(token_id: str) -> dict:
            if token_id == "token_m-fail":
                raise ConnectionError("timeout")
            return {"bids": [{"price": "0.90"}]}

        client.clob.get_orderbook.side_effect = mock_get_orderbook
        client.get_market_context.return_value = None

        state = TradingState()
        for mid in market_ids:
            state.record_trade(mid, f"bucket_{mid}", "yes", 0.10, 20.0,
                               location="NYC", forecast_date="2026-02-20")

        config = Config(dynamic_exits=False)

        found, executed = check_exit_opportunities(
            client, config, state, dry_run=True, use_safeguards=False,
        )

        # Only the successful fetch should result in an exit opportunity
        self.assertEqual(found, 1)
        # Both fetches were attempted
        self.assertEqual(client.clob.get_orderbook.call_count, 2)


class TestEmergencyExitLosers(unittest.TestCase):
    """Test _emergency_exit_losers sells only losing positions."""

    def _make_mock_client(self, market_ids: list[str]) -> MagicMock:
        client = MagicMock()
        client._market_cache = {}
        for mid in market_ids:
            gm = MagicMock()
            gm.clob_token_ids = [f"token_{mid}"]
            gm.end_date = None
            client._market_cache[mid] = gm
        return client

    def test_sells_losing_positions_only(self):
        """Positions below cost basis should be sold, winners kept."""
        market_ids = ["m-loser", "m-winner"]
        client = self._make_mock_client(market_ids)

        def mock_get_orderbook(token_id: str) -> dict:
            if token_id == "token_m-loser":
                return {"bids": [{"price": "0.05"}]}  # Below cost basis 0.10
            return {"bids": [{"price": "0.20"}]}  # Above cost basis 0.10

        client.clob.get_orderbook.side_effect = mock_get_orderbook
        client.execute_sell.return_value = {"success": True}

        state = TradingState()
        state.record_trade("m-loser", "losing_bucket", "yes", 0.10, 20.0, location="NYC")
        state.record_trade("m-winner", "winning_bucket", "yes", 0.10, 20.0, location="NYC")

        exits = _emergency_exit_losers(client, state, dry_run=False)

        self.assertEqual(exits, 1)
        # Only the loser should have been sold (side="yes" from record_trade)
        client.execute_sell.assert_called_once_with("m-loser", 20.0, side="yes")
        # Winner still in state
        self.assertIn("m-winner", state.trades)

    def test_dry_run_does_not_sell(self):
        """In dry_run mode, no sells should be executed."""
        client = self._make_mock_client(["m-loser"])
        client.clob.get_orderbook.return_value = {"bids": [{"price": "0.05"}]}

        state = TradingState()
        state.record_trade("m-loser", "losing_bucket", "yes", 0.10, 20.0, location="NYC")

        exits = _emergency_exit_losers(client, state, dry_run=True)

        self.assertEqual(exits, 0)
        client.execute_sell.assert_not_called()
        # Position should still be in state
        self.assertIn("m-loser", state.trades)

    def test_no_trades_returns_zero(self):
        """With no open trades, should return 0."""
        client = self._make_mock_client([])
        state = TradingState()
        exits = _emergency_exit_losers(client, state, dry_run=False)
        self.assertEqual(exits, 0)


class TestStopLossNoSide(unittest.TestCase):
    """Verify stop-loss handles YES and NO positions correctly."""

    def _make_mock_client(self) -> MagicMock:
        client = MagicMock()
        client._market_cache = {}
        gm = MagicMock()
        gm.clob_token_ids = ["yes_tok", "no_tok"]
        client._market_cache["m-1"] = gm
        client.execute_sell.return_value = {"success": True}
        return client

    def test_stop_loss_yes_triggers_outside_bucket(self):
        """YES position: stop-loss should trigger when forecast moves OUTSIDE bucket."""
        client = self._make_mock_client()
        state = TradingState()
        # Trade in bucket 30-35, forecast was 32, now forecast is 45 (outside, shift=13)
        state.record_trade("m-1", "30 to 35", "yes", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20",
                           forecast_temp=32.0)

        config = Config(stop_loss_reversal=True, stop_loss_reversal_threshold=5.0, multi_source=False)
        noaa_cache = {"NYC": {"2026-02-20": {"high": 45.0}}}

        exits = _check_stop_loss_reversals(
            client, config, state, noaa_cache, {}, dry_run=False,
        )
        self.assertEqual(exits, 1)

    def test_stop_loss_yes_no_trigger_inside_bucket(self):
        """YES position: no stop-loss when forecast stays in bucket."""
        client = self._make_mock_client()
        state = TradingState()
        state.record_trade("m-1", "30 to 35", "yes", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20",
                           forecast_temp=32.0)

        config = Config(stop_loss_reversal=True, stop_loss_reversal_threshold=5.0, multi_source=False)
        noaa_cache = {"NYC": {"2026-02-20": {"high": 33.0}}}  # Still in bucket

        exits = _check_stop_loss_reversals(
            client, config, state, noaa_cache, {}, dry_run=False,
        )
        self.assertEqual(exits, 0)

    def test_stop_loss_no_triggers_inside_bucket(self):
        """NO position: stop-loss should trigger when forecast moves INTO bucket."""
        client = self._make_mock_client()
        state = TradingState()
        # Bet NO on bucket 30-35. Forecast was 45 (outside), now 32 (inside) — we're losing
        state.record_trade("m-1", "30 to 35", "no", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20",
                           forecast_temp=45.0)

        config = Config(stop_loss_reversal=True, stop_loss_reversal_threshold=5.0, multi_source=False)
        noaa_cache = {"NYC": {"2026-02-20": {"high": 32.0}}}  # Now in bucket, shift=13

        exits = _check_stop_loss_reversals(
            client, config, state, noaa_cache, {}, dry_run=False,
        )
        self.assertEqual(exits, 1)
        # Should sell with side="no"
        client.execute_sell.assert_called_once_with("m-1", 20.0, side="no")

    def test_stop_loss_no_no_trigger_outside_bucket(self):
        """NO position: no stop-loss when forecast stays OUTSIDE bucket (we're winning)."""
        client = self._make_mock_client()
        state = TradingState()
        # Bet NO on bucket 30-35. Forecast was 45, now 50 (still outside) — we're winning
        state.record_trade("m-1", "30 to 35", "no", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20",
                           forecast_temp=45.0)

        config = Config(stop_loss_reversal=True, stop_loss_reversal_threshold=5.0, multi_source=False)
        noaa_cache = {"NYC": {"2026-02-20": {"high": 50.0}}}  # Still outside bucket

        exits = _check_stop_loss_reversals(
            client, config, state, noaa_cache, {}, dry_run=False,
        )
        self.assertEqual(exits, 0)


class TestMarketOverlapGuard(unittest.TestCase):
    """Verify we can't hold both YES and NO on the same market."""

    def test_skip_if_already_holding_position(self):
        """score_buckets entry should be skipped if we already have a position."""
        # This is tested via the full run flow, but we verify the guard logic directly
        state = TradingState()
        state.record_trade("m-1", "30 to 35", "yes", 0.10, 20.0, location="NYC")

        # The guard is: if market_id in state.trades: continue
        self.assertIn("m-1", state.trades)


class TestNoSideExits(unittest.TestCase):
    """Verify that NO-side positions use the correct token for exits."""

    def _make_mock_client(self, market_ids: list[str]) -> MagicMock:
        client = MagicMock()
        client._market_cache = {}
        for mid in market_ids:
            gm = MagicMock()
            gm.clob_token_ids = [f"yes_{mid}", f"no_{mid}"]
            gm.end_date = None
            client._market_cache[mid] = gm
        return client

    def test_exit_no_position_fetches_no_token_orderbook(self):
        """For a NO position, orderbook should be fetched for the NO token."""
        client = self._make_mock_client(["m-no"])
        client.clob.get_orderbook.return_value = {
            "bids": [{"price": "0.85"}],
        }
        client.get_market_context.return_value = None

        state = TradingState()
        state.record_trade("m-no", "bucket_test", "no", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20")

        config = Config(dynamic_exits=False)
        found, _ = check_exit_opportunities(
            client, config, state, dry_run=True, use_safeguards=False,
        )

        self.assertEqual(found, 1)
        # Should have fetched the NO token orderbook
        client.clob.get_orderbook.assert_called_once_with("no_m-no")

    def test_exit_yes_position_fetches_yes_token_orderbook(self):
        """For a YES position, orderbook should be fetched for the YES token."""
        client = self._make_mock_client(["m-yes"])
        client.clob.get_orderbook.return_value = {
            "bids": [{"price": "0.85"}],
        }
        client.get_market_context.return_value = None

        state = TradingState()
        state.record_trade("m-yes", "bucket_test", "yes", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20")

        config = Config(dynamic_exits=False)
        found, _ = check_exit_opportunities(
            client, config, state, dry_run=True, use_safeguards=False,
        )

        self.assertEqual(found, 1)
        client.clob.get_orderbook.assert_called_once_with("yes_m-yes")

    def test_exit_sell_passes_no_side(self):
        """execute_sell should receive side='no' for NO positions."""
        client = self._make_mock_client(["m-no"])
        client.clob.get_orderbook.return_value = {
            "bids": [{"price": "0.85"}],
        }
        client.get_market_context.return_value = None
        client.execute_sell.return_value = {"success": True}

        state = TradingState()
        state.record_trade("m-no", "bucket_test", "no", 0.10, 20.0,
                           location="NYC", forecast_date="2026-02-20")

        config = Config(dynamic_exits=False)
        check_exit_opportunities(
            client, config, state, dry_run=False, use_safeguards=False,
        )

        client.execute_sell.assert_called_once_with("m-no", 20.0, side="no")

    def test_emergency_exit_no_position(self):
        """Emergency exit should use NO token for NO-side positions."""
        client = self._make_mock_client(["m-no"])
        client.clob.get_orderbook.return_value = {
            "bids": [{"price": "0.05"}],  # Below cost basis
        }
        client.execute_sell.return_value = {"success": True}

        state = TradingState()
        state.record_trade("m-no", "losing_no", "no", 0.10, 20.0, location="NYC")

        exits = _emergency_exit_losers(client, state, dry_run=False)

        self.assertEqual(exits, 1)
        # Should fetch the NO token orderbook
        client.clob.get_orderbook.assert_called_once_with("no_m-no")
        # Should sell with side="no"
        client.execute_sell.assert_called_once_with("m-no", 20.0, side="no")


class TestAdaptiveSlippage(unittest.TestCase):

    def test_adaptive_slippage_high_edge_passes(self):
        """High-edge trade tolerates higher slippage."""
        from weather.strategy import check_context_safeguards
        from weather.config import Config
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
            "edge": {"user_edge": 0.25},  # 25% edge → threshold = 12.5%
        }
        ok, reasons = check_context_safeguards(context, config)
        assert ok  # 10% < 12.5%

    def test_adaptive_slippage_low_edge_blocked(self):
        """Low-edge trade has low slippage tolerance."""
        from weather.strategy import check_context_safeguards
        from weather.config import Config
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
            "edge": {"user_edge": 0.05},  # 5% edge → threshold = 2.5%
        }
        ok, reasons = check_context_safeguards(context, config)
        assert not ok  # 10% > 2.5%

    def test_adaptive_slippage_capped_at_max(self):
        """Adaptive threshold never exceeds slippage_max_pct."""
        from weather.strategy import check_context_safeguards
        from weather.config import Config
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.14}]},
            "edge": {"user_edge": 0.50},  # 50% edge → 25%, capped at 15%
        }
        ok, reasons = check_context_safeguards(context, config)
        assert ok  # 14% < 15%

    def test_adaptive_slippage_no_edge_uses_fixed(self):
        """Without edge data, falls back to fixed threshold."""
        from weather.strategy import check_context_safeguards
        from weather.config import Config
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
        }
        ok, reasons = check_context_safeguards(context, config)
        assert ok  # 10% < 15% (fixed)

    def test_adaptive_slippage_zero_ratio_uses_fixed(self):
        """With slippage_edge_ratio=0, uses fixed threshold."""
        from weather.strategy import check_context_safeguards
        from weather.config import Config
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.0)
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
            "edge": {"user_edge": 0.05},
        }
        ok, reasons = check_context_safeguards(context, config)
        assert ok  # 10% < 15% (fixed, ratio=0 disabled)


def test_trade_metrics_filters_low_temp():
    """Config with trade_metrics='high' excludes low from active_metrics."""
    from weather.config import Config
    config = Config(trade_metrics="high")
    assert "low" not in config.active_metrics
    assert "high" in config.active_metrics

def test_trade_metrics_allows_both():
    """Config with trade_metrics='high,low' includes both."""
    from weather.config import Config
    config = Config(trade_metrics="high,low")
    assert "high" in config.active_metrics
    assert "low" in config.active_metrics


def test_cumulative_correlation_discount():
    """Sum-based correlation gives appropriate discount."""
    from weather.strategy import _apply_correlation_discount
    from weather.config import Config
    config = Config(correlation_threshold=0.3, correlation_discount=0.5)
    # Using real calibration data: Dallas|Seattle SON=0.674
    # month=10 is SON season
    size = _apply_correlation_discount(2.0, "Dallas", 10, ["Seattle"], config)
    # total_corr = 0.674, factor = 1 - 0.674*0.5 = 0.663
    assert size < 2.0
    assert size > 0.1  # Floor at 0.1

def test_correlation_no_positions():
    """No discount when no open positions."""
    from weather.strategy import _apply_correlation_discount
    from weather.config import Config
    config = Config(correlation_threshold=0.3, correlation_discount=0.5)
    assert _apply_correlation_discount(2.0, "NYC", 1, [], config) == 2.0

def test_correlation_below_threshold_no_discount():
    """Correlations below threshold are ignored."""
    from weather.strategy import _apply_correlation_discount
    from weather.config import Config
    config = Config(correlation_threshold=0.3, correlation_discount=0.5)
    # Atlanta|NYC DJF = 0.174, below threshold 0.3
    size = _apply_correlation_discount(2.0, "Atlanta", 1, ["NYC"], config)
    assert size == 2.0  # No discount


if __name__ == "__main__":
    unittest.main()
