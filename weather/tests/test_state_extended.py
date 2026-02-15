"""Tests for extended state features: forecast tracking, calibration, correlation guard, pending orders."""

import tempfile
import unittest
from pathlib import Path

from weather.state import PendingOrder, PredictionRecord, TradingState


class TestForecastChangeDetection(unittest.TestCase):

    def test_first_forecast_returns_none(self):
        state = TradingState()
        delta = state.get_forecast_delta("NYC", "2025-03-15", "high", 52.0)
        self.assertIsNone(delta)

    def test_store_then_detect_change(self):
        state = TradingState()
        state.store_forecast("NYC", "2025-03-15", "high", 52.0)
        delta = state.get_forecast_delta("NYC", "2025-03-15", "high", 55.0)
        self.assertAlmostEqual(delta, 3.0)

    def test_negative_change(self):
        state = TradingState()
        state.store_forecast("NYC", "2025-03-15", "high", 52.0)
        delta = state.get_forecast_delta("NYC", "2025-03-15", "high", 48.0)
        self.assertAlmostEqual(delta, -4.0)

    def test_different_locations_independent(self):
        state = TradingState()
        state.store_forecast("NYC", "2025-03-15", "high", 52.0)
        state.store_forecast("Chicago", "2025-03-15", "high", 45.0)
        delta = state.get_forecast_delta("NYC", "2025-03-15", "high", 52.0)
        self.assertAlmostEqual(delta, 0.0)

    def test_different_metrics_independent(self):
        state = TradingState()
        state.store_forecast("NYC", "2025-03-15", "high", 52.0)
        state.store_forecast("NYC", "2025-03-15", "low", 38.0)
        delta_high = state.get_forecast_delta("NYC", "2025-03-15", "high", 54.0)
        delta_low = state.get_forecast_delta("NYC", "2025-03-15", "low", 35.0)
        self.assertAlmostEqual(delta_high, 2.0)
        self.assertAlmostEqual(delta_low, -3.0)


class TestCalibration(unittest.TestCase):

    def test_empty_calibration(self):
        state = TradingState()
        stats = state.get_calibration_stats()
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["brier"])

    def test_perfect_calibration(self):
        state = TradingState()
        pred = PredictionRecord(
            market_id="m-1", event_id="e-1", location="NYC",
            forecast_date="2025-03-15", metric="high",
            our_probability=0.80, forecast_temp=52.0,
            bucket_low=50, bucket_high=54,
            resolved=True, actual_outcome=True,
        )
        state.record_prediction(pred)
        stats = state.get_calibration_stats()
        self.assertEqual(stats["count"], 1)
        # Brier = (0.80 - 1.0)^2 = 0.04
        self.assertAlmostEqual(stats["brier"], 0.04, places=4)
        self.assertAlmostEqual(stats["accuracy"], 1.0)

    def test_wrong_prediction(self):
        state = TradingState()
        pred = PredictionRecord(
            market_id="m-1", event_id="e-1", location="NYC",
            forecast_date="2025-03-15", metric="high",
            our_probability=0.80, forecast_temp=52.0,
            bucket_low=50, bucket_high=54,
            resolved=True, actual_outcome=False,
        )
        state.record_prediction(pred)
        stats = state.get_calibration_stats()
        # Brier = (0.80 - 0.0)^2 = 0.64
        self.assertAlmostEqual(stats["brier"], 0.64, places=4)
        self.assertAlmostEqual(stats["accuracy"], 0.0)

    def test_unresolved_not_counted(self):
        state = TradingState()
        pred = PredictionRecord(
            market_id="m-1", event_id="e-1", location="NYC",
            forecast_date="2025-03-15", metric="high",
            our_probability=0.80, forecast_temp=52.0,
            bucket_low=50, bucket_high=54,
            resolved=False, actual_outcome=None,
        )
        state.record_prediction(pred)
        stats = state.get_calibration_stats()
        self.assertEqual(stats["count"], 0)

    def test_mixed_predictions(self):
        state = TradingState()
        for i, (prob, outcome) in enumerate([(0.90, True), (0.70, False), (0.60, True)]):
            state.record_prediction(PredictionRecord(
                market_id=f"m-{i}", event_id=f"e-{i}", location="NYC",
                forecast_date="2025-03-15", metric="high",
                our_probability=prob, forecast_temp=52.0,
                bucket_low=50, bucket_high=54,
                resolved=True, actual_outcome=outcome,
            ))
        stats = state.get_calibration_stats()
        self.assertEqual(stats["count"], 3)
        # Brier = ((0.9-1)^2 + (0.7-0)^2 + (0.6-1)^2) / 3
        # = (0.01 + 0.49 + 0.16) / 3 = 0.22
        self.assertAlmostEqual(stats["brier"], 0.22, places=4)
        self.assertAlmostEqual(stats["accuracy"], 2 / 3, places=4)


class TestCorrelationGuard(unittest.TestCase):

    def test_no_position_initially(self):
        state = TradingState()
        self.assertFalse(state.has_event_position("evt-1"))

    def test_record_and_check(self):
        state = TradingState()
        state.record_event_position("evt-1", "m-1")
        self.assertTrue(state.has_event_position("evt-1"))
        self.assertFalse(state.has_event_position("evt-2"))

    def test_remove_event_position(self):
        state = TradingState()
        state.record_event_position("evt-1", "m-1")
        state.remove_event_position("evt-1")
        self.assertFalse(state.has_event_position("evt-1"))

    def test_remove_nonexistent_no_error(self):
        state = TradingState()
        state.remove_event_position("evt-999")  # Should not raise


class TestExtendedStateRoundtrip(unittest.TestCase):
    """Verify new fields survive save/load cycle."""

    def test_roundtrip_all_fields(self):
        state = TradingState()
        state.record_trade(
            market_id="m-1", outcome_name="50-54", side="yes",
            cost_basis=0.10, shares=20.0, location="NYC",
        )
        state.store_forecast("NYC", "2025-03-15", "high", 52.0)
        state.record_event_position("evt-1", "m-1")
        state.record_prediction(PredictionRecord(
            market_id="m-1", event_id="evt-1", location="NYC",
            forecast_date="2025-03-15", metric="high",
            our_probability=0.75, forecast_temp=52.0,
            bucket_low=50, bucket_high=54,
            timestamp="2025-03-14T12:00:00+00:00",
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = TradingState.load(path)

        # Trades
        self.assertIn("m-1", loaded.trades)

        # Forecast change detection
        self.assertIn("NYC|2025-03-15|high", loaded.previous_forecasts)
        self.assertAlmostEqual(loaded.previous_forecasts["NYC|2025-03-15|high"], 52.0)

        # Predictions
        self.assertIn("m-1", loaded.predictions)
        pred = loaded.predictions["m-1"]
        self.assertEqual(pred.location, "NYC")
        self.assertAlmostEqual(pred.our_probability, 0.75)
        self.assertEqual(pred.bucket_low, 50)

        # Event positions
        self.assertIn("evt-1", loaded.event_positions)
        self.assertEqual(loaded.event_positions["evt-1"], "m-1")

        Path(path).unlink()

    def test_load_old_format_backward_compatible(self):
        """State files without new fields should load without error."""
        import json

        old_data = {
            "trades": {},
            "analyzed_markets": [],
            "last_run": "2025-03-14T12:00:00+00:00",
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(old_data, f)
            path = f.name

        loaded = TradingState.load(path)
        self.assertEqual(len(loaded.previous_forecasts), 0)
        self.assertEqual(len(loaded.predictions), 0)
        self.assertEqual(len(loaded.event_positions), 0)
        self.assertEqual(len(loaded.pending_orders), 0)

        Path(path).unlink()


class TestPendingOrder(unittest.TestCase):

    def test_to_dict_roundtrip(self):
        po = PendingOrder(
            order_id="order-123",
            market_id="cond-1",
            side="BUY",
            price=0.35,
            size=20.0,
            timestamp="2025-03-15T12:00:00+00:00",
            token_id="token-yes-1",
        )
        d = po.to_dict()
        loaded = PendingOrder.from_dict(d)
        self.assertEqual(loaded.order_id, "order-123")
        self.assertEqual(loaded.market_id, "cond-1")
        self.assertEqual(loaded.side, "BUY")
        self.assertAlmostEqual(loaded.price, 0.35)
        self.assertAlmostEqual(loaded.size, 20.0)
        self.assertEqual(loaded.token_id, "token-yes-1")


class TestPendingOrderStateRoundtrip(unittest.TestCase):

    def test_save_load_pending_orders(self):
        state = TradingState()
        state.pending_orders["order-123"] = PendingOrder(
            order_id="order-123",
            market_id="cond-1",
            side="BUY",
            price=0.35,
            size=20.0,
            timestamp="2025-03-15T12:00:00+00:00",
            token_id="token-yes-1",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = TradingState.load(path)

        self.assertIn("order-123", loaded.pending_orders)
        po = loaded.pending_orders["order-123"]
        self.assertEqual(po.market_id, "cond-1")
        self.assertAlmostEqual(po.price, 0.35)
        self.assertAlmostEqual(po.size, 20.0)

        Path(path).unlink()

    def test_empty_pending_orders_roundtrip(self):
        state = TradingState()
        self.assertEqual(len(state.pending_orders), 0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = TradingState.load(path)
        self.assertEqual(len(loaded.pending_orders), 0)

        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
