"""Tests for weather.backtest — backtesting engine."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from weather.backtest import (
    BacktestResult,
    BacktestTrade,
    compute_calibration_curve,
    run_backtest,
    _compute_max_drawdown,
    _compute_sharpe,
    _load_price_snapshots,
    _simulate_market_price,
)


class TestBacktestTrade(unittest.TestCase):

    def test_dataclass_creation(self):
        trade = BacktestTrade(
            date="2025-06-15",
            location="NYC",
            metric="high",
            bucket=(80, 84),
            our_probability=0.65,
            simulated_price=0.14,
            forecast_temp=82.0,
            actual_temp=83.0,
            won=True,
            pnl=0.86,
        )
        self.assertTrue(trade.won)
        self.assertAlmostEqual(trade.pnl, 0.86)


class TestBacktestResult(unittest.TestCase):

    def test_summary_output(self):
        result = BacktestResult(
            trades=[],
            brier_score=0.18,
            accuracy=0.65,
            total_pnl=12.50,
            roi=0.25,
            max_drawdown=3.00,
            sharpe_ratio=1.2,
        )
        summary = result.summary()
        self.assertIn("Brier score", summary)
        self.assertIn("0.1800", summary)
        self.assertIn("Total P&L", summary)

    def test_empty_result(self):
        result = BacktestResult()
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.brier_score, 0.0)
        summary = result.summary()
        self.assertIn("Total trades:      0", summary)


class TestComputeCalibrationCurve(unittest.TestCase):

    def test_basic_curve(self):
        predictions = [
            (0.1, False), (0.1, False), (0.1, True),
            (0.5, True), (0.5, False), (0.5, True),
            (0.9, True), (0.9, True), (0.9, True),
        ]
        curve = compute_calibration_curve(predictions, n_bins=10)
        self.assertGreater(len(curve), 0)
        # Each entry is (mean_predicted, fraction_positive)
        for pred, actual in curve:
            self.assertGreaterEqual(pred, 0.0)
            self.assertLessEqual(pred, 1.0)
            self.assertGreaterEqual(actual, 0.0)
            self.assertLessEqual(actual, 1.0)

    def test_empty_predictions(self):
        curve = compute_calibration_curve([])
        self.assertEqual(curve, [])

    def test_perfect_calibration(self):
        # All predictions at 1.0, all outcomes True
        predictions = [(1.0, True)] * 10
        curve = compute_calibration_curve(predictions, n_bins=10)
        self.assertGreater(len(curve), 0)
        # Should show ~1.0 predicted → ~1.0 actual
        last = curve[-1]
        self.assertAlmostEqual(last[1], 1.0)


class TestMaxDrawdown(unittest.TestCase):

    def test_no_drawdown(self):
        trades = [
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 0.90),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 0.90),
        ]
        self.assertAlmostEqual(_compute_max_drawdown(trades), 0.0)

    def test_full_drawdown(self):
        trades = [
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 1.00),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, False, -0.50),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, False, -0.50),
        ]
        dd = _compute_max_drawdown(trades)
        self.assertAlmostEqual(dd, 1.0)

    def test_empty_trades(self):
        self.assertAlmostEqual(_compute_max_drawdown([]), 0.0)


class TestSharpeRatio(unittest.TestCase):

    def test_zero_variance(self):
        trades = [
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 1.0),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 1.0),
        ]
        self.assertAlmostEqual(_compute_sharpe(trades), 0.0)

    def test_positive_sharpe(self):
        trades = [
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 0.90),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 0.80),
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, False, -0.10),
        ]
        sharpe = _compute_sharpe(trades)
        self.assertGreater(sharpe, 0)

    def test_single_trade(self):
        trades = [
            BacktestTrade("", "", "", (0, 0), 0.5, 0.1, 0, 0, True, 1.0),
        ]
        self.assertAlmostEqual(_compute_sharpe(trades), 0.0)


class TestRunBacktest(unittest.TestCase):

    def setUp(self):
        """Reset calibration cache to avoid pollution from other test modules."""
        import weather.probability as _prob
        self._saved_cache = _prob._calibration_cache
        self._saved_mtime = _prob._calibration_mtime
        _prob._calibration_cache = {}  # Empty = use hardcoded defaults
        _prob._calibration_mtime = 0.0

    def tearDown(self):
        import weather.probability as _prob
        _prob._calibration_cache = self._saved_cache
        _prob._calibration_mtime = self._saved_mtime

    @patch("weather.backtest.get_historical_actuals")
    @patch("weather.backtest.get_historical_forecasts")
    def test_basic_backtest(self, mock_forecasts, mock_actuals):
        # Forecasts are keyed by target_date (API deduplication)
        mock_forecasts.return_value = {
            "2025-06-04": {
                "2025-06-04": {
                    "gfs_high": 82.0, "gfs_low": 65.0,
                    "ecmwf_high": 84.0, "ecmwf_low": 66.0,
                },
            },
        }
        mock_actuals.return_value = {
            "2025-06-04": {"high": 83.0, "low": 64.5},
        }

        result = run_backtest(
            locations=["NYC"],
            start_date="2025-06-04",
            end_date="2025-06-04",
            horizon=1,
        )

        self.assertIsInstance(result, BacktestResult)
        # Should have some trades (high-probability bucket should trigger)
        self.assertGreater(len(result.trades), 0)
        self.assertGreater(result.brier_score, 0)

    @patch("weather.backtest.get_historical_actuals")
    @patch("weather.backtest.get_historical_forecasts")
    def test_no_data_returns_empty(self, mock_forecasts, mock_actuals):
        mock_forecasts.return_value = {}
        mock_actuals.return_value = {}

        result = run_backtest(
            locations=["NYC"],
            start_date="2025-06-01",
            end_date="2025-06-30",
        )

        self.assertEqual(len(result.trades), 0)

    @patch("weather.backtest.get_historical_actuals")
    @patch("weather.backtest.get_historical_forecasts")
    def test_hundred_percent_win(self, mock_forecasts, mock_actuals):
        """When forecast perfectly matches actual, trades should win."""
        # Forecasts are keyed by target_date (API deduplication)
        mock_forecasts.return_value = {
            "2025-06-04": {
                "2025-06-04": {
                    "gfs_high": 80.0, "gfs_low": 60.0,
                    "ecmwf_high": 80.0, "ecmwf_low": 60.0,
                },
            },
        }
        mock_actuals.return_value = {
            "2025-06-04": {"high": 80.0, "low": 60.0},
        }

        result = run_backtest(
            locations=["NYC"],
            start_date="2025-06-04",
            end_date="2025-06-04",
            horizon=3,
            entry_threshold=0.0,  # Take all trades
        )

        # At least the center bucket should win
        winning_trades = [t for t in result.trades if t.won]
        self.assertGreater(len(winning_trades), 0)

    def test_unknown_location_handled(self):
        result = run_backtest(
            locations=["UNKNOWN"],
            start_date="2025-06-01",
            end_date="2025-06-30",
        )
        self.assertEqual(len(result.trades), 0)


class TestSimulateMarketPrice(unittest.TestCase):

    def test_range_clamped(self):
        """Price should always be in [0.02, 0.98]."""
        for prob in [0.0, 0.01, 0.5, 0.99, 1.0]:
            for seed in range(100):
                price = _simulate_market_price(prob, 7, seed)
                self.assertGreaterEqual(price, 0.02)
                self.assertLessEqual(price, 0.98)

    def test_deterministic(self):
        """Same seed should produce same price."""
        p1 = _simulate_market_price(0.5, 7, 42)
        p2 = _simulate_market_price(0.5, 7, 42)
        self.assertEqual(p1, p2)

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different prices."""
        p1 = _simulate_market_price(0.5, 7, 1)
        p2 = _simulate_market_price(0.5, 7, 2)
        # They could theoretically be equal but extremely unlikely
        self.assertNotAlmostEqual(p1, p2, places=6)

    def test_center_probability_price_near_prob(self):
        """For the center bucket (high prob), price should be roughly near prob."""
        prices = [_simulate_market_price(0.60, 7, seed) for seed in range(200)]
        avg = sum(prices) / len(prices)
        # Average should be close to 0.60 (within noise stddev)
        self.assertAlmostEqual(avg, 0.60, delta=0.03)


class TestLoadPriceSnapshots(unittest.TestCase):

    def test_loads_and_indexes(self):
        snapshots = [
            {
                "date": "2026-02-15",
                "location": "NYC",
                "metric": "high",
                "bucket_lo": 50,
                "bucket_hi": 54,
                "best_ask": 0.45,
                "best_bid": 0.42,
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(snapshots, f)
            tmp_path = f.name
        try:
            result = _load_price_snapshots(tmp_path)
            key = "2026-02-15|NYC|high|50,54"
            self.assertIn(key, result)
            self.assertAlmostEqual(result[key]["best_ask"], 0.45)
        finally:
            os.unlink(tmp_path)

    def test_missing_file_returns_empty(self):
        result = _load_price_snapshots("/nonexistent/path.json")
        self.assertEqual(result, {})

    def test_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            tmp_path = f.name
        try:
            result = _load_price_snapshots(tmp_path)
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp_path)


class TestBacktestWithSnapshots(unittest.TestCase):

    @patch("weather.backtest.get_historical_actuals")
    @patch("weather.backtest.get_historical_forecasts")
    def test_backtest_uses_snapshot_price(self, mock_forecasts, mock_actuals):
        """When snapshots are available, their prices should be used."""
        mock_forecasts.return_value = {
            "2025-06-04": {
                "2025-06-04": {
                    "gfs_high": 82.0, "gfs_low": 65.0,
                    "ecmwf_high": 84.0, "ecmwf_low": 66.0,
                },
            },
        }
        mock_actuals.return_value = {
            "2025-06-04": {"high": 83.0, "low": 64.5},
        }

        # Create a snapshot file with a known price for the center bucket
        snapshots = [{
            "date": "2025-06-04",
            "location": "NYC",
            "metric": "high",
            "bucket_lo": 80,
            "bucket_hi": 84,
            "best_ask": 0.55,
            "best_bid": 0.52,
        }]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(snapshots, f)
            tmp_path = f.name

        try:
            result = run_backtest(
                locations=["NYC"],
                start_date="2025-06-04",
                end_date="2025-06-04",
                horizon=3,
                entry_threshold=0.0,
                snapshot_path=tmp_path,
            )

            # Find the trade for bucket (80, 84)
            matching = [t for t in result.trades if t.bucket == (80, 84)]
            if matching:
                self.assertAlmostEqual(matching[0].simulated_price, 0.55)
        finally:
            os.unlink(tmp_path)

    @patch("weather.backtest.get_historical_actuals")
    @patch("weather.backtest.get_historical_forecasts")
    def test_fallback_when_no_snapshots(self, mock_forecasts, mock_actuals):
        """Without snapshots, prices should use probabilistic model (not 1/N)."""
        mock_forecasts.return_value = {
            "2025-06-04": {
                "2025-06-04": {
                    "gfs_high": 82.0, "gfs_low": 65.0,
                    "ecmwf_high": 84.0, "ecmwf_low": 66.0,
                },
            },
        }
        mock_actuals.return_value = {
            "2025-06-04": {"high": 83.0, "low": 64.5},
        }

        result = run_backtest(
            locations=["NYC"],
            start_date="2025-06-04",
            end_date="2025-06-04",
            horizon=3,
            entry_threshold=0.0,
        )

        # With 7 buckets, old 1/N would give ~0.14. The new model should
        # produce prices correlated with probability (center bucket higher,
        # tail buckets lower)
        if result.trades:
            prices = [t.simulated_price for t in result.trades]
            # At least one price should be significantly different from 0.14
            self.assertTrue(any(abs(p - 0.14) > 0.05 for p in prices),
                            f"All prices too close to 0.14: {prices}")


if __name__ == "__main__":
    unittest.main()
