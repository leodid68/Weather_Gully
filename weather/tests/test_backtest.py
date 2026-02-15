"""Tests for weather.backtest — backtesting engine."""

import unittest
from unittest.mock import patch

from weather.backtest import (
    BacktestResult,
    BacktestTrade,
    compute_calibration_curve,
    run_backtest,
    _compute_max_drawdown,
    _compute_sharpe,
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
            horizon=3,
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


if __name__ == "__main__":
    unittest.main()
