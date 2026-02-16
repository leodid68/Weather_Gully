"""Tests for weather.metrics module."""

import unittest
from weather.metrics import brier_score, sharpe_ratio, win_rate, average_edge, calibration_table


class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        predictions = [(1.0, True), (0.0, False), (1.0, True)]
        self.assertAlmostEqual(brier_score(predictions), 0.0)

    def test_worst_predictions(self):
        predictions = [(0.0, True), (1.0, False)]
        self.assertAlmostEqual(brier_score(predictions), 1.0)

    def test_empty_returns_none(self):
        self.assertIsNone(brier_score([]))

    def test_fifty_fifty(self):
        predictions = [(0.5, True), (0.5, False)]
        self.assertAlmostEqual(brier_score(predictions), 0.25)


class TestSharpeRatio(unittest.TestCase):
    def test_positive_sharpe(self):
        returns = [0.05, 0.03, 0.04, 0.06, 0.02]
        result = sharpe_ratio(returns)
        self.assertGreater(result, 0)

    def test_negative_sharpe(self):
        returns = [-0.05, -0.03, -0.04, -0.06, -0.02]
        result = sharpe_ratio(returns)
        self.assertLess(result, 0)

    def test_empty_returns_none(self):
        self.assertIsNone(sharpe_ratio([]))

    def test_single_return_none(self):
        self.assertIsNone(sharpe_ratio([0.05]))


class TestWinRate(unittest.TestCase):
    def test_all_wins(self):
        self.assertAlmostEqual(win_rate([0.1, 0.2, 0.3]), 1.0)

    def test_all_losses(self):
        self.assertAlmostEqual(win_rate([-0.1, -0.2]), 0.0)

    def test_empty_returns_none(self):
        self.assertIsNone(win_rate([]))

    def test_mixed(self):
        self.assertAlmostEqual(win_rate([0.1, -0.1, 0.2, -0.2]), 0.5)


class TestAverageEdge(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(average_edge([0.1, -0.2, 0.3]), 0.2)

    def test_empty_returns_none(self):
        self.assertIsNone(average_edge([]))


class TestCalibrationTable(unittest.TestCase):
    def test_bins_predictions(self):
        predictions = [(0.15, True), (0.15, False), (0.85, True), (0.85, True)]
        table = calibration_table(predictions)
        self.assertIn("0.1-0.2", table)
        self.assertAlmostEqual(table["0.1-0.2"]["actual_freq"], 0.5)
        self.assertEqual(table["0.1-0.2"]["count"], 2)
        self.assertIn("0.8-0.9", table)
        self.assertAlmostEqual(table["0.8-0.9"]["actual_freq"], 1.0)

    def test_empty_predictions(self):
        table = calibration_table([])
        self.assertEqual(len(table), 0)


if __name__ == "__main__":
    unittest.main()
