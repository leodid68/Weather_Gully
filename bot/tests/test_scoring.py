"""Tests for bot.scoring — proper scoring rules."""

import math
import unittest

from bot.scoring import brier_score, calibration_curve, log_score


class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        preds = [1.0, 0.0, 1.0]
        outcomes = [1, 0, 1]
        self.assertAlmostEqual(brier_score(preds, outcomes), 0.0)

    def test_worst_predictions(self):
        preds = [0.0, 1.0]
        outcomes = [1, 0]
        self.assertAlmostEqual(brier_score(preds, outcomes), 1.0)

    def test_uniform_predictions(self):
        preds = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1, 0, 1, 0]
        self.assertAlmostEqual(brier_score(preds, outcomes), 0.25)

    def test_empty(self):
        self.assertTrue(math.isnan(brier_score([], [])))

    def test_single(self):
        self.assertAlmostEqual(brier_score([0.7], [1]), 0.09)


class TestLogScore(unittest.TestCase):
    def test_good_predictions(self):
        preds = [0.9, 0.1]
        outcomes = [1, 0]
        score = log_score(preds, outcomes)
        self.assertLess(score, 0)  # log score is negative
        self.assertGreater(score, -1)  # but not too bad

    def test_bad_predictions(self):
        preds = [0.1, 0.9]
        outcomes = [1, 0]
        score = log_score(preds, outcomes)
        self.assertLess(score, -1)  # penalised heavily

    def test_empty(self):
        self.assertTrue(math.isnan(log_score([], [])))

    def test_clipping(self):
        # Should not raise with extreme values
        score = log_score([0.0, 1.0], [1, 0])
        self.assertTrue(math.isfinite(score))

    def test_perfect(self):
        # Near-perfect predictions
        preds = [0.999, 0.001]
        outcomes = [1, 0]
        score = log_score(preds, outcomes)
        self.assertGreater(score, -0.01)


class TestCalibrationCurve(unittest.TestCase):
    def test_basic_structure(self):
        preds = [0.1, 0.2, 0.3, 0.8, 0.9]
        outcomes = [0, 0, 1, 1, 1]
        result = calibration_curve(preds, outcomes, n_bins=5)
        self.assertIn("bins", result)
        self.assertIn("predicted", result)
        self.assertIn("actual", result)
        self.assertIn("count", result)
        self.assertEqual(len(result["bins"]), 5)

    def test_ten_bins_default(self):
        preds = [i / 10 for i in range(10)]
        outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        result = calibration_curve(preds, outcomes)
        self.assertEqual(len(result["bins"]), 10)

    def test_empty(self):
        result = calibration_curve([], [])
        self.assertEqual(sum(result["count"]), 0)


if __name__ == "__main__":
    unittest.main()
