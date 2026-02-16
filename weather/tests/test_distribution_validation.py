"""Tests for distribution validation module."""

import math
import unittest

from weather.distribution_validation import _bucket_hit, score_distribution
from weather.probability import _normal_cdf


class TestBucketHit(unittest.TestCase):

    def test_center_bucket_high_probability(self):
        """Error near 0 with narrow sigma -> high probability for center bucket."""
        prob = _bucket_hit(0.5, 2.0, -2.5, 2.5, _normal_cdf)
        self.assertGreater(prob, 0.5)

    def test_tail_bucket_low_probability(self):
        """Error near 0, far bucket -> low probability."""
        prob = _bucket_hit(0.0, 2.0, 10, 15, _normal_cdf)
        self.assertLess(prob, 0.01)

    def test_open_lower_bucket(self):
        """Sentinel -999 -> open lower bound."""
        prob = _bucket_hit(-5.0, 2.0, -999, -2.5, _normal_cdf)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_open_upper_bucket(self):
        """Sentinel 999 -> open upper bound."""
        prob = _bucket_hit(5.0, 2.0, 2.5, 999, _normal_cdf)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_zero_sigma_clamped(self):
        prob = _bucket_hit(0.0, 0.0, -2.5, 2.5, _normal_cdf)
        self.assertGreater(prob, 0.0)


class TestScoreDistribution(unittest.TestCase):

    def test_perfect_sigma_scores_well(self):
        """Errors drawn from N(0, sigma) should score well with Normal CDF."""
        import random
        random.seed(42)
        sigma = 3.0
        errors = [random.gauss(0, sigma) for _ in range(500)]
        result = score_distribution(errors, _normal_cdf, sigma)
        self.assertLess(result["brier"], 0.8)
        self.assertEqual(result["n"], 500)

    def test_wrong_sigma_scores_worse(self):
        """Using wrong sigma should give worse Brier score."""
        import random
        random.seed(42)
        sigma = 3.0
        errors = [random.gauss(0, sigma) for _ in range(500)]
        good = score_distribution(errors, _normal_cdf, sigma)
        bad = score_distribution(errors, _normal_cdf, sigma * 0.3)
        self.assertLess(good["brier"], bad["brier"])

    def test_empty_errors(self):
        result = score_distribution([], _normal_cdf, 3.0)
        self.assertEqual(result["n"], 0)

    def test_returns_required_keys(self):
        result = score_distribution([1.0, -1.0, 0.5], _normal_cdf, 2.0)
        self.assertIn("brier", result)
        self.assertIn("log_loss", result)
        self.assertIn("n", result)
