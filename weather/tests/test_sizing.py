"""Tests for Kelly criterion sizing and dynamic exit thresholds."""

import unittest

from weather.sizing import compute_exit_threshold, compute_position_size, kelly_fraction


class TestKellyFraction(unittest.TestCase):

    def test_no_edge_returns_zero(self):
        """When p equals the implied probability, no bet."""
        # price=0.50 → b=1.0, p=0.50 → full_kelly = (0.5*1 - 0.5)/1 = 0
        self.assertEqual(kelly_fraction(0.50, 1.0), 0.0)

    def test_negative_edge_returns_zero(self):
        """When probability < implied, definitely no bet."""
        self.assertEqual(kelly_fraction(0.30, 1.0), 0.0)

    def test_strong_edge(self):
        """High probability, low price → significant fraction."""
        # p=0.85, b=(1/0.10)-1=9 → full=(0.85*9-0.15)/9=0.833
        f = kelly_fraction(0.85, 9.0, fraction=0.25)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1.0)

    def test_quarter_kelly_is_conservative(self):
        """Quarter Kelly < Full Kelly."""
        full = kelly_fraction(0.80, 4.0, fraction=1.0)
        quarter = kelly_fraction(0.80, 4.0, fraction=0.25)
        self.assertAlmostEqual(quarter, full * 0.25, places=6)

    def test_zero_odds_returns_zero(self):
        self.assertEqual(kelly_fraction(0.80, 0.0), 0.0)

    def test_zero_probability_returns_zero(self):
        self.assertEqual(kelly_fraction(0.0, 2.0), 0.0)

    def test_probability_one_returns_fraction(self):
        # Edge case: certain win
        f = kelly_fraction(1.0, 2.0)
        # p=1 is clamped by the p >= 1 check
        self.assertEqual(f, 0.0)

    def test_half_kelly(self):
        f_half = kelly_fraction(0.70, 3.0, fraction=0.50)
        f_full = kelly_fraction(0.70, 3.0, fraction=1.0)
        self.assertAlmostEqual(f_half, f_full * 0.50, places=6)


class TestComputePositionSize(unittest.TestCase):

    def test_basic_sizing(self):
        size = compute_position_size(
            probability=0.85, price=0.10, balance=100.0,
            max_position_usd=5.0, kelly_frac=0.25,
        )
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 5.0)

    def test_no_edge_returns_zero(self):
        size = compute_position_size(
            probability=0.10, price=0.10, balance=100.0,
            max_position_usd=5.0,
        )
        self.assertEqual(size, 0.0)

    def test_capped_at_max_position(self):
        size = compute_position_size(
            probability=0.95, price=0.05, balance=10000.0,
            max_position_usd=2.0,
        )
        self.assertLessEqual(size, 2.0)

    def test_capped_at_balance(self):
        size = compute_position_size(
            probability=0.95, price=0.05, balance=1.50,
            max_position_usd=100.0,
        )
        self.assertLessEqual(size, 1.50)

    def test_zero_price_returns_zero(self):
        self.assertEqual(
            compute_position_size(0.80, 0.0, 100.0, 5.0), 0.0
        )

    def test_price_one_returns_zero(self):
        self.assertEqual(
            compute_position_size(0.80, 1.0, 100.0, 5.0), 0.0
        )

    def test_tiny_edge_below_min_trade_returns_zero(self):
        """When Kelly says bet $0.10 but min_trade is $1.00, return 0 (don't override Kelly)."""
        size = compute_position_size(
            probability=0.52, price=0.50, balance=10.0,
            max_position_usd=5.0, kelly_frac=0.25, min_trade=1.0,
        )
        self.assertEqual(size, 0.0)


class TestComputeExitThreshold(unittest.TestCase):

    def test_far_from_resolution(self):
        """Far away → full target."""
        t = compute_exit_threshold(0.10, hours_to_resolution=200)
        # base_target = min(0.20, 0.80) = 0.20, time_factor=1.0
        # threshold = 0.10 + (0.20 - 0.10) * 1.0 = 0.20
        self.assertAlmostEqual(t, 0.20, places=2)

    def test_close_to_resolution_lower_target(self):
        """Close to resolution → reduced target."""
        t_far = compute_exit_threshold(0.10, hours_to_resolution=200)
        t_close = compute_exit_threshold(0.10, hours_to_resolution=3)
        self.assertLess(t_close, t_far)

    def test_minimum_profit_floor(self):
        """Always at least 5 cents above cost basis."""
        t = compute_exit_threshold(0.90, hours_to_resolution=1)
        self.assertGreaterEqual(t, 0.95)

    def test_high_cost_basis_cap(self):
        """High cost basis: target capped at 0.80."""
        t = compute_exit_threshold(0.50, hours_to_resolution=200)
        # base_target = min(1.00, 0.80) = 0.80
        self.assertLessEqual(t, 0.80)

    def test_very_low_cost_basis(self):
        t = compute_exit_threshold(0.05, hours_to_resolution=100)
        self.assertGreater(t, 0.05)


if __name__ == "__main__":
    unittest.main()
