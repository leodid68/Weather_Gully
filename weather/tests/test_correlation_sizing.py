"""Tests for correlation-based sizing discount in strategy."""

import unittest
from unittest.mock import patch

from weather.strategy import _apply_correlation_discount
from weather.config import Config


class TestCorrelationDiscount(unittest.TestCase):

    def test_no_open_positions_no_discount(self):
        adjusted = _apply_correlation_discount(
            base_size=2.0, location="NYC", month=1,
            open_locations=[], config=Config(correlation_threshold=0.5, correlation_discount=0.5),
        )
        self.assertEqual(adjusted, 2.0)

    @patch("weather.strategy.get_correlation", return_value=0.8)
    def test_correlated_position_reduces_sizing(self, mock_corr):
        adjusted = _apply_correlation_discount(
            base_size=2.0, location="NYC", month=1,
            open_locations=["Chicago"],
            config=Config(correlation_threshold=0.5, correlation_discount=0.5),
        )
        self.assertAlmostEqual(adjusted, 1.2, places=2)

    @patch("weather.strategy.get_correlation", return_value=0.3)
    def test_below_threshold_no_discount(self, mock_corr):
        adjusted = _apply_correlation_discount(
            base_size=2.0, location="NYC", month=1,
            open_locations=["Miami"],
            config=Config(correlation_threshold=0.5, correlation_discount=0.5),
        )
        self.assertEqual(adjusted, 2.0)

    def test_multiple_correlated_cumulative(self):
        def mock_corr(l1, l2, m):
            pair = tuple(sorted([l1, l2]))
            if pair == ("Chicago", "NYC"):
                return 0.8
            if pair == ("Dallas", "NYC"):
                return 0.6
            return 0.0

        with patch("weather.strategy.get_correlation", side_effect=mock_corr):
            adjusted = _apply_correlation_discount(
                base_size=2.0, location="NYC", month=1,
                open_locations=["Chicago", "Dallas"],
                config=Config(correlation_threshold=0.5, correlation_discount=0.5),
            )
        # Cumulative: total_corr = 0.8 + 0.6 = 1.4, factor = max(0.1, 1 - 1.4*0.5) = 0.3
        self.assertAlmostEqual(adjusted, 0.6, places=2)

    @patch("weather.strategy.get_correlation", return_value=0.95)
    def test_floor_at_ten_percent(self, mock_corr):
        adjusted = _apply_correlation_discount(
            base_size=2.0, location="NYC", month=1,
            open_locations=["Chicago"],
            config=Config(correlation_threshold=0.5, correlation_discount=1.0),
        )
        # 1 - 0.95 * 1.0 = 0.05, but floor at 0.1 -> 2.0 * 0.1 = 0.2
        self.assertAlmostEqual(adjusted, 0.2, places=2)
