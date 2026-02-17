"""Tests for multi-choice sum-YES deviation detection."""

import pytest
from weather.strategy import compute_yes_sum_deviation


class TestComputeYesSumDeviation:
    def test_normal_event(self):
        """Event with prices summing to ~1.0."""
        markets = [
            {"best_ask": "0.15", "outcome_name": "60-61"},
            {"best_ask": "0.30", "outcome_name": "62-63"},
            {"best_ask": "0.35", "outcome_name": "64-65"},
            {"best_ask": "0.20", "outcome_name": "66+"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 4
        assert abs(yes_sum - 1.0) < 0.01
        assert abs(deviation) < 0.01

    def test_overpriced_event(self):
        """Sum > 1.0 -- positive deviation (arb: buy all NO)."""
        markets = [
            {"best_ask": "0.20"},
            {"best_ask": "0.35"},
            {"best_ask": "0.30"},
            {"best_ask": "0.20"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 4
        assert yes_sum == pytest.approx(1.05)
        assert deviation == pytest.approx(0.05)

    def test_underpriced_event(self):
        """Sum < 1.0 -- negative deviation (arb: buy all YES)."""
        markets = [
            {"best_ask": "0.10"},
            {"best_ask": "0.30"},
            {"best_ask": "0.25"},
            {"best_ask": "0.25"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 4
        assert yes_sum == pytest.approx(0.90)
        assert deviation == pytest.approx(-0.10)

    def test_insufficient_buckets(self):
        """Less than 3 buckets -- returns zeros."""
        markets = [
            {"best_ask": "0.50"},
            {"best_ask": "0.50"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 0
        assert yes_sum == 0.0
        assert deviation == 0.0

    def test_missing_prices(self):
        """Markets with missing/zero prices are excluded."""
        markets = [
            {"best_ask": "0.30"},
            {"best_ask": "0"},
            {"best_ask": "0.30"},
            {"outcome_name": "no-price"},  # no best_ask
            {"best_ask": "0.30"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 3  # only 3 with valid prices
        assert yes_sum == pytest.approx(0.90)

    def test_external_price_fallback(self):
        """Falls back to external_price_yes when best_ask missing."""
        markets = [
            {"external_price_yes": "0.25"},
            {"external_price_yes": "0.25"},
            {"external_price_yes": "0.25"},
            {"external_price_yes": "0.25"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 4
        assert yes_sum == pytest.approx(1.0)

    def test_invalid_price_string(self):
        """Invalid price strings are treated as 0 and excluded."""
        markets = [
            {"best_ask": "abc"},
            {"best_ask": "0.30"},
            {"best_ask": "0.30"},
            {"best_ask": "0.30"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 3
        assert yes_sum == pytest.approx(0.90)

    def test_empty_markets(self):
        """Empty market list returns zeros."""
        yes_sum, deviation, n = compute_yes_sum_deviation([])
        assert n == 0
        assert yes_sum == 0.0
        assert deviation == 0.0

    def test_single_bucket(self):
        """Single bucket is insufficient (need >= 3)."""
        markets = [{"best_ask": "0.80"}]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 0

    def test_exactly_three_buckets(self):
        """Exactly 3 buckets is the minimum for a valid sum."""
        markets = [
            {"best_ask": "0.40"},
            {"best_ask": "0.35"},
            {"best_ask": "0.25"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 3
        assert yes_sum == pytest.approx(1.0)
        assert deviation == pytest.approx(0.0)

    def test_mixed_best_ask_and_external(self):
        """best_ask takes priority over external_price_yes."""
        markets = [
            {"best_ask": "0.30", "external_price_yes": "0.50"},
            {"best_ask": "0.30", "external_price_yes": "0.50"},
            {"best_ask": "0.30", "external_price_yes": "0.50"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 3
        assert yes_sum == pytest.approx(0.90)  # uses best_ask, not external

    def test_none_best_ask_falls_through(self):
        """None best_ask falls back to external_price_yes."""
        markets = [
            {"best_ask": None, "external_price_yes": "0.30"},
            {"best_ask": None, "external_price_yes": "0.30"},
            {"best_ask": None, "external_price_yes": "0.40"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 3
        assert yes_sum == pytest.approx(1.0)

    def test_large_event_eight_buckets(self):
        """Realistic 8-bucket event with slight overpricing."""
        markets = [
            {"best_ask": "0.05"},
            {"best_ask": "0.08"},
            {"best_ask": "0.12"},
            {"best_ask": "0.20"},
            {"best_ask": "0.25"},
            {"best_ask": "0.15"},
            {"best_ask": "0.10"},
            {"best_ask": "0.08"},
        ]
        yes_sum, deviation, n = compute_yes_sum_deviation(markets)
        assert n == 8
        assert yes_sum == pytest.approx(1.03)
        assert deviation == pytest.approx(0.03)
