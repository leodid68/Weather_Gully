"""Tests for bridge execution helpers (VWAP, depth)."""

from weather.bridge import compute_available_depth, compute_vwap


class TestComputeAvailableDepth:
    def test_basic_depth(self):
        book = [
            {"price": "0.50", "size": "100"},
            {"price": "0.55", "size": "200"},
        ]
        expected = 100 * 0.5 + 200 * 0.55
        assert abs(compute_available_depth(book, max_levels=5) - expected) < 0.01

    def test_max_levels_limit(self):
        book = [
            {"price": "0.50", "size": "100"},
            {"price": "0.55", "size": "200"},
            {"price": "0.60", "size": "300"},
        ]
        depth = compute_available_depth(book, max_levels=2)
        expected = 100 * 0.5 + 200 * 0.55
        assert abs(depth - expected) < 0.01

    def test_empty_book(self):
        assert compute_available_depth([], max_levels=5) == 0.0

    def test_malformed_level_skipped(self):
        book = [{"price": "0.50"}, {"price": "0.60", "size": "100"}]
        assert abs(compute_available_depth(book, max_levels=5) - 60.0) < 0.01


class TestComputeVwap:
    def test_single_level_fill(self):
        book = [{"price": "0.50", "size": "100"}]
        vwap = compute_vwap(book, 10.0)
        assert abs(vwap - 0.50) < 1e-6

    def test_multi_level_fill(self):
        book = [
            {"price": "0.50", "size": "20"},   # 10 USD available
            {"price": "0.60", "size": "100"},   # 60 USD available
        ]
        vwap = compute_vwap(book, 20.0)
        # Level 0: 10 USD → 20 shares. Level 1: 10 USD → 16.67 shares
        # VWAP = 20 / (20 + 16.667) = 0.5455
        expected = 20.0 / (20 + 10.0 / 0.60)
        assert abs(vwap - expected) < 0.01

    def test_empty_book(self):
        assert compute_vwap([], 10.0) == 0.0

    def test_partial_fill_returns_vwap(self):
        book = [{"price": "0.50", "size": "10"}]  # Only 5 USD available
        vwap = compute_vwap(book, 100.0)
        assert abs(vwap - 0.50) < 1e-6  # Still returns 0.50

    def test_zero_price_skipped(self):
        book = [{"price": "0.00", "size": "100"}, {"price": "0.50", "size": "100"}]
        vwap = compute_vwap(book, 10.0)
        assert abs(vwap - 0.50) < 1e-6

    def test_vwap_larger_order_worse_price(self):
        asks = [
            {"price": "0.35", "size": "50"},
            {"price": "0.38", "size": "30"},
            {"price": "0.42", "size": "100"},
        ]
        vwap_small = compute_vwap(asks, 5.0)
        vwap_large = compute_vwap(asks, 30.0)
        assert vwap_large >= vwap_small  # Larger order gets worse avg price
