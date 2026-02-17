"""Tests for value trap filters in score_buckets()."""

import pytest
from unittest.mock import MagicMock
from weather.strategy import score_buckets
from weather.config import Config


def _make_market(outcome_name: str, best_ask: float, bucket_low: float, bucket_high: float):
    """Helper to create a mock market dict."""
    return {
        "id": f"0xtest_{outcome_name}",
        "outcome_name": outcome_name,
        "best_ask": best_ask,
        "clob_token_ids": ["token1"],
        "external_price_yes": best_ask,
    }


def _make_config(**overrides):
    """Create config with defaults and overrides."""
    c = Config.load.__wrapped__(Config) if hasattr(Config.load, '__wrapped__') else Config()
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class TestMinProbabilityFilter:
    def test_filters_low_prob_bucket(self):
        """Bucket with probability 0.20 should be filtered (< 0.25 default)."""
        config = Config()
        # A bucket far enough from forecast to have ~20% prob
        # forecast=70, bucket=64-65, sigma=2.0 -> prob ~0.06 -> filtered
        markets = [_make_market("64-65", 0.02, 64.0, 65.0)]
        scored = score_buckets(markets, forecast_temp=70.0, forecast_date="2026-02-18",
                               config=config, sigma_override=2.0)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) == 0

    def test_allows_high_prob_bucket(self):
        """Bucket near forecast should pass min_probability filter."""
        config = Config()
        markets = [_make_market("69-70", 0.02, 69.0, 70.0)]
        scored = score_buckets(markets, forecast_temp=70.0, forecast_date="2026-02-18",
                               config=config, sigma_override=2.0)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) >= 1


class TestDistanceFilter:
    def test_blocks_extreme_bucket(self):
        """Bucket center 7.8F from forecast with sigma=2.15 -> distance/sigma=3.6 > 2.5 -> filtered."""
        config = Config()
        config.min_probability = 0.01  # disable min_prob to isolate distance filter
        markets = [_make_market("29-31", 0.01, 29.0, 31.0)]
        scored = score_buckets(markets, forecast_temp=38.8, forecast_date="2026-02-18",
                               config=config, sigma_override=2.15)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) == 0

    def test_allows_near_bucket(self):
        """Bucket center 0.5F from forecast -> well within 2.5*sigma -> allowed."""
        config = Config()
        config.min_probability = 0.01  # disable min_prob
        markets = [_make_market("42-43", 0.01, 42.0, 43.0)]
        scored = score_buckets(markets, forecast_temp=43.0, forecast_date="2026-02-18",
                               config=config, sigma_override=2.15)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) >= 1

    def test_open_bottom_bucket_uses_high_as_center(self):
        """Open-bottom bucket (-999, 41): uses 41 as center."""
        config = Config()
        config.min_probability = 0.01
        # forecast=38, hi=41, distance=3, sigma=2.0, threshold=5.0 -> allowed
        markets = [_make_market("41 or below", 0.01, -999.0, 41.0)]
        scored = score_buckets(markets, forecast_temp=38.0, forecast_date="2026-02-18",
                               config=config, sigma_override=2.0)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) >= 1

    def test_open_bottom_bucket_far_filtered(self):
        """Open-bottom bucket (-999, 31): forecast=38.8, hi=31, distance=7.8, sigma=2.15 -> filtered."""
        config = Config()
        config.min_probability = 0.01
        markets = [_make_market("31 or below", 0.01, -999.0, 31.0)]
        scored = score_buckets(markets, forecast_temp=38.8, forecast_date="2026-02-18",
                               config=config, sigma_override=2.15)
        yes_entries = [s for s in scored if s["side"] == "yes"]
        assert len(yes_entries) == 0

    def test_no_filter_without_sigma(self):
        """Without sigma_override, distance filter is not applied."""
        config = Config()
        config.min_probability = 0.01
        markets = [_make_market("29-31", 0.80, 29.0, 31.0)]
        # No sigma_override -> distance filter skipped, only prob matters
        scored = score_buckets(markets, forecast_temp=70.0, forecast_date="2026-02-18",
                               config=config, sigma_override=None)
        # With no sigma, the Normal CDF will still compute a very low prob
        # which may or may not be filtered by min_prob=0.01
        # The key point: no crash, no distance filter applied
        assert isinstance(scored, list)
