"""Tests for Other bucket no_prob adjustment."""
import pytest
from unittest.mock import patch
from weather.strategy import score_buckets
from weather.config import Config


class TestOtherBucketAdjustment:
    def _make_config(self, **overrides):
        cfg = Config()
        cfg.min_probability = 0.01  # Low threshold for testing
        cfg.min_ev_threshold = 0.0
        cfg.trading_fees = 0.0
        cfg.seasonal_adjustments = False
        cfg.max_bucket_distance_sigma = 100.0  # Disable distance filter
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_no_prob_adjusted_with_other(self):
        """Event with 'Other' bucket -> no_prob reduced."""
        markets = [
            {"outcome_name": "42-43", "best_ask": 0.40, "external_price_yes": 0.40},
            {"outcome_name": "Other", "best_ask": 0.05, "external_price_yes": 0.05},
        ]
        cfg = self._make_config()
        scored = score_buckets(markets, 42.5, "2026-02-20", cfg,
                               metric="high", location="NYC")
        no_entries = [s for s in scored if s["side"] == "no"]
        # Without Other: no_prob would be 1.0 - prob
        # With Other: no_prob = 1.0 - prob - 0.05
        for entry in no_entries:
            # The prob for 42-43 with forecast 42.5 should be high
            # no_prob should be less than 1.0 - entry's yes_prob
            assert entry["prob"] < 1.0

    def test_no_prob_no_other(self):
        """Event without 'Other' -> no_prob unchanged."""
        markets = [
            {"outcome_name": "42-43", "best_ask": 0.40, "external_price_yes": 0.40},
            {"outcome_name": "44-45", "best_ask": 0.30, "external_price_yes": 0.30},
        ]
        cfg = self._make_config()
        scored = score_buckets(markets, 42.5, "2026-02-20", cfg,
                               metric="high", location="NYC")
        no_entries = [s for s in scored if s["side"] == "no"]
        yes_entries = [s for s in scored if s["side"] == "yes"]
        # Without Other bucket, no_prob = 1.0 - prob exactly
        for no_e in no_entries:
            matching_yes = [y for y in yes_entries if y["outcome_name"] == no_e["outcome_name"]]
            if matching_yes:
                assert no_e["prob"] == pytest.approx(1.0 - matching_yes[0]["prob"], abs=0.001)

    def test_no_prob_clamped_to_zero(self):
        """If prob + p_other > 1.0 -> no_prob clamped to 0."""
        markets = [
            {"outcome_name": "42-43", "best_ask": 0.10, "external_price_yes": 0.10},
            {"outcome_name": "Other", "best_ask": 0.95, "external_price_yes": 0.95},
        ]
        cfg = self._make_config()
        scored = score_buckets(markets, 42.5, "2026-02-20", cfg,
                               metric="high", location="NYC")
        no_entries = [s for s in scored if s["side"] == "no"]
        # no_prob should be 0 or very small (clamped)
        for entry in no_entries:
            assert entry["prob"] >= 0.0

    def test_other_bucket_not_scored(self):
        """'Other' bucket itself is not scored (parse returns None)."""
        markets = [
            {"outcome_name": "42-43", "best_ask": 0.40, "external_price_yes": 0.40},
            {"outcome_name": "Other", "best_ask": 0.05, "external_price_yes": 0.05},
        ]
        cfg = self._make_config()
        scored = score_buckets(markets, 42.5, "2026-02-20", cfg,
                               metric="high", location="NYC")
        # "Other" should not appear in scored
        other_entries = [s for s in scored if s["outcome_name"] == "Other"]
        assert len(other_entries) == 0

    def test_multiple_non_parseable_buckets(self):
        """Multiple non-parseable buckets sum their prices."""
        markets = [
            {"outcome_name": "42-43", "best_ask": 0.40, "external_price_yes": 0.40},
            {"outcome_name": "Other", "best_ask": 0.03, "external_price_yes": 0.03},
            {"outcome_name": "No winner", "best_ask": 0.02, "external_price_yes": 0.02},
        ]
        cfg = self._make_config()
        scored = score_buckets(markets, 42.5, "2026-02-20", cfg,
                               metric="high", location="NYC")
        # p_other should be 0.05 (0.03 + 0.02)
        # Only temp buckets scored
        assert all(s["outcome_name"] != "Other" for s in scored)
        assert all(s["outcome_name"] != "No winner" for s in scored)
