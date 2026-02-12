"""Tests for the NOAA probability model."""

import unittest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from weather.probability import (
    estimate_bucket_probability,
    get_horizon_days,
    get_noaa_probability,
)


class TestGetHorizonDays(unittest.TestCase):

    def test_today(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(get_horizon_days(today), 0)

    def test_tomorrow(self):
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertIn(get_horizon_days(tomorrow), (0, 1, 2))  # rounding

    def test_week_ahead(self):
        future = (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d")
        days = get_horizon_days(future)
        self.assertGreaterEqual(days, 6)
        self.assertLessEqual(days, 8)

    def test_invalid_date_returns_large_value(self):
        """Invalid date should return large horizon to be filtered out."""
        self.assertGreater(get_horizon_days("not-a-date"), 100)


class TestGetNoaaProbability(unittest.TestCase):

    def test_decreases_with_horizon(self):
        """Probability should decrease as forecast horizon increases."""
        now = datetime.now(timezone.utc)
        probs = []
        for days in [0, 1, 3, 5, 7, 10]:
            date_str = (now + timedelta(days=days)).strftime("%Y-%m-%d")
            p = get_noaa_probability(date_str, apply_seasonal=False)
            probs.append(p)

        for i in range(len(probs) - 1):
            self.assertGreaterEqual(probs[i], probs[i + 1],
                                     f"prob should decrease: {probs}")

    def test_never_exceeds_one(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = get_noaa_probability(today, apply_seasonal=False)
        self.assertLessEqual(p, 1.0)

    def test_never_below_zero(self):
        far_future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        p = get_noaa_probability(far_future, apply_seasonal=False)
        self.assertGreaterEqual(p, 0.0)

    def test_seasonal_reduces_in_winter(self):
        """Winter months should reduce probability."""
        date_str = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        p_noseasonal = get_noaa_probability(date_str, apply_seasonal=False)

        # Mock winter month
        with patch("weather.probability.datetime") as mock_dt:
            mock_now = datetime(2025, 1, 15, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.strptime = datetime.strptime
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            p_winter = get_noaa_probability(date_str, apply_seasonal=True)

        # Winter probability should be <= non-seasonal
        self.assertLessEqual(p_winter, p_noseasonal)


class TestEstimateBucketProbability(unittest.TestCase):

    def test_forecast_in_bucket_high_probability(self):
        """Forecast inside bucket → high probability."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = estimate_bucket_probability(52, 50, 54, today, apply_seasonal=False)
        self.assertGreater(p, 0.5)

    def test_forecast_far_from_bucket_low_probability(self):
        """Forecast far from bucket → low probability."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = estimate_bucket_probability(70, 50, 54, today, apply_seasonal=False)
        self.assertLess(p, 0.1)

    def test_open_below_bucket(self):
        """Open bucket (-999, X) for very low forecast."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = estimate_bucket_probability(30, -999, 35, today, apply_seasonal=False)
        self.assertGreater(p, 0.5)

    def test_open_above_bucket(self):
        """Open bucket (X, 999) for very high forecast."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = estimate_bucket_probability(75, 70, 999, today, apply_seasonal=False)
        self.assertGreater(p, 0.5)

    def test_wider_bucket_higher_probability(self):
        """Wider bucket (same center) → higher probability."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_narrow = estimate_bucket_probability(52, 51, 53, today, apply_seasonal=False)
        p_wide = estimate_bucket_probability(52, 48, 56, today, apply_seasonal=False)
        self.assertGreater(p_wide, p_narrow)

    def test_probabilities_sum_near_one(self):
        """All buckets in a market should sum approximately to 1."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        buckets = [(-999, 44), (45, 49), (50, 54), (55, 59), (60, 999)]
        total = sum(
            estimate_bucket_probability(52, lo, hi, today, apply_seasonal=False)
            for lo, hi in buckets
        )
        self.assertAlmostEqual(total, 1.0, delta=0.05)

    def test_longer_horizon_wider_distribution(self):
        """Longer horizon → more spread → center bucket gets less probability."""
        now = datetime.now(timezone.utc)
        near = now.strftime("%Y-%m-%d")
        far = (now + timedelta(days=10)).strftime("%Y-%m-%d")

        p_near = estimate_bucket_probability(52, 50, 54, near, apply_seasonal=False)
        p_far = estimate_bucket_probability(52, 50, 54, far, apply_seasonal=False)
        self.assertGreater(p_near, p_far)


if __name__ == "__main__":
    unittest.main()
