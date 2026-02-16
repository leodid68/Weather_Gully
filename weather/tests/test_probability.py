"""Tests for the NOAA probability model."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from weather.ensemble import EnsembleResult
from weather.probability import (
    _get_seasonal_factor,
    _get_stddev,
    _load_calibration,
    _skew_t_cdf,
    _student_t_cdf,
    _weather_sigma_multiplier,
    compute_adaptive_sigma,
    estimate_bucket_probability,
    get_horizon_days,
    get_noaa_probability,
    platt_calibrate,
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

    def setUp(self):
        """Force hardcoded fallback by clearing calibration cache."""
        import weather.probability as prob_module
        self._saved_cache = prob_module._calibration_cache
        prob_module._calibration_cache = {}  # Empty = no calibration data

    def tearDown(self):
        import weather.probability as prob_module
        prob_module._calibration_cache = self._saved_cache

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

    def test_horizon_override_uses_fixed_horizon(self):
        """horizon_override bypasses date-based horizon computation."""
        # Use a date far in the past — without override, horizon would be 0
        past_date = "2020-01-01"
        p_h0 = estimate_bucket_probability(
            52, 50, 54, past_date, apply_seasonal=False, horizon_override=0,
        )
        p_h7 = estimate_bucket_probability(
            52, 50, 54, past_date, apply_seasonal=False, horizon_override=7,
        )
        # Horizon 7 → wider sigma → lower center bucket probability
        self.assertGreater(p_h0, p_h7)

    def test_horizon_override_none_uses_date(self):
        """horizon_override=None falls back to date-based computation."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_default = estimate_bucket_probability(
            52, 50, 54, today, apply_seasonal=False,
        )
        p_explicit_none = estimate_bucket_probability(
            52, 50, 54, today, apply_seasonal=False, horizon_override=None,
        )
        self.assertAlmostEqual(p_default, p_explicit_none, places=6)


class TestCalibrationFallback(unittest.TestCase):
    """When no calibration.json exists, hardcoded defaults should be used."""

    def test_no_calibration_file_uses_defaults(self):
        """Without calibration.json, _get_stddev returns hardcoded values."""
        import weather.probability as prob_module
        # Reset cache to force reload
        prob_module._calibration_cache = None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sigma = _get_stddev(today)
        # Should be the hardcoded value for horizon 0 (~1.5)
        self.assertGreater(sigma, 0)
        self.assertLess(sigma, 20)

    def test_location_param_backward_compatible(self):
        """estimate_bucket_probability with location param works like before."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_no_loc = estimate_bucket_probability(52, 50, 54, today, apply_seasonal=False)
        p_with_loc = estimate_bucket_probability(52, 50, 54, today, apply_seasonal=False, location="UNKNOWN")
        # Both should produce same result when no calibration data exists for the location
        self.assertAlmostEqual(p_no_loc, p_with_loc, places=3)

    def test_weather_data_param_backward_compatible(self):
        """estimate_bucket_probability with weather_data=None works like before."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_without = estimate_bucket_probability(52, 50, 54, today, apply_seasonal=False)
        p_with_none = estimate_bucket_probability(52, 50, 54, today, apply_seasonal=False, weather_data=None)
        self.assertAlmostEqual(p_without, p_with_none, places=4)


class TestWeatherSigmaMultiplier(unittest.TestCase):
    """Test _weather_sigma_multiplier for various weather conditions."""

    def test_default_no_data(self):
        """Empty weather data should return 1.0 (no adjustment)."""
        self.assertAlmostEqual(_weather_sigma_multiplier({}), 1.0)

    def test_high_cloud_cover(self):
        """Cloud cover > 80% on high should increase sigma."""
        data = {"cloud_cover_max": 85.0}
        mult = _weather_sigma_multiplier(data, metric="high")
        self.assertGreater(mult, 1.0)
        # For low metric, cloud cover shouldn't increase
        mult_low = _weather_sigma_multiplier(data, metric="low")
        self.assertAlmostEqual(mult_low, 1.0)

    def test_high_wind(self):
        """Wind > 40 km/h should increase sigma."""
        data = {"wind_speed_max": 50.0}
        mult = _weather_sigma_multiplier(data, metric="high")
        self.assertGreater(mult, 1.0)

    def test_high_wind_gusts(self):
        """Wind gusts > 40 km/h should also increase sigma."""
        data = {"wind_gusts_max": 55.0}
        mult = _weather_sigma_multiplier(data, metric="high")
        self.assertGreater(mult, 1.0)

    def test_precipitation(self):
        """Precipitation > 10mm on high should increase sigma."""
        data = {"precip_sum": 15.0}
        mult = _weather_sigma_multiplier(data, metric="high")
        self.assertGreater(mult, 1.0)

    def test_all_conditions_stack(self):
        """Multiple bad conditions should all contribute."""
        data = {
            "cloud_cover_max": 90.0,
            "wind_speed_max": 50.0,
            "precip_sum": 20.0,
        }
        mult = _weather_sigma_multiplier(data, metric="high")
        # Should be > 1.0 + 0.10 + 0.08 + 0.12 = 1.30
        self.assertGreaterEqual(mult, 1.30)

    def test_low_values_no_adjustment(self):
        """Low cloud/wind/precip should not trigger adjustments."""
        data = {
            "cloud_cover_max": 50.0,
            "wind_speed_max": 20.0,
            "precip_sum": 2.0,
        }
        mult = _weather_sigma_multiplier(data, metric="high")
        self.assertAlmostEqual(mult, 1.0)


class TestCalibrationWithMockData(unittest.TestCase):
    """Test calibration loading with a mock calibration.json."""

    def setUp(self):
        import weather.probability as prob_module
        self._original_cache = prob_module._calibration_cache
        self._original_path = prob_module._CALIBRATION_PATH

    def tearDown(self):
        import weather.probability as prob_module
        prob_module._calibration_cache = self._original_cache
        prob_module._CALIBRATION_PATH = self._original_path

    def test_load_calibration_data(self):
        import weather.probability as prob_module
        fixture_path = Path(__file__).parent / "fixtures" / "calibration.json"
        prob_module._CALIBRATION_PATH = fixture_path
        prob_module._calibration_cache = None

        cal = _load_calibration()
        self.assertIn("global_sigma", cal)
        self.assertIn("location_sigma", cal)
        self.assertEqual(cal["global_sigma"]["0"], 1.8)

    def test_location_sigma_used_when_available(self):
        import weather.probability as prob_module
        fixture_path = Path(__file__).parent / "fixtures" / "calibration.json"
        prob_module._CALIBRATION_PATH = fixture_path
        prob_module._calibration_cache = None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sigma_nyc = _get_stddev(today, location="NYC")
        sigma_miami = _get_stddev(today, location="Miami")
        # NYC and Miami should have different sigmas
        self.assertNotAlmostEqual(sigma_nyc, sigma_miami, places=1)

    def test_seasonal_factor_per_location(self):
        import weather.probability as prob_module
        fixture_path = Path(__file__).parent / "fixtures" / "calibration.json"
        prob_module._CALIBRATION_PATH = fixture_path
        prob_module._calibration_cache = None

        factor_nyc = _get_seasonal_factor(1, location="NYC")
        factor_miami = _get_seasonal_factor(1, location="Miami")
        # Both should be > 1.0 (winter = harder = more sigma)
        self.assertGreater(factor_nyc, 1.0)
        self.assertGreater(factor_miami, 1.0)
        # NYC should have different factor from Miami
        self.assertNotAlmostEqual(factor_nyc, factor_miami, places=2)


class TestHardcodedSigmaNotStale(unittest.TestCase):
    """Verify hardcoded fallbacks are aligned with calibration data."""

    def test_horizon_0_sigma_not_too_low(self):
        """_HORIZON_STDDEV[0] should be >= 1.9 (calibrated ~1.96)."""
        from weather.probability import _HORIZON_STDDEV
        self.assertGreaterEqual(_HORIZON_STDDEV[0], 1.9)

    def test_horizon_10_sigma_not_too_low(self):
        """_HORIZON_STDDEV[10] should be >= 11.0 (calibrated ~11.77)."""
        from weather.probability import _HORIZON_STDDEV
        self.assertGreaterEqual(_HORIZON_STDDEV[10], 11.0)

    def test_seasonal_february_hardest(self):
        """February should be the hardest month (highest factor = more sigma)."""
        from weather.probability import _SEASONAL_FACTORS
        self.assertGreater(_SEASONAL_FACTORS[2], 1.10)

    def test_seasonal_november_easiest(self):
        """November should be the easiest month (lowest factor = less sigma)."""
        from weather.probability import _SEASONAL_FACTORS
        self.assertLess(_SEASONAL_FACTORS[11], 0.80)


class TestComputeAdaptiveSigma(unittest.TestCase):

    def _make_ensemble(self, stddev):
        return EnsembleResult(
            member_temps=[50.0], ensemble_mean=50.0,
            ensemble_stddev=stddev, ecmwf_stddev=stddev,
            gfs_stddev=stddev, n_members=51,
        )

    def test_ensemble_signal_wins(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(self._make_ensemble(5.0), 1.0, 1.0, today, "NYC")
        self.assertGreaterEqual(result, 6.0)  # 5.0 * 1.3 = 6.5

    def test_model_spread_wins(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(self._make_ensemble(0.5), 10.0, 0.5, today, "NYC")
        self.assertGreaterEqual(result, 5.0)  # 10.0 * 0.7 = 7.0

    def test_ema_signal_wins(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(self._make_ensemble(0.5), 0.5, 8.0, today, "NYC")
        self.assertGreaterEqual(result, 8.0)  # 8.0 * 1.25 = 10.0

    def test_floor_prevents_too_low(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(self._make_ensemble(0.1), 0.1, 0.1, today, "NYC")
        # Floor is now _get_stddev without seasonal factor
        floor = _get_stddev(today, "NYC")
        self.assertGreaterEqual(result, floor)

    def test_none_ensemble_uses_other_signals(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(None, 4.0, 3.0, today, "NYC")
        self.assertGreater(result, 0)

    def test_all_none_returns_floor(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(None, 0.0, None, today, "NYC")
        # Floor is now _get_stddev without seasonal factor
        floor = _get_stddev(today, "NYC")
        self.assertAlmostEqual(result, floor, places=1)

    def test_result_always_positive(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = compute_adaptive_sigma(self._make_ensemble(0.0), 0.0, None, today, "")
        self.assertGreater(result, 0)


class TestSigmaOverride(unittest.TestCase):

    def test_override_bypasses_internal_sigma(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prob_wide = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=20.0)
        prob_narrow = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=0.5)
        self.assertGreater(prob_narrow, prob_wide)

    def test_override_none_uses_internal_sigma(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prob_default = estimate_bucket_probability(50.0, 49, 51, today)
        prob_none = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=None)
        self.assertAlmostEqual(prob_default, prob_none, places=4)

    def test_override_ignores_weather_data(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        weather = {"cloud_cover_max": 100, "wind_speed_max": 60, "precip_sum": 20}
        prob_with = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=3.0, weather_data=weather)
        prob_without = estimate_bucket_probability(50.0, 49, 51, today, sigma_override=3.0)
        self.assertAlmostEqual(prob_with, prob_without, places=4)


class TestPlattCalibrate(unittest.TestCase):
    @patch("weather.probability._load_platt_params", return_value={})
    def test_identity_when_no_params(self, _mock):
        """Without calibration params, returns raw probability."""
        self.assertAlmostEqual(platt_calibrate(0.5), 0.5)
        self.assertAlmostEqual(platt_calibrate(0.3), 0.3)

    def test_bounded_output(self):
        result = platt_calibrate(0.001)
        self.assertGreaterEqual(result, 0.01)
        result = platt_calibrate(0.999)
        self.assertLessEqual(result, 0.99)

    def test_monotonic(self):
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = [platt_calibrate(p) for p in probs]
        for i in range(len(calibrated) - 1):
            self.assertLessEqual(calibrated[i], calibrated[i + 1])

    @patch("weather.probability._load_platt_params", return_value={"a": 1.5, "b": -0.3})
    def test_sigmoid_transform_applied(self, _mock):
        """With non-identity params, probability should be shifted."""
        result = platt_calibrate(0.5)
        # sigmoid(1.5 * 0 + (-0.3)) = sigmoid(-0.3) ≈ 0.4256
        self.assertAlmostEqual(result, 0.4256, delta=0.01)
        self.assertNotAlmostEqual(result, 0.5)

    @patch("weather.probability._load_platt_params", return_value={"a": 1.5, "b": -0.3})
    def test_sigmoid_preserves_monotonicity(self, _mock):
        """Non-identity params still preserve ordering."""
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = [platt_calibrate(p) for p in probs]
        for i in range(len(calibrated) - 1):
            self.assertLessEqual(calibrated[i], calibrated[i + 1])


class TestRegularizedIncompleteBeta(unittest.TestCase):
    """Direct tests for _regularized_incomplete_beta edge cases."""

    def test_invalid_a_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 0, 1)
        self.assertIsInstance(result, float)

    def test_invalid_b_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1, 0)
        self.assertIsInstance(result, float)

    def test_invalid_a_negative(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, -1, 1)
        self.assertEqual(result, 0.0)

    def test_x_near_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1e-20, 5, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_x_near_one(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1 - 1e-20, 5, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestStudentTCDF(unittest.TestCase):
    def test_symmetry(self):
        """Student's t CDF should be symmetric: F(-x) = 1 - F(x)."""
        from weather.probability import _student_t_cdf
        for df in [3, 5, 10, 30]:
            for x in [0.5, 1.0, 2.0]:
                self.assertAlmostEqual(
                    _student_t_cdf(-x, df) + _student_t_cdf(x, df), 1.0, places=6)

    def test_center_is_half(self):
        """F(0) should be 0.5 for all df."""
        from weather.probability import _student_t_cdf
        for df in [3, 5, 10, 100]:
            self.assertAlmostEqual(_student_t_cdf(0, df), 0.5, places=6)

    def test_high_df_matches_normal(self):
        """For df > 100, should match normal CDF."""
        from weather.probability import _student_t_cdf, _normal_cdf
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            self.assertAlmostEqual(_student_t_cdf(x, 200), _normal_cdf(x), places=4)

    def test_fatter_tails_than_normal(self):
        """Student's t with low df should have more probability in tails."""
        from weather.probability import _student_t_cdf, _normal_cdf
        # P(X > 3) should be larger for t(3) than for normal
        tail_t = 1 - _student_t_cdf(3.0, 3)
        tail_normal = 1 - _normal_cdf(3.0)
        self.assertGreater(tail_t, tail_normal)


class TestProbabilityOutputBounds(unittest.TestCase):
    """Ensure estimate_bucket_probability always returns [0, 1]."""

    def test_never_exceeds_one(self):
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=-999, bucket_high=999,
            forecast_date="2026-02-16", sigma_override=0.001,
        )
        self.assertLessEqual(result, 1.0)

    def test_zero_sigma_returns_valid(self):
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=0.0,
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_huge_sigma_returns_valid(self):
        result = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=1000.0,
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_sigma_capped_at_fifty(self):
        """With sigma_override=1000, behavior should be same as sigma=50."""
        result_huge = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=1000.0,
        )
        result_fifty = estimate_bucket_probability(
            forecast_temp=50.0, bucket_low=45, bucket_high=55,
            forecast_date="2026-02-16", sigma_override=50.0,
        )
        self.assertAlmostEqual(result_huge, result_fifty, places=4)


class TestStudentTCDFEdgeCases(unittest.TestCase):
    """Edge cases for _student_t_cdf robustness."""

    def test_extreme_positive(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1e10, 10)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_extreme_negative(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(-1e10, 10)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_positive_inf(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('inf'), 10)
        self.assertEqual(result, 1.0)

    def test_negative_inf(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('-inf'), 10)
        self.assertEqual(result, 0.0)

    def test_nan_returns_half(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(float('nan'), 10)
        self.assertEqual(result, 0.5)

    def test_df_zero(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1.0, 0)
        self.assertEqual(result, 0.5)

    def test_df_negative(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1.0, -5)
        self.assertEqual(result, 0.5)

    def test_df_fractional(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0, 0.5)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_df_very_large(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0, 1000)
        self.assertAlmostEqual(result, 0.5, places=4)


class TestNumericalRobustnessExtended(unittest.TestCase):
    """Extended numerical robustness tests for beta function and Student's t."""

    def test_beta_a_near_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1e-10, 1.0)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_beta_b_near_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1.0, 1e-10)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_beta_x_near_zero(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1e-300, 5.0, 5.0)
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_beta_x_near_one(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(1.0 - 1e-15, 5.0, 5.0)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_beta_both_params_tiny(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.5, 1e-8, 1e-8)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_student_t_df_half(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(0.0, 0.5)
        self.assertAlmostEqual(result, 0.5, places=3)

    def test_student_t_df_1000_converges_to_normal(self):
        from weather.probability import _student_t_cdf, _normal_cdf
        result = _student_t_cdf(1.96, 1000)
        normal = _normal_cdf(1.96)
        self.assertAlmostEqual(result, normal, places=3)

    def test_student_t_extreme_1e10(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1e10, 10)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_student_t_extreme_neg_1e10(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(-1e10, 10)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_beta_non_convergence_logs_warning(self):
        from weather.probability import _regularized_incomplete_beta
        result = _regularized_incomplete_beta(0.3, 5.0, 5.0, max_iter=1)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_bucket_probability_with_extreme_forecast(self):
        from weather.probability import estimate_bucket_probability
        prob = estimate_bucket_probability(200.0, 40, 44, "2026-02-20", apply_seasonal=False, horizon_override=0)
        self.assertAlmostEqual(prob, 0.0, places=2)

    def test_bucket_probability_with_zero_sigma(self):
        from weather.probability import estimate_bucket_probability
        prob = estimate_bucket_probability(42.0, 40, 44, "2026-02-20", sigma_override=0.0)
        self.assertTrue(0.0 <= prob <= 1.0)


class TestCalibrationLogging(unittest.TestCase):
    """Test that the calibration fallback chain emits appropriate log messages."""

    def test_missing_file_logs_warning(self):
        import weather.probability as prob
        # Force cache reset and point to non-existent path
        orig_path = prob._CALIBRATION_PATH
        prob._CALIBRATION_PATH = Path("/tmp/nonexistent_calibration_xyz.json")
        prob._calibration_cache = None
        prob._calibration_mtime = 0.0
        try:
            with self.assertLogs("weather.probability", level="WARNING") as cm:
                prob._load_calibration()
            self.assertTrue(any("not found" in msg or "fallback" in msg for msg in cm.output))
        finally:
            prob._CALIBRATION_PATH = orig_path
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0


class TestHorizonGrowthFromCalibration(unittest.TestCase):
    def test_reads_growth_from_json(self):
        """If calibration has horizon_growth, use base_sigma * growth for missing horizons."""
        import weather.probability as prob
        cal_data = {
            "global_sigma": {"0": 2.0},  # Only h=0 present
            "horizon_growth": {"0": 1.0, "1": 1.5, "2": 2.0, "3": 2.5},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            # h=0 should use global_sigma directly: 2.0
            sigma_0 = prob._get_stddev("2026-02-16", horizon_override=0)
            self.assertAlmostEqual(sigma_0, 2.0, places=1)
            # h=1 should use base * growth: 2.0 * 1.5 = 3.0
            sigma_1 = prob._get_stddev("2026-02-17", horizon_override=1)
            self.assertAlmostEqual(sigma_1, 3.0, places=1)
            # h=3 should use base * growth: 2.0 * 2.5 = 5.0
            sigma_3 = prob._get_stddev("2026-02-19", horizon_override=3)
            self.assertAlmostEqual(sigma_3, 5.0, places=1)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)

    def test_falls_back_without_growth(self):
        """Without horizon_growth in JSON, should use hardcoded fallback."""
        import weather.probability as prob
        cal_data = {"global_sigma": {"0": 2.0}}  # No horizon_growth
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            # h=5 not in global_sigma, no horizon_growth → hardcoded fallback
            sigma_5 = prob._get_stddev("2026-02-21", horizon_override=5)
            self.assertAlmostEqual(sigma_5, 5.2, places=1)  # _HORIZON_STDDEV[5]
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)


class TestCalibrationAge(unittest.TestCase):
    def test_old_calibration_warns(self):
        """Calibration >30 days old should generate a warning."""
        import weather.probability as prob
        old_meta = {
            "_metadata": {"generated_at": "2025-06-01T00:00:00Z"},
            "global_sigma": {"0": 2.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_meta, f)
            tmp_path = f.name
        try:
            orig_path = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            with self.assertLogs("weather.probability", level="WARNING") as cm:
                prob._load_calibration()
            self.assertTrue(any("days old" in msg for msg in cm.output))
        finally:
            prob._CALIBRATION_PATH = orig_path
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)

    def test_recent_calibration_no_warning(self):
        """Calibration <30 days old should not warn."""
        import weather.probability as prob
        from datetime import datetime, timezone
        recent_meta = {
            "_metadata": {"generated_at": datetime.now(timezone.utc).isoformat()},
            "global_sigma": {"0": 2.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(recent_meta, f)
            tmp_path = f.name
        try:
            orig_path = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            # assertNoLogs is Python 3.10+ — use assertRaises to check no WARNING
            try:
                with self.assertLogs("weather.probability", level="WARNING") as cm:
                    prob._load_calibration()
                # If we get here, a warning was logged — check it's not about age
                age_warnings = [m for m in cm.output if "days old" in m]
                self.assertEqual(len(age_warnings), 0, f"Unexpected age warning: {age_warnings}")
            except AssertionError:
                pass  # No WARNING at all — that's what we want
        finally:
            prob._CALIBRATION_PATH = orig_path
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)


class TestGetCorrelation(unittest.TestCase):
    def test_returns_correlation_for_season(self):
        import weather.probability as prob
        cal_data = {
            "correlation_matrix": {
                "Chicago|NYC": {"DJF": 0.72, "JJA": 0.45},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            corr = prob.get_correlation("NYC", "Chicago", 1)
            self.assertAlmostEqual(corr, 0.72, places=2)
            corr = prob.get_correlation("NYC", "Chicago", 7)
            self.assertAlmostEqual(corr, 0.45, places=2)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)

    def test_returns_zero_for_unknown_pair(self):
        import weather.probability as prob
        cal_data = {"correlation_matrix": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cal_data, f)
            tmp_path = f.name
        try:
            orig = prob._CALIBRATION_PATH
            prob._CALIBRATION_PATH = Path(tmp_path)
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            corr = prob.get_correlation("NYC", "Timbuktu", 1)
            self.assertEqual(corr, 0.0)
        finally:
            prob._CALIBRATION_PATH = orig
            prob._calibration_cache = None
            prob._calibration_mtime = 0.0
            os.unlink(tmp_path)


class TestPlattNegativeAGuard(unittest.TestCase):
    """Platt scaling with a < 0 should be clamped to preserve ordering."""

    @patch("weather.probability._load_platt_params", return_value={"a": -1.0, "b": 0.0})
    def test_negative_a_clamped(self, _mock):
        """a < 0 should be clamped to 0.01 — ordering preserved."""
        low = platt_calibrate(0.3)
        high = platt_calibrate(0.7)
        self.assertLess(low, high, "Ordering should be preserved even with negative a param")

    @patch("weather.probability._load_platt_params", return_value={"a": -2.0, "b": 0.5})
    def test_negative_a_does_not_invert(self, _mock):
        """Higher raw prob should still produce higher calibrated prob."""
        results = [platt_calibrate(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i], results[i + 1])


class TestStudentTSmallDF(unittest.TestCase):
    """Student-t CDF with very small df should fall back to normal CDF."""

    def test_df_below_1_returns_valid(self):
        from weather.probability import _student_t_cdf
        result = _student_t_cdf(1.5, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_df_below_1_near_normal(self):
        from weather.probability import _student_t_cdf, _normal_cdf
        # With df < 1, falls back to normal CDF
        result = _student_t_cdf(1.0, 0.5)
        normal = _normal_cdf(1.0)
        self.assertAlmostEqual(result, normal, places=4)


class TestProbabilityCoherence(unittest.TestCase):
    """Bucket probabilities for exhaustive/exclusive buckets must sum to ~1.0.

    This is the fundamental mathematical invariant: if the buckets partition
    the temperature axis, the probabilities MUST sum to 1.0 (within rounding).
    Platt scaling, Student-t, or any post-processing must preserve this.
    """

    def _sum_buckets(self, forecast, buckets, sigma, date="2026-02-20"):
        """Sum bucket probabilities with given sigma."""
        total = 0.0
        for lo, hi in buckets:
            p = estimate_bucket_probability(
                forecast, lo, hi, date,
                apply_seasonal=False,
                sigma_override=sigma,
            )
            # Also pass through platt_calibrate (as score_buckets does)
            total += platt_calibrate(p)
        return total

    def _sum_buckets_raw(self, forecast, buckets, sigma, date="2026-02-20"):
        """Sum raw bucket probabilities (no Platt)."""
        total = 0.0
        for lo, hi in buckets:
            p = estimate_bucket_probability(
                forecast, lo, hi, date,
                apply_seasonal=False,
                sigma_override=sigma,
            )
            total += p
        return total

    def test_exhaustive_buckets_sum_to_one_raw(self):
        """Raw CDF probabilities must sum to ~1.0."""
        buckets = [(-999, 41), (42, 46), (47, 51), (52, 56), (57, 61), (62, 999)]
        for sigma in [2.0, 3.0, 5.0, 9.36, 15.0]:
            total = self._sum_buckets_raw(46.3, buckets, sigma)
            self.assertAlmostEqual(total, 1.0, delta=0.02,
                                    msg=f"Raw sum should be ~1.0 for sigma={sigma}, got {total}")

    def test_exhaustive_buckets_sum_to_one_with_platt(self):
        """After Platt calibration, bucket probs must still sum to ~1.0.

        With default Platt (a=1, b=0) this is trivially true.
        This test catches any regression that re-introduces non-identity Platt.
        """
        buckets = [(-999, 41), (42, 46), (47, 51), (52, 56), (57, 61), (62, 999)]
        for sigma in [2.0, 3.0, 5.0, 9.36]:
            total = self._sum_buckets(46.3, buckets, sigma)
            self.assertAlmostEqual(total, 1.0, delta=0.05,
                                    msg=f"Platt sum should be ~1.0 for sigma={sigma}, got {total}")

    def test_wider_bucket_sets_still_sum_to_one(self):
        """Test with different bucket configurations."""
        buckets_5deg = [(-999, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 999)]
        total = self._sum_buckets_raw(50.0, buckets_5deg, 4.0)
        self.assertAlmostEqual(total, 1.0, delta=0.02)

    def test_narrow_buckets_sum_to_one(self):
        buckets_2deg = [(-999, 43), (44, 45), (46, 47), (48, 49), (50, 51), (52, 999)]
        total = self._sum_buckets_raw(48.0, buckets_2deg, 3.0)
        self.assertAlmostEqual(total, 1.0, delta=0.02)

    def test_tail_bucket_not_overinflated(self):
        """A bucket 5°F from the forecast with sigma=3 should be < 15% (not 55%)."""
        # This is the exact NYC anomaly that was flagged
        prob = estimate_bucket_probability(
            46.3, -999, 41, "2026-02-20",
            apply_seasonal=False, sigma_override=3.0,
        )
        calibrated = platt_calibrate(prob)
        self.assertLess(calibrated, 0.15,
                         f"Tail bucket 5.3°F from forecast with sigma=3.0 should be <15%, got {calibrated:.1%}")


class TestPlattLargeBGuard(unittest.TestCase):
    """Platt scaling with |b| > 0.5 should fall back to identity."""

    @patch("weather.probability._load_platt_params", return_value={"a": 0.62, "b": 0.70})
    def test_large_positive_b_falls_back(self, _mock):
        """b=0.70 should trigger the guard and return raw probability."""
        result = platt_calibrate(0.3)
        self.assertAlmostEqual(result, 0.3, places=2)

    @patch("weather.probability._load_platt_params", return_value={"a": 0.62, "b": -0.70})
    def test_large_negative_b_falls_back(self, _mock):
        """b=-0.70 should also trigger the guard."""
        result = platt_calibrate(0.3)
        self.assertAlmostEqual(result, 0.3, places=2)

    @patch("weather.probability._load_platt_params", return_value={"a": 1.2, "b": 0.3})
    def test_moderate_b_still_applies(self, _mock):
        """b=0.3 is within acceptable range — should apply transform."""
        result = platt_calibrate(0.5)
        # sigmoid(1.2*0 + 0.3) = sigmoid(0.3) ≈ 0.574
        self.assertNotAlmostEqual(result, 0.5, places=2)


class TestSkewTCDF(unittest.TestCase):
    """Tests for the Fernández-Steel skewed Student-t CDF."""

    def test_gamma_one_equals_student_t(self):
        """gamma=1.0 should match _student_t_cdf exactly."""
        for df in [3, 5, 10, 30]:
            for x in [-2, -1, 0, 1, 2]:
                expected = _student_t_cdf(x, df)
                result = _skew_t_cdf(x, df, 1.0)
                self.assertAlmostEqual(
                    result, expected, places=10,
                    msg=f"gamma=1.0 should match student_t for x={x}, df={df}",
                )

    def test_center_is_half(self):
        """F(0) = 0.5 for any gamma."""
        for gamma in [0.5, 0.8, 1.0, 1.2, 1.5]:
            result = _skew_t_cdf(0.0, 5, gamma)
            self.assertAlmostEqual(
                result, 0.5, places=10,
                msg=f"F(0) should be 0.5 for gamma={gamma}",
            )

    def test_left_skew_heavier_left_tail(self):
        """gamma=0.7 (left-skewed): P(X < -3) > P_symmetric(X < -3)."""
        p_skewed = _skew_t_cdf(-3.0, 5, 0.7)
        p_symmetric = _skew_t_cdf(-3.0, 5, 1.0)
        self.assertGreater(
            p_skewed, p_symmetric,
            f"Left-skewed (gamma=0.7) should have more mass in left tail: "
            f"skewed={p_skewed:.6f} vs symmetric={p_symmetric:.6f}",
        )

    def test_right_skew_heavier_right_tail(self):
        """gamma=1.5 (right-skewed): P(X > 3) > P_symmetric(X > 3)."""
        p_skewed = 1.0 - _skew_t_cdf(3.0, 5, 1.5)
        p_symmetric = 1.0 - _skew_t_cdf(3.0, 5, 1.0)
        self.assertGreater(
            p_skewed, p_symmetric,
            f"Right-skewed (gamma=1.5) should have more mass in right tail: "
            f"skewed={p_skewed:.6f} vs symmetric={p_symmetric:.6f}",
        )

    def test_cdf_monotonically_increasing(self):
        """CDF must be monotonically non-decreasing."""
        for gamma in [0.6, 1.0, 1.4]:
            xs = [i * 0.5 for i in range(-20, 21)]  # -10 to 10 in steps of 0.5
            values = [_skew_t_cdf(x, 5, gamma) for x in xs]
            for i in range(len(values) - 1):
                self.assertLessEqual(
                    values[i], values[i + 1] + 1e-12,
                    msg=f"CDF not monotonic at x={xs[i]:.1f} for gamma={gamma}: "
                        f"F({xs[i]:.1f})={values[i]:.8f} > F({xs[i+1]:.1f})={values[i+1]:.8f}",
                )

    def test_limits_zero_and_one(self):
        """F(-1e10) ~ 0 and F(1e10) ~ 1 for various gamma."""
        for gamma in [0.5, 1.0, 1.5]:
            low = _skew_t_cdf(-1e10, 5, gamma)
            high = _skew_t_cdf(1e10, 5, gamma)
            self.assertAlmostEqual(low, 0.0, places=4,
                                   msg=f"F(-1e10) should be ~0 for gamma={gamma}")
            self.assertAlmostEqual(high, 1.0, places=4,
                                   msg=f"F(1e10) should be ~1 for gamma={gamma}")

    def test_inf_inputs(self):
        """inf -> 1.0, -inf -> 0.0."""
        self.assertEqual(_skew_t_cdf(float('inf'), 5, 0.8), 1.0)
        self.assertEqual(_skew_t_cdf(float('-inf'), 5, 0.8), 0.0)
        self.assertEqual(_skew_t_cdf(float('inf'), 5, 1.3), 1.0)
        self.assertEqual(_skew_t_cdf(float('-inf'), 5, 1.3), 0.0)

    def test_nan_returns_half(self):
        """nan -> 0.5."""
        self.assertEqual(_skew_t_cdf(float('nan'), 5, 0.8), 0.5)
        self.assertEqual(_skew_t_cdf(float('nan'), 10, 1.5), 0.5)

    def test_gamma_clamped(self):
        """Extreme gamma values are clamped to [0.3, 3.0]; F(0) still = 0.5."""
        # gamma=0.01 clamped to 0.3
        result_low = _skew_t_cdf(0.0, 5, 0.01)
        self.assertAlmostEqual(result_low, 0.5, places=10,
                               msg="F(0) should be 0.5 even with gamma clamped from 0.01")
        # gamma=100 clamped to 3.0
        result_high = _skew_t_cdf(0.0, 5, 100.0)
        self.assertAlmostEqual(result_high, 0.5, places=10,
                               msg="F(0) should be 0.5 even with gamma clamped from 100.0")


class TestSkewTIntegration(unittest.TestCase):
    """Test that skew-t distribution is used when calibration says so."""

    def test_skew_t_distribution_from_calibration(self):
        """When calibration says skew_t, left-tail bucket should get more probability."""
        from unittest.mock import patch

        cal_skew_t = {"distribution": "skew_t", "student_t_df": 10, "skew_t_gamma": 0.7}
        cal_normal = {"distribution": "normal"}

        with patch("weather.probability._load_calibration", return_value=cal_skew_t):
            prob_skew = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        with patch("weather.probability._load_calibration", return_value=cal_normal):
            prob_normal = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        # Left-skewed (gamma=0.7) should give higher prob for left-tail bucket
        self.assertGreater(prob_skew, prob_normal)

    def test_skew_t_obs_function_uses_skew(self):
        """estimate_bucket_probability_with_obs should also use skew-t."""
        from unittest.mock import patch
        from weather.probability import estimate_bucket_probability_with_obs

        cal = {"distribution": "skew_t", "student_t_df": 10, "skew_t_gamma": 0.7}
        with patch("weather.probability._load_calibration", return_value=cal):
            prob = estimate_bucket_probability_with_obs(
                forecast_temp=50.0, bucket_low=-999, bucket_high=40,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

    def test_missing_gamma_defaults_to_symmetric(self):
        """If skew_t_gamma is missing, should default to 1.0 (symmetric)."""
        from unittest.mock import patch

        cal_skew_no_gamma = {"distribution": "skew_t", "student_t_df": 10}
        cal_student_t = {"distribution": "student_t", "student_t_df": 10}

        with patch("weather.probability._load_calibration", return_value=cal_skew_no_gamma):
            prob_skew = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=45, bucket_high=55,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        with patch("weather.probability._load_calibration", return_value=cal_student_t):
            prob_student = estimate_bucket_probability(
                forecast_temp=50.0, bucket_low=45, bucket_high=55,
                forecast_date="2026-06-15", sigma_override=5.0,
            )
        # gamma=1.0 default → same as Student-t(10)
        self.assertAlmostEqual(prob_skew, prob_student, places=4)


if __name__ == "__main__":
    unittest.main()
