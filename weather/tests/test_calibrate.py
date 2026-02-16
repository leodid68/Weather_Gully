"""Tests for weather.calibrate — calibration script."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from weather.calibrate import (
    _compute_adaptive_factors,
    _compute_base_sigma,
    _compute_platt_params,
    _compute_sigma_by_horizon,
    _expand_sigma_by_horizon,
    _fit_student_t_df,
    _HORIZON_GROWTH,
    _test_normality,
    build_calibration_tables,
    compute_empirical_sigma,
    compute_forecast_errors,
    compute_horizon_errors,
    compute_model_weights,
    compute_exponential_weights,
    _weighted_base_sigma,
    build_weighted_calibration_tables,
)


class TestComputeEmpiricalSigma(unittest.TestCase):

    def test_by_horizon(self):
        errors = [
            {"horizon": 1, "month": 1, "error": 2.0},
            {"horizon": 1, "month": 1, "error": -2.0},
            {"horizon": 1, "month": 2, "error": 1.0},
            {"horizon": 2, "month": 1, "error": 4.0},
            {"horizon": 2, "month": 1, "error": -4.0},
            {"horizon": 2, "month": 2, "error": 2.0},
        ]
        result = compute_empirical_sigma(errors, group_by="horizon")
        self.assertIn("1", result)
        self.assertIn("2", result)
        # horizon=1: errors [2, -2, 1] → mean=0.33, stddev ≈ 1.70
        self.assertGreater(result["1"], 0)
        # horizon=2 should have higher sigma
        self.assertGreater(result["2"], result["1"])

    def test_by_month(self):
        errors = [
            {"horizon": 1, "month": 1, "error": 3.0},
            {"horizon": 1, "month": 1, "error": -3.0},
            {"horizon": 1, "month": 1, "error": 1.0},
            {"horizon": 1, "month": 7, "error": 1.0},
            {"horizon": 1, "month": 7, "error": -1.0},
            {"horizon": 1, "month": 7, "error": 0.5},
        ]
        result = compute_empirical_sigma(errors, group_by="month")
        self.assertIn("1", result)
        self.assertIn("7", result)
        # Winter (month 1) should have higher sigma than summer (month 7)
        self.assertGreater(result["1"], result["7"])

    def test_skips_small_groups(self):
        errors = [
            {"horizon": 1, "error": 1.0},
            {"horizon": 1, "error": -1.0},
            # Only 2 samples for horizon 1 — should be skipped (min 3)
        ]
        result = compute_empirical_sigma(errors, group_by="horizon")
        self.assertEqual(result, {})

    def test_empty_errors(self):
        result = compute_empirical_sigma([], group_by="horizon")
        self.assertEqual(result, {})


class TestComputeModelWeights(unittest.TestCase):

    def test_inverse_rmse_weighting(self):
        errors = [
            {"location": "NYC", "model": "gfs", "forecast": 50.0, "actual": 52.0},
            {"location": "NYC", "model": "gfs", "forecast": 48.0, "actual": 50.0},
            {"location": "NYC", "model": "ecmwf", "forecast": 51.0, "actual": 52.0},
            {"location": "NYC", "model": "ecmwf", "forecast": 49.5, "actual": 50.0},
        ]
        result = compute_model_weights(errors, group_by="location")
        self.assertIn("NYC", result)
        weights = result["NYC"]
        # ECMWF should have higher weight (lower RMSE)
        self.assertGreater(weights.get("ecmwf_ifs025", 0), weights.get("gfs_seamless", 0))
        # All weights should sum to ~1.0
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_includes_noaa_weight(self):
        errors = [
            {"location": "NYC", "model": "gfs", "forecast": 50.0, "actual": 52.0},
            {"location": "NYC", "model": "ecmwf", "forecast": 51.0, "actual": 52.0},
        ]
        result = compute_model_weights(errors, group_by="location")
        weights = result["NYC"]
        self.assertIn("noaa", weights)
        self.assertAlmostEqual(weights["noaa"], 0.20, places=2)


class TestBuildCalibrationTables(unittest.TestCase):

    def test_output_structure(self):
        import random
        errors = []
        for day in range(1, 20):
            for month in [1, 6]:
                for model in ["gfs", "ecmwf"]:
                    err = random.gauss(0, 2.5)
                    errors.append({
                        "location": "NYC",
                        "target_date": f"2025-{month:02d}-{day:02d}",
                        "month": month,
                        "metric": "high",
                        "model": model,
                        "forecast": 50.0 + err,
                        "actual": 50.0,
                        "error": err,
                        "model_spread": abs(random.gauss(0, 1.5)),
                    })

        result = build_calibration_tables(errors, locations=["NYC"])

        # Check structure
        self.assertIn("global_sigma", result)
        self.assertIn("location_sigma", result)
        self.assertIn("seasonal_factors", result)
        self.assertIn("location_seasonal", result)
        self.assertIn("model_weights", result)
        self.assertIn("metadata", result)

        # Metadata
        self.assertEqual(result["metadata"]["samples"], len(errors))
        self.assertIn("NYC", result["metadata"]["locations"])
        self.assertIn("base_sigma_global", result["metadata"])

        # Global sigma should have entries for horizons 0-10
        self.assertEqual(len(result["global_sigma"]), 11)
        self.assertIn("0", result["global_sigma"])
        self.assertIn("10", result["global_sigma"])

        # Sigma should increase with horizon (growth model)
        self.assertLess(result["global_sigma"]["0"], result["global_sigma"]["5"])
        self.assertLess(result["global_sigma"]["5"], result["global_sigma"]["10"])

        # Location sigma should also have full horizon table
        self.assertEqual(len(result["location_sigma"]["NYC"]), 11)

    def test_empty_errors_returns_fallback(self):
        result = build_calibration_tables([], locations=["NYC"])
        # With no errors, base sigma fallback (1.5) is used
        self.assertEqual(len(result["global_sigma"]), 11)
        self.assertAlmostEqual(result["global_sigma"]["0"], 1.5, places=1)
        self.assertEqual(result["model_weights"], {})
        self.assertEqual(result["metadata"]["samples"], 0)


class TestComputeForecastErrors(unittest.TestCase):

    @patch("weather.calibrate.get_historical_actuals")
    @patch("weather.calibrate.get_historical_forecasts")
    def test_computes_errors(self, mock_forecasts, mock_actuals):
        mock_forecasts.return_value = {
            "2025-01-14": {
                "2025-01-15": {"gfs_high": 42.0, "gfs_low": 30.0, "ecmwf_high": 44.0, "ecmwf_low": 31.0},
            },
        }
        mock_actuals.return_value = {
            "2025-01-15": {"high": 43.2, "low": 29.5},
        }

        errors = compute_forecast_errors(
            location="NYC", lat=40.77, lon=-73.87,
            start_date="2025-01-14", end_date="2025-01-15",
        )

        self.assertGreater(len(errors), 0)
        # Should have errors for gfs_high, gfs_low, ecmwf_high, ecmwf_low
        models = {e["model"] for e in errors}
        self.assertEqual(models, {"gfs", "ecmwf"})
        metrics = {e["metric"] for e in errors}
        self.assertEqual(metrics, {"high", "low"})

        # Check error calculation
        gfs_high_err = [e for e in errors if e["model"] == "gfs" and e["metric"] == "high"][0]
        self.assertAlmostEqual(gfs_high_err["error"], 42.0 - 43.2, places=1)

        # Should have model_spread (|gfs - ecmwf|)
        self.assertIn("model_spread", gfs_high_err)
        # |42.0 - 44.0| = 2.0
        self.assertAlmostEqual(gfs_high_err["model_spread"], 2.0, places=1)

        # Errors should be deduplicated — no "horizon" field
        self.assertNotIn("horizon", gfs_high_err)

    @patch("weather.calibrate.get_historical_actuals")
    @patch("weather.calibrate.get_historical_forecasts")
    def test_no_actuals_returns_empty(self, mock_forecasts, mock_actuals):
        mock_forecasts.return_value = {"2025-01-14": {"2025-01-15": {"gfs_high": 42.0}}}
        mock_actuals.return_value = {}

        errors = compute_forecast_errors(
            location="NYC", lat=40.77, lon=-73.87,
            start_date="2025-01-14", end_date="2025-01-15",
        )
        self.assertEqual(errors, [])


class TestHorizonGrowthModel(unittest.TestCase):

    def test_growth_factors_monotonically_increase(self):
        prev = 0.0
        for h in range(11):
            factor = _HORIZON_GROWTH[h]
            self.assertGreater(factor, prev)
            prev = factor

    def test_base_sigma_computation(self):
        errors = [{"error": 2.0}, {"error": -2.0}, {"error": 1.0}, {"error": -1.0}]
        sigma = _compute_base_sigma(errors)
        self.assertGreater(sigma, 0)
        self.assertLess(sigma, 3)

    def test_expand_sigma_by_horizon(self):
        result = _expand_sigma_by_horizon(2.0)
        self.assertEqual(len(result), 11)
        self.assertAlmostEqual(result["0"], 2.0, places=1)
        # h=10 should be 2.0 * 6.0 = 12.0
        self.assertAlmostEqual(result["10"], 12.0, places=1)
        # All values should be positive and increasing
        for h in range(10):
            self.assertLess(result[str(h)], result[str(h + 1)])

    def test_fallback_on_empty_errors(self):
        sigma = _compute_base_sigma([])
        self.assertAlmostEqual(sigma, 1.5, places=1)


class TestComputeAdaptiveFactors(unittest.TestCase):

    def _make_errors(self, n_days=30, base_spread=2.0, base_error=1.5):
        """Generate synthetic error records for n_days."""
        errors = []
        for day in range(1, n_days + 1):
            date_str = f"2025-01-{day:02d}" if day <= 28 else f"2025-02-{day - 28:02d}"
            for metric in ["high", "low"]:
                spread = base_spread + (day % 5) * 0.2
                for model in ["gfs", "ecmwf"]:
                    err = base_error * (1 if model == "gfs" else -0.8)
                    errors.append({
                        "location": "NYC",
                        "target_date": date_str,
                        "month": 1 if day <= 28 else 2,
                        "metric": metric,
                        "model": model,
                        "forecast": 50.0 + err,
                        "actual": 50.0,
                        "error": err,
                        "model_spread": spread,
                    })
        return errors

    def test_returns_all_keys(self):
        errors = self._make_errors()
        result = _compute_adaptive_factors(errors)
        self.assertIn("underdispersion_factor", result)
        self.assertIn("spread_to_sigma_factor", result)
        self.assertIn("ema_to_sigma_factor", result)
        self.assertIn("samples", result)

    def test_underdispersion_is_literature_default(self):
        errors = self._make_errors()
        result = _compute_adaptive_factors(errors)
        self.assertAlmostEqual(result["underdispersion_factor"], 1.3)

    def test_spread_factor_positive(self):
        errors = self._make_errors()
        result = _compute_adaptive_factors(errors)
        self.assertGreater(result["spread_to_sigma_factor"], 0)

    def test_ema_factor_positive(self):
        errors = self._make_errors()
        result = _compute_adaptive_factors(errors)
        self.assertGreater(result["ema_to_sigma_factor"], 0)

    def test_sample_count(self):
        errors = self._make_errors(n_days=10)
        result = _compute_adaptive_factors(errors)
        # 10 days × 2 metrics = 20 day-metric groups
        self.assertEqual(result["samples"], 20)

    def test_empty_errors_returns_defaults(self):
        result = _compute_adaptive_factors([])
        self.assertAlmostEqual(result["spread_to_sigma_factor"], 0.7)
        self.assertAlmostEqual(result["ema_to_sigma_factor"], 1.25)
        self.assertEqual(result["samples"], 0)

    def test_zero_spread_returns_fallback(self):
        errors = self._make_errors(n_days=5, base_spread=0.0)
        # Override all spreads to exactly 0
        for e in errors:
            e["model_spread"] = 0.0
        result = _compute_adaptive_factors(errors)
        # Should fall back to default spread factor
        self.assertAlmostEqual(result["spread_to_sigma_factor"], 0.7)

    def test_larger_spread_gives_smaller_factor(self):
        """If model spread is large relative to errors, the factor should be smaller."""
        small_spread = self._make_errors(n_days=20, base_spread=1.0, base_error=1.5)
        large_spread = self._make_errors(n_days=20, base_spread=5.0, base_error=1.5)
        r_small = _compute_adaptive_factors(small_spread)
        r_large = _compute_adaptive_factors(large_spread)
        self.assertGreater(r_small["spread_to_sigma_factor"],
                           r_large["spread_to_sigma_factor"])

    def test_integrated_in_build_calibration_tables(self):
        """build_calibration_tables should include adaptive_sigma key."""
        errors = self._make_errors(n_days=10)
        result = build_calibration_tables(errors, locations=["NYC"])
        self.assertIn("adaptive_sigma", result)
        self.assertIn("spread_to_sigma_factor", result["adaptive_sigma"])
        self.assertIn("ema_to_sigma_factor", result["adaptive_sigma"])


class TestComputePlattParams(unittest.TestCase):
    def test_perfect_calibration_gives_identity(self):
        predictions = [0.2, 0.4, 0.6, 0.8]
        actuals = [0.2, 0.4, 0.6, 0.8]
        params = _compute_platt_params(predictions, actuals)
        self.assertIn("a", params)
        self.assertIn("b", params)
        self.assertAlmostEqual(params["a"], 1.0, delta=0.3)
        self.assertAlmostEqual(params["b"], 0.0, delta=0.3)

    def test_empty_data_returns_identity(self):
        params = _compute_platt_params([], [])
        self.assertAlmostEqual(params["a"], 1.0)
        self.assertAlmostEqual(params["b"], 0.0)

    def test_overconfident_model(self):
        """Overconfident model (predicted > actual) should stretch probabilities."""
        predictions = [0.2, 0.35, 0.5, 0.65, 0.8]
        actuals = [0.05, 0.15, 0.30, 0.50, 0.70]
        params = _compute_platt_params(predictions, actuals)
        # a should be > 1 (stretches logit space to correct overconfidence)
        self.assertGreater(params["a"], 1.0)


class TestWeightedCalibration(unittest.TestCase):

    def _make_dated_errors(self, n_days=60, base_error=2.0):
        """Generate errors with dates spread over n_days ending today."""
        from datetime import datetime, timedelta
        errors = []
        today = datetime.now()
        for day_offset in range(n_days):
            date = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            month = (today - timedelta(days=day_offset)).month
            for metric in ["high", "low"]:
                for model in ["gfs", "ecmwf"]:
                    err = base_error * (1 if model == "gfs" else -0.8)
                    errors.append({
                        "location": "NYC",
                        "target_date": date,
                        "month": month,
                        "metric": metric,
                        "model": model,
                        "forecast": 50.0 + err,
                        "actual": 50.0,
                        "error": err,
                        "model_spread": abs(base_error * 0.5),
                    })
        return errors

    def test_compute_weights_recent_higher(self):
        from datetime import datetime, timedelta
        today = datetime.now()
        dates = [
            today.strftime("%Y-%m-%d"),
            (today - timedelta(days=60)).strftime("%Y-%m-%d"),
        ]
        weights = compute_exponential_weights(dates, half_life=30.0)
        today_str = today.strftime("%Y-%m-%d")
        old_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        # Today's weight should be higher than a 60-day-old weight
        self.assertGreater(weights[today_str], weights[old_str])
        # Today should be approximately 1.0
        self.assertAlmostEqual(weights[today_str], 1.0, places=2)

    def test_weighted_base_sigma_differs_from_unweighted(self):
        errors = self._make_dated_errors()
        from datetime import datetime
        dates = list({e["target_date"] for e in errors})
        weights = compute_exponential_weights(dates, half_life=30.0)
        weighted_sigma = _weighted_base_sigma(errors, weights)
        unweighted_sigma = _compute_base_sigma(errors)
        # Both should be positive
        self.assertGreater(weighted_sigma, 0)
        self.assertGreater(unweighted_sigma, 0)
        # Both should be in reasonable range
        self.assertLess(weighted_sigma, 10.0)
        self.assertLess(unweighted_sigma, 10.0)

    def test_build_weighted_tables_has_all_keys(self):
        errors = self._make_dated_errors()
        result = build_weighted_calibration_tables(errors, locations=["NYC"])
        self.assertIn("global_sigma", result)
        self.assertIn("location_sigma", result)
        self.assertIn("seasonal_factors", result)
        self.assertIn("model_weights", result)
        self.assertIn("adaptive_sigma", result)
        self.assertIn("platt_scaling", result)
        self.assertIn("metadata", result)

    def test_effective_samples_in_metadata(self):
        errors = self._make_dated_errors()
        result = build_weighted_calibration_tables(errors, locations=["NYC"])
        metadata = result["metadata"]
        self.assertIn("samples_effective", metadata)
        self.assertGreater(metadata["samples_effective"], 0)
        self.assertLessEqual(metadata["samples_effective"], metadata["samples"])


class TestComputeSigmaByHorizon(unittest.TestCase):

    def test_real_sigmas_computed(self):
        """Known horizons should have real RMSE values."""
        errors = []
        for h in [0, 3]:
            for _ in range(40):
                errors.append({"horizon": h, "error": (h + 1) * 1.0})
                errors.append({"horizon": h, "error": -(h + 1) * 1.0})
        result = _compute_sigma_by_horizon(errors)
        self.assertIn("0", result)
        self.assertIn("3", result)
        # Horizon 0: errors are +1/-1, RMSE = 1.0
        self.assertAlmostEqual(result["0"], 1.0, places=1)
        # Horizon 3: errors are +4/-4, RMSE = 4.0
        self.assertAlmostEqual(result["3"], 4.0, places=1)
        self.assertGreater(float(result["3"]), float(result["0"]))

    def test_interpolation_for_missing_horizons(self):
        """Missing horizons should be interpolated."""
        errors = []
        for _ in range(40):
            errors.append({"horizon": 0, "error": 1.0})
            errors.append({"horizon": 0, "error": -1.0})
            errors.append({"horizon": 7, "error": 5.0})
            errors.append({"horizon": 7, "error": -5.0})
        result = _compute_sigma_by_horizon(errors)
        # Horizon 3 should be between 0 and 7
        self.assertGreater(float(result["3"]), float(result["0"]))
        self.assertLess(float(result["3"]), float(result["7"]))

    def test_insufficient_samples_ignored(self):
        """Horizons with < 20 samples should be interpolated, not used directly."""
        errors = [{"horizon": 0, "error": 1.0}] * 40
        errors += [{"horizon": 5, "error": 3.0}] * 10  # only 10 samples
        errors += [{"horizon": 7, "error": 5.0}] * 40
        result = _compute_sigma_by_horizon(errors)
        # Horizon 5 should be interpolated (not 3.0)
        self.assertIn("5", result)

    def test_extrapolation_beyond_last_known(self):
        """Horizons beyond last known should be extrapolated."""
        errors = []
        for _ in range(40):
            errors.append({"horizon": 0, "error": 1.0})
            errors.append({"horizon": 0, "error": -1.0})
            errors.append({"horizon": 3, "error": 4.0})
            errors.append({"horizon": 3, "error": -4.0})
        result = _compute_sigma_by_horizon(errors)
        # Horizons 4-10 should be extrapolated and increasing
        self.assertIn("10", result)
        self.assertGreater(float(result["10"]), float(result["3"]))

    def test_all_horizons_present(self):
        """Result should have entries for all horizons 0-10."""
        errors = []
        for _ in range(40):
            errors.append({"horizon": 0, "error": 1.0})
            errors.append({"horizon": 0, "error": -1.0})
            errors.append({"horizon": 5, "error": 3.0})
            errors.append({"horizon": 5, "error": -3.0})
        result = _compute_sigma_by_horizon(errors)
        self.assertEqual(len(result), 11)
        for h in range(11):
            self.assertIn(str(h), result)

    def test_fallback_when_no_sufficient_horizons(self):
        """When no horizon has >= 20 samples, fall back to growth model."""
        errors = [{"horizon": 0, "error": 1.5}] * 5  # Only 5 samples
        result = _compute_sigma_by_horizon(errors)
        # Should still return 11 horizons (via fallback)
        self.assertEqual(len(result), 11)

    def test_empty_errors(self):
        """Empty errors should fall back to growth model."""
        result = _compute_sigma_by_horizon([])
        self.assertEqual(len(result), 11)


class TestComputeHorizonErrors(unittest.TestCase):

    def test_basic_errors(self):
        prev_runs = {
            0: {"2025-06-01": {"gfs_high": 82.0, "gfs_low": 65.0, "ecmwf_high": 84.0, "ecmwf_low": 66.0}},
            3: {"2025-06-01": {"gfs_high": 80.0, "gfs_low": 63.0, "ecmwf_high": 81.0, "ecmwf_low": 64.0}},
        }
        actuals = {"2025-06-01": {"high": 83.0, "low": 64.5}}
        errors = compute_horizon_errors(prev_runs, actuals, "NYC")
        self.assertTrue(len(errors) > 0)
        horizons = {e["horizon"] for e in errors}
        self.assertIn(0, horizons)
        self.assertIn(3, horizons)
        # All errors should have the standard fields
        for e in errors:
            self.assertIn("error", e)
            self.assertIn("horizon", e)
            self.assertIn("location", e)
            self.assertIn("target_date", e)
            self.assertIn("month", e)
            self.assertIn("metric", e)
            self.assertIn("model", e)
            self.assertIn("forecast", e)
            self.assertIn("actual", e)
            self.assertIn("model_spread", e)

    def test_error_values_correct(self):
        prev_runs = {
            0: {"2025-06-01": {"gfs_high": 82.0, "gfs_low": 65.0, "ecmwf_high": 84.0, "ecmwf_low": 66.0}},
        }
        actuals = {"2025-06-01": {"high": 83.0, "low": 64.5}}
        errors = compute_horizon_errors(prev_runs, actuals, "NYC")
        gfs_high = [e for e in errors if e["model"] == "gfs" and e["metric"] == "high"][0]
        # error = forecast - actual = 82.0 - 83.0 = -1.0
        self.assertAlmostEqual(gfs_high["error"], -1.0, places=1)
        self.assertEqual(gfs_high["horizon"], 0)
        self.assertEqual(gfs_high["location"], "NYC")

    def test_missing_actuals_skipped(self):
        prev_runs = {
            0: {"2025-06-01": {"gfs_high": 82.0, "gfs_low": 65.0, "ecmwf_high": 84.0, "ecmwf_low": 66.0}},
        }
        actuals = {}  # No actuals
        errors = compute_horizon_errors(prev_runs, actuals, "NYC")
        self.assertEqual(len(errors), 0)

    def test_model_spread_computed(self):
        prev_runs = {
            0: {"2025-06-01": {"gfs_high": 82.0, "gfs_low": 65.0, "ecmwf_high": 84.0, "ecmwf_low": 66.0}},
        }
        actuals = {"2025-06-01": {"high": 83.0, "low": 64.5}}
        errors = compute_horizon_errors(prev_runs, actuals, "NYC")
        gfs_high = [e for e in errors if e["model"] == "gfs" and e["metric"] == "high"][0]
        # |82.0 - 84.0| = 2.0
        self.assertAlmostEqual(gfs_high["model_spread"], 2.0, places=1)

    def test_multiple_dates_and_horizons(self):
        prev_runs = {
            0: {
                "2025-06-01": {"gfs_high": 82.0, "gfs_low": 65.0, "ecmwf_high": 84.0, "ecmwf_low": 66.0},
                "2025-06-02": {"gfs_high": 80.0, "gfs_low": 62.0, "ecmwf_high": 81.0, "ecmwf_low": 63.0},
            },
            3: {
                "2025-06-01": {"gfs_high": 79.0, "gfs_low": 61.0, "ecmwf_high": 80.0, "ecmwf_low": 62.0},
            },
        }
        actuals = {
            "2025-06-01": {"high": 83.0, "low": 64.5},
            "2025-06-02": {"high": 81.0, "low": 63.0},
        }
        errors = compute_horizon_errors(prev_runs, actuals, "NYC")
        # 2 models * 2 metrics * 2 dates at h=0 + 2 models * 2 metrics * 1 date at h=3 = 12
        self.assertEqual(len(errors), 12)


class TestBuildCalibrationTablesWithHorizonErrors(unittest.TestCase):

    def test_uses_real_rmse_when_horizon_errors_provided(self):
        """When horizon_errors is given, should use real RMSE, not growth model."""
        import random
        # Regular errors for the standard calibration fields
        all_errors = []
        for day in range(1, 20):
            for month in [1, 6]:
                for model in ["gfs", "ecmwf"]:
                    err = random.gauss(0, 2.5)
                    all_errors.append({
                        "location": "NYC",
                        "target_date": f"2025-{month:02d}-{day:02d}",
                        "month": month,
                        "metric": "high",
                        "model": model,
                        "forecast": 50.0 + err,
                        "actual": 50.0,
                        "error": err,
                        "model_spread": abs(random.gauss(0, 1.5)),
                    })

        # Horizon errors with known RMSE pattern
        horizon_errors = []
        for h in [0, 3, 7]:
            sigma = (h + 1) * 1.0
            for _ in range(40):
                horizon_errors.append({
                    "location": "NYC",
                    "target_date": "2025-06-01",
                    "horizon": h,
                    "error": sigma,
                })
                horizon_errors.append({
                    "location": "NYC",
                    "target_date": "2025-06-01",
                    "horizon": h,
                    "error": -sigma,
                })

        result = build_calibration_tables(all_errors, ["NYC"], horizon_errors=horizon_errors)
        self.assertIn("global_sigma", result)
        # Should use real RMSE model
        self.assertIn("real RMSE", result["metadata"]["horizon_growth_model"])
        # Horizon 0 RMSE = 1.0
        self.assertAlmostEqual(result["global_sigma"]["0"], 1.0, places=1)

    def test_backward_compatible_without_horizon_errors(self):
        """Without horizon_errors, should use old growth model."""
        import random
        errors = []
        for day in range(1, 20):
            for model in ["gfs", "ecmwf"]:
                err = random.gauss(0, 2.5)
                errors.append({
                    "location": "NYC",
                    "target_date": f"2025-01-{day:02d}",
                    "month": 1,
                    "metric": "high",
                    "model": model,
                    "forecast": 50.0 + err,
                    "actual": 50.0,
                    "error": err,
                    "model_spread": 1.0,
                })
        result = build_calibration_tables(errors, ["NYC"])
        self.assertIn("NWP linear", result["metadata"]["horizon_growth_model"])


class TestNormality(unittest.TestCase):
    def test_normal_data_passes(self):
        """Normal data should not be rejected."""
        import random
        random.seed(42)
        errors = [random.gauss(0, 2) for _ in range(200)]
        result = _test_normality(errors)
        self.assertTrue(result["normal"])

    def test_heavy_tailed_data_rejected(self):
        """Data with fat tails should be rejected."""
        import random
        random.seed(42)
        # Mix of normal + outliers (simulates fat tails)
        errors = [random.gauss(0, 2) for _ in range(180)]
        errors += [random.choice([-15, 15]) for _ in range(20)]
        result = _test_normality(errors)
        self.assertFalse(result["normal"])
        self.assertGreater(result["jb_statistic"], 5.991)

    def test_insufficient_data(self):
        """< 30 samples should return normal=True (not enough data to reject)."""
        result = _test_normality([1.0, -1.0, 2.0])
        self.assertTrue(result["normal"])


class TestStudentTFitting(unittest.TestCase):
    def test_fit_returns_valid_df(self):
        import random
        random.seed(42)
        errors = [random.gauss(0, 2) for _ in range(200)]
        df = _fit_student_t_df(errors)
        self.assertIn(df, [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100])

    def test_heavy_tails_give_low_df(self):
        """Heavy-tailed data should fit low df (broader tails)."""
        import random
        random.seed(42)
        # Generate Student's t with df=3 (heavy tails)
        errors = []
        for _ in range(500):
            # Approximate t(3) by normal / sqrt(chi2(3)/3)
            z = random.gauss(0, 1)
            chi2 = sum(random.gauss(0, 1)**2 for _ in range(3)) / 3
            errors.append(z / (chi2**0.5) * 2)
        df = _fit_student_t_df(errors)
        self.assertLessEqual(df, 10)  # Should prefer low df


if __name__ == "__main__":
    unittest.main()
