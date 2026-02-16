"""Tests for weather.recalibrate -- recalibration orchestrator."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from weather.recalibrate import (
    MIN_EFFECTIVE_SAMPLES,
    _compute_delta,
    _fetch_horizon_errors,
    filter_window,
    run_recalibration,
)


class TestFilterWindow(unittest.TestCase):
    """Tests for filter_window()."""

    def test_filters_to_window(self):
        """Errors outside the 90-day window are filtered out."""
        ref = "2026-02-16"
        errors = [
            {"target_date": "2026-02-10"},   # 6 days ago  -> keep
            {"target_date": "2026-01-01"},   # 46 days ago -> keep
            {"target_date": "2025-10-01"},   # ~138 days ago -> drop
        ]
        result = filter_window(errors, window_days=90, reference_date=ref)
        self.assertEqual(len(result), 2)
        dates = [e["target_date"] for e in result]
        self.assertIn("2026-02-10", dates)
        self.assertIn("2026-01-01", dates)
        self.assertNotIn("2025-10-01", dates)

    def test_empty_errors(self):
        """Empty input returns empty list."""
        result = filter_window([], window_days=90, reference_date="2026-02-16")
        self.assertEqual(result, [])


class TestComputeDelta(unittest.TestCase):
    """Tests for _compute_delta()."""

    def test_delta_between_calibrations(self):
        old_cal = {
            "global_sigma": {"0": 1.80},
            "platt_scaling": {"a": 0.75, "b": 0.30},
        }
        new_cal = {
            "global_sigma": {"0": 2.00},
            "platt_scaling": {"a": 0.80, "b": 0.25},
        }
        delta = _compute_delta(old_cal, new_cal)
        self.assertAlmostEqual(delta["base_sigma"], 0.20, places=4)
        self.assertAlmostEqual(delta["platt_a"], 0.05, places=4)
        self.assertAlmostEqual(delta["platt_b"], -0.05, places=4)

    def test_delta_no_old(self):
        """Empty old calibration returns empty dict."""
        new_cal = {
            "global_sigma": {"0": 2.00},
            "platt_scaling": {"a": 0.80, "b": 0.25},
        }
        delta = _compute_delta({}, new_cal)
        self.assertEqual(delta, {})


def _make_errors(n_days=30):
    """Generate plausible error records for testing."""
    errors = []
    today = datetime.now()
    for day_offset in range(n_days):
        date = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        month = (today - timedelta(days=day_offset)).month
        for metric in ["high", "low"]:
            for model in ["gfs", "ecmwf"]:
                errors.append({
                    "location": "NYC",
                    "target_date": date,
                    "month": month,
                    "metric": metric,
                    "model": model,
                    "forecast": 50.0 + (2.0 if model == "gfs" else -1.6),
                    "actual": 50.0,
                    "error": 2.0 if model == "gfs" else -1.6,
                    "model_spread": 1.5,
                })
    return errors


def _make_horizon_errors(n_days=30):
    """Generate plausible horizon error records for testing."""
    errors = []
    today = datetime.now()
    for day_offset in range(n_days):
        date = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        month = (today - timedelta(days=day_offset)).month
        for horizon in [0, 1, 3, 5, 7]:
            # Errors grow with horizon
            base_err = 1.0 + horizon * 0.3
            for metric in ["high", "low"]:
                for model in ["gfs", "ecmwf"]:
                    errors.append({
                        "location": "NYC",
                        "target_date": date,
                        "month": month,
                        "metric": metric,
                        "model": model,
                        "forecast": 50.0 + base_err,
                        "actual": 50.0,
                        "error": base_err if model == "gfs" else -base_err * 0.8,
                        "model_spread": 1.5,
                        "horizon": horizon,
                    })
    return errors


class TestRunRecalibration(unittest.TestCase):
    """Tests for run_recalibration() with mocked external dependencies."""

    @patch("weather.recalibrate._fetch_horizon_errors")
    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.save_error_cache")
    @patch("weather.recalibrate.load_error_cache")
    def test_successful_recalibration(
        self, mock_load, mock_save, mock_fetch, mock_horizon
    ):
        """Successful run writes calibration.json and log."""
        errors = _make_errors(n_days=45)  # 45 * 4 = 180 records, ~113 effective
        cache = {
            "version": 1,
            "errors": errors,
            "last_fetched": {"NYC": "2026-02-14"},
        }
        mock_load.return_value = cache
        mock_fetch.return_value = cache
        mock_save.return_value = None
        mock_horizon.return_value = []  # No horizon data — fallback to growth model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "calibration.json")
            log_dir = os.path.join(tmpdir, "log")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=output_path,
                log_dir=log_dir,
                reference_date=datetime.now().strftime("%Y-%m-%d"),
            )

            self.assertTrue(result["success"])
            self.assertIn("samples", result)
            self.assertIn("samples_effective", result)
            self.assertIn("delta", result)

            # calibration.json was written
            self.assertTrue(os.path.exists(output_path))
            with open(output_path) as f:
                cal = json.load(f)
            self.assertIn("global_sigma", cal)

            # log directory was created with a log file
            self.assertTrue(os.path.isdir(log_dir))
            log_files = os.listdir(log_dir)
            self.assertGreater(len(log_files), 0)

    @patch("weather.recalibrate._fetch_horizon_errors")
    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.save_error_cache")
    @patch("weather.recalibrate.load_error_cache")
    def test_insufficient_samples_aborts(
        self, mock_load, mock_save, mock_fetch, mock_horizon
    ):
        """Too few samples -> success=False, calibration.json NOT written."""
        errors = _make_errors(n_days=2)  # 2 * 4 = 8 records
        cache = {
            "version": 1,
            "errors": errors,
            "last_fetched": {"NYC": "2026-02-14"},
        }
        mock_load.return_value = cache
        mock_fetch.return_value = cache
        mock_save.return_value = None
        mock_horizon.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "calibration.json")
            log_dir = os.path.join(tmpdir, "log")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=output_path,
                log_dir=log_dir,
                reference_date=datetime.now().strftime("%Y-%m-%d"),
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["reason"], "insufficient_samples")
            # calibration.json was NOT written
            self.assertFalse(os.path.exists(output_path))

    @patch("weather.recalibrate._fetch_horizon_errors")
    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.save_error_cache")
    @patch("weather.recalibrate.load_error_cache")
    def test_with_horizon_errors(
        self, mock_load, mock_save, mock_fetch, mock_horizon
    ):
        """When horizon errors are provided, calibration uses real RMSE."""
        errors = _make_errors(n_days=45)
        horizon_errors = _make_horizon_errors(n_days=45)
        cache = {
            "version": 1,
            "errors": errors,
            "last_fetched": {"NYC": "2026-02-14"},
        }
        mock_load.return_value = cache
        mock_fetch.return_value = cache
        mock_save.return_value = None
        mock_horizon.return_value = horizon_errors

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "calibration.json")
            log_dir = os.path.join(tmpdir, "log")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=output_path,
                log_dir=log_dir,
                reference_date=datetime.now().strftime("%Y-%m-%d"),
            )

            self.assertTrue(result["success"])

            with open(output_path) as f:
                cal = json.load(f)

            # Should use real RMSE from Previous Runs data
            self.assertIn("global_sigma", cal)
            self.assertEqual(
                cal["metadata"]["horizon_growth_model"],
                "real RMSE from Previous Runs data",
            )

            # Sigma should increase with horizon
            sigma_0 = float(cal["global_sigma"]["0"])
            sigma_7 = float(cal["global_sigma"]["7"])
            self.assertGreater(sigma_7, sigma_0)

            # Distribution info should be present
            self.assertIn("distribution", cal)

            # Log should record horizon errors count
            log_files = os.listdir(log_dir)
            with open(os.path.join(log_dir, log_files[0])) as f:
                log = json.load(f)
            self.assertGreater(log["horizon_errors_count"], 0)

    @patch("weather.recalibrate._fetch_horizon_errors")
    @patch("weather.recalibrate.fetch_new_errors")
    @patch("weather.recalibrate.save_error_cache")
    @patch("weather.recalibrate.load_error_cache")
    def test_horizon_fetch_failure_falls_back(
        self, mock_load, mock_save, mock_fetch, mock_horizon
    ):
        """If horizon fetch fails, should fall back to growth model gracefully."""
        errors = _make_errors(n_days=45)
        cache = {
            "version": 1,
            "errors": errors,
            "last_fetched": {"NYC": "2026-02-14"},
        }
        mock_load.return_value = cache
        mock_fetch.return_value = cache
        mock_save.return_value = None
        mock_horizon.side_effect = Exception("API timeout")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "calibration.json")
            log_dir = os.path.join(tmpdir, "log")

            result = run_recalibration(
                locations=["NYC"],
                cache_path=os.path.join(tmpdir, "cache.json"),
                output_path=output_path,
                log_dir=log_dir,
                reference_date=datetime.now().strftime("%Y-%m-%d"),
            )

            # Should still succeed with growth model fallback
            self.assertTrue(result["success"])

            with open(output_path) as f:
                cal = json.load(f)
            # Should use growth model as fallback
            self.assertNotEqual(
                cal["metadata"]["horizon_growth_model"],
                "real RMSE from Previous Runs data",
            )


class TestFetchHorizonErrors(unittest.TestCase):
    """Tests for _fetch_horizon_errors()."""

    @patch("weather.recalibrate.compute_horizon_errors")
    @patch("weather.recalibrate.get_historical_metar_actuals")
    @patch("weather.recalibrate.fetch_previous_runs")
    def test_fetches_for_known_locations(
        self, mock_prev, mock_metar, mock_compute
    ):
        mock_prev.return_value = {0: {"2026-01-15": {"gfs_high": 42.0}}}
        mock_metar.return_value = {"2026-01-15": {"high": 43.0, "low": 30.0}}
        mock_compute.return_value = [
            {"location": "NYC", "horizon": 0, "error": -1.0,
             "target_date": "2026-01-15", "month": 1,
             "metric": "high", "model": "gfs",
             "forecast": 42.0, "actual": 43.0, "model_spread": 0.0},
        ]

        result = _fetch_horizon_errors(["NYC"], "2026-01-01", "2026-01-31")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["horizon"], 0)
        mock_prev.assert_called_once()
        mock_metar.assert_called_once()

    @patch("weather.recalibrate.fetch_previous_runs")
    def test_skips_unknown_location(self, mock_prev):
        result = _fetch_horizon_errors(["UNKNOWN"], "2026-01-01", "2026-01-31")
        self.assertEqual(result, [])
        mock_prev.assert_not_called()

    @patch("weather.recalibrate.get_historical_metar_actuals")
    @patch("weather.recalibrate.fetch_previous_runs")
    def test_skips_location_with_no_actuals(self, mock_prev, mock_metar):
        mock_prev.return_value = {0: {"2026-01-15": {"gfs_high": 42.0}}}
        mock_metar.return_value = {}

        result = _fetch_horizon_errors(["NYC"], "2026-01-01", "2026-01-31")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
