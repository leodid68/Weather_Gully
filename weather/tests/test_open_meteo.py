"""Tests for Open-Meteo multi-source forecasting."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from weather.open_meteo import compute_ensemble_forecast, _get_model_weights


class TestComputeEnsembleForecast(unittest.TestCase):

    def test_all_three_sources(self):
        """NOAA + GFS + ECMWF should produce weighted average."""
        om_data = {"gfs_high": 50, "ecmwf_high": 54}
        temp, spread = compute_ensemble_forecast(52.0, om_data, "high")
        self.assertIsNotNone(temp)
        # Weighted: NOAA=52*0.20, GFS=50*0.30, ECMWF=54*0.50
        # = (10.4 + 15.0 + 27.0) / 1.0 = 52.4
        self.assertAlmostEqual(temp, 52.4, places=0)
        self.assertGreater(spread, 0)

    def test_noaa_only(self):
        """Only NOAA available — should return NOAA value."""
        temp, spread = compute_ensemble_forecast(55.0, None, "high")
        self.assertAlmostEqual(temp, 55.0, places=1)
        self.assertEqual(spread, 0.0)

    def test_open_meteo_only(self):
        """No NOAA, only Open-Meteo models."""
        om_data = {"gfs_low": 38, "ecmwf_low": 36}
        temp, spread = compute_ensemble_forecast(None, om_data, "low")
        self.assertIsNotNone(temp)
        # GFS=38*0.30, ECMWF=36*0.50 → (11.4 + 18.0) / 0.80 = 36.75
        self.assertAlmostEqual(temp, 36.8, places=0)

    def test_no_data_returns_none(self):
        """No sources → None."""
        temp, spread = compute_ensemble_forecast(None, None, "high")
        self.assertIsNone(temp)
        self.assertEqual(spread, 0.0)

    def test_empty_om_data(self):
        """Open-Meteo dict with no matching keys."""
        temp, spread = compute_ensemble_forecast(50.0, {}, "high")
        self.assertAlmostEqual(temp, 50.0, places=1)

    def test_spread_increases_with_disagreement(self):
        """Larger model disagreement → larger spread."""
        om_close = {"gfs_high": 51, "ecmwf_high": 53}
        _, spread_close = compute_ensemble_forecast(52.0, om_close, "high")

        om_far = {"gfs_high": 45, "ecmwf_high": 60}
        _, spread_far = compute_ensemble_forecast(52.0, om_far, "high")

        self.assertGreater(spread_far, spread_close)

    def test_low_metric(self):
        """Metric 'low' should use gfs_low and ecmwf_low keys."""
        om_data = {"gfs_low": 30, "ecmwf_low": 28, "gfs_high": 99, "ecmwf_high": 99}
        temp, _ = compute_ensemble_forecast(32.0, om_data, "low")
        self.assertIsNotNone(temp)
        # Should NOT use the high values
        self.assertLess(temp, 40)


class TestLocationWeights(unittest.TestCase):
    """Test location-based model weight loading (Phase 2)."""

    def test_default_weights_when_no_calibration(self):
        from weather.open_meteo import MODEL_WEIGHTS
        weights = _get_model_weights("")
        self.assertEqual(weights, MODEL_WEIGHTS)

    @patch("weather.probability._load_calibration", return_value={})
    def test_default_weights_for_unknown_location(self, mock_cal):
        from weather.open_meteo import MODEL_WEIGHTS
        weights = _get_model_weights("UNKNOWN_LOCATION")
        self.assertEqual(weights, MODEL_WEIGHTS)

    @patch("weather.probability._load_calibration")
    def test_calibrated_weights_used(self, mock_cal):
        fixture_path = Path(__file__).parent / "fixtures" / "calibration.json"
        import json
        with open(fixture_path) as f:
            cal_data = json.load(f)
        mock_cal.return_value = cal_data

        weights = _get_model_weights("NYC")
        # Should use calibrated weights from fixture
        self.assertAlmostEqual(weights["ecmwf_ifs025"], 0.55, places=2)
        self.assertAlmostEqual(weights["noaa"], 0.20, places=2)


class TestEnsembleWithLocation(unittest.TestCase):
    """Test that location parameter is accepted and doesn't break ensemble."""

    def test_location_param_accepted(self):
        om_data = {"gfs_high": 50, "ecmwf_high": 54}
        temp, spread = compute_ensemble_forecast(52.0, om_data, "high", location="NYC")
        self.assertIsNotNone(temp)
        self.assertGreater(spread, 0)

    def test_location_empty_string_uses_defaults(self):
        om_data = {"gfs_high": 50, "ecmwf_high": 54}
        temp_no_loc, _ = compute_ensemble_forecast(52.0, om_data, "high")
        temp_empty_loc, _ = compute_ensemble_forecast(52.0, om_data, "high", location="")
        self.assertAlmostEqual(temp_no_loc, temp_empty_loc, places=1)


class TestAuxiliaryWeatherVariables(unittest.TestCase):
    """Test that auxiliary variables are extracted from Open-Meteo response."""

    @patch("weather.open_meteo._fetch_json")
    def test_extracts_auxiliary_variables(self, mock_fetch):
        from weather.open_meteo import get_open_meteo_forecast

        mock_fetch.return_value = {
            "daily": {
                "time": ["2025-06-15"],
                "temperature_2m_max_gfs_seamless": [85.0],
                "temperature_2m_min_gfs_seamless": [68.0],
                "temperature_2m_max_ecmwf_ifs025": [87.0],
                "temperature_2m_min_ecmwf_ifs025": [69.0],
                "cloud_cover_max_gfs_seamless": [60.0],
                "cloud_cover_max_ecmwf_ifs025": [70.0],
                "wind_speed_10m_max_gfs_seamless": [25.0],
                "wind_speed_10m_max_ecmwf_ifs025": [30.0],
                "wind_gusts_10m_max_gfs_seamless": [40.0],
                "wind_gusts_10m_max_ecmwf_ifs025": [45.0],
                "precipitation_sum_gfs_seamless": [5.0],
                "precipitation_sum_ecmwf_ifs025": [3.0],
                "precipitation_probability_max_gfs_seamless": [60.0],
                "precipitation_probability_max_ecmwf_ifs025": [50.0],
            }
        }

        result = get_open_meteo_forecast(40.77, -73.87)
        self.assertIn("2025-06-15", result)
        day = result["2025-06-15"]

        # Temperature keys should still be present
        self.assertIn("gfs_high", day)
        self.assertIn("ecmwf_high", day)

        # Auxiliary variables should be present (averaged across models)
        self.assertIn("cloud_cover_max", day)
        self.assertAlmostEqual(day["cloud_cover_max"], 65.0, places=1)  # avg(60, 70)

        self.assertIn("wind_speed_max", day)
        self.assertAlmostEqual(day["wind_speed_max"], 27.5, places=1)  # avg(25, 30)

        self.assertIn("wind_gusts_max", day)
        self.assertIn("precip_sum", day)
        self.assertIn("precip_prob_max", day)

    @patch("weather.open_meteo._fetch_json")
    def test_handles_missing_aux_variables(self, mock_fetch):
        """When auxiliary variables are absent, entry should still have temps."""
        from weather.open_meteo import get_open_meteo_forecast

        mock_fetch.return_value = {
            "daily": {
                "time": ["2025-06-15"],
                "temperature_2m_max_gfs_seamless": [85.0],
                "temperature_2m_min_gfs_seamless": [68.0],
                "temperature_2m_max_ecmwf_ifs025": [87.0],
                "temperature_2m_min_ecmwf_ifs025": [69.0],
                # No auxiliary variables
            }
        }

        result = get_open_meteo_forecast(40.77, -73.87)
        self.assertIn("2025-06-15", result)
        day = result["2025-06-15"]
        self.assertIn("gfs_high", day)
        # Auxiliary keys should just be absent
        self.assertNotIn("cloud_cover_max", day)


class TestGetOpenMeteoForecastMulti(unittest.TestCase):
    """Tests for multi-location batch forecast fetching."""

    def _make_daily(self, dates, gfs_highs, gfs_lows, ecmwf_highs, ecmwf_lows):
        """Helper to build a daily response dict."""
        return {
            "daily": {
                "time": dates,
                "temperature_2m_max_gfs_seamless": gfs_highs,
                "temperature_2m_min_gfs_seamless": gfs_lows,
                "temperature_2m_max_ecmwf_ifs025": ecmwf_highs,
                "temperature_2m_min_ecmwf_ifs025": ecmwf_lows,
            }
        }

    @patch("weather.open_meteo._fetch_json")
    def test_single_location(self, mock_fetch):
        """Single location — API returns a dict with 'daily' key."""
        from weather.open_meteo import get_open_meteo_forecast_multi

        mock_fetch.return_value = self._make_daily(
            ["2025-06-15"], [85.0], [68.0], [87.0], [69.0],
        )

        locations = {"NYC": {"lat": 40.77, "lon": -73.87, "tz": "America/New_York"}}
        result = get_open_meteo_forecast_multi(locations)

        self.assertIn("NYC", result)
        self.assertIn("2025-06-15", result["NYC"])
        day = result["NYC"]["2025-06-15"]
        self.assertEqual(day["gfs_high"], 85)
        self.assertEqual(day["ecmwf_high"], 87)
        mock_fetch.assert_called_once()

    @patch("weather.open_meteo._fetch_json")
    def test_multiple_locations_same_timezone(self, mock_fetch):
        """Multiple locations in the same timezone — API returns a list."""
        from weather.open_meteo import get_open_meteo_forecast_multi

        mock_fetch.return_value = [
            self._make_daily(["2025-06-15"], [85.0], [68.0], [87.0], [69.0]),
            self._make_daily(["2025-06-15"], [78.0], [62.0], [80.0], [63.0]),
        ]

        locations = {
            "NYC": {"lat": 40.77, "lon": -73.87, "tz": "America/New_York"},
            "Atlanta": {"lat": 33.75, "lon": -84.39, "tz": "America/New_York"},
        }
        result = get_open_meteo_forecast_multi(locations)

        self.assertIn("NYC", result)
        self.assertIn("Atlanta", result)
        self.assertEqual(result["NYC"]["2025-06-15"]["gfs_high"], 85)
        self.assertEqual(result["Atlanta"]["2025-06-15"]["gfs_high"], 78)
        # Same timezone → single API call
        mock_fetch.assert_called_once()

    @patch("weather.open_meteo._fetch_json")
    def test_multiple_timezone_groups(self, mock_fetch):
        """Locations in different timezones — multiple API calls."""
        from weather.open_meteo import get_open_meteo_forecast_multi

        # First call for America/New_York, second for America/Chicago
        mock_fetch.side_effect = [
            self._make_daily(["2025-06-15"], [85.0], [68.0], [87.0], [69.0]),
            self._make_daily(["2025-06-15"], [90.0], [72.0], [92.0], [73.0]),
        ]

        locations = {
            "NYC": {"lat": 40.77, "lon": -73.87, "tz": "America/New_York"},
            "Chicago": {"lat": 41.88, "lon": -87.63, "tz": "America/Chicago"},
        }
        result = get_open_meteo_forecast_multi(locations)

        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)
        self.assertEqual(result["NYC"]["2025-06-15"]["gfs_high"], 85)
        self.assertEqual(result["Chicago"]["2025-06-15"]["gfs_high"], 90)
        self.assertEqual(mock_fetch.call_count, 2)

    @patch("weather.open_meteo._fetch_json")
    def test_api_failure_partial_data(self, mock_fetch):
        """One timezone group fails — that group returns empty, others succeed."""
        from weather.open_meteo import get_open_meteo_forecast_multi

        # First call succeeds, second returns None (failure)
        mock_fetch.side_effect = [
            self._make_daily(["2025-06-15"], [85.0], [68.0], [87.0], [69.0]),
            None,
        ]

        locations = {
            "NYC": {"lat": 40.77, "lon": -73.87, "tz": "America/New_York"},
            "Chicago": {"lat": 41.88, "lon": -87.63, "tz": "America/Chicago"},
        }
        result = get_open_meteo_forecast_multi(locations)

        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)
        # NYC should have data
        self.assertIn("2025-06-15", result["NYC"])
        # Chicago group failed — empty dict
        self.assertEqual(result["Chicago"], {})

    @patch("weather.open_meteo._fetch_json")
    def test_empty_locations(self, mock_fetch):
        """Empty locations dict — returns empty result, no API calls."""
        from weather.open_meteo import get_open_meteo_forecast_multi

        result = get_open_meteo_forecast_multi({})

        self.assertEqual(result, {})
        mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
