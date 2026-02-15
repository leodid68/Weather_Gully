"""Tests for weather.historical — Open-Meteo historical forecast and actuals client."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from weather.historical import get_historical_actuals, get_historical_forecasts

FIXTURES = Path(__file__).parent / "fixtures"


class TestGetHistoricalForecasts(unittest.TestCase):

    @patch("weather.historical._fetch_json")
    def test_parses_forecast_response(self, mock_fetch):
        with open(FIXTURES / "historical_forecast_response.json") as f:
            mock_fetch.return_value = json.load(f)

        result = get_historical_forecasts(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-15",
            tz_name="America/New_York",
        )

        self.assertIn("2025-01-15", result)
        day = result["2025-01-15"]
        # Should have forecast for 2025-01-15 itself
        self.assertIn("2025-01-15", day)
        entry = day["2025-01-15"]
        self.assertIn("gfs_high", entry)
        self.assertIn("ecmwf_high", entry)
        self.assertAlmostEqual(entry["gfs_high"], 42.0, places=1)
        self.assertAlmostEqual(entry["ecmwf_high"], 44.0, places=1)

    @patch("weather.historical._fetch_json")
    def test_returns_empty_on_api_failure(self, mock_fetch):
        mock_fetch.return_value = None

        result = get_historical_forecasts(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-15",
        )
        self.assertEqual(result, {})

    @patch("weather.historical._fetch_json")
    def test_handles_missing_model_data(self, mock_fetch):
        mock_fetch.return_value = {
            "daily": {
                "time": ["2025-01-15"],
                "temperature_2m_max_gfs_seamless": [42.0],
                "temperature_2m_min_gfs_seamless": [30.0],
                # No ECMWF data
            }
        }

        result = get_historical_forecasts(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-15",
        )

        self.assertIn("2025-01-15", result)
        entry = result["2025-01-15"]["2025-01-15"]
        self.assertIn("gfs_high", entry)
        self.assertNotIn("ecmwf_high", entry)


class TestGetHistoricalActuals(unittest.TestCase):

    @patch("weather.historical._fetch_json")
    def test_parses_actuals_response(self, mock_fetch):
        with open(FIXTURES / "historical_actuals_response.json") as f:
            mock_fetch.return_value = json.load(f)

        result = get_historical_actuals(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-17",
        )

        self.assertEqual(len(result), 3)
        self.assertIn("2025-01-15", result)
        self.assertAlmostEqual(result["2025-01-15"]["high"], 43.2, places=1)
        self.assertAlmostEqual(result["2025-01-15"]["low"], 29.5, places=1)

    @patch("weather.historical._fetch_json")
    def test_returns_empty_on_api_failure(self, mock_fetch):
        mock_fetch.return_value = None

        result = get_historical_actuals(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-17",
        )
        self.assertEqual(result, {})

    @patch("weather.historical._fetch_json")
    def test_handles_none_values(self, mock_fetch):
        mock_fetch.return_value = {
            "daily": {
                "time": ["2025-01-15", "2025-01-16"],
                "temperature_2m_max": [43.2, None],
                "temperature_2m_min": [29.5, 33.2],
            }
        }

        result = get_historical_actuals(
            lat=40.7769, lon=-73.8740,
            start_date="2025-01-15", end_date="2025-01-16",
        )

        self.assertIn("2025-01-15", result)
        self.assertIn("high", result["2025-01-15"])
        # Second day: high is None, should only have "low"
        self.assertIn("2025-01-16", result)
        self.assertNotIn("high", result["2025-01-16"])
        self.assertIn("low", result["2025-01-16"])


if __name__ == "__main__":
    unittest.main()
