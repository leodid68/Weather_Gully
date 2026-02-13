"""Tests for the Aviation Weather (METAR) client."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from weather.aviation import (
    STATION_MAP,
    _celsius_to_fahrenheit,
    compute_daily_extremes,
    get_aviation_daily_data,
    get_metar_observations,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str):
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestStationMap(unittest.TestCase):

    def test_all_polymarket_locations_mapped(self):
        expected = {"NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami"}
        self.assertEqual(set(STATION_MAP.keys()), expected)

    def test_station_codes_are_icao(self):
        for loc, code in STATION_MAP.items():
            self.assertTrue(code.startswith("K"), f"{loc} station {code} should start with K")
            self.assertEqual(len(code), 4, f"{loc} station {code} should be 4 chars")


class TestCelsiusToFahrenheit(unittest.TestCase):

    def test_freezing_point(self):
        self.assertAlmostEqual(_celsius_to_fahrenheit(0), 32.0, places=1)

    def test_boiling_point(self):
        self.assertAlmostEqual(_celsius_to_fahrenheit(100), 212.0, places=1)

    def test_negative(self):
        self.assertAlmostEqual(_celsius_to_fahrenheit(-40), -40.0, places=1)

    def test_body_temp(self):
        self.assertAlmostEqual(_celsius_to_fahrenheit(37), 98.6, places=1)

    def test_typical_weather(self):
        # 20°C = 68°F
        self.assertAlmostEqual(_celsius_to_fahrenheit(20), 68.0, places=1)


class TestGetMetarObservations(unittest.TestCase):

    @patch("weather.aviation._fetch_json")
    def test_parses_fixture_correctly(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = get_metar_observations(["NYC", "Chicago"])

        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)

        # NYC should have 6 observations (5 on Mar 15, 1 on Mar 14)
        self.assertEqual(len(result["NYC"]), 6)
        # Chicago should have 3 observations
        self.assertEqual(len(result["Chicago"]), 3)

    @patch("weather.aviation._fetch_json")
    def test_temps_converted_to_fahrenheit(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = get_metar_observations(["NYC"])

        # First observation (by time) for NYC is 3°C at 05:51
        # Sorted ascending: earliest first
        earliest = result["NYC"][0]
        # 8°C (Mar 14 22:51) → 46.4°F
        self.assertAlmostEqual(earliest["temp_f"], 46.4, places=1)

    @patch("weather.aviation._fetch_json")
    def test_sorted_by_time_ascending(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = get_metar_observations(["NYC"])
        times = [obs["time"] for obs in result["NYC"]]
        self.assertEqual(times, sorted(times))

    @patch("weather.aviation._fetch_json")
    def test_unknown_location_skipped(self, mock_fetch):
        mock_fetch.return_value = []
        result = get_metar_observations(["UnknownCity"])
        self.assertEqual(result, {})

    @patch("weather.aviation._fetch_json")
    def test_api_failure_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None
        result = get_metar_observations(["NYC"])
        self.assertEqual(result, {})

    @patch("weather.aviation._fetch_json")
    def test_single_api_call_for_multiple_stations(self, mock_fetch):
        mock_fetch.return_value = []
        get_metar_observations(["NYC", "Chicago", "Miami"])
        # Should be called exactly once (batched)
        mock_fetch.assert_called_once()
        call_url = mock_fetch.call_args[0][0]
        # All three stations should be in the URL
        self.assertIn("KLGA", call_url)
        self.assertIn("KORD", call_url)
        self.assertIn("KMIA", call_url)

    @patch("weather.aviation._fetch_json")
    def test_skips_obs_with_missing_temp(self, mock_fetch):
        mock_fetch.return_value = [
            {"icaoId": "KLGA", "reportTime": "2025-03-15T12:00:00Z", "temp": None},
            {"icaoId": "KLGA", "reportTime": "2025-03-15T13:00:00Z", "temp": 10.0},
        ]
        result = get_metar_observations(["NYC"])
        self.assertEqual(len(result["NYC"]), 1)

    @patch("weather.aviation._fetch_json")
    def test_skips_obs_with_missing_time(self, mock_fetch):
        mock_fetch.return_value = [
            {"icaoId": "KLGA", "temp": 10.0},
        ]
        result = get_metar_observations(["NYC"])
        self.assertEqual(len(result["NYC"]), 0)


class TestComputeDailyExtremes(unittest.TestCase):

    def test_computes_high_and_low(self):
        observations = [
            {"time": "2025-03-15T05:00:00Z", "temp_f": 37.4},
            {"time": "2025-03-15T08:00:00Z", "temp_f": 42.8},
            {"time": "2025-03-15T12:00:00Z", "temp_f": 50.0},
            {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
            {"time": "2025-03-15T18:00:00Z", "temp_f": 48.2},
        ]
        result = compute_daily_extremes(observations, "2025-03-15")
        self.assertAlmostEqual(result["high"], 53.6)
        self.assertAlmostEqual(result["low"], 37.4)
        self.assertEqual(result["obs_count"], 5)
        self.assertEqual(result["latest_obs_time"], "2025-03-15T18:00:00Z")

    def test_filters_by_date(self):
        observations = [
            {"time": "2025-03-14T22:00:00Z", "temp_f": 46.0},
            {"time": "2025-03-15T08:00:00Z", "temp_f": 42.8},
            {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
        ]
        result = compute_daily_extremes(observations, "2025-03-15")
        self.assertEqual(result["obs_count"], 2)
        self.assertAlmostEqual(result["high"], 53.6)
        self.assertAlmostEqual(result["low"], 42.8)

    def test_no_matching_date_returns_none(self):
        observations = [
            {"time": "2025-03-14T22:00:00Z", "temp_f": 46.0},
        ]
        result = compute_daily_extremes(observations, "2025-03-15")
        self.assertIsNone(result)

    def test_single_observation(self):
        observations = [
            {"time": "2025-03-15T12:00:00Z", "temp_f": 50.0},
        ]
        result = compute_daily_extremes(observations, "2025-03-15")
        self.assertAlmostEqual(result["high"], 50.0)
        self.assertAlmostEqual(result["low"], 50.0)
        self.assertEqual(result["obs_count"], 1)

    def test_empty_observations(self):
        result = compute_daily_extremes([], "2025-03-15")
        self.assertIsNone(result)


class TestGetAviationDailyData(unittest.TestCase):

    @patch("weather.aviation.get_metar_observations")
    def test_aggregates_by_date(self, mock_obs):
        mock_obs.return_value = {
            "NYC": [
                {"time": "2025-03-15T05:00:00Z", "temp_f": 37.4},
                {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
                {"time": "2025-03-14T22:00:00Z", "temp_f": 46.0},
            ],
        }

        result = get_aviation_daily_data(["NYC"])

        self.assertIn("NYC", result)
        self.assertIn("2025-03-15", result["NYC"])
        self.assertIn("2025-03-14", result["NYC"])

        mar15 = result["NYC"]["2025-03-15"]
        self.assertAlmostEqual(mar15["obs_high"], 53.6)
        self.assertAlmostEqual(mar15["obs_low"], 37.4)
        self.assertEqual(mar15["obs_count"], 2)

    @patch("weather.aviation.get_metar_observations")
    def test_empty_observations_excluded(self, mock_obs):
        mock_obs.return_value = {"NYC": []}

        result = get_aviation_daily_data(["NYC"])
        self.assertNotIn("NYC", result)

    @patch("weather.aviation.get_metar_observations")
    def test_multiple_locations(self, mock_obs):
        mock_obs.return_value = {
            "NYC": [
                {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
            ],
            "Chicago": [
                {"time": "2025-03-15T14:00:00Z", "temp_f": 41.0},
            ],
        }

        result = get_aviation_daily_data(["NYC", "Chicago"])
        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)

    @patch("weather.aviation._fetch_json")
    def test_end_to_end_with_fixture(self, mock_fetch):
        """Full pipeline: fixture → observations → daily data."""
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = get_aviation_daily_data(["NYC", "Chicago"])

        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)

        # NYC Mar 15: temps are 3°C, 6°C, 10°C, 11°C, 12°C → 37.4, 42.8, 50.0, 51.8, 53.6°F
        nyc_mar15 = result["NYC"]["2025-03-15"]
        self.assertAlmostEqual(nyc_mar15["obs_high"], 53.6, places=1)
        self.assertAlmostEqual(nyc_mar15["obs_low"], 37.4, places=1)
        self.assertEqual(nyc_mar15["obs_count"], 5)

        # Chicago Mar 15: temps are -2°C, 5°C → 28.4, 41.0°F
        chi_mar15 = result["Chicago"]["2025-03-15"]
        self.assertAlmostEqual(chi_mar15["obs_high"], 41.0, places=1)
        self.assertAlmostEqual(chi_mar15["obs_low"], 28.4, places=1)
        self.assertEqual(chi_mar15["obs_count"], 2)


if __name__ == "__main__":
    unittest.main()
