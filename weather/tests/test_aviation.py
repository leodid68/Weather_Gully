"""Tests for the Aviation Weather (METAR) client."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from weather.aviation import (
    STATION_MAP,
    _celsius_to_fahrenheit,
    _utc_to_local_date,
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
        expected = {
            "NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami",
            "London", "Paris", "Seoul", "Toronto",
            "BuenosAires", "SaoPaulo", "Ankara", "Wellington",
        }
        self.assertEqual(set(STATION_MAP.keys()), expected)

    def test_station_codes_are_icao(self):
        for loc, code in STATION_MAP.items():
            self.assertEqual(len(code), 4, f"{loc} station {code} should be 4 chars")
            self.assertTrue(code.isalpha(), f"{loc} station {code} should be alphabetic")


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

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_parses_fixture_correctly(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = await get_metar_observations(["NYC", "Chicago"])

        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)

        # NYC should have 6 observations (5 on Mar 15, 1 on Mar 14)
        self.assertEqual(len(result["NYC"]), 6)
        # Chicago should have 3 observations
        self.assertEqual(len(result["Chicago"]), 3)

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_temps_converted_to_fahrenheit(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = await get_metar_observations(["NYC"])

        # First observation (by time) for NYC is 3°C at 05:51
        # Sorted ascending: earliest first
        earliest = result["NYC"][0]
        # 8°C (Mar 14 22:51) → 46.4°F
        self.assertAlmostEqual(earliest["temp_f"], 46.4, places=1)

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_sorted_by_time_ascending(self, mock_fetch):
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = await get_metar_observations(["NYC"])
        times = [obs["time"] for obs in result["NYC"]]
        self.assertEqual(times, sorted(times))

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_unknown_location_skipped(self, mock_fetch):
        mock_fetch.return_value = []
        result = await get_metar_observations(["UnknownCity"])
        self.assertEqual(result, {})

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_api_failure_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None
        result = await get_metar_observations(["NYC"])
        self.assertEqual(result, {})

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_single_api_call_for_multiple_stations(self, mock_fetch):
        mock_fetch.return_value = []
        await get_metar_observations(["NYC", "Chicago", "Miami"])
        # Should be called exactly once (batched)
        mock_fetch.assert_called_once()
        call_url = mock_fetch.call_args[0][0]
        # All three stations should be in the URL
        self.assertIn("KLGA", call_url)
        self.assertIn("KORD", call_url)
        self.assertIn("KMIA", call_url)

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_skips_obs_with_missing_temp(self, mock_fetch):
        mock_fetch.return_value = [
            {"icaoId": "KLGA", "reportTime": "2025-03-15T12:00:00Z", "temp": None},
            {"icaoId": "KLGA", "reportTime": "2025-03-15T13:00:00Z", "temp": 10.0},
        ]
        result = await get_metar_observations(["NYC"])
        self.assertEqual(len(result["NYC"]), 1)

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_skips_obs_with_missing_time(self, mock_fetch):
        mock_fetch.return_value = [
            {"icaoId": "KLGA", "temp": 10.0},
        ]
        result = await get_metar_observations(["NYC"])
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

    @pytest.mark.asyncio
    @patch("weather.aviation.get_metar_observations", new_callable=AsyncMock)
    async def test_aggregates_by_date(self, mock_obs):
        mock_obs.return_value = {
            "NYC": [
                {"time": "2025-03-15T05:00:00Z", "temp_f": 37.4},
                {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
                {"time": "2025-03-14T22:00:00Z", "temp_f": 46.0},
            ],
        }

        result = await get_aviation_daily_data(["NYC"])

        self.assertIn("NYC", result)
        self.assertIn("2025-03-15", result["NYC"])
        self.assertIn("2025-03-14", result["NYC"])

        mar15 = result["NYC"]["2025-03-15"]
        self.assertAlmostEqual(mar15["obs_high"], 53.6)
        self.assertAlmostEqual(mar15["obs_low"], 37.4)
        self.assertEqual(mar15["obs_count"], 2)

    @pytest.mark.asyncio
    @patch("weather.aviation.get_metar_observations", new_callable=AsyncMock)
    async def test_empty_observations_excluded(self, mock_obs):
        mock_obs.return_value = {"NYC": []}

        result = await get_aviation_daily_data(["NYC"])
        self.assertNotIn("NYC", result)

    @pytest.mark.asyncio
    @patch("weather.aviation.get_metar_observations", new_callable=AsyncMock)
    async def test_multiple_locations(self, mock_obs):
        mock_obs.return_value = {
            "NYC": [
                {"time": "2025-03-15T14:00:00Z", "temp_f": 53.6},
            ],
            "Chicago": [
                {"time": "2025-03-15T14:00:00Z", "temp_f": 41.0},
            ],
        }

        result = await get_aviation_daily_data(["NYC", "Chicago"])
        self.assertIn("NYC", result)
        self.assertIn("Chicago", result)

    @pytest.mark.asyncio
    @patch("weather.aviation.fetch_json", new_callable=AsyncMock)
    async def test_end_to_end_with_fixture(self, mock_fetch):
        """Full pipeline: fixture → observations → daily data."""
        fixture = _load_fixture("metar_response.json")
        mock_fetch.return_value = fixture

        result = await get_aviation_daily_data(["NYC", "Chicago"])

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


class TestUtcToLocalDate(unittest.TestCase):
    """Test the _utc_to_local_date helper used for timezone conversion."""

    def test_nyc_early_morning_utc_is_previous_local_day(self):
        """2025-01-16T03:00:00Z in NYC (UTC-5) is still Jan 15 local."""
        result = _utc_to_local_date("2025-01-16T03:00:00Z", "America/New_York")
        self.assertEqual(result, "2025-01-15")

    def test_nyc_late_utc_is_same_local_day(self):
        """2025-01-16T18:00:00Z in NYC (UTC-5) is Jan 16 local."""
        result = _utc_to_local_date("2025-01-16T18:00:00Z", "America/New_York")
        self.assertEqual(result, "2025-01-16")

    def test_chicago_utc_offset(self):
        """2025-03-15T04:00:00Z in Chicago (UTC-5 in March DST) is Mar 14 local."""
        # March 15 is after DST spring-forward (CDT = UTC-5)
        result = _utc_to_local_date("2025-03-15T04:00:00Z", "America/Chicago")
        self.assertEqual(result, "2025-03-14")

    def test_seattle_utc_offset(self):
        """2025-06-15T06:00:00Z in Seattle (UTC-7 PDT) is Jun 14 local."""
        result = _utc_to_local_date("2025-06-15T06:00:00Z", "America/Los_Angeles")
        self.assertEqual(result, "2025-06-14")

    def test_miami_same_day(self):
        """2025-01-16T12:00:00Z in Miami (UTC-5) is Jan 16 local."""
        result = _utc_to_local_date("2025-01-16T12:00:00Z", "America/New_York")
        self.assertEqual(result, "2025-01-16")


class TestComputeDailyExtremesWithTimezone(unittest.TestCase):
    """Test that compute_daily_extremes correctly assigns UTC obs to local dates."""

    def test_utc_early_morning_obs_assigned_to_previous_local_day(self):
        """An observation at 03:00 UTC should be on the previous local day in NYC."""
        observations = [
            {"time": "2025-01-16T03:00:00Z", "temp_f": 28.0},  # Jan 15 local in NYC
            {"time": "2025-01-16T12:00:00Z", "temp_f": 35.0},  # Jan 16 local in NYC
            {"time": "2025-01-16T18:00:00Z", "temp_f": 40.0},  # Jan 16 local in NYC
        ]
        # Asking for Jan 15 local: should only include the 03:00Z obs
        result = compute_daily_extremes(observations, "2025-01-15", tz_name="America/New_York")
        self.assertIsNotNone(result)
        self.assertEqual(result["obs_count"], 1)
        self.assertAlmostEqual(result["high"], 28.0)

        # Asking for Jan 16 local: should include the 12:00Z and 18:00Z obs
        result_16 = compute_daily_extremes(observations, "2025-01-16", tz_name="America/New_York")
        self.assertIsNotNone(result_16)
        self.assertEqual(result_16["obs_count"], 2)
        self.assertAlmostEqual(result_16["high"], 40.0)
        self.assertAlmostEqual(result_16["low"], 35.0)

    def test_without_tz_name_uses_utc_date(self):
        """Without tz_name, falls back to UTC date prefix (legacy behaviour)."""
        observations = [
            {"time": "2025-01-16T03:00:00Z", "temp_f": 28.0},
        ]
        # UTC date is Jan 16, so asking for Jan 16 should match
        result = compute_daily_extremes(observations, "2025-01-16")
        self.assertIsNotNone(result)
        self.assertEqual(result["obs_count"], 1)

        # And Jan 15 should not match (UTC date is Jan 16)
        result_15 = compute_daily_extremes(observations, "2025-01-15")
        self.assertIsNone(result_15)

    def test_chicago_timezone_grouping(self):
        """Chicago CDT (UTC-5 in summer): 04:30Z should be previous local day."""
        observations = [
            {"time": "2025-06-15T04:30:00Z", "temp_f": 65.0},  # Jun 14 local (CDT)
            {"time": "2025-06-15T15:00:00Z", "temp_f": 82.0},  # Jun 15 local
        ]
        result_14 = compute_daily_extremes(observations, "2025-06-14", tz_name="America/Chicago")
        self.assertIsNotNone(result_14)
        self.assertEqual(result_14["obs_count"], 1)
        self.assertAlmostEqual(result_14["high"], 65.0)

        result_15 = compute_daily_extremes(observations, "2025-06-15", tz_name="America/Chicago")
        self.assertIsNotNone(result_15)
        self.assertEqual(result_15["obs_count"], 1)
        self.assertAlmostEqual(result_15["high"], 82.0)


class TestGetAviationDailyDataWithTimezone(unittest.TestCase):
    """Test that get_aviation_daily_data uses local timezone for date grouping."""

    @pytest.mark.asyncio
    @patch("weather.aviation.get_metar_observations", new_callable=AsyncMock)
    async def test_utc_midnight_crossing_grouped_correctly(self, mock_obs):
        """Observations near UTC midnight should be grouped by local date, not UTC."""
        mock_obs.return_value = {
            "NYC": [
                # 2025-01-16T03:00:00Z = 2025-01-15 22:00 EST → Jan 15 local
                {"time": "2025-01-16T03:00:00Z", "temp_f": 25.0},
                # 2025-01-16T06:00:00Z = 2025-01-16 01:00 EST → Jan 16 local
                {"time": "2025-01-16T06:00:00Z", "temp_f": 22.0},
                # 2025-01-16T18:00:00Z = 2025-01-16 13:00 EST → Jan 16 local
                {"time": "2025-01-16T18:00:00Z", "temp_f": 38.0},
            ],
        }

        result = await get_aviation_daily_data(["NYC"])

        self.assertIn("NYC", result)
        # Jan 15 should have 1 obs (the 03:00Z one)
        self.assertIn("2025-01-15", result["NYC"])
        jan15 = result["NYC"]["2025-01-15"]
        self.assertEqual(jan15["obs_count"], 1)
        self.assertAlmostEqual(jan15["obs_high"], 25.0)

        # Jan 16 should have 2 obs (the 06:00Z and 18:00Z ones)
        self.assertIn("2025-01-16", result["NYC"])
        jan16 = result["NYC"]["2025-01-16"]
        self.assertEqual(jan16["obs_count"], 2)
        self.assertAlmostEqual(jan16["obs_high"], 38.0)
        self.assertAlmostEqual(jan16["obs_low"], 22.0)


if __name__ == "__main__":
    unittest.main()
