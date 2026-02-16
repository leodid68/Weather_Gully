"""Tests for weather.previous_runs — Previous Runs API client."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from weather.previous_runs import (
    _hourly_to_daily_max_min,
    _tz_offset_for_location,
    fetch_previous_runs,
)


def _utc_times(start_iso: str, count: int) -> list[str]:
    """Generate *count* hourly UTC timestamps starting at *start_iso*."""
    dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M")
    return [(dt + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(count)]


class TestHourlyToDailyMaxMin(unittest.TestCase):
    """Tests for _hourly_to_daily_max_min conversion."""

    def test_basic_conversion(self):
        """Full 24 hours of data yields correct max/min."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(24)]
        # Temps ramp from 30 at midnight to 53 at 11am, then back down
        values = [
            30, 29, 28, 27, 26, 27, 30, 35, 40, 45, 50, 53,
            52, 50, 48, 45, 42, 40, 38, 36, 34, 33, 32, 31,
        ]
        # tz_offset=0 means times are already local
        result = _hourly_to_daily_max_min(times, values, tz_offset=0)

        self.assertIn("2025-01-15", result)
        day_max, day_min = result["2025-01-15"]
        self.assertAlmostEqual(day_max, 53.0)
        self.assertAlmostEqual(day_min, 26.0)

    def test_timezone_offset_shifts_date(self):
        """Timezone offset shifts UTC hours into correct local date."""
        # 24 hours starting at 2025-01-15T05:00 UTC
        # With tz_offset=-5 (Eastern), 05:00 UTC = 00:00 local on Jan 15
        times = _utc_times("2025-01-15T05:00", 24)
        values = [float(30 + h) for h in range(24)]  # 30..53

        result = _hourly_to_daily_max_min(times, values, tz_offset=-5)

        # All hours should map to 2025-01-15 local time
        self.assertIn("2025-01-15", result)
        day_max, day_min = result["2025-01-15"]
        self.assertAlmostEqual(day_max, 53.0)
        self.assertAlmostEqual(day_min, 30.0)

    def test_timezone_offset_splits_across_days(self):
        """UTC hours spanning midnight local time split into two days."""
        # 2025-01-15T00:00 UTC to 2025-01-15T09:00 UTC (10 hours)
        # With tz_offset=-5: maps to 2025-01-14T19:00 .. 2025-01-15T04:00
        times = [f"2025-01-15T{h:02d}:00" for h in range(10)]
        values = [40.0 + h for h in range(10)]  # 40..49

        result = _hourly_to_daily_max_min(times, values, tz_offset=-5)

        # Neither day should have >= 12 hours, so both excluded
        self.assertEqual(len(result), 0)

    def test_null_values_skipped(self):
        """None values are ignored in max/min computation."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(24)]
        values: list[float | None] = [None] * 24
        # Set 15 non-null values (above threshold of 12)
        for i in range(15):
            values[i] = 30.0 + i * 2  # 30, 32, 34, ..., 58

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)

        self.assertIn("2025-01-15", result)
        day_max, day_min = result["2025-01-15"]
        self.assertAlmostEqual(day_max, 58.0)
        self.assertAlmostEqual(day_min, 30.0)

    def test_all_null_returns_empty(self):
        """All None values produce no result."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(24)]
        values: list[float | None] = [None] * 24

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertEqual(len(result), 0)

    def test_insufficient_hours_excluded(self):
        """Days with fewer than 12 valid hours are excluded."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(11)]
        values = [40.0 + h for h in range(11)]

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertNotIn("2025-01-15", result)

    def test_exactly_12_hours_included(self):
        """Days with exactly 12 valid hours are included."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(12)]
        values = [40.0 + h for h in range(12)]

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertIn("2025-01-15", result)
        day_max, day_min = result["2025-01-15"]
        self.assertAlmostEqual(day_max, 51.0)
        self.assertAlmostEqual(day_min, 40.0)

    def test_multi_day_data(self):
        """Two full days produce two entries."""
        times = []
        values = []
        for day in range(15, 17):
            for h in range(24):
                times.append(f"2025-01-{day:02d}T{h:02d}:00")
                values.append(30.0 + h + (day - 15) * 5)

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)

        self.assertEqual(len(result), 2)
        self.assertIn("2025-01-15", result)
        self.assertIn("2025-01-16", result)

    def test_values_are_rounded(self):
        """Output values are rounded to 1 decimal."""
        times = [f"2025-01-15T{h:02d}:00" for h in range(24)]
        values = [30.123456 + h * 0.1 for h in range(24)]

        result = _hourly_to_daily_max_min(times, values, tz_offset=0)
        self.assertIn("2025-01-15", result)
        day_max, day_min = result["2025-01-15"]
        # Verify rounded to 1 decimal
        self.assertEqual(day_max, round(day_max, 1))
        self.assertEqual(day_min, round(day_min, 1))


class TestTzOffset(unittest.TestCase):
    """Tests for _tz_offset_for_location."""

    def test_new_york(self):
        self.assertEqual(_tz_offset_for_location("America/New_York"), -5)

    def test_chicago(self):
        self.assertEqual(_tz_offset_for_location("America/Chicago"), -6)

    def test_denver(self):
        self.assertEqual(_tz_offset_for_location("America/Denver"), -7)

    def test_los_angeles(self):
        self.assertEqual(_tz_offset_for_location("America/Los_Angeles"), -8)

    def test_unknown_defaults_to_eastern(self):
        self.assertEqual(_tz_offset_for_location("Europe/Paris"), -5)

    def test_empty_string_defaults_to_eastern(self):
        self.assertEqual(_tz_offset_for_location(""), -5)


class TestFetchPreviousRuns(unittest.TestCase):
    """Tests for fetch_previous_runs with mocked HTTP."""

    def _make_hourly_response(self, date: str = "2025-01-15"):
        """Build a realistic Previous Runs API response for one day.

        Creates 24 hours of hourly data with GFS and ECMWF temps for
        horizon 0 (temperature_2m) and horizon 1 (temperature_2m_previous_day1).
        Times are in UTC; for tz_offset=-5 the local date maps to *date*.
        """
        # Start at 05:00 UTC so that local Eastern (UTC-5) = 00:00
        # Use _utc_times to properly handle date rollover at UTC midnight
        times = _utc_times(f"{date}T05:00", 24)

        gfs_h0 = [40.0 + h * 0.5 for h in range(24)]
        ecmwf_h0 = [41.0 + h * 0.5 for h in range(24)]
        gfs_h1 = [39.0 + h * 0.5 for h in range(24)]
        ecmwf_h1 = [40.0 + h * 0.5 for h in range(24)]

        return {
            "hourly": {
                "time": times,
                "temperature_2m_gfs_seamless": gfs_h0,
                "temperature_2m_ecmwf_ifs025": ecmwf_h0,
                "temperature_2m_previous_day1_gfs_seamless": gfs_h1,
                "temperature_2m_previous_day1_ecmwf_ifs025": ecmwf_h1,
            }
        }

    @patch("weather.previous_runs._fetch_json")
    def test_basic_structure(self, mock_fetch):
        """Returned dict has correct horizon keys and date structure."""
        mock_fetch.return_value = self._make_hourly_response("2025-01-15")

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0, 1],
            tz_name="America/New_York",
        )

        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertIn("2025-01-15", result[0])
        self.assertIn("2025-01-15", result[1])

    @patch("weather.previous_runs._fetch_json")
    def test_entry_keys(self, mock_fetch):
        """Each date entry has gfs_high, gfs_low, ecmwf_high, ecmwf_low."""
        mock_fetch.return_value = self._make_hourly_response("2025-01-15")

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0],
            tz_name="America/New_York",
        )

        entry = result[0]["2025-01-15"]
        for key in ("gfs_high", "gfs_low", "ecmwf_high", "ecmwf_low"):
            self.assertIn(key, entry)
            self.assertIsInstance(entry[key], float)

    @patch("weather.previous_runs._fetch_json")
    def test_max_min_values_correct(self, mock_fetch):
        """Daily max/min are correctly computed from hourly data."""
        mock_fetch.return_value = self._make_hourly_response("2025-01-15")

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0],
            tz_name="America/New_York",
        )

        entry = result[0]["2025-01-15"]
        # GFS h0: values from 40.0 to 40.0 + 23*0.5 = 51.5
        self.assertAlmostEqual(entry["gfs_high"], 51.5, places=1)
        self.assertAlmostEqual(entry["gfs_low"], 40.0, places=1)
        # ECMWF h0: 41.0 to 52.5
        self.assertAlmostEqual(entry["ecmwf_high"], 52.5, places=1)
        self.assertAlmostEqual(entry["ecmwf_low"], 41.0, places=1)

    @patch("weather.previous_runs._fetch_json")
    def test_empty_response_returns_empty_horizons(self, mock_fetch):
        """API returning None produces empty dicts per horizon."""
        mock_fetch.return_value = None

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0, 1],
            tz_name="America/New_York",
        )

        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 0)

    @patch("weather.previous_runs._fetch_json")
    def test_no_hourly_key_returns_empty(self, mock_fetch):
        """API returning dict without 'hourly' key produces empty results."""
        mock_fetch.return_value = {"error": True, "reason": "invalid param"}

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0],
        )

        self.assertEqual(len(result[0]), 0)

    @patch("weather.previous_runs._fetch_json")
    def test_default_horizons(self, mock_fetch):
        """When horizons=None, uses default [0, 1, 2, 3, 5, 7]."""
        mock_fetch.return_value = self._make_hourly_response("2025-01-15")

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            tz_name="America/New_York",
        )

        for h in [0, 1, 2, 3, 5, 7]:
            self.assertIn(h, result)

    @patch("weather.previous_runs._fetch_json")
    def test_chunking_for_large_range(self, mock_fetch):
        """Date range > 90 days triggers multiple API calls."""
        mock_fetch.return_value = self._make_hourly_response("2025-01-15")

        fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-01", end_date="2025-06-30",
            horizons=[0],
            tz_name="America/New_York",
        )

        # 181 days / 90 = 3 chunks (day 1-90, 91-180, 181)
        self.assertEqual(mock_fetch.call_count, 3)

    @patch("weather.previous_runs._fetch_json")
    def test_partial_model_data(self, mock_fetch):
        """Missing one model still returns the other model's data."""
        times = _utc_times("2025-01-15T05:00", 24)
        mock_fetch.return_value = {
            "hourly": {
                "time": times,
                "temperature_2m_gfs_seamless": [42.0 + h for h in range(24)],
                # No ECMWF data
            }
        }

        result = fetch_previous_runs(
            lat=40.78, lon=-73.87,
            start_date="2025-01-15", end_date="2025-01-15",
            horizons=[0],
            tz_name="America/New_York",
        )

        entry = result[0]["2025-01-15"]
        self.assertIn("gfs_high", entry)
        self.assertIn("gfs_low", entry)
        self.assertNotIn("ecmwf_high", entry)
        self.assertNotIn("ecmwf_low", entry)


if __name__ == "__main__":
    unittest.main()
