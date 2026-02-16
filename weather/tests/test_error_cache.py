"""Tests for weather.error_cache — incremental error history cache."""

import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from weather.error_cache import (
    CACHE_FORMAT_VERSION,
    _BOOTSTRAP_DAYS,
    _BUFFER_DAYS,
    fetch_new_errors,
    load_error_cache,
    prune_old_errors,
    save_error_cache,
)


class TestLoadSaveCache(unittest.TestCase):

    def test_load_missing_file_returns_empty(self):
        path = os.path.join(tempfile.gettempdir(), "nonexistent_error_cache.json")
        if os.path.exists(path):
            os.unlink(path)
        cache = load_error_cache(path)
        self.assertEqual(cache["version"], CACHE_FORMAT_VERSION)
        self.assertEqual(cache["errors"], [])
        self.assertEqual(cache["last_fetched"], {})

    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cache = {
                "version": CACHE_FORMAT_VERSION,
                "errors": [
                    {"location": "NYC", "target_date": "2026-01-15", "error": -1.2},
                ],
                "last_fetched": {"NYC": "2026-01-15"},
            }
            save_error_cache(cache, path)
            loaded = load_error_cache(path)
            self.assertEqual(loaded["version"], CACHE_FORMAT_VERSION)
            self.assertEqual(len(loaded["errors"]), 1)
            self.assertEqual(loaded["errors"][0]["location"], "NYC")
            self.assertEqual(loaded["last_fetched"]["NYC"], "2026-01-15")
        finally:
            os.unlink(path)

    def test_load_corrupt_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("NOT VALID JSON {{{")
            path = f.name
        try:
            cache = load_error_cache(path)
            self.assertEqual(cache["version"], CACHE_FORMAT_VERSION)
            self.assertEqual(cache["errors"], [])
            self.assertEqual(cache["last_fetched"], {})
        finally:
            os.unlink(path)


class TestPruneOldErrors(unittest.TestCase):

    def test_prunes_old_keeps_recent(self):
        today = date.today()
        old_date = (today - timedelta(days=400)).isoformat()
        recent_date = (today - timedelta(days=30)).isoformat()
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [
                {"location": "NYC", "target_date": old_date, "error": -1.0},
                {"location": "NYC", "target_date": recent_date, "error": 0.5},
            ],
            "last_fetched": {"NYC": recent_date},
        }
        pruned = prune_old_errors(cache, max_age_days=365)
        self.assertEqual(len(pruned["errors"]), 1)
        self.assertEqual(pruned["errors"][0]["target_date"], recent_date)

    def test_empty_cache_no_crash(self):
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {},
        }
        pruned = prune_old_errors(cache, max_age_days=365)
        self.assertEqual(pruned["errors"], [])


class TestFetchNewErrors(unittest.TestCase):

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_fetches_from_last_fetched_plus_one(self, mock_compute):
        mock_compute.return_value = [
            {"location": "NYC", "target_date": "2026-02-10", "month": 2,
             "metric": "high", "model": "gfs",
             "forecast": 42.0, "actual": 43.2, "error": -1.2, "model_spread": 1.5},
        ]
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {"NYC": "2026-02-08"},
        }
        ref_date = date(2026, 2, 16)
        result = fetch_new_errors(cache, ["NYC"], reference_date=ref_date)

        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        # start_date should be last_fetched + 1 = 2026-02-09
        self.assertEqual(call_kwargs[1]["start_date"], "2026-02-09")
        # end_date should be ref_date - BUFFER_DAYS = 2026-02-14
        self.assertEqual(call_kwargs[1]["end_date"], "2026-02-14")
        # errors should be appended
        self.assertEqual(len(result["errors"]), 1)
        # last_fetched should be updated to end_date
        self.assertEqual(result["last_fetched"]["NYC"], "2026-02-14")

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_bootstrap_fetches_90_days(self, mock_compute):
        mock_compute.return_value = []
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {},
        }
        ref_date = date(2026, 2, 16)
        result = fetch_new_errors(cache, ["NYC"], reference_date=ref_date)

        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        expected_start = (ref_date - timedelta(days=_BOOTSTRAP_DAYS)).isoformat()
        self.assertEqual(call_kwargs[1]["start_date"], expected_start)
        expected_end = (ref_date - timedelta(days=_BUFFER_DAYS)).isoformat()
        self.assertEqual(call_kwargs[1]["end_date"], expected_end)

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_skips_location_when_up_to_date(self, mock_compute):
        ref_date = date(2026, 2, 16)
        end_date = (ref_date - timedelta(days=_BUFFER_DAYS)).isoformat()
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {"NYC": end_date},
        }
        result = fetch_new_errors(cache, ["NYC"], reference_date=ref_date)

        mock_compute.assert_not_called()

    @patch("weather.error_cache._compute_errors_with_metar")
    def test_unknown_location_skipped(self, mock_compute):
        cache = {
            "version": CACHE_FORMAT_VERSION,
            "errors": [],
            "last_fetched": {},
        }
        ref_date = date(2026, 2, 16)
        result = fetch_new_errors(cache, ["UNKNOWN"], reference_date=ref_date)

        mock_compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
