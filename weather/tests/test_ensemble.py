"""Tests for the ensemble data model, disk cache, and API client."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from weather.ensemble import (
    EnsembleResult,
    _cache_path,
    _read_cache,
    _write_cache,
    fetch_ensemble_spread,
)


class TestEnsembleResult(unittest.TestCase):

    def test_creation_defaults(self):
        result = EnsembleResult()
        self.assertEqual(result.member_temps, [])
        self.assertAlmostEqual(result.ensemble_mean, 0.0)
        self.assertAlmostEqual(result.ensemble_stddev, 0.0)
        self.assertAlmostEqual(result.ecmwf_stddev, 0.0)
        self.assertAlmostEqual(result.gfs_stddev, 0.0)
        self.assertEqual(result.n_members, 0)

    def test_creation_with_values(self):
        temps = [70.0, 71.5, 72.0]
        result = EnsembleResult(
            member_temps=temps,
            ensemble_mean=71.17,
            ensemble_stddev=0.83,
            ecmwf_stddev=0.7,
            gfs_stddev=0.9,
            n_members=3,
        )
        self.assertEqual(result.member_temps, temps)
        self.assertAlmostEqual(result.ensemble_mean, 71.17)
        self.assertAlmostEqual(result.ensemble_stddev, 0.83)
        self.assertAlmostEqual(result.ecmwf_stddev, 0.7)
        self.assertAlmostEqual(result.gfs_stddev, 0.9)
        self.assertEqual(result.n_members, 3)

    def test_empty_returns_defaults(self):
        result = EnsembleResult.empty()
        self.assertEqual(result.member_temps, [])
        self.assertAlmostEqual(result.ensemble_mean, 0.0)
        self.assertAlmostEqual(result.ensemble_stddev, 0.0)
        self.assertEqual(result.n_members, 0)

    def test_empty_returns_new_instance(self):
        a = EnsembleResult.empty()
        b = EnsembleResult.empty()
        self.assertIsNot(a, b)
        # Mutating one should not affect the other
        a.member_temps.append(99.0)
        self.assertEqual(b.member_temps, [])


class TestCachePath(unittest.TestCase):

    def test_format(self):
        cache_dir = Path("/tmp/test_cache")
        path = _cache_path(cache_dir, 40.71, -74.01, "2026-02-15", "temperature_2m_max")
        self.assertEqual(
            path,
            Path("/tmp/test_cache/40.71_-74.01_2026-02-15_temperature_2m_max.json"),
        )

    def test_returns_path_object(self):
        path = _cache_path(Path("/tmp"), 0.0, 0.0, "2026-01-01", "t")
        self.assertIsInstance(path, Path)


class TestCache(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ensemble_cache_test_")
        self.cache_dir = Path(self.tmp_dir) / "cache" / "ensemble"

    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_write_read_roundtrip(self):
        """Write a result, then read it back and verify fields match."""
        original = EnsembleResult(
            member_temps=[68.0, 69.5, 70.2, 71.0],
            ensemble_mean=69.675,
            ensemble_stddev=1.1,
            ecmwf_stddev=0.9,
            gfs_stddev=1.3,
            n_members=4,
        )

        _write_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "temperature_2m_max", original)
        loaded = _read_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "temperature_2m_max")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.member_temps, original.member_temps)
        self.assertAlmostEqual(loaded.ensemble_mean, original.ensemble_mean)
        self.assertAlmostEqual(loaded.ensemble_stddev, original.ensemble_stddev)
        self.assertAlmostEqual(loaded.ecmwf_stddev, original.ecmwf_stddev)
        self.assertAlmostEqual(loaded.gfs_stddev, original.gfs_stddev)
        self.assertEqual(loaded.n_members, original.n_members)

    def test_write_creates_parent_dirs(self):
        """Cache dir should be created automatically."""
        self.assertFalse(self.cache_dir.exists())
        _write_cache(self.cache_dir, 0.0, 0.0, "2026-01-01", "t", EnsembleResult.empty())
        self.assertTrue(self.cache_dir.exists())

    def test_write_adds_cached_at_timestamp(self):
        """The JSON file should contain a _cached_at field."""
        _write_cache(self.cache_dir, 0.0, 0.0, "2026-01-01", "t", EnsembleResult.empty())
        path = _cache_path(self.cache_dir, 0.0, 0.0, "2026-01-01", "t")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("_cached_at", data)
        self.assertIsInstance(data["_cached_at"], float)

    def test_read_missing_returns_none(self):
        """Reading a non-existent cache file should return None."""
        result = _read_cache(self.cache_dir, 99.0, 99.0, "2099-01-01", "missing")
        self.assertIsNone(result)

    def test_expired_cache_returns_none(self):
        """A cache entry older than TTL should return None."""
        _write_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "t", EnsembleResult.empty())

        # Patch time.time to simulate 7 hours later (> 6h default TTL)
        real_time = time.time()
        with patch("weather.ensemble.time") as mock_time:
            mock_time.time.return_value = real_time + 25200  # 7 hours
            result = _read_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "t")

        self.assertIsNone(result)

    def test_not_expired_within_ttl(self):
        """A cache entry within TTL should be returned."""
        _write_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "t", EnsembleResult.empty())

        # Patch time.time to simulate 1 hour later (< 6h default TTL)
        real_time = time.time()
        with patch("weather.ensemble.time") as mock_time:
            mock_time.time.return_value = real_time + 3600  # 1 hour
            result = _read_cache(self.cache_dir, 40.71, -74.01, "2026-02-15", "t")

        self.assertIsNotNone(result)

    def test_corrupted_cache_returns_none(self):
        """A corrupted JSON file should return None, not raise."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(self.cache_dir, 0.0, 0.0, "2026-01-01", "t")
        path.write_text("{corrupted!!")

        result = _read_cache(self.cache_dir, 0.0, 0.0, "2026-01-01", "t")
        self.assertIsNone(result)

    def test_custom_ttl(self):
        """Custom TTL should be respected."""
        _write_cache(self.cache_dir, 0.0, 0.0, "2026-01-01", "t", EnsembleResult.empty())

        real_time = time.time()
        with patch("weather.ensemble.time") as mock_time:
            # 10 seconds later, with 5-second TTL -> expired
            mock_time.time.return_value = real_time + 10
            result = _read_cache(self.cache_dir, 0.0, 0.0, "2026-01-01", "t", ttl_seconds=5)

        self.assertIsNone(result)


class TestFetchEnsembleSpread(unittest.TestCase):

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_basic_fetch(self, mock_fetch):
        """Mock API returns member temps, verify stddev computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Return a response simulating 5 ECMWF + 3 GFS members
            mock_fetch.return_value = [
                {"daily": {"time": ["2026-02-15"], "temperature_2m_max_member0": [50.0], "temperature_2m_max_member1": [52.0], "temperature_2m_max_member2": [54.0], "temperature_2m_max_member3": [48.0], "temperature_2m_max_member4": [56.0]}, "model": "ecmwf_ifs025"},
                {"daily": {"time": ["2026-02-15"], "temperature_2m_max_member0": [49.0], "temperature_2m_max_member1": [53.0], "temperature_2m_max_member2": [51.0]}, "model": "gfs025"},
            ]
            result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high",
                                            cache_dir=tmpdir)
            self.assertGreater(result.n_members, 0)
            self.assertGreater(result.ensemble_stddev, 0)
            self.assertEqual(result.n_members, 8)

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_api_failure_returns_empty(self, mock_fetch):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_fetch.return_value = None
            result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high",
                                            cache_dir=tmpdir)
            self.assertEqual(result.n_members, 0)

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_cache_hit_skips_api(self, mock_fetch):
        with tempfile.TemporaryDirectory() as tmpdir:
            from weather.ensemble import _write_cache, EnsembleResult
            cached = EnsembleResult(member_temps=[50.0, 52.0], ensemble_mean=51.0,
                                     ensemble_stddev=1.41, ecmwf_stddev=1.0,
                                     gfs_stddev=1.5, n_members=2)
            _write_cache(Path(tmpdir), 40.77, -73.87, "2026-02-15", "high", cached)
            result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "high", cache_dir=tmpdir)
            mock_fetch.assert_not_called()
            self.assertAlmostEqual(result.ensemble_stddev, 1.41)

    def test_stddev_basic(self):
        from weather.ensemble import _stddev
        self.assertAlmostEqual(_stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 2.0, places=0)

    def test_stddev_single_value(self):
        from weather.ensemble import _stddev
        self.assertEqual(_stddev([5.0]), 0.0)

    def test_stddev_empty(self):
        from weather.ensemble import _stddev
        self.assertEqual(_stddev([]), 0.0)

    @patch("weather.ensemble._fetch_ensemble_json")
    def test_low_metric_uses_min(self, mock_fetch):
        """Metric 'low' should query temperature_2m_min."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_fetch.return_value = [
                {"daily": {"time": ["2026-02-15"], "temperature_2m_min_member0": [30.0], "temperature_2m_min_member1": [32.0]}, "model": "ecmwf_ifs025"},
            ]
            result = fetch_ensemble_spread(40.77, -73.87, "2026-02-15", "low",
                                            cache_dir=tmpdir)
            self.assertEqual(result.n_members, 2)


if __name__ == "__main__":
    unittest.main()
