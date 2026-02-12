"""Tests for Open-Meteo multi-source forecasting."""

import unittest

from weather.open_meteo import compute_ensemble_forecast


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


if __name__ == "__main__":
    unittest.main()
