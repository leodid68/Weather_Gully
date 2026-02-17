"""Tests for model conviction signal (dominant model sizing adjustment)."""
import unittest
from unittest.mock import patch


class TestGetDominantModelInfo(unittest.TestCase):
    """Test get_dominant_model_info() from open_meteo.py."""

    def test_get_dominant_model_gfs(self):
        """GFS with weight 0.85 should be returned as dominant with its temp."""
        mock_weights = {"gfs_seamless": 0.85, "ecmwf_ifs025": 0.0, "noaa": 0.0}
        om_data = {"gfs_high": 52.0, "ecmwf_high": 54.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=50.0, om_data=om_data, metric="high")
        self.assertEqual(name, "gfs_seamless")
        self.assertEqual(temp, 52.0)
        self.assertEqual(weight, 0.85)

    def test_get_dominant_model_noaa(self):
        """NOAA with weight 0.5 should return the NOAA temperature."""
        mock_weights = {"gfs_seamless": 0.2, "ecmwf_ifs025": 0.1, "noaa": 0.5}
        om_data = {"gfs_high": 52.0, "ecmwf_high": 54.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=48.0, om_data=om_data, metric="high")
        self.assertEqual(name, "noaa")
        self.assertEqual(temp, 48.0)
        self.assertEqual(weight, 0.5)

    def test_no_dominant_model(self):
        """All weights below 0.4 should return empty result."""
        mock_weights = {"gfs_seamless": 0.35, "ecmwf_ifs025": 0.35, "noaa": 0.30}
        om_data = {"gfs_high": 52.0, "ecmwf_high": 54.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=50.0, om_data=om_data, metric="high")
        self.assertEqual(name, "")
        self.assertIsNone(temp)
        self.assertEqual(weight, 0.0)

    def test_get_dominant_model_ecmwf(self):
        """ECMWF with weight 0.6 should return ecmwf temp from om_data."""
        mock_weights = {"gfs_seamless": 0.2, "ecmwf_ifs025": 0.6, "noaa": 0.1}
        om_data = {"gfs_high": 52.0, "ecmwf_high": 54.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=50.0, om_data=om_data, metric="high")
        self.assertEqual(name, "ecmwf_ifs025")
        self.assertEqual(temp, 54.0)
        self.assertEqual(weight, 0.6)

    def test_dominant_model_no_om_data(self):
        """GFS dominant but no om_data: should return name and weight but no temp."""
        mock_weights = {"gfs_seamless": 0.85, "ecmwf_ifs025": 0.0, "noaa": 0.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=50.0, om_data=None, metric="high")
        self.assertEqual(name, "gfs_seamless")
        self.assertIsNone(temp)
        self.assertEqual(weight, 0.85)

    def test_dominant_model_low_metric(self):
        """Should use 'low' suffix when metric is 'low'."""
        mock_weights = {"gfs_seamless": 0.85, "ecmwf_ifs025": 0.0, "noaa": 0.0}
        om_data = {"gfs_low": 38.0, "ecmwf_low": 36.0}
        with patch("weather.open_meteo._get_model_weights", return_value=mock_weights):
            from weather.open_meteo import get_dominant_model_info
            name, temp, weight = get_dominant_model_info("NYC", noaa_temp=40.0, om_data=om_data, metric="low")
        self.assertEqual(name, "gfs_seamless")
        self.assertEqual(temp, 38.0)


class TestConvictionSizingMath(unittest.TestCase):
    """Test the conviction sizing adjustment logic in isolation."""

    @staticmethod
    def _apply_conviction(position_size, dom_model_temp, dom_model_weight,
                          model_disagreement, bucket, adaptive_sigma_value):
        """Replicate the conviction sizing logic from strategy.py."""
        if dom_model_temp is not None and dom_model_weight >= 0.5 and model_disagreement:
            if bucket and adaptive_sigma_value and adaptive_sigma_value > 0:
                _lo, _hi = bucket
                bc = _hi if _lo < -900 else (_lo if _hi > 900 else (_lo + _hi) / 2.0)
                dist = abs(dom_model_temp - bc)
                if dist <= adaptive_sigma_value * 0.5:
                    position_size = round(position_size * 1.5, 2)
                elif dist > adaptive_sigma_value * 1.5:
                    position_size = round(position_size * 0.5, 2)
        return position_size

    def test_conviction_boost_bucket_agrees(self):
        """Dominant model temp within 0.5*sigma of bucket center -> 1.5x boost."""
        # Bucket 48-52, center=50, sigma=4, dominant model temp=50
        # dist = |50-50| = 0 <= 4*0.5 = 2 -> boost
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=50.0,
            dom_model_weight=0.85,
            model_disagreement=True,
            bucket=(48, 52),
            adaptive_sigma_value=4.0,
        )
        self.assertAlmostEqual(result, 15.0)

    def test_conviction_penalty_bucket_disagrees(self):
        """Dominant model temp > 1.5*sigma from bucket center -> 0.5x penalty."""
        # Bucket 48-52, center=50, sigma=4, dominant model temp=60
        # dist = |60-50| = 10 > 4*1.5 = 6 -> penalty
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=60.0,
            dom_model_weight=0.85,
            model_disagreement=True,
            bucket=(48, 52),
            adaptive_sigma_value=4.0,
        )
        self.assertAlmostEqual(result, 5.0)

    def test_no_conviction_no_disagreement(self):
        """model_disagreement=False -> no adjustment."""
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=50.0,
            dom_model_weight=0.85,
            model_disagreement=False,
            bucket=(48, 52),
            adaptive_sigma_value=4.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_no_conviction_low_weight(self):
        """Weight below 0.5 -> no adjustment even with disagreement."""
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=50.0,
            dom_model_weight=0.45,
            model_disagreement=True,
            bucket=(48, 52),
            adaptive_sigma_value=4.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_no_conviction_neutral_distance(self):
        """Distance between 0.5*sigma and 1.5*sigma -> no adjustment."""
        # Bucket 48-52, center=50, sigma=4, dominant model temp=54
        # dist = |54-50| = 4; 0.5*4 = 2 < 4 <= 1.5*4 = 6 -> no change
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=54.0,
            dom_model_weight=0.85,
            model_disagreement=True,
            bucket=(48, 52),
            adaptive_sigma_value=4.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_conviction_open_ended_low_bucket(self):
        """Open-ended low bucket (lo < -900): center = hi."""
        # Bucket (-999, 40), center=40, sigma=3, dominant model temp=40
        # dist = 0 <= 1.5 -> boost
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=40.0,
            dom_model_weight=0.7,
            model_disagreement=True,
            bucket=(-999, 40),
            adaptive_sigma_value=3.0,
        )
        self.assertAlmostEqual(result, 15.0)

    def test_conviction_open_ended_high_bucket(self):
        """Open-ended high bucket (hi > 900): center = lo."""
        # Bucket (60, 999), center=60, sigma=3, dominant model temp=70
        # dist = 10 > 4.5 -> penalty
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=70.0,
            dom_model_weight=0.7,
            model_disagreement=True,
            bucket=(60, 999),
            adaptive_sigma_value=3.0,
        )
        self.assertAlmostEqual(result, 5.0)

    def test_no_conviction_no_sigma(self):
        """No adaptive sigma -> no adjustment."""
        result = self._apply_conviction(
            position_size=10.0,
            dom_model_temp=50.0,
            dom_model_weight=0.85,
            model_disagreement=True,
            bucket=(48, 52),
            adaptive_sigma_value=None,
        )
        self.assertAlmostEqual(result, 10.0)
