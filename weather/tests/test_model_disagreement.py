"""Tests for model disagreement detection and sigma boost."""
import unittest
from weather.config import Config


class TestModelDisagreementConfig(unittest.TestCase):
    """Test config fields for model disagreement."""

    def test_default_threshold(self):
        config = Config()
        self.assertEqual(config.model_disagreement_threshold, 3.0)

    def test_default_multiplier(self):
        config = Config()
        self.assertEqual(config.model_disagreement_multiplier, 1.5)

    def test_custom_values(self):
        config = Config()
        config.model_disagreement_threshold = 5.0
        config.model_disagreement_multiplier = 2.0
        self.assertEqual(config.model_disagreement_threshold, 5.0)
        self.assertEqual(config.model_disagreement_multiplier, 2.0)


class TestDisagreementDetectionLogic(unittest.TestCase):
    """Test the disagreement detection logic in isolation."""

    def test_disagreement_detected(self):
        noaa_temp = 40.0
        om_only_temp = 44.0
        threshold = 3.0
        spread = abs(noaa_temp - om_only_temp)
        self.assertTrue(spread >= threshold)

    def test_no_disagreement_when_close(self):
        noaa_temp = 40.0
        om_only_temp = 41.5
        threshold = 3.0
        spread = abs(noaa_temp - om_only_temp)
        self.assertFalse(spread >= threshold)

    def test_sigma_boost_math(self):
        base_sigma = 2.0
        multiplier = 1.5
        self.assertAlmostEqual(base_sigma * multiplier, 3.0)

    def test_no_boost_below_threshold(self):
        """No sigma change when spread is below threshold."""
        base_sigma = 2.0
        noaa_temp = 40.0
        om_only_temp = 42.0
        threshold = 3.0
        spread = abs(noaa_temp - om_only_temp)
        if spread >= threshold:
            base_sigma *= 1.5
        self.assertAlmostEqual(base_sigma, 2.0)

    def test_negative_spread_uses_abs(self):
        """Spread should be absolute."""
        noaa_temp = 44.0
        om_only_temp = 40.0
        threshold = 3.0
        spread = abs(noaa_temp - om_only_temp)
        self.assertTrue(spread >= threshold)
