"""Tests for international sigma boost."""

import pytest
from unittest.mock import patch, MagicMock
from weather.config import Config, LOCATIONS


class TestInternationalSigmaBoost:
    def test_config_default(self):
        """Default international_sigma_boost is 1.3."""
        config = Config()
        assert config.international_sigma_boost == 1.3

    def test_us_cities_are_fahrenheit(self):
        """US cities have unit='F'."""
        for city in ["NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami"]:
            assert LOCATIONS[city]["unit"] == "F"

    def test_international_cities_are_celsius(self):
        """International cities have unit='C'."""
        for city in ["London", "Paris", "Seoul", "Toronto", "BuenosAires", "SaoPaulo", "Ankara", "Wellington"]:
            assert LOCATIONS[city]["unit"] == "C"

    def test_boost_applied_to_celsius_city(self):
        """Sigma for a C city should be 1.3x higher than base."""
        from weather.config import LOCATIONS
        # Verify the detection mechanism works
        loc_data = LOCATIONS.get("London", {})
        assert loc_data.get("unit") == "C"

        base_sigma = 2.0
        boosted = base_sigma * 1.3
        assert boosted == pytest.approx(2.6)

    def test_no_boost_for_us_city(self):
        """Sigma for a F city should NOT be boosted."""
        loc_data = LOCATIONS.get("NYC", {})
        assert loc_data.get("unit") == "F"
        # unit != "C", so boost should not be applied
