"""Tests for weather event and temperature bucket parsing."""

import unittest
from datetime import datetime, timezone

from weather.parsing import parse_temperature_bucket, parse_weather_event


class TestParseWeatherEvent(unittest.TestCase):
    """~15 cases covering locations, dates, metrics, edge cases."""

    def test_nyc_high(self):
        result = parse_weather_event("What will be the highest temperature in NYC on March 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "NYC")
        self.assertEqual(result["metric"], "high")
        self.assertIn("-03-15", result["date"])

    def test_new_york_alias(self):
        result = parse_weather_event("Highest temp in New York on Jan 5?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "NYC")

    def test_laguardia_alias(self):
        result = parse_weather_event("High temp at LaGuardia on Feb 20?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "NYC")

    def test_chicago_low(self):
        result = parse_weather_event("What will be the lowest temperature in Chicago on March 16?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Chicago")
        self.assertEqual(result["metric"], "low")

    def test_ohare_alias(self):
        result = parse_weather_event("Low temp at O'Hare on Apr 1?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Chicago")

    def test_seattle(self):
        result = parse_weather_event("Highest temperature in Seattle on June 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Seattle")

    def test_atlanta(self):
        result = parse_weather_event("High temp at Hartsfield on Jul 4?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Atlanta")

    def test_dallas(self):
        result = parse_weather_event("Highest temp at DFW on Aug 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Dallas")

    def test_miami(self):
        result = parse_weather_event("Highest temperature in Miami on Dec 25?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Miami")

    def test_abbreviated_month(self):
        result = parse_weather_event("Highest temp in NYC on Feb 14?")
        self.assertIsNotNone(result)
        self.assertIn("-02-14", result["date"])

    def test_unknown_location_returns_none(self):
        result = parse_weather_event("Highest temp in Denver on March 15?")
        self.assertIsNone(result)

    def test_missing_date_returns_none(self):
        result = parse_weather_event("Highest temp in NYC")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        result = parse_weather_event("")
        self.assertIsNone(result)

    def test_none_input_returns_none(self):
        result = parse_weather_event(None)
        self.assertIsNone(result)

    def test_invalid_date_returns_none(self):
        result = parse_weather_event("Highest temp in NYC on February 30?")
        self.assertIsNone(result)

    # --- International cities ---

    def test_london(self):
        result = parse_weather_event("Highest temperature in London on March 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "London")

    def test_heathrow_alias(self):
        result = parse_weather_event("High temp at Heathrow on March 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "London")

    def test_lhr_alias(self):
        result = parse_weather_event("High temp at LHR on March 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "London")

    def test_paris(self):
        result = parse_weather_event("Highest temperature in Paris on Apr 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Paris")

    def test_cdg_alias(self):
        result = parse_weather_event("High temp at CDG on Apr 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Paris")

    def test_seoul(self):
        result = parse_weather_event("Highest temperature in Seoul on May 20?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Seoul")

    def test_incheon_alias(self):
        result = parse_weather_event("High temp at Incheon on May 20?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Seoul")

    def test_toronto(self):
        result = parse_weather_event("Highest temperature in Toronto on Jun 1?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Toronto")

    def test_pearson_alias(self):
        result = parse_weather_event("High temp at Pearson on Jun 1?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Toronto")

    def test_buenos_aires(self):
        result = parse_weather_event("High temp in Buenos Aires on Jul 4?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "BuenosAires")

    def test_ezeiza_alias(self):
        result = parse_weather_event("High temp at Ezeiza on Jul 4?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "BuenosAires")

    def test_sao_paulo(self):
        result = parse_weather_event("High temp in Sao Paulo on Aug 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "SaoPaulo")

    def test_sao_paulo_accented(self):
        result = parse_weather_event("High temp in São Paulo on Aug 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "SaoPaulo")

    def test_ankara(self):
        result = parse_weather_event("Highest temperature in Ankara on Sep 5?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Ankara")

    def test_esenboga_alias(self):
        result = parse_weather_event("High temp at Esenboga on Sep 5?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Ankara")

    def test_wellington(self):
        result = parse_weather_event("Highest temperature in Wellington on Oct 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "Wellington")

    def test_london_low_metric(self):
        result = parse_weather_event("What will be the lowest temperature in London on March 15?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "London")
        self.assertEqual(result["metric"], "low")

    def test_buenosaires_no_space(self):
        result = parse_weather_event("High temp in BuenosAires on Jul 4?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "BuenosAires")

    def test_saopaulo_no_space(self):
        result = parse_weather_event("High temp in SaoPaulo on Aug 10?")
        self.assertIsNotNone(result)
        self.assertEqual(result["location"], "SaoPaulo")


class TestParseTemperatureBucket(unittest.TestCase):
    """Covers open/closed buckets, various formats, edge cases."""

    def test_range_dash(self):
        self.assertEqual(parse_temperature_bucket("50-54°F"), (50, 54))

    def test_range_endash(self):
        self.assertEqual(parse_temperature_bucket("50–54"), (50, 54))

    def test_range_reversed(self):
        # Handles reversed ranges gracefully
        self.assertEqual(parse_temperature_bucket("54-50°F"), (50, 54))

    def test_or_below(self):
        self.assertEqual(parse_temperature_bucket("44°F or below"), (-999, 44))

    def test_or_less(self):
        self.assertEqual(parse_temperature_bucket("44 or less"), (-999, 44))

    def test_or_higher(self):
        self.assertEqual(parse_temperature_bucket("60°F or higher"), (60, 999))

    def test_or_above(self):
        self.assertEqual(parse_temperature_bucket("60 or above"), (60, 999))

    def test_or_more(self):
        self.assertEqual(parse_temperature_bucket("60°F or more"), (60, 999))

    def test_plus_sign(self):
        self.assertEqual(parse_temperature_bucket("60°F+"), (60, 999))

    def test_no_degree_symbol(self):
        self.assertEqual(parse_temperature_bucket("45-49"), (45, 49))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_temperature_bucket(""))

    def test_none_returns_none(self):
        self.assertIsNone(parse_temperature_bucket(None))

    def test_no_numbers_returns_none(self):
        self.assertIsNone(parse_temperature_bucket("Sunny weather"))

    def test_and_below(self):
        self.assertEqual(parse_temperature_bucket("30 and below"), (-999, 30))

    def test_and_above(self):
        self.assertEqual(parse_temperature_bucket("70 and above"), (70, 999))

    def test_negative_range(self):
        self.assertEqual(parse_temperature_bucket("-5 to 0"), (-5, 0))

    def test_negative_or_below(self):
        self.assertEqual(parse_temperature_bucket("-10°F or below"), (-999, -10))

    def test_range_with_to_word(self):
        """'X to Y' should match as range, not false-match 't' and 'o' chars."""
        self.assertEqual(parse_temperature_bucket("50 to 54"), (50, 54))


if __name__ == "__main__":
    unittest.main()
