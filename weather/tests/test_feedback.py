"""Tests for the feedback loop module."""

import tempfile
import unittest
from pathlib import Path

from weather.feedback import FeedbackEntry, FeedbackState, _season_key


class TestSeasonKey(unittest.TestCase):

    def test_winter_months(self):
        for m in (12, 1, 2):
            self.assertEqual(_season_key("NYC", m), "NYC|winter")

    def test_spring_months(self):
        for m in (3, 4, 5):
            self.assertEqual(_season_key("Chicago", m), "Chicago|spring")

    def test_summer_months(self):
        for m in (6, 7, 8):
            self.assertEqual(_season_key("Miami", m), "Miami|summer")

    def test_fall_months(self):
        for m in (9, 10, 11):
            self.assertEqual(_season_key("Seattle", m), "Seattle|fall")


class TestFeedbackEntry(unittest.TestCase):

    def test_first_update_sets_values(self):
        entry = FeedbackEntry()
        entry.update(forecast_temp=75.0, actual_temp=72.0)
        self.assertAlmostEqual(entry.bias_ema, 3.0)  # 75 - 72
        self.assertAlmostEqual(entry.abs_error_ema, 3.0)
        self.assertEqual(entry.sample_count, 1)

    def test_ema_convergence(self):
        """After many identical errors, EMA should converge to that error."""
        entry = FeedbackEntry()
        for _ in range(50):
            entry.update(forecast_temp=80.0, actual_temp=78.0)
        # Should converge close to +2.0
        self.assertAlmostEqual(entry.bias_ema, 2.0, places=1)
        self.assertAlmostEqual(entry.abs_error_ema, 2.0, places=1)

    def test_ema_negative_bias(self):
        """Underpredicting should give negative bias."""
        entry = FeedbackEntry()
        for _ in range(50):
            entry.update(forecast_temp=60.0, actual_temp=65.0)
        self.assertAlmostEqual(entry.bias_ema, -5.0, places=1)

    def test_has_enough_data(self):
        entry = FeedbackEntry()
        self.assertFalse(entry.has_enough_data)
        for i in range(7):
            entry.update(70.0, 70.0)
        self.assertTrue(entry.has_enough_data)


class TestFeedbackState(unittest.TestCase):

    def test_record_and_get_bias(self):
        state = FeedbackState()
        # Not enough data yet
        self.assertIsNone(state.get_bias("NYC", 1))

        # Feed 7 samples (MIN_SAMPLES)
        for _ in range(7):
            state.record("NYC", 1, 50.0, 47.0)

        bias = state.get_bias("NYC", 1)
        self.assertIsNotNone(bias)
        self.assertGreater(bias, 0)  # overpredict

    def test_different_seasons_independent(self):
        state = FeedbackState()
        for _ in range(10):
            state.record("NYC", 1, 50.0, 47.0)   # winter
            state.record("NYC", 7, 80.0, 82.0)   # summer

        winter_bias = state.get_bias("NYC", 1)
        summer_bias = state.get_bias("NYC", 7)

        self.assertIsNotNone(winter_bias)
        self.assertIsNotNone(summer_bias)
        self.assertGreater(winter_bias, 0)   # overpredict in winter
        self.assertLess(summer_bias, 0)      # underpredict in summer

    def test_min_samples_guard(self):
        """Should not return bias with fewer than MIN_SAMPLES."""
        state = FeedbackState()
        for _ in range(6):  # one fewer than threshold
            state.record("NYC", 1, 50.0, 47.0)
        self.assertIsNone(state.get_bias("NYC", 1))

    def test_save_load_roundtrip(self):
        state = FeedbackState()
        for _ in range(10):
            state.record("NYC", 1, 50.0, 47.0)
            state.record("Chicago", 7, 80.0, 82.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        state.save(path)
        loaded = FeedbackState.load(path)

        # Check entries preserved
        self.assertEqual(set(loaded.entries.keys()), set(state.entries.keys()))

        # Check values match
        for key in state.entries:
            self.assertAlmostEqual(
                loaded.entries[key].bias_ema,
                state.entries[key].bias_ema,
                places=4,
            )
            self.assertEqual(
                loaded.entries[key].sample_count,
                state.entries[key].sample_count,
            )

        Path(path).unlink()

    def test_load_missing_file_returns_empty(self):
        state = FeedbackState.load("/tmp/nonexistent_feedback_test.json")
        self.assertEqual(len(state.entries), 0)

    def test_load_corrupted_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{corrupted!!")
            path = f.name

        state = FeedbackState.load(path)
        self.assertEqual(len(state.entries), 0)

        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
