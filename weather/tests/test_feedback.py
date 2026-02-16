"""Tests for the feedback loop module."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather.feedback import (
    FeedbackEntry,
    FeedbackState,
    _HALF_LIFE_DAYS,
    _season_key,
)


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


class TestGetAbsErrorEma(unittest.TestCase):

    def test_returns_ema_when_enough_data(self):
        state = FeedbackState()
        for i in range(10):
            state.record("NYC", 1, 50.0, 50.0 + (i % 3))
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNotNone(ema)
        self.assertGreater(ema, 0)

    def test_returns_none_when_no_data(self):
        state = FeedbackState()
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNone(ema)

    def test_returns_none_when_too_few_samples(self):
        state = FeedbackState()
        state.record("NYC", 1, 50.0, 52.0)
        state.record("NYC", 1, 50.0, 48.0)
        ema = state.get_abs_error_ema("NYC", 1)
        self.assertIsNone(ema)


class TestFeedbackDecay(unittest.TestCase):

    def test_recent_entry_minimal_decay(self):
        """Entry updated just now should have decay_factor close to 1.0."""
        entry = FeedbackEntry(bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
                              last_updated=datetime.now(timezone.utc).isoformat())
        self.assertAlmostEqual(entry.decay_factor(), 1.0, delta=0.01)

    def test_half_life_decay(self):
        """Entry updated 30 days ago should have decay_factor ~0.5."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        entry = FeedbackEntry(bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
                              last_updated=old_time)
        self.assertAlmostEqual(entry.decay_factor(), 0.5, delta=0.05)

    def test_stale_entry_below_floor(self):
        """Entry updated 100+ days ago should have decay < floor."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        entry = FeedbackEntry(bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
                              last_updated=old_time)
        self.assertLess(entry.decay_factor(), 0.1)

    def test_stale_get_bias_returns_none(self):
        """FeedbackState.get_bias should return None for very old entries."""
        state = FeedbackState()
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        state.entries["NYC|winter"] = FeedbackEntry(
            bias_ema=2.0, abs_error_ema=3.0, sample_count=10, last_updated=old_time)
        self.assertIsNone(state.get_bias("NYC", 1))

    def test_recent_get_bias_returns_value(self):
        """FeedbackState.get_bias should return decayed value for recent entries."""
        state = FeedbackState()
        state.entries["NYC|winter"] = FeedbackEntry(
            bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
            last_updated=datetime.now(timezone.utc).isoformat())
        result = state.get_bias("NYC", 1)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, delta=0.1)

    def test_stale_get_abs_error_ema_returns_none(self):
        """FeedbackState.get_abs_error_ema should return None for very old entries."""
        state = FeedbackState()
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        state.entries["NYC|winter"] = FeedbackEntry(
            bias_ema=2.0, abs_error_ema=3.0, sample_count=10, last_updated=old_time)
        self.assertIsNone(state.get_abs_error_ema("NYC", 1))

    def test_recent_get_abs_error_ema_returns_value(self):
        """FeedbackState.get_abs_error_ema should return decayed value for recent entries."""
        state = FeedbackState()
        state.entries["NYC|winter"] = FeedbackEntry(
            bias_ema=2.0, abs_error_ema=3.0, sample_count=10,
            last_updated=datetime.now(timezone.utc).isoformat())
        result = state.get_abs_error_ema("NYC", 1)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 3.0, delta=0.1)

    def test_empty_last_updated_returns_zero_decay(self):
        """Entry with no timestamp should have decay_factor 0."""
        entry = FeedbackEntry(bias_ema=2.0, sample_count=10)
        self.assertEqual(entry.decay_factor(), 0.0)

    def test_update_sets_last_updated(self):
        """Calling update should set last_updated."""
        entry = FeedbackEntry()
        entry.update(70.0, 68.0)
        self.assertTrue(len(entry.last_updated) > 0)

    def test_serialization_roundtrip(self):
        """last_updated field should survive save/load."""
        state = FeedbackState()
        state.record("NYC", 1, 70.0, 68.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            loaded = FeedbackState.load(path)
            entry = loaded.entries.get("NYC|winter")
            self.assertIsNotNone(entry)
            self.assertTrue(len(entry.last_updated) > 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
