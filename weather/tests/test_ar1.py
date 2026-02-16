"""Tests for AR(1) forecast error autocorrelation in feedback.py."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from weather.feedback import (
    FeedbackEntry,
    FeedbackState,
    _MIN_AR_SAMPLES,
    _MIN_SAMPLES,
    _season_key,
)


def _make_recent_timestamp() -> str:
    """Return a recent ISO timestamp so decay_factor() is near 1.0."""
    return datetime.now(timezone.utc).isoformat()


def _feed_errors(entry: FeedbackEntry, errors: list[float]) -> None:
    """Feed a list of errors into the entry as forecast=error, actual=0."""
    for err in errors:
        entry.update(forecast_temp=err, actual_temp=0.0)


class TestPhiConvergesPositive:
    """Feed +3, +3, +3... errors (20x) -> phi near 1.0 (but clamped to 0.8)."""

    def test_phi_converges_positive(self):
        entry = FeedbackEntry()
        _feed_errors(entry, [3.0] * 20)
        # All same-sign errors: covariance and variance both positive
        # raw_phi = cov_sum / var_sum should be ~1.0, clamped to 0.8
        assert entry.ar_phi == 0.8
        assert entry.ar_count == 19  # 20 errors, 19 pairs


class TestPhiConvergesNegative:
    """Feed +3, -3, +3, -3... (20x) -> phi near -1.0 (but clamped to -0.8)."""

    def test_phi_converges_negative(self):
        entry = FeedbackEntry()
        errors = [3.0 if i % 2 == 0 else -3.0 for i in range(20)]
        _feed_errors(entry, errors)
        # Alternating signs: cov_sum is negative, var_sum is positive
        # raw_phi should be ~-1.0, clamped to -0.8
        assert entry.ar_phi == -0.8
        assert entry.ar_count == 19


class TestPhiClamped:
    """Extreme sequences stay in [-0.8, 0.8]."""

    def test_phi_clamped_positive(self):
        entry = FeedbackEntry()
        # Very large constant errors
        _feed_errors(entry, [100.0] * 30)
        assert -0.8 <= entry.ar_phi <= 0.8

    def test_phi_clamped_negative(self):
        entry = FeedbackEntry()
        # Large alternating errors
        errors = [1000.0 if i % 2 == 0 else -1000.0 for i in range(30)]
        _feed_errors(entry, errors)
        assert -0.8 <= entry.ar_phi <= 0.8

    def test_phi_clamped_mixed_extreme(self):
        entry = FeedbackEntry()
        # Start with large positives then large negatives
        _feed_errors(entry, [50.0] * 15 + [-50.0] * 15)
        assert -0.8 <= entry.ar_phi <= 0.8


class TestNoCorrectionBelowMinSamples:
    """< 10 AR pairs -> plain bias only (no AR correction)."""

    def test_no_correction_below_min_samples(self):
        state = FeedbackState()
        # Feed exactly _MIN_SAMPLES errors (gives _MIN_SAMPLES - 1 AR pairs < 10)
        for i in range(_MIN_SAMPLES):
            state.record("NYC", 1, 73.0, 70.0)  # constant +3 error

        entry = state.entries["NYC|winter"]
        assert entry.ar_count == _MIN_SAMPLES - 1  # 6 AR pairs
        assert entry.ar_count < _MIN_AR_SAMPLES

        # Verify AR correction is NOT applied by checking the entry directly.
        # With too few AR samples, get_bias should return base_bias only.
        # We verify by computing manually: base_bias = bias_ema * decay
        decay = entry.decay_factor()
        expected_base = entry.bias_ema * decay

        bias_with_ar = state.get_bias("NYC", 1, use_autocorrelation=True)
        assert bias_with_ar is not None
        # The result should be just base_bias (no AR term).
        # Allow tiny float tolerance from time elapsed during the call.
        assert bias_with_ar == pytest.approx(expected_base, rel=1e-6)

        # Also verify the flag makes no difference
        bias_without_ar = state.get_bias("NYC", 1, use_autocorrelation=False)
        assert bias_without_ar is not None
        assert bias_with_ar == pytest.approx(bias_without_ar, rel=1e-6)


class TestNoCorrectionWhenPhiSmall:
    """|phi| < 0.1 -> no AR correction."""

    def test_no_correction_when_phi_small(self):
        state = FeedbackState()
        # Feed errors that produce near-zero phi:
        # Random-looking errors that don't have strong serial correlation
        import random

        rng = random.Random(42)
        errors = [rng.gauss(0, 3) for _ in range(30)]
        for err in errors:
            state.record("NYC", 1, err, 0.0)

        entry = state.entries["NYC|winter"]
        # If phi happens to be small enough (< 0.1), no AR correction
        if abs(entry.ar_phi) < 0.1:
            bias_with_ar = state.get_bias("NYC", 1, use_autocorrelation=True)
            bias_without_ar = state.get_bias("NYC", 1, use_autocorrelation=False)
            assert bias_with_ar == pytest.approx(bias_without_ar, rel=1e-6)
        else:
            # If phi ended up >= 0.1 (unlikely with this seed), we still
            # verify the AR path works without error
            bias_with_ar = state.get_bias("NYC", 1, use_autocorrelation=True)
            assert bias_with_ar is not None

    def test_force_small_phi(self):
        """Directly set small phi and verify no correction."""
        state = FeedbackState()
        # Build up enough samples for bias
        for _ in range(15):
            state.record("NYC", 1, 73.0, 70.0)

        entry = state.entries["NYC|winter"]
        # Force small phi
        entry.ar_phi = 0.05
        entry.ar_count = 20
        entry.last_error = 5.0

        bias_with_ar = state.get_bias("NYC", 1, use_autocorrelation=True)
        bias_without_ar = state.get_bias("NYC", 1, use_autocorrelation=False)
        assert bias_with_ar == pytest.approx(bias_without_ar, rel=1e-6)


class TestBackwardCompatibleLoad:
    """Old JSON without AR fields loads fine."""

    def test_backward_compatible_load(self):
        old_data = {
            "NYC|winter": {
                "bias_ema": 2.5,
                "abs_error_ema": 3.0,
                "sample_count": 15,
                "last_updated": _make_recent_timestamp(),
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_data, f)
            tmp_path = f.name

        try:
            state = FeedbackState.load(tmp_path)
            entry = state.entries["NYC|winter"]
            assert entry.bias_ema == 2.5
            assert entry.abs_error_ema == 3.0
            assert entry.sample_count == 15
            # New AR fields should have defaults
            assert entry.last_error is None
            assert entry.ar_phi == 0.0
            assert entry.cov_sum == 0.0
            assert entry.var_sum == 0.0
            assert entry.ar_count == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestArCorrectionDirection:
    """Positive phi + positive last_error increases bias."""

    def test_ar_correction_direction(self):
        state = FeedbackState()
        # Feed enough constant positive errors to build up data
        for _ in range(25):
            state.record("NYC", 1, 73.0, 70.0)  # +3 error each time

        entry = state.entries["NYC|winter"]
        # With constant positive errors: phi should be clamped to 0.8
        assert entry.ar_phi == 0.8
        assert entry.last_error == 3.0
        assert entry.ar_count >= _MIN_AR_SAMPLES

        bias_with_ar = state.get_bias("NYC", 1, use_autocorrelation=True)
        bias_without_ar = state.get_bias("NYC", 1, use_autocorrelation=False)

        assert bias_with_ar is not None
        assert bias_without_ar is not None
        # Positive phi * positive last_error = positive correction
        # So bias_with_ar > bias_without_ar
        assert bias_with_ar > bias_without_ar


class TestSaveLoadRoundtripWithAr:
    """New fields survive serialization."""

    def test_save_load_roundtrip_with_ar(self):
        state = FeedbackState()
        # Build up AR data
        for _ in range(20):
            state.record("Chicago", 7, 85.0, 82.0)  # +3 error

        entry = state.entries["Chicago|summer"]
        original_phi = entry.ar_phi
        original_cov_sum = entry.cov_sum
        original_var_sum = entry.var_sum
        original_ar_count = entry.ar_count
        original_last_error = entry.last_error

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            state.save(tmp_path)

            loaded = FeedbackState.load(tmp_path)
            loaded_entry = loaded.entries["Chicago|summer"]

            assert loaded_entry.ar_phi == original_phi
            assert loaded_entry.cov_sum == original_cov_sum
            assert loaded_entry.var_sum == original_var_sum
            assert loaded_entry.ar_count == original_ar_count
            assert loaded_entry.last_error == original_last_error
        finally:
            Path(tmp_path).unlink(missing_ok=True)
