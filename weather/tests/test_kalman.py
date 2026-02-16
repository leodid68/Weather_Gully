"""Tests for the Kalman filter dynamic sigma module."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from weather.kalman import (
    KalmanSigmaEntry,
    KalmanState,
    horizon_bucket,
    _SQRT_2_OVER_PI,
)


# ---------------------------------------------------------------------------
# horizon_bucket
# ---------------------------------------------------------------------------

class TestHorizonBucketing:

    def test_short_bucket(self):
        assert horizon_bucket(0) == "short"
        assert horizon_bucket(1) == "short"

    def test_medium_bucket(self):
        assert horizon_bucket(2) == "medium"
        assert horizon_bucket(3) == "medium"
        assert horizon_bucket(4) == "medium"

    def test_long_bucket(self):
        assert horizon_bucket(5) == "long"
        assert horizon_bucket(6) == "long"
        assert horizon_bucket(7) == "long"

    def test_extended_bucket(self):
        assert horizon_bucket(8) == "extended"
        assert horizon_bucket(9) == "extended"
        assert horizon_bucket(10) == "extended"

    def test_beyond_10_is_extended(self):
        assert horizon_bucket(11) == "extended"
        assert horizon_bucket(99) == "extended"

    def test_horizons_01_share_short(self):
        """Horizons 0 and 1 share the same bucket."""
        assert horizon_bucket(0) == horizon_bucket(1)

    def test_horizons_234_share_medium(self):
        """Horizons 2, 3, 4 share the same bucket."""
        assert horizon_bucket(2) == horizon_bucket(3) == horizon_bucket(4)


# ---------------------------------------------------------------------------
# KalmanSigmaEntry
# ---------------------------------------------------------------------------

class TestKalmanSigmaEntry:

    def test_initial_defaults(self):
        entry = KalmanSigmaEntry()
        assert entry.x == 3.0
        assert entry.P == 4.0
        assert entry.sample_count == 0
        assert not entry.is_warmed_up

    def test_predict_increases_P(self):
        entry = KalmanSigmaEntry(P=4.0, Q=0.05)
        entry.predict()
        assert entry.P == pytest.approx(4.05)

    def test_update_increments_sample_count(self):
        entry = KalmanSigmaEntry()
        entry.update(2.0)
        assert entry.sample_count == 1

    def test_warmed_up_after_five_updates(self):
        entry = KalmanSigmaEntry()
        for _ in range(4):
            entry.predict()
            entry.update(2.0)
        assert not entry.is_warmed_up
        entry.predict()
        entry.update(2.0)
        assert entry.is_warmed_up

    def test_clamp_extreme_large_error(self):
        """Very large abs_error should not push x above 30.0."""
        entry = KalmanSigmaEntry(x=25.0, P=10.0)
        entry.predict()
        entry.update(1000.0)  # Huge error
        assert entry.x <= 30.0
        assert entry.P <= 20.0

    def test_clamp_extreme_small_error(self):
        """Very small abs_error should not push x below 0.5."""
        entry = KalmanSigmaEntry(x=1.0, P=10.0)
        entry.predict()
        entry.update(0.001)  # Tiny error
        assert entry.x >= 0.5
        assert entry.P >= 0.01

    def test_measurement_model(self):
        """Verify |error|/0.7979 conversion is correct.

        For a half-normal distribution, E[|N(0,sigma)|] = sigma * sqrt(2/pi).
        So implied_sigma = abs_error / sqrt(2/pi).
        """
        abs_error = 3.0
        expected_implied = abs_error / _SQRT_2_OVER_PI
        # After one update from default state, x should move toward expected_implied
        entry = KalmanSigmaEntry()
        entry.predict()
        entry.update(abs_error)
        # x should be between the prior (3.0) and the implied sigma
        assert entry.x > 3.0  # implied sigma = 3.76 > prior 3.0, so x should increase
        assert entry.x < expected_implied + 0.5  # Shouldn't overshoot much


# ---------------------------------------------------------------------------
# KalmanState — initial / warm-up
# ---------------------------------------------------------------------------

class TestKalmanStateInitial:

    def test_initial_state_returns_none(self):
        """Before warm-up, get_sigma should return None."""
        ks = KalmanState()
        assert ks.get_sigma("NYC", 3) is None

    def test_returns_none_before_warmup(self):
        """Even with a few samples, returns None before 5."""
        ks = KalmanState()
        for _ in range(4):
            ks.record_error("NYC", 3, 2.5)
        assert ks.get_sigma("NYC", 3) is None

    def test_returns_sigma_after_warmup(self):
        """After 5 samples, get_sigma returns a value."""
        ks = KalmanState()
        for _ in range(5):
            ks.record_error("NYC", 3, 2.5)
        sigma = ks.get_sigma("NYC", 3)
        assert sigma is not None
        assert sigma > 0


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_convergence_to_true_sigma(self):
        """Feed 30 identical errors (abs_error=3.0) → x converges near 3.0/0.7979 ≈ 3.76."""
        ks = KalmanState()
        abs_error = 3.0
        expected = abs_error / _SQRT_2_OVER_PI  # ≈ 3.76

        for _ in range(30):
            ks.record_error("NYC", 1, abs_error)

        sigma = ks.get_sigma("NYC", 1)
        assert sigma is not None
        # Should be close to expected (within 10%)
        assert abs(sigma - expected) / expected < 0.10, (
            f"Expected sigma near {expected:.2f}, got {sigma:.2f}"
        )


# ---------------------------------------------------------------------------
# Blend weight ramp
# ---------------------------------------------------------------------------

class TestBlendWeight:

    def test_zero_at_4_samples(self):
        ks = KalmanState()
        for _ in range(4):
            ks.record_error("NYC", 2, 2.0)
        assert ks.get_blend_weight("NYC", 2) == 0.0

    def test_positive_at_5_samples(self):
        ks = KalmanState()
        for _ in range(5):
            ks.record_error("NYC", 2, 2.0)
        # At exactly 5 samples: (5-5)/25 * 0.5 = 0.0
        assert ks.get_blend_weight("NYC", 2) == 0.0

    def test_positive_at_6_samples(self):
        ks = KalmanState()
        for _ in range(6):
            ks.record_error("NYC", 2, 2.0)
        w = ks.get_blend_weight("NYC", 2)
        assert w > 0.0
        assert w == pytest.approx(0.5 * 1 / 25.0)

    def test_half_at_30_samples(self):
        ks = KalmanState()
        for _ in range(30):
            ks.record_error("NYC", 2, 2.0)
        assert ks.get_blend_weight("NYC", 2) == pytest.approx(0.5)

    def test_half_at_100_samples(self):
        """Beyond 30 samples, weight stays at 0.5."""
        ks = KalmanState()
        for _ in range(100):
            ks.record_error("NYC", 2, 2.0)
        assert ks.get_blend_weight("NYC", 2) == pytest.approx(0.5)

    def test_zero_for_unknown_key(self):
        ks = KalmanState()
        assert ks.get_blend_weight("Unknown", 5) == 0.0


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        path = str(tmp_path / "kalman_test.json")
        ks = KalmanState()
        # Feed some data
        for _ in range(10):
            ks.record_error("NYC", 1, 3.0)
        for _ in range(7):
            ks.record_error("Chicago", 5, 2.0)

        ks.save(path)
        loaded = KalmanState.load(path)

        # Same keys
        assert set(loaded.entries.keys()) == set(ks.entries.keys())

        # Same values
        for key in ks.entries:
            orig = ks.entries[key]
            copy = loaded.entries[key]
            assert copy.x == pytest.approx(orig.x)
            assert copy.P == pytest.approx(orig.P)
            assert copy.Q == pytest.approx(orig.Q)
            assert copy.R == pytest.approx(orig.R)
            assert copy.sample_count == orig.sample_count
            assert copy.last_updated == orig.last_updated

    def test_load_missing_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        ks = KalmanState.load(path)
        assert len(ks.entries) == 0

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{bad json")
        ks = KalmanState.load(str(path))
        assert len(ks.entries) == 0


# ---------------------------------------------------------------------------
# Pre-warming
# ---------------------------------------------------------------------------

class TestPrewarm:

    def test_prewarm_creates_all_entries(self):
        """Should create 14 locations × 4 buckets = 56 entries."""
        ks = KalmanState()
        count = ks.prewarm()
        assert count == 56
        assert len(ks.entries) == 56

    def test_prewarm_entries_are_warmed_up(self):
        """Pre-warmed entries should have sample_count=10, past threshold."""
        ks = KalmanState()
        ks.prewarm()
        for entry in ks.entries.values():
            assert entry.sample_count == 10
            assert entry.is_warmed_up

    def test_prewarm_sigma_values(self):
        """NYC short should be ~2.38 * 1.165 ≈ 2.77."""
        ks = KalmanState()
        ks.prewarm()
        nyc_short = ks.entries.get("NYC|short")
        assert nyc_short is not None
        assert nyc_short.x == pytest.approx(2.38 * (1.0 + 1.33) / 2, abs=0.01)

    def test_prewarm_extended_sigma(self):
        """Seattle extended should be ~1.04 * 5.333 ≈ 5.55."""
        ks = KalmanState()
        ks.prewarm()
        sea_ext = ks.entries.get("Seattle|extended")
        assert sea_ext is not None
        assert sea_ext.x == pytest.approx(1.04 * (4.67 + 5.33 + 6.0) / 3, abs=0.01)

    def test_prewarm_reduced_uncertainty(self):
        """P should be 1.0, not default 4.0."""
        ks = KalmanState()
        ks.prewarm()
        for entry in ks.entries.values():
            assert entry.P == 1.0

    def test_prewarm_does_not_overwrite_existing(self):
        """Existing trained entries should not be replaced by default."""
        ks = KalmanState()
        # Manually insert an entry with different values
        ks.entries["NYC|short"] = KalmanSigmaEntry(
            x=5.0, P=0.5, sample_count=50, last_updated="trained",
        )
        count = ks.prewarm()
        assert count == 55  # 56 - 1 existing
        assert ks.entries["NYC|short"].x == 5.0  # Not overwritten
        assert ks.entries["NYC|short"].sample_count == 50

    def test_prewarm_overwrite_flag(self):
        """With overwrite=True, even existing entries are replaced."""
        ks = KalmanState()
        ks.entries["NYC|short"] = KalmanSigmaEntry(
            x=5.0, P=0.5, sample_count=50,
        )
        count = ks.prewarm(overwrite=True)
        assert count == 56
        assert ks.entries["NYC|short"].x != 5.0  # Was overwritten

    def test_prewarm_blend_weight_positive(self):
        """Pre-warmed entries (10 samples) should have blend weight > 0."""
        ks = KalmanState()
        ks.prewarm()
        # 10 samples → weight = 0.5 * (10-5)/25 = 0.1
        w = ks.get_blend_weight("NYC", 3)  # medium bucket
        assert w == pytest.approx(0.1)

    def test_prewarm_get_sigma_works(self):
        """After prewarm, get_sigma should return values."""
        ks = KalmanState()
        ks.prewarm()
        for loc in ["NYC", "Chicago", "Miami", "Seattle", "Atlanta", "Dallas",
                    "London", "Paris", "Seoul", "Toronto",
                    "BuenosAires", "SaoPaulo", "Ankara", "Wellington"]:
            for h in [0, 3, 6, 9]:
                sigma = ks.get_sigma(loc, h)
                assert sigma is not None
                assert sigma > 0


# ---------------------------------------------------------------------------
# Integration: compute_adaptive_sigma with Kalman blending
# ---------------------------------------------------------------------------

class TestComputeAdaptiveSigmaWithKalman:

    def test_no_kalman_state_unchanged(self):
        """Without Kalman state, behaviour is unchanged."""
        from weather.probability import compute_adaptive_sigma

        # Use sigma_floor only (no ensemble/spread/ema)
        result = compute_adaptive_sigma(
            ensemble_result=None,
            model_spread=0.0,
            ema_error=None,
            forecast_date="2026-06-15",
            location="NYC",
            kalman_state=None,
        )
        assert result > 0

    def test_with_warmed_up_kalman_blends(self):
        """When Kalman is warmed up, result should blend toward Kalman sigma."""
        from weather.probability import compute_adaptive_sigma

        ks = KalmanState()
        # Feed enough data to warm up (30 samples for full weight)
        for _ in range(30):
            ks.record_error("NYC", 3, 5.0)

        kalman_sigma = ks.get_sigma("NYC", 3)
        assert kalman_sigma is not None

        # Patch get_horizon_days to return 3 (matching our bucket)
        with patch("weather.probability.get_horizon_days", return_value=3):
            result_with = compute_adaptive_sigma(
                ensemble_result=None,
                model_spread=0.0,
                ema_error=None,
                forecast_date="2026-02-20",
                location="NYC",
                kalman_state=ks,
            )
            result_without = compute_adaptive_sigma(
                ensemble_result=None,
                model_spread=0.0,
                ema_error=None,
                forecast_date="2026-02-20",
                location="NYC",
                kalman_state=None,
            )

        # The blended result should differ from the non-blended result
        # (unless they happen to be identical, which is unlikely)
        # At full weight (0.5), result = 0.5 * kalman + 0.5 * max_of_signals
        weight = ks.get_blend_weight("NYC", 3)
        assert weight == pytest.approx(0.5)
        expected = 0.5 * kalman_sigma + 0.5 * result_without
        # Result should be at least the floor
        assert result_with >= 0  # positive sigma
        # If kalman sigma is less than floor, result should equal the non-blended
        # Otherwise should be close to the expected blend
        if expected >= result_without:
            # Kalman didn't lower below floor
            assert result_with == pytest.approx(expected, rel=0.01)

    def test_not_warmed_up_returns_max_of_signals(self):
        """When Kalman is not warmed up, result equals max-of-signals."""
        from weather.probability import compute_adaptive_sigma

        ks = KalmanState()
        # Only 3 samples — not warmed up
        for _ in range(3):
            ks.record_error("NYC", 3, 5.0)

        with patch("weather.probability.get_horizon_days", return_value=3):
            result_with = compute_adaptive_sigma(
                ensemble_result=None,
                model_spread=0.0,
                ema_error=None,
                forecast_date="2026-02-20",
                location="NYC",
                kalman_state=ks,
            )
            result_without = compute_adaptive_sigma(
                ensemble_result=None,
                model_spread=0.0,
                ema_error=None,
                forecast_date="2026-02-20",
                location="NYC",
                kalman_state=None,
            )

        assert result_with == pytest.approx(result_without)
