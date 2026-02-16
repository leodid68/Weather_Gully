"""Tests for weather.mean_reversion — price Z-score, sizing, persistence."""

import json
import math
import os
import tempfile

import pytest

from weather.mean_reversion import (
    PriceHistory,
    PriceTracker,
    _MIN_SNAPSHOTS,
    _WINDOW_SIZE,
)


# --------------------------------------------------------------------------
# PriceHistory Z-score
# --------------------------------------------------------------------------

def test_no_zscore_below_min_snapshots():
    """Fewer than _MIN_SNAPSHOTS prices should return None."""
    ph = PriceHistory()
    for i in range(_MIN_SNAPSHOTS - 1):
        ph.add(0.5 + i * 0.01, f"2026-01-01T00:0{i}:00Z")
    assert ph.z_score(0.5) is None


def test_zscore_zero_for_mean_price():
    """When current price equals the window mean, Z-score should be ~0."""
    ph = PriceHistory()
    prices = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
    for i, p in enumerate(prices):
        ph.add(p, f"2026-01-01T00:{i:02d}:00Z")
    mean = sum(prices) / len(prices)
    z = ph.z_score(mean)
    assert z is not None
    assert abs(z) < 0.01, f"Expected ~0, got {z}"


def test_zscore_negative_for_depressed():
    """A price well below the window mean should yield negative Z."""
    ph = PriceHistory()
    # Use slightly varying prices around 0.50 so stddev > 0.001
    prices = [0.48, 0.49, 0.50, 0.51, 0.52, 0.48, 0.49, 0.50, 0.51, 0.52]
    for i, p in enumerate(prices):
        ph.add(p, f"2026-01-01T00:{i:02d}:00Z")
    z = ph.z_score(0.30)
    assert z is not None
    assert z < -1.0, f"Expected z < -1, got {z}"


def test_zscore_positive_for_elevated():
    """A price well above the window mean should yield positive Z."""
    ph = PriceHistory()
    # Use slightly varying prices around 0.30 so stddev > 0.001
    prices = [0.28, 0.29, 0.30, 0.31, 0.32, 0.28, 0.29, 0.30, 0.31, 0.32]
    for i, p in enumerate(prices):
        ph.add(p, f"2026-01-01T00:{i:02d}:00Z")
    z = ph.z_score(0.50)
    assert z is not None
    assert z > 1.0, f"Expected z > 1, got {z}"


def test_zero_stddev_guard():
    """All identical prices should return z=0 (stddev < 0.001 guard)."""
    ph = PriceHistory()
    for i in range(10):
        ph.add(0.42, f"2026-01-01T00:{i:02d}:00Z")
    z = ph.z_score(0.42)
    assert z == 0.0


# --------------------------------------------------------------------------
# PriceTracker sizing
# --------------------------------------------------------------------------

def test_sizing_boost_depressed():
    """A Z-score of -2.0 should yield a multiplier > 1.0."""
    tracker = PriceTracker()
    # Build a history with mean=0.50, stddev ~0.05
    # We need z = (current - mean) / stddev = -2.0
    # Use current_price = mean - 2*stddev = 0.50 - 0.10 = 0.40
    prices = [0.45, 0.50, 0.55, 0.50, 0.45, 0.55, 0.50, 0.45, 0.55, 0.50]
    for i, p in enumerate(prices):
        tracker.record_price("NYC", "2026-02-20", "high", (30, 35), p,
                             timestamp=f"2026-02-19T{i:02d}:00:00Z")
    # Compute mean and stddev to find a depressed price
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / (len(prices) - 1)
    stddev = variance ** 0.5
    depressed_price = mean - 2.0 * stddev  # z = -2.0
    mult = tracker.sizing_multiplier("NYC", "2026-02-20", "high", (30, 35), depressed_price)
    assert mult > 1.0, f"Expected > 1.0, got {mult}"
    assert mult <= 1.5, f"Expected <= 1.5, got {mult}"


def test_sizing_reduction_elevated():
    """A Z-score of +2.0 should yield a multiplier < 1.0."""
    tracker = PriceTracker()
    prices = [0.45, 0.50, 0.55, 0.50, 0.45, 0.55, 0.50, 0.45, 0.55, 0.50]
    for i, p in enumerate(prices):
        tracker.record_price("NYC", "2026-02-20", "high", (30, 35), p,
                             timestamp=f"2026-02-19T{i:02d}:00:00Z")
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / (len(prices) - 1)
    stddev = variance ** 0.5
    elevated_price = mean + 2.0 * stddev  # z = +2.0
    mult = tracker.sizing_multiplier("NYC", "2026-02-20", "high", (30, 35), elevated_price)
    assert mult < 1.0, f"Expected < 1.0, got {mult}"
    assert mult >= 0.5, f"Expected >= 0.5, got {mult}"


def test_sizing_neutral_zone():
    """Z-score in [-1, +1] should yield multiplier = 1.0."""
    tracker = PriceTracker()
    prices = [0.45, 0.50, 0.55, 0.50, 0.45, 0.55, 0.50, 0.45, 0.55, 0.50]
    for i, p in enumerate(prices):
        tracker.record_price("NYC", "2026-02-20", "high", (30, 35), p,
                             timestamp=f"2026-02-19T{i:02d}:00:00Z")
    mean = sum(prices) / len(prices)
    # current_price = mean => z = 0 => neutral
    mult = tracker.sizing_multiplier("NYC", "2026-02-20", "high", (30, 35), mean)
    assert mult == 1.0


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

def test_prune_limits():
    """Tracker with > 200 markets should prune to 200."""
    tracker = PriceTracker()
    # Use unique bucket ranges to guarantee 250 distinct market keys
    for i in range(250):
        tracker.record_price("NYC", "2026-01-15", "high", (i, i + 5), 0.50,
                             timestamp=f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00Z")
    assert len(tracker.histories) == 250
    tracker.prune(max_markets=200)
    assert len(tracker.histories) == 200


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_save_load_roundtrip():
    """Save + load should preserve all data."""
    tracker = PriceTracker()
    prices = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    for i, p in enumerate(prices):
        tracker.record_price("Chicago", "2026-03-01", "low", (20, 25), p,
                             timestamp=f"2026-02-28T{i:02d}:00:00Z")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        tracker.save(tmp_path)
        loaded = PriceTracker.load(tmp_path)
        assert len(loaded.histories) == 1
        key = list(loaded.histories.keys())[0]
        assert loaded.histories[key].prices == prices
        assert len(loaded.histories[key].timestamps) == len(prices)
    finally:
        os.unlink(tmp_path)


# --------------------------------------------------------------------------
# Window trimming
# --------------------------------------------------------------------------

def test_window_trimming():
    """Adding 50 prices should keep only last 40 (2 * _WINDOW_SIZE)."""
    ph = PriceHistory()
    for i in range(50):
        ph.add(0.50 + i * 0.001, f"2026-01-01T00:{i % 60:02d}:{i % 60:02d}Z")
    expected_len = 2 * _WINDOW_SIZE  # 40
    assert len(ph.prices) == expected_len
    assert len(ph.timestamps) == expected_len
    # Should be the last 40 prices
    assert ph.prices[0] == pytest.approx(0.50 + 10 * 0.001)
