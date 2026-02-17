"""Tests for cross-temporal arbitrage signal detection."""
import pytest
from weather.strategy import detect_cross_temporal_signals


class TestCrossTemporalSignals:
    def _make_market(self, event_name, outcome_name, best_ask):
        return {
            "event_name": event_name,
            "outcome_name": outcome_name,
            "best_ask": best_ask,
            "external_price_yes": best_ask,
        }

    def test_detects_overpriced_far_date(self):
        """Far date priced nearly same as near -> signal."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
            "e2": [self._make_market("NYC High Temp on February 19", "42-43", 0.28)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 1
        assert signals[0]["ratio"] == pytest.approx(0.28 / 0.30, abs=0.01)
        assert signals[0]["date_near"] == "2026-02-18"
        assert signals[0]["date_far"] == "2026-02-19"

    def test_skips_low_price_near(self):
        """Near date price < $0.05 -> skip."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.03)],
            "e2": [self._make_market("NYC High Temp on February 19", "42-43", 0.02)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0

    def test_no_signal_when_far_much_cheaper(self):
        """Far date < 85% of near -> no signal (expected decay)."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
            "e2": [self._make_market("NYC High Temp on February 19", "42-43", 0.15)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0  # ratio 0.5 < 0.85

    def test_skips_low_price_far(self):
        """Far date price < $0.10 -> skip even if ratio high."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.10)],
            "e2": [self._make_market("NYC High Temp on February 19", "42-43", 0.09)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0  # far price 0.09 < 0.10

    def test_multiple_dates_generates_pairs(self):
        """Three dates -> checks consecutive pairs."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
            "e2": [self._make_market("NYC High Temp on February 19", "42-43", 0.28)],
            "e3": [self._make_market("NYC High Temp on February 20", "42-43", 0.25)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        # 18->19 ratio=0.93 > 0.85 -> signal
        # 19->20 ratio=0.89 > 0.85 -> signal
        assert len(signals) == 2

    def test_ignores_different_location(self):
        """Markets for different location -> no cross-temporal."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
            "e2": [self._make_market("Chicago High Temp on February 19", "42-43", 0.28)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0

    def test_ignores_different_metric(self):
        """High vs low -> separate, no cross-temporal."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
            "e2": [self._make_market("NYC Low Temp on February 19", "42-43", 0.28)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0

    def test_empty_events(self):
        """No events -> no signals."""
        signals = detect_cross_temporal_signals({}, "NYC", "high")
        assert len(signals) == 0

    def test_single_date_no_signal(self):
        """Only one date per bucket -> no cross-temporal comparison."""
        events = {
            "e1": [self._make_market("NYC High Temp on February 18", "42-43", 0.30)],
        }
        signals = detect_cross_temporal_signals(events, "NYC", "high")
        assert len(signals) == 0
