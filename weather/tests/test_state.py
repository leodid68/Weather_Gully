"""Tests for weather.state — TradingState persistence and circuit breaker fields."""

import os
import tempfile
import unittest

from weather.state import TradingState


class TestCircuitBreakerState(unittest.TestCase):
    def test_daily_pnl_default_zero(self):
        state = TradingState()
        self.assertEqual(state.get_daily_pnl("2026-02-16"), 0.0)

    def test_record_daily_pnl(self):
        state = TradingState()
        state.record_daily_pnl("2026-02-16", -3.50)
        state.record_daily_pnl("2026-02-16", -2.00)
        self.assertAlmostEqual(state.get_daily_pnl("2026-02-16"), -5.50)

    def test_positions_opened_today(self):
        state = TradingState()
        state.record_position_opened("2026-02-16")
        state.record_position_opened("2026-02-16")
        self.assertEqual(state.positions_opened_today("2026-02-16"), 2)

    def test_circuit_break_timestamp(self):
        state = TradingState()
        self.assertIsNone(state.last_circuit_break)
        state.last_circuit_break = "2026-02-16T12:00:00+00:00"
        self.assertIsNotNone(state.last_circuit_break)

    def test_serialization_roundtrip(self):
        """New fields survive save/load cycle."""
        state = TradingState()
        state.record_daily_pnl("2026-02-16", -5.0)
        state.record_position_opened("2026-02-16")
        state.last_circuit_break = "2026-02-16T12:00:00+00:00"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            loaded = TradingState.load(path)
            self.assertAlmostEqual(loaded.get_daily_pnl("2026-02-16"), -5.0)
            self.assertEqual(loaded.positions_opened_today("2026-02-16"), 1)
            self.assertEqual(loaded.last_circuit_break, "2026-02-16T12:00:00+00:00")
        finally:
            os.unlink(path)

    def test_daily_pnl_separate_days(self):
        """P&L tracking is per-day, not cumulative across days."""
        state = TradingState()
        state.record_daily_pnl("2026-02-15", -1.0)
        state.record_daily_pnl("2026-02-16", -2.0)
        self.assertAlmostEqual(state.get_daily_pnl("2026-02-15"), -1.0)
        self.assertAlmostEqual(state.get_daily_pnl("2026-02-16"), -2.0)

    def test_positions_count_separate_days(self):
        """Position count is per-day."""
        state = TradingState()
        state.record_position_opened("2026-02-15")
        state.record_position_opened("2026-02-16")
        state.record_position_opened("2026-02-16")
        self.assertEqual(state.positions_opened_today("2026-02-15"), 1)
        self.assertEqual(state.positions_opened_today("2026-02-16"), 2)


if __name__ == "__main__":
    unittest.main()
