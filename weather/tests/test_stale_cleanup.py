"""Tests for stale position auto-cleanup in strategy."""
import unittest
from datetime import datetime, timezone

from weather.state import TradingState


class TestStalePositionCleanup(unittest.TestCase):
    """Test that expired positions are correctly identified and removed."""

    def _make_state_with_trades(self, dates: list[str]) -> TradingState:
        state = TradingState()
        for i, d in enumerate(dates):
            mid = f"market-{i}"
            state.record_trade(
                market_id=mid,
                outcome_name=f"Bucket {i}",
                side="yes",
                cost_basis=0.10,
                shares=10.0,
                location="NYC",
                forecast_date=d,
                forecast_temp=45.0,
                event_id=f"event-{i}",
            )
            state.record_event_position(f"event-{i}", mid)
        return state

    def test_expired_positions_removed(self):
        """Positions with forecast_date < today should be removed."""
        state = self._make_state_with_trades(["2026-02-15", "2026-02-17", "2026-02-19"])
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        for mid in stale_ids:
            trade = state.trades[mid]
            if trade.event_id:
                state.remove_event_position_market(trade.event_id, mid)
            state.remove_trade(mid)

        self.assertEqual(len(state.trades), 2)
        self.assertNotIn("market-0", state.trades)
        self.assertIn("market-1", state.trades)
        self.assertIn("market-2", state.trades)
        self.assertNotIn("event-0", state.event_positions)

    def test_no_stale_positions(self):
        """No crash when all positions are current."""
        state = self._make_state_with_trades(["2099-12-31"])
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        self.assertEqual(len(stale_ids), 0)
        self.assertEqual(len(state.trades), 1)

    def test_empty_forecast_date_not_removed(self):
        """Positions without forecast_date should NOT be removed."""
        state = TradingState()
        state.record_trade(
            market_id="m1", outcome_name="Bucket", side="yes",
            cost_basis=0.10, shares=10.0, forecast_date="",
        )
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        self.assertEqual(len(stale_ids), 0)

    def test_all_expired_clears_state(self):
        """If all positions are expired, state.trades should be empty."""
        state = self._make_state_with_trades(["2026-01-01", "2026-01-15"])
        today_str = "2026-02-17"
        stale_ids = [
            mid for mid, t in state.trades.items()
            if t.forecast_date and t.forecast_date < today_str
        ]
        for mid in stale_ids:
            trade = state.trades[mid]
            if trade.event_id:
                state.remove_event_position_market(trade.event_id, mid)
            state.remove_trade(mid)
        self.assertEqual(len(state.trades), 0)
        self.assertEqual(len(state.event_positions), 0)
