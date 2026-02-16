"""Tests for enriched CLI report."""

import unittest
from weather.report import _format_pnl, _format_position_row, format_report
from weather.state import TradingState


class TestPnlFormatting(unittest.TestCase):
    def test_positive_pnl(self):
        result = _format_pnl(2.15)
        self.assertIn("+$2.15", result)

    def test_negative_pnl(self):
        result = _format_pnl(-1.50)
        self.assertIn("-$1.50", result)

    def test_zero_pnl(self):
        result = _format_pnl(0.0)
        self.assertIn("$0.00", result)


class TestPositionRow(unittest.TestCase):
    def test_basic_row(self):
        row = _format_position_row(
            location="NYC", bucket="48-52F", side="YES",
            entry_price=0.35, unrealized=0.07, days_left=2,
        )
        self.assertIn("NYC", row)
        self.assertIn("48-52F", row)
        self.assertIn("YES", row)


class TestFormatReport(unittest.TestCase):
    def test_empty_state_no_crash(self):
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Weather Gully Report", output)
        self.assertIn("Open Positions", output)

    def test_report_includes_circuit_breaker(self):
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Circuit Breaker", output)

    def test_report_includes_calibration(self):
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Calibration", output)

    def test_report_includes_metrics(self):
        state = TradingState()
        output = format_report(state, trade_log=[])
        self.assertIn("Metrics", output)
