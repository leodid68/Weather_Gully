"""Tests for weather.trade_log — structured trade logging."""

import json
import os
import tempfile
import unittest

from weather.trade_log import load_trade_log, log_trade, resolve_trades


class TestLogTrade(unittest.TestCase):
    def test_creates_file_and_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(50, 54), prob_raw=0.35, prob_platt=0.38,
                market_price=0.25, position_usd=2.0, shares=8.0,
                forecast_temp=52.0, horizon=3, path=path,
            )
            entries = load_trade_log(path)
            self.assertEqual(len(entries), 1)
            e = entries[0]
            self.assertEqual(e["location"], "NYC")
            self.assertEqual(e["bucket"], [50, 54])
            self.assertAlmostEqual(e["prob_raw"], 0.35, places=4)
            self.assertAlmostEqual(e["prob_platt"], 0.38, places=4)
            self.assertAlmostEqual(e["edge_predicted"], 0.13, places=4)
            self.assertIsNone(e["outcome"])

            # Append second trade
            log_trade(
                location="Chicago", date="2026-02-15", metric="low",
                bucket=(30, 34), prob_raw=0.20, prob_platt=0.22,
                market_price=0.10, position_usd=1.5, shares=15.0,
                forecast_temp=32.0, path=path,
            )
            entries = load_trade_log(path)
            self.assertEqual(len(entries), 2)

    def test_load_missing_file(self):
        entries = load_trade_log("/nonexistent/path.json")
        self.assertEqual(entries, [])


class TestResolveTrades(unittest.TestCase):
    def test_resolves_winning_and_losing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            # Trade 1: bucket [50, 54], actual 52 → win
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(50, 54), prob_raw=0.35, prob_platt=0.38,
                market_price=0.25, position_usd=2.0, shares=8.0,
                forecast_temp=52.0, path=path,
            )
            # Trade 2: bucket [55, 59], actual 52 → lose
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(55, 59), prob_raw=0.15, prob_platt=0.18,
                market_price=0.08, position_usd=1.0, shares=12.5,
                forecast_temp=52.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 52.0, "low": 35.0}}
            resolved = resolve_trades(actuals, path=path)
            self.assertEqual(resolved, 2)

            entries = load_trade_log(path)
            # Trade 1: won
            self.assertEqual(entries[0]["outcome"], 1)
            self.assertAlmostEqual(entries[0]["pnl"], 0.75, places=4)
            self.assertAlmostEqual(entries[0]["actual_temp"], 52.0)
            # Trade 2: lost
            self.assertEqual(entries[1]["outcome"], 0)
            self.assertAlmostEqual(entries[1]["pnl"], -0.08, places=4)

    def test_skips_already_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(50, 54), prob_raw=0.35, prob_platt=0.38,
                market_price=0.25, position_usd=2.0, shares=8.0,
                forecast_temp=52.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 52.0}}
            resolve_trades(actuals, path=path)
            # Second call should resolve 0
            resolved = resolve_trades(actuals, path=path)
            self.assertEqual(resolved, 0)

    def test_skips_missing_actuals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(50, 54), prob_raw=0.35, prob_platt=0.38,
                market_price=0.25, position_usd=2.0, shares=8.0,
                forecast_temp=52.0, path=path,
            )
            # No actuals for this date
            resolved = resolve_trades({}, path=path)
            self.assertEqual(resolved, 0)


class TestOpenEndedBuckets(unittest.TestCase):
    """Ensure open-ended buckets resolve correctly."""

    def test_lower_open_ended_bucket_win(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(-999, 54), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=50.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 52.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)  # 52 <= 54

    def test_lower_open_ended_bucket_lose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(-999, 54), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=50.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 60.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 0)  # 60 > 54

    def test_upper_open_ended_bucket_win(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(55, 999), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=58.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 60.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)  # 60 >= 55

    def test_upper_open_ended_bucket_lose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-15", metric="high",
                bucket=(55, 999), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=58.0, path=path,
            )
            actuals = {"2026-02-15": {"high": 50.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 0)  # 50 < 55


if __name__ == "__main__":
    unittest.main()
