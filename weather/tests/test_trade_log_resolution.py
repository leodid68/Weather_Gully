"""Tests for trade_log resolution via METAR observations.

Covers resolve_trades() with various bucket types and the actuals dict
transformation from TradingState.daily_observations format.
"""

import json
import os
import tempfile
import unittest

from weather.trade_log import load_trade_log, log_trade, resolve_trades


class TestResolveTradesWin(unittest.TestCase):
    """Actual temp inside bucket => outcome=1, pnl = 1 - price."""

    def test_resolve_trades_win(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-17", metric="high",
                bucket=(42, 43), prob_raw=0.30, prob_platt=0.32,
                market_price=0.40, position_usd=2.0, shares=5.0,
                forecast_temp=42.5, path=path,
            )
            actuals = {"2026-02-17": {"high": 42.5}}
            resolved = resolve_trades(actuals, path=path)
            self.assertEqual(resolved, 1)

            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)
            self.assertAlmostEqual(entries[0]["pnl"], 1.0 - 0.40, places=4)
            self.assertAlmostEqual(entries[0]["actual_temp"], 42.5)


class TestResolveTradesLoss(unittest.TestCase):
    """Actual temp outside bucket => outcome=0, pnl = -price."""

    def test_resolve_trades_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-17", metric="high",
                bucket=(42, 43), prob_raw=0.30, prob_platt=0.32,
                market_price=0.40, position_usd=2.0, shares=5.0,
                forecast_temp=42.5, path=path,
            )
            actuals = {"2026-02-17": {"high": 45.0}}
            resolved = resolve_trades(actuals, path=path)
            self.assertEqual(resolved, 1)

            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 0)
            self.assertAlmostEqual(entries[0]["pnl"], -0.40, places=4)
            self.assertAlmostEqual(entries[0]["actual_temp"], 45.0)


class TestResolveSentinelBelow(unittest.TestCase):
    """Lower-open bucket [-999, hi]: actual <= hi => outcome=1."""

    def test_resolve_sentinel_below(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-17", metric="high",
                bucket=(-999, 41), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=38.0, path=path,
            )
            actuals = {"2026-02-17": {"high": 39.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)  # 39 <= 41


class TestResolveSentinelAbove(unittest.TestCase):
    """Upper-open bucket [lo, 999]: actual >= lo => outcome=1."""

    def test_resolve_sentinel_above(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-17", metric="high",
                bucket=(46, 999), prob_raw=0.10, prob_platt=0.12,
                market_price=0.05, position_usd=1.0, shares=20.0,
                forecast_temp=48.0, path=path,
            )
            actuals = {"2026-02-17": {"high": 50.0}}
            resolve_trades(actuals, path=path)
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)  # 50 >= 46


class TestResolveSkipsAlreadyResolved(unittest.TestCase):
    """Trades with outcome already set should not be re-resolved."""

    def test_resolve_skips_already_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.json")
            log_trade(
                location="NYC", date="2026-02-17", metric="high",
                bucket=(42, 43), prob_raw=0.30, prob_platt=0.32,
                market_price=0.40, position_usd=2.0, shares=5.0,
                forecast_temp=42.5, path=path,
            )
            # First resolve
            actuals = {"2026-02-17": {"high": 42.5}}
            resolved1 = resolve_trades(actuals, path=path)
            self.assertEqual(resolved1, 1)

            # Second call should resolve 0
            resolved2 = resolve_trades(actuals, path=path)
            self.assertEqual(resolved2, 0)

            # Outcome unchanged
            entries = load_trade_log(path)
            self.assertEqual(entries[0]["outcome"], 1)


class TestBuildActualsFromDailyObs(unittest.TestCase):
    """Verify the dict transformation from daily_observations format to actuals."""

    def test_build_actuals_from_daily_obs(self):
        # Simulate state.daily_observations keyed as "location|date"
        daily_observations = {
            "NYC|2026-02-17": {"obs_high": 43.0, "obs_low": 28.0},
            "Chicago|2026-02-17": {"obs_high": 35.0},
            "NYC|2026-02-18": {"obs_low": 22.5},
        }

        # Replicate the transformation logic from paper_trade._async_main
        actuals: dict[str, dict[str, float]] = {}
        for key, obs in daily_observations.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            _, date_str = parts
            for metric_name in ("high", "low"):
                temp = obs.get(f"obs_{metric_name}")
                if temp is not None:
                    actuals.setdefault(date_str, {})[metric_name] = temp

        # Verify: both dates present
        self.assertIn("2026-02-17", actuals)
        self.assertIn("2026-02-18", actuals)
        # 2026-02-17 has both high and low (Chicago overwrites NYC high)
        self.assertIn("high", actuals["2026-02-17"])
        self.assertEqual(actuals["2026-02-17"]["low"], 28.0)
        # 2026-02-18 only has low from NYC
        self.assertIn("low", actuals["2026-02-18"])
        self.assertEqual(actuals["2026-02-18"]["low"], 22.5)
        self.assertNotIn("high", actuals["2026-02-18"])

    def test_build_actuals_skips_malformed_keys(self):
        """Keys without '|' separator are silently skipped."""
        daily_observations = {
            "bad_key_no_pipe": {"obs_high": 50.0},
            "NYC|2026-02-17": {"obs_high": 43.0},
        }
        actuals: dict[str, dict[str, float]] = {}
        for key, obs in daily_observations.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            _, date_str = parts
            for metric_name in ("high", "low"):
                temp = obs.get(f"obs_{metric_name}")
                if temp is not None:
                    actuals.setdefault(date_str, {})[metric_name] = temp

        self.assertEqual(len(actuals), 1)
        self.assertIn("2026-02-17", actuals)

    def test_build_actuals_multiple_locations_merge(self):
        """Multiple locations for the same date: last write wins per metric."""
        daily_observations = {
            "NYC|2026-02-17": {"obs_high": 43.0},
            "Chicago|2026-02-17": {"obs_high": 35.0},
        }
        actuals: dict[str, dict[str, float]] = {}
        for key, obs in daily_observations.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            _, date_str = parts
            for metric_name in ("high", "low"):
                temp = obs.get(f"obs_{metric_name}")
                if temp is not None:
                    actuals.setdefault(date_str, {})[metric_name] = temp

        # Chicago's high (35.0) overwrites NYC's high (43.0) for same date
        self.assertEqual(actuals["2026-02-17"]["high"], 35.0)


if __name__ == "__main__":
    unittest.main()
