"""Tests for bot.daemon."""

import contextlib
import os
import signal
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from bot.config import Config
from bot.daemon import (
    _cleanup_pid,
    _interruptible_sleep,
    _write_heartbeat,
    _write_pid,
    check_health,
    run_daemon,
)


class TestPidHeartbeat(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmpdir, "state.json")

    def tearDown(self):
        for ext in (".pid", ".heartbeat"):
            try:
                os.remove(self.state_path + ext)
            except FileNotFoundError:
                pass
        os.rmdir(self.tmpdir)

    def test_write_and_read_pid(self):
        _write_pid(self.state_path)
        pid_text = Path(self.state_path + ".pid").read_text()
        self.assertEqual(int(pid_text), os.getpid())

    def test_write_heartbeat(self):
        _write_heartbeat(self.state_path)
        hb_text = Path(self.state_path + ".heartbeat").read_text()
        hb = datetime.fromisoformat(hb_text)
        age = (datetime.now(timezone.utc) - hb).total_seconds()
        self.assertLess(age, 5)

    def test_cleanup_pid(self):
        _write_pid(self.state_path)
        self.assertTrue(os.path.exists(self.state_path + ".pid"))
        _cleanup_pid(self.state_path)
        self.assertFalse(os.path.exists(self.state_path + ".pid"))

    def test_cleanup_pid_missing(self):
        # Should not raise
        _cleanup_pid(self.state_path)


class TestInterruptibleSleep(unittest.TestCase):
    def test_exits_early_when_check_false(self):
        calls = 0

        def check():
            nonlocal calls
            calls += 1
            return calls < 2  # return False on 2nd check

        start = time.monotonic()
        _interruptible_sleep(10.0, check)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.0)

    def test_sleeps_full_duration(self):
        start = time.monotonic()
        _interruptible_sleep(0.1, lambda: True)
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.09)


class TestCheckHealth(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmpdir, "state.json")

    def tearDown(self):
        for ext in (".pid", ".heartbeat"):
            try:
                os.remove(self.state_path + ext)
            except FileNotFoundError:
                pass
        os.rmdir(self.tmpdir)

    def test_no_pid_file(self):
        ok, msg = check_health(self.state_path)
        self.assertFalse(ok)
        self.assertIn("No PID file", msg)

    def test_stale_pid(self):
        # Write a PID that doesn't exist
        Path(self.state_path + ".pid").write_text("999999999")
        ok, msg = check_health(self.state_path)
        self.assertFalse(ok)
        self.assertIn("not running", msg)

    def test_alive_with_fresh_heartbeat(self):
        _write_pid(self.state_path)
        _write_heartbeat(self.state_path)
        ok, msg = check_health(self.state_path)
        self.assertTrue(ok)
        self.assertIn("alive", msg)

    def test_alive_no_heartbeat(self):
        _write_pid(self.state_path)
        ok, msg = check_health(self.state_path)
        self.assertTrue(ok)
        self.assertIn("no heartbeat yet", msg)

    def test_stale_heartbeat(self):
        _write_pid(self.state_path)
        # Write a heartbeat 10 minutes ago
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        Path(self.state_path + ".heartbeat").write_text(old_time.isoformat())
        ok, msg = check_health(self.state_path)
        self.assertFalse(ok)
        self.assertIn("stale", msg)


class TestRunDaemon(unittest.TestCase):
    def test_graceful_shutdown_on_signal(self):
        """Daemon should stop after receiving SIGINT-like shutdown."""
        config = Config()
        config.run_interval_seconds = 0  # no sleep between runs

        tmpdir = tempfile.mkdtemp()
        state_path = os.path.join(tmpdir, "state.json")

        run_count = 0

        def mock_run_strategy(client, cfg, state, dry_run, sp):
            nonlocal run_count
            run_count += 1
            if run_count >= 2:
                # Simulate shutdown signal by sending SIGINT to self
                os.kill(os.getpid(), signal.SIGINT)

        mock_client = MagicMock()
        mock_client_factory = MagicMock(return_value=mock_client)

        with (
            patch("bot.daemon.run_strategy", side_effect=mock_run_strategy),
            patch("bot.daemon.TradingState.load", return_value=MagicMock()),
            patch("bot.daemon.state_lock"),
        ):
            run_daemon(mock_client_factory, config, state_path, dry_run=True)

        self.assertGreaterEqual(run_count, 2)
        mock_client.close.assert_called_once()

        # Cleanup
        for ext in ("", ".pid", ".heartbeat"):
            try:
                os.remove(state_path + ext)
            except FileNotFoundError:
                pass
        os.rmdir(tmpdir)

    def test_network_error_retry(self):
        """Daemon retries on network errors and recreates client after max attempts."""
        import httpx

        config = Config()
        config.run_interval_seconds = 0
        config.retry_max_attempts = 2
        config.retry_backoff_base = 0.01
        config.retry_backoff_max = 0.01

        tmpdir = tempfile.mkdtemp()
        state_path = os.path.join(tmpdir, "state.json")

        call_count = 0

        def mock_run_strategy(client, cfg, state, dry_run, sp):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise httpx.ConnectError("connection refused")
            # Signal stop after recovery
            os.kill(os.getpid(), signal.SIGINT)

        mock_client = MagicMock()
        mock_client_factory = MagicMock(return_value=mock_client)

        with (
            patch("bot.daemon.run_strategy", side_effect=mock_run_strategy),
            patch("bot.daemon.TradingState.load", return_value=MagicMock()),
            patch("bot.daemon.state_lock"),
            patch("bot.daemon.time.sleep"),
        ):
            run_daemon(mock_client_factory, config, state_path, dry_run=True)

        # Should have recreated client once (after 2 consecutive failures)
        self.assertEqual(mock_client.close.call_count, 2)  # recreate + final close
        self.assertGreaterEqual(call_count, 4)

        # Cleanup
        for ext in ("", ".pid", ".heartbeat"):
            try:
                os.remove(state_path + ext)
            except FileNotFoundError:
                pass
        os.rmdir(tmpdir)

    def test_lock_held_skips_run(self):
        """Daemon skips run when state lock is held."""
        config = Config()
        config.run_interval_seconds = 0

        tmpdir = tempfile.mkdtemp()
        state_path = os.path.join(tmpdir, "state.json")

        call_count = 0

        @contextlib.contextmanager
        def mock_state_lock(sp):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("lock held")
            # Stop on second call
            os.kill(os.getpid(), signal.SIGINT)
            yield

        mock_client = MagicMock()

        with (
            patch("bot.daemon.run_strategy"),
            patch("bot.daemon.TradingState.load", return_value=MagicMock()),
            patch("bot.daemon.state_lock", side_effect=mock_state_lock),
        ):
            run_daemon(lambda: mock_client, config, state_path, dry_run=True)

        self.assertEqual(call_count, 2)

        # Cleanup
        for ext in ("", ".pid", ".heartbeat"):
            try:
                os.remove(state_path + ext)
            except FileNotFoundError:
                pass
        os.rmdir(tmpdir)


class TestConfigDaemonParams(unittest.TestCase):
    def test_default_values(self):
        config = Config()
        self.assertEqual(config.run_interval_seconds, 60)
        self.assertEqual(config.retry_max_attempts, 5)
        self.assertEqual(config.retry_backoff_base, 2.0)
        self.assertEqual(config.retry_backoff_max, 300.0)
        self.assertTrue(config.weather_enabled)


if __name__ == "__main__":
    unittest.main()
