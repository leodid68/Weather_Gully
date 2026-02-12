"""Daemon mode — continuous strategy loop with retry, graceful shutdown, and health check."""

import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import Config
from .state import TradingState, state_lock
from .strategy import run_strategy

logger = logging.getLogger(__name__)


def run_daemon(client_factory, config: Config, state_path: str, dry_run: bool = True):
    """Main daemon loop — runs strategy on interval with retry + graceful shutdown."""
    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        running = False
        logger.info("Received signal %s — shutting down after current run", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _write_pid(state_path)
    logger.info(
        "Daemon started (pid=%d, interval=%ds, dry_run=%s)",
        os.getpid(), config.run_interval_seconds, dry_run,
    )

    client = client_factory()
    consecutive_failures = 0

    while running:
        run_start = time.monotonic()
        try:
            state = TradingState.load(state_path)
            try:
                lock_ctx = state_lock(state_path)
                lock_ctx.__enter__()
            except OSError:
                logger.warning("State lock held — skipping this run")
                continue
            try:
                run_strategy(client, config, state, dry_run, state_path)
            finally:
                lock_ctx.__exit__(None, None, None)
            consecutive_failures = 0
            _write_heartbeat(state_path)

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            consecutive_failures += 1
            delay = min(config.retry_backoff_base ** consecutive_failures, config.retry_backoff_max)
            logger.error(
                "Run failed (attempt %d): %s — backing off %.0fs",
                consecutive_failures, exc, delay,
            )
            if consecutive_failures >= config.retry_max_attempts:
                logger.error("Max retries reached — recreating client")
                client.close()
                client = client_factory()
                consecutive_failures = 0
            time.sleep(delay)
            continue

        except Exception as exc:
            consecutive_failures += 1
            logger.exception("Unexpected error in run: %s", exc)
            if consecutive_failures >= config.retry_max_attempts * 2:
                logger.critical("Too many failures — exiting")
                break

        # Sleep until next interval
        elapsed = time.monotonic() - run_start
        sleep_time = max(0, config.run_interval_seconds - elapsed)
        if sleep_time > 0 and running:
            _interruptible_sleep(sleep_time, lambda: running)

    client.close()
    _cleanup_pid(state_path)
    logger.info("Daemon stopped cleanly")


# ── Utilities ─────────────────────────────────────────────────────────


def _write_pid(state_path: str) -> None:
    """Write current PID to state_path + '.pid'."""
    pid_path = state_path + ".pid"
    Path(pid_path).write_text(str(os.getpid()))


def _write_heartbeat(state_path: str) -> None:
    """Write ISO timestamp to state_path + '.heartbeat'."""
    hb_path = state_path + ".heartbeat"
    Path(hb_path).write_text(datetime.now(timezone.utc).isoformat())


def _cleanup_pid(state_path: str) -> None:
    """Remove PID file."""
    pid_path = state_path + ".pid"
    try:
        os.remove(pid_path)
    except FileNotFoundError:
        pass


def _interruptible_sleep(seconds: float, check_fn) -> None:
    """Sleep in 1-second increments, checking check_fn() between each."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not check_fn():
            return
        time.sleep(max(0, min(1.0, end - time.monotonic())))


def check_health(state_path: str) -> tuple[bool, str]:
    """Check daemon health: PID alive + heartbeat < 5 min.

    Returns (ok, message) for CLI / monitoring.
    """
    pid_path = state_path + ".pid"
    hb_path = state_path + ".heartbeat"

    # Check PID file
    if not os.path.exists(pid_path):
        return False, "No PID file — daemon not running"

    try:
        pid = int(Path(pid_path).read_text().strip())
    except (ValueError, OSError):
        return False, "Invalid PID file"

    # Check if process is alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, f"PID {pid} not running (stale PID file)"
    except PermissionError:
        pass  # process exists but we can't signal it — still alive

    # Check heartbeat
    if not os.path.exists(hb_path):
        return True, f"PID {pid} alive, no heartbeat yet"

    try:
        hb_text = Path(hb_path).read_text().strip()
        hb_time = datetime.fromisoformat(hb_text)
        age = (datetime.now(timezone.utc) - hb_time).total_seconds()
        if age > 300:
            return False, f"PID {pid} alive, heartbeat stale ({age:.0f}s ago)"
        return True, f"PID {pid} alive, last heartbeat {age:.0f}s ago"
    except (ValueError, OSError):
        return True, f"PID {pid} alive, heartbeat unreadable"
