"""Paper trading loop — runs paper_trade every N seconds with live terminal output."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

INTERVAL = 300  # seconds between runs
STATE_PATH = Path(__file__).parent / "paper_state.json"
SNAPSHOTS_PATH = Path(__file__).parent / "price_snapshots.json"

# Colors
G = "\033[32m"  # green
Y = "\033[33m"  # yellow
C = "\033[36m"  # cyan
R = "\033[31m"  # red
B = "\033[1m"   # bold
D = "\033[0m"   # reset


def _read_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _count_snapshots() -> int:
    try:
        with open(SNAPSHOTS_PATH) as f:
            return len(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _print_positions(state: dict) -> None:
    trades = state.get("trades", {})
    if not trades:
        print(f"  {Y}No open positions{D}")
        return
    total = 0.0
    for mid, t in sorted(trades.items(), key=lambda x: x[1].get("location", "")):
        side = t.get("side", "?")
        loc = t.get("location", "?")
        name = t.get("outcome_name", "?")
        date = t.get("forecast_date", "?")
        cost = t.get("cost_basis", 0) * t.get("shares", 0)
        total += cost
        color = R if side == "no" else G
        print(f"  {color}{side.upper():3s}{D}  {loc:12s} {name:25s} {date}  ${cost:.2f}")
    print(f"  {B}Exposition totale: ${total:.2f}{D}")


def _print_summary(state: dict, snapshots: int, duration: float) -> None:
    trades = state.get("trades", {})
    preds = state.get("predictions", {})
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print()
    print(f"  {C}Positions: {len(trades):3d}  |  Predictions: {len(preds):3d}  |  Snapshots: {snapshots:,d}  |  {now}  |  {duration:.1f}s{D}")


def main() -> None:
    args = [
        sys.executable, "-m", "weather.paper_trade",
        "--set", "entry_threshold=0.30",
        "--no-safeguards",
    ]
    run_count = 0
    snapshots_before = _count_snapshots()

    print(f"\n{B}{C}{'=' * 60}{D}")
    print(f"{B}{C}  PAPER TRADING LOOP — every {INTERVAL}s — Ctrl+C to stop{D}")
    print(f"{B}{C}{'=' * 60}{D}\n")

    while True:
        run_count += 1
        now = datetime.now().strftime("%H:%M:%S")
        print(f"{B}[{now}] Run #{run_count}{D}")
        print(f"{'─' * 60}")

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                capture_output=True, text=True,
                timeout=240,
            )
            duration = time.monotonic() - t0
            output = proc.stderr + proc.stdout

            # Extract key lines for display
            for line in output.splitlines():
                if any(kw in line for kw in (
                    "[PAPER] BUY", "[PAPER] SELL",
                    "Sigma boosted", "Limit cap",
                    "expired position", "Cleaned up",
                    "Summary:", "Mean-reversion",
                )):
                    # Strip timestamp prefix for cleaner display
                    parts = line.split("] ", 1)
                    msg = parts[-1] if len(parts) > 1 else line
                    if "[PAPER] BUY" in line:
                        print(f"  {G}>> {msg}{D}")
                    elif "[PAPER] SELL" in line:
                        print(f"  {R}>> {msg}{D}")
                    elif "Summary:" in line:
                        print(f"  {B}{msg}{D}")
                    elif "expired" in line or "Cleaned" in line:
                        print(f"  {Y}{msg}{D}")
                    else:
                        print(f"  {msg}")

            if proc.returncode != 0:
                print(f"  {R}Exit code {proc.returncode}{D}")

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - t0
            print(f"  {R}Timeout after 240s{D}")
        except Exception as e:
            duration = time.monotonic() - t0
            print(f"  {R}Error: {e}{D}")

        # Show positions and stats
        state = _read_state()
        snapshots_now = _count_snapshots()
        new_snaps = snapshots_now - snapshots_before
        snapshots_before = snapshots_now

        print()
        _print_positions(state)
        _print_summary(state, snapshots_now, duration)
        print(f"  {C}(+{new_snaps} snapshots this run){D}")
        print()

        # Countdown
        try:
            for remaining in range(INTERVAL, 0, -1):
                mins, secs = divmod(remaining, 60)
                print(f"\r  Next run in {mins}m{secs:02d}s  ", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 40 + "\r", end="")
        except KeyboardInterrupt:
            print(f"\n\n{Y}Stopped after {run_count} runs.{D}\n")
            break


if __name__ == "__main__":
    main()
