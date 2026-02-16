"""CLI report -- enriched trading dashboard with positions, P&L, metrics, calibration.

Usage: python3 -m weather.report [--state-file PATH] [--trade-log PATH] [--watch N]
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import brier_score, calibration_table, sharpe_ratio, win_rate, average_edge
from .state import TradingState

# ANSI color codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _format_pnl(value: float) -> str:
    """Format P&L with color: green for positive, red for negative."""
    if value > 0:
        return f"{_GREEN}+${value:.2f}{_RESET}"
    elif value < 0:
        return f"{_RED}-${abs(value):.2f}{_RESET}"
    return "$0.00"


def _format_position_row(
    location: str,
    bucket: str,
    side: str,
    entry_price: float,
    unrealized: float,
    days_left: int,
) -> str:
    """Format a single position row for the report."""
    pnl_str = _format_pnl(unrealized)
    return f"  {location:<8s} {bucket:<10s} {side:<4s} ${entry_price:.2f}  {pnl_str}  ({days_left}d left)"


def _load_trade_log(path: str = "weather/trade_log.jsonl") -> list[dict]:
    """Load JSONL trade log."""
    entries = []
    p = Path(path)
    if not p.exists():
        return entries
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _get_calibration_age() -> tuple[int | None, str]:
    """Get calibration.json age in days and status string."""
    cal_path = Path(__file__).parent / "calibration.json"
    if not cal_path.exists():
        return None, f"{_RED}MISSING{_RESET}"
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        gen = cal.get("metadata", {}).get("generated") or cal.get("_metadata", {}).get("generated_at")
        if gen:
            gen_dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - gen_dt).days
            if age > 90:
                return age, f"{_RED}{age}d — STALE{_RESET}"
            elif age > 30:
                return age, f"{_YELLOW}{age}d — consider recalibrating{_RESET}"
            return age, f"{_GREEN}{age}d — OK{_RESET}"
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    return None, f"{_YELLOW}unknown{_RESET}"


def format_report(
    state: TradingState,
    trade_log: list[dict] | None = None,
) -> str:
    """Generate the full report as a string."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    lines = []

    lines.append(f"\n{_BOLD}{'=' * 55}{_RESET}")
    lines.append(f"{_BOLD}  Weather Gully Report{_RESET}")
    lines.append(f"  {_DIM}Last update: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}{_RESET}")
    lines.append(f"{_BOLD}{'=' * 55}{_RESET}")

    # Open Positions
    lines.append(f"\n{_BOLD}-- Open Positions ({len(state.trades)}) --{_RESET}")
    if not state.trades:
        lines.append(f"  {_DIM}No open positions{_RESET}")
    else:
        for mid, trade in state.trades.items():
            days_left = 0
            if trade.forecast_date:
                try:
                    end = datetime.strptime(trade.forecast_date, "%Y-%m-%d").date()
                    days_left = max(0, (end - now.date()).days)
                except ValueError:
                    pass
            lines.append(_format_position_row(
                location=trade.location or "?",
                bucket=trade.outcome_name[:12],
                side=trade.side.upper(),
                entry_price=trade.cost_basis,
                unrealized=0.0,
                days_left=days_left,
            ))

    # Today's P&L
    daily_pnl = state.get_daily_pnl(today)
    lines.append(f"\n{_BOLD}-- Today's P&L --{_RESET}")
    lines.append(f"  Realized: {_format_pnl(daily_pnl)}")

    # Metrics
    resolved = [p for p in state.predictions.values() if p.resolved and p.actual_outcome is not None]
    lines.append(f"\n{_BOLD}-- Metrics ({len(resolved)} resolved) --{_RESET}")
    if resolved:
        preds = [(p.our_probability, p.actual_outcome) for p in resolved]
        bs = brier_score(preds)
        wr = win_rate([1.0 if o else -1.0 for _, o in preds])
        lines.append(f"  Brier: {bs:.4f}" if bs is not None else "  Brier: N/A")
        lines.append(f"  Win rate: {wr:.0%}" if wr is not None else "  Win rate: N/A")

        if trade_log:
            edges = []
            for entry in trade_log:
                prob = entry.get("prob_platt", 0)
                price = entry.get("market_price", 0)
                if price > 0:
                    edges.append(prob - price)
            ae = average_edge(edges)
            if ae is not None:
                lines.append(f"  Avg edge: {ae:.1%}")
    else:
        lines.append(f"  {_DIM}No resolved predictions yet{_RESET}")

    # Calibration
    age, status = _get_calibration_age()
    lines.append(f"\n{_BOLD}-- Calibration --{_RESET}")
    lines.append(f"  Age: {status}")

    # Circuit Breaker
    lines.append(f"\n{_BOLD}-- Circuit Breaker --{_RESET}")
    positions_today = state.positions_opened_today(today)
    lines.append(f"  Daily loss: {_format_pnl(daily_pnl)} / $10.00")
    lines.append(f"  Positions today: {positions_today} / 20")
    if state.last_circuit_break:
        lines.append(f"  Last break: {state.last_circuit_break}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Weather Gully trading report")
    parser.add_argument("--state-file", default="weather/paper_state.json")
    parser.add_argument("--trade-log", default="weather/trade_log.jsonl")
    parser.add_argument("--watch", type=int, default=0, metavar="N",
                        help="Refresh every N seconds (0 = once)")
    args = parser.parse_args()

    def _render():
        state = TradingState.load(args.state_file)
        trade_log = _load_trade_log(args.trade_log)
        print(format_report(state, trade_log))

    if args.watch > 0:
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            os.system("clear")
            _render()
            time.sleep(args.watch)
    else:
        _render()


if __name__ == "__main__":
    main()
