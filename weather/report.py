"""CLI report -- print trading state summary, P&L, and forecast quality metrics.

Usage: python3 -m weather.report [--state-file PATH] [--trade-log PATH]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .metrics import brier_score, calibration_table, sharpe_ratio, win_rate
from .state import TradingState


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


def main():
    parser = argparse.ArgumentParser(description="Weather Gully trading report")
    parser.add_argument("--state-file", default="weather/paper_state.json")
    parser.add_argument("--trade-log", default="weather/trade_log.jsonl")
    args = parser.parse_args()

    state = TradingState.load(args.state_file)
    trade_log = _load_trade_log(args.trade_log)

    print("=" * 60)
    print("  WEATHER GULLY -- TRADING REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Open positions
    print(f"\nOpen positions: {len(state.trades)}")
    for mid, trade in state.trades.items():
        print(f"  {trade.location} {trade.forecast_date} {trade.outcome_name} "
              f"[{trade.side}] @ ${trade.cost_basis:.2f} ({trade.shares:.0f} shares)")

    # Calibration stats
    cal_stats = state.get_calibration_stats()
    print(f"\nResolved predictions: {cal_stats['count']}")
    if cal_stats["brier"] is not None:
        print(f"  Brier score: {cal_stats['brier']:.4f}")
        print(f"  Accuracy:    {cal_stats['accuracy']:.1%}")

    # Detailed metrics from resolved predictions
    resolved = [p for p in state.predictions.values() if p.resolved and p.actual_outcome is not None]
    if resolved:
        preds = [(p.our_probability, p.actual_outcome) for p in resolved]
        bs = brier_score(preds)
        if bs is not None:
            print(f"  Brier (computed): {bs:.4f}")

        cal = calibration_table(preds)
        if cal:
            print("\n  Calibration table:")
            for bin_key, data in cal.items():
                bar = "#" * int(data["actual_freq"] * 20)
                print(f"    {bin_key:>7s}: n={data['count']:>3d}  actual={data['actual_freq']:.1%}  {bar}")

    # Trade log stats
    if trade_log:
        print(f"\nTrade log entries: {len(trade_log)}")

    # Daily P&L
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = state.get_daily_pnl(today)
    print(f"\nDaily P&L ({today}): ${daily:+.2f}")

    # Circuit breaker status
    if state.last_circuit_break:
        print(f"Last circuit break: {state.last_circuit_break}")
    positions_today = state.positions_opened_today(today)
    print(f"Positions opened today: {positions_today}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
