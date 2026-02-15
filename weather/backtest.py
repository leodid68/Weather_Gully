"""Backtesting engine — simulate strategy on historical data.

Usage::

    python -m weather.backtest --locations NYC --start-date 2025-06-01 --end-date 2025-12-31
"""

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .config import LOCATIONS
from .historical import get_historical_actuals, get_historical_forecasts
from .open_meteo import compute_ensemble_forecast
from .probability import estimate_bucket_probability

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    date: str
    location: str
    metric: str
    bucket: tuple[int, int]
    our_probability: float
    simulated_price: float
    forecast_temp: float
    actual_temp: float
    won: bool
    pnl: float


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    brier_score: float = 0.0
    accuracy: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calibration_curve: list[tuple[float, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total trades:      {len(self.trades)}",
            f"Brier score:       {self.brier_score:.4f}",
            f"Accuracy:          {self.accuracy:.1%}",
            f"Total P&L:         ${self.total_pnl:.2f}",
            f"ROI:               {self.roi:.1%}",
            f"Max drawdown:      ${self.max_drawdown:.2f}",
            f"Sharpe ratio:      {self.sharpe_ratio:.2f}",
            "=" * 60,
        ]
        if self.calibration_curve:
            lines.append("Calibration curve (predicted → actual):")
            for predicted, actual in self.calibration_curve:
                lines.append(f"  {predicted:.2f} → {actual:.2f}")
        return "\n".join(lines)


def _load_price_snapshots(path: str) -> dict[str, dict]:
    """Load price snapshots from JSON and index by bucket key.

    Returns a dict keyed by ``"{date}|{location}|{metric}|{lo},{hi}"``
    with values ``{"best_ask": float, "best_bid": float}``.
    """
    snapshots: dict[str, dict] = {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return snapshots

    for snap in raw:
        date = snap.get("date", "")
        location = snap.get("location", "")
        metric = snap.get("metric", "")
        lo = snap.get("bucket_lo")
        hi = snap.get("bucket_hi")
        if not (date and location and metric and lo is not None and hi is not None):
            continue
        key = f"{date}|{location}|{metric}|{lo},{hi}"
        # Keep the latest snapshot for each key
        snapshots[key] = {
            "best_ask": snap.get("best_ask", 0.0),
            "best_bid": snap.get("best_bid", 0.0),
        }
    return snapshots


def _simulate_market_price(our_prob: float, n_buckets: int, seed_val: int) -> float:
    """Simulate a realistic market price from probability + noise.

    Models a semi-efficient market: prices are correlated with true
    probabilities but with ~5% Gaussian noise (model/market disagreement).
    Deterministic for a given seed (reproducible backtests).

    Returns a price clamped to [0.02, 0.98].
    """
    rng = random.Random(seed_val)
    noise = rng.gauss(0, 0.05)
    return max(0.02, min(0.98, our_prob + noise))


def run_backtest(
    locations: list[str],
    start_date: str,
    end_date: str,
    horizon: int = 3,
    entry_threshold: float = 0.03,
    bucket_width: int = 5,
    snapshot_path: str | None = None,
) -> BacktestResult:
    """Run backtest simulation over historical data.

    For each day in the range, simulates what the bot would have done:
    1. Get the forecast from N days before (``horizon``)
    2. Compute bucket probabilities
    3. Simulate a trade if EV > threshold
    4. Resolve against ERA5 actuals

    Args:
        locations: Location keys to backtest.
        start_date: Start of backtesting period (YYYY-MM-DD).
        end_date: End of backtesting period (YYYY-MM-DD).
        horizon: Forecast horizon in days (default 3).
        entry_threshold: Minimum EV to take a trade (default 0.03).
        bucket_width: Width of simulated buckets in °F (default 5).
        snapshot_path: Path to price_snapshots.json from paper trading.
            When provided, real recorded prices are used where available.
    """
    trades: list[BacktestTrade] = []

    # Load price snapshots if available
    price_snapshots: dict[str, dict] = {}
    if snapshot_path:
        price_snapshots = _load_price_snapshots(snapshot_path)
        if price_snapshots:
            logger.info("Loaded %d price snapshots from %s", len(price_snapshots), snapshot_path)

    for loc in locations:
        loc_data = LOCATIONS.get(loc)
        if not loc_data:
            logger.warning("Unknown location: %s", loc)
            continue

        lat, lon = loc_data["lat"], loc_data["lon"]
        tz_name = loc_data.get("tz", "America/New_York")

        # Fetch data — extend start back by horizon days for forecasts
        forecast_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=horizon)
        ).strftime("%Y-%m-%d")

        logger.info("Fetching historical data for %s (%s to %s)...", loc, forecast_start, end_date)
        forecasts = get_historical_forecasts(lat, lon, forecast_start, end_date, tz_name=tz_name)
        actuals = get_historical_actuals(lat, lon, start_date, end_date, tz_name=tz_name)

        if not actuals:
            logger.warning("No actuals data for %s", loc)
            continue

        for target_date_str, actual_data in actuals.items():
            # Forecasts are keyed by target_date (API deduplication — no true
            # horizon distinction), so look up by target_date directly.
            target_forecasts = forecasts.get(target_date_str, {})
            run_forecasts = target_forecasts.get(target_date_str)
            if not run_forecasts:
                continue

            for metric in ["high", "low"]:
                actual_temp = actual_data.get(metric)
                if actual_temp is None:
                    continue

                # Compute ensemble forecast from historical model data
                ensemble_temp, _ = compute_ensemble_forecast(
                    noaa_temp=None,
                    open_meteo_data=run_forecasts,
                    metric=metric,
                )
                if ensemble_temp is None:
                    continue

                # Generate buckets around the forecast
                center = round(ensemble_temp / bucket_width) * bucket_width
                buckets = []
                for offset in range(-3, 4):
                    lo = center + offset * bucket_width
                    hi = lo + bucket_width - 1
                    buckets.append((lo, hi))
                # Add open-ended buckets
                buckets[0] = (-999, buckets[0][1])
                buckets[-1] = (buckets[-1][0], 999)

                # Score each bucket
                for bucket_lo, bucket_hi in buckets:
                    prob = estimate_bucket_probability(
                        ensemble_temp, bucket_lo, bucket_hi,
                        target_date_str, apply_seasonal=True,
                        location=loc,
                    )

                    # Simulate market price: use real snapshot if available,
                    # otherwise model a semi-efficient market (prob + noise)
                    n_buckets = len(buckets)
                    snap_key = f"{target_date_str}|{loc}|{metric}|{bucket_lo},{bucket_hi}"
                    snapshot = price_snapshots.get(snap_key)
                    if snapshot and snapshot["best_ask"] > 0:
                        simulated_price = snapshot["best_ask"]
                    else:
                        seed = hash(f"{target_date_str}{loc}{metric}{bucket_lo}")
                        simulated_price = _simulate_market_price(prob, n_buckets, seed)

                    ev = prob - simulated_price
                    if ev < entry_threshold:
                        continue

                    # Did we win?
                    if bucket_lo <= -900:
                        won = actual_temp <= bucket_hi
                    elif bucket_hi >= 900:
                        won = actual_temp >= bucket_lo
                    else:
                        won = bucket_lo <= actual_temp <= bucket_hi

                    pnl = (1.0 - simulated_price) if won else -simulated_price

                    trades.append(BacktestTrade(
                        date=target_date_str,
                        location=loc,
                        metric=metric,
                        bucket=(bucket_lo, bucket_hi),
                        our_probability=prob,
                        simulated_price=simulated_price,
                        forecast_temp=ensemble_temp,
                        actual_temp=actual_temp,
                        won=won,
                        pnl=pnl,
                    ))

    if not trades:
        return BacktestResult()

    # Compute metrics
    brier_score = sum(
        (t.our_probability - (1.0 if t.won else 0.0)) ** 2
        for t in trades
    ) / len(trades)

    wins = sum(1 for t in trades if t.won)
    accuracy = wins / len(trades)
    total_pnl = sum(t.pnl for t in trades)
    total_risked = sum(t.simulated_price for t in trades)
    roi = total_pnl / total_risked if total_risked > 0 else 0.0
    max_drawdown = _compute_max_drawdown(trades)
    sharpe = _compute_sharpe(trades)
    cal_curve = compute_calibration_curve(
        [(t.our_probability, t.won) for t in trades],
    )

    return BacktestResult(
        trades=trades,
        brier_score=round(brier_score, 4),
        accuracy=round(accuracy, 4),
        total_pnl=round(total_pnl, 2),
        roi=round(roi, 4),
        max_drawdown=round(max_drawdown, 2),
        sharpe_ratio=round(sharpe, 2),
        calibration_curve=cal_curve,
    )


def _compute_max_drawdown(trades: list[BacktestTrade]) -> float:
    """Compute maximum drawdown from trade P&L series."""
    if not trades:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_sharpe(trades: list[BacktestTrade], risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio from trade P&L series."""
    if len(trades) < 2:
        return 0.0
    returns = [t.pnl for t in trades]
    mean_ret = sum(returns) / len(returns)
    var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean_ret - risk_free) / std


def compute_calibration_curve(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> list[tuple[float, float]]:
    """Compute reliability diagram data.

    Args:
        predictions: List of ``(predicted_probability, actual_outcome)`` pairs.
        n_bins: Number of bins for the calibration curve.

    Returns:
        List of ``(mean_predicted, fraction_positive)`` per bin.
    """
    if not predictions:
        return []

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for pred, outcome in predictions:
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, outcome))

    curve: list[tuple[float, float]] = []
    for bin_items in bins:
        if not bin_items:
            continue
        mean_pred = sum(p for p, _ in bin_items) / len(bin_items)
        frac_pos = sum(1 for _, o in bin_items if o) / len(bin_items)
        curve.append((round(mean_pred, 3), round(frac_pos, 3)))

    return curve


def generate_report(result: BacktestResult, output_path: str) -> None:
    """Write backtest report as JSON."""
    report = {
        "summary": {
            "total_trades": len(result.trades),
            "brier_score": result.brier_score,
            "accuracy": result.accuracy,
            "total_pnl": result.total_pnl,
            "roi": result.roi,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
        },
        "calibration_curve": [
            {"predicted": p, "actual": a}
            for p, a in result.calibration_curve
        ],
        "trades": [
            {
                "date": t.date,
                "location": t.location,
                "metric": t.metric,
                "bucket": list(t.bucket),
                "our_probability": t.our_probability,
                "simulated_price": t.simulated_price,
                "forecast_temp": t.forecast_temp,
                "actual_temp": t.actual_temp,
                "won": t.won,
                "pnl": t.pnl,
            }
            for t in result.trades
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Backtest report written to %s", output_path)


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(
        description="Backtest weather trading strategy on historical data",
    )
    parser.add_argument(
        "--locations", type=str, default="NYC",
        help="Comma-separated location keys (default: NYC)",
    )
    parser.add_argument(
        "--start-date", type=str, required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon", type=int, default=3,
        help="Forecast horizon in days (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default="backtest_report.json",
        help="Output path for report (default: backtest_report.json)",
    )
    parser.add_argument(
        "--snapshot-path", type=str, default=None,
        help="Path to price_snapshots.json from paper trading (uses real prices where available)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loc_keys = [l.strip() for l in args.locations.split(",")]

    result = run_backtest(
        locations=loc_keys,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        snapshot_path=args.snapshot_path,
    )

    print(result.summary())

    if result.trades:
        generate_report(result, args.output)


if __name__ == "__main__":
    main()
