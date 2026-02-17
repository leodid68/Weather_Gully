"""Paper trading CLI — simulate trading with real Polymarket prices.

Usage::

    python -m weather.paper_trade [--verbose] [--set locations=NYC,Chicago]

Uses a PublicClient (read-only, no private key needed) wrapped in
PaperBridge so that the strategy code is identical to real trading
but no orders are ever submitted.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from typing import TYPE_CHECKING

from .config import Config
from .feedback import FeedbackState
from .paper_bridge import PaperBridge
from .state import TradingState, state_lock
from .strategy import run_weather_strategy

if TYPE_CHECKING:
    from .kalman import KalmanState

logger = logging.getLogger(__name__)

# Paths relative to this package directory
_PKG_DIR = str(Path(__file__).parent)
_PAPER_STATE_FILE = str(Path(_PKG_DIR) / "paper_state.json")
_SNAPSHOTS_FILE = str(Path(_PKG_DIR) / "price_snapshots.json")


def _setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    root.addHandler(handler)


async def _resolve_predictions(state: TradingState, gamma) -> int:
    """Resolve pending predictions via Gamma API.

    Returns the number of newly resolved predictions.
    """
    resolved_count = 0
    for market_id, pred in list(state.predictions.items()):
        if pred.resolved:
            continue
        result = await gamma.check_resolution(market_id)
        if result and result.get("resolved"):
            pred.resolved = True
            pred.actual_outcome = result["outcome"]
            resolved_count += 1
            logger.info(
                "Resolved prediction %s: outcome=%s (prob was %.2f)",
                market_id[:16], result["outcome"], pred.our_probability,
            )
    if resolved_count:
        logger.info("Resolved %d pending predictions", resolved_count)
    return resolved_count


def _feed_feedback(state: TradingState, feedback: FeedbackState,
                   kalman: "KalmanState | None" = None) -> int:
    """Feed resolved predictions into the feedback loop using daily observations.

    When *kalman* is provided, also records the absolute forecast error
    into the Kalman filter for dynamic sigma learning.

    Returns the number of entries fed.
    """
    fed = 0
    for pred in state.predictions.values():
        if not pred.resolved or pred.forecast_temp is None or pred.fed_to_feedback:
            continue
        # Get actual temperature from daily observations
        obs = state.get_daily_obs(pred.location, pred.forecast_date)
        if not obs:
            continue
        actual_key = f"obs_{pred.metric}"
        actual_temp = obs.get(actual_key)
        if actual_temp is None:
            continue
        # Parse month from forecast_date
        try:
            month = int(pred.forecast_date.split("-")[1])
        except (IndexError, ValueError):
            continue
        feedback.record(pred.location, month, pred.forecast_temp, actual_temp)
        # Feed Kalman filter
        if kalman is not None:
            abs_error = abs(pred.forecast_temp - actual_temp)
            kalman.record_error(pred.location, pred.horizon, abs_error)
        pred.fed_to_feedback = True
        fed += 1
    if fed:
        logger.info("Fed %d resolved predictions to feedback loop", fed)
    return fed


def _print_pnl_summary(state: TradingState) -> None:
    """Print P&L summary from resolved predictions."""
    resolved = [p for p in state.predictions.values() if p.resolved and p.actual_outcome is not None]
    if not resolved:
        logger.info("No resolved predictions yet — run paper trading daily to accumulate data")
        return

    wins = sum(1 for p in resolved if p.actual_outcome)
    losses = len(resolved) - wins
    # Estimate P&L (simplified: won → +1-price, lost → -price; use prob as proxy for price)
    brier_sum = sum((p.our_probability - (1.0 if p.actual_outcome else 0.0)) ** 2 for p in resolved)
    brier = brier_sum / len(resolved)

    logger.info("=" * 50)
    logger.info("PAPER TRADING P&L SUMMARY")
    logger.info("=" * 50)
    logger.info("Resolved predictions: %d (wins=%d, losses=%d)", len(resolved), wins, losses)
    logger.info("Win rate: %.1f%%", 100.0 * wins / len(resolved))
    logger.info("Brier score: %.4f", brier)
    logger.info("=" * 50)


async def _async_main(
    config: Config,
    paper_bridge: PaperBridge,
    state: TradingState,
    gamma,
    explain: bool,
    use_safeguards: bool,
) -> None:
    """Async entry point — resolves predictions, runs strategy, cleans up."""
    try:
        # Resolve past predictions (must be in same event loop as gamma client)
        await _resolve_predictions(state, gamma)

        # Feed feedback from resolved predictions
        feedback = FeedbackState.load()
        from .kalman import KalmanState
        kalman = KalmanState.load() if config.kalman_sigma else None
        _feed_feedback(state, feedback, kalman=kalman)
        feedback.save()
        if kalman is not None:
            kalman.save()
        _print_pnl_summary(state)

        # Run strategy with PaperBridge — dry_run=False so trades go through
        # the bridge (which simulates them), not skipped as dry_run
        await run_weather_strategy(
            client=paper_bridge,
            config=config,
            state=state,
            dry_run=explain,  # paper_trade normally runs with dry_run=False (paper bridge handles it)
            explain=explain,
            use_safeguards=use_safeguards,
            state_path=_PAPER_STATE_FILE,
        )
    finally:
        from .http_client import close_session
        await close_session()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weather.paper_trade",
        description="Paper trading — simulate weather trading with real Polymarket prices",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="Set config value (e.g., --set locations=NYC,Chicago)",
    )
    parser.add_argument("--no-safeguards", action="store_true", help="Disable context safeguards")
    parser.add_argument("--no-aviation", action="store_true", help="Disable METAR aviation observations")
    parser.add_argument("--explain", action="store_true",
                        help="Show detailed decision reasoning (implies dry-run)")

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    _setup_logging(level=log_level)

    # Load config (no private key needed for paper trading)
    config_dir = _PKG_DIR
    config = Config.load(config_dir)

    # Handle --set updates
    if args.set:
        updates: dict = {}
        for item in args.set:
            if "=" in item:
                key, value = item.split("=", 1)
                updates[key] = value
        if updates:
            config.update(updates)
            logger.info("Config overrides: %s", updates)

    if args.no_aviation:
        config.aviation_obs = False

    logger.info("=" * 50)
    logger.info("PAPER TRADING MODE — no real orders will be placed")
    logger.info("=" * 50)
    logger.info("Locations: %s", ", ".join(config.active_locations))

    # Build real bridge (read-only) then wrap in PaperBridge
    from bot.gamma import GammaClient
    from polymarket.public import PublicClient
    from .bridge import CLOBWeatherBridge

    gamma = GammaClient()
    clob = PublicClient()
    real_bridge = CLOBWeatherBridge(
        clob_client=clob,
        gamma_client=gamma,
        max_exposure=config.max_exposure,
    )
    paper_bridge = PaperBridge(real_bridge)

    # Acquire exclusive lock to prevent concurrent paper runs from corrupting state
    with state_lock(_PAPER_STATE_FILE):
        # Load paper-specific state (separate from real trading state)
        state = TradingState.load(_PAPER_STATE_FILE)
        logger.info("Loaded paper state: %d trades, %d predictions",
                    len(state.trades), len(state.predictions))

        asyncio.run(_async_main(
            config=config,
            paper_bridge=paper_bridge,
            state=state,
            gamma=gamma,
            explain=args.explain,
            use_safeguards=not args.no_safeguards,
        ))

        # Save price snapshots
        paper_bridge.save_snapshots(_SNAPSHOTS_FILE)

        # Save paper state
        state.save(_PAPER_STATE_FILE)
        logger.info("Paper state saved to %s", _PAPER_STATE_FILE)


if __name__ == "__main__":
    main()
