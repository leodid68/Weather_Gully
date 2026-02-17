"""CLI entry point — ``python -m weather`` (CLOB direct)."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .state import TradingState, state_lock
from .strategy import run_weather_strategy


def _setup_logging(level: str = "INFO", json_log: bool = False) -> None:
    """Configure structured logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if json_log:
        fmt = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}'
        )
    else:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    root.addHandler(handler)


def _build_bridge(config: Config, live: bool):
    """Build CLOBWeatherBridge from config.

    Dry-run uses a public read-only client; --live uses an authenticated client.
    """
    from bot.gamma import GammaClient
    from .bridge import CLOBWeatherBridge

    gamma = GammaClient()

    if live:
        key = config.private_key or os.environ.get("POLY_PRIVATE_KEY")
        if not key:
            print("Error: set POLY_PRIVATE_KEY env var or --set private_key=0x...")
            sys.exit(1)

        from polymarket.client import PolymarketClient

        api_creds = config.load_api_creds(str(Path(__file__).parent))
        clob = PolymarketClient(private_key=key, api_creds=api_creds)
    else:
        # Public read-only client (can't post orders)
        from polymarket.public import PublicClient
        clob = PublicClient()

    return CLOBWeatherBridge(
        clob_client=clob,
        gamma_client=gamma,
        max_exposure=config.max_exposure,
    )


async def _async_main(
    config: Config,
    state: TradingState,
    bridge,
    dry_run: bool,
    explain: bool,
    positions_only: bool,
    use_safeguards: bool,
    state_path: str,
    show_config: bool = False,
) -> None:
    """Async entry point — runs strategy and cleans up HTTP session."""
    try:
        await run_weather_strategy(
            client=bridge,
            config=config,
            state=state,
            dry_run=dry_run,
            explain=explain,
            positions_only=positions_only,
            show_config=show_config,
            use_safeguards=use_safeguards,
            state_path=state_path,
        )
    finally:
        from .http_client import close_session
        await close_session()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weather",
        description="Weather Trading Bot — trades Polymarket weather markets using NOAA forecasts (CLOB direct)",
    )
    parser.add_argument("--live", action="store_true", help="Execute real trades (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="(Default) Show opportunities without trading")
    parser.add_argument("--positions", action="store_true", help="Show current positions only")
    parser.add_argument("--config", action="store_true", help="Show current config")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="Set config value (e.g., --set entry_threshold=0.20)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--json-log", action="store_true", help="Output structured JSON logs")
    parser.add_argument("--no-safeguards", action="store_true", help="Disable context safeguards")
    parser.add_argument("--no-aviation", action="store_true", help="Disable METAR aviation observations")
    parser.add_argument("--explain", action="store_true",
                        help="Show detailed decision reasoning (implies --dry-run)")

    args = parser.parse_args()

    # Determine config directory (same as this package)
    config_dir = str(Path(__file__).parent)

    # Load config
    config = Config.load(config_dir)

    # Setup logging early so config.update() warnings are visible
    log_level = "DEBUG" if args.verbose else config.log_level
    _setup_logging(level=log_level, json_log=args.json_log)

    logger = logging.getLogger(__name__)

    # Handle --set updates
    if args.set:
        updates: dict = {}
        for item in args.set:
            if "=" in item:
                key, value = item.split("=", 1)
                updates[key] = value
        if updates:
            config.update(updates)
            config.save(config_dir)
            safe_updates = {k: v for k, v in updates.items() if k != "private_key"}
            logger.info("Config updated: %s", safe_updates)

    # Resolve state file path relative to config dir
    state_path = config.state_file
    if not Path(state_path).is_absolute():
        state_path = str(Path(config_dir) / state_path)

    # Show config only
    if args.config:
        asyncio.run(_async_main(
            config=config,
            state=TradingState(),
            bridge=None,  # type: ignore[arg-type]
            dry_run=True,
            explain=False,
            positions_only=False,
            use_safeguards=True,
            state_path=state_path,
            show_config=True,
        ))
        return

    # Disable aviation observations if requested
    if args.no_aviation:
        config.aviation_obs = False
        logger.info("Aviation observations disabled via --no-aviation")

    # Build CLOB bridge
    dry_run = not args.live
    bridge = _build_bridge(config, live=args.live)

    # Acquire exclusive lock to prevent concurrent runs from corrupting state
    with state_lock(state_path):
        # Load persistent state
        state = TradingState.load(state_path)

        asyncio.run(_async_main(
            config=config,
            state=state,
            bridge=bridge,
            dry_run=dry_run or args.explain,
            explain=args.explain,
            positions_only=args.positions,
            use_safeguards=not args.no_safeguards,
            state_path=state_path,
        ))


if __name__ == "__main__":
    main()
