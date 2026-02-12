"""CLI entry point — ``python3 -m bot``."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from polymarket.constants import CLOB_BASE_URL

from .config import Config
from .scanner import run_scan_pipeline
from .state import TradingState, state_lock
from .strategy import run_strategy


@runtime_checkable
class TradingClient(Protocol):
    """Protocol for trading client implementations."""

    def get_markets(self, **filters) -> list[dict]: ...
    def get_orderbook(self, token_id: str) -> dict: ...
    def get_price(self, token_id: str) -> dict: ...
    def post_order(self, *a, **kw) -> dict: ...
    def close(self) -> None: ...


class _PublicClient:
    """Lightweight read-only client for public CLOB endpoints (no private key)."""

    def __init__(self):
        self._http = httpx.Client(base_url=CLOB_BASE_URL, timeout=15)

    def get_markets(self, **filters) -> list[dict]:
        limit = filters.pop("limit", None)
        params = "&".join(f"{k}={v}" for k, v in filters.items())
        path = "/sampling-markets" + (f"?{params}" if params else "")
        resp = self._http.get(path)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            items = data.get("data", data.get("markets", []))
        else:
            items = data
        if limit is not None:
            items = items[:int(limit)]
        return items

    def get_orderbook(self, token_id: str) -> dict:
        resp = self._http.get(f"/book?token_id={token_id}")
        resp.raise_for_status()
        return resp.json()

    def get_price(self, token_id: str) -> dict:
        resp = self._http.get(f"/midpoint?token_id={token_id}")
        resp.raise_for_status()
        data = resp.json()
        # Normalize: /midpoint returns {"mid": "0.xx"}, callers expect {"price": ...}
        if "mid" in data and "price" not in data:
            data["price"] = data["mid"]
        return data

    def post_order(self, *args, **kwargs):
        raise RuntimeError("Cannot post orders without private key (use --live with POLY_PRIVATE_KEY)")

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _setup_logging(level: str = "INFO", json_log: bool = False) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m bot",
        description="Trading bot — Polymarket CLOB direct",
    )
    parser.add_argument("--live", action="store_true", help="Execute real trades")
    parser.add_argument("--positions", action="store_true", help="Show open positions and exit")
    parser.add_argument("--config", action="store_true", help="Show config and exit")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="Override config value (e.g. --set entry_threshold=0.12)",
    )
    parser.add_argument("--scan", action="store_true", help="Show tradeable markets and exit")
    parser.add_argument("--signals", action="store_true", help="Show detected signals and exit")
    parser.add_argument("--weather", action="store_true", help="Scan weather/temperature markets and exit")
    parser.add_argument("--calibration", action="store_true", help="Show calibration stats and exit")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously (daemon mode for OpenClaw)")
    parser.add_argument("--health", action="store_true",
                        help="Check daemon health and exit")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument("--json-log", action="store_true", help="Structured JSON logs (for OpenClaw)")

    args = parser.parse_args()
    config_dir = str(Path(__file__).parent)
    config = Config.load(config_dir)

    log_level = "DEBUG" if args.verbose else config.log_level
    _setup_logging(level=log_level, json_log=args.json_log)
    logger = logging.getLogger(__name__)

    # --set overrides
    if args.set:
        updates = {}
        for item in args.set:
            if "=" in item:
                k, v = item.split("=", 1)
                updates[k] = v
        if updates:
            config.update(updates)
            config.save(config_dir)
            logger.info("Config updated: %s", updates)

    # State path
    state_path = config.state_file
    if not Path(state_path).is_absolute():
        state_path = str(Path(config_dir) / state_path)

    # --health
    if args.health:
        from .daemon import check_health
        ok, msg = check_health(state_path)
        print(msg)
        sys.exit(0 if ok else 1)

    # --config
    if args.config:
        from dataclasses import fields
        data = {f.name: getattr(config, f.name) for f in fields(config)}
        data.pop("private_key", None)
        print(json.dumps(data, indent=2))
        return

    # --positions
    if args.positions:
        state = TradingState.load(state_path)
        positions = state.open_positions()
        if not positions:
            print("No open positions.")
            return
        for p in positions:
            print(f"  {p.token_id[:16]}...  {p.side}  price={p.price:.4f}  size={p.size:.2f}")
        print(f"\n{len(positions)} position(s) | last run: {state.last_run}")
        return

    # --calibration
    if args.calibration:
        state = TradingState.load(state_path)
        cal = state.get_calibration()
        if cal["n"] == 0:
            print("No resolved predictions yet.")
            return
        print(f"Brier score: {cal['brier']:.6f}")
        print(f"Log score:   {cal['log']:.6f}")
        print(f"Predictions: {cal['n']}")
        from .scoring import calibration_curve
        preds = [p["our_prob"] for p in state.predictions.values()
                 if p.get("resolved") and p.get("outcome") is not None]
        outcomes = [p["outcome"] for p in state.predictions.values()
                    if p.get("resolved") and p.get("outcome") is not None]
        if preds:
            curve = calibration_curve(preds, outcomes)
            print("\nCalibration curve:")
            print(f"  {'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Count':>6}")
            for b, p, a, c in zip(curve["bins"], curve["predicted"], curve["actual"], curve["count"]):
                print(f"  {b:<12} {p:>10.3f} {a:>10.3f} {c:>6}")
        return

    # --daemon (before client build — daemon manages its own client lifecycle)
    if args.daemon:
        from .daemon import run_daemon

        if args.live and not config.private_key:
            print("Error: set POLY_PRIVATE_KEY env var or --set private_key=0x...")
            sys.exit(1)

        def _make_client():
            if args.live:
                from polymarket.client import PolymarketClient
                api_creds = config.load_api_creds(config_dir)
                return PolymarketClient(
                    private_key=config.private_key,
                    api_creds=api_creds,
                )
            return _PublicClient()

        run_daemon(_make_client, config, state_path, dry_run=not args.live)
        return

    # ── Build client ─────────────────────────────────────────────────
    needs_auth = args.live
    client = None

    if needs_auth:
        if not config.private_key:
            print("Error: set POLY_PRIVATE_KEY env var or --set private_key=0x...")
            sys.exit(1)
        from polymarket.client import PolymarketClient
        api_creds = config.load_api_creds(config_dir)
        client = PolymarketClient(
            private_key=config.private_key,
            api_creds=api_creds,
        )
    else:
        client = _PublicClient()

    # --weather
    if args.weather:
        from .gamma import GammaClient, group_multi_choice, gamma_to_scanner_format
        from .signals import scan_for_signals
        with GammaClient() as gamma:
            events, all_markets = gamma.fetch_events_with_markets(
                tag_slug="weather", limit=100,
            )

            # Group multi-choice and convert to scanner format
            mc_groups = group_multi_choice(all_markets, gamma_client=gamma)
            tradeable = gamma_to_scanner_format(all_markets)

            # Display events summary
            print(f"\n── Weather events ({len(events)}) ──")
            for ev in events:
                title = ev.get("title", "?")[:60]
                vol = float(ev.get("volume", 0))
                n = len(ev.get("markets", []))
                print(f"  {title:<60} vol=${vol:>10,.0f}  ({n} outcomes)")

            # Display multi-choice arb opportunities
            if mc_groups:
                print(f"\n── Multi-choice groups ({len(mc_groups)}) ──")
                print(f"  {'Event':<55} {'N':>3} {'Σ YES':>7} {'Dev':>7}")
                print("  " + "-" * 75)
                for g in mc_groups:
                    title = g.event_title[:53] if g.event_title else f"event_{g.event_id}"
                    print(f"  {title:<55} {len(g.markets):>3} {g.yes_sum:>6.3f} {g.deviation:>+6.3f}")

            # Detect signals if we have data
            if tradeable:
                token_ids = [t["token_id"] for m in tradeable for t in m.get("tokens", [])]
                token_prices = {
                    t["token_id"]: t.get("price", 0)
                    for m in tradeable for t in m.get("tokens", [])
                }
                token_pairs = {}
                for m in tradeable:
                    toks = m.get("tokens", [])
                    if len(toks) == 2:
                        token_pairs[m["condition_id"]] = (toks[0]["token_id"], toks[1]["token_id"])

                signals = scan_for_signals(
                    client, token_ids, config,
                    multi_choice_groups=mc_groups,
                    token_prices=token_prices,
                    token_pairs=token_pairs,
                )
                if signals:
                    print(f"\n── Signals ({len(signals)}) ──")
                    print(f"  {'Side':<5} {'Token':<18} {'Price':>6} {'Edge':>6} {'Method':<20} {'Conf':>5}")
                    print("  " + "-" * 64)
                    for s in signals:
                        print(
                            f"  {s.side:<4} {s.token_id[:16]:<18} {s.market_price:>5.3f} "
                            f"{s.edge:>6.4f} {s.method:<20} {s.confidence:>5.2f}"
                        )
                else:
                    print("\nNo signals detected on weather markets.")

            print(f"\n{len(all_markets)} weather markets total, {len(events)} events")
        return

    # --scan
    if args.scan:
        try:
            tradeable, mc_groups, _, _, _ = run_scan_pipeline(client, config)

            # Display markets
            print(f"{'Question':<50} {'Grade':>5} {'Spread':>8} {'Vol24h':>10} {'Liq':>8}")
            print("-" * 84)
            for m in tradeable:
                q = m.get("question", "")[:48]
                grade = m.get("liquidity_grade", "?")
                gamma = m.get("gamma", {})
                spread = gamma.get("spread", 0)
                vol24h = gamma.get("volume_24hr", 0)
                liq = gamma.get("liquidity", 0)
                print(f"  {q:<48} {grade:>5} {spread:>7.4f} {vol24h:>9.0f} {liq:>7.0f}")
            print(f"\n{len(tradeable)} tradeable market(s)")

            # Display multi-choice groups
            if mc_groups:
                print(f"\n── Multi-choice groups ({len(mc_groups)}) ──")
                print(f"{'Event':<50} {'N':>3} {'Σ YES':>7} {'Dev':>7}")
                print("-" * 70)
                for g in mc_groups:
                    title = g.event_title[:48] if g.event_title else f"event_{g.event_id}"
                    print(f"  {title:<48} {len(g.markets):>3} {g.yes_sum:>6.3f} {g.deviation:>+6.3f}")
        finally:
            client.close()
        return

    # --signals
    if args.signals:
        from .signals import scan_for_signals
        try:
            tradeable, mc_groups, token_ids, token_prices, token_pairs = run_scan_pipeline(client, config)
            signals = scan_for_signals(
                client, token_ids, config,
                multi_choice_groups=mc_groups,
                token_prices=token_prices,
                token_pairs=token_pairs,
            )
            if not signals:
                print("No signals detected.")
            else:
                print(f"{'Side':<5} {'Token':<18} {'Price':>6} {'Edge':>6} {'Method':<20} {'Conf':>5}")
                print("-" * 64)
                for s in signals:
                    print(
                        f"  {s.side:<4} {s.token_id[:16]:<18} {s.market_price:>5.3f} "
                        f"{s.edge:>6.4f} {s.method:<20} {s.confidence:>5.2f}"
                    )
                print(f"\n{len(signals)} signal(s)")
        finally:
            client.close()
        return

    state = TradingState.load(state_path)
    dry_run = not args.live

    if dry_run:
        logger.info("DRY RUN — no trades will be executed")

    try:
        with state_lock(state_path):
            run_strategy(
                client=client,
                config=config,
                state=state,
                dry_run=dry_run,
                state_path=state_path,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
